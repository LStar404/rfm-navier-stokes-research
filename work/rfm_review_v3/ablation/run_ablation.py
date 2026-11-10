"""Frozen, bounded CPU accuracy ablation; no elapsed value is a speed comparison."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ARCHIVE = WORK / "rfm_v8/rank_fair_cache_qualification_output"
SEEDS = [7331, 7687, 8011, 8353, 9041]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze():
    target = HERE / "protocol.json"
    if target.exists():
        raise FileExistsError("Protocol already frozen; use run, never overwrite it")
    original = read(ARCHIVE / "protocol.json")
    baselines = []
    jobs = []
    for resolution, name in (("mid", "rfm37_k72"), ("high", "rfm41_k80")):
        selected = [(i, j) for i, j in enumerate(original["slots"]) if j["name"] == name
                    and j["arm"] == "uniform" and j["policy"] == "empty_first"]
        assert [j["seed"] for _, j in selected] == SEEDS
        for index, job in selected:
            folder = ARCHIVE / f"slot_{index:03d}"
            baseline = dict(resolution=resolution, name=name, seed=job["seed"], slot=index,
                config=job["config"], steps=job["args"]["steps"],
                trajectory=str((folder / "trajectory.npz").relative_to(WORK)),
                trajectory_sha256=digest(folder / "trajectory.npz"),
                independent_score=str((folder / "independent_score.json").relative_to(WORK)),
                independent_score_sha256=digest(folder / "independent_score.json"))
            baselines.append(baseline)
            config = dict(job["config"], midpoint=False, nonzero_layers=False)
            jobs.append(dict(id=f"{resolution}_random_no_layers_{job['seed']}",
                resolution=resolution, role="random-no-layers", seed=job["seed"],
                config=config, steps=job["args"]["steps"]))
        first = selected[0][1]
        for role, midpoint, layers in (("midpoint-layers", True, True),
                                      ("midpoint-no-layers", True, False),
                                      ("cpu-control", False, True)):
            jobs.append(dict(id=f"{resolution}_{role.replace('-', '_')}_{SEEDS[0]}",
                resolution=resolution, role=role, seed=SEEDS[0],
                config=dict(first["config"], midpoint=midpoint, nonzero_layers=layers),
                steps=first["args"]["steps"]))
    sources = [HERE / "ablation_model.py", HERE / "run_ablation.py",
               WORK / "rfm_v7/basis_recurrence_model.py", WORK / "rfm_v7/cpu_implicit_stride.py",
               WORK / "rfm_v6/full_galerkin.py", WORK / "rfm_v6/full_galerkin_buffered.py",
               WORK / "rfm_v6/full_case.py", WORK / "rfm_v6/full_nonlinear.py",
               WORK / "rfm_v6/radial_layers.py", WORK / "rfm_v3/cylinder_rfm.py"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORK.parent, capture_output=True, text=True)
    protocol = dict(status="frozen-before-execution", created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Accuracy-only 2x2 frequency jitter / nonzero radial-layer ablation",
        seeds=SEEDS, horizon=72, times=original["physical_times"], probe_spacing=.125,
        physical_gate=original["physical_gate"], max_workers=6, blas_threads=1,
        mean_layers_unchanged=[16, 36, 80, 165, 340, 660], mean_order=24,
        svd_cutoff=1e-10, post_failure_tuning=False, timings_are_diagnostic_only=True,
        deterministic_replicates="One unique deterministic trajectory per cell; no pseudo seed replicates",
        baseline_reuse="Ten frozen GPU trajectories; one new CPU enriched random control per resolution",
        rank_policy="Fixed original raw m and cutoff. Actual retained ranks reported; no rank-matching claim.",
        reference_protocol="rfm_v7/switch_qualification_output/protocol.json",
        reference_protocol_sha256=digest(WORK / "rfm_v7/switch_qualification_output/protocol.json"),
        archive_protocol_sha256=digest(ARCHIVE / "protocol.json"),
        sources_sha256={str(p.relative_to(WORK)):digest(p) for p in sources},
        git_head_at_registration=git.stdout.strip(), python=platform.python_version(),
        baselines=baselines, jobs=jobs)
    write(target, protocol)
    print(json.dumps(dict(protocol=str(target), new_cpu_jobs=len(jobs), archived_baselines=len(baselines))))

def execute(job, horizon=None):
    from ablation_model import AblationGalerkin, case, np
    protocol = read(HERE / "protocol.json")
    folder = HERE / "runs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    sidecar = folder / "result.json"
    if sidecar.exists():
        return read(sidecar)
    started = time.perf_counter()
    result = dict(job=job, status="running", protocol_sha256=digest(HERE / "protocol.json"),
        started_utc=datetime.now(timezone.utc).isoformat(), diagnostic_elapsed_seconds=None,
        timing_comparison_valid=False, process_id=os.getpid(),
        thread_environment={k:os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    write(folder / "started.json", result)
    try:
        np.seterr(over="raise", invalid="raise", divide="raise")
        inspected = AblationGalerkin(dt=72 / job["steps"], **job["config"])
        result["construction_metadata"] = inspected.metadata
        result["mean_basis_sha256"] = hashlib.sha256(inspected.mean_samples.tobytes()).hexdigest()
        result["mean_operator_sha256"] = hashlib.sha256(inspected.mean_inverse.tobytes()).hexdigest()
        write(folder / "constructed.json", result)
        sol = case.solve(AblationGalerkin, steps=job["steps"], horizon=72,
            snapshots=protocol["times"], probe_spacing=.125, **job["config"])
        target = folder / "trajectory.npz"
        case.store_solution(target, sol, job["config"])
        result.update(status="complete", metadata=sol.row, trajectory=str(target.relative_to(HERE)),
            trajectory_sha256=digest(target))
    except Exception as exc:
        result.update(status="failed", exception=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc())
    result["diagnostic_elapsed_seconds"] = time.perf_counter() - started
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write(sidecar, result)
    return result

def run(workers):
    protocol = read(HERE / "protocol.json")
    for name, expected in protocol["sources_sha256"].items():
        assert digest(WORK / name) == expected, f"Registered source changed: {name}"
    assert 1 <= workers <= protocol["max_workers"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, j):j for j in protocol["jobs"]}
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps(dict(id=row["job"]["id"], status=row["status"],
                elapsed=row["diagnostic_elapsed_seconds"], failure=row.get("failure"))), flush=True)

def assess():
    from ablation_model import AblationGalerkin, case, np
    from basis_recurrence_model import RecurrenceGalerkin
    from verify_switch_qualification import absolute_reference
    protocol = read(HERE / "protocol.json")
    grids = case.evaluation_grids()
    grids["layer144"] = case.layer_scoring_grid()
    truth, budget = absolute_reference(read(WORK / protocol["reference_protocol"]), grids)
    results = []
    for item in protocol["baselines"]:
        path = WORK / item["trajectory"]
        assert digest(path) == item["trajectory_sha256"]
        # Reconstruct and rescore archived fields with the same CPU basis evaluator.
        sol = case.load_solution(path, RecurrenceGalerkin)
        scores = {name:case.score(sol, fields, grids) for name, fields in truth.items()}
        worst = case.maxima([r for rows in scores.values() for r in rows])
        row = dict(id=f"{item['resolution']}_archived_random_layers_{item['seed']}",
            resolution=item["resolution"], seed=item["seed"], role="random-layers",
            provenance="frozen-v8-GPU-trajectory-rescored", status="complete",
            retained_rank=sol.model.width, mean_rank=sol.model.mean_width,
            physical_worst=worst, physical_pass=max(worst.values()) <= protocol["physical_gate"],
            source=item, scores=scores)
        results.append(row)
        print(json.dumps({k:row[k] for k in ("id", "physical_worst", "physical_pass")}), flush=True)
    for job in protocol["jobs"]:
        result = read(HERE / "runs" / job["id"] / "result.json")
        row = dict(id=job["id"], resolution=job["resolution"], seed=job["seed"], role=job["role"],
            provenance="new-CPU-accuracy-only", status=result["status"],
            result_path=str((HERE / "runs" / job["id"] / "result.json").relative_to(HERE)))
        if result["status"] == "complete":
            path = HERE / result["trajectory"]
            assert digest(path) == result["trajectory_sha256"]
            sol = case.load_solution(path, AblationGalerkin)
            scores = {name:case.score(sol, fields, grids) for name, fields in truth.items()}
            worst = case.maxima([r for rows in scores.values() for r in rows])
            row.update(retained_rank=sol.model.width, mean_rank=sol.model.mean_width,
                physical_worst=worst, physical_pass=max(worst.values()) <= protocol["physical_gate"], scores=scores,
                retained_scaled_condition=sol.model.metadata["retained_scaled_condition"])
            if job["role"] == "cpu-control":
                old = next(x for x in protocol["baselines"] if x["resolution"]==job["resolution"] and x["seed"]==job["seed"])
                oldsol = case.load_solution(WORK / old["trajectory"], RecurrenceGalerkin)
                previous = case.reference_fields(oldsol, grids)
                delta = case.score(sol, previous, grids)
                row["archived_gpu_control_difference"] = case.maxima(delta)
                row["archived_gpu_control_rows"] = delta
        else:
            row.update(failure=result["failure"], exception=result["exception"], physical_pass=False)
        results.append(row)
        write(HERE / "assessment_partial.json", dict(status="in-progress", results=results))
        print(json.dumps({k:row[k] for k in ("id", "status", "physical_pass")}, ensure_ascii=False), flush=True)
    summary = dict(status="complete", protocol_sha256=digest(HERE / "protocol.json"),
        horizon=72, checkpoints=protocol["times"], grid_names=list(grids), reference_names=list(truth),
        metrics=["velocity", "vorticity", "near_velocity", "near_vorticity"], physical_gate=protocol["physical_gate"],
        reference_budget=budget, new_cpu_jobs=len(protocol["jobs"]), reused_gpu_baselines=len(protocol["baselines"]),
        no_new_gpu_timing=True, original_78_gpu_qualification_unchanged=True,
        limitations=["Enrichment removal changes retained rank; this is a fixed raw-m design, not equal-rank attribution.",
            "One midpoint design is tested; it cannot establish optimality among deterministic frequencies.",
            "CPU elapsed times were collected concurrently and are not performance evidence.",
            "Mean layers remain present; this ablates nonzero-mode enrichment only."], results=results)
    write(HERE / "summary.json", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "assess"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    {"freeze":freeze, "run":lambda:run(args.workers), "assess":assess}[args.action]()
