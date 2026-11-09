"""Checkpointed, matched-resolution CPU/native CUDA dispatch ablation."""

import argparse
import json
from contextlib import nullcontext
import math
from pathlib import Path
import platform
import statistics
import time

import cpu_formal as base
import fd_pow2_reference_chain as reference_chain
import joint_hardware as matched
from basis_recurrence_model import RecurrenceGalerkin, native_model
from fd_cpu_buffered import BufferedFD
from fd_cuda_native import FDRuntime, solve_fd_native
from galerkin_cuda_native import Runtime, solve_native


case, np = base.case, base.case.np
ORDER_SEED = 116293
BACKENDS = ("cpu", "cuda_eager", "cuda_graph")
GRAPH_STEPS = 24
MATCH_TOLERANCE = 1e-5


def settings_hashes():
    return (matched.hashes() | reference_chain.fingerprint() |
            {Path(__file__).relative_to(base.WORK).as_posix():base.digest(Path(__file__))})


METRICS = ("velocity","vorticity","near_velocity","near_vorticity")


def checked_scores(rows, grids, times):
    expected = {(grid,float(t)) for grid in grids for t in times}
    observed = [(r["grid"],r["time"]) for r in rows]
    if len(observed)!=len(expected) or set(observed)!=expected:
        raise ValueError("Expected one score per grid and absolute checkpoint")
    if any(not math.isfinite(r[k]) or r[k]<0 for r in rows for k in METRICS):
        raise FloatingPointError("Nonfinite or negative field score")
    return case.maxima(rows)


def pow2_reference_record(path, protocol):
    chain = base.read(reference_chain.JOURNAL)
    frozen = chain["frozen"]
    if chain["status"]!="complete" or frozen["protocol"]!=base.read(reference_chain.PROTOCOL):
        raise RuntimeError("The independent pow2 reference chain is incomplete or changed")
    if frozen["source_hashes"]!=reference_chain.fingerprint():
        raise RuntimeError("Pow2 reference implementation differs from frozen inputs")
    scope = frozen["protocol"]
    spec = scope["reference_path"][-1]
    expected = base.WORK/spec["source"]
    if path is not None and path.resolve()!=expected.resolve():
        raise ValueError("Full isolated scaling uses the time_fine artifact from the pow2 chain")
    path = expected
    details = next(row for row in chain["rows"] if row["spec"]["name"]==spec["name"])
    if details["spec"]!=spec or details["status"]!="complete" or base.digest(path)!=details["sha256"]:
        raise RuntimeError("Final pow2 trajectory does not match the completed arm")
    audit_path = reference_chain.OUT/chain["reference_audit"]
    audit = base.read(audit_path)
    if ([e["axis"] for e in audit["edges"]]!=["radial","angle","time"] or
            not audit["independent_axes"] or audit["passes"]!=chain["reference_screen"] or
            audit["composed_difference"]!=chain["composed_difference"]):
        raise RuntimeError("Reference axis evidence differs from the completed chain")
    grids = base.grid_set()
    maxima = []
    for edge in audit["edges"]:
        worst = checked_scores(edge["scores"],grids,protocol["checkpoints"])
        if edge["worst"]!=worst:
            raise RuntimeError("Reference edge maxima differ from raw scores")
        maxima.append(max(worst.values()))
    composed = math.prod(1+v for v in maxima)-1
    limit = scope["gate"]*scope["reference_budget_fraction"]
    if (composed!=audit["composed_difference"] or audit["threshold"]!=limit or
            audit["passes"]!=(composed<=limit)):
        raise RuntimeError("Reference composition or screening decision changed")
    with np.load(path,allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"]))
        settings = json.loads(str(artifact["config"]))
    expected_config = dict(reference_chain.grid(spec),n=spec["n"])
    expected_class = reference_chain.radial_driver.CLASS
    if (metadata!=details["metadata"] or settings!=expected_config or
            metadata["reconstruction_class"]!=expected_class or
            metadata["equilibration"]!="row_power_of_two" or
            metadata["steps"]!=scope["horizon"]*spec["rate"]):
        raise RuntimeError("Pow2 reference parameters or reconstruction class differ")
    return path, dict(source=spec["source"],sha256=details["sha256"],spec=spec,
        reconstruction_class=expected_class,configuration=settings,metadata=metadata,
        chain_sha256=base.digest(reference_chain.JOURNAL),audit_sha256=base.digest(audit_path),
        independent_axes=True,empirical_reference_change=composed,
        screen_0p1=bool(audit["passes"]),limit_0p1=limit,
        screen_0p5=bool(composed<=protocol["gate"]*protocol["reference_budget_fraction"]),
        limit_0p5=protocol["gate"]*protocol["reference_budget_fraction"],
        evidence_note="Independent-grid comparisons estimate reference sensitivity. "
            "An unmet 0.1% screen does not support a 0.1% speed conclusion.")


def validate_trajectory(solution, horizon, snapshots, steps):
    samples = np.arange(0.,horizon+.001,.125)
    if (solution.row["steps"]!=steps or solution.row["horizon"]!=horizon or
            not np.array_equal(solution.times,snapshots) or
            not np.array_equal(solution.probe_times,samples) or
            solution.states.shape[0]!=len(snapshots) or
            solution.transverse.shape!=solution.vorticity.shape or
            solution.transverse.shape!=(len(samples),len(case.PROBES))):
        raise RuntimeError("Trajectory state or physical observation times differ")
    if any(not np.isfinite(array).all() for array in (
            solution.states,solution.transverse,solution.vorticity,solution.times,solution.probe_times)):
        raise FloatingPointError("Nonfinite trajectory data")


def load_reference(path, record, protocol):
    if record["reconstruction_class"]==reference_chain.radial_driver.CLASS:
        model = reference_chain.MODEL
        steps = record["spec"]["rate"]*protocol["horizon"]
    else:
        model = base.FullFD
        steps = record["metadata"]["steps"]
    solution = case.load_solution(path,model)
    validate_trajectory(solution,protocol["horizon"],protocol["checkpoints"],steps)
    if model is reference_chain.MODEL and solution.row["row_scale"]!=solution.model.scale_metadata():
        raise RuntimeError("Reassembled reference row-scaling fingerprint differs")
    if not base.wake(solution)["sustained_alternating_probe_signs"]:
        raise RuntimeError("The frozen reference did not reproduce sustained shedding")
    return solution


def matched_scales(protocol):
    lookup = {s["name"]:s for s in protocol["variants"]}
    pairs = []
    for scale, name in (("low", "rfm35_k64"), ("mid", "rfm37_k72"), ("high", "rfm41_k80")):
        spec = dict(lookup[name], scale=scale)
        pairs.append(spec)
        pairs.append(dict(spec, name=f"matched_leg{spec['m']}_k{spec['k']}", method="legendre"))
    return pairs + [dict(lookup["fd176_k80"], scale="mid")]


def solve(spec, settings, backend, *, horizon, snapshots, runtime=None):
    is_fd = spec["method"] == "fd4"
    steps = round(horizon*spec["rate"])
    args = dict(settings, steps=steps, horizon=horizon, snapshots=snapshots)
    if backend == "cpu":
        return case.solve(BufferedFD if is_fd else RecurrenceGalerkin, **args)
    graph_steps = GRAPH_STEPS if backend == "cuda_graph" else 0
    if is_fd:
        return solve_fd_native(**args, graph_steps=graph_steps, cuda_linear_solve="dense",
                               profile_steps=0,runtime=runtime)
    with native_model():
        return solve_native(**args, graph_steps=graph_steps, profile_steps=0,runtime=runtime)


def artifact_model(method):
    return base.FullFD if method == "fd4" else RecurrenceGalerkin


def record_run(job, directory, protocol, horizon, snapshots, runtime=None):
    spec = job["spec"]
    stem = f"block{job['block']:02d}_{job['backend']}_{spec['name']}_{job['seed']}"
    trajectory = directory / f"{stem}.npz"
    record = directory / f"{stem}.json"
    directory.mkdir(parents=True, exist_ok=True)
    if record.exists():
        row = base.read(record)
        if any(row.get(key) != value for key, value in job.items()):
            raise RuntimeError(f"Saved job differs from schedule: {record.name}")
        if "artifact" in row and (row["artifact"] != trajectory.name or base.digest(trajectory) != row["sha256"]):
            raise RuntimeError(f"Saved trajectory differs: {trajectory.name}")
        return row
    if trajectory.exists():
        trajectory.rename(trajectory.with_suffix(f".orphan_{time.time_ns()}.npz"))
    _, settings = base.config(spec, protocol, job["seed"])
    started = time.perf_counter()
    try:
        solution = solve(spec, settings, job["backend"], horizon=horizon, snapshots=snapshots, runtime=runtime)
        elapsed = time.perf_counter()-started
        metadata = solution.row
        validate_trajectory(solution,horizon,snapshots,round(horizon*spec["rate"]))
        if job["backend"] == "cpu":
            setup = metadata["setup_seconds"]
            march = metadata["seconds"]-setup
            download = 0.
        else:
            setup = sum(metadata.get(k,0.) for k in ("host_setup_seconds",
                "host_linear_precompute_seconds","runtime_init_seconds","upload_seconds",
                "plan_seconds","graph_setup_seconds"))
            march = metadata["march_seconds"]
            download = metadata["download_seconds"]
        tmp = trajectory.with_suffix(".tmp.npz")
        case.store_solution(tmp, solution, settings)
        tmp.replace(trajectory)
        residual = elapsed-setup-march-download
        if any(not math.isfinite(value) or value<0 for value in (elapsed,setup,march,download,residual)):
            raise RuntimeError("Timing components are invalid")
        row = dict(job, wall_seconds=elapsed, setup_seconds=setup,march_seconds=march,
                   download_seconds=download, residual_wall_seconds=residual,
                   metadata=metadata,
                   artifact=trajectory.name, sha256=base.digest(trajectory))
    except (ArithmeticError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        row = dict(job, failure=str(exc), failure_wall_seconds=time.perf_counter()-started)
    base.save(record, row)
    return row


def score_matches(rows, directory, reference, gate, horizon, snapshots):
    grids = base.grid_set()
    truth = case.reference_fields(reference, grids) if reference is not None else None
    key = lambda row:(row["block"], row["spec"]["name"], row["seed"])
    controls = {}
    for row in rows:
        if row["backend"] == "cpu" and "artifact" in row:
            sol = case.load_solution(directory/row["artifact"], artifact_model(row["spec"]["method"]))
            validate_trajectory(sol,horizon,snapshots,round(horizon*row["spec"]["rate"]))
            controls[key(row)] = (sol, case.reference_fields(sol, grids))
    for row in rows:
        if "artifact" not in row:
            continue
        pair = controls.get(key(row))
        if pair is None:
            row["implementation_pass"] = False
            continue
        previous, previous_fields = pair
        solution = (previous if row["backend"] == "cpu" else case.load_solution(
            directory/row["artifact"], artifact_model(row["spec"]["method"])))
        validate_trajectory(solution,horizon,snapshots,round(horizon*row["spec"]["rate"]))
        if not np.array_equal(solution.times, previous.times) or not np.array_equal(
                solution.probe_times, previous.probe_times):
            raise RuntimeError("Matched runs must share physical checkpoint and probe times")
        field_rows = case.score(solution, previous_fields, grids)
        row["same_discretization_scores"] = field_rows
        row["same_discretization_field_difference"] = checked_scores(field_rows,grids,solution.times)
        row["same_discretization_probe_max_abs"] = {
            name:float(np.max(np.abs(getattr(solution,name)-getattr(previous,name))))
            for name in ("transverse","vorticity")}
        probe_relative = {}
        for name in ("transverse", "vorticity"):
            reference_probe = getattr(previous,name)
            norm = float(np.linalg.norm(reference_probe))
            delta = float(np.linalg.norm(getattr(solution,name)-reference_probe))
            probe_relative[name] = delta/norm if norm>1e-12 else None
        row["same_discretization_probe_relative"] = probe_relative
        probe_pass = all(
            v<=MATCH_TOLERANCE if v is not None else row["same_discretization_probe_max_abs"][key]<=1e-10
            for key,v in probe_relative.items())
        row["implementation_pass"] = (max(row["same_discretization_field_difference"].values())
                                       <=MATCH_TOLERANCE and probe_pass)
        if truth is not None:
            row["reference_scores"] = case.score(solution, truth, grids)
            row["reference_errors"] = checked_scores(row["reference_scores"],grids,solution.times)
            row["wake"] = base.wake(solution)
            row["reference_accuracy_pass"] = (max(row["reference_errors"].values())<=gate and
                                               row["wake"]["sustained_alternating_probe_signs"])


def summarize(rows, specs, rounds):
    groups = []
    for spec in specs:
        for backend in BACKENDS:
            same = [r for r in rows if r["spec"]["name"] == spec["name"] and r["backend"] == backend]
            good = [r for r in same if "wall_seconds" in r]
            wall = [r["wall_seconds"] for r in good]
            setup = [r["setup_seconds"] for r in good]
            march = [r["march_seconds"] for r in good]
            groups.append(dict(spec=spec, backend=backend, count=len(same),
                failures=sum("failure" in r for r in same),
                matched_implementation_pass=len(same)==rounds and all(
                    r.get("implementation_pass",False) for r in same),
                wall_seconds=wall, setup_seconds=setup, march_seconds=march,
                wall_median=statistics.median(wall) if wall else None,
                setup_median=statistics.median(setup) if setup else None,
                march_median=statistics.median(march) if march else None,
                reference_accuracy_pass=len(good)==rounds and all(r.get("reference_accuracy_pass",False) for r in good)
                    if any("reference_accuracy_pass" in r for r in good) else None))
    return groups


def run_reuse(fresh, fresh_rows, directory, protocol, horizon, snapshots, repeats):
    if repeats < 2:
        raise ValueError("Amortization requires at least two solves")
    journal = directory / "reuse.json"
    if journal.exists():
        previous = base.read(journal)
        if previous["status"]=="complete" and previous["repeats_per_method"]==repeats:
            for group in previous["groups"]:
                for row in group["rows"]:
                    path = directory/"reuse"/group["spec"]["name"]/row["artifact"]
                    if base.digest(path)!=row["sha256"]:
                        raise RuntimeError(f"Reuse artifact differs: {path.name}")
            return
        raise FileExistsError("Inspect a partial reuse session before starting a fresh context")
    result = dict(status="running", groups=[], comparison="same-process context and module reuse; "
                  "model, transfers, FFT plans and graphs are built again every solve",
                  repeats_per_method=repeats)
    base.save(journal, result)
    for selected in (s for s in fresh if s["backend"]=="cuda_graph" and s["spec"]["scale"]=="mid"):
        spec = selected["spec"]
        runtime_type = FDRuntime if spec["method"] == "fd4" else Runtime
        seed = protocol["seeds"][0] if spec["method"]=="rfm" else protocol["deterministic_seed"]
        matching = [r for r in fresh_rows if r["spec"]==spec and r["backend"]=="cuda_graph"
                    and r["seed"]==seed and r.get("implementation_pass",False)]
        if not matching:
            continue
        control = matching[0]
        model = artifact_model(spec["method"])
        previous = case.load_solution(directory/control["artifact"], model)
        reuse_directory = directory/"reuse"/spec["name"]
        start = time.perf_counter()
        group = dict(spec=spec, seed=seed, rows=[], matching_fresh_samples_seconds=[
            r["wall_seconds"] for r in matching], matching_fresh_wall_median=statistics.median(
                r["wall_seconds"] for r in matching))
        result["groups"].append(group)
        with runtime_type() as runtime:
            group["runtime_initialization_seconds"] = time.perf_counter()-start
            for block in range(1,repeats+1):
                job = dict(spec=spec, seed=seed, backend="cuda_graph", block=block)
                row = record_run(job,reuse_directory,protocol,horizon,snapshots,runtime)
                if "artifact" in row:
                    actual = case.load_solution(reuse_directory/row["artifact"], model)
                    row["bitwise_fresh_match"] = all(np.array_equal(getattr(actual,key),getattr(previous,key))
                        for key in ("states","transverse","vorticity","times","probe_times"))
                group["rows"].append(row)
                base.save(journal, result)
            cleanup_started = time.perf_counter()
        group["runtime_cleanup_seconds"] = time.perf_counter()-cleanup_started
        group["session_elapsed_with_output_logging_seconds"] = time.perf_counter()-start
        values = [r["wall_seconds"] for r in group["rows"] if "wall_seconds" in r]
        if len(values) == repeats and all(r.get("bitwise_fresh_match",False) for r in group["rows"]):
            median = statistics.median(values)
            group["reused_wall_median"] = median
            difference = group["matching_fresh_wall_median"]-median
            lifecycle = group["runtime_initialization_seconds"] + group["runtime_cleanup_seconds"]
            group["break_even_solves_approx"] = math.ceil(lifecycle/difference) if difference>0 else None
        base.save(journal, result)
    complete = len(result["groups"])==3 and all(len(g["rows"])==repeats and all(
        r.get("bitwise_fresh_match",False) for r in g["rows"]) for g in result["groups"])
    result["status"] = "complete" if complete else "incomplete"
    base.save(journal, result)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, required=True)
    cli.add_argument("--mode", choices=("pilot", "isolated"), default="pilot")
    cli.add_argument("--confirm-isolated", action="store_true")
    cli.add_argument("--horizon", type=float, default=2.)
    cli.add_argument("--rounds", type=int, default=1)
    cli.add_argument("--reference", type=Path, help="Optional explicit path to the frozen pow2 time_fine reference")
    cli.add_argument("--reuse-repeats", type=int, default=0)
    cli.add_argument("--freeze-only", action="store_true")
    args = cli.parse_args()
    if args.mode == "isolated" and not (args.confirm_isolated or args.freeze_only):
        cli.error("Confirm all competing research processes are stopped")
    if not 1<=args.rounds<=5 or args.horizon<=0:
        cli.error("Require 1–5 paired rounds and a positive horizon")
    if args.mode=="isolated" and (args.horizon!=72. or args.rounds!=3 or args.reuse_repeats!=3):
        cli.error("The isolated scale matrix fixes t=72, three paired rounds, and three reuse repeats")
    if args.reference and args.horizon != 72.:
        cli.error("Full reference scoring requires all seven t=72 checkpoints")
    protocol = base.read(base.PROTOCOL)
    if len(protocol["seeds"])!=5 or len(set(protocol["seeds"]))!=5:
        raise RuntimeError("The original method protocol requires five distinct RFM seeds")
    scale_seeds = protocol["seeds"][:args.rounds]
    specs = matched_scales(protocol)
    snapshots = protocol["checkpoints"] if args.horizon == 72. else [args.horizon/2,args.horizon]
    pairs = [(backend,spec) for spec in specs for backend in BACKENDS]
    rng = np.random.default_rng(ORDER_SEED)
    jobs = []
    for block in range(1,args.rounds+1):
        for i in rng.permutation(len(pairs)):
            backend, spec = pairs[i]
            seed = scale_seeds[block-1] if spec["method"]=="rfm" else protocol["deterministic_seed"]
            jobs.append(dict(block=block,backend=backend,spec=spec,seed=seed))
    if len(jobs)!=len(specs)*len(BACKENDS)*args.rounds or len(specs)!=7:
        raise RuntimeError("The matched scale schedule must cover seven configurations and all backends")
    if args.reference or args.mode=="isolated":
        reference_path, reference_record = pow2_reference_record(args.reference,protocol)
        if args.mode=="isolated" and not reference_record["screen_0p5"]:
            raise RuntimeError("The independent reference failed the 0.5% empirical screen")
    else:
        reference_path, reference_record = None, None
    fixed = dict(mode=args.mode,horizon=args.horizon,snapshots=snapshots,rounds=args.rounds,
        rfm_seeds=scale_seeds,deterministic_seed=protocol["deterministic_seed"],
        schedule=jobs,source_hashes=settings_hashes(),reference=reference_record,
        reuse_repeats=args.reuse_repeats,
        graph_steps=GRAPH_STEPS,match_tolerance=MATCH_TOLERANCE,
        purpose="Matched configuration scaling and CUDA dispatch ablation. "
                "Three paired RFM seeds measure hardware trends and do not replace the five-seed method conclusion. "
                "Reference accuracy scores alone do not certify a cross-discretization speed claim.",
        timing_scope="Uninstrumented fresh in-process wall clock includes construction, setup, transfer, "
                    "graph or eager dispatch, synchronization, output and teardown. "
                    "No process-level cache reset. Profile graphs are separate diagnostics.")
    path = args.directory/"protocol.json"
    if path.exists() and base.read(path) != fixed:
        raise RuntimeError("Frozen scale inputs changed")
    if not path.exists():
        base.save(path, fixed)
    if args.freeze_only:
        print("FROZEN",len(jobs),path,flush=True)
        return
    reference = load_reference(reference_path,reference_record,protocol) if reference_path else None
    affinity = base.pin() if args.mode == "isolated" else None
    journal = args.directory/"timing.json"
    if journal.exists():
        result = base.read(journal)
        if result["protocol_sha256"]!=base.digest(path) or result["cpu"]!=affinity:
            raise RuntimeError("Frozen protocol or CPU affinity differs")
    else:
        result = dict(status="running", protocol_sha256=base.digest(path), rows=[], cpu=affinity,
                      gpu=matched.prior.nvidia(), platform=platform.platform())
    rows = []
    for job in jobs:
        row = record_run(job,args.directory,protocol,args.horizon,snapshots)
        rows.append(row)
        result.update(rows=rows,status="running")
        base.save(journal,result)
        print(job["block"],job["spec"]["name"],job["backend"],row.get("wall_seconds",row.get("failure")),flush=True)
    score_matches(rows,args.directory,reference,protocol["gate"],args.horizon,snapshots)
    if settings_hashes()!=fixed["source_hashes"] or (reference_path and
            pow2_reference_record(reference_path,protocol)[1]!=reference_record):
        raise RuntimeError("Benchmark input changed")
    result.update(status="complete", summary=summarize(rows,specs,args.rounds))
    base.save(journal,result)
    if args.reuse_repeats:
        run_reuse(result["summary"],rows,args.directory,protocol,args.horizon,snapshots,args.reuse_repeats)


if __name__ == "__main__":
    main()
