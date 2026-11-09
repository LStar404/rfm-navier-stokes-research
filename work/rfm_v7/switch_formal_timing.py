"""Draft isolated 60-arm comparison, one fresh Python child per solve."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time

import fd_early_multiscale as fd_qual
import switch_qualification as switch
from basis_recurrence_model import native_model
from fd_cpu_buffered import BufferedFD
from fd_cuda_native import solve_fd_native
from galerkin_cuda_native import solve_native


base, case, np, old = switch.base, switch.case, switch.np, switch.old
HERE = base.HERE
OUT = HERE/"switch_formal_timing_output"
SMOKE = HERE/"switch_formal_transport_smoke_output"
TEMPLATE = HERE/"protocol_switch_formal_timing_draft.json"
VERIFIER = HERE/"verify_switch_qualification.py"
VERIFICATION = HERE/"switch_qualification_verification.json"
WRAPPER = HERE/"run_switch_formal_timing_isolated.ps1"
FROZEN, TIMING, EXECUTION, MONITOR, SUMMARY = (OUT/name for name in (
    "protocol.json", "timing.json", "execution.json", "isolation.json", "summary.json"))
MODEL = switch.switched
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MARKER = b"SWITCH_SOLVE_RETURN "
TOKEN_ENV = "RFM_SWITCH_FORMAL_PROTOCOL_SHA256"
JOB_FIELDS = ("block", "name", "seed", "arm")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def path_record(path):
    return dict(path=path.relative_to(base.WORK).as_posix(), sha256=base.digest(path))


def check_hashes(fingerprints):
    root = base.WORK.resolve()
    for name, sha in fingerprints.items():
        path = base.WORK/name
        require(path.resolve().relative_to(root).as_posix() == name, f"Unsafe frozen path: {name}")
        old.sha_matches(path, sha)


def add_record(fingerprints, record):
    name, sha = record["path"], record["sha256"]
    require(fingerprints.get(name, sha) == sha, f"Conflicting frozen input: {name}")
    fingerprints[name] = sha
    return record


def artifact_record(row, directory, fingerprints):
    path = directory/row["artifact"]
    require(path.parent == directory and path.name == row["artifact"], "Unsafe trajectory basename")
    return add_record(fingerprints, dict(path=path.relative_to(base.WORK).as_posix(), sha256=row["sha256"]))


def draft_matches(p, s, f):
    t = s["template"]
    return (p["physical_times"] == s["physical_times"] == f["checkpoints"] and len(p["physical_times"]) == 19 and
            len(s["checkpoints"]) == 21 and p["families"] == t["methods"] and p["rfm_seeds"] == t["rfm_seeds"] and
            p["deterministic_seed"] == t["deterministic_seed"] == f["template"]["seed"] and
            p["horizon"] == t["horizon"] == f["template"]["horizon"] and
            p["probe_spacing"] == t["probe_spacing"] == f["template"]["probe_spacing"] and
            p["references"] == t["references"] == f["template"]["references"] and
            all(p[k] == t[k] == f["template"][k] for k in ("physical_gate", "implementation_gate",
                "reference_budget_fraction", "graph_steps")) and
            p["switch"] == {k:t[k] for k in ("startup_rate", "late_rate", "switch_time")} and
            p["uniform_rate"] == t["startup_rate"] == f["template"]["rate"] and
            p["cuda_fd_linear"] == f["template"]["cuda_linear_solve"] == "dense" and
            p["galerkin_arms"] == ["cpu_recurrence128", "cpu_native128", "cpu_native_switch", "cuda_graph128", "cuda_graph_switch"] and
            p["fd_arms"] == ["cpu_buffered128", "cuda_graph128"] and p["blocks"] == len(p["rfm_seeds"]) == 5)


def make_protocol():
    """Resolve finished qualification, immutable references, sources and block order."""
    p, s, f = (base.read(path) for path in (TEMPLATE, switch.FROZEN, fd_qual.FROZEN))
    require(draft_matches(p, s, f), "Draft differs from the accuracy-qualified grid, rates or arms")
    sr, sq, verified, fr, fq = (base.read(path) for path in (switch.RUNS, switch.SUMMARY, VERIFICATION,
                                                            fd_qual.RUNS, fd_qual.SUMMARY))
    require(s["status"] == f["status"] == "frozen" and sr["status"] == sq["status"] == fr["status"] == fq["status"] == "complete" and
            sr["passes"] and sq["passes"] and len(sr["rows"]) == 24 and len(sq["rows"]) == 18 and
            all(r["qualified"] for r in sq["rows"]) and len(sq["anchors"]) == 6 and
            verified["status"] == "passed" and verified["qualification_passes"] and verified["candidate_identities"] == 18 and
            verified["time_solves_run"] == 0, "Need independent archived verification of all switch trajectories")
    for source, raw_path, summary_path, raw, summary in ((switch.FROZEN,switch.RUNS,switch.SUMMARY,sr,sq),
                                                          (fd_qual.FROZEN,fd_qual.RUNS,fd_qual.SUMMARY,fr,fq)):
        require(raw["protocol_sha256"] == summary["protocol_sha256"] == base.digest(source) and
                summary["runs_sha256"] == base.digest(raw_path), f"Accuracy freeze differs: {summary_path.name}")
    require(verified["protocol_sha256"] == sq["protocol_sha256"] and verified["runs_sha256"] == sq["runs_sha256"] and
            verified["qualification_sha256"] == base.digest(switch.SUMMARY) and
            verified["verifier_source_sha256"] == base.digest(VERIFIER), "Independent switch evidence changed")
    reference_names = p["references"]
    for name in reference_names:
        require(f["references"]["early"][name] == s["early_references"][name], "The FD and Galerkin early references differ")
        late = f["references"]["late"][name]
        require(s["late_reference"]["reference_hashes"].get(late["path"]) == late["sha256"],
                "The FD and Galerkin late references differ")
    fd_rows = {r["job"]["arm"]: r for r in fq["rows"] if r["job"]["name"] == p["fd_name"]}
    require(set(fd_rows) == {"cpu_buffered", "cuda_graph"} and all(r["qualified"] for r in fd_rows.values()),
            "The preregistered FD176 CPU/GPU pair has not qualified at all 19 times")
    raw_fd = {(r["job"]["name"],r["job"]["arm"]):r for r in fr["rows"]}
    for arm, scored in fd_rows.items():
        raw = raw_fd[p["fd_name"], arm]
        require(raw["status"] == "complete" and raw["job"] == scored["job"] and
                (raw["artifact"], raw["sha256"]) == (scored["artifact"], scored["sha256"]) and
                raw["job"]["n"] == 176 and raw["job"]["k"] == 80 and raw["job"]["rate"] == 128,
                "FD176 score cites a different raw trajectory or resolution")
    require(all(r["physical"][ref]["early_undefined"] == 0 for r in fd_rows.values() for ref in reference_names),
            "A registered early physical error is undefined")
    sources = {}
    for mapping in (s["source_hashes"], f["source_hashes"]):
        for path, sha in mapping.items():
            add_record(sources, dict(path=path, sha256=sha))
    for path in (Path(__file__), TEMPLATE, WRAPPER, switch.FROZEN, switch.RUNS, switch.SUMMARY, VERIFIER,
                 VERIFICATION, fd_qual.FROZEN, fd_qual.RUNS, fd_qual.SUMMARY):
        add_record(sources, path_record(path))
    for phase in f["references"].values():
        for record in phase.values():
            add_record(sources, record)
    switched = {(r["job"]["name"], r["job"]["seed"]): r for r in sq["rows"] if r["job"]["arm"] == "cpu_recurrence"}
    controls = []
    physical_keys = {(g, t) for g in base.grid_set() for t in p["physical_times"]}
    for r in sq["anchors"]:
        name, seed = r["name"], r["seed"]
        require(all(max(old.score_rows(r["physical_scores"][ref], physical_keys).values()) <= p["physical_gate"]
                    for ref in reference_names), "A uniform rate128 CPU anchor misses the common-time physical gate")
        controls.append(dict(name=name, seed=seed, early=artifact_record(r, switch.OUT, sources),
                             late=add_record(sources, r["late_source"]),
                             switched=artifact_record(switched[name, seed], switch.OUT, sources),
                             baseline_scores=r["physical_scores"]))
    expected = {(name, seed) for name in p["families"] for seed in
                (p["rfm_seeds"] if name == p["families"][0] else [p["deterministic_seed"]])}
    require(len(controls) == 6 and {(r["name"],r["seed"]) for r in controls} == expected and len(switched) == 6,
            "Missing registered Galerkin controls")
    for row in sq["rows"]:
        artifact_record(row, switch.OUT, sources)
    for row in fd_rows.values():
        artifact_record(row, fd_qual.OUT, sources)
    check_hashes(sources)
    arms = [(name, arm) for name in p["families"] for arm in p["galerkin_arms"]]
    arms += [(p["fd_name"], arm) for arm in p["fd_arms"]]
    rng = np.random.default_rng(p["order_seed"])
    schedule = []
    for block, rfm_seed in enumerate(p["rfm_seeds"], 1):
        for index in rng.permutation(len(arms)):
            name, arm = arms[index]
            seed = rfm_seed if name == p["families"][0] else p["deterministic_seed"]
            schedule.append(dict(block=block, name=name, seed=seed, arm=arm))
    specs = {name:s["specs"][name] for name in p["families"]}
    require(len(schedule) == 60, "Expected five complete randomized twelve-arm blocks")
    return dict(status="frozen", template=p, source_hashes=sources, schedule=schedule,
                solve_protocol=s["solve_protocol"], specs=specs, gal_controls=controls,
                fd_config=fd_qual.config(fd_rows["cpu_buffered"]["job"], f["template"]),
                fd_control=artifact_record(fd_rows["cpu_buffered"], fd_qual.OUT, sources),
                fd_reference_protocol=f, isolation=p["isolation"])


def freeze():
    result = make_protocol()
    if FROZEN.exists():
        require(base.read(FROZEN) == result, "The existing formal freeze differs")
    else:
        base.save(FROZEN, result)
    return result


def split_cases(frozen):
    """Load scoring controls once in the coordinator outside solver timing."""
    p = frozen["template"]
    times = np.asarray(p["physical_times"])
    grids = base.grid_set()
    phases, _ = fd_qual.reference_fields(frozen["fd_reference_protocol"], grids)
    truth = {name:{g:phases["early"][name][g]+phases["late"][name][g] for g in grids} for name in p["references"]}

    def load(row, cls):
        return case.load_solution(base.WORK/row["path"], cls)

    controls = {}
    for row in frozen["gal_controls"]:
        identity = row["name"], row["seed"]
        early = load(row["early"], MODEL.RecurrenceGalerkin)
        late = switch.select(load(row["late"], MODEL.RecurrenceGalerkin), times[12:])
        selected = switch.select(load(row["switched"], MODEL.RecurrenceGalerkin), times)
        require(np.array_equal(early.times, times[:12]) and MODEL.equal_spatial_models((early.model, late.model, selected.model)),
                "Uniform and switched scoring controls use different spatial coordinates")
        uniform = case.Solution(late.model, np.concatenate((early.states, late.states)), times, late.probe_times,
                                late.transverse, late.vorticity, late.row)
        for kind, sol in (("uniform", uniform), ("switch", selected)):
            controls[(*identity, kind)] = (sol, case.reference_fields(sol, grids), model_signature(sol.model), row["baseline_scores"])
    sol = load(frozen["fd_control"], base.FullFD)
    require(np.array_equal(sol.times, times), "FD matched control omits prescribed physical times")
    controls[(p["fd_name"], p["deterministic_seed"], "uniform")] = (sol, case.reference_fields(sol, grids), model_signature(sol.model), None)
    return grids, truth, controls


def model_signature(model):
    """Check the exact spatial coordinates before using an archived scoring model."""
    if isinstance(model, base.FullFD):
        values = [model.s, model.r, model.right_mean, model.left_d1, model.right_d1]
        for matrix in (model.d1, model.d2, model.wall_d2[0]):
            values.extend((matrix.data, matrix.indices, matrix.indptr))
    else:
        names = ("transform", "mean_transform", "samples", "mean_samples", "M0", "M1", "M_lift",
                 "mean_mass", "test", "test_s", "mean_test", "r", "lift", "k2")
        values = [getattr(model, name) for name in names]
    result = hashlib.sha256()
    for value in values:
        result.update(repr((value.shape, value.dtype.str)).encode("ascii"))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def settings(job, frozen):
    name = job["name"]
    return (frozen["fd_config"] if name == frozen["template"]["fd_name"] else
            base.config(frozen["specs"][name], frozen["solve_protocol"], job["seed"])[1])


def solve(job, frozen):
    p = frozen["template"]
    name, arm = job["name"], job["arm"]
    config = settings(job, frozen)
    args = dict(horizon=p["horizon"], snapshots=p["physical_times"], probe_spacing=p["probe_spacing"], **config)
    steps = p["uniform_rate"]*p["horizon"]
    graph = dict(graph_steps=p["graph_steps"])
    if name == p["fd_name"]:
        if arm == "cpu_buffered128":
            solution = case.solve(BufferedFD, steps=steps, **args)
        else:
            solution = solve_fd_native(steps=steps, cuda_linear_solve=p["cuda_fd_linear"], **graph, **args)
    elif arm == "cpu_recurrence128":
        solution = case.solve(MODEL.RecurrenceGalerkin, steps=steps, **args)
    elif arm == "cpu_native128":
        solution = case.solve(MODEL.NativePrecontractedGalerkin, steps=steps, **args)
    elif arm == "cpu_native_switch":
        solution = MODEL.solve_cpu_switch(MODEL.NativePrecontractedGalerkin, **p["switch"], **args)
    elif arm == "cuda_graph128":
        with native_model():
            solution = solve_native(steps=steps, **graph, **args)
    elif arm == "cuda_graph_switch":
        solution = switch.solve_cuda_switch(**p["switch"], **graph, **args)
    else:
        raise ValueError(f"Unknown formal arm: {arm}")
    return solution, config


def stem(job):
    return "b{block}_{name}_{arm}_{seed}".format(**job)


def emit_marker(payload):
    sys.stdout.buffer.write(MARKER+json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("ascii")+b"\n")
    sys.stdout.buffer.flush()


def affinity_mask():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    address = ctypes.POINTER(ctypes.c_size_t)
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, address, address]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return process.value


def child(index, output=OUT):
    protocol_path = output/"protocol.json"
    frozen = base.read(protocol_path)
    require(os.environ.get(TOKEN_ENV) == base.digest(protocol_path), "Child requires its matching run protocol")
    job = frozen["schedule"][index]
    target = output/(stem(job)+".npz")
    require(not target.exists(), "A timed child cannot replace an existing trajectory")
    inherited = affinity_mask()
    cpu = base.pin()
    started = time.perf_counter()
    try:
        solution, config = solve(job, frozen)
    except (ArithmeticError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        emit_marker(dict(status="failed", index=index, pid=os.getpid(), failure_type=type(exc).__name__, failure=str(exc)))
        return 1
    core = time.perf_counter()-started
    signature = model_signature(solution.model)
    release_started = time.perf_counter()
    solution.model = None
    released = time.perf_counter()-release_started
    emit_marker(dict(status="solved", index=index, pid=os.getpid(), logical_cpu=cpu, inherited_affinity_mask=inherited,
                     solver_seconds=core+released,
                     solver_core_seconds=core, model_release_seconds=released, spatial_sha256=signature))
    require(sys.stdin.buffer.readline() == b"SAVE\n", "Coordinator did not acknowledge the completion marker")
    temp = target.with_suffix(".tmp.npz")
    case.store_solution(temp, solution, config)
    temp.replace(target)
    return 0


def read_trajectory(path, model, expected_config):
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved["config"]))
        require(config == json.loads(json.dumps(expected_config)), "Child wrote a different registered solver configuration")
        return case.Solution(model, saved["states"], saved["times"], saved["probe_times"],
                             saved["transverse"], saved["vorticity"], json.loads(str(saved["metadata"])))


def score_solution(solution, job, control, grids, truth, p):
    baseline, fields, _, guard_input = control
    keyset = {(grid, t) for grid in grids for t in p["physical_times"]}
    require(np.array_equal(solution.times, p["physical_times"]) and np.array_equal(solution.times, baseline.times) and
            np.array_equal(solution.probe_times, baseline.probe_times) and
            solution.states.shape == baseline.states.shape and solution.transverse.shape == baseline.transverse.shape and
            solution.vorticity.shape == baseline.vorticity.shape and
            all(np.isfinite(getattr(solution, k)).all() for k in ("states", "transverse", "vorticity")),
            "Timed trajectory shape, absolute observations or finite-state check failed")
    is_switch = job["arm"].endswith("_switch")
    steps = 8192 if is_switch else 9216
    row = solution.row
    require(row["steps"] == steps and row["horizon"] == p["horizon"] and row["dt"] == 1/128,
            "Timed trajectory differs from the registered schedule")
    if is_switch:
        require(row["switch_step"] == 1024 and row["late_rate"] == 112 and row["nonlinear_evaluations"] == 8191 and
                row["marching_fft_batches"] == 16382 and row["nonlinear_by_phase"] == dict(startup=1024, late=7167),
                "Measured switch work differs from qualification")
    if job["arm"].startswith("cuda_"):
        require(row["graph_steps"] == p["graph_steps"], "CUDA graph chunk changed")
        if is_switch:
            require(row["bridge_extra_fft_batches"] == 0 and row["priming_fft_batches"] == 2 and row["device_final_step"] == steps,
                    "GPU bridge FFT or execution inventory changed")
        if job["name"] == p["fd_name"]:
            require(row["cuda_linear_solve"] == p["cuda_fd_linear"], "GPU FD linear arm changed")
    physical = {name:case.score(solution, target, grids) for name, target in truth.items()}
    maxima = {name:old.score_rows(entries, keyset) for name, entries in physical.items()}
    implementation = case.score(solution, fields, grids)
    implementation_worst = old.score_rows(implementation, keyset)
    probes = {}
    for name in ("transverse", "vorticity"):
        a, b = getattr(solution, name), getattr(baseline, name)
        norm = np.linalg.norm(b)
        require(norm > 1e-12, "Matched implementation probe is too small")
        probes[name] = float(np.linalg.norm(a-b)/norm)
    wake = base.wake(solution)
    guard = {}
    if is_switch:
        # Qualification already guards the schedule relative to uniform r128.
        # Guard the timed replay against the same uniform physical fields too.
        uniform_fields, uniform_scores = guard_input
        delta = case.score(solution, uniform_fields, grids)
        old.score_rows(delta, keyset)
        for ref, records in uniform_scores.items():
            indexed = {(r["grid"],r["time"]):r for r in records}
            guard[ref] = [dict(grid=d["grid"],time=d["time"],**{
                k:d[k]*(1+indexed[d["grid"],d["time"]][k])+indexed[d["grid"],d["time"]][k]
                for k in switch.METRICS}) for d in delta]
    guard_worst = {name:old.score_rows(rows, keyset) for name, rows in guard.items()}
    imp_pass = all(math.isfinite(x) and 0 <= x <= p["implementation_gate"] for x in (*implementation_worst.values(), *probes.values()))
    phy_pass = all(max(values.values()) <= p["physical_gate"] for values in maxima.values())
    guard_pass = all(max(values.values()) <= p["physical_gate"] for values in guard_worst.values())
    return dict(physical_scores=physical, physical_worst=maxima, implementation_scores=implementation,
                implementation_worst=implementation_worst, implementation_probes=probes, guard_scores=guard,
                guard_worst=guard_worst, wake=wake, implementation_pass=imp_pass, physical_pass=phy_pass,
                guard_pass=guard_pass, qualified=bool(imp_pass and phy_pass and guard_pass and wake["sustained_alternating_probe_signs"]),
                work=dict(steps=steps, nonlinear_evaluations=steps-1, marching_fft_batches=2*(steps-1),
                          source="measured_switch_kernel" if is_switch else "uniform_registered_kernel"))


def launch_child(row, index, artifact, cpu, *, output=OUT, action="--child", environment=None):
    name = artifact.stem
    marker, returned = None, None
    with (output/(name+".stderr.log")).open("xb") as errors, (output/(name+".stdout.log")).open("xb") as raw:
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), action, str(index)]
        launched = time.perf_counter_ns()
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, env=environment,
                              creationflags=subprocess.CREATE_NO_WINDOW) as process:
            row["child_pid"] = process.pid
            for line in process.stdout:
                if line.startswith(MARKER) and marker is None:
                    returned = time.perf_counter_ns()
                    marker = json.loads(line[len(MARKER):])
                    if marker["status"] == "solved":
                        require(not artifact.exists(), "Child serialized before its measured completion marker")
                        process.stdin.write(b"SAVE\n")
                        process.stdin.flush()
                    process.stdin.close()
                raw.write(line)
            code = process.wait()
    row.update(child_exit_code=code, marker=marker, parent_popen_monotonic_ns=launched, parent_marker_monotonic_ns=returned)
    require(marker is not None and marker["pid"] == process.pid and marker["index"] == index,
            "Missing or mismatched flushed child completion marker")
    if marker["status"] != "solved":
        raise RuntimeError(marker.get("failure", "Child solve failed"))
    row.update(solver_seconds=marker["solver_seconds"], solver_core_seconds=marker["solver_core_seconds"],
               model_release_seconds=marker["model_release_seconds"], fresh_process_return_seconds=(returned-launched)/1e9)
    row["fresh_process_overhead_seconds"] = row["fresh_process_return_seconds"]-row["solver_seconds"]
    require(code == 0 and artifact.is_file() and marker["logical_cpu"] == cpu and marker["inherited_affinity_mask"] == 1<<cpu and
            row["fresh_process_return_seconds"] >= row["solver_seconds"] > 0, "Child completion, affinity, save or clock check failed")
    return marker


def run_job(index, frozen, grids, truth, controls, cpu):
    job = frozen["schedule"][index]
    name = stem(job)
    artifact = OUT/(name+".npz")
    record = OUT/(name+".json")
    require(not record.exists() and not artifact.exists(), "A formal job cannot be retried or resumed")
    row = dict(job, index=index, qualified=False)
    try:
        marker = launch_child(row, index, artifact, cpu)
        p = frozen["template"]
        kind = "switch" if job["arm"].endswith("_switch") else "uniform"
        control = controls[job["name"], job["seed"], kind]
        if kind == "switch":
            _, uniform_fields, _, baseline_scores = controls[job["name"], job["seed"], "uniform"]
            control = (*control[:3], (uniform_fields, baseline_scores))
        require(marker["spatial_sha256"] == control[2], "Timed basis differs from the archived scoring coordinates")
        solution = read_trajectory(artifact, control[0].model, settings(job, frozen))
        row.update(artifact=artifact.name, sha256=base.digest(artifact), metadata=solution.row)
        row.update(score_solution(solution, job, control, grids, truth, p))
        row["status"] = "complete"
    except (ArithmeticError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        if artifact.exists() and "artifact" not in row:
            row.update(artifact=artifact.name, sha256=base.digest(artifact))
        row.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
    base.save(record, row)
    return row


def smoke_transport():
    """Exercise the actual marker and save handshake on one independent T9 solve."""
    require(not FROZEN.exists() and not EXECUTION.exists() and not SMOKE.exists(),
            "The short transport smoke requires its own unused directory before formal freeze")
    s = base.read(switch.FROZEN)
    p = dict(base.read(TEMPLATE), horizon=9, physical_times=[t for t in MODEL.SNAPSHOTS if t <= 9])
    job = dict(block=0, name="rfm37_k72", seed=7331, arm="cpu_native_switch")
    smoke = dict(status="transport_smoke", template=p, schedule=[job], specs=s["specs"], solve_protocol=s["solve_protocol"])
    protocol_path = SMOKE/"protocol.json"
    base.save(protocol_path, smoke)
    environment = dict(os.environ)
    environment[TOKEN_ENV] = base.digest(protocol_path)
    environment.update({key:"1" for key in THREADS})
    cpu = base.pin()
    artifact = SMOKE/(stem(job)+".npz")
    result = dict(status="running", job=job)
    try:
        launch_child(result, 0, artifact, cpu, output=SMOKE, action="--smoke-child", environment=environment)
        solution = read_trajectory(artifact, None, settings(job, smoke))
        require(np.array_equal(solution.times, p["physical_times"]) and solution.row["switch_step"] == 1024 and
                solution.row["steps"] == 1136 and solution.row["nonlinear_evaluations"] == 1135 and
                np.isfinite(solution.states).all(), "The T9 transport did not complete the actual bridge")
        result.update(status="passed", artifact=artifact.name, sha256=base.digest(artifact),
                      snapshots=len(solution.times), measured_steps=solution.row["steps"],
                      marker_before_serialization=True, scope="Short transport exercise. Neither clock is a formal sample.")
    except BaseException:
        result["status"] = "failed"
        raise
    finally:
        base.save(SMOKE/"transport.json", result)
    print("TRANSPORT", result["status"], result["solver_seconds"], result["fresh_process_return_seconds"], flush=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def execute():
    frozen = base.read(FROZEN)
    sha = base.digest(FROZEN)
    require(os.environ.get(TOKEN_ENV) == sha and all(os.environ.get(key) == "1" for key in THREADS),
            "Use the frozen isolation wrapper with one BLAS thread")
    require(not EXECUTION.exists() and not TIMING.exists(), "Only one formal timing attempt is registered")
    execution = dict(status="running", pid=os.getpid(), start_utc=utc(), protocol_sha256=sha)
    base.save(EXECUTION, execution)
    try:
        check_hashes(frozen["source_hashes"])
        cpu = base.pin()
        grids, truth, controls = split_cases(frozen)
        result = dict(status="running", protocol_sha256=sha, parent_pid=os.getpid(), logical_cpu=cpu,
                      gpu=old.old_joint.nvidia(), platform=platform.platform(), python_version=platform.python_version(),
                      environment={key:os.environ[key] for key in THREADS}, rows=[])
        base.save(TIMING, result)
        for i, job in enumerate(frozen["schedule"]):
            row = run_job(i, frozen, grids, truth, controls, cpu)
            result["rows"].append(row)
            base.save(TIMING, result)
            print(i+1, job, row["status"], row.get("solver_seconds"), row.get("fresh_process_return_seconds"), flush=True)
        check_hashes(frozen["source_hashes"])
        old.sha_matches(FROZEN, sha)
        result.update(status="complete", numerical_qualified=all(r["qualified"] for r in result["rows"]))
        base.save(TIMING, result)
        execution["status"] = "complete"
    except BaseException:
        execution["status"] = "failed"
        raise
    finally:
        execution["end_utc"] = utc()
        base.save(EXECUTION, execution)


def statistics(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values), samples=values) if values else None


def monitor_status(monitor, execution, isolation, rows):
    targets = list(range(isolation["sample_seconds"], isolation["window_seconds"]+1, isolation["sample_seconds"]))
    samples, preflight = monitor["samples"], monitor["preflight"]
    require([r["target_elapsed_seconds"] for r in samples] == targets and
            len(preflight) == isolation["preflight_windows"], "The isolation schedule does not cover the full window")
    start, deadline, end = [datetime.fromisoformat(monitor[k]) for k in ("start_utc", "planned_end_utc", "end_utc")]
    worker_start, worker_end = [datetime.fromisoformat(execution[k]) for k in ("start_utc", "end_utc")]
    within = (execution["status"] == "complete" and start <= worker_start <= worker_end <= deadline)
    require((deadline-start).total_seconds() == isolation["window_seconds"] and end >= deadline and
            within == monitor["solver_finished_within_window"], "Monitor and coordinator disagree on the complete window")
    t = isolation["thresholds"]

    def active(sample):
        background = [r["cpu_seconds"] for r in sample["external_python"] if r["label"] != "python"]
        unknown = [r["cpu_seconds"] for r in sample["external_python"] if r["label"] == "python"]
        return (any(x > t["background_individual_cpu_seconds"] for x in background) or
                sum(background) > t["background_total_cpu_seconds"] or
                any(x > t["unexpected_python_cpu_seconds"] for x in unknown))

    require(all(not active(r) and r["gpu_percent"] <= t["preflight_gpu_percent"] and
                all(p["label"] != "python" for p in r["external_python"]) for r in preflight),
            "The monitored attempt did not pass its quiet preflight")
    expected_activity = any(active(r) for r in samples)
    expected_overlap = any(len(r["cohort_children"]) > 1 for r in samples)
    require(expected_activity == monitor["external_activity_detected"] and expected_overlap == monitor["child_overlap_detected"],
            "External or overlapping child flags disagree with sampled process evidence")
    allowed = {(row["child_pid"], row["index"]) for row in rows if "child_pid" in row}
    if execution["status"] == "complete":
        require(all({(r["pid"],r["slot"]) for r in sample["cohort_child_slots"]} <= allowed and
                    sample["cohort_children"] == [r["pid"] for r in sample["cohort_child_slots"]] and
                    (sample["cohort_coordinator_present"] or not sample["cohort_children"]) for sample in samples),
                "Monitor excluded an unregistered cohort child")
    coverage = monitor["sample_coverage_pass"]
    if coverage:
        require(monitor["final_gpu_percent"] is not None and monitor["coordinator_creation_utc_ticks"] is not None and all(
            not r["snapshot_failure_type"] and -1e-6 <= r["elapsed_seconds"]-target <= t["maximum_sample_lateness_seconds"]+1e-6
            for r,target in zip(samples,targets,strict=True)), "Coverage flag differs from raw observation times")
    expected = ("failed" if monitor["exit_code"] != 0 else "completed_outside_fixed_window" if not (within and coverage)
                else "completed_with_overlapping_children" if expected_overlap
                else "completed_with_observed_external_activity" if expected_activity
                else "completed_no_sampled_external_activity")
    require(monitor["status"] == expected, "Monitor status differs from the frozen isolation rules")
    return expected


def summarize():
    p = base.read(FROZEN)
    timing, monitor, execution = (base.read(path) for path in (TIMING, MONITOR, EXECUTION))
    sha = base.digest(FROZEN)
    rows, schedule = timing["rows"], p["schedule"]
    require(len(rows) <= len(schedule) == 60 and
            all({k:r[k] for k in JOB_FIELDS} == job for r, job in zip(rows, schedule)) and
            timing["protocol_sha256"] == monitor["protocol_sha256"] == execution["protocol_sha256"] == sha and
            monitor["execution_sha256"] == base.digest(EXECUTION) and
            timing["parent_pid"] == monitor["pid"] == execution["pid"], "Incomplete formal schedule or monitoring bindings")
    isolation = monitor_status(monitor, execution, p["isolation"], rows)
    require(not SUMMARY.exists(), "Preserve the existing formal timing summary")
    hashes = dict(protocol_sha256=sha, timing_sha256=base.digest(TIMING), monitor_sha256=base.digest(MONITOR),
                  execution_sha256=base.digest(EXECUTION))
    for row in rows:
        path = OUT/(stem(row)+".json")
        require(base.read(path) == row, "A saved timing row differs from the journal")
        if "artifact" in row:
            old.sha_matches(OUT/row["artifact"], row["sha256"])
    if timing["status"] != "complete" or execution["status"] != "complete" or len(rows) != 60:
        base.save(SUMMARY, dict(status="diagnostic_incomplete", rows=len(rows), isolated_speed_claim_supported=False,
                                monitor_status=isolation, **hashes))
        return
    numerical = all(r["qualified"] for r in rows)
    clean = isolation == "completed_no_sampled_external_activity"
    groups = {}
    clocks = ("solver_seconds", "fresh_process_return_seconds")
    for row in rows:
        groups.setdefault((row["name"], row["arm"]), []).append(row)
    require(len(groups) == 12 and all(len(rows) == 5 for rows in groups.values()), "Need exactly five observations per arm")
    summary = []
    for (name, arm), subset in groups.items():
        valid = [r for r in subset if r["qualified"]]
        phases = sorted(set.intersection(*(set(r["metadata"]) for r in valid))) if len(valid) == 5 else []
        summary.append(dict(name=name, arm=arm, qualified=len(valid), phase_seconds={
            key:statistics([r["metadata"][key] for r in valid]) for key in phases if key.endswith("_seconds")}, **{
            key:statistics([r[key] for r in valid]) if len(valid) == 5 else None
            for key in (*clocks,"solver_core_seconds","model_release_seconds","fresh_process_overhead_seconds")}))
    t = p["template"]
    pairs = []
    for name in t["families"]:
        pairs.extend([((name,a),(name,b)) for a,b in (("cpu_recurrence128","cpu_native128"),
                     ("cpu_native128","cpu_native_switch"),("cuda_graph128","cuda_graph_switch"))])
        pairs.extend([((t["fd_name"],a),(name,b)) for a,b in (("cpu_buffered128","cpu_native128"),
                     ("cpu_buffered128","cpu_native_switch"),("cuda_graph128","cuda_graph128"),
                     ("cuda_graph128","cuda_graph_switch"))])
    rfm, leg = t["families"]
    pairs.extend([((leg,arm),(rfm,arm)) for arm in t["galerkin_arms"]])
    paired = []
    for a, b in pairs:
        x, y = sorted(groups[a],key=lambda r:r["block"]), sorted(groups[b],key=lambda r:r["block"])
        stats = {}
        for clock in clocks:
            values = [u[clock]/v[clock] for u,v in zip(x,y,strict=True)] if all(r["qualified"] for r in x+y) else []
            stats[clock] = dict(statistics(values), denominator_wins=sum(v>1 for v in values)) if values else None
        paired.append(dict(numerator=a, denominator=b, **stats))
    result = dict(status="passed" if numerical and clean else "diagnostic_only", rows=60,
                  isolated_speed_claim_supported=bool(numerical and clean), numerical_qualified=numerical,
                  monitor_status=isolation, groups=summary, paired=paired, **hashes)
    base.save(SUMMARY, result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-only", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--summarize", action="store_true")
    actions.add_argument("--child", type=int)
    actions.add_argument("--smoke-transport", action="store_true")
    actions.add_argument("--smoke-child", type=int, help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.freeze_only:
        print("FROZEN", len(freeze()["schedule"]), flush=True)
    elif args.execute:
        execute()
    elif args.summarize:
        summarize()
    elif args.smoke_transport:
        smoke_transport()
    elif args.smoke_child is not None:
        raise SystemExit(child(args.smoke_child, SMOKE))
    else:
        raise SystemExit(child(args.child))
