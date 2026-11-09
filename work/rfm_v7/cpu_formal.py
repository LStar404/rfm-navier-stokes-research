"""Freeze, qualify and time the v7 CPU comparison with a checked reference ladder."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
V6 = WORK / "rfm_v6"
OUT = HERE / "cpu_formal_output"
PROTOCOL = HERE / "protocol_cpu.json"
sys.path.extend([str(V6), str(WORK / "rfm_v5")])

import full_case as case
from diagnostic_wake import analyze
from fd_full import FullFD
from full_formal import boundary_errors
from full_galerkin_buffered import BufferedGalerkin
from timing_v5 import pin


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def source_hashes():
    sources = [PROTOCOL, Path(__file__)]
    sources += [V6 / n for n in (
        "full_case.py", "diagnostic_wake.py", "full_formal.py", "fd_full.py",
        "fd_full_band.c", "full_galerkin.py", "full_galerkin_buffered.py",
        "full_nonlinear.py", "radial_layers.py")]
    native = V6 / ("fd_full_band.dll" if sys.platform == "win32" else "fd_full_band.so")
    sources.append(native)
    sources += [WORK / "rfm_v3" / n for n in ("cylinder_case.py", "cylinder_rfm.py", "cylinder_fd.py")]
    sources.append(WORK / "rfm_v5" / "timing_v5.py")
    return {path.relative_to(WORK).as_posix(): digest(path) for path in sources}


def grid_set():
    grids = case.evaluation_grids()
    grids["layer144"] = case.layer_scoring_grid()
    return grids


def config(spec, protocol, seed=None):
    if "n" in spec:
        return FullFD, dict(n=spec["n"], k=spec["k"], stretch=1.0, grid="outer",
                            angular="fft", linear="native")
    return BufferedGalerkin, dict(m=spec["m"], k=spec["k"], q=spec["q"], scale=62.0,
        mean_order=24, layers=protocol["layers"], split=True, backend="fft",
        seed=seed, family=spec["method"])


def wake(solution):
    return analyze(solution.probe_times, solution.transverse, solution.vorticity)


def reference_step(protocol):
    """Compute the one missing independent-axis reference, preserving existing data."""
    spec = protocol["reference_path"][2]
    target = WORK / spec["source"]
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cls, settings = config(spec, protocol)
    sol = case.solve(cls, steps=protocol["horizon"] * spec["rate"], **settings)
    temp = target.with_suffix(".tmp.npz")
    case.store_solution(temp, sol, settings)
    temp.replace(target)
    print(f"reference angular_fine saved in {sol.row['seconds']:.3f}s", flush=True)


def open_reference(spec, protocol):
    path = WORK / spec["source"]
    with case.np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"]))
        settings = json.loads(str(artifact["config"]))
    _, expected = config(spec, protocol)
    if settings != expected or metadata["steps"] != protocol["horizon"] * spec["rate"]:
        raise ValueError(f"Unexpected reference configuration {path.name}")
    sol = case.load_solution(path, FullFD)
    if not case.np.array_equal(sol.times, protocol["checkpoints"]):
        raise ValueError(f"Unexpected reference checkpoint times {path.name}")
    if not wake(sol)["sustained_alternating_probe_signs"]:
        raise ValueError(f"Expected a sustained full wake in {path.name}")
    return sol


def reference_audit(protocol, allow_missing=False):
    specs = protocol["reference_path"]
    if allow_missing and not (WORK / specs[2]["source"]).exists():
        selected, axes = [specs[0], specs[1], specs[3]], ("radial", "angle_and_time")
    else:
        selected, axes = specs, ("radial", "angle", "time")
    solutions = [open_reference(s, protocol) for s in selected]
    grids = grid_set()
    edges = []
    compound = 1.0
    for axis, left, right in zip(axes, solutions, solutions[1:]):
        scores = case.score(left, case.reference_fields(right, grids), grids)
        worst = case.maxima(scores)
        compound *= 1.0 + max(worst.values())
        edges.append(dict(axis=axis, worst=worst, scores=scores))
    truth = case.reference_fields(solutions[-1], grids)
    old = case.load_solution(WORK / protocol["legacy_reference"], FullFD)
    legacy = case.maxima(case.score(old, truth, grids))
    direct = case.maxima(case.score(solutions[0], truth, grids))
    hashes = {s["source"]: digest(WORK / s["source"]) for s in selected}
    hashes[protocol["legacy_reference"]] = digest(WORK / protocol["legacy_reference"])
    threshold = protocol["gate"] * protocol["reference_budget_fraction"]
    status = dict(edges=edges, composed_difference=compound - 1.0,
        legacy_to_v7=legacy, direct_coarse_to_fine=direct, threshold=threshold,
        reference_hashes=hashes, independent_axes=len(axes)==3,
        passes=len(axes)==3 and compound - 1.0<=threshold, wake=wake(solutions[-1]))
    return status, truth


_truth = None
_protocol = None


def init_worker(protocol):
    global _truth, _protocol
    _protocol = protocol
    _truth = case.reference_fields(open_reference(protocol["reference_path"][-1], protocol), grid_set())


def run(job):
    phase, sequence, spec, seed = job
    cls, settings = config(spec, _protocol, seed)
    row = dict(phase=phase, sequence=sequence, name=spec["name"], method=spec["method"], seed=seed)
    path = OUT / f"{phase}_{sequence:02d}_{spec['name']}_{seed}.npz"
    record_path = path.with_suffix(".json")
    if record_path.exists():
        saved = read(record_path)
        if any(saved[key] != value for key, value in row.items()):
            raise RuntimeError(f"Checkpoint job changed for {record_path.name}")
        if "artifact" in saved and (saved["artifact"]!=path.name or digest(path)!=saved["sha256"]):
            raise RuntimeError(f"Checkpoint artifact changed for {path.name}")
        return saved
    if path.exists():
        path.rename(path.with_suffix(f".orphan_{time.time_ns()}.npz"))
    start = time.perf_counter()
    try:
        with case.np.errstate(over="ignore", invalid="ignore"):
            sol = case.solve(cls, steps=_protocol["horizon"] * spec["rate"], **settings)
        elapsed = time.perf_counter() - start
        scores = case.score(sol, _truth, grid_set())
        errors = case.maxima(scores)
        diagnostics = wake(sol)
        temp = path.with_suffix(".tmp.npz")
        case.store_solution(temp, sol, settings)
        temp.replace(path)
        row.update(metadata=sol.row, scores=scores, errors=errors, wake=diagnostics,
            boundary_errors=boundary_errors(sol), artifact=path.name, sha256=digest(path),
            numerical_pass=max(errors.values())<=_protocol["gate"] and
                diagnostics["sustained_alternating_probe_signs"])
        row["seconds" if phase=="timing" else "concurrent_diagnostic_seconds"] = elapsed
    except (ArithmeticError, ValueError, case.np.linalg.LinAlgError) as exc:
        row.update(failure=str(exc), numerical_pass=False)
    save(record_path,row)
    return row


def journal(path, phase, protocol, jobs, reference, cpu=None):
    fixed = dict(phase=phase, protocol=protocol, jobs=jobs, source_hashes=source_hashes(),
                 reference=reference, cpu=cpu, machine=platform.platform())
    if path.exists():
        previous = read(path)
        if previous["frozen"] != json.loads(json.dumps(fixed)):
            raise RuntimeError("Frozen plan, reference or implementation changed")
        for row in previous["rows"]:
            if "artifact" in row and digest(OUT / row["artifact"]) != row["sha256"]:
                raise RuntimeError(f"Checkpoint hash changed for {row['artifact']}")
        return previous
    record = dict(status="running", frozen=fixed, rows=[])
    save(path, record)
    return record


def summarize(rows, phase, protocol):
    summary = []
    for spec in protocol["variants"]:
        group = [r for r in rows if r["name"] == spec["name"]]
        if not group:
            continue
        passed = bool(all(r["numerical_pass"] for r in group))
        worst = {key:max(r["errors"][key] for r in group if "errors" in r)
                 for key in ("velocity","vorticity","near_velocity","near_vorticity")} if any(
                     "errors" in r for r in group) else None
        entry = dict(name=spec["name"], method=spec["method"], numerical_pass=passed,
                     worst=worst, count=len(group), failures=sum("failure" in r for r in group))
        if phase=="timing":
            times = [r["seconds"] for r in group if "seconds" in r]
            entry.update(samples_seconds=times, median_seconds=statistics.median(times) if times else None,
                         min_seconds=min(times) if times else None, max_seconds=max(times) if times else None)
        summary.append(entry)
    return summary


def qualify(protocol, workers, new_seeds_only=False):
    reference, _ = reference_audit(protocol, allow_missing=True)
    jobs = []
    for spec in protocol["variants"]:
        seeds = protocol["seeds"] if spec["method"]=="rfm" else [protocol["deterministic_seed"]]
        jobs.extend(("qualify", 0, spec, seed) for seed in seeds)
    path = OUT / "qualification.json"
    if path.exists():
        original = read(path)["frozen"]["reference"]
        if any(reference["reference_hashes"].get(name)!=sha for name,sha in
               original["reference_hashes"].items()):
            raise RuntimeError("A qualification reference changed")
        reference = original
    result = journal(path, "qualify", protocol, jobs, reference)
    known = {(r["name"],r["seed"]) for r in result["rows"]}
    pending = [j for j in jobs if (j[2]["name"],j[3]) not in known]
    if new_seeds_only:
        pending = [j for j in pending if j[2]["name"] in ("rfm37_k72", "rfm41_k80")
                   and j[3] in protocol["seeds"][-2:]]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(protocol,)) as pool:
        for future in as_completed([pool.submit(run, j) for j in pending]):
            row = future.result()
            result["rows"].append(row)
            save(path, result)
            print(row["name"], row["seed"], row.get("errors",row.get("failure")), flush=True)
    status = "complete" if len(result["rows"])==len(jobs) else "incomplete"
    result.update(status=status, summary=summarize(result["rows"], "qualify", protocol))
    save(path, result)


def timing(protocol):
    qualification = read(OUT / "qualification.json")
    if qualification["status"]!="complete":
        raise RuntimeError("Complete the preregistered qualification first")
    reference, _ = reference_audit(protocol)
    if not reference["passes"]:
        raise RuntimeError("The independent-axis reference screen is required")
    frozen = qualification["frozen"]
    if (frozen["protocol"]!=protocol or frozen["source_hashes"]!=source_hashes() or
            any(reference["reference_hashes"].get(name)!=sha for name,sha in
                frozen["reference"]["reference_hashes"].items())):
        raise RuntimeError("Qualification and timing input fingerprints differ")
    eligible = {r["name"] for r in qualification["summary"] if r["numerical_pass"]}
    # Predeclared low-to-high ordering, never selected by concurrent timings.
    ordered = [s for s in protocol["variants"] if s["name"] in eligible]
    choices = [next((s for s in ordered if s["method"]==m), None) for m in ("rfm","legendre","fd4")]
    if any(s is None for s in choices):
        raise RuntimeError("Need at least one passing configuration for each method")
    rng = case.np.random.default_rng(protocol["order_seed"])
    jobs = [("timing", block, choices[i], seed if choices[i]["method"]=="rfm" else protocol["deterministic_seed"])
            for block,seed in enumerate(protocol["seeds"],1) for i in rng.permutation(len(choices))]
    cpu = pin()
    path = OUT / "isolated.json"
    result = journal(path, "timing", protocol, jobs, reference, cpu=cpu)
    known = {(r["sequence"],r["name"]) for r in result["rows"]}
    init_worker(protocol)
    case.np.linalg.svd(case.np.eye(10))
    for j in jobs:
        if (j[1],j[2]["name"]) in known:
            continue
        row=run(j)
        result["rows"].append(row)
        save(path,result)
        print(row["name"],row["seed"],row.get("seconds"),row.get("errors",row.get("failure")),flush=True)
    if source_hashes()!=result["frozen"]["source_hashes"]:
        raise RuntimeError("Frozen sources changed during isolated timing")
    result.update(status="complete", summary=summarize(result["rows"], "timing", protocol))
    save(path,result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference","audit","qualify","timing","smoke"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--new-seeds-only", action="store_true",
                        help="Qualify only the two additional RFM seeds at medium/high resolution.")
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    if args.phase=="reference":
        reference_step(protocol)
    elif args.phase=="audit":
        result,_ = reference_audit(protocol, allow_missing=True)
        save(OUT / "reference_audit.json", result)
        print(json.dumps({k:v for k,v in result.items() if k!="edges"},indent=2))
    elif args.phase=="qualify":
        if not 1<=args.workers<=16:
            parser.error("Choose 1 to 16 single-thread workers")
        qualify(protocol,args.workers,args.new_seeds_only)
    elif args.phase=="timing":
        timing(protocol)
    else:
        for spec in (protocol["variants"][1],protocol["variants"][4],protocol["variants"][7]):
            cls,settings = config(spec,protocol,protocol["seeds"][0])
            sol=case.solve(cls,steps=spec["rate"],horizon=1,snapshots=(0.5,1.0),**settings)
            assert case.np.isfinite(sol.states).all()
            print(spec["name"], sol.states.shape, sol.row["seconds"])


if __name__=="__main__":
    main()
