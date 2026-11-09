"""Qualify three independent constant-rate FD4 controls at nineteen physical times."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path

import cpu_formal as base
import early_reference_chain as early
import fd_pow2_reference_chain as late
from fd_cpu_buffered import BufferedFD
from fd_cuda_native import solve_fd_native


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE/"protocol_fd_early_multiscale.json"
OUT = HERE/"fd_early_multiscale_output"
FROZEN = OUT/"protocol.json"
RUNS = OUT/"runs.json"
SUMMARY = OUT/"qualification.json"
EARLY = HERE/"early_reference_continuation_output"
case, np = base.case, base.case.np
METRICS = ("velocity", "vorticity", "near_velocity", "near_vorticity")


def reference_record(path):
    return dict(path=path.relative_to(base.WORK).as_posix(), sha256=base.digest(path))


def check_hashes(mapping):
    for name, expected in mapping.items():
        if base.digest(base.WORK/name) != expected:
            raise RuntimeError(f"Frozen digest differs: {name}")


def freeze():
    p = base.read(TEMPLATE)
    old = base.read(HERE/"hardware_pow2_output"/"protocol.json")
    ep, er, ea = (base.read(EARLY/name) for name in ("protocol.json", "runs.json", "reference_audit.json"))
    lr = base.read(late.JOURNAL)
    la_path = late.OUT/lr["reference_audit"]
    la = base.read(la_path)
    budget = p["physical_gate"]*p["reference_budget_fraction"]
    if (p["rate"], p["k"], p["horizon"], p["probe_spacing"], p["cpu_workers"], p["seed"]) != (128, 80, 72., .125, 3, 17):
        raise ValueError("Unexpected fixed FD march configuration")
    if ([(r["name"], r["n"], r["scale"]) for r in p["cases"]] != [(f"fd{n}_k80", n, s) for n, s in
            ((152, "low"), (176, "mid"), (184, "high"))] or p["arms"] != ["cpu_buffered", "cuda_graph"] or
            p["references"] != ["angular_fine", "time_fine"] or p["physical_gate"] != .005 or
            p["implementation_gate"] != 1e-5 or p["graph_steps"] != 24 or p["cuda_linear_solve"] != "dense" or
            p["grid"] != "outer" or p["stretch"] != 1.):
        raise ValueError("Unexpected FD grid, arms or gate")
    if (ep["protocol"]["checkpoints"] != p["early_checkpoints"] or lr["frozen"]["protocol"]["checkpoints"] != p["late_checkpoints"]
            or len(p["early_checkpoints"]+p["late_checkpoints"]) != 19 or er["status"] != "complete" or
            not er["bitwise_t8_pass"] or not er["reference_pass"] or not ea["passes"] or
            max(ea["worst"].values()) > budget or lr["status"] != "complete" or la["composed_difference"] > budget):
        raise ValueError("Both reference phases need independently passing convergence screens")
    late_files = old["reference"]["files"]
    if (base.digest(EARLY/"protocol.json") != er["protocol_sha256"] or ea["protocol_sha256"] != er["protocol_sha256"] or
            base.digest(EARLY/"reference_audit.json") != er["reference_audit_sha256"] or
            any(base.digest(path) != late_files[path.relative_to(base.WORK).as_posix()] for path in (late.JOURNAL, la_path))):
        raise ValueError("Reference protocol or independent-axis audit digest differs")
    early_arms = {row["spec"]["name"]: row for row in er["arms"]}
    late_arms = {row["spec"]["name"]: row for row in lr["rows"]}
    refs = {"early": {}, "late": {}}
    for phase, specs, records, directory in (("early", ep["protocol"]["reference_path"], early_arms, EARLY),
                                              ("late", lr["frozen"]["protocol"]["reference_path"], late_arms, late.OUT)):
        for spec in specs:
            name = spec["name"]
            if phase == "early":
                row = records[name]
                path, expected = directory/row["artifact"], row["sha256"]
                if not row["bitwise_t8_pass"] or ea["arm_hashes"][row["artifact"]] != expected:
                    raise ValueError("The T16 reference failed its immutable continuation check")
            else:
                path = base.WORK/spec["source"]
                expected = spec["sha256"] if "sha256" in spec else records[name]["sha256"]
            record = reference_record(path)
            if record["sha256"] != expected:
                raise ValueError(f"Reference trajectory hash differs: {name}")
            refs[phase][name] = record
    sources = old["source_hashes"] | ep["source_hashes"] | late.fingerprint()
    inputs = (Path(__file__), TEMPLATE, EARLY/"protocol.json", EARLY/"runs.json", EARLY/"reference_audit.json",
              late.JOURNAL, la_path, HERE/"hardware_pow2_output"/"protocol.json")
    for path in inputs:
        record = reference_record(path)
        sources[record["path"]] = record["sha256"]
    check_hashes(sources)
    jobs = [dict(name=s["name"], n=s["n"], k=p["k"], rate=p["rate"], seed=p["seed"], arm=arm)
            for s in p["cases"] for arm in p["arms"]]
    fixed = dict(status="frozen", template=p, checkpoints=p["early_checkpoints"]+p["late_checkpoints"],
                 jobs=jobs, references=refs, early_reference_protocol=ep["protocol"], source_hashes=sources)
    if FROZEN.exists():
        if base.read(FROZEN) != fixed:
            raise RuntimeError("Frozen FD qualification inputs differ")
    else:
        OUT.mkdir(exist_ok=True)
        base.save(FROZEN, fixed)
    return fixed


def artifact(job):
    return OUT/f"{job['name']}_{job['seed']}_{job['arm']}.npz"


def config(job, p):
    return dict(n=job["n"], k=job["k"], stretch=p["stretch"], grid=p["grid"], angular="fft", linear="native")


def validate(sol, job, frozen):
    p = frozen["template"]
    probe_times = np.arange(0., p["horizon"]+.01*p["probe_spacing"], p["probe_spacing"])
    if (not np.array_equal(sol.times, frozen["checkpoints"]) or not np.array_equal(sol.probe_times, probe_times) or
            sol.row["steps"] != int(job["rate"]*p["horizon"]) or sol.row["dt"] != 1/job["rate"] or
            sol.states.shape != (19, job["k"]+1, job["n"]-1) or
            any(getattr(sol, key).shape != (len(probe_times), len(case.PROBES)) for key in ("transverse", "vorticity")) or
            not all(np.isfinite(getattr(sol, key)).all() for key in ("states", "transverse", "vorticity"))):
        raise ArithmeticError("FD trajectory fields, probes or time grid differ from the freeze")


def run(job, frozen):
    path = artifact(job)
    record = path.with_suffix(".json")
    if record.exists():
        row = base.read(record)
        if row["job"] != job or ("artifact" in row and base.digest(path) != row["sha256"]):
            raise RuntimeError("Existing FD qualification trajectory differs")
        return row
    if path.exists():
        raise FileExistsError(path)
    p = frozen["template"]
    settings = config(job, p)
    args = dict(steps=int(job["rate"]*p["horizon"]), horizon=p["horizon"],
                snapshots=frozen["checkpoints"], probe_spacing=p["probe_spacing"], **settings)
    row = dict(job=job)
    print("START", job, flush=True)
    try:
        if job["arm"] == "cuda_graph":
            solution = solve_fd_native(graph_steps=p["graph_steps"], cuda_linear_solve=p["cuda_linear_solve"], **args)
        else:
            solution = case.solve(BufferedFD, **args)
        validate(solution, job, frozen)
        temporary = path.with_suffix(".tmp.npz")
        case.store_solution(temporary, solution, settings)
        temporary.replace(path)
        row.update(status="complete", artifact=path.name, sha256=base.digest(path), metadata=solution.row)
    except (ArithmeticError, ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as exc:
        row.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
    base.save(record, row)
    print("DONE", job, row["status"], flush=True)
    return row


def subset(solution, times):
    index = np.flatnonzero(np.isin(solution.times, times))
    if not np.array_equal(solution.times[index], times):
        raise ValueError("Physical score snapshots differ")
    return case.Solution(solution.model, solution.states[index], solution.times[index], solution.probe_times,
                         solution.transverse, solution.vorticity, solution.row)


def reference_fields(frozen, grids):
    p = frozen["template"]
    fields = {}
    for phase in ("early", "late"):
        truth = {}
        for name in p["references"]:
            item = frozen["references"][phase][name]
            check_hashes({item["path"]: item["sha256"]})
            sol = case.load_solution(base.WORK/item["path"], late.MODEL)
            if not np.array_equal(sol.times, p[phase+"_checkpoints"]) or sol.row["row_scale"] != sol.model.scale_metadata():
                raise ValueError("Physical reference reconstruction differs")
            truth[name] = case.reference_fields(sol, grids)
        fields[phase] = truth
    fine = fields["early"]["time_fine"]
    times = p["early_checkpoints"]
    norms = early.norm_entries(fine, fine, grids, times)
    floor_ratio = frozen["early_reference_protocol"]["near_zero_relative_floor"]
    floors = {(g, key): max(np.finfo(float).tiny, floor_ratio*max(row["metrics"][key] for row in norms if row["grid"] == g))
              for g in grids for key in METRICS}
    return fields, floors


def score_all(frozen, rows):
    p = frozen["template"]
    grids = base.grid_set()
    truth, floors = reference_fields(frozen, grids)
    expected = {(g, t) for g in grids for t in frozen["checkpoints"]}
    keyed = {(r["job"]["name"], r["job"]["arm"]): r for r in rows}
    if len(keyed) != len(rows) or len(rows) != len(frozen["jobs"]):
        raise ValueError("Incomplete FD raw records")
    controls, scores = {}, []
    for job in frozen["jobs"]:
        name, arm = job["name"], job["arm"]
        row = keyed[name, arm]
        if row["status"] != "complete":
            scores.append(dict(row, implementation_pass=False, physical_pass=False, qualified=False))
            continue
        path = artifact(job)
        if base.digest(path) != row["sha256"] or row["artifact"] != path.name:
            raise ValueError("A scored FD trajectory does not match its raw record")
        sol = case.load_solution(path, base.FullFD)
        validate(sol, job, frozen)
        if arm == "cpu_buffered":
            controls[name] = sol, case.reference_fields(sol, grids)
        physical = {}
        early_fields = case.reference_fields(subset(sol, p["early_checkpoints"]), grids)
        late_sol = subset(sol, p["late_checkpoints"])
        for ref in p["references"]:
            early_scores = early.norm_entries(early_fields, truth["early"][ref], grids, p["early_checkpoints"], floors)
            early_worst, undefined = early.summarize(early_scores)
            late_scores = case.score(late_sol, truth["late"][ref], grids)
            if len(early_scores) != 36 or len(late_scores) != 21:
                raise ValueError("Missing early or late physical score rows")
            physical[ref] = dict(early_scores=early_scores, early_worst=early_worst, early_undefined=undefined,
                                 late_scores=late_scores, late_worst=case.maxima(late_scores))
        wake = base.wake(sol)
        physical_pass = all(item["early_undefined"] == 0 and max(item["early_worst"].values()) <= p["physical_gate"]
                            and max(item["late_worst"].values()) <= p["physical_gate"] for item in physical.values())
        implementation = dict(implementation_pass=False, implementation_reason="missing CPU control")
        if name in controls:
            control, fields = controls[name]
            values = case.score(sol, fields, grids)
            if len(values) != len(expected) or {(r["grid"], r["time"]) for r in values} != expected:
                raise ValueError("Incomplete implementation score rows")
            maxima = case.maxima(values)
            probe = {key: float(np.linalg.norm(getattr(sol, key)-getattr(control, key))/np.linalg.norm(getattr(control, key)))
                     for key in ("transverse", "vorticity")}
            implementation_pass = all(math.isfinite(value) and 0 <= value <= p["implementation_gate"]
                                      for value in (*maxima.values(), *probe.values()))
            implementation = dict(implementation_scores=values, implementation_worst=maxima,
                                  implementation_probe=probe, implementation_pass=implementation_pass)
        scores.append(dict(row, physical=physical, wake=wake, physical_pass=physical_pass, **implementation,
                           qualified=bool(physical_pass and wake["sustained_alternating_probe_signs"] and
                                          implementation["implementation_pass"])))
    return scores


def execute(frozen):
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    protocol_sha = base.digest(FROZEN)
    if RUNS.exists() and base.read(RUNS)["protocol_sha256"] != protocol_sha:
        raise RuntimeError("FD run journal differs from its freeze")
    journal = dict(status="running", protocol_sha256=protocol_sha, rows=[])
    base.save(RUNS, journal)
    jobs = frozen["jobs"]
    with ProcessPoolExecutor(max_workers=frozen["template"]["cpu_workers"]) as pool:
        cpu = [pool.submit(run, job, frozen) for job in jobs if job["arm"] == "cpu_buffered"]
        for job in (j for j in jobs if j["arm"] == "cuda_graph"):
            journal["rows"].append(run(job, frozen))
            base.save(RUNS, journal)
        for task in cpu:
            journal["rows"].append(task.result())
            base.save(RUNS, journal)
    if freeze() != frozen or base.digest(FROZEN) != protocol_sha:
        raise RuntimeError("FD qualification freeze changed during the march")
    rows = score_all(frozen, journal["rows"])
    scales = []
    for spec in frozen["template"]["cases"]:
        both = [r for r in rows if r["job"]["name"] == spec["name"]]
        scales.append(dict(spec, qualified=len(both) == 2 and all(r["qualified"] for r in both),
                           arms={r["job"]["arm"]: r["qualified"] for r in both}))
    journal["status"] = "complete"
    base.save(RUNS, journal)
    base.save(SUMMARY, dict(status="complete", protocol_sha256=protocol_sha, runs_sha256=base.digest(RUNS),
                            rows=rows, scales=scales, all_passed=all(s["qualified"] for s in scales),
                            isolated_speed_claim_supported=False, scope=frozen["template"]["scope"]))
    print("SCALES", [(r["name"], r["qualified"]) for r in scales], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = cli.parse_args()
    if args.execute and any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        cli.error("Use one BLAS thread per worker")
    protocol = freeze()
    print("FROZEN", base.digest(FROZEN), len(protocol["jobs"]), flush=True)
    if args.execute:
        execute(protocol)
