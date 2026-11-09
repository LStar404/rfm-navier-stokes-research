"""Qualify one FD184 CUDA band replay against the frozen CPU184 trajectory."""

import argparse
import math
import os
from pathlib import Path

import fd_early_multiscale as original
from fd_cuda_native import solve_fd_native


base, case, np = original.base, original.case, original.np
HERE = Path(__file__).resolve().parent
OUT = HERE/"fd184_band_followup"
PROTOCOL = OUT/"protocol.json"
RESULT = OUT/"qualification.json"
RAW = OUT/"fd184_k80_17_cuda_graph_band.npz"
RAW_RECORD = RAW.with_suffix(".json")


def old_controls():
    frozen, runs, report = [base.read(path) for path in (original.FROZEN, original.RUNS, original.SUMMARY)]
    protocol_sha = base.digest(original.FROZEN)
    if (frozen["status"] != "frozen" or runs["status"] != "complete" or report["status"] != "complete" or
            runs["protocol_sha256"] != protocol_sha or report["protocol_sha256"] != protocol_sha or
            report["runs_sha256"] != base.digest(original.RUNS)):
        raise RuntimeError("The completed original FD qualification is required")
    rows = {row["job"]["arm"]: row for row in report["rows"] if row["job"]["name"] == "fd184_k80"}
    cpu, dense = rows["cpu_buffered"], rows["cuda_graph"]
    if (not cpu["qualified"] or not dense["physical_pass"] or dense["implementation_pass"] or dense["qualified"] or
            dense["metadata"]["cuda_linear_solve"] != "dense" or
            not dense["wake"]["sustained_alternating_probe_signs"]):
        raise RuntimeError("The retained FD184 dense failure differs from the original result")
    for row in (cpu, dense):
        path = original.OUT/row["artifact"]
        stored = base.read(path.with_suffix(".json"))
        if (row["job"] != stored["job"] or row["metadata"] != stored["metadata"] or
                row["sha256"] != stored["sha256"] or base.digest(path) != row["sha256"]):
            raise RuntimeError("FD184 raw archive or individual record changed")
    return frozen, cpu, dense


def freeze():
    original_frozen, cpu, dense = old_controls()
    p = original_frozen["template"]
    job = dict(dense["job"])
    if (job["n"], job["k"], job["rate"], job["seed"], p["horizon"], p["probe_spacing"], p["graph_steps"],
            p["physical_gate"], p["implementation_gate"], len(original_frozen["checkpoints"])) != (
            184, 80, 128, 17, 72., .125, 24, .005, 1e-5, 19):
        raise ValueError("FD184 follow-up must keep the registered physical and implementation gates")
    sources = dict(original_frozen["source_hashes"])
    inputs = [Path(__file__), original.FROZEN, original.RUNS, original.SUMMARY]
    for row in (cpu, dense):
        path = original.OUT/row["artifact"]
        inputs.extend((path, path.with_suffix(".json")))
    for path in inputs:
        item = original.reference_record(path)
        sources[item["path"]] = item["sha256"]
    original.check_hashes(sources)
    for refs in original_frozen["references"].values():
        original.check_hashes({item["path"]: item["sha256"] for item in refs.values()})
    frozen = dict(status="frozen", job=job, cuda_linear_solve="band", checkpoints=original_frozen["checkpoints"],
                  graph_steps=p["graph_steps"], physical_gate=p["physical_gate"],
                  implementation_gate=p["implementation_gate"],
                  source_hashes=sources, references=original_frozen["references"],
                  control=original.reference_record(original.OUT/cpu["artifact"]),
                  retained_dense=original.reference_record(original.OUT/dense["artifact"]),
                  scope="One accuracy-only CUDA replay. The FD4 equation, grid, time step and original CPU control stay fixed. "
                        "The unscaled cached band factors replace the explicit dense inverse. Solver times are diagnostic.")
    if PROTOCOL.exists():
        if base.read(PROTOCOL) != frozen:
            raise RuntimeError("FD184 band protocol changed")
    else:
        OUT.mkdir(exist_ok=True)
        base.save(PROTOCOL, frozen)
    return frozen


def score(candidate, control, frozen, original_frozen):
    grids = base.grid_set()
    p = original_frozen["template"]
    if not np.array_equal(candidate.times, control.times) or not np.array_equal(candidate.probe_times, control.probe_times):
        raise ValueError("FD184 comparison observations differ")
    fields = case.score(candidate, case.reference_fields(control, grids), grids)
    keys = {(g, t) for g in grids for t in frozen["checkpoints"]}
    if len(fields) != 57 or {(row["grid"], row["time"]) for row in fields} != keys:
        raise ValueError("FD184 implementation grid is incomplete")
    maxima = case.maxima(fields)
    probes = {}
    for name in ("transverse", "vorticity"):
        a, b = getattr(candidate, name), getattr(control, name)
        norm = np.linalg.norm(b)
        if norm <= 1e-12:
            raise ValueError("Reference probe is too small for a relative FD184 check")
        probes[name] = float(np.linalg.norm(a-b)/norm)
    implementation = all(math.isfinite(x) and 0 <= x <= frozen["implementation_gate"]
                         for x in (*maxima.values(), *probes.values()))
    truth, floors = original.reference_fields(original_frozen, grids)
    early = case.reference_fields(original.subset(candidate, p["early_checkpoints"]), grids)
    late = original.subset(candidate, p["late_checkpoints"])
    physical = {}
    for name in p["references"]:
        a = original.early.norm_entries(early, truth["early"][name], grids, p["early_checkpoints"], floors)
        b = case.score(late, truth["late"][name], grids)
        worst, undefined = original.early.summarize(a)
        if len(a) != 36 or len(b) != 21:
            raise ValueError("Incomplete FD184 dual-reference field scores")
        physical[name] = dict(early_scores=a, early_worst=worst, early_undefined=undefined,
                              late_scores=b, late_worst=case.maxima(b))
    physical_pass = all(item["early_undefined"] == 0 and max(item["early_worst"].values()) <= frozen["physical_gate"]
                        and max(item["late_worst"].values()) <= frozen["physical_gate"] for item in physical.values())
    wake = base.wake(candidate)
    return dict(implementation_scores=fields, implementation_worst=maxima, implementation_probe=probes,
                implementation_pass=implementation, physical=physical, physical_pass=physical_pass, wake=wake,
                qualified=bool(implementation and physical_pass and wake["sustained_alternating_probe_signs"]))


def execute(frozen):
    if any(path.exists() for path in (RAW, RAW_RECORD, RESULT)):
        raise FileExistsError("The single authorized FD184 band replay already started")
    original_frozen, cpu_row, dense = old_controls()
    p = original_frozen["template"]
    job = frozen["job"]
    config = original.config(job, p)
    print("START FD184 band", flush=True)
    result = solve_fd_native(steps=int(job["rate"]*p["horizon"]), horizon=p["horizon"],
                             snapshots=frozen["checkpoints"], probe_spacing=p["probe_spacing"],
                             graph_steps=frozen["graph_steps"], cuda_linear_solve=frozen["cuda_linear_solve"], **config)
    original.validate(result, job, original_frozen)
    temporary = RAW.with_suffix(".tmp.npz")
    case.store_solution(temporary, result, config)
    temporary.replace(RAW)
    raw = dict(status="complete", job=job, artifact=RAW.name, sha256=base.digest(RAW), metadata=result.row)
    base.save(RAW_RECORD, raw)
    control = case.load_solution(base.WORK/frozen["control"]["path"], base.FullFD)
    original.validate(control, cpu_row["job"], original_frozen)
    scores = score(result, control, frozen, original_frozen)
    if freeze() != frozen:
        raise RuntimeError("FD184 band protocol changed during accuracy work")
    base.save(RESULT, dict(raw, protocol_sha256=base.digest(PROTOCOL), raw_record_sha256=base.digest(RAW_RECORD),
                           control_sha256=cpu_row["sha256"], retained_dense=dict(sha256=dense["sha256"],
                           physical_pass=dense["physical_pass"], implementation_pass=dense["implementation_pass"],
                           implementation_worst=dense["implementation_worst"], implementation_probe=dense["implementation_probe"]),
                           **scores, isolated_speed_claim_supported=False, scope=frozen["scope"]))
    print("FD184 BAND QUALIFIED", scores["qualified"], scores["implementation_worst"], scores["implementation_probe"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = cli.parse_args()
    if args.execute and any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        cli.error("Use one BLAS thread for the follow-up")
    frozen = freeze()
    print("FROZEN FD184 BAND", base.digest(PROTOCOL), flush=True)
    if args.execute:
        execute(frozen)
