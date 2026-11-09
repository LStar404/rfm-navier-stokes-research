"""Score the archived mixed-rate CUDA pilots against CPU and physical references."""

import argparse
from pathlib import Path

import cpu_formal as base
import early_reference_chain as early
import early_reference_continuation as continuation
from cpu_precontracted_native import NativePrecontractedGalerkin
from cpu_sbdf2_switch import EARLY, LATE, SNAPSHOTS
from fd_cuda_pow2_replay import write_json
from galerkin_cuda_variable_audit import METHODS, OUT, PAIR_GATE, relative, source_hashes
from galerkin_cuda_variable import RecurrenceGalerkin, case, np
from verify_v7_operator_round import REPEAT, dual_truth


HERE = Path(__file__).resolve().parent
CPU = HERE/"cpu_sbdf2_switch_pilot_output"


def subset(solution, times):
    index = np.flatnonzero(np.isin(solution.times, times))
    if not np.array_equal(solution.times[index], times):
        raise ValueError("Requested absolute times are missing")
    return case.Solution(solution.model, solution.states[index], solution.times[index], solution.probe_times,
                         solution.transverse, solution.vorticity, solution.row)


def checked_solution(path, row, cls):
    if base.digest(path) != row["sha256"]:
        raise ValueError(f"Archive digest differs: {path.name}")
    result = case.load_solution(path, cls)
    if not all(np.isfinite(getattr(result, key)).all() for key in ("states", "transverse", "vorticity")):
        raise ValueError(f"Archive is not finite: {path.name}")
    return result


def early_references(grids):
    protocol = base.read(continuation.FROZEN)["protocol"]
    runs = base.read(continuation.RUNS)
    audit = base.read(continuation.REFERENCE)
    if (runs["status"] != "complete" or not runs["reference_pass"] or not audit["passes"] or
            base.digest(continuation.FROZEN) != runs["protocol_sha256"] or
            base.digest(continuation.REFERENCE) != runs["reference_audit_sha256"]):
        raise ValueError("Early/bridge reference audit is incomplete")
    fields = {}
    for row in runs["arms"]:
        name = row["spec"]["name"]
        if name in ("angular_fine", "time_fine"):
            solution = checked_solution(continuation.OUT/row["artifact"], row, early.late.MODEL)
            if not np.array_equal(solution.times, protocol["checkpoints"]):
                raise ValueError("Early physical reference checkpoints changed")
            fields[name] = case.reference_fields(solution, grids)
    times = protocol["checkpoints"]
    truth = fields["time_fine"]
    scale = early.norm_entries(truth, truth, grids, times)
    floors = {(g, k): max(np.finfo(float).tiny, protocol["near_zero_relative_floor"]*max(
        row["metrics"][k] for row in scale if row["grid"] == g)) for g in grids for k in early.METRICS}
    return times, fields, floors, runs["reference_audit_sha256"]


def compare_gpu(gpu, cpu, grids):
    if not np.array_equal(gpu.times, cpu.times) or not np.array_equal(gpu.probe_times, cpu.probe_times):
        raise ValueError("Mixed-rate CPU and GPU observations use different schedules")
    scores = case.score(gpu, case.reference_fields(cpu, grids), grids)
    worst = case.maxima(scores)
    probes = {key: relative(getattr(gpu, key), getattr(cpu, key)) for key in ("transverse", "vorticity")}
    values = [row[k] for row in scores for k in worst]+list(probes.values())
    return dict(passed=all(np.isfinite(v) and v <= PAIR_GATE for v in values), scores=scores, worst=worst, probes=probes)


def run(physical_only=False):
    march_path = OUT/"march.json"
    march = base.read(march_path)
    if march["status"] != "complete" or march["source_hashes"] != source_hashes():
        raise ValueError("The complete source-matched CUDA pilot is required")
    if march["fixed_sha256"] != base.digest(OUT/"fixed.json"):
        raise ValueError("CUDA fixed-state audit changed")
    grids = base.grid_set()
    early_times, early_truth, floors, early_audit = early_references(grids)
    late_truth = dual_truth(base.read(REPEAT/"protocol.json"), grids)
    physical_times = set(early_times) | set(LATE)
    missing = [t for t in SNAPSHOTS if t not in physical_times]
    if missing != list(SNAPSHOTS[6:8]):
        raise ValueError("Only the two first post-bridge steps lack FD reference times")
    rows = []
    for entry in march["rows"]:
        name, seed = entry["name"], entry["seed"]
        if METHODS[name] != seed:
            raise ValueError("Unexpected pilot method or seed")
        gpu = checked_solution(OUT/entry["artifact"], entry, RecurrenceGalerkin)
        if not np.array_equal(gpu.times, SNAPSHOTS):
            raise ValueError("CUDA mixed pilot checkpoints changed")
        comparisons = {}
        if not physical_only:
            for arm, cls in (("cpu_recurrence", RecurrenceGalerkin), ("cpu_native_contract", NativePrecontractedGalerkin)):
                path = CPU/f"{name}_{seed}_{arm}.npz"
                info = base.read(path.with_suffix(".json"))
                cpu = checked_solution(path, info, cls)
                comparisons[arm] = dict(cpu_sha256=info["sha256"], **compare_gpu(gpu, cpu, grids))
        control_path = HERE/"r128_operator_qualification_output"/f"{name}_{seed}_cuda_graph.npz"
        control = checked_solution(control_path, base.read(control_path.with_suffix(".json")), RecurrenceGalerkin)
        prefix_states = np.array_equal(subset(gpu, EARLY).states, subset(control, EARLY).states)
        mask = gpu.probe_times <= 8
        prefix_probes = all(np.array_equal(getattr(gpu, key)[mask], getattr(control, key)[mask])
                            for key in ("transverse", "vorticity"))
        early_fields = case.reference_fields(subset(gpu, early_times), grids)
        physical = {}
        for ref, truth in early_truth.items():
            scores = early.norm_entries(early_fields, truth, grids, early_times, floors)
            worst, undefined = early.summarize(scores)
            physical[ref] = dict(early_scores=scores, early_worst=worst, early_undefined=undefined)
        for ref, truth in late_truth.items():
            scores = case.score(subset(gpu, LATE), truth, grids)
            physical[ref].update(late_scores=scores, late_worst=case.maxima(scores))
        physical_pass = all(row["early_undefined"] == 0 and max(row["early_worst"].values()) <= .005 and
                            max(row["late_worst"].values()) <= .005 for row in physical.values())
        rows.append(dict(name=name, seed=seed, cuda_sha256=entry["sha256"], paired=comparisons,
                         startup_cuda128_bitwise_states=prefix_states, startup_cuda128_bitwise_probes=prefix_probes,
                         physical=physical, physical_pass=physical_pass,
                         alternating_wake=base.wake(gpu)["sustained_alternating_probe_signs"]))
    if len(rows) != 2:
        raise ValueError("Need the two registered pilot trajectories")
    passed = all(r["physical_pass"] and r["alternating_wake"] and r["startup_cuda128_bitwise_states"] and
                 r["startup_cuda128_bitwise_probes"] and all(p["passed"] for p in r["paired"].values()) for r in rows)
    return dict(status="complete", passed=passed, paired_checked=not physical_only,
                rows=rows, early_physical_times=early_times, late_physical_times=LATE,
                no_physical_reference_times=missing, implementation_gate=PAIR_GATE, physical_gate=.005,
                early_reference_audit_sha256=early_audit, cuda_march_sha256=base.digest(march_path),
                scope="Two mixed-rate pilots. No isolated speed result or five-seed generalization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT/"comparison.json")
    parser.add_argument("--physical-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.physical_only)
    write_json(args.output, result)
    print("PAIR", result["passed"], [r["physical_pass"] for r in result["rows"]], flush=True)
