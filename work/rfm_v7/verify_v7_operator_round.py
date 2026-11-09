"""Verify the operator-round release with archived fields and an optional T=8 replay."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median

import cpu_formal as base
from cpu_precontracted_native import NativePrecontractedGalerkin, RecurrenceGalerkin
from fd_cuda_pow2_replay import PROTOCOL as POW2_PROTOCOL, compare as compare_pow2, reference as pow2_reference
from fd_cuda_rowcomp_replay import PROTOCOL as ROWCOMP_PROTOCOL, verify as verify_rowcomp
from fd_pow2_long import Pow2EquilibratedBufferedFD
import summarize_cpu_native_joint as timing_summary


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE/"cpu_native_joint_output"
REPEAT = HERE/"cpu_native_joint_repeat_output"
ROOT = HERE.parents[1]
case, np = base.case, base.case.np
METRICS = ("velocity", "vorticity", "near_velocity", "near_vorticity")
JOB_KEYS = ("block", "name", "arm", "rate", "seed")
REL_TOL = 1e-8
ABS_TOL = 1e-12


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def archived(root, name):
    path = (root/name).resolve()
    check(path.is_relative_to(root.resolve()), f"Archive path escapes its directory: {name}")
    return path


def numeric_match(actual, expected, where):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Different record keys: {where}")
        for key, value in expected.items():
            numeric_match(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), f"Different row count: {where}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            numeric_match(a, b, f"{where}[{index}]")
    elif isinstance(expected, float):
        check(isinstance(actual, (float, int)) and not isinstance(actual, bool)
              and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL),
              f"Numerical difference at {where}: {actual!r} versus {expected!r}")
    else:
        check(type(actual) is type(expected) and actual == expected, f"Different stored value at {where}")


def verify_digest(path, expected):
    check(base.digest(path) == expected, f"Digest mismatch: {path.relative_to(ROOT)}")


def scores_by_key(rows, expected, where):
    indexed = {(r["grid"], r["time"]): r for r in rows}
    check(len(indexed) == len(rows) and indexed.keys() == expected, f"Incomplete field grid or time keys: {where}")
    check(all(math.isfinite(r[key]) and r[key] >= 0 for r in rows for key in METRICS),
          f"Nonfinite or negative field score: {where}")
    return indexed


def compare_scores(actual, expected, keys, where):
    a = scores_by_key(actual, keys, where)
    b = scores_by_key(expected, keys, where)
    for key in sorted(keys):
        for metric in METRICS:
            numeric_match(a[key][metric], b[key][metric], f"{where} {key} {metric}")


def worst(rows):
    return {key: max(r[key] for r in rows) for key in METRICS}


def frozen_inputs(frozen):
    maps = [frozen["files"], frozen["trajectories"]]
    inputs = frozen["inputs"]
    maps.extend(inputs[key] for key in ("source_hashes", "qualification_files", "reference_hashes"))
    hashes = {}
    for entries in maps:
        for relative, digest in entries.items():
            check(hashes.get(relative, digest) == digest, f"Conflicting frozen digests: {relative}")
            hashes[relative] = digest
    for job in inputs["jobs"]:
        for control in job["controls"]:
            path = archived(base.WORK, control["source"])
            verify_digest(path, control["sha256"])
            record = base.read(path)
            verify_digest(archived(HERE, record["artifact"]), record["sha256"])
    for name, digest in hashes.items():
        verify_digest(archived(base.WORK, name), digest)
    return len(hashes)


def check_schedule(frozen, rows):
    template = frozen["template"]
    spec = frozen["inputs"]["solve_protocol"]
    arms = [(name, arm) for name in template["galerkin_names"] for arm in template["galerkin_arms"]]
    arms.extend((template["fd_name"], arm) for arm in template["fd_arms"])
    check(len(arms) == 10 and len(set(arms)) == 10 and len(spec["seeds"]) == template["blocks"] == 5,
          "Expected five randomized ten-arm blocks")
    families = {row["name"]: row["method"] for row in spec["variants"]}
    schedule = []
    random = np.random.default_rng(template["order_seed"])
    for block, seed in enumerate(spec["seeds"], 1):
        for i in random.permutation(len(arms)):
            name, arm = arms[i]
            if name == template["fd_name"]:
                rate = template["fd_rate"]
            elif arm == "cuda_graph_rate128":
                rate = template["galerkin_cuda_control_rate"]
            else:
                rate = template["galerkin_rate"]
            schedule.append(dict(block=block, name=name, arm=arm, rate=rate,
                                 seed=seed if families[name] == "rfm" else spec["deterministic_seed"]))
    check(schedule == frozen["schedule"], "Reconstructed randomized schedule differs")
    check([{k: row[k] for k in JOB_KEYS} for row in rows] == schedule,
          "Timed rows do not follow the frozen schedule")
    return arms


def check_archive(path, row, schedule, probe_times, config):
    verify_digest(path, row["sha256"])
    with np.load(path, allow_pickle=False) as archive:
        check(json.loads(str(archive["metadata"])) == row["metadata"], f"Archive metadata differs: {path.name}")
        check(json.loads(str(archive["config"])) == config, f"Archive configuration differs: {path.name}")
        check(np.array_equal(archive["times"], schedule) and np.array_equal(archive["probe_times"], probe_times),
              f"Observation schedule differs: {path.name}")
        states = archive["states"]
        check(states.ndim == 3 and states.shape[0] == len(schedule) and states.dtype == np.complex128
              and np.isfinite(states).all(), f"Invalid saved state: {path.name}")
        for key in ("transverse", "vorticity"):
            a = archive[key]
            check(a.shape == (len(probe_times), len(case.PROBES)) and np.isfinite(a).all(),
                  f"Invalid archived {key}: {path.name}")


def check_groups_and_pairs(timing, template, arms):
    buckets = defaultdict(dict)
    for row in timing["rows"]:
        key = row["name"], row["arm"]
        check(row["block"] not in buckets[key], f"Duplicate block for {key}")
        buckets[key][row["block"]] = row
    groups = {(g["name"], g["arm"]): g for g in timing["groups"]}
    check(len(groups) == len(timing["groups"]) == len(arms) and groups.keys() == set(arms),
          "Timing group inventory differs")
    blocks = list(range(1, template["blocks"]+1))
    for key in arms:
        check(set(buckets[key]) == set(blocks), f"Missing group blocks: {key}")
        rows = [buckets[key][b] for b in blocks]
        times = [row["seconds"] for row in rows]
        expected = dict(name=key[0], arm=key[1], count=5, successes=5, passes=True, seconds=times,
                        median=median(times), minimum=min(times), maximum=max(times),
                        worst={ref: {k: max(r["worst"][ref][k] for r in rows) for k in METRICS}
                               for ref in template["reference_names"]})
        check(groups[key] == expected, f"Reconstructed group statistics differ: {key}")
    rfm, leg = template["galerkin_names"]
    fd = template["fd_name"]
    comparisons = {((leg, arm), (rfm, arm)) for arm in template["galerkin_arms"]}
    comparisons.add(((fd, "cpu_buffered"), (fd, "cuda_graph")))
    for name in (rfm, leg):
        comparisons.update((((name, "cpu_recurrence"), (name, "cpu_native_contract")),
                            ((name, "cuda_graph_rate128"), (name, "cuda_graph")),
                            ((fd, "cpu_buffered"), (name, "cpu_native_contract")),
                            ((fd, "cuda_graph"), (name, "cuda_graph")),
                            ((name, "cpu_native_contract"), (name, "cuda_graph"))))
    pairs = {(tuple(p["numerator"]), tuple(p["denominator"])): p for p in timing["ratios"]}
    check(len(pairs) == len(timing["ratios"]) == 15 and pairs.keys() == comparisons,
          "Paired contrast inventory differs")
    for (numerator, denominator), recorded in pairs.items():
        values = [buckets[numerator][b]["seconds"]/buckets[denominator][b]["seconds"] for b in blocks]
        expected = dict(numerator=list(numerator), denominator=list(denominator),
                        pairs=[dict(block=b, ratio=v) for b, v in zip(blocks, values)],
                        aggregate=dict(median=median(values), minimum=min(values), maximum=max(values),
                                       denominator_wins=sum(v > 1 for v in values)))
        check(expected == recorded, f"Reconstructed paired statistics differ: {numerator}/{denominator}")


def dual_truth(frozen, grids):
    inputs = frozen["inputs"]
    folder = HERE/"cpu_formal_point1_pow2_chain_output"
    journal = base.read(folder/"runs.json")
    arms = {row["spec"]["name"]: row for row in journal["rows"]}
    truth = {}
    for name in frozen["template"]["reference_names"]:
        row = arms[name]
        spec = row["spec"]
        check(row["status"] == "complete" and row["sha256"] == inputs["reference_hashes"][spec["source"]],
              f"Reference cross-hash differs: {name}")
        sol = case.load_solution(archived(base.WORK, spec["source"]), Pow2EquilibratedBufferedFD)
        expected = dict(n=spec["n"], k=spec["k"], steps=spec["rate"]*inputs["solve_protocol"]["horizon"],
                        row_scale=sol.model.scale_metadata())
        check(all(sol.row[k] == value for k, value in expected.items()) and
              np.array_equal(sol.times, inputs["solve_protocol"]["checkpoints"]) and
              base.wake(sol)["sustained_alternating_probe_signs"], f"Reference reconstruction differs: {name}")
        truth[name] = case.reference_fields(sol, grids)
    return truth


def verify_timing(directory, *, rescore):
    path = directory/"protocol.json"
    frozen = base.read(path)
    timing = base.read(directory/"timing.json")
    template, solve_protocol = frozen["template"], frozen["inputs"]["solve_protocol"]
    check(frozen["status"] == "frozen" and timing["status"] == "complete" and len(timing["rows"]) == 50,
          "Need the completed 50-run timing archive")
    check(template["reference_names"] == ["angular_fine", "time_fine"] and
          (template["galerkin_rate"], template["galerkin_cuda_control_rate"], template["fd_rate"]) == (112, 128, 128),
          "Unexpected reference or selected/control step rate")
    verify_digest(path, timing["protocol_sha256"])
    hashes = frozen_inputs(frozen) if rescore else 0
    arms = check_schedule(frozen, timing["rows"])
    grids = base.grid_set()
    checkpoints = solve_protocol["checkpoints"]
    probes = np.arange(0., solve_protocol["horizon"]+.001, .125)
    keys = {(grid, t) for grid in grids for t in checkpoints}
    check(len(keys) == 21, "Expected 21 grid/time scores per reference")
    specs = {s["name"]: s for s in solve_protocol["variants"]}
    artifacts = set()
    for row in timing["rows"]:
        tag = f"b{row['block']}_{row['name']}_{row['arm']}_{row['seed']}"
        check(row["artifact"] == f"{tag}.npz" and tag not in artifacts, f"Unexpected artifact: {tag}")
        artifacts.add(tag)
        check(row == base.read(directory/f"{tag}.json"), f"Individual timing record differs: {tag}")
        _, config = base.config(specs[row["name"]], solve_protocol, row["seed"])
        check_archive(directory/row["artifact"], row, checkpoints, probes, config)
        check(math.isfinite(row["seconds"]) and row["seconds"] > 0 and
              row["metadata"]["steps"] == row["rate"]*solve_protocol["horizon"] and
              row["numerical_pass"] and row["implementation_pass"] and
              row["wake"]["sustained_alternating_probe_signs"], f"Unexpected timing failure: {tag}")
        check(row["scores"].keys() == set(template["reference_names"]), f"Reference inventory differs: {tag}")
        for name, scores in row["scores"].items():
            scores_by_key(scores, keys, tag)
            check(worst(scores) == row["worst"][name] and max(row["worst"][name].values()) <= template["gate"],
                  f"Recorded score maxima or gate differ: {tag}")
        check(all(math.isfinite(v) and 0 <= v <= template["implementation_tolerance"]
                  for values in (row["implementation_field"], row["implementation_probe"]) for v in values.values()),
              f"Implementation gate differs: {tag}")
    check_groups_and_pairs(timing, template, arms)
    result = dict(timing_rows=len(timing["rows"]), frozen_hashes=hashes, timing_groups=10, timing_pairs=15)
    if not rescore:
        summary = base.read(directory/"summary.json")
        check(summary["runs"] == len(timing["rows"]) and summary["paired"] == timing["ratios"] and
              not summary["isolated_speed_claim_supported"], "Original timing was not archived as diagnostic-only")
        verify_digest(directory/"timing.json", summary["timing_sha256"])
        return frozen, dict(result, publication_role="diagnostic_only", field_errors_recomputed=0,
                            observed_isolation_status=summary["isolation"]["status"])
    recorded_summary = base.read(directory/"summary.json")
    check(recorded_summary == timing_summary.summarize(directory), "Published detailed timing summary differs")
    truth = dual_truth(frozen, grids)
    selected = {(name, arm) for name in template["galerkin_names"] for arm in ("cpu_native_contract", "cuda_graph")}
    selected.update((template["fd_name"], arm) for arm in template["fd_arms"])
    rows = [r for r in timing["rows"] if r["block"] == 1 and (r["name"], r["arm"]) in selected]
    check(len(rows) == len(selected) == 6, "Expected six selected first-block method/backend trajectories")
    for row in rows:
        if row["name"] == template["fd_name"]:
            cls = base.FullFD
        elif row["arm"] == "cpu_native_contract":
            cls = NativePrecontractedGalerkin
        else:
            cls = RecurrenceGalerkin
        path = directory/row["artifact"]
        sol = case.load_solution(path, cls)
        check(sol.states.shape[1:] == sol.model.rhs_shape and base.wake(sol)["sustained_alternating_probe_signs"],
              f"Saved shape or shedding differs: {path.name}")
        for name, ref in truth.items():
            scores = case.score(sol, ref, grids)
            compare_scores(scores, row["scores"][name], keys, f"{path.name} {name}")
            check(max(worst(scores).values()) <= template["gate"], f"Reconstructed field gate differs: {path.name}")
    role = "sampled_window_eligible" if recorded_summary["isolated_speed_claim_supported"] else "diagnostic_only"
    return frozen, dict(result, publication_role=role, representative_trajectories=len(rows), references=len(truth),
                        field_errors_recomputed=len(rows)*len(truth)*len(keys)*len(METRICS))


def timestamp(value):
    instant = datetime.fromisoformat(value)
    check(instant.utcoffset() is not None, f"Expected a timezone in monitoring timestamp {value}")
    return instant


def monitor_activity(rows, allowed, worker, thresholds):
    seen = set()
    background = []
    unexpected = []
    for item in rows:
        pid, label, seconds = item["pid"], item["label"], item["cpu_seconds"]
        check(pid != worker and pid not in seen and math.isfinite(seconds) and
              label in ("known_server", "companion", "python") and ((pid in allowed) == (label != "python")),
              "Unexpected process identity or counter in fixed-window observations")
        seen.add(pid)
        (unexpected if label == "python" else background).append(seconds)
    return (any(v > thresholds["background_individual_cpu_seconds"] for v in background) or
            sum(background) > thresholds["background_total_cpu_seconds"] or
            any(v > thresholds["unexpected_python_cpu_seconds"] for v in unexpected))


def verify_isolation():
    original_protocol = ORIGINAL/"protocol.json"
    repeat_protocol = REPEAT/"protocol.json"
    supplement_path = REPEAT/"isolation_protocol.json"
    isolation_path = REPEAT/"isolation_repeat.json"
    execution_path = REPEAT/"execution.json"
    supplement, isolation, execution = map(base.read, (supplement_path, isolation_path, execution_path))
    original_digest = base.digest(original_protocol)
    supplement_digest = base.digest(supplement_path)
    verify_digest(repeat_protocol, original_digest)
    check(supplement["status"] == "frozen" and supplement["original_protocol_sha256"] == original_digest,
          "Isolation supplement is not bound to the unchanged original experiment protocol")
    required_sources = {"rfm_v7/cpu_native_joint.py", "rfm_v7/cpu_native_joint_repeat.py",
                        "rfm_v7/run_cpu_native_joint_repeat.ps1"}
    check(required_sources <= supplement["sources"].keys(), "Isolation runner source inventory is incomplete")
    for name, digest in supplement["sources"].items():
        verify_digest(archived(base.WORK, name), digest)
    check(isolation["isolation_protocol_sha256"] == execution["isolation_protocol_sha256"] == supplement_digest,
          "Worker and monitor used a different isolation supplement")
    verify_digest(execution_path, isolation["execution_sha256"])
    timing = base.read(REPEAT/"timing.json")
    check(isolation["pid"] == execution["pid"] == timing["parent_pid"], "Monitored PID differs from the timed worker")
    check(execution["status"] == "complete", "Monitored worker did not complete")
    seconds, cadence = supplement["window_seconds"], supplement["sample_seconds"]
    check((seconds, cadence, supplement["preflight_windows"]) == (600, 10, 2) and
          isolation["window_seconds"] == seconds and isolation["sampling_interval_seconds"] == cadence,
          "Fixed observation duration or cadence changed")
    start, end, planned = [timestamp(isolation[k]) for k in ("start_utc", "end_utc", "planned_end_utc")]
    worker_start, worker_end = [timestamp(execution[k]) for k in ("start_utc", "end_utc")]
    check((planned-start).total_seconds() == seconds and start <= worker_start < worker_end and planned <= end,
          "Invalid worker or fixed observation window chronology")
    finished_within = worker_end <= planned
    samples = isolation["samples"]
    thresholds = supplement["thresholds"]
    expected_thresholds = dict(preflight_gpu_percent=5, background_individual_cpu_seconds=.5,
                               background_total_cpu_seconds=1., unexpected_python_cpu_seconds=.2,
                               maximum_sample_lateness_seconds=2.)
    check(thresholds == expected_thresholds, "Fixed-window activity thresholds changed")
    allowed = set(isolation["allowed_background_pids"])
    worker = execution["pid"]
    check(worker not in allowed, "The worker is listed as external background")
    preflight = isolation["preflight"]
    check(len(preflight) == supplement["preflight_windows"] and
          timestamp(preflight[0]["utc"]) < timestamp(preflight[1]["utc"]) <= start and
          (start-timestamp(preflight[0]["utc"])).total_seconds() >= cadence and
          isolation["initial_gpu_percent"] == preflight[-1]["gpu_percent"],
          "Preflight coverage or initial GPU reading differs")
    for row in preflight:
        check(0 <= row["gpu_percent"] <= thresholds["preflight_gpu_percent"] and
              all(item["pid"] in allowed for item in row["external_python"]) and
              not monitor_activity(row["external_python"], allowed, worker, thresholds),
              "An initial background monitoring interval exceeded its gate")
    targets = list(range(cadence, seconds+1, cadence))
    check([row["target_elapsed_seconds"] for row in samples] == targets, "Fixed-window samples are missing or reordered")
    elapsed = [row["elapsed_seconds"] for row in samples]
    coverage = all(math.isfinite(t) and 0 <= t-target <= thresholds["maximum_sample_lateness_seconds"]
                   for t, target in zip(elapsed, targets))
    check(all(a < b for a, b in zip(elapsed, elapsed[1:])) and
          all(start <= timestamp(row["utc"]) <= end for row in samples), "Monitor observation ordering differs")
    activity = [r["target_elapsed_seconds"] for r in samples
                if monitor_activity(r["external_python"], allowed, worker, thresholds)]
    if not coverage or not finished_within:
        status = "completed_outside_fixed_window"
    elif activity:
        status = "completed_with_observed_external_activity"
    else:
        status = "completed_no_sampled_external_activity"
    check(isolation["status"] == status and isolation["exit_code"] == 0 and
          isolation["external_activity_detected"] == bool(activity) and isolation["sample_coverage_pass"] == coverage and
          isolation["solver_finished_within_window"] == finished_within,
          "Recorded isolation classification differs from fixed-window telemetry")
    report = base.read(REPEAT/"summary.json")
    summary = report["isolation"]
    shared = ("status", "start_utc", "end_utc", "planned_end_utc", "isolation_protocol_sha256", "execution_sha256",
              "initial_gpu_percent", "final_gpu_percent", "sampling_interval_seconds")
    check(summary["fixed_window_seconds"] == seconds and summary["record"] == isolation_path.name and
          summary["samples"] == len(samples) and all(summary[key] == isolation[key] for key in shared),
          "Final summary refers to another isolation record")
    verify_digest(isolation_path, summary["record_sha256"])
    isolated_claim = status == "completed_no_sampled_external_activity" and not report["failures"]
    check(report["isolated_speed_claim_supported"] == isolated_claim, "Speed claim differs from the observed isolation")
    return dict(status=status, fixed_window_seconds=seconds, sampled_windows=len(samples),
                external_activity_windows=activity, isolated_speed_claim_supported=isolated_claim,
                separate_cohorts=True, scope=supplement["scope"])


def verify_reference_replays():
    pow2_protocol = base.read(POW2_PROTOCOL)
    cpu = pow2_reference(pow2_protocol)
    statuses = {}
    for arm, accepted in (("dense", False), ("band", True)):
        path = HERE/f"fd_cuda_pow2_replay_{arm}.npz"
        recorded = base.read(path.with_suffix(".json"))
        verify_digest(path, recorded["artifact_sha256"])
        verify_digest(POW2_PROTOCOL, recorded["protocol_sha256"])
        sol = case.load_solution(path, Pow2EquilibratedBufferedFD)
        check(recorded["protocol"] == pow2_protocol and recorded["cpu"] == cpu.row and sol.row == recorded["gpu"],
              f"Pow2 {arm} protocol or metadata differs")
        scored = compare_pow2(sol, cpu, pow2_protocol)
        check(scored["passed"] is accepted, f"Pow2 {arm} status differs")
        numeric_match(scored, {key: recorded[key] for key in scored}, f"pow2 {arm}")
        statuses[arm] = accepted
    rowcomp_protocol = base.read(ROWCOMP_PROTOCOL)
    path = HERE/"fd_cuda_rowcomp_verification.json"
    recorded = base.read(path)
    verify_digest(ROWCOMP_PROTOCOL, recorded["protocol_sha256"])
    check(recorded["protocol"] == rowcomp_protocol, "Row-moment verification protocol differs")
    gpu_record = HERE/"fd_cuda_rowcomp_t72.json"
    march = base.read(gpu_record)
    verify_digest(HERE/"fd_cuda_rowcomp_pilot.json", march["pilot_sha256"])
    recomputed = verify_rowcomp(rowcomp_protocol, gpu_record)
    check(recomputed["paired"]["passed"] and len(recomputed["paired"]["fields"]) == 21,
          "Row-moment paired accuracy, unshifted probes or shedding gate differs")
    numeric_match(recomputed, {key: recorded[key] for key in recomputed}, "row-moment paired replay")
    return dict(pow2_accepted=statuses, rowcomp_paired_passed=True, rowcomp_field_rows=21,
                rowcomp_probes=6, rowcomp_crossing_series=6)


def short_replays(frozen):
    from basis_recurrence_model import native_model
    from galerkin_cuda_native import solve_native

    template = frozen["template"]
    protocol = frozen["inputs"]["solve_protocol"]
    configs = {r["name"]: r for r in protocol["variants"]}
    horizon, rate = 8., template["galerkin_rate"]
    seeds = {"rfm37_k72": 7331, "leg37_k72": 17}
    check(set(seeds) == set(template["galerkin_names"]) and rate == 112 and
          7331 in protocol["seeds"] and protocol["deterministic_seed"] == 17,
          "Unexpected short-replay selection")
    grids = base.grid_set()
    rows = []
    for name, seed in seeds.items():
        _, settings = base.config(configs[name], protocol, seed)
        args = dict(steps=int(horizon*rate), horizon=horizon, snapshots=(4., 8.), **settings)
        cpu = case.solve(RecurrenceGalerkin, **args)
        native = case.solve(NativePrecontractedGalerkin, **args)
        with native_model():
            gpu = solve_native(graph_steps=template["cuda_graph_steps"], **args)
        fields = case.reference_fields(cpu, grids)
        for arm, sol in (("cpu_native_contract", native), ("cuda_graph", gpu)):
            check(np.array_equal(sol.times, cpu.times) and np.array_equal(sol.probe_times, cpu.probe_times) and
                  np.isfinite(sol.states).all(), f"Invalid short trajectory for {name}/{arm}")
            scores = worst(case.score(sol, fields, grids))
            probe = {}
            for key in ("transverse", "vorticity"):
                truth = getattr(cpu, key)
                norm = np.linalg.norm(truth)
                check(norm > 1e-12, f"Short replay probe too small: {name}/{key}")
                probe[key] = float(np.linalg.norm(getattr(sol, key)-truth)/norm)
            check(all(math.isfinite(v) and v <= template["implementation_tolerance"]
                      for v in (*scores.values(), *probe.values())), f"Short implementation gate failed: {name}/{arm}")
            rows.append(dict(name=name, seed=seed, arm=arm, rate=rate, horizon=horizon, fields=scores, probes=probe))
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true", help="Also run six T=8 trajectories for implementation consistency")
    args = cli.parse_args()
    isolation = verify_isolation()
    _, diagnostic = verify_timing(ORIGINAL, rescore=False)
    frozen, timing = verify_timing(REPEAT, rescore=True)
    reference = verify_reference_replays()
    rows = short_replays(frozen) if args.replay else []
    print(json.dumps(dict(status="passed", repeat_timing=timing, original_diagnostic=diagnostic,
                          isolation=isolation, reference=reference, short_replay=rows,
                          trajectory_solves_run=6 if args.replay else 0,
                          short_replay_scope="Implementation consistency only, without a T=72 physical acceptance claim."),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
