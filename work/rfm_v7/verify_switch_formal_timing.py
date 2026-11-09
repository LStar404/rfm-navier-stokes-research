"""Verify archived variable-step timing and representative fields without solving NS."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import switch_formal_timing as formal
import verify_switch_qualification as calc

base, case, np = formal.base, formal.case, formal.np
require = calc.require
OUT = formal.OUT
RECORD = formal.HERE / "switch_formal_timing_verification.json"
ADMINISTRATIVE = "rfm_v7/operator_round_checkpoint.json"
CLOCKS = ("solver_seconds", "fresh_process_return_seconds")
COMPONENTS = ("solver_core_seconds", "model_release_seconds", "fresh_process_overhead_seconds")
PROBES = ("transverse", "vorticity")


def verify_schedule(frozen, timing):
    p = frozen["template"]
    arms = [(family, arm) for family in p["families"] for arm in p["galerkin_arms"]]
    arms += [(p["fd_name"], arm) for arm in p["fd_arms"]]
    require(len(set(arms)) == len(arms) == 12 and len(set(p["rfm_seeds"])) == p["blocks"] == 5,
            "Unexpected arm or seed inventory")
    rng = np.random.default_rng(p["order_seed"])
    expected = []
    for block, rfm_seed in enumerate(p["rfm_seeds"], 1):
        for i in rng.permutation(len(arms)):
            name, arm = arms[i]
            seed = rfm_seed if name == p["families"][0] else p["deterministic_seed"]
            expected.append(dict(block=block, name=name, seed=seed, arm=arm))
    require(frozen["schedule"] == expected and
            [{k: r[k] for k in formal.JOB_FIELDS} for r in timing["rows"]] == expected,
            "Schedule differs from the frozen randomization")


def verify_monitor(frozen, timing, monitor, execution):
    spec = frozen["isolation"]
    thresholds = spec["thresholds"]
    start, deadline, end = (datetime.fromisoformat(monitor[k]) for k in ("start_utc", "planned_end_utc", "end_utc"))
    first, last = (datetime.fromisoformat(execution[k]) for k in ("start_utc", "end_utc"))
    require(monitor["window_seconds"] == spec["window_seconds"] == 900 and
            monitor["sampling_interval_seconds"] == spec["sample_seconds"] == 10 and
            (deadline - start).total_seconds() == 900 and start <= first <= last <= deadline <= end and
            monitor["solver_finished_within_window"] and monitor["exit_code"] == 0,
            "The observed solver and monitor do not fit the fixed window")
    known = {spec["known_server_pid"]: "known_server", **{pid: "companion" for pid in spec["companion_pids"]}}
    require(set(monitor["allowed_background_pids"]) == set(known), "Background PID policy differs")

    def active(sample):
        entries = sample["external_python"]
        require(len({e["pid"] for e in entries}) == len(entries) and all(
            e["label"] == known.get(e["pid"], "python") and math.isfinite(e["cpu_seconds"]) and
            e["cpu_seconds"] >= 0 for e in entries), "Malformed external-process sample")
        background = [e["cpu_seconds"] for e in entries if e["pid"] in known]
        return (max(background, default=0) > thresholds["background_individual_cpu_seconds"] or
                sum(background) > thresholds["background_total_cpu_seconds"] or any(
                    e["pid"] not in known and e["cpu_seconds"] > thresholds["unexpected_python_cpu_seconds"] for e in entries))

    preflight = monitor["preflight"]
    require(len(preflight) == spec["preflight_windows"] == 2 and all(
        not active(s) and 0 <= s["gpu_percent"] <= thresholds["preflight_gpu_percent"] and
        all(e["pid"] in known for e in s["external_python"]) for s in preflight) and
        datetime.fromisoformat(preflight[0]["utc"]) < datetime.fromisoformat(preflight[1]["utc"]) <= start and
        0 <= monitor["initial_gpu_percent"] <= thresholds["preflight_gpu_percent"], "Quiet preflight differs")
    slots = {(r["child_pid"], r["index"]) for r in timing["rows"]}
    child_pids = {pid for pid, _ in slots}
    samples = monitor["samples"]
    targets = list(range(10, 901, 10))
    require([s["target_elapsed_seconds"] for s in samples] == targets, "Fixed sample targets differ")
    observed, outside, unmatched_children = [], [], []
    previous = start
    for sample, target in zip(samples, targets, strict=True):
        utc = datetime.fromisoformat(sample["utc"])
        registered = [(r["pid"], r["slot"]) for r in sample["cohort_child_slots"]]
        elapsed = sample["elapsed_seconds"]
        require(previous < utc <= end and abs((utc-start).total_seconds()-elapsed) < 0.2 and
                not sample["snapshot_failure_type"] and 0 <= sample["process_snapshot_seconds"] <= elapsed and
                0 <= elapsed-target <= thresholds["maximum_sample_lateness_seconds"] and
                len(registered) <= 1 and set(registered) <= slots and
                sample["cohort_children"] == [pid for pid, _ in registered] and
                (sample["cohort_coordinator_present"] or not registered), "Coverage, child identity or overlap differs")
        previous = utc
        if active(sample):
            observed.append(target)
            for e in sample["external_python"]:
                if e["label"] == "python" and e["cpu_seconds"] > thresholds["unexpected_python_cpu_seconds"]:
                    (unmatched_children if e["pid"] in child_pids else outside).append(target)
    require(outside and monitor["external_activity_detected"] and monitor["sample_coverage_pass"] and
            not monitor["child_overlap_detected"] and monitor["coordinator_creation_utc_ticks"] and
            monitor["final_gpu_percent"] is not None and
            monitor["status"] == "completed_with_observed_external_activity", "Contaminated-window verdict differs")
    return dict(samples=len(samples), observed_external_activity_targets_seconds=observed,
                targets_with_non_cohort_python_seconds=sorted(set(outside)),
                targets_with_external_label_matching_a_cohort_pid_seconds=sorted(set(unmatched_children)),
                status=monitor["status"], telemetry_scope="10-second sampled Python process counters")


def read_artifact(row, frozen, times):
    stem = formal.stem(row)
    path = OUT / (stem+".npz")
    require(row["artifact"] == path.name and base.digest(path) == row["sha256"] and
            base.read(OUT / (stem+".json")) == row, "Raw trajectory hash or sidecar differs")
    marker_lines = (OUT / (stem+".stdout.log")).read_bytes().splitlines()
    require([json.loads(line[len(formal.MARKER):]) for line in marker_lines if line.startswith(formal.MARKER)] ==
            [row["marker"]], "Saved return marker differs from journal")
    with np.load(path, allow_pickle=False) as data:
        require(json.loads(str(data["config"])) == json.loads(json.dumps(formal.settings(row, frozen))) and
                json.loads(str(data["metadata"])) == row["metadata"], "Raw solver metadata differs")
        raw = SimpleNamespace(**{k: data[k] for k in ("states", "times", "probe_times", *PROBES)})
    p = frozen["template"]
    probe_times = np.arange(0, p["horizon"]+p["probe_spacing"], p["probe_spacing"])
    require(np.array_equal(raw.times, times) and np.array_equal(raw.probe_times, probe_times) and
            raw.states.shape[0] == len(times) and all(getattr(raw,k).shape == (len(probe_times),3) for k in PROBES) and
            all(np.isfinite(getattr(raw,k)).all() for k in ("states", *PROBES)), "Raw observation coverage differs")
    return path, raw


def verify_row(row, frozen, timing, times, keys, previous_marker):
    p = frozen["template"]
    marker = row["marker"]
    begin, returned = row["parent_popen_monotonic_ns"], row["parent_marker_monotonic_ns"]
    core, release, overhead = (row[k] for k in COMPONENTS)
    solver, fresh = (row[k] for k in CLOCKS)
    require(row["status"] == "complete" and row["child_exit_code"] == 0 and marker["status"] == "solved" and
            marker["pid"] == row["child_pid"] and marker["index"] == row["index"] and
            marker["logical_cpu"] == timing["logical_cpu"] and marker["inherited_affinity_mask"] == 1 << timing["logical_cpu"] and
            all(marker[k] == row[k] for k in (CLOCKS[0],*COMPONENTS[:2])) and
            all(math.isfinite(x) and x >= 0 for x in (core,release,overhead,solver,fresh)) and
            previous_marker < begin < returned and 0 < solver <= fresh and core > 0 and
            solver == core+release and fresh == (returned-begin)/1e9 and overhead == fresh-solver,
            "Clock arithmetic, ordering, completion marker or affinity differs")
    path, raw = read_artifact(row, frozen, times)
    require(base.wake(raw) == row["wake"] and row["wake"]["sustained_alternating_probe_signs"], "Wake diagnostic differs")
    switched = row["arm"].endswith("_switch")
    steps = 8192 if switched else 9216
    work_source = "measured_switch_kernel" if switched else "uniform_registered_kernel"
    meta = row["metadata"]
    require(row["work"] == dict(steps=steps, nonlinear_evaluations=steps-1, marching_fft_batches=2*(steps-1),source=work_source) and
            (meta["steps"],meta["horizon"],meta["dt"]) == (steps,p["horizon"],1/p["uniform_rate"]) and
            all(math.isfinite(v) and v >= 0 for k,v in meta.items() if k.endswith("_seconds")), "Archived work differs")
    if switched:
        require(meta["switch_step"] == 1024 and meta["late_rate"] == 112 and meta["nonlinear_evaluations"] == steps-1 and
                meta["marching_fft_batches"] == 2*(steps-1) and meta["nonlinear_by_phase"] == {"startup":1024,"late":7167},
                "Bridge marching counters differ")
    if row["arm"].startswith("cuda_"):
        require(meta["graph_steps"] == p["graph_steps"], "CUDA graph chunk differs")
        if switched:
            require((meta["bridge_extra_fft_batches"],meta["priming_fft_batches"],meta["device_final_step"]) == (0,2,steps),
                    "CUDA bridge FFT counters differ")
        if row["name"] == p["fd_name"]:
            require(meta["cuda_linear_solve"] == p["cuda_fd_linear"], "FD CUDA linear solver differs")
    physical = {r:calc.indexed(s,keys) for r,s in row["physical_scores"].items()}
    guard = {r:calc.indexed(s,keys) for r,s in row["guard_scores"].items()}
    impl = calc.indexed(row["implementation_scores"],keys)
    require(set(physical) == set(p["references"]) and set(guard) == (set(physical) if switched else set()) and
            {r:calc.maxima(s) for r,s in physical.items()} == row["physical_worst"] and
            {r:calc.maxima(s) for r,s in guard.items()} == row["guard_worst"] and
            calc.maxima(impl) == row["implementation_worst"] and set(row["implementation_probes"]) == set(PROBES),
            "Worst errors differ from the complete grid-time table")
    require(all(physical[r][gt][k] <= guard[r][gt][k]+2e-13 for r in guard for gt in keys for k in calc.METRICS),
            "Physical discrepancy exceeds archived triangle guard")
    field_gate, impl_gate = p["physical_gate"],p["implementation_gate"]
    require(all(0 <= x <= field_gate for w in (row["physical_worst"],row["guard_worst"]) for s in w.values() for x in s.values()) and
            all(0 <= x <= impl_gate for w in (row["implementation_worst"],row["implementation_probes"]) for x in w.values()) and
            all(row[k] is True for k in ("qualified","physical_pass","guard_pass","implementation_pass")),
            "An archived trajectory failed a numerical gate")
    return path, returned


def verify_groups(rows, summary, p):
    def stats(values):
        return dict(median=median(values), minimum=min(values), maximum=max(values), samples=values)

    groups = {}
    for row in rows:
        groups.setdefault((row["name"],row["arm"]),[]).append(row)
    require(len(groups) == 12 and all([r["block"] for r in v] == list(range(1,6)) for v in groups.values()),
            "Missing block or duplicate arm observation")
    expected = []
    for (name, arm), group in groups.items():
        common = set.intersection(*(set(r["metadata"]) for r in group))
        phases = {k:stats([r["metadata"][k] for r in group]) for k in common if k.endswith("_seconds")}
        expected.append(dict(name=name,arm=arm,qualified=5,phase_seconds=phases,**{
            k:stats([r[k] for r in group]) for k in (*CLOCKS,*COMPONENTS)}))
    require(expected == summary["groups"], "Twelve per-arm time or phase aggregates differ")
    rfm, leg = p["families"]
    fd = p["fd_name"]
    pairs = []
    for family in (rfm,leg):
        pairs += [((family,a),(family,b)) for a,b in (("cpu_recurrence128","cpu_native128"),
                  ("cpu_native128","cpu_native_switch"),("cuda_graph128","cuda_graph_switch"))]
        pairs += [((fd,a),(family,b)) for a,b in (("cpu_buffered128","cpu_native128"),
                  ("cpu_buffered128","cpu_native_switch"),("cuda_graph128","cuda_graph128"),("cuda_graph128","cuda_graph_switch"))]
    pairs += [((leg,a),(rfm,a)) for a in p["galerkin_arms"]]
    rebuilt = []
    for a,b in pairs:
        comparison = dict(numerator=list(a),denominator=list(b))
        for clock in CLOCKS:
            values = [x[clock]/y[clock] for x,y in zip(groups[a],groups[b],strict=True)]
            comparison[clock] = dict(stats(values),denominator_wins=sum(v>1 for v in values))
        rebuilt.append(comparison)
    require(len(set(pairs)) == len(rebuilt) == 19 and rebuilt == summary["paired"], "Nineteen within-block ratios differ")


def rescore(representatives, frozen, grids, times, keys):
    qualification = base.read(formal.switch.FROZEN)
    require(qualification["physical_times"] == times, "Formal and reference evaluation times differ")
    truth, budget = calc.absolute_reference(qualification,grids)
    model = formal.MODEL.RecurrenceGalerkin
    controls = {}

    def load(record, cls):
        path = base.WORK / record["path"]
        require(base.digest(path) == record["sha256"], "Archived matched control changed")
        return case.load_solution(path,cls)

    for row,path in representatives:
        name,seed = row["name"],row["seed"]
        identity = (name,seed)
        if identity in controls:
            continue
        if name == frozen["template"]["fd_name"]:
            uniform = load(frozen["fd_control"],base.FullFD)
            controls[identity] = (uniform,calc.fields_at(uniform,times,grids),None,None)
        else:
            rec = next(r for r in frozen["gal_controls"] if (r["name"],r["seed"]) == identity)
            early,late,switch = (load(rec[k],model) for k in ("early","late","switched"))
            a = calc.fields_at(early,times[:12],grids)
            b = calc.fields_at(late,times[12:],grids)
            controls[identity] = (late,{g:a[g]+b[g] for g in grids},switch,calc.fields_at(switch,times,grids))
    for row,path in representatives:
        uniform,uniform_fields,switch,switch_fields = controls[row["name"],row["seed"]]
        switched = row["arm"].endswith("_switch")
        matched,matched_fields = (switch,switch_fields) if switched else (uniform,uniform_fields)
        # The child serialized spatial coefficients. Reconstruct on the matched spatial basis.
        actual = case.load_solution(path,type(matched.model))
        require(np.array_equal(actual.probe_times,matched.probe_times) and actual.states.shape[1:] == matched.states.shape[1:] and
                formal.model_signature(actual.model) == formal.model_signature(matched.model) == row["marker"]["spatial_sha256"],
                "Representative coordinate or observation shape differs")
        fields = calc.fields_at(actual,times,grids)
        calc.compare_scores(calc.measured(fields,matched_fields,grids,times),calc.indexed(row["implementation_scores"],keys))
        for k in PROBES:
            a,b = getattr(actual,k),getattr(matched,k)
            norm = np.linalg.norm(b)
            require(norm > 1e-12 and calc.same(np.linalg.norm(a-b)/norm,row["implementation_probes"][k]),
                    "Recomputed full-horizon probe discrepancy differs")
        delta = calc.measured(fields,uniform_fields,grids,times) if switched else None
        for ref, ref_fields in truth.items():
            calc.compare_scores(calc.measured(fields,ref_fields,grids,times),calc.indexed(row["physical_scores"][ref],keys))
            if switched:
                anchor = calc.measured(uniform_fields,ref_fields,grids,times)
                guard = {gt:{k:delta[gt][k]*(1+anchor[gt][k])+anchor[gt][k] for k in calc.METRICS} for gt in keys}
                calc.compare_scores(guard,calc.indexed(row["guard_scores"][ref],keys))
    return budget


def verify():
    paths = {f"{k}.json":OUT / f"{k}.json" for k in ("protocol","timing","execution","isolation","summary")}
    frozen,timing,execution,monitor,summary = (base.read(p) for p in paths.values())
    digests = {k:base.digest(path) for k,path in paths.items()}
    require(frozen["status"] == "frozen" and timing["status"] == execution["status"] == "complete" and
            timing["parent_pid"] == execution["pid"] == monitor["pid"] and
            all(record["protocol_sha256"] == digests["protocol.json"] for record in (timing,execution,monitor,summary)) and
            monitor["execution_sha256"] == summary["execution_sha256"] == digests["execution.json"] and
            summary["monitor_sha256"] == digests["isolation.json"] and summary["timing_sha256"] == digests["timing.json"] and
            timing["environment"] == {k:"1" for k in formal.THREADS}, "Input hashes, process identity or thread policy differs")
    fingerprints = frozen["source_hashes"]
    numerical = {p:s for p,s in fingerprints.items() if p != ADMINISTRATIVE}
    calc.checksums(numerical)
    verify_schedule(frozen,timing)
    times = frozen["template"]["physical_times"]
    grids = base.grid_set()
    require(len(set(times)) == len(times) == 19 and len(grids) == 3, "Formal physical observation grid differs")
    keys = {(grid,t) for grid in grids for t in times}
    rows = timing["rows"]
    previous_marker, representatives = 0, []
    for i,row in enumerate(rows):
        require(row["index"] == i,"Duplicate or reordered timing row index")
        path,previous_marker = verify_row(row,frozen,timing,times,keys,previous_marker)
        if row["block"] == 1:
            representatives.append((row,path))
    verify_groups(rows,summary,frozen["template"])
    observed = verify_monitor(frozen,timing,monitor,execution)
    require(summary["rows"] == len(rows) == 60 and timing["numerical_qualified"] and summary["numerical_qualified"] and
            summary["status"] == "diagnostic_only" and summary["monitor_status"] == observed["status"] and
            summary["isolated_speed_claim_supported"] is False, "Summary does not preserve the isolation verdict")
    budget = rescore(representatives,frozen,grids,times,keys)
    return dict(status="passed", timing_verdict=summary["status"], isolated_speed_claim_supported=False,
                numerical_qualified_rows=len(rows), raw_trajectory_hashes_verified=len(rows), sidecars_verified=len(rows),
                timing_groups=12, within_block_pairs=19, measured_clocks=list(CLOCKS), monitoring=observed,
                representative_rows_rescored=[r["index"] for r,_ in representatives], physical_times=len(times),
                grids=len(grids), norms=len(calc.METRICS), reference_budget=budget, time_solves_run=0,
                frozen_numerical_input_hashes_verified=len(numerical),
                administrative_input_policy="Historical checkpoint SHA retained in protocol; administrative text excluded from numerical replay",
                input_sha256=digests, verifier_source_sha256=base.digest(Path(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",action="store_true",help="Write the audit JSON once")
    result = verify()
    if parser.parse_args().record:
        require(not RECORD.exists(),"Preserve the existing independent verification")
        base.save(RECORD,result)
    print(json.dumps(result,indent=2))
