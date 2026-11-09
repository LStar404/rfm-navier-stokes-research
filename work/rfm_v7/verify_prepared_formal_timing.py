"""Independently audit the archived prepared CUDA timings without running a solver."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import random
from statistics import median
from types import SimpleNamespace

import switch_formal_timing as archived
import verify_switch_qualification as fields_check

base, case, np = archived.base, archived.case, archived.np
require = fields_check.require
HERE = base.HERE
RECORD = HERE / "prepared_formal_timing_verification.json"
FILES = ("protocol", "timing", "execution", "isolation", "identity_ledger", "coordinator_admission", "summary")
COUNTS = {"mid": 55, "high": 40}
MODEL = archived.MODEL.RecurrenceGalerkin
METRICS = fields_check.METRICS
CLOCKS = ("solver_seconds", "fresh_process_return_seconds")
PROBES = ("transverse", "vorticity")
JOB_FIELDS = ("block", "name", "arm", "seed")
MARKER = b"PREPARED_TIMING "
SCHEMA = "win32-process-identity/v1"
TERMINAL_FIELDS = (("exit_filetime_100ns", "exit_filetime_100ns"),
                   ("kernel_cpu_100ns", "exit_kernel_cpu_100ns"),
                   ("user_cpu_100ns", "exit_user_cpu_100ns"))


def identity(item):
    return item["pid"], item["birth_filetime_100ns"]


def stem(job):
    return f"b{job['block']}_{job['name']}_{job['arm']}_{job['seed']}"


def design(protocol, scale):
    d, s = protocol["design"], protocol["scale_design"]
    rfm, leg = s["rfm"], s["legendre"]
    arms = s["timing_arms"]
    fd = [item["name"] for item in s["fd"]]
    require(protocol["status"] == "frozen" and protocol["scale"] == scale == s["name"] and
            d["rfm_seeds"] == [7331,7687,8011,8353,9041] and d["deterministic_seed"] == 17 and
            d["physical_gate"] == .005 and d["implementation_gate"] == 1e-5 and d["reference_budget_fraction"] == .25 and
            d["references"] == ["angular_fine","time_fine"] and d["probe_spacing"] == .125 and d["horizon"] == 72 and
            d["physical_times"] == [.5,1,2,4,6,8,8.125,8.5,9,10,12,16,24,36,48,54,60,66,72],
            "Prepared comparison differs from the registered observations and seeds")
    expected_arms = ["uniform","switch_old","switch_module","switch_host","switch_both"] if scale == "mid" else ["uniform","switch_old","switch_both"]
    expected_fd = ["fd176_k80"] if scale == "mid" else ["fd176_k80","fd184_k80"]
    rates = [128,112] if scale == "mid" else [152,136]
    names = ("rfm37_k72","leg37_k72") if scale == "mid" else ("rfm41_k80","leg39_k80")
    q = protocol["step_schedule"]
    require(arms == expected_arms and fd == expected_fd and
            [(x["rate"], x["linear"], x["seed"]) for x in s["fd"]] == [(128,"dense",17)] + ([(128,"band",17)] if scale == "high" else []) and
            s["rates"] == rates == [q["startup_rate"],q["late_rate"]] and q["steps"] == 8*rates[0]+64*rates[1] and
            (rfm,leg) == names and d["switch_time"] == 8 and d["cuda_graph_steps"] == 24 and
            q["checkpoints"] == sorted(d["physical_times"]+[8+i/rates[1] for i in (1,2)]),
            "Prepared comparison omitted the within-window FD control or an ablation")
    groups = [(name, arm) for name in (rfm,leg) for arm in arms] + [(name,"fd") for name in fd]
    rng = random.Random(s["order_seed"])
    order = []
    for block, seed in enumerate(d["rfm_seeds"],1):
        jobs = [dict(block=block,name=name,arm=arm,seed=seed if name == rfm else 17) for name,arm in groups]
        rng.shuffle(jobs)
        order.extend(jobs)
    require(order == protocol["schedule"] and len(order) == COUNTS[scale] and
            [list(group) for group in groups] == protocol["groups"], "Frozen schedule is not the five registered complete blocks")
    return groups


def check_clock(row, slot, token, cpu):
    require(identity(slot) == (row["child_pid"],row["child_birth_filetime_100ns"]) and row["index"] == slot["index"] and
            row["child_exit_code"] == slot["exit_code"] and row["launch_monotonic_ns"] == slot["launch_monotonic_ns"] and
            row["exit_monotonic_ns"] == slot["exit_monotonic_ns"], "Timed child does not match its owned process ledger")
    for key,event in (("ready","READY"),("marker","RETURN")):
        m = row[key]
        require(m["event"] == event and m["index"] == row["index"] and identity(m) == identity(slot) and
                m["run_token"] == token and m["inherited_affinity_mask"] == 1 << cpu, "Timed marker identity or affinity differs")
    names = ("launch_monotonic_ns","ready_received_monotonic_ns","go_sent_monotonic_ns",
             "return_received_monotonic_ns","save_sent_monotonic_ns","exit_monotonic_ns")
    if "solver_seconds" in row:
        launch, ready, go, ret, save, exit_ = (row[k] for k in names)
        solver,fresh,core,release,overhead = (row[k] for k in (*CLOCKS,"solver_core_seconds","model_release_seconds","fresh_process_overhead_seconds"))
        require(launch < ready <= go < ret <= save < exit_ and row["marker"]["status"] == "solved" and
                all(math.isfinite(v) for v in (solver,fresh,core,release,overhead)) and core > 0 and release >= 0 and
                0 < solver <= (ret-go)/1e9 <= fresh and (solver,fresh,overhead) == (core+release,(ret-launch)/1e9,fresh-solver) and
                all(row[k] == row["marker"][k] for k in ("solver_seconds","solver_core_seconds","model_release_seconds")),
                "Dual clocks differ from their raw transport boundaries")


def check_raw(directory, row, job, protocol, times):
    name = stem(job)
    require(base.read(directory / (name+".json")) == row, "Raw sidecar differs from journal")
    if "artifact" not in row:
        require(not row["qualified"] and not (directory / (name+".npz")).exists(), "Untracked trajectory or qualification")
        return None
    artifact = directory / (name+".npz")
    require(row["artifact"] == artifact.name and base.digest(artifact) == row["sha256"], "Raw trajectory SHA differs")
    if row["qualified"]:
        transcript = (directory / (name+".stdout.log")).read_bytes()
        require(transcript == b"".join(MARKER+json.dumps(row[k],separators=(",",":"),allow_nan=False).encode("ascii")+b"\n"
                                          for k in ("ready","marker")), "Raw READY/RETURN transcript differs")
    if job["arm"] == "fd":
        cfg = protocol["fd_controls"][job["name"]]["config"]
    else:
        cfg = base.config(protocol["specs"][job["name"]],protocol["solve_protocol"],job["seed"])[1]
    with np.load(artifact,allow_pickle=False) as data:
        require(set(data.files) == {"config","metadata","states","times","probe_times",*PROBES} and
                json.loads(str(data["config"])) == json.loads(json.dumps(cfg)), "Saved solver configuration differs")
        metadata = json.loads(str(data["metadata"]))
        raw = SimpleNamespace(**{k:data[k] for k in ("states","times","probe_times",*PROBES)})
    require(("metadata" not in row or row["metadata"] == metadata) and np.array_equal(raw.times,times) and
            np.array_equal(raw.probe_times,np.arange(577)/8) and raw.states.ndim == 3 and
            raw.states.shape[0] == len(times) and raw.states.dtype == np.complex128 and
            all(getattr(raw,k).shape == (577,3) for k in PROBES) and
            all(np.isfinite(getattr(raw,k)).all() for k in ("states",*PROBES)), "Raw observations, state or metadata differ")
    if "wake" in row:
        require(base.wake(raw) == row["wake"], "Archived wake does not match the saved probes")
    return artifact


def check_numerical(row, p, keys):
    if "implementation" not in row:
        require(not row["qualified"], "A qualified row omitted its error inventory")
        return
    d, job, metadata = p["design"], row["job"], row["metadata"]
    arm = job["arm"]
    fd = arm == "fd"
    switched = arm.startswith("switch_")
    rate = p["fd_controls"][job["name"]]["rate"] if fd else p["step_schedule"]["startup_rate"]
    steps = p["step_schedule"]["steps"] if switched else 72*rate
    modules = 2 if arm in ("switch_old","switch_host") else 1
    shared = arm in ("switch_host","switch_both")
    work = dict(steps=steps,nonlinear_evaluations=steps-1,marching_fft_batches=2*(steps-1),
                inverse_field_transforms=(4 if fd else 3)*(steps-1),forward_field_transforms=(1 if fd else 2)*(steps-1),
                compiled_cuda_modules=modules,phase_preparation_shared=shared,
                source="verified_device_switch_inventory" if switched else "uniform_registered_kernel")
    require(row["work"] == work and (metadata["steps"],metadata["horizon"],metadata["dt"],metadata["graph_steps"]) ==
            (steps,72,1/rate,d["cuda_graph_steps"]), "Saved time, work or graph inventory differs")
    if switched:
        split = 8*rate
        require(metadata["device_final_step"] == steps and metadata["switch_step"] == split and
                metadata["late_rate"] == p["step_schedule"]["late_rate"] and metadata["priming_fft_batches"] == 2 and
                metadata["bridge_extra_fft_batches"] == 0 and metadata["nonlinear_by_phase"] == {"startup":split,"late":steps-split-1} and
                all(metadata[k] == work[k] for k in ("nonlinear_evaluations","marching_fft_batches")), "Saved CUDA phase work differs")
        if arm != "switch_old":
            require(metadata["preparation"] == p["template"]["prepared_arms"][arm] and metadata["compiled_cuda_modules"] == modules and
                    metadata["phase_preparation_shared"] == shared, "Preparation variant was not actually applied")
    if fd:
        require(metadata["cuda_linear_solve"] == p["fd_controls"][job["name"]]["linear"], "FD implementation differs")
    impl = row["implementation"]
    physical = {k:fields_check.indexed(v,keys) for k,v in row["physical_scores"].items()}
    guards = {k:fields_check.indexed(v,keys) for k,v in row["guard_scores"].items()}
    expected_impl = {"cpu_switch","cuda_switch_old"} if switched else {"cpu_fd" if fd else "cpu_uniform"}
    require(set(impl) == expected_impl and set(physical) == set(d["references"]) and
            set(guards) == (set(physical) if switched else set()), "An error comparison is missing")
    ip = True
    for comp in impl.values():
        worst = fields_check.maxima(fields_check.indexed(comp["scores"],keys))
        require(worst == comp["worst"] and set(comp["probes"]) == set(PROBES), "Implementation maxima or probes differ")
        passed = all(math.isfinite(v) and 0 <= v <= d["implementation_gate"] for v in (*worst.values(),*comp["probes"].values()))
        require(comp["passed"] == passed, "Implementation gate differs from saved norms")
        ip &= passed
    flags = []
    for label, rows in (("physical",physical),("guard",guards)):
        worst = {ref:fields_check.maxima(scores) for ref,scores in rows.items()}
        flags.append(all(max(s.values()) <= d["physical_gate"] for s in worst.values()))
        require(worst == row[label+"_worst"] and flags[-1] == row[label+"_pass"], "Physical or guard gate differs from saved norms")
    require(all(physical[ref][gt][k] <= guards[ref][gt][k]+2e-13 for ref in guards for gt in keys for k in METRICS) and
            ip == row["implementation_pass"] and row["qualified"] == bool(ip and all(flags) and row["wake"]["sustained_alternating_probe_signs"]),
            "Archived triangle guard or combined qualification differs")


def ticks(iso):
    whole = datetime.fromisoformat(iso).astimezone(timezone.utc) - datetime(1601,1,1,tzinfo=timezone.utc)
    decimal = iso.partition(".")[2][:7]
    last = int(decimal[-1]) if len(decimal) == 7 and decimal.isdecimal() else 0
    return ((whole.days*86400+whole.seconds)*1_000_000+whole.microseconds)*10+last


def ledger_registry(p, m, t, l, a):
    known = identity(p["isolation"]["known_background"][0])
    coordinator = identity(a["coordinator"])
    require(known != coordinator and identity(m["coordinator"]) == identity(t["coordinator"]) == identity(l["coordinator"]) == coordinator and
            l["schema"] == a["schema"] == m["identity_schema"] == SCHEMA and a["status"] == "admitted", "Exact coordinator identity differs")
    rows, children = t["rows"], l["children"]
    roles = {known:("known_background",None),coordinator:("cohort_coordinator",None)}
    valid = len(rows) == len(children) == len(p["schedule"]) and t["status"] == l["status"] == "complete"
    candidate, terminal = {},{}
    last = 0
    for index,(child,row) in enumerate(zip(children,rows)):
        ident = identity(child)
        matched = (ident not in roles and ident not in candidate and child["status"] == "exited" and child["index"] == index and
                   (child["parent_pid"],child["parent_birth_filetime_100ns"]) == coordinator and
                   child["exit_filetime_100ns"] > ident[1] and min(child["exit_kernel_cpu_100ns"],child["exit_user_cpu_100ns"]) >= 0 and
                   last < child["launch_monotonic_ns"] < child["exit_monotonic_ns"] and
                   (row.get("child_pid"),row.get("child_birth_filetime_100ns")) == ident and all(k in row and identity(row[k]) == ident and
                       row[k]["index"] == index and row[k]["run_token"] == a["run_token"] for k in ("ready","marker")))
        valid &= matched
        if matched:
            check_clock(row,child,a["run_token"],t["logical_cpu"])
        candidate[ident] = ("cohort_child",index)
        terminal[ident] = child
        last = child.get("exit_monotonic_ns",last)
    require(valid == m["identity_ledger_valid"], "Monitor accepted a different identity ledger")
    return (roles | candidate, terminal, known, coordinator) if valid else (roles, {}, known, coordinator)


def check_monitor(p, m, e, t, l, a):
    roles,terminal,known,coordinator = ledger_registry(p,m,t,l,a)
    s = p["isolation"]
    duration, interval = s["window_seconds"],s["sample_seconds"]
    targets = list(range(interval,duration+1,interval))
    require(m["window_seconds"] == duration == ({"mid":600,"high":480}[p["scale"]]) and m["sampling_interval_seconds"] == interval == 10 and
            [r["target_elapsed_seconds"] for r in m["samples"]] == targets and m["known_background"] == s["known_background"], "Sample schedule differs")
    threshold = {k:Decimal(str(v))*10_000_000 for k,v in s["thresholds"].items() if k.endswith("_cpu_seconds")}

    def difference(before, after, registry):
        old = {identity(x):x for x in before["python"]}
        current = after["python"]
        require(len(old) == len(before["python"]) and len({identity(x) for x in current}) == len(current), "Duplicate snapshot identity")
        report = []
        for item in current:
            ident = identity(item)
            prev = old.get(ident,{})
            lab, index = registry.get(ident,("external_python",None))
            kernel = item["kernel_cpu_100ns"] - prev.get("kernel_cpu_100ns",0)
            user = item["user_cpu_100ns"] - prev.get("user_cpu_100ns",0)
            report.append(dict(pid=ident[0],birth_filetime_100ns=ident[1],label=lab,child_index=index,
                               kernel_delta_100ns=kernel,user_delta_100ns=user,cpu_delta_100ns=kernel+user,
                               previous_identity_observed=ident in old,exited=item["exit_filetime_100ns"] != 0))
        bg = [x["cpu_delta_100ns"] for x in report if x["label"] == "known_background"]
        ext = [x for x in report if x["label"] == "external_python" and x["cpu_delta_100ns"] > threshold["unexpected_python_cpu_seconds"]]
        active = bool(ext) or sum(bg)>threshold["background_total_cpu_seconds"] or max(bg,default=0)>threshold["background_individual_cpu_seconds"]
        return report,active,ext

    def resolved(snapshot):
        return not (snapshot["snapshot_failure_type"] or snapshot["unresolved_python"])

    def quiet(snapshot):
        py = snapshot["python"]
        return resolved(snapshot) and len(py) == 1 and identity(py[0]) == known and py[0]["exit_filetime_100ns"] == 0

    previous = m["initial_snapshot"]
    require(quiet(previous) and [r["window"] for r in m["preflight"]] == [1,2], "Preflight did not meet the frozen quiet policy")
    for pre in m["preflight"]:
        snap = pre["snapshot"]
        observed,activity,_ = difference(previous,snap,{known:("known_background",None)})
        require(quiet(snap) and observed == pre["classification"] and not activity and
                all(min(x["kernel_delta_100ns"],x["user_delta_100ns"]) >= 0 for x in observed) and
                0 <= pre["gpu_percent"] <= s["thresholds"]["preflight_gpu_percent"], "Raw preflight disagrees")
        previous = snap
    begin,deadline,end = (ticks(m[k]) for k in ("start_utc","planned_end_utc","end_utc"))
    admitted,start,stop = (ticks(v) for v in (a["admitted_utc"],e["start_utc"],e["end_utc"]))
    require(deadline == m["planned_end_filetime_100ns"] and deadline-begin == duration*10_000_000 and
            end >= deadline and m["end_elapsed_seconds"] >= duration and m["initial_gpu_percent"] == m["preflight"][-1]["gpu_percent"],
            "Fixed monitor window differs from the raw UTC and elapsed evidence")
    final = m["coordinator_exit"]
    within = bool(e["status"] == "complete" and begin <= admitted <= start <= stop <= deadline and final and
                  identity(final) == coordinator and stop <= final["exit_filetime_100ns"] <= deadline)
    coverage = m["final_gpu_percent"] is not None and 0 <= m["final_gpu_percent"] <= 100
    overlap,parents,ends = False,True,True
    activity_at, external_at, unresolved_at = [],[],[]
    terminal_rows = 0
    for sample in m["samples"]:
        target,snap = sample["target_elapsed_seconds"],sample["snapshot"]
        observed,activity,ext = difference(previous,snap,roles)
        require(observed == sample["classification"] and activity == sample["external_activity_detected"], "Raw CPU delta classification differs")
        if activity:
            activity_at.append(target)
        if ext:
            external_at.append(target)
        if not resolved(snap):
            unresolved_at.append(target)
        overlap |= sum(x["label"] == "cohort_child" and not x["exited"] for x in observed)>1
        for process in snap["python"]:
            ident = identity(process)
            if ident in terminal:
                parents &= process["enumerated_parent_pid"] == coordinator[0]
                if process["exit_filetime_100ns"]:
                    terminal_rows += 1
                    ends &= all(process[k] == terminal[ident][q] for k,q in TERMINAL_FIELDS)
        elapsed = sample["elapsed_seconds"]
        coverage &= (resolved(snap) and all(min(x["kernel_delta_100ns"],x["user_delta_100ns"]) >= 0 for x in observed) and
                     sample["sample_started_elapsed_seconds"] <= elapsed and 0 <= elapsed-target <= s["thresholds"]["maximum_sample_lateness_seconds"])
        previous = snap
    coverage &= parents and ends
    require((coverage,within,overlap,parents,ends,bool(activity_at)) == tuple(m[k] for k in (
        "sample_coverage_pass","solver_finished_within_window","child_overlap_detected","observed_child_parent_consistent",
        "observed_child_exit_consistent","external_activity_detected")), "Window verdict does not match the raw observations")
    status = "completed_no_sampled_external_activity"
    for condition, label in ((m["coordinator_force_stopped"],"stopped_after_post_window_grace"),(m["exit_code"] != 0,"failed"),
        (not m["identity_ledger_valid"],"completed_with_invalid_identity_ledger"),(not (coverage and within),"completed_outside_fixed_window"),
        (overlap,"completed_with_overlapping_children"),(bool(activity_at),"completed_with_observed_external_activity")):
        if condition:
            status = label
            break
    require(status == m["status"], "Terminal monitor status differs")
    return dict(samples=len(targets),status=status,external_activity_targets_seconds=activity_at,
                non_cohort_python_activity_targets_seconds=external_at,unresolved_snapshot_targets_seconds=unresolved_at,
                matched_terminal_process_rows=terminal_rows,coverage=coverage,coordinator_exited_within_window=within,
                scope="Sampled Python image prefixes, retained terminal CPU counters and aggregate GPU endpoints")


def stats(values):
    return dict(median=median(values),minimum=min(values),maximum=max(values),samples=values) if values else None


def paired_identities(protocol):
    s = protocol["scale_design"]
    rfm,leg = s["rfm"],s["legendre"]
    fds = [x["name"] for x in s["fd"]]
    pairs = set()
    for family in (rfm,leg):
        for kind,left,right in (("combined_preparation","switch_old","switch_both"),
                                ("combined_schedule","uniform","switch_both")):
            pairs.add((kind,(family,left),(family,right)))
        if s["name"] == "mid":
            for kind,left,right in (("module_at_legacy_host","switch_old","switch_module"),
                                    ("module_at_shared_host","switch_host","switch_both"),
                                    ("host_at_legacy_module","switch_old","switch_host"),
                                    ("host_at_bundled_module","switch_module","switch_both")):
                pairs.add((kind,(family,left),(family,right)))
        for arm in s["timing_arms"]:
            pairs.update(("same_cohort_fd_control",(fd,"fd"),(family,arm)) for fd in fds)
    pairs.update(("legendre_rfm",(leg,arm),(rfm,arm)) for arm in s["timing_arms"])
    if len(fds) == 2:
        pairs.add(("fd_radial_scale",(fds[0],"fd"),(fds[1],"fd")))
    return pairs


def check_statistics(protocol, rows, summary, groups):
    def components(row):
        phases = [(k,v) for data in (row,row["metadata"]) for k,v in data.items() if k.endswith("_seconds")]
        require(len(dict(phases)) == len(phases) and all(math.isfinite(v) and v >= 0 for _,v in phases), "Invalid timing component")
        return dict(phases)

    by_arm = {g:[r for r in rows if (r["job"]["name"],r["job"]["arm"]) == g] for g in groups}
    usable = {}
    expected = []
    for (name,arm), selection in by_arm.items():
        require([x["job"]["block"] for x in selection] == [1,2,3,4,5], "Incomplete paired block")
        qualified = sum(x["qualified"] for x in selection)
        usable[name,arm] = [components(r) for r in selection] if qualified == 5 else []
        records = usable[name,arm]
        common = set.intersection(*(set(x) for x in records)) if records else set()
        expected.append(dict(name=name,arm=arm,qualified=qualified,phase_seconds={k:stats([x[k] for x in records]) for k in common}))
    require(expected == summary["groups"], "Per-arm phase statistics differ")
    isolated = summary["isolated_speed_claim_supported"]
    comparisons = protocol["comparisons"]
    pair_keys = {(x["kind"],tuple(x["baseline"]),tuple(x["candidate"])) for x in comparisons}
    require(len(comparisons) == len(pair_keys) == {"mid":27,"high":20}[protocol["scale"]] and
            pair_keys == paired_identities(protocol),
            "Registered comparison inventory differs")
    rebuilt = []
    for comp in comparisons:
        a,b = (usable[tuple(comp[k])] for k in ("baseline","candidate"))
        clocks = dict.fromkeys(CLOCKS)
        phases = {}
        practical = False
        if a and b:
            for clock in CLOCKS:
                ratios = [x[clock]/y[clock] for x,y in zip(a,b,strict=True)]
                clocks[clock] = dict(stats(ratios),candidate_wins=sum(r>1 for r in ratios))
            shared = set.intersection(*(set(x) for x in a+b))
            phases = {k:stats([y[k]-x[k] for x,y in zip(a,b,strict=True)]) for k in shared}
            practical = all(clocks[k]["candidate_wins"] == 5 and clocks[k]["median"] >= 1.05 for k in CLOCKS)
        rebuilt.append(dict(comp,clocks=clocks,phase_delta_candidate_minus_baseline_seconds=phases,
                            meets_practical_threshold=practical,isolated_practical_improvement_supported=practical and isolated))
    require(rebuilt == summary["paired"], "Paired dual-clock or signed phase statistics differ")
    interactions = []
    for item in protocol["factorials"]:
        sign = item["coefficient"]
        require(sign == {"switch_old":1,"switch_module":-1,"switch_host":-1,"switch_both":1}, "Factorial coefficient differs")
        series = [usable[item["name"],arm] for arm in sign]
        components = {}
        for k in item["components"]:
            series_values = [[record.get(k) for record in group] for group in series]
            components[k] = stats([sum(c*v for c,v in zip(sign.values(),block,strict=True)) for block in zip(*series_values,strict=True)]) if (
                all(len(group) == 5 and None not in group for group in series_values)) else None
        interactions.append(dict(item,seconds=components))
    s = protocol["scale_design"]
    families = {s["rfm"],s["legendre"]} if protocol["scale"] == "mid" else set()
    require(interactions == summary["factorials"] and len(interactions) == len(families) and
            {x["name"] for x in interactions} == families, "Factorial interaction differs")
    return dict(groups=len(groups),paired_comparisons=len(comparisons),factorials=len(interactions))


def rescore(protocol, candidates, grids, times, keys):
    if not candidates:
        return [],None
    ref = protocol["reference_protocol"]
    path = base.WORK / ref["path"]
    require(base.digest(path) == ref["sha256"], "Independent fine reference protocol changed")
    truth, budget = fields_check.absolute_reference(base.read(path),grids)
    controls = {}

    def load(item, cls):
        target = base.WORK / item["path"]
        require(base.digest(target) == item["sha256"], "An archived same-schedule control changed")
        return case.load_solution(target,cls)

    for row, artifact in candidates:
        j = row["job"]
        ident = j["name"],j["seed"]
        fd = j["arm"] == "fd"
        if ident not in controls:
            if fd:
                sol = load(protocol["fd_controls"][j["name"]]["cpu"],base.FullFD)
                controls[ident] = {"cpu_fd":(sol,fields_check.fields_at(sol,times,grids))}
            else:
                c = next(c for c in protocol["gal_controls"] if (c["name"],c["seed"]) == ident)
                early, late, cpu, gpu = (load(c[k],MODEL) for k in ("early","late","cpu_recurrence","cuda_graph"))
                a = fields_check.fields_at(early,times[:12],grids)
                b = fields_check.fields_at(late,times[12:],grids)
                controls[ident] = {"cpu_uniform":(late,{g:a[g]+b[g] for g in grids}),
                                   "cpu_switch":(cpu,fields_check.fields_at(cpu,times,grids)),
                                   "cuda_switch_old":(gpu,fields_check.fields_at(gpu,times,grids))}
        actual = case.load_solution(artifact,base.FullFD if fd else MODEL)
        require(archived.model_signature(actual.model) == row["marker"]["spatial_sha256"], "Raw spatial signature differs")
        # Reconstruct physical fields directly from the raw solution. No timed-runner cache is used.
        actual_fields = fields_check.fields_at(actual,times,grids)
        for name, recorded in row["implementation"].items():
            control, target = controls[ident][name]
            require(np.array_equal(control.probe_times,actual.probe_times) and control.states.shape[1:] == actual.states.shape[1:],
                    "Raw implementation-control coordinates differ")
            scores = fields_check.measured(actual_fields,target,grids,times)
            fields_check.compare_scores(scores,fields_check.indexed(recorded["scores"],keys))
            for k in PROBES:
                a,b = getattr(actual,k),getattr(control,k)
                norm = np.linalg.norm(b)
                require(norm > 1e-12 and fields_check.same(float(np.linalg.norm(a-b)/norm),recorded["probes"][k]), "Full probe discrepancy differs")
        switched = bool(row["guard_scores"])
        if switched:
            uniform = controls[ident]["cpu_uniform"][1]
            delta = fields_check.measured(actual_fields,uniform,grids,times)
        for name, target in truth.items():
            physical = fields_check.measured(actual_fields,target,grids,times)
            fields_check.compare_scores(physical,fields_check.indexed(row["physical_scores"][name],keys))
            if switched:
                anchor = fields_check.measured(uniform,target,grids,times)
                bound = {gt:{k:anchor[gt][k]+(1+anchor[gt][k])*delta[gt][k] for k in METRICS} for gt in keys}
                fields_check.compare_scores(bound,fields_check.indexed(row["guard_scores"][name],keys))
    return [row["index"] for row,_ in candidates],budget


def verify_cohort(scale):
    directory = HERE / f"prepared_formal_timing_{scale}_output"
    paths = {k:directory / (k+".json") for k in FILES if (directory / (k+".json")).is_file()}
    data = {k:base.read(path) for k,path in paths.items()}
    digests = {path.name:base.digest(path) for path in paths.values()}
    p,m,s = (data[k] for k in ("protocol","isolation","summary"))
    groups = design(p,scale)
    require(s["scale"] == m["scale"] == scale and m["status"] == s["monitor_status"] and
            m["status"] not in ("preflight","launching","running") and
            all(s[k+"_sha256"] == digests[k+".json"] for k in paths if k != "summary") and
            m["protocol_sha256"] == digests["protocol.json"], "Summary hashes or terminal monitor differ")
    fields_check.checksums(p["source_hashes"])
    journal = data.get("timing",{})
    raw_rows = journal.get("rows",[])
    times,grids = p["design"]["physical_times"],base.grid_set()
    keys = {(g,t) for g in grids for t in times}
    require(len(raw_rows) == s["rows"] <= COUNTS[scale] and len(grids) == 3, "Archived row count or grid inventory differs")
    checked,candidates,rows,raw_hashes = 0,[],[],{}
    for i,raw in enumerate(raw_rows):
        job = {k:raw[k] for k in JOB_FIELDS}
        require("job" not in raw and raw["index"] == i and job == p["schedule"][i] and
                all(type(job[k]) is type(p["schedule"][i][k]) for k in JOB_FIELDS),
                "Flat timing row is outside the registered schedule")
        path = check_raw(directory,raw,job,p,times)
        sidecar = directory / (stem(job)+".json")
        raw_hashes[sidecar.name] = base.digest(sidecar)
        if path is not None:
            raw_hashes[path.name] = raw["sha256"]
        row = dict(raw,job=job)
        rows.append(row)
        check_numerical(row,p,keys)
        checked += path is not None
        if path is not None and "implementation" in row and row["job"]["block"] == 1:
            candidates.append((row,path))
    require(not any((directory / (stem(job)+ext)).is_file() for job in p["schedule"][len(rows):] for ext in (".json",".npz")),
            "An orphaned trajectory requires separate partial-evidence review")
    finished = len(rows) == COUNTS[scale] and journal.get("status") == data.get("execution",{}).get("status") == "complete"
    full = finished and m["status"] != "monitor_failed"
    numerical = full and all(row["qualified"] for row in rows)
    if finished:
        require(journal["numerical_qualified"] == all(row["qualified"] for row in rows), "Journal numerical flag differs")
    isolated = numerical and m["status"] == "completed_no_sampled_external_activity"
    if full or s["status"] in ("passed","diagnostic_only"):
        require(full and set(paths) == set(FILES), "Complete cohort is missing a required input")
    bound = {"timing","execution","identity_ledger","coordinator_admission"}
    monitor = dict(status=m["status"],samples=len(m["samples"]),scope="Incomplete window or missing terminal evidence; no isolation claim")
    if m["status"] != "monitor_failed" and bound <= set(data) and all(m.get(k+"_sha256") == digests[k+".json"] for k in bound) and "end_utc" in data["execution"]:
        t,e,l,a = (data[k] for k in ("timing","execution","identity_ledger","coordinator_admission"))
        require(all(r["protocol_sha256"] == digests["protocol.json"] and r["run_token"] == m["run_token"] for r in (t,e,l,a)) and
                e["identity_ledger_sha256"] == digests["identity_ledger.json"] and
                identity(e) == identity(a["coordinator"]) and t["environment"] == {k:"1" for k in archived.THREADS}, "Process/input binding differs")
        monitor = check_monitor(p,m,e,t,l,a)
    else:
        require(not full and not isolated, "A complete cohort lacks monitor/ledger bindings")
        token = m["run_token"]
        cpu = journal.get("logical_cpu")
        slots = {c["index"]:c for c in data.get("identity_ledger",{}).get("children",[]) if c["status"] == "exited"}
        for row in rows:
            slot = slots.get(row["index"])
            if slot is not None and cpu is not None and {"ready","marker"} <= row.keys():
                check_clock(row,slot,token,cpu)
    verdict = "passed" if isolated else "diagnostic_only" if full else "diagnostic_incomplete" if rows else "diagnostic_no_cohort"
    require(s["status"] == verdict and s["numerical_qualified"] == numerical and s["isolated_speed_claim_supported"] == isolated,
            "Formal summary overstates the numerical or isolation verdict")
    if "summary_postprocessor_sha256" in s:
        processor = HERE / "prepared_formal_summary.py"
        evidence = s["isolation_findings"]
        require(base.digest(processor) == s["summary_postprocessor_sha256"] and s["raw_evidence_sha256"] == raw_hashes and
                s["source_schema"]["raw_job_fields"] == list(JOB_FIELDS) and s["source_schema"]["job_view"] == "in-memory only" and
                evidence["external_activity_targets_seconds"] == monitor["external_activity_targets_seconds"] and
                [x["target_elapsed_seconds"] for x in evidence["unresolved_process_windows"]] == monitor["unresolved_snapshot_targets_seconds"] and
                all(x["unresolved_python"] == m["samples"][x["target_elapsed_seconds"]//10-1]["snapshot"]["unresolved_python"]
                    for x in evidence["unresolved_process_windows"]) and
                all(evidence[k] == m[k] for k in ("sample_coverage_pass","solver_finished_within_window","external_activity_detected")),
                "Derived flat-row adapter provenance or findings differ from the raw evidence")
    aggregate = check_statistics(p,rows,s,groups) if full else {}
    selected,budget = rescore(p,candidates,grids,times,keys)
    if numerical:
        require(len(selected) == len(groups), "Representative block omitted a registered arm")
    return dict(status="passed",timing_verdict=verdict,isolated_speed_claim_supported=isolated,rows=len(rows),
                qualified_rows=sum(r["qualified"] for r in rows),sidecars_verified=len(rows),raw_trajectory_hashes_verified=checked,
                registered_rows=COUNTS[scale],representative_rows_rescored=selected,physical_times=len(times),grids=len(grids),
                norms=len(METRICS),monitor_samples=monitor["samples"],monitoring=monitor,reference_budget=budget,
                frozen_numerical_input_hashes_verified=len(p["source_hashes"]),input_sha256=digests,**aggregate)


def verify():
    return dict(status="passed",time_solves_run=0,verifier_source_sha256=base.digest(Path(__file__)),
                cohorts={scale:verify_cohort(scale) for scale in COUNTS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",action="store_true",help="Save the verification certificate once")
    args = parser.parse_args()
    result = verify()
    if args.record:
        require(not RECORD.exists(), "Preserve the existing verification certificate")
        base.save(RECORD,result)
    print(json.dumps(result,indent=2))
