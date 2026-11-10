"""Independently verify dual clocks, numerical identities and sampled isolation without a new solve."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import freeze_rank_fair_timing as frozen
import verify_thin_formal as legacy
from rank_fair_owned_transport import SECOND, key, latch, read, require, require_identity


T,timing,qualify = frozen.T,frozen.timing,frozen.qualify
CLOCKS = ("solver_seconds","fresh_process_return_seconds")
PHASES = ("seconds","host_setup_seconds","runtime_init_seconds","nvrtc_compile_load_seconds","upload_seconds",
          "plan_seconds","graph_setup_seconds","march_seconds","download_seconds")


def verify_transport(output,row,request,entry,ledger,plan,opened):
    i = request["index"]
    request_file = output/f"request_{i:03d}.json"
    require(i == row["index"] == entry["index"] and row["status"] == "complete" and row["exit_code"] == 0 and
            entry["status"] == "exited" and not row["watchdog"] and row["artifact"] == request["trajectory"] and
            read(request_file) == request and T.digest(request_file) == row["request_sha256"] == entry["request_sha256"] and
            row["child_identity"] == entry["identity"] and row["terminal_identity"] == entry["terminal"] and
            key(entry["identity"]) == key(entry["terminal"]) and entry["terminal"]["exit_filetime_100ns"] > key(entry["identity"])[1],
            "Transport ownership or frozen request differs")
    launch,ready,go,returned,saved,exited = chronology = [row[k] for k in (
        "launch_monotonic_ns","ready_received_monotonic_ns","go_sent_monotonic_ns","return_received_monotonic_ns",
        "save_sent_monotonic_ns","exit_monotonic_ns")]
    hard = min(opened+plan["maximum_active_ns"],launch+round(plan["slot_hard_timeout_seconds"]*SECOND))
    observed_exit = row["exit_wait_completed_monotonic_ns"]
    require(chronology == sorted(chronology) and opened <= launch == entry["launch_monotonic_ns"] and exited == entry["exit_monotonic_ns"] and
            launch <= row["monitor_registration_qpc_ns"] == entry["monitor_registration_qpc_ns"] <= go and
            ready <= launch+plan["ready_timeout_seconds"]*SECOND and returned <= go+plan["return_timeout_seconds"]*SECOND and
            row["hard_deadline_ns"] == hard and saved <= observed_exit <= min(exited,hard,saved+plan["exit_timeout_seconds"]*SECOND),
            "Registered worker chronology or hard deadline differs")
    values = {event:read(latch(output,i,event)) for event in timing.LATCHES}
    for event,value in values.items():
        require_identity(value,event,request,row["request_sha256"],row["child_identity"])
    ready_value,marker = values["READY"],values["RETURN"]
    require(row["ready"] == ready_value and row["marker"] == marker and launch <= ready_value["published_monotonic_ns"] <= ready and
            all(values[event]["coordinator"] == ledger["coordinator"] and values[event]["sent_monotonic_ns"] == stamp
                for event,stamp in (("GO",go),("SAVE",saved))),"Authenticated machine latches differ from the journal")
    start,solved,release,released = [marker[k] for k in ("solve_started_ns","solve_completed_ns","model_release_started_ns","model_released_ns")]
    core,release_time = (solved-start)/SECOND,(released-release)/SECOND
    solver,fresh = row["solver_seconds"],row["fresh_process_return_seconds"]
    require(go <= start <= solved <= release <= released <= marker["published_monotonic_ns"] <= returned and marker["status"] == "solved" and
            marker["solver_core_seconds"] == core and marker["model_release_seconds"] == release_time and
            0 < solver == marker["solver_seconds"] == core+release_time <= row["go_to_return_seconds"] == (returned-go)/SECOND <= fresh and
            fresh == (returned-launch)/SECOND and row["fresh_process_overhead_seconds"] == fresh-solver,
            "Two independent clock boundaries differ")
    require(not (output/f"worker_{i:03d}.stdout.log").stat().st_size and not (output/f"worker_{i:03d}.stderr.log").stat().st_size,
            "Unexpected worker output in a successful slot")
    return legacy.checked_file(output,row["artifact"],row["trajectory_sha256"])


def verify_trajectory(path, request, control, row):
    import numpy as np

    p = qualify.shared.bound(control["artifact"])
    require(row["marker"]["spatial_sha256"] == control["spatial_sha256"],"Exact Galerkin or FD space differs")
    with np.load(path,allow_pickle=False) as data, np.load(p,allow_pickle=False) as old:
        keys = set(legacy.thin.ARRAYS)|{"config","metadata"}
        require(set(data.files) == set(old.files) == keys and json.loads(str(data["config"])) == request["config"] == json.loads(str(old["config"])),
                "Raw timed archive or configuration differs")
        metadata,original = (json.loads(str(a["metadata"])) for a in (data,old))
        require(legacy.thin.non_time_metadata(metadata) == legacy.thin.non_time_metadata(original),
                "Timed exact-runtime work or metadata differs from qualification")
        times = request["solve_args"]["snapshots"]
        require(np.array_equal(data["times"],times) and np.array_equal(old["times"],control["job"]["args"]["snapshots"]) and
                np.array_equal(data["probe_times"],np.arange(577)/8),"Timed physical or probe clock differs")
        ids = [int(np.flatnonzero(old["times"] == t).item()) for t in times]
        expected = {k:old[k] for k in ("probe_times","transverse","vorticity")}
        expected["states"] = old["states"][ids]
        require(all(data[k].shape == a.shape and data[k].dtype == a.dtype and np.isfinite(data[k]).all() and
                    data[k].tobytes() == a.tobytes() for k,a in expected.items()),
                "Saved trajectory or full probes differ from the qualified exact runtime")
    phases = {name:metadata[name] for name in PHASES}
    require(all(math.isfinite(value) and 0 <= value <= phases["seconds"] for value in phases.values()) and
            0 < phases["seconds"] <= row["solver_seconds"] and phases["nvrtc_compile_load_seconds"] <= phases["runtime_init_seconds"],
            "Nested phase clocks or full solver duration differ")
    return dict(qualified=True,gate="exact_qualified_runtime_saved_states_and_577_probes",phases=phases,
                trajectory_sha256=row["trajectory_sha256"],clocks={k:row[k] for k in CLOCKS})


def compare_scores(a,b):
    if isinstance(a,dict):
        return set(a) == set(b) and all(compare_scores(value,b[k]) for k,value in a.items())
    if isinstance(a,list):
        return len(a) == len(b) and all(compare_scores(x,y) for x,y in zip(a,b,strict=True))
    if isinstance(a,float):
        return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-13)
    return a == b


def independent_rescore(directory,q,indices):
    evaluate,_ = qualify.shared.assessment(directory,q)
    evidence = []
    for i in sorted(set(indices)):
        slot = directory/f"slot_{i:03d}"
        saved = read(slot/"qualification.json")
        replay = evaluate(i,read(slot/"result.json"))
        strict = qualify.strict_observer(slot/"trajectory.npz",q["slots"][i],q)
        require(replay["qualified"] and all(compare_scores(value,saved[k]) for k,value in replay.items()) and
                compare_scores(strict,saved["strict_observer"]),"Independent full-field rescore differs from saved qualification")
        evidence.append(dict(index=i,job=qualify.order(saved["job"]),physical_worst=replay["physical_worst"],
                             guard_worst=replay["guard_worst"],implementation_worst=replay["implementation_worst"],
                             probes=replay["probes"],field_source="same arrays as this block's actual timed NPZ"))
    return evidence


def verify(output):
    raw_plan = (output/"protocol.json").read_bytes()
    plan = json.loads(raw_plan)
    q,directory = frozen.verify_frozen(plan)
    summary = read(output/"summary.json")
    require(summary["protocol_sha256"] == sha256(raw_plan).hexdigest() and summary["intended_slots"] == len(plan["requests"]) and
            not summary["isolated_speed_claim_supported"],"Raw coordinator promoted timing or changed registration")
    hashes = summary["raw_hashes"]
    for name,digest in hashes.items():
        legacy.checked_file(output,name,digest)
    environment_ok,environment_sha = legacy.guard_accepted(output,plan,summary)
    raw = read(output/"runs.json") if "runs.json" in hashes else []
    ledger = read(output/"identity_ledger.json") if "identity_ledger.json" in hashes else {"children":[]}
    require(len(raw) == summary["attempted_slots"] <= len(plan["requests"]) and
            summary["unattempted_slots"] == list(range(len(raw),len(plan["requests"]))) and
            summary["successful_slots"] == sum(row["status"] == "complete" for row in raw),"Failed/unattempted queue was altered")
    entries = {entry["index"]:entry for entry in ledger["children"]}
    require(len(entries) == len(ledger["children"]) and set(entries) <= set(range(len(raw))),"Owned lifecycle ledger differs")
    try:
        isolation = legacy.verify_monitor(output,plan,summary,ledger)
    except Exception as error:
        isolation = dict(verified=False,eligible=False,reason="independent_monitor_audit_failed",failure_type=type(error).__name__)
    rows,recompute = [],[]
    for i,record in enumerate(raw):
        request = plan["requests"][i]
        index = plan["slot_controls"][i]
        job = {k:request[k] for k in ("name","arm","imports","engine","block","seed")}
        row = dict(index=i,job=job,control_index=index,qualified=False)
        name = f"attempt_{i:03d}.json"
        require(name in hashes and read(output/name) == record and record["index"] == i,"Raw attempt was not preserved")
        try:
            require(all(p.name in hashes for p in timing.evidence_files(output,request) if p.is_file()) and
                    all(latch(output,i,event).name in hashes for event in timing.LATCHES),"Attempt omitted committed or partial evidence")
            p = verify_transport(output,record,request,entries[i],ledger,plan,summary["open_qpc_ns"])
            row.update(verify_trajectory(p,request,plan["controls"][index],record))
            if job["block"] == 1:
                recompute.append(index)
        except Exception as error:
            row.update(qualified=False,gate="failed_closed",failure_type=type(error).__name__,raw_status=record["status"])
        rows.append(row)
    fields,recompute_failure = [],None
    if recompute:
        try:
            fields = independent_rescore(directory,q,recompute)
        except Exception as error:
            recompute_failure = type(error).__name__
    all_numerical = (len(rows) == len(plan["requests"]) and all(row["qualified"] for row in rows) and
                     len(fields) == len(plan["groups"]) and recompute_failure is None)
    accepted = bool(all_numerical and isolation["eligible"] and environment_ok and
                    summary["status"] == "awaiting_independent_numerical_qualification")
    groups,pairs = legacy.paired_statistics(plan,rows,accepted)
    for group in groups:
        matching = sorted((r for r in rows if [r["job"][k] for k in ("name","arm","imports")] == group["group"]),key=lambda r:r["job"]["block"])
        group["phases"] = {name:legacy.distribution([r["phases"][name] for r in matching]) for name in PHASES} if group["clocks"] else {}
    for pair in pairs:
        pair["descriptive_rank_contrast"] = pair["kind"] in ("same_retained_rank_descriptive","rfm_vs_lower_rank_legendre","legendre_rank_cost")
    T.check_sources(plan["source_hashes"])
    require((output/"protocol.json").read_bytes() == raw_plan,"Protocol changed during independent audit")
    return dict(status="sampled_isolated_and_numerically_qualified" if accepted else "diagnostic_only",
                scale=plan["scale"],protocol_sha256=summary["protocol_sha256"],coordinator_summary_sha256=T.digest(output/"summary.json"),
                verifier_source_sha256=T.digest(Path(__file__)),environment_guard_sha256=environment_sha,scope=
                "Observed Python process lifetimes and aggregate GPU preflight endpoints. Short unsampled work can be missed.",
                registered_slots=len(plan["requests"]),attempted_slots=len(raw),numerical_qualified_slots=sum(r["qualified"] for r in rows),
                isolated_speed_claim_supported=accepted,numerical_gate_all_slots=all_numerical,monitor=isolation,rows=rows,groups=groups,
                paired=pairs,independent_field_rescores=fields,independent_rescore_failure_type=recompute_failure,solves_run=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    for name in T.THREADS:
        os.environ[name] = "1"
    require(not args.output.exists(),"Preserve previous independent evidence")
    certificate = verify(args.cohort.resolve())
    T.save(args.output,certificate)
    print(json.dumps({k:certificate[k] for k in ("status","registered_slots","attempted_slots","numerical_qualified_slots","solves_run")}))
