"""Independently check registered thin timing trajectories, clocks and paired claims."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

import freeze_thin_formal as frozen
import win32_isolation_monitor as monitor


T=frozen.transport
thin=frozen.thin
require=frozen.require
read=frozen.read
CLOCKS=("solver_seconds","fresh_process_return_seconds")
IDENTITY=("pid","birth_filetime_100ns")


def json_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def checked_file(output,name,expected):
    require(Path(name).name==name and not any(c in name for c in ":\\/") and T.digest(output/name)==expected,
            "A raw cohort file hash or name differs")
    return output/name


def compare_trajectory(path,request,control,spatial_sha,rescore):
    """Raw shape, observations and model identity gate both qualification paths."""
    import numpy as np

    require(request["config"]==control["config"] and request["solve_args"]==control["solve_args"] and
            spatial_sha==control["spatial_sha256"],"Timed solver settings or spatial reconstruction signature differ")
    with np.load(path,allow_pickle=False) as data:
        require(set(data.files)==set(thin.ARRAYS)|{"config","metadata"} and
                json.loads(str(data["config"]))==control["config"],"Timed NPZ schema or configuration differs")
        metadata=json.loads(str(data["metadata"]))
        require(frozen.canonical_sha(thin.non_time_metadata(metadata))==control["non_time_metadata_sha256"],
                "Timed non-time metadata or work inventory differs")
        require(np.array_equal(data["times"],request["solve_args"]["snapshots"]) and
                np.array_equal(data["probe_times"],np.arange(577)/8) and
                data["transverse"].shape==data["vorticity"].shape==(577,3),"Timed observations or full probe series differ")
        arrays={key:frozen.array_identity(data[key]) for key in thin.ARRAYS}
        require(all(arrays[key]["shape"]==control["arrays"][key]["shape"] and arrays[key]["dtype"]==control["arrays"][key]["dtype"] and
                    np.isfinite(data[key]).all() for key in arrays),"Timed dtype, array shape, or finiteness differs")
    comparisons={key:dict(value,byte_exact=value==control["arrays"][key]) for key,value in arrays.items()}
    result=dict(control_index=control["index"],arrays=comparisons,non_time_metadata_equal=True,spatial_signature_equal=True)
    if all(row["byte_exact"] for row in comparisons.values()):
        return dict(result,qualified=True,gate="inherited_bitwise_thin_control"),metadata
    scores=rescore(path,control,spatial_sha)
    return dict(result,qualified=bool(scores["qualified"]),gate="independent_v7_full_norm_rescore",scores=scores),metadata


def rescorer():
    context={}

    def score(path,control,spatial_sha):
        import prepared_formal_timing as v7

        protocol_record=control["archive_protocol"]
        key=protocol_record["sha256"]
        if key not in context:
            plan=read(thin.check_record(protocol_record))
            context[key]=(plan,v7.context(plan))
        plan,scoring=context[key]
        job=control["archive_job"]
        candidate=dict(job=job,marker=dict(spatial_sha256=spatial_sha))
        candidate.update(v7.score_solution(path,candidate,job,plan,scoring))
        v7.verify_numerical_row(candidate,plan)
        return candidate
    return score


def verify_slot(output,row,request,slot):
    index=request["index"]
    raw_request=output/f"request_{index:03d}.json"
    require(row["index"]==slot["index"]==index and row["status"]=="complete" and row["exit_code"]==0 and
            row["artifact"]==request["trajectory"] and read(raw_request)==request and
            T.digest(raw_request)==row["request_sha256"]==slot["request_sha256"] and
            row["terminal_identity"]==slot["terminal"] and row["child_identity"]==slot["identity"] and
            slot["status"]=="exited","Timed transport row differs from its exact request or ledger")
    keys=("launch_monotonic_ns","ready_received_monotonic_ns","go_sent_monotonic_ns","return_received_monotonic_ns",
          "save_sent_monotonic_ns","exit_monotonic_ns")
    launch,ready,go,returned,saved,exited=chronology=[row[key] for key in keys]
    core=row["marker"]["solver_core_seconds"]
    release=row["marker"]["model_release_seconds"]
    solver,fresh=row["solver_seconds"],row["fresh_process_return_seconds"]
    require(chronology==sorted(chronology) and launch==slot["launch_monotonic_ns"] and exited==slot["exit_monotonic_ns"] and
            launch<=row["monitor_registration_qpc_ns"]==slot["monitor_registration_qpc_ns"]<=go and
            0<solver==row["marker"]["solver_seconds"]==core+release<=row["go_to_return_seconds"]==(returned-go)/T.SECOND and
            fresh==(returned-launch)/T.SECOND and row["fresh_process_overhead_seconds"]==fresh-solver and
            fresh>=row["go_to_return_seconds"] and row["marker"]["status"]=="solved","Timed dual-clock chronology differs")
    for marker,event in ((row["ready"],"READY"),(row["marker"],"RETURN")):
        require(marker["event"]==event and marker["index"]==index and marker["run_token"]==request["run_token"] and
                marker["request_sha256"]==row["request_sha256"] and marker["inherited_affinity_mask"]==request["affinity_mask"] and
                all(marker[key]==row["child_identity"][key]==row["terminal_identity"][key] for key in IDENTITY),
                "Timed worker READY or RETURN identity differs")
    require(row["terminal_identity"]["exit_filetime_100ns"]>row["child_identity"]["birth_filetime_100ns"],
            "Timed terminal FILETIME is invalid")
    stdout=output/f"worker_{index:03d}.stdout.log"
    exact=b"".join(T.MARKER+T.raw_json(row[key]) for key in ("ready","marker"))
    require(stdout.read_bytes()==exact,"Timed stdout is not the two original machine marker lines")
    return checked_file(output,row["artifact"],row["trajectory_sha256"])


def verify_monitor(output,plan,summary,ledger):
    monitor_files=summary["monitor_evidence_sha256"]
    for name,sha in monitor_files.items():
        checked_file(output,name,sha)
    if not {"preflight.json","process_samples.jsonl","monitor_events.jsonl"}<=monitor_files.keys():
        return dict(verified=False,eligible=False,reason="incomplete_monitor_evidence")
    preflight=read(output/"preflight.json")
    samples=json_lines(output/"process_samples.jsonl")
    events=json_lines(output/"monitor_events.jsonl")
    require(all(command["serial"]==reply["serial"]==i and command["run_token"]==reply["run_token"]==plan["run_token"] and
                reply["event"]==command["event"]+"_ACK" for i,entry in enumerate(events,1)
                for command,reply in ((entry["command"],entry["reply"]),)),"Monitor event sequence differs")
    require(events and events[0]["command"]["event"]=="PREFLIGHT" and ledger["run_token"]==plan["run_token"] and
            events[0]["command"]["coordinator"]==ledger["coordinator"] and
            monitor.process_key(events[0]["reply"]["identity"])==monitor.process_key(ledger["monitor"]),
            "Monitor preflight/owned identities differ")
    known={monitor.process_key(row) for row in plan["known_background"]}
    own,coordinator=map(monitor.process_key,(ledger["monitor"],ledger["coordinator"]))
    anchors=preflight["sample_anchor_indices"]
    require(len(anchors)==3 and anchors==sorted(set(anchors)) and anchors[-1]<len(samples),"Preflight boundary anchors differ")
    computed=monitor.evaluate_quiet_preflight([samples[a:b+1] for a,b in zip(anchors,anchors[1:])],
        [row["gpu_percent"] for row in preflight["windows"]],plan["max_poll_gap_ns"],known_background=known|{coordinator},monitor_key=own)
    require(computed=={key:preflight[key] for key in ("passed","windows")} and
            preflight["passed"]==events[0]["reply"]["passed"],"Quiet preflight does not match the raw samples")
    if not preflight["passed"]:
        require(len(events)==1 and not ledger["children"] and summary["status"]=="preflight_rejected",
                "A rejected preflight launched registered work")
        return dict(verified=True,eligible=False,reason="preflight_rejected",preflight_passed=False)
    if "isolation.json" not in monitor_files:
        return dict(verified=False,eligible=False,reason="incomplete_active_monitor_evidence",preflight_passed=True)
    isolation=read(output/"isolation.json")
    require(isolation["preflight"]==preflight and isolation["close_raw_hashes"]==summary["raw_hashes"] and
            isolation["process_samples_sha256"]==monitor_files["process_samples.jsonl"] and
            summary["isolation_sha256"]==monitor_files["isolation.json"],"Active sample or close hash bindings differ")
    kinds=[entry["command"]["event"] for entry in events]
    require(kinds==["PREFLIGHT","OPEN",*(["REGISTER","EXIT"]*len(ledger["children"])),"CLOSE"],"Active event boundary sequence differs")
    opened,closed=(summary[key] for key in ("open_qpc_ns","close_qpc_ns"))
    opening,closing=events[1]["reply"],events[-1]
    first,last=opening["boundary_sample_index"],isolation["end_sample_index"]
    require(opened==opening["open_qpc_ns"]==isolation["open_qpc_ns"] and
            closed==closing["command"]["close_qpc_ns"]==isolation["close_qpc_ns"] and
            closing["command"]["raw_hashes"]==summary["raw_hashes"] and
            closing["reply"]["isolation_sha256"]==summary["isolation_sha256"] and
            samples[first]["collection_end_qpc_ns"]<=opened<closed<=samples[last]["qpc_start_ns"],
            "Active interval is not bracketed by the original observations")
    roles={own:"monitor",coordinator:"cohort_coordinator"}
    terminals={}
    previous=opened
    for i,slot in enumerate(ledger["children"]):
        reg,end=events[2+2*i:4+2*i]
        left,right=reg["command"],end["command"]
        key=monitor.process_key(slot["identity"])
        require(slot["index"]==left["index"]==right["index"]==i and slot["status"]=="exited" and key not in roles and
                left["identity"]==slot["identity"] and right["terminal"]==slot["terminal"] and
                monitor.process_key(slot["terminal"])==key and slot["terminal"]["exit_filetime_100ns"]>key[1] and
                previous<=left["launch_ns"]==slot["launch_monotonic_ns"]<=
                slot["monitor_registration_qpc_ns"]==reg["reply"]["ack_qpc_ns"]<=right["exit_ns"]==slot["exit_monotonic_ns"]<=closed,
                "Owned lifetime ledger or monitor registration differs")
        previous=right["exit_ns"]
        roles[key]="cohort_child"
        terminals[key]=slot["terminal"]
    try:
        decision=monitor.evaluate_active_interval(samples[:last+1],opened,closed,plan["maximum_active_ns"],
            plan["max_poll_gap_ns"],known_background=known,owned_roles=roles,owned_terminal=terminals)
    except ValueError:
        decision=dict(status="coverage_incomplete",reason="registered_active_duration_or_ledger",full_lifecycle_verified=False)
    if any(row["parent_pid"]!=coordinator[0] for sample in samples[:last+1] for row in sample["live"]
           if roles.get(monitor.process_key(row))=="cohort_child"):
        decision.update(status="coverage_incomplete",reason="observed_child_parent_differs")
    require(decision==isolation["active"] and decision["status"]==summary["active_verdict"] and
            not decision["full_lifecycle_verified"],"Active monitoring verdict does not match the frozen scope")
    eligible=decision["status"]=="no_sampled_external_activity"
    return dict(verified=True,eligible=eligible,reason=decision["status"],preflight_passed=True,
                active_samples=decision.get("active_samples"),max_poll_gap_ns=decision.get("max_poll_gap_ns"),
                full_lifecycle_verified=False)


def distribution(values):
    return dict(values=values,median=median(values),minimum=min(values),maximum=max(values))


def paired_statistics(plan,rows,eligible):
    groups={tuple(key):{} for key in plan["groups"]}
    for row in rows:
        job=row["job"]
        group=groups[job["name"],job["arm"],job["imports"]]
        require(job["block"] not in group,"Duplicate paired timing block")
        group[job["block"]]=row
    blocks=list(range(1,6))
    table,pairs=[],[]
    for key,group in groups.items():
        complete=set(group)==set(blocks) and all(row["qualified"] for row in group.values())
        stats={clock:distribution([group[b]["clocks"][clock] for b in blocks]) for clock in CLOCKS} if complete else {}
        table.append(dict(group=list(key),registered_blocks=blocks,qualified=sum(row["qualified"] for row in group.values()),
                          clocks=stats))
    rules=plan["design"]["reporting"]
    for pair in plan["paired"]:
        a,b=(groups[tuple(pair[key])] for key in ("baseline","candidate"))
        complete=set(a)==set(b)==set(blocks) and all(row["qualified"] for row in (*a.values(),*b.values()))
        clocks={}
        for clock in CLOCKS:
            if not complete:
                clocks[clock]=None
                continue
            ratios=[a[i]["clocks"][clock]/b[i]["clocks"][clock] for i in blocks]
            require(all(math.isfinite(v) and v>0 for v in ratios),"Paired clock ratio is invalid")
            wins=sum(v>1 for v in ratios)
            stats=distribution(ratios)
            practical=wins==rules["consistent_wins_required_per_clock"] and stats["median"]>=rules["practical_improvement_ratio"]
            clocks[clock]=dict(stats,candidate_wins=wins,meets_practical_threshold=practical,
                               isolated_practical_improvement_supported=bool(eligible and practical))
        solver=bool(complete and clocks[CLOCKS[0]]["meets_practical_threshold"])
        fresh=bool(complete and clocks[CLOCKS[1]]["meets_practical_threshold"])
        pairs.append(dict(pair,clocks=clocks,complete_five_pairs=complete,
            overall_two_clock_practical_improvement=bool(eligible and solver and fresh),
            startup_only_practical_improvement=bool(eligible and fresh and not solver and pair["kind"]=="import_only"),
            descriptive_rank_contrast=pair["kind"]=="rfm_legendre_descriptive"))
    return table,pairs


def guard_accepted(output,plan,summary):
    path=output/"execution_environment_guard.json"
    if not path.is_file():
        return False,None
    guard=read(path)
    if guard["status"]!="passed":
        return False,T.digest(path)
    require(guard["protocol_sha256"]==summary["protocol_sha256"] and guard["opening_environment"]==
            guard["closing_environment"]==plan["environment"] and guard["coordinator_summary_sha256"]==T.digest(output/"summary.json"),
            "Opening/closing hardware or raw journal hash differs")
    if "open_qpc_ns" in summary:
        require(guard["opening_guard_qpc_ns"]<=summary["open_qpc_ns"]<=summary["close_qpc_ns"]<=guard["closing_guard_qpc_ns"],
                "Environment guard did not bracket the complete timing cohort")
    return True,T.digest(path)


def verify(output):
    plan,protocol_bytes=T.protocol(output/"protocol.json")
    frozen.verify_frozen(plan)
    controls,_,qualification=frozen.load_controls(plan["design"])
    require(controls==plan["controls"] and qualification==plan["qualification"],"Prequalified control identities differ")
    summary=read(output/"summary.json")
    require(summary["protocol_sha256"]==sha256(protocol_bytes).hexdigest() and
            summary["intended_slots"]==len(plan["requests"]) and not summary["isolated_speed_claim_supported"],
            "The raw coordinator journal was prematurely promoted or differs")
    raw_hashes=summary["raw_hashes"]
    for name,sha in raw_hashes.items():
        checked_file(output,name,sha)
    environment_ok,environment_sha=guard_accepted(output,plan,summary)
    raw=read(output/"runs.json") if "runs.json" in raw_hashes else []
    ledger=read(output/"identity_ledger.json") if "identity_ledger.json" in raw_hashes else {"children":[]}
    require(len(raw)==summary["attempted_slots"]<=len(plan["requests"]) and
            sum(row["status"]=="complete" for row in raw)==summary["successful_slots"] and
            summary["unattempted_slots"]==list(range(len(raw),len(plan["requests"]))),
            "Failed or unattempted registered slots changed")
    try:
        isolation=verify_monitor(output,plan,summary,ledger)
    except Exception as error:
        isolation=dict(verified=False,eligible=False,reason="independent_monitor_audit_failed",failure_type=type(error).__name__)
    original_qualification=read(thin.check_record(plan["qualification"]["qualification"]))
    score=rescorer()
    rows=[]
    slots={row["index"]:row for row in ledger["children"]}
    require(len(slots)==len(ledger["children"]),"Duplicate owned child slot")
    for index,raw_row in enumerate(raw):
        request=plan["requests"][index]
        control=controls[plan["slot_controls"][index]]
        row=dict(index=index,job={key:request[key] for key in ("name","seed","block","arm","imports")})
        name=f"attempt_{index:03d}.json"
        require(name in raw_hashes and read(output/name)==raw_row and raw_row["index"]==index,"Raw attempt sidecar differs")
        try:
            require(all(name in raw_hashes for name in (request["trajectory"],f"request_{index:03d}.json",
                        f"worker_{index:03d}.stdout.log",f"worker_{index:03d}.stderr.log")),"Incomplete successful slot evidence")
            artifact=verify_slot(output,raw_row,request,slots[index])
            result,metadata=compare_trajectory(artifact,request,control,raw_row["marker"]["spatial_sha256"],score)
            if result["gate"]=="inherited_bitwise_thin_control":
                inherited=original_qualification["rows"][control["index"]]
                result.update(physical_worst=inherited["physical_worst"],guard_worst=inherited["guard_worst"],
                              implementation=inherited["implementation"],wake_passed=inherited["wake"]["sustained_alternating_probe_signs"])
            clocks={key:raw_row[key] for key in CLOCKS}
            require(all(math.isfinite(v) and v>0 for v in clocks.values()),"Invalid saved timing clock")
            row.update(result,clocks=clocks,trajectory_sha256=raw_row["trajectory_sha256"],
                       phases={key:value for key,value in metadata.items() if key.endswith("_seconds")})
        except Exception as error:
            row.update(qualified=False,gate="failed_closed",failure_type=type(error).__name__,raw_status=raw_row["status"])
        rows.append(row)
    all_numerical=len(rows)==len(plan["requests"]) and all(row["qualified"] for row in rows)
    eligible=all_numerical and isolation["eligible"] and environment_ok and summary["status"]=="awaiting_independent_numerical_qualification"
    groups,pairs=paired_statistics(plan,rows,eligible)
    T.check_sources(plan["source_hashes"])
    require((output/"protocol.json").read_bytes()==protocol_bytes,"Registered protocol changed during independent verification")
    return dict(status="sampled_isolated_and_numerically_qualified" if eligible else "diagnostic_only",
                protocol_sha256=summary["protocol_sha256"],coordinator_summary_sha256=T.digest(output/"summary.json"),
                verifier_source_sha256=T.digest(Path(__file__)),environment_guard_sha256=environment_sha,
                scope="Observed Python process scope with GPU preflight endpoints; complete lifecycle coverage is unavailable.",
                scale=plan["scale"],registered_slots=len(plan["requests"]),attempted_slots=len(raw),
                numerical_qualified_slots=sum(row["qualified"] for row in rows),isolated_speed_claim_supported=bool(eligible),
                numerical_gate_all_slots=all_numerical,monitor=isolation,rows=rows,groups=groups,paired=pairs,solves_run=0)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    require(not args.output.exists(),"Preserve prior independent verification")
    certificate=verify(args.cohort)
    T.save(args.output,certificate)
    print(json.dumps({key:certificate[key] for key in ("status","scale","registered_slots","attempted_slots","numerical_qualified_slots")}))
