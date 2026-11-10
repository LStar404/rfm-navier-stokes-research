"""Verify exact cache-policy timing evidence and combine only complete fixed blocks."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import median

import freeze_rank_fair_cache_formal as frozen
import verify_rank_fair_timing as transport
from rank_fair_cache_formal_worker import CACHE_FIELDS
from rank_fair_owned_transport import latch,read,require


T,qual = frozen.T,frozen.qual
legacy = transport.legacy
CLOCKS = transport.CLOCKS


def verify_cache(row,request,metadata,output):
    meta = row["marker"]["cache"]
    first = request["policy"] == "empty_first"
    require(set(meta) == set(CACHE_FIELDS) and all(meta[k] == metadata[k] for k in CACHE_FIELDS) and
            meta["ptx_cache_policy"] == request["policy"] and meta["ptx_cache_observed_mode"] == ("miss" if first else "hit") and
            meta["ptx_cache_key"] == request["ptx_cache_key"] and meta["ptx_sha256"] == request["ptx_sha256"] and
            meta["ptx_artifact_sha256"] == request["ptx_artifact"]["sha256"] and meta["cuda_context_mode"] == request["runtime_id"] and
            meta["private_context_finally_unbound"] is True and meta["cuda_driver_api"] == 13020 and
            0 <= meta["ptx_nvrtc_seconds"] <= metadata["seconds"] and (meta["ptx_nvrtc_seconds"] > 0) == first,
            "A timed policy changed artifact, context or compiled unexpectedly")
    directory = output/f"explicit_cache_{request['index']:03d}"
    if first:
        actual = directory/(request["ptx_cache_key"]+".rfmptx")
        require(T.digest(actual) == meta["ptx_artifact_sha256"],"Freshly built timed artifact does not match RETURN and its fixed reference")
        return dict(path=actual.relative_to(output).as_posix(),sha256=meta["ptx_artifact_sha256"])
    require(not directory.exists(),"A strict-hit timed slot created a local cache namespace")
    return None


def verify_trajectory(path,request,control,row,output):
    import numpy as np

    require(row["marker"]["spatial_sha256"] == control["spatial_sha256"] and request["policy"] == control["job"]["policy"],
            "The timed spatial or exact policy identity differs")
    target = qual.shared.bound(control["artifact"])
    with np.load(path,allow_pickle=False) as a,np.load(target,allow_pickle=False) as b:
        keys = {"config","metadata","states","times","probe_times","transverse","vorticity"}
        require(set(a.files) == set(b.files) == keys and json.loads(str(a["config"])) == request["config"] == json.loads(str(b["config"])),
                "Timed result configuration differs")
        meta,reference = (json.loads(str(data["metadata"])) for data in (a,b))
        require(legacy.thin.non_time_metadata(meta) == legacy.thin.non_time_metadata(reference),"Timed kernel or work inventory differs")
        t = request["solve_args"]["snapshots"]
        require(len(t) == 19 and np.array_equal(a["times"],t) and np.array_equal(a["probe_times"],np.arange(577)/8) and
                np.array_equal(b["times"],control["job"]["args"]["snapshots"]),"Physical or complete probe times differ")
        ids = [int(np.flatnonzero(b["times"] == sample).item()) for sample in t]
        expect = {k:b[k] for k in ("probe_times","transverse","vorticity")}
        expect["states"] = b["states"][ids]
        require(all(a[k].shape == v.shape and a[k].dtype == v.dtype and np.isfinite(a[k]).all() and a[k].tobytes() == v.tobytes()
                    for k,v in expect.items()),"A timed field or complete probe array differs from exact qualification")
    local_ptx = verify_cache(row,request,meta,output)
    phases = {k:v for k,v in meta.items() if k == "seconds" or k.endswith("_seconds")}
    require(all(math.isfinite(value) and 0 <= value <= phases["seconds"] for value in phases.values()) and
            0 < phases["seconds"] <= row["solver_seconds"] and
            phases["ptx_nvrtc_seconds"] <= phases["ptx_cache_total_seconds"] and
            phases["nvrtc_compile_load_seconds"] <= phases["runtime_init_seconds"],"Invalid solver phase nesting")
    return dict(qualified=True,phases=phases,cache_artifact=local_ptx,trajectory_sha256=row["trajectory_sha256"],
                clocks={key:row[key] for key in CLOCKS})


def verify_window(output):
    raw = (output/"protocol.json").read_bytes()
    p = json.loads(raw)
    frozen.verify_plan(p)
    s = read(output/"summary.json")
    require(s["protocol_sha256"] == sha256(raw).hexdigest() and s["intended_slots"] == len(p["requests"]) and
            not s["isolated_speed_claim_supported"],"Raw coordinator prematurely promoted timing or changed protocol")
    hashes = s["raw_hashes"]
    for name,digest in hashes.items():
        legacy.checked_file(output,name,digest)
    rows = read(output/"runs.json") if "runs.json" in hashes else []
    ledger = read(output/"identity_ledger.json") if "identity_ledger.json" in hashes else {"children":[]}
    entries = {r["index"]:r for r in ledger["children"]}
    require(len(entries) == len(ledger["children"]) and set(entries) <= set(range(len(rows))) and
            len(rows) == s["attempted_slots"] <= len(p["requests"]) and
            sum(r["status"] == "complete" for r in rows) == s["successful_slots"] and
            s["unattempted_slots"] == list(range(len(rows),len(p["requests"]))),"Attempt or terminal journal changed")
    guard,guard_sha = legacy.guard_accepted(output,p,s)
    try:
        isolation = legacy.verify_monitor(output,p,s,ledger)
    except Exception as error:
        isolation = dict(verified=False,eligible=False,reason="independent_monitor_audit_failed",failure_type=type(error).__name__)
    measured = []
    for i,record in enumerate(rows):
        rq = p["requests"][i]
        name = f"attempt_{i:03d}.json"
        require(name in hashes and read(output/name) == record and record["index"] == i,"Raw attempt sidecar changed")
        row = dict(job={k:rq[k] for k in ("name","arm","policy","seed","block")},index=i,qualified=False)
        try:
            require(all(latch(output,i,event).name in hashes for event in frozen.timing.LATCHES) and
                    all(path.name in hashes for path in frozen.timing.evidence_files(output,rq) if path.is_file()),
                    "A raw transport latch or artifact is omitted")
            path = transport.verify_transport(output,record,rq,entries[i],ledger,p,s["open_qpc_ns"])
            row.update(verify_trajectory(path,rq,p["controls"][i],record,output))
        except Exception as error:
            row.update(qualified=False,raw_status=record["status"],failure_type=type(error).__name__)
        measured.append(row)
    complete = len(measured) == len(p["requests"]) and all(r["qualified"] for r in measured)
    accepted = bool(complete and guard and isolation["eligible"] and s["status"] == "awaiting_independent_numerical_qualification")
    T.check_sources(p["source_hashes"])
    require((output/"protocol.json").read_bytes() == raw,"Formal timing protocol changed during audit")
    report = dict(window=p["window"],scale=p["scale"],blocks=p["window_spec"]["blocks"],status="verified" if accepted else "diagnostic_only",
                  protocol_sha256=s["protocol_sha256"],summary_sha256=T.digest(output/"summary.json"),environment_guard_sha256=guard_sha,
                  eligible=accepted,complete=complete,monitor=isolation,rows=measured,unattempted=s["unattempted_slots"],
                  open_qpc_ns=s.get("open_qpc_ns"),close_qpc_ns=s.get("close_qpc_ns"))
    return p,report


def summary_of(values):
    return dict(median=median(values),minimum=min(values),maximum=max(values),values=values)


def compare_pair(table,baseline,candidate,eligible):
    a,b = table.get(tuple(baseline),{}),table.get(tuple(candidate),{})
    blocks = list(range(1,6))
    complete = set(a) == set(b) == set(blocks) and all(r["qualified"] for r in (*a.values(),*b.values()))
    ratios = {}
    if complete:
        for clock in CLOCKS:
            values = [a[i]["clocks"][clock]/b[i]["clocks"][clock] for i in blocks]
            wins = sum(value>1 for value in values)
            ratios[clock] = dict(**summary_of(values),wins=wins,practical=wins==5 and median(values)>=1.05)
    return dict(baseline=list(baseline),candidate=list(candidate),complete_five_pairs=complete,clocks=ratios,
                supports_both_clock_practical_gain=bool(eligible and complete and all(r["practical"] for r in ratios.values())))


def summarize(reports):
    plans = {p["window"]:p for p,_ in reports}
    require(len(reports) == len(plans) == 3 and set(plans) == set(frozen.WINDOWS) and all(
            frozen.evidence(p) == frozen.evidence(reports[0][0]) for p,_ in reports),"Three registered windows with a common exact scope are required")
    by_name = {r["window"]:r for _,r in reports}
    ordered = [by_name[name] for name in frozen.WINDOWS]
    intervals = [(r["open_qpc_ns"],r["close_qpc_ns"]) for r in ordered]
    chronological = all(type(a) is int and type(b) is int and 0 < a < b for a,b in intervals) and all(
        left[1] <= right[0] for left,right in zip(intervals,intervals[1:]))
    eligible = chronological and all(r["eligible"] for r in ordered)
    report = dict(status="sampled_isolated_and_numerically_qualified" if eligible else "diagnostic_only",
                  isolated_speed_claim_supported=eligible,scope="Scoped sampled isolation for all three independent prospective windows",
                  solves_run=0,windows=ordered,scales={},nonoverlapping_registered_order=chronological,verifier_sha256=T.digest(Path(__file__)))
    if not eligible:
        return report
    scoped = {}
    for scale in ("mid","high"):
        peers = [(p,r) for p,r in reports if p["scale"] == scale]
        spec = peers[0][0]["window_spec"]
        require(all(p["window_spec"]["method_contrasts"] == spec["method_contrasts"] and
                    p["window_spec"]["cache_contrasts"] == spec["cache_contrasts"] for p,_ in peers),"Fixed comparison scope changed across windows")
        table = {}
        for _,cohort in peers:
            for row in cohort["rows"]:
                job = row["job"]
                key = job["name"],job["arm"],job["policy"]
                group = table.setdefault(key,{})
                require(job["block"] not in group,"A timing block was duplicated or replayed")
                group[job["block"]] = row
        arms,pairs = [],[]
        for cache in spec["cache_contrasts"]:
            name,arm = cache["name"],cache["arm"]
            pairs.append(dict(kind="explicit_cache_policy",**compare_pair(table,(name,arm,cache["baseline"]),(name,arm,cache["candidate"]),eligible)))
            for policy in qual.inventory.POLICIES:
                key = name,arm,policy
                group = table.get(key,{})
                complete = set(group) == set(range(1,6)) and all(row["qualified"] for row in group.values())
                stats,phases = {},{}
                if complete:
                    order = [group[i] for i in range(1,6)]
                    stats = {k:summary_of([r["clocks"][k] for r in order]) for k in CLOCKS}
                    fields = set.intersection(*(set(r["phases"]) for r in order))
                    phases = {k:summary_of([r["phases"][k] for r in order]) for k in sorted(fields)}
                arms.append(dict(arm=list(key),numerically_qualified=sum(row["qualified"] for row in group.values()),clocks=stats,phases=phases))
        for method in spec["method_contrasts"]:
            for policy in qual.inventory.POLICIES:
                pairs.append(dict(kind=method["kind"],**compare_pair(table,(*method["baseline"],policy),(*method["candidate"],policy),eligible)))
        scoped[scale] = dict(arms=arms,pairs=pairs)
    return dict(report,scales=scoped)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--windows",type=Path,nargs=3,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args = cli.parse_args()
    for name in T.THREADS:
        os.environ[name] = "1"
    report = summarize([verify_window(p.resolve()) for p in args.windows])
    T.save(args.output,report)
    print(json.dumps(dict(status=report["status"],windows=[r["status"] for r in report["windows"]],solves_run=0)))
