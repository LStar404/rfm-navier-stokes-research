"""Separate serial GPU trajectory capture from independent parallel CPU audit."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import rank_fair_cache_inventory as inventory
import rank_fair_cache_worker as gpu_worker
from rank_fair_owned_transport import key, read, require


HERE,WORK = inventory.HERE,inventory.WORK
original = inventory.base.original
shared,T = original.shared,original.timing.prior
SOURCE = Path(__file__).resolve()
AUDIT_CPUS = (10,12,14,16)
MAX_FOLLOW_SECONDS = 2800
THIN_API = HERE/"cuda_private_scope_solver.py"
CLOSURE = HERE/"cuda_private_scope_solver_closure_audit.json"


def build_registered():
    draft = inventory.draft()
    evidence = read(CLOSURE)
    require(evidence["status"] == "passed" and evidence["thin_source_sha256"] == shared.sha(THIN_API) and
            evidence["context_mode"] == inventory.RUNTIME and evidence["driver_api"] == 13020 and
            evidence["artifact_manifest_sha256"] == draft["ptx_manifest"]["sha256"] and evidence["gpu_contexts_created"] == 0,
            "Thin private adapter closure was not independently verified")
    fixture = read(HERE/"rank_fair_cache_qualification_fixture.json")
    require(fixture["status"] == "passed" and fixture["gpu_contexts_created"] == fixture["flow_solves_run"] == 0 and
            fixture["qualification_source_sha256"] == shared.sha(SOURCE) and
            fixture["worker_source_sha256"] == shared.sha(Path(gpu_worker.__file__)) and all(fixture["expected_rejections"].values()),
            "The exact qualification source lacks its CPU-only fixture")
    deps = dict(draft["sources_sha256"])
    original.merge(deps,evidence["local_dependency_sha256"])
    originals = (SOURCE,CLOSURE,THIN_API,Path(gpu_worker.__file__).resolve(),HERE/"freeze_rank_fair_timing.py",
                 HERE/"verify_cuda_private_scope_solver.py",HERE/"rank_fair_cache_qualification_fixture.json",
                 HERE/"verify_rank_fair_cache_qualification_fixture.py")
    original.bind(deps,*(shared.record(path) for path in originals))
    driver = Path(os.environ["SystemRoot"])/"System32"/"nvcuda.dll"
    return dict(draft,status="frozen",exact_runtime_adapter_selected=True,runtime_id=inventory.RUNTIME,
                closure=shared.record(CLOSURE),sources_sha256=deps,nvcuda_sha256=shared.sha(driver))


def verify_registered_plan(plan):
    require(plan["status"] == "frozen" and plan["exact_runtime_adapter_selected"] and
            plan["runtime_id"] == inventory.RUNTIME and plan["cpu_auditor_affinities"] == list(AUDIT_CPUS) and len(plan["slots"]) == 78,
            "A reviewed 78-slot exact-runtime qualification freeze is required")
    require(plan == build_registered(),"The registered runtime diverges from its 39 declared two-policy numerical identities")
    require(all(shared.sha(WORK/name) == digest for name,digest in plan["sources_sha256"].items()),"Frozen source or fixed control differs")


def cache_evidence(output,job,result):
    metadata = result["metadata"]
    policy = job["policy"]
    first = policy == "empty_first"
    require(metadata["ptx_cache_policy"] == policy and metadata["ptx_cache_observed_mode"] == ("miss" if first else "hit") and
            metadata["ptx_cache_key"] == job["ptx_cache_key"] and metadata["ptx_artifact_sha256"] == job["ptx_artifact"]["sha256"] and
            metadata["ptx_sha256"] == job["ptx_sha256"] and metadata["cuda_context_mode"] == inventory.RUNTIME and
            metadata["cuda_driver_api"] == 13020 and metadata["private_context_finally_unbound"] is True and
            metadata["nonlinear_engine"] == "warp_cufft" and metadata["observation_reduction"] == "warp" and
            0 <= metadata["ptx_nvrtc_seconds"] <= metadata["seconds"] and (metadata["ptx_nvrtc_seconds"] > 0) == first,
            "The cache observation, no-fallback rule, PTX bytes or private context differs")
    namespace = output/"explicit_cache"
    if first:
        path = namespace/(job["ptx_cache_key"]+".rfmptx")
        return shared.record(path,job["ptx_artifact"]["sha256"])
    require(not namespace.exists(),"Strict-hit unexpectedly wrote a job-local explicit cache")
    return None


def verify_phases(path,index,watchdog):
    phases = [json.loads(line) for line in path.read_bytes().splitlines()]
    require([p["event"] for p in phases] == ["python_ready","solver_enter","solver_return_after_context_cleanup"] and
            all(p["index"] == index and key(p["identity"]) == key(watchdog["identity"]) for p in phases) and
            [p["qpc_ns"] for p in phases] == sorted({p["qpc_ns"] for p in phases}),"Worker phase or ownership evidence differs")


def capture(protocol,output):
    raw = protocol.read_bytes()
    plan = json.loads(raw)
    verify_registered_plan(plan)
    output.mkdir()
    with (output/"protocol.json").open("xb") as file:
        file.write(raw)
    digest = shared.sha(output/"protocol.json")
    shared.pin(6)
    census = shared.WindowsCensus()
    identity = census.times(census.current_process())
    for name in shared.THREADS:
        os.environ[name] = "1"
    rows = []
    for i,job in enumerate(plan["slots"]):
        slot = output/f"slot_{i:03d}"
        # The shared watchdog owns the Popen handle. Environment is restored before the next slot.
        previous = os.environ.get(gpu_worker.OWNER_TOKEN)
        os.environ[gpu_worker.OWNER_TOKEN] = f"{digest}:{identity['pid']}:{identity['birth_filetime_100ns']}:{i}"
        try:
            command = [sys.executable,"-B",str(Path(gpu_worker.__file__).resolve()),str(output),str(i)]
            watchdog = shared.run_owned(command,slot,deadline_seconds=plan["hard_deadline_seconds"])
        finally:
            if previous is None:
                os.environ.pop(gpu_worker.OWNER_TOKEN,None)
            else:
                os.environ[gpu_worker.OWNER_TOKEN] = previous
        T.save(slot/"watchdog.json",watchdog)
        row = dict(status="failed",index=i,job=job,protocol_sha256=digest,watchdog=shared.record(slot/"watchdog.json"))
        phase_path = slot/"phases.jsonl"
        if phase_path.is_file():
            row["phases"] = shared.record(phase_path)
        if watchdog["status"] == "complete":
            try:
                result = read(slot/"result.json")
                require(result["job"] == job and result["index"] == i and result["protocol_sha256"] == digest and
                        result["affinity_mask"] == 16 and key(result["identity"]) == key(watchdog["identity"]) == key(watchdog["terminal_identity"]),
                        "Captured solution is not from the exact registered child")
                verify_phases(phase_path,i,watchdog)
                cold = cache_evidence(slot,job,result)
                row.update(status="complete",result=shared.record(slot/"result.json"),
                           artifact=shared.record(slot/"trajectory.npz",result["trajectory_sha256"]),fresh_artifact=cold)
            except Exception as error:
                row.update(failure_type=type(error).__name__)
        T.save(slot/"capture.json",row)
        rows.append(shared.record(slot/"capture.json"))
        print("capture",i,job["name"],job["seed"],job["arm"],job["policy"],row["status"],flush=True)
        if row["status"] != "complete":
            break
    verify_registered_plan(plan)
    require(protocol.read_bytes() == raw == (output/"protocol.json").read_bytes(),"GPU capture protocol changed")
    report = dict(status="complete" if len(rows) == len(plan["slots"]) and row["status"] == "complete" else "stopped",
        protocol_sha256=digest,rows=rows,unattempted=list(range(len(rows),len(plan["slots"]))),speed_claim_supported=False)
    T.save(output/"capture_summary.json",report)
    return report


def wait_for_capture(path,summary,index,deadline):
    while time.monotonic() < deadline:
        try:
            return "committed",read(path)
        except (FileNotFoundError,PermissionError):
            try:
                if index in read(summary)["unattempted"]:
                    return "unattempted",None
            except (FileNotFoundError,PermissionError):
                pass
            time.sleep(.25)
    return "capture_wait_timeout",None


def audit_lane(output,lane):
    require(lane in range(len(AUDIT_CPUS)),"Unknown offline audit lane")
    for name in shared.THREADS:
        os.environ[name] = "1"
    shared.pin(AUDIT_CPUS[lane])
    protocol = output/"protocol.json"
    plan = read(protocol)
    verify_registered_plan(plan)
    digest = shared.sha(protocol)
    evaluate,_ = shared.assessment(output,plan)
    deadline = time.monotonic()+MAX_FOLLOW_SECONDS
    checked,missing = [],[]
    for i in range(lane,len(plan["slots"]),len(AUDIT_CPUS)):
        job = plan["slots"][i]
        slot = output/f"slot_{i:03d}"
        evidence = slot/"capture.json"
        status,capture_row = wait_for_capture(evidence,output/"capture_summary.json",i,deadline)
        if status != "committed":
            missing.append(dict(index=i,status=status))
            continue
        row = dict(index=i,job=job,qualified=False,protocol_sha256=digest,capture=shared.record(evidence))
        try:
            require(capture_row["status"] == "complete" and capture_row["index"] == i and capture_row["job"] == job and
                    capture_row["protocol_sha256"] == digest,"GPU capture did not complete or changed")
            watchdog = read(shared.bound(capture_row["watchdog"]))
            result = read(shared.bound(capture_row["result"]))
            raw = shared.bound(capture_row["artifact"])
            require(watchdog["status"] == "complete" and watchdog["exit_code"] == 0 and raw == slot/"trajectory.npz" and
                    key(watchdog["identity"]) == key(watchdog["terminal_identity"]) == key(result["identity"]) and
                    result["protocol_sha256"] == digest and result["job"] == job and result["index"] == i,
                    "GPU raw artifact or exact terminal evidence differs")
            verify_phases(shared.bound(capture_row["phases"]),i,watchdog)
            require(cache_evidence(slot,job,result) == capture_row["fresh_artifact"],"Serialized PTX policy evidence changed")
            row.update(evaluate(i,result))
            if job["serial_control"] is None:
                row["observer_attribution"] = dict(status="no_matched_serial_trajectory",strict_comparison_run=False)
            else:
                row["observer_attribution"] = dict(status="checked_matched_serial",strict_comparison_run=True,
                                                    result=original.strict_observer(raw,job,plan))
        except Exception as error:
            row.update(qualified=False,failure_type=type(error).__name__)
        T.save(slot/"independent_score.json",row)
        checked.append(shared.record(slot/"independent_score.json"))
    T.save(output/f"audit_lane_{lane}.json",dict(status="complete",lane=lane,cpu=AUDIT_CPUS[lane],protocol_sha256=digest,
        rows=checked,missing=missing,flow_solves_run=0))


def audit(output):
    plan = read(output/"protocol.json")
    verify_registered_plan(plan)

    def run(lane):
        command = [sys.executable,"-B",str(SOURCE),"--audit-lane",str(output),str(lane)]
        log = output/f"audit_lane_{lane}.log"
        with log.open("xb") as handle:
            try:
                child = subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,timeout=MAX_FOLLOW_SECONDS+180,check=False,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                return dict(lane=lane,status="complete" if child.returncode == 0 else "failed",exit_code=child.returncode)
            except subprocess.TimeoutExpired:
                return dict(lane=lane,status="timeout")

    with ThreadPoolExecutor(max_workers=len(AUDIT_CPUS)) as pool:
        worker_status = list(pool.map(run,range(len(AUDIT_CPUS))))
    deadline = time.monotonic()+60
    while not (output/"capture_summary.json").is_file() and time.monotonic() < deadline:
        time.sleep(.25)
    return verify_audit(output,worker_status)


def verify_audit(output,worker_status):
    from freeze_rank_fair_timing import REFERENCES,validate_table
    import numpy as np

    protocol = output/"protocol.json"
    plan = read(protocol)
    verify_registered_plan(plan)
    digest = shared.sha(protocol)
    rows,missing,lane_records,pair_sources = {},[],[],{}
    for lane,cpu in enumerate(AUDIT_CPUS):
        file = output/f"audit_lane_{lane}.json"
        if not file.is_file():
            continue
        value = read(file)
        require(value["protocol_sha256"] == digest and value["lane"] == lane and value["cpu"] == cpu and value["flow_solves_run"] == 0,
                "Offline audit process used the wrong scope")
        lane_records.append(shared.record(file))
        missing.extend(value["missing"])
        for ref in value["rows"]:
            row = read(shared.bound(ref))
            i = row["index"]
            require(i not in rows and i%len(AUDIT_CPUS) == lane and row["job"] == plan["slots"][i] and row["protocol_sha256"] == digest,
                    "Offline scores do not cover the exact declared identities")
            cap = read(shared.bound(row["capture"]))
            require(cap["index"] == i and cap["job"] == row["job"] and cap["protocol_sha256"] == digest,"Audited capture differs")
            if row["qualified"]:
                require(cap["status"] == "complete" and set(row["physical_worst"]) == REFERENCES and
                        row["wake"]["sustained_alternating_probe_signs"] and row["reference_budget"]["budget"] == .00125 and
                        set(row["tables"]) == REFERENCES and set(row["probes"]) == set(shared.HISTORY) and
                        all(0 <= x <= plan["implementation_gate"] for x in row["probes"].values()),"Numerical gate differs")
                observer = row["observer_attribution"]
                if row["job"]["serial_control"] is None:
                    require(observer == {"status":"no_matched_serial_trajectory","strict_comparison_run":False},
                            "A strict observer comparison was attributed without a matched trajectory")
                else:
                    strict = observer["result"]
                    require(observer["status"] == "checked_matched_serial" and observer["strict_comparison_run"] and
                            strict["saved_states_bitwise"] and strict["common_times"] >= 19 and set(strict["probes"]) == set(shared.HISTORY) and
                            all(set(v) == {"relative","scaled_absolute"} and all(0 <= x <= 1e-12 for x in v.values()) for v in strict["probes"].values()),
                            "The archived strict observer comparison differs")
                validate_table(row["implementation_rows"],row["job"]["args"]["snapshots"],row["implementation_worst"],1e-5)
                for name,table in row["tables"].items():
                    validate_table(table["direct"],plan["physical_times"],row["physical_worst"][name],.005)
                    if row["job"]["arm"] == "combined":
                        validate_table(table["guard"],plan["physical_times"],row["guard_worst"][name],.005)
                    else:
                        require(not table["guard"],"Uniform or FD row claims a mixed-step triangle guard")
                pair_sources.setdefault(row["job"]["identity_index"],{})[row["job"]["policy"]] = (cap,row["job"])
            rows[i] = dict(index=i,qualified=row["qualified"],score=ref,source_capture=row["capture"])
    capture_path = output/"capture_summary.json"
    cap_summary = read(capture_path) if capture_path.is_file() else None
    pairs = []
    for identity in range(39):
        policies = pair_sources.get(identity,{})
        row = dict(identity_index=identity,passed=False)
        if set(policies) == set(inventory.POLICIES):
            left,right = (policies[policy] for policy in inventory.POLICIES)
            la,ra = (shared.bound(pair[0]["artifact"]) for pair in (left,right))
            lm,rm = (read(shared.bound(pair[0]["result"])) for pair in (left,right))
            stable = ("steps","nonlinear_evaluations","logical_fft_groups","logical_field_transforms_all_radial_rows",
                      "inverse_field_transforms","forward_field_transforms","fft_radial_lines","cufft_marching_calls",
                      "priming_logical_field_transforms","graph_steps","ptx_sha256","ptx_artifact_sha256","cuda_context_mode")
            with np.load(la,allow_pickle=False) as a,np.load(ra,allow_pickle=False) as b:
                arrays = {k:a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes()
                          for k in ("states","times","probe_times","transverse","vorticity")}
            match = lm["spatial_sha256"] == rm["spatial_sha256"] and all(lm["metadata"][k] == rm["metadata"][k] for k in stable)
            row.update(passed=bool(match and all(arrays.values())),arrays_bitwise=arrays,spatial_work_and_ptx_equal=match,
                       name=left[1]["name"],seed=left[1]["seed"],arm=left[1]["arm"])
        pairs.append(row)
    all_completed = bool(cap_summary and cap_summary["status"] == "complete" and not cap_summary["unattempted"] and
                         len(cap_summary["rows"]) == len(rows) == len(plan["slots"]) and cap_summary["protocol_sha256"] == digest and
                         all(cap_summary["rows"][i] == rows[i]["source_capture"] and rows[i]["qualified"] for i in range(len(rows))) and
                         len(lane_records) == len(AUDIT_CPUS) and not missing and all(r["status"] == "complete" for r in worker_status) and
                         all(r["passed"] for r in pairs))
    report = dict(status="passed" if all_completed else "incomplete_or_failed",protocol_sha256=digest,rows=[rows[i] for i in sorted(rows)],
                  lanes=lane_records,worker_status=worker_status,missing=missing,policy_pairs=pairs,flow_solves_run_by_audit=0,speed_claim_supported=False,
                  capture_summary=shared.record(capture_path) if cap_summary else None)
    T.save(output/"qualification_summary.json",report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture",type=Path)
    group.add_argument("--register",type=Path)
    group.add_argument("--audit",type=Path)
    group.add_argument("--audit-lane",nargs=2,help=argparse.SUPPRESS)
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    if args.capture:
        if args.output is None:
            cli.error("Capture requires a new output directory")
        result = capture(args.capture.resolve(),args.output.resolve())
        raise SystemExit(0 if result["status"] == "complete" else 2)
    elif args.register:
        plan = build_registered()
        verify_registered_plan(plan)
        T.save(args.register.resolve(),plan)
        print(json.dumps(dict(status="frozen",rows=len(plan["slots"]),source_sha256=shared.sha(args.register))))
    elif args.audit:
        result = audit(args.audit.resolve())
        print(json.dumps(dict(status=result["status"],audited=len(result["rows"]),flow_solves_run_by_audit=0)))
        raise SystemExit(0 if result["status"] == "passed" else 2)
    else:
        audit_lane(Path(args.audit_lane[0]).resolve(),int(args.audit_lane[1]))
