"""Exercise the actual bounded coordinator without importing a flow solver or CUDA."""

import argparse
from hashlib import sha256
from pathlib import Path
import time

import formal_timing_coordinator as prior
import rank_fair_coordinator as coordinator
from rank_fair_owned_transport import SECOND, TERMINATION_CODE, key, latch, publish, read, require
from rank_fair_solver_worker import SCHEMA, admission, receive_control


HERE,WORK = coordinator.HERE,coordinator.WORK
SOURCE = Path(__file__).resolve()


def fake(path, digest):
    request,target,identity,parent,marker = admission(path,digest)
    output,index = target.parent,request["index"]
    mode = request["synthetic_mode"]
    if mode == "before_ready":
        time.sleep(30)
    publish(latch(output,index,"READY"),dict(marker,event="READY",published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target,"GO",request,identity,parent,marker)
    if mode == "before_return":
        with latch(output,index,"RETURN").with_suffix(".pending").open("xb") as pending:
            pending.write(b"incomplete synthetic return\n")
            pending.flush()
            time.sleep(30)
    started = time.perf_counter_ns()
    time.sleep(.035)
    solved = time.perf_counter_ns()
    before = time.perf_counter_ns()
    released = time.perf_counter_ns()
    core,release = (solved-started)/SECOND,(released-before)/SECOND
    publish(latch(output,index,"RETURN"),dict(marker,event="RETURN",status="solved",solver_seconds=core+release,
        solver_core_seconds=core,model_release_seconds=release,spatial_sha256="synthetic_no_flow",
        solve_started_ns=started,solve_completed_ns=solved,model_release_started_ns=before,model_released_ns=released,
        published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target,"SAVE",request,identity,parent,marker)
    temporary = target.with_suffix(".tmp.npz")
    with temporary.open("xb") as artifact:
        artifact.write(b"synthetic transport artifact, not a trajectory\n")
        artifact.flush()
        if mode == "after_save":
            time.sleep(30)
    temporary.rename(target)
    return 0


def fixture(root, mode):
    protocol_path, output = root/f"{mode}_protocol.json",root/mode
    token = sha256(f"no-gpu rank fair transport {mode}".encode()).hexdigest()
    names = prior.REQUIRED_SOURCES|{p.relative_to(WORK).as_posix() for p in (
        SOURCE,HERE/"rank_fair_owned_transport.py",HERE/"rank_fair_coordinator.py",HERE/"rank_fair_solver_worker.py")}
    plan = dict(schema=coordinator.SCHEMA,status="frozen",mode="transport_only",run_token=token,
                coordinator_cpu=4,monitor_cpu=6,poll_interval_ns=SECOND//10,max_poll_gap_ns=SECOND//4,
                maximum_active_ns=8*SECOND,budget_interval_ns=10*SECOND,max_budget_lateness_ns=2*SECOND,
                ready_timeout_seconds=1.2,return_timeout_seconds=1.1,exit_timeout_seconds=1.1,
                slot_hard_timeout_seconds=3.5,known_background=[],source_hashes={name:prior.digest(WORK/name) for name in sorted(names)})
    plan["requests"] = [dict(schema=SCHEMA,index=i,run_token=token,affinity_mask=16,trajectory=f"trajectory_{i:03d}.npz",
                            synthetic_mode=branch) for i,branch in enumerate((mode,"success"))]
    prior.save(protocol_path,plan)
    summary = coordinator.run(protocol_path,output,plan,protocol_path.read_bytes(),worker=SOURCE)
    rows = read(output/"runs.json")
    ledger = read(output/"identity_ledger.json")
    require(not summary["isolated_speed_claim_supported"] and len(rows) == len(ledger["children"]), "Fixture promoted or lost a child")
    for row,entry in zip(rows,ledger["children"],strict=True):
        require(key(row["child_identity"]) == key(row["terminal_identity"]) and row["terminal_identity"] == entry["terminal"],
                "Exact fixture child identity was not retained")
    if mode == "success":
        require(summary["status"] == "transport_complete" and len(rows) == 2 and not summary["unattempted_slots"] and
                rows[0]["exit_monotonic_ns"] <= rows[1]["launch_monotonic_ns"] and
                all(0 < r["solver_seconds"] <= r["go_to_return_seconds"] <= r["fresh_process_return_seconds"] and
                    r["return_received_monotonic_ns"] <= r["save_sent_monotonic_ns"] <= r["exit_monotonic_ns"] for r in rows),
                "Successful dual clock or serial-worker fixture failed")
    else:
        require(summary["status"] == "transport_failed" and len(rows) == 1 and summary["unattempted_slots"] == [1] and
                rows[0]["status"] == "failed" and rows[0]["failure_type"] == "TimeoutError" and
                rows[0]["exit_code"] == TERMINATION_CODE and rows[0]["watchdog"]["terminate_process_ok"] and
                rows[0]["watchdog"]["post_terminate_wait_status"] == 0 and not (output/"trajectory_000.npz").exists(),
                "Stalled worker was not bounded or the following slot was started")
        if mode == "before_return":
            require("return_000.pending" in summary["raw_hashes"],"Partial RETURN evidence was lost")
        if mode == "after_save":
            require("trajectory_000.tmp.npz" in summary["raw_hashes"],"Partial SAVE evidence was lost")
    return dict(mode=mode,status=summary["status"],attempts=len(rows),successful=summary["successful_slots"],
                termination_codes=[r["exit_code"] for r in rows],summary_sha256=prior.digest(output/"summary.json"),
                terminal_ownership_checked=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--worker",nargs=2)
    parser.add_argument("worker_args",nargs="*")
    args = parser.parse_args()
    if args.worker_args:
        raise SystemExit(fake(Path(args.worker_args[0]).resolve(),args.worker_args[1]))
    root = args.output.resolve()
    root.mkdir()
    modes = ("success","before_ready","before_return","after_save")
    records = [fixture(root,mode) for mode in modes]
    prior.save(root/"certificate.json",dict(status="passed",test_cases=records,time_solves_run=0,
        source_sha256=prior.digest(SOURCE)))
