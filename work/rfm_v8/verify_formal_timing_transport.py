"""Run small transport-only cases through the new coordinator and actual worker handshake."""

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import formal_timing_coordinator as timing


HERE = Path(__file__).resolve().parent
FAKE = HERE/"formal_timing_fake_worker.py"
SECOND = timing.SECOND


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_protocol(root, name, slots, cadence, max_seconds=12):
    token = sha256(name.encode("ascii")).hexdigest()
    requests = [dict(index=i, schema="rfm-v8-worker-1", run_token=token, affinity_mask=16,
                     trajectory=f"trajectory_{i:03d}.npz", block=1, name="transport_fixture", seed=17, arm="uniform",
                     family="legendre", imports="minimal", config={}, solve_args=dict(seconds=seconds,fail=fail))
                for i,(seconds,fail) in enumerate(slots)]
    sources = sorted(timing.REQUIRED_SOURCES | {FAKE.relative_to(timing.WORK).as_posix(),
                                               Path(__file__).resolve().relative_to(timing.WORK).as_posix()})
    value = dict(schema=timing.SCHEMA, status="frozen", mode="transport_only", run_token=token,
                 coordinator_cpu=4, monitor_cpu=6, budget_interval_ns=10*SECOND, max_budget_lateness_ns=2*SECOND,
                 poll_interval_ns=cadence, max_poll_gap_ns=5*cadence//2, maximum_active_ns=int(max_seconds*SECOND),
                 ready_timeout_seconds=4, return_timeout_seconds=4, exit_timeout_seconds=2, known_background=[],
                 source_hashes={name:timing.digest(timing.WORK/name) for name in sources}, requests=requests)
    path = root/f"{name}_protocol.json"
    timing.save(path,value)
    return path


def run_case(root, name, slots, cadence, max_seconds=12):
    path = write_protocol(root,name,slots,cadence,max_seconds=max_seconds)
    output = root/name
    summary = timing.run(path,output,fake_worker=FAKE)
    return output,summary


def audit_trace(output, summary, expected_attempts, expected_successes):
    ledger = read(output/"identity_ledger.json")
    journal = read(output/"runs.json")
    isolation = read(output/"isolation.json")
    events = lines(output/"monitor_events.jsonl")
    samples = lines(output/"process_samples.jsonl")
    checks = [(name,timing.digest(output/name)) for name in summary["raw_hashes"]]
    timing.require(len(journal) == len(ledger["children"]) == summary["attempted_slots"] == expected_attempts and
                   summary["successful_slots"] == expected_successes and all(summary["raw_hashes"][name] == digest for name,digest in checks),
                   "Attempt or raw hash mismatch")
    timing.require(summary["isolation_sha256"] == timing.digest(output/"isolation.json") and
                   isolation["process_samples_sha256"] == timing.digest(output/"process_samples.jsonl") and
                   isolation["close_raw_hashes"] == summary["raw_hashes"], "Raw monitor bindings differ")
    monitor_hashes = {name:timing.digest(output/name) for name in
                      ("preflight.json", "process_samples.jsonl", "monitor_events.jsonl", "isolation.json")}
    timing.require(summary["monitor_evidence_sha256"] == monitor_hashes and read(output/"preflight.json") == isolation["preflight"],
                   "Preflight or monitor evidence hashes differ")
    registered = {event["command"]["index"]:event for event in events if event["command"]["event"] == "REGISTER"}
    previous = summary["open_qpc_ns"]
    for row,slot in zip(journal,ledger["children"],strict=True):
        index = row["index"]
        ack = registered[index]["reply"]["ack_qpc_ns"]
        timing.require(row["terminal_identity"] == slot["terminal"] and
                       timing.process_key(row["terminal_identity"]) == timing.process_key(slot["identity"]) and
                       previous <= row["launch_monotonic_ns"] <= ack <= row["go_sent_monotonic_ns"] <= row["exit_monotonic_ns"],
                       "Owned terminal or pre-GO registration chronology differs")
        previous = row["exit_monotonic_ns"]
        if row["status"] == "complete":
            timing.require(0 < row["solver_seconds"] <= row["go_to_return_seconds"] <= row["fresh_process_return_seconds"] and
                           row["return_received_monotonic_ns"] <= row["save_sent_monotonic_ns"] <= row["exit_monotonic_ns"] and
                           (output/row["artifact"]).read_bytes() == b"V8_FORMAL_SYNTHETIC_TRANSPORT_ONLY\n", "Fresh/solver or SAVE boundary differs")
    timing.require(samples[events[1]["reply"]["boundary_sample_index"]]["collection_end_qpc_ns"] <= summary["open_qpc_ns"] and
                   previous <= summary["close_qpc_ns"] <= samples[isolation["end_sample_index"]]["qpc_start_ns"] and
                   not summary["isolated_speed_claim_supported"] and not isolation["active"]["full_lifecycle_verified"],
                   "Active boundaries were not bracketed or transport was promoted")
    return journal,isolation


def audit_overhead(root, cadence):
    name = "cadence_"+str(cadence//1_000_000)+"ms"
    output, summary = run_case(root,name,[(2.4,False)],cadence)
    timing.require(summary["status"] == "transport_complete", "Cadence transport failed")
    _, isolation = audit_trace(output,summary,1,1)
    before, after = (isolation["monitor_cpu"][key] for key in ("opening", "after_post_close_sample"))
    active = (summary["close_qpc_ns"]-summary["open_qpc_ns"])/SECOND
    measured = (isolation["monitor_cpu"]["post_close_cpu_sample_qpc_ns"]-summary["open_qpc_ns"])/SECOND
    cpu = sum(after[key]-before[key] for key in ("kernel_cpu_100ns","user_cpu_100ns"))/10_000_000
    return dict(poll_interval_ns=cadence, monitor_cpu_seconds=cpu, active_seconds=active,
                monitor_cpu_measurement_span_seconds=measured, monitor_cpu_fraction_over_measurement_span=cpu/measured,
                active_samples=isolation["active"].get("active_samples"), actual_max_poll_gap_ns=isolation["active"].get("max_poll_gap_ns"),
                activity_verdict_not_used=True)


def verify(root):
    root.mkdir()
    cadence = SECOND//10
    invalid = root/"mutated_worker_request.json"
    timing.save(invalid,dict(schema="rfm-v8-worker-1"))
    rejected = subprocess.run([sys.executable,"-B",str(FAKE),str(invalid),"0"*64], capture_output=True,
                              timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
    timing.require(rejected.returncode != 0 and not rejected.stdout and b"Request hash mismatch" in rejected.stderr,
                   "Worker did not reject a changed request before READY")
    output, summary = run_case(root,"attempt_preservation",[(.06,False),(.03,True),(.06,False)],cadence)
    journal,_ = audit_trace(output,summary,3,2)
    timing.require(summary["status"] == "transport_failed" and journal[1]["marker"]["status"] == "failed" and
                   not (output/"trajectory_001.npz").exists() and journal[2]["status"] == "complete", "Failed slot was discarded or retried")

    original_save = timing.save
    def corrupt_ledger(path, value, **options):
        if path.name == "identity_ledger.json" and value["children"] and value["children"][-1]["status"] == "monitor_registered":
            value = copy.deepcopy(value)
            value["coordinator"]["birth_filetime_100ns"] += 1
        return original_save(path,value,**options)
    with patch.object(timing,"save",side_effect=corrupt_ledger):
        output,summary = run_case(root,"bad_parent_birth",[(.03,False)],cadence)
    journal,_ = audit_trace(output,summary,1,0)
    timing.require(journal[0]["status"] == "failed" and "ready" in journal[0] and "marker" not in journal[0] and
                   not (output/"trajectory_000.npz").exists(), "Worker accepted a wrong parent birth time")

    output,summary = run_case(root,"deadline",[(3.,False),(.03,False)],cadence,max_seconds=.8)
    journal,_ = audit_trace(output,summary,1,0)
    timing.require(journal[0]["status"] == "failed" and summary["unattempted_slots"] == [1] and
                   not (output/"trajectory_000.npz").exists(), "Timeout did not retain the attempt and unlaunched slot")

    overhead = [audit_overhead(root,value) for value in (SECOND,SECOND//10)]
    sources = sorted(timing.REQUIRED_SOURCES | {FAKE.relative_to(timing.WORK).as_posix(),
                                               Path(__file__).resolve().relative_to(timing.WORK).as_posix()})
    certificate = dict(status="passed", scope="Synthetic transport and cadence cost only, with no fluid or GPU work",
                       source_sha256={name:timing.digest(timing.WORK/name) for name in sources},
                       deliberately_failed_slot_preserved=True, exact_wrong_parent_birth_rejected=True,
                       changed_request_hash_rejected_before_ready=True, preflight_evidence_bound=True,
                       hard_deadline_preserves_unattempted_slots=True, cadence_overhead=overhead,
                       output_summary_sha256={path.parent.name:timing.digest(path) for path in root.glob("*/summary.json")})
    timing.save(root/"certificate.json",certificate)
    print("passed")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    verify(cli.parse_args().output)
