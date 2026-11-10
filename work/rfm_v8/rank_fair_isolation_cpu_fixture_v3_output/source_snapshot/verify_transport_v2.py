"""Unchanged strict clock/exit checks plus independent v2 affinity attestation."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import verify_rank_fair_timing as old
from transport_v2 import SECOND, key, latch, read, require, require_identity
from worker_bootstrap_v2 import BOOTSTRAP_SCHEMA
import coordinator_v2 as timing
T, legacy = old.T, old.legacy


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
    bootstrap_path = output/f"affinity_bootstrap_{i:03d}.json"
    proof = read(bootstrap_path)
    proof_sha = T.digest(bootstrap_path)
    require(all(v["affinity_bootstrap_sha256"] == proof_sha for v in values.values()), "Affinity proof is not bound by every latch")
    require(proof["schema"] == BOOTSTRAP_SCHEMA and proof["index"] == i and proof["run_token"] == request["run_token"] and
            proof["request_sha256"] == row["request_sha256"] and key(proof["identity"]) == key(row["child_identity"]) and
            key(proof["parent_identity"]) == key(ledger["coordinator"]) and proof["inherited_affinity_mask"] == 256 and
            proof["effective_affinity_mask"] == 16 and not proof["scientific_modules_before_pin"] and not proof["scientific_modules_after_pin"] and
            launch <= proof["admission_entered_ns"] <= proof["effective_pin_completed_ns"] <= proof["before_solver_import_ns"] <= ready_value["published_monotonic_ns"] and
            proof["thread_policy"] == {n:"1" for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
            "CPU4 was not established before scientific imports")
    active = read(output/f"outer_slot_active_{i:03d}.json")
    terminal = read(output/f"outer_slot_terminal_{i:03d}.json")
    require(active == dict(schema="rfm-v8-outer-slot-1",run_token=plan["run_token"],index=i,phase="active",
                           launch_monotonic_ns=launch,hard_deadline_ns=hard,coordinator=ledger["coordinator"]) and
            terminal == dict(active,phase="terminal",child_identity=row["child_identity"],terminal_identity=row["terminal_identity"],
                             exit_monotonic_ns=exited,exit_code=0), "Independent active-slot deadline or exact terminal differs")
    return legacy.checked_file(output,row["artifact"],row["trajectory_sha256"])

