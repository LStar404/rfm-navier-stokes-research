"""Bind unchanged numerical/paired gates to strict v2 transport and outer ownership."""
import argparse
from pathlib import Path
import sys
from types import FunctionType

sys.path.append(str(Path(__file__).resolve().parent.parent))
import verify_rank_fair_cache_formal as old
import binding_v2 as frozen
import verify_transport_v2 as transport
from transport_v2 import read, require, key

T = frozen.T


def verify_outer(output, plan, report):
    guard = read(output / "execution_environment_guard.json")
    outer = frozen.WORK / guard["outer_evidence_directory"]
    ready = read(outer / "owner_ready.json")
    terminal = read(outer / "watchdog_terminal.json")
    owner = read(outer / "owner_terminal.json")
    inputs = read(outer / "watchdog_input_guard.json")
    opening = read(outer / "watchdog_open.json")
    request_path = Path(opening["request_path"])
    request = read(request_path)
    ledger = read(output / "identity_ledger.json")
    require(T.digest(outer / "owner_ready.json") == guard["outer_owner_ready_sha256"] and
            key(ready["coordinator_identity"]) == key(ledger["coordinator"]) == key(owner["coordinator_terminal"]) and
            ready["assigned_before_resume"] and ready["job_kill_on_close"] and not ready["job_handle_inherited"], "Outer exact Job ownership differs")
    require(terminal["status"] == "owned_tree_terminal" and terminal["reason"] == "owner_exited" and terminal["termination_confirmed"] and terminal["owner_exited"] and
            terminal["closing_input_guard_passed"] and terminal["owner_exit_code"] == 0 and terminal["owner_terminal"] == owner and
            owner["status"] == "job_empty" and owner["reason"] == "coordinator_exited" and owner["job_empty_confirmed"] and
            owner["coordinator_exit_code"] == 0 and inputs["status"] == "passed" and T.digest(outer / "watchdog_input_guard.json") == terminal["input_guard_sha256"],
            "Outer tree did not close normally with source guards")
    require(T.digest(request_path) == terminal["request_sha256"] == opening["request_sha256"] == inputs["request_sha256"] and
            Path(request["directory"]).resolve() == outer.resolve() and Path(request["slot_state_path"]).resolve() == (output / "outer_slot_state.json").resolve() and
            request["run_token"] == plan["run_token"] and request["slot_count"] == len(plan["requests"]) and request["slot_timeout_seconds"] == 45 and
            request["deadline_seconds"] == plan["outer_supervision"]["whole_window_watchdog_seconds"] == 840 and
            request["confirmation_seconds"] == plan["outer_supervision"]["confirmation_seconds"] == 5 and
            request["observer_cpu"] == 10 and request["owner_cpu"] == 12 and opening["observer_affinity_mask"] == 1024 and ready["owner_affinity_mask"] == 4096,
            "Outer registered slot deadline or CPU assignment changed")
    # The external request binds the exact protocol and implementation. Retain
    # absolute path keys only in local evidence, not in the public report.
    for name, expected in plan["source_hashes"].items():
        path = str((frozen.WORK / name).resolve())
        require(inputs["inputs_sha256"].get(path) == expected, "Outer input guard omitted a frozen input")
    require(inputs["inputs_sha256"].get(str((output / "protocol.json").resolve())) == T.digest(output / "protocol.json") or
            any(v == T.digest(output / "protocol.json") for v in inputs["inputs_sha256"].values()), "Outer protocol binding missing")
    return dict(verified=True, termination_confirmed=True, normal_coordinator_exit=True,
                terminal_sha256=T.digest(outer / "watchdog_terminal.json"), owner_terminal_sha256=T.digest(outer / "owner_terminal.json"))


def verify_window(output):
    f = old.verify_window
    bound = FunctionType(f.__code__, dict(f.__globals__, frozen=frozen, transport=transport), f.__name__, f.__defaults__, f.__closure__)
    plan, report = bound(output)
    try:
        outer = verify_outer(output, plan, report)
    except Exception as error:
        outer = dict(verified=False, failure_type=type(error).__name__)
        report.update(status="diagnostic_only", eligible=False)
    report["outer_supervision"] = outer
    report["clock_revision"] = frozen.FRESH_BOUNDARY
    return plan, report


def summarize(reports):
    f = old.summarize
    bound = FunctionType(f.__code__, dict(f.__globals__, frozen=frozen), f.__name__, f.__defaults__, f.__closure__)
    result = bound(reports)
    result["verifier_sha256"] = T.digest(Path(__file__))
    result["clock_revision"] = frozen.FRESH_BOUNDARY
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize([verify_window(p.resolve()) for p in args.windows])
    T.save(args.output, report)
