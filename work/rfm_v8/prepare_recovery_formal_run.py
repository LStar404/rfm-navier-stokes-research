"""Freeze a complete fresh three-window attempt only after CPU and T9 entry proof."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"rank_fair_isolation_v2"))
import binding_v2 as binding
import verifier_v2 as verifier
from transport_v2 import read, require

T, WORK = binding.T, binding.WORK
WATCHDOG = HERE/"rank_fair_cache_recovery_watchdog_fixture_v6_output"


def prepare(root, cpu, t9):
    require(not root.exists(), "A recovery attempt must have a new output namespace")
    root.mkdir()
    readiness = dict(schema=binding.READINESS_SCHEMA, status="passed", sources_sha256=binding.source_hashes(),
        cpu_fixture=binding.qual.shared.record(cpu), t9_equivalence=binding.qual.shared.record(t9),
        outer_watchdog_exe=binding.qual.shared.record(WATCHDOG/"rank_fair_outer_watchdog.exe"),
        outer_watchdog_source=binding.qual.shared.record(HERE/"rank_fair_outer_watchdog.cs"),
        outer_cpu_fixture=binding.qual.shared.record(WATCHDOG/"certificate.json"),
        generator=binding.qual.shared.record(Path(__file__).resolve()), old_partial_rows_reused=False)
    certificate = read(WATCHDOG/"certificate.json")
    require(certificate["status"] == "passed" and certificate["source_sha256"] == readiness["outer_watchdog_source"]["sha256"] and
            certificate["executable_sha256"] == readiness["outer_watchdog_exe"]["sha256"], "Watchdog fixture no longer matches")
    T.save(root/"readiness.json", readiness)
    ready_record = binding.qual.shared.record(root/"readiness.json")
    binding.checked_readiness(ready_record)
    background = binding.qual.shared.record(HERE/"rank_fair_cache_recovery_identified_background.json")
    environment = binding.old.environment()
    plans = [binding.make_window(name, HERE/"rank_fair_cache_qualification_output", ready_record, environment, background)
             for name in binding.WINDOWS]
    require([len(p["requests"]) for p in plans] == [50, 66, 44] and all(binding.evidence(p) == binding.evidence(plans[0]) for p in plans),
            "The complete fixed 160-slot scope differs")
    windows = []
    for plan in plans:
        folder = root/plan["window"]
        folder.mkdir()
        protocol, output, outer = folder/"protocol.json", folder/"capture", folder/"outer"
        T.save(protocol, plan)
        command = [sys.executable, "-B", str(binding.HERE/"binding_v2.py"), "--execute", str(protocol),
                   "--output", str(output), "--outer-evidence", str(outer)]
        inputs = {str((WORK/name).resolve()):digest for name,digest in plan["source_hashes"].items()}
        inputs[str(protocol)] = T.digest(protocol)
        request = dict(directory=str(outer), application=sys.executable, command_line=subprocess.list2cmdline(command), working_directory=str(HERE),
            observer_cpu=10, owner_cpu=12, deadline_seconds=840, confirmation_seconds=5, inputs_sha256=inputs,
            slot_state_path=str(output/"outer_slot_state.json"), run_token=plan["run_token"], slot_count=len(plan["requests"]), slot_timeout_seconds=45)
        T.save(folder/"watchdog_request.json", request)
        windows.append(dict(window=plan["window"], slots=len(plan["requests"]), protocol=binding.qual.shared.record(protocol),
            request=binding.qual.shared.record(folder/"watchdog_request.json"),
            launch_argv=[str(WATCHDOG/"rank_fair_outer_watchdog.exe"), "--observe", str(folder/"watchdog_request.json")],
            gate_argv=[sys.executable, "-B", str(Path(__file__).resolve()), "--validate-window", str(folder)]))
    manifest = dict(status="prepared_not_executed", complete_new_cohort=True, old_partial_rows_reused=False, windows=windows,
        no_automatic_gpu_execution=True, stop_after_any_failed_or_pending_window=True,
        required_window_gates=["outer normal owner/coordinator exit", "termination confirmed", "closing source/runtime guards",
            "every exact worker terminal and affinity proof", "all numerical fields/probes bitwise against exact qualification", "independent isolation"],
        final_audit_argv=[sys.executable, "-B", str(binding.HERE/"verifier_v2.py"), "--windows",
            *(str(root/name/"capture") for name in binding.WINDOWS), "--output", str(root/"formal_verification.json")])
    T.save(root/"attempt_manifest.json", manifest)
    return dict(status=manifest["status"], total_slots=160, windows=3)


def gate(folder):
    _, result = verifier.verify_window(folder/"capture")
    require(result["eligible"] and result["complete"] and result["status"] == "verified" and result["outer_supervision"]["verified"],
            "Window is ineligible: do not start another GPU window")
    T.save(folder/"independent_window_acceptance.json", result)
    return dict(status="accepted", window=result["window"], next_window_only_if_registered_order=True)


if __name__ == "__main__":
    import json
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--validate-window", type=Path)
    cli.add_argument("--cpu-certificate", type=Path)
    cli.add_argument("--t9-certificate", type=Path)
    args = cli.parse_args()
    if args.prepare:
        require(args.cpu_certificate and args.t9_certificate, "CPU and actual T9 entry proof both required")
        result = prepare(args.prepare.resolve(), args.cpu_certificate.resolve(), args.t9_certificate.resolve())
    else:
        result = gate(args.validate_window.resolve())
    print(json.dumps(result))
