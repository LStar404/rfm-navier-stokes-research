"""Build CPU-only recovery design evidence, without scheduling any GPU work."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORK = HERE.parent


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def record(path):
    return dict(path=path.relative_to(WORK).as_posix(), sha256=digest(path))


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def design():
    original = HERE/"rank_fair_cache_formal_protocols"
    names = ("mid_b1_b5", "high_b1_b3", "high_b4_b5")
    plans = [read(original/(name+".json")) for name in names]
    qualification = HERE/"rank_fair_cache_qualification_output"
    qp = read(qualification/"protocol.json")
    audit_path = HERE/"rank_fair_cache_recovery_cpu_audit.json"
    audit = read(audit_path)
    watchdog = HERE/"rank_fair_cache_recovery_watchdog_fixture_v5_output"
    certificate = read(watchdog/"certificate.json")
    assert audit["all_three_original_protocols_verify_live"] and certificate["status"] == "passed"
    assert [len(p["requests"]) for p in plans] == [50, 66, 44]
    scientific_fields = ("name", "arm", "seed", "policy", "block", "qualification_index", "family", "engine", "config",
                         "ptx_artifact", "ptx_cache_key", "ptx_sha256", "runtime_id", "imports", "solve_args", "compiler_identity")
    windows = []
    for plan in plans:
        requests = [{k:r[k] for k in scientific_fields} for r in plan["requests"]]
        windows.append(dict(name=plan["window"], slots=len(requests), original_protocol=record(original/(plan["window"]+".json")),
                            unchanged_ordered_scientific_requests=requests,
                            ordered_scientific_request_sha256=canonical(requests), window_spec=plan["window_spec"]))
    old = HERE/"rank_fair_cache_formal_mid_output"
    raw_inventory = {p.relative_to(old).as_posix():digest(p) for p in sorted(old.rglob("*")) if p.is_file()}
    original_bindings = dict(plans[0]["source_hashes"])
    assert all(p["source_hashes"] == original_bindings for p in plans)
    original_changed = [name for name,wanted in original_bindings.items() if digest(WORK/name) != wanted]
    assert not original_changed
    return dict(schema="rfm-v8-recovery-design-1", status="cpu_prepared_not_formal_execution_authorized",
        created_utc=datetime.now(timezone.utc).isoformat(), flow_solves_run=0, gpu_contexts_created=0,
        prior_attempt=dict(disposition="incomplete_and_timeout_not_provably_closed", output=old.relative_to(WORK).as_posix(),
            counts=dict(completed=2, started_without_return=1, not_attempted=47),
            absent=["summary.json", "execution_environment_guard.json", "isolation.json"],
            root_cause="unknown", full_file_inventory_sha256=raw_inventory,
            independent_interruption_review=record(HERE/"formal_interruption_review_20261110/formal_interruption_review.json"),
            admissible_for_new_formal_pool=False, raw_rewritten=False),
        numerical_qualification=dict(protocol=record(qualification/"protocol.json"), summary=record(qualification/"qualification_summary.json"),
            exact_slots=78, policy_pairs=39, total_probe_times=577, horizon=qp["horizon"],
            runtime_id=qp["runtime_id"], implementation_gate=qp["implementation_gate"], physical_gate=qp["physical_gate"],
            serial_probe_gate=qp["serial_probe_relative_and_scaled_absolute_gate"],
            numerical_api=record(HERE/"cuda_private_scope_solver.py"), all_bound_files_currently_match=True),
        prospective_cohort=dict(windows=windows, total_slots=160, reuse_old_complete_rows=False,
            all_three_new_windows_required=True, no_replacement_of_failed_slots=True, chronological_nonoverlap_required=True,
            comparison_thresholds_unchanged=True, maximum_active_seconds=720, monitor_poll_seconds=1,
            max_monitor_gap_seconds=2.5, budget_interval_seconds=10, max_budget_lateness_seconds=2,
            ready_timeout_seconds=plans[0]["ready_timeout_seconds"], return_timeout_seconds=plans[0]["return_timeout_seconds"],
            exit_timeout_seconds=plans[0]["exit_timeout_seconds"], whole_slot_timeout_seconds=45,
            coordinator_cpu=8, monitor_cpu=6, worker_cpu=4, outer_observer_cpu=10, job_owner_cpu=12,
            worker_affinity_semantics="New schema must distinguish inherited CPU8 mask256 from explicit effective CPU4 mask16 before scientific imports.",
            clock_semantics="Worker bootstrap and explicit affinity overhead stay in fresh-process-return clock; numerical solver clock boundaries remain unchanged.",
            known_background=[], unknown_python_policy="Original preflight rejects any unknown Python identity. Active windows retain original 0.2 CPU-second per 10 seconds threshold.",
            no_inference_that_idle_unknown_python_is_whitelisted=True),
        environment=dict(cpu_audit=record(audit_path), frozen_current_environment=audit["current_environment"],
            new_package_tree_sha256={name:value["tree_sha256"] for name,value in audit["current_package_byte_identities"].items()},
            package_hash_scope_note=audit["package_hash_scope_note"]),
        outer_watchdog=dict(source=record(HERE/"rank_fair_outer_watchdog.cs"), executable=record(watchdog/"rank_fair_outer_watchdog.exe"),
            fixture=record(watchdog/"certificate.json"), fixture_cases=len(certificate["cases"]),
            owner_launch="Native CreateProcess with inheritance disabled; coordinator atomically created in owned Job suspended then resumed.",
            observer_holds_job_handle=False, termination_apis_in_observer=False,
            whole_window_observer_seconds=840, termination_confirmation_seconds=4,
            active_slot_state_schema="rfm-v8-outer-slot-1", active_slot_cap_seconds=45,
            state_authentication=["run_token", "slot index", "exact coordinator PID and birth", "immutable launch/deadline QPC", "monotonic index/phase"],
            deadline_evidence_persisted_before_termination_request=True, pending_is_immutable=True,
            terminal_observation_does_not_prove_all_slot_terminals=True,
            no_next_gpu_work_after_any_pending_or_failed_window=True,
            success_requires=["watchdog reason owner_exited", "termination_confirmed true", "coordinator_exit_code zero",
                "closing_input_guard_passed true", "all expected per-slot raw/terminal evidence", "closing runtime/environment guard passed",
                "independent numerical and isolation verification passed", "three complete chronological nonoverlapping windows"],
            limitation=certificate["limitation"]),
        required_before_any_formal_execution=["Independently review new isolation_v2 source and CPU8/4 saturation fixture",
            "Freeze complete new source snapshot and same scientific requests into distinct attempt protocols",
            "Bind every new protocol, its original source/qualification evidence, source snapshots, executable and fixtures in outer input guards",
            "Resolve quiet preflight by observation; unknown Python remains unwhitelisted",
            "Root explicitly authorizes bounded GPU diagnostic, then evaluates whether formal execution is warranted"],
        original_bound_source_hashes=original_bindings, manifest_generator=record(Path(__file__).resolve()))


if __name__ == "__main__":
    result = design()
    target = HERE/"rank_fair_cache_recovery_design.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    print(json.dumps(dict(status=result["status"], planned_slots=160, gpu_contexts_created=0, flow_solves_run=0,
                         manifest_sha256=digest(target))))
