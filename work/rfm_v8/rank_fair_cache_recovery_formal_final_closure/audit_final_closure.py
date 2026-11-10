"""Read-only independent closure; preserve failures and compute no speed ratios."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
V8 = OUT.parent
WORK = V8.parent
ROOT = V8 / "rank_fair_cache_recovery_formal_v3_output"
RAW = ROOT / "mid_b1_b5/capture"
OUTER = ROOT / "mid_b1_b5/outer"
sys.path.insert(0, str(V8))
sys.path.insert(0, str(V8 / "formal_interruption_review_20261110"))
import audit_interruption as independent
import verify_rank_fair_cache_formal as numeric
import win32_isolation_monitor as monitor

read = independent.read
sha = independent.digest
key = independent.identity


def record(path):
    return dict(path=path.relative_to(WORK).as_posix(), sha256=sha(path))


def inventory():
    return {f.relative_to(RAW).as_posix(): sha(f) for f in RAW.rglob("*") if f.is_file()}


def largest(tree):
    if isinstance(tree, dict):
        return max([largest(x) for x in tree.values()] + [0])
    if isinstance(tree, list):
        return max([largest(x) for x in tree] + [0])
    return tree if isinstance(tree, (int, float)) and not isinstance(tree, bool) else 0


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def main():
    before = inventory()
    p, summary, rows, ledger, isolation, pre, guard = [read(RAW / n) for n in (
        "protocol.json", "summary.json", "runs.json", "identity_ledger.json", "isolation.json", "preflight.json", "execution_environment_guard.json")]
    events, samples = lines(RAW / "monitor_events.jsonl"), lines(RAW / "process_samples.jsonl")
    assert summary["status"] == "diagnostic_only" and summary["attempted_slots"] == 12 and summary["successful_slots"] == 11
    assert summary["unattempted_slots"] == list(range(12, 50)) and len(rows) == len(ledger["children"]) == 12
    assert all(r["status"] == "complete" for r in rows[:11]) and rows[11]["status"] == "failed" and rows[11]["cleanup_failure_type"] == "PermissionError"
    assert sha(RAW / "protocol.json") == summary["protocol_sha256"] == sha(ROOT / "mid_b1_b5/protocol.json") and not summary["isolated_speed_claim_supported"]
    for table in ("raw_hashes", "monitor_evidence_sha256"):
        for name, digest in summary[table].items():
            assert sha(RAW / name) == digest
    qroot = V8 / "rank_fair_cache_qualification_output"
    qs, qp = read(qroot / "qualification_summary.json"), read(qroot / "protocol.json")
    assert sha(qroot / "qualification_summary.json") == "1da5cd2cf5771a701ba172a3e979b2c8003f84328810ade76a692756f33231df"
    assert len(qs["rows"]) == 78 and all(r["qualified"] for r in qs["rows"]) and len(qs["policy_pairs"]) == 39 and all(r["passed"] for r in qs["policy_pairs"])
    results = []
    for i, (rq, row, entry, control) in enumerate(zip(p["requests"], rows, ledger["children"], p["controls"])):
        qi = rq["qualification_index"]
        qr = qs["rows"][qi]
        capture, score = read(WORK / qr["source_capture"]["path"]), read(WORK / qr["score"]["path"])
        file, ref = RAW / row["artifact"], WORK / control["artifact"]["path"]
        assert sha(file) == row["trajectory_sha256"] and read(RAW / f"attempt_{i:03d}.json") == row and read(RAW / f"request_{i:03d}.json") == rq
        for r in (qr["score"], qr["source_capture"], capture["artifact"], capture["result"], capture["phases"], capture["watchdog"], rq["ptx_artifact"]):
            assert sha(WORK / r["path"]) == r["sha256"]
        assert qr["qualified"] and score["qualified"] and capture["job"] == score["job"] == qp["slots"][qi] == control["job"] and control["artifact"] == capture["artifact"]
        # This numerical function never promotes row status or window eligibility.
        gate = numeric.verify_trajectory(file, rq, control, row, RAW)
        assert gate["qualified"]
        a, b = independent.archive(file), independent.archive(ref)
        times, fulltimes = independent.doubles(a["times"]), independent.doubles(b["times"])
        ids = [fulltimes.index(t) for t in times]
        assert times == rq["solve_args"]["snapshots"] and len(times) == 19 and independent.doubles(a["probe_times"]) == [j / 8 for j in range(577)]
        h, data = b["states"]
        stride = math.prod(h["shape"][1:]) * int(h["descr"][2:])
        expected = b"".join(data[j * stride:(j + 1) * stride] for j in ids)
        assert a["states"][1] == expected and a["states"][0]["shape"] == (19, *h["shape"][1:]) and a["states"][0]["descr"] == h["descr"]
        assert all(a[k] == b[k] for k in ("probe_times", "transverse", "vorticity")) and all(independent.finite(a[k]) for k in ("states", "times", "probe_times", "transverse", "vorticity"))
        assert independent.non_time(independent.unicode_json(a["metadata"])) == independent.non_time(independent.unicode_json(b["metadata"]))
        assert row["exit_code"] == 0 and key(row["child_identity"]) == key(row["terminal_identity"]) and row["terminal_identity"]["exit_filetime_100ns"] > row["child_identity"]["birth_filetime_100ns"]
        assert row["exit_wait_completed_monotonic_ns"] <= min(row["hard_deadline_ns"], row["save_sent_monotonic_ns"] + p["exit_timeout_seconds"] * 1e9)
        proof = read(RAW / f"affinity_bootstrap_{i:03d}.json")
        assert proof["inherited_affinity_mask"] == 256 and proof["effective_affinity_mask"] == 16 and not proof["scientific_modules_before_pin"] and not proof["scientific_modules_after_pin"]
        assert row["launch_monotonic_ns"] <= proof["effective_pin_completed_ns"] <= proof["before_solver_import_ns"] <= row["ready"]["published_monotonic_ns"]
        for event in ("READY", "GO", "RETURN", "SAVE"):
            marker = read(RAW / f"{event.lower()}_{i:03d}.json")
            assert marker["affinity_bootstrap_sha256"] == sha(RAW / f"affinity_bootstrap_{i:03d}.json") and marker["inherited_affinity_mask"] == 256 and marker["effective_affinity_mask"] == 16 and key(marker) == key(entry["identity"])
        terminal = read(RAW / f"outer_slot_terminal_{i:03d}.json")
        assert terminal["terminal_identity"] == row["terminal_identity"] and terminal["exit_code"] == 0
        results.append(dict(index=i, job={k: rq[k] for k in ("name", "arm", "seed", "policy", "block")}, qualification_index=qi,
            raw_status=row["status"], numerical_values_match_existing_full_qualification=True, physical_times=19, probe_times=577,
            configuration_and_non_time_metadata_exact=True, cache_and_phase_checks_passed=True,
            original_qualification_physical_max=largest(score["physical_worst"]), original_qualification_guard_max=largest(score["guard_worst"]),
            original_qualification_implementation_max=largest(score["implementation_worst"]), original_qualification_probes=score["probes"],
            raw_sha256=sha(file), qualification_score=qr["score"], exact_process_exit_code=0, lifecycle_accepted=(i < 11), formal_timing_eligible=False))
    known = {key(x) for x in p["known_background"]}
    own, coordinator = key(ledger["monitor"]), key(ledger["coordinator"])
    anchors = pre["sample_anchor_indices"]
    computed_pre = monitor.evaluate_quiet_preflight([samples[a:b + 1] for a, b in zip(anchors, anchors[1:])],
        [x["gpu_percent"] for x in pre["windows"]], p["max_poll_gap_ns"], known_background=known | {coordinator}, monitor_key=own)
    assert computed_pre == {k: pre[k] for k in ("passed", "windows")} and computed_pre["passed"]
    roles, terminals, registered, exited = {own: "monitor", coordinator: "cohort_coordinator"}, {}, [], []
    for e in events:
        c = e["command"]
        if c["event"] == "REGISTER":
            roles[key(c["identity"])] = "cohort_child"
            registered.append(c["index"])
        elif c["event"] == "EXIT":
            terminals[key(c["terminal"])] = c["terminal"]
            exited.append(c["index"])
    assert registered == list(range(12)) and exited == list(range(11))
    computed = monitor.evaluate_active_interval(samples[:isolation["end_sample_index"] + 1], summary["open_qpc_ns"], summary["close_qpc_ns"],
        p["maximum_active_ns"], p["max_poll_gap_ns"], known_background=known, owned_roles=roles, owned_terminal=terminals)
    computed.update(status="coverage_incomplete", reason="missing_owned_terminal")
    assert computed == isolation["active"] and computed["external_activity_detected"] and not computed["full_lifecycle_verified"]
    external = [dict(budget_window=i, identity=x["identity"], cpu_seconds=x["total_delta_100ns"] / 1e7, terminal=x["terminal"],
                     over_original_point2_second_budget=x["total_delta_100ns"] > 2000000)
                for i, budget in enumerate(computed["budget_windows"]) for x in budget["deltas"] if x["role"] == "external_python"]
    state, temporary, terminal11 = [read(RAW / n) for n in ("outer_slot_state.json", "outer_slot_state.json.tmp", "outer_slot_terminal_011.json")]
    assert state["index"] == 11 and state["phase"] == "active" and temporary == terminal11 and ledger["children"][11]["status"] == "monitor_registered"
    assert sha(RAW / "outer_slot_state.json.tmp") == sha(RAW / "outer_slot_terminal_011.json")
    final, owner = read(OUTER / "watchdog_terminal.json"), read(OUTER / "owner_terminal.json")
    assert final["termination_confirmed"] and final["closing_input_guard_passed"] and final["reason"] == "owner_exited" and final["owner_exit_code"] == 0 and owner["coordinator_exit_code"] == 0 and owner["job_empty_confirmed"]
    assert guard["status"] == "passed" and guard["opening_environment"] == guard["closing_environment"] == p["environment"] and guard["coordinator_summary_sha256"] == sha(RAW / "summary.json")
    paths = subprocess.run(["rg", "--files", str(V8), "-g", "identity_ledger.json"], capture_output=True, text=True, check=True).stdout.splitlines()
    research = {}
    for name in paths:
        f, l = Path(name), read(Path(name))
        for role, identity in [("coordinator", l.get("coordinator")), ("monitor", l.get("monitor"))] + [("child", r.get("identity")) for r in l.get("children", [])]:
            if identity and "pid" in identity and "birth_filetime_100ns" in identity:
                research.setdefault(key(identity), []).append(dict(path=f.relative_to(WORK).as_posix(), role=role))
    attribution = []
    for cohort, base, pids in [("recovery_v2", V8 / "rank_fair_cache_recovery_formal_v2_output/mid_b1_b5/capture", [2172, 11512]),
                              ("recovery_v3", RAW, [12956, 11556, 29744, 36444])]:
        seen = {}
        for sample in lines(base / "process_samples.jsonl"):
            bypid = {x["pid"]: x for x in sample["live"]}
            for x in sample["live"]:
                if x["pid"] not in pids:
                    continue
                k, parent = key(x), bypid.get(x["parent_pid"])
                parent_key = key(parent) if parent else None
                result = seen.setdefault(k, dict(cohort=cohort, identity=list(k), parent_pid=x["parent_pid"], exact_research_ledger_matches=research.get(k, []),
                    sampled_parent_identity=list(parent_key) if parent_key else None, parent_exact_research_ledger_matches=research.get(parent_key, []) if parent_key else [],
                    direct_child_of_identified_41960=(x["parent_pid"] == 41960), origin="unattributed"))
                if parent_key and result["sampled_parent_identity"] is None:
                    result.update(sampled_parent_identity=list(parent_key), parent_exact_research_ledger_matches=research.get(parent_key, []))
        attribution.extend(seen.values())
    assert all(not x["exact_research_ledger_matches"] and not x["parent_exact_research_ledger_matches"] and not x["direct_child_of_identified_41960"] for x in attribution)
    windows = []
    for name, slots in (("mid_b1_b5", 50), ("high_b1_b3", 66), ("high_b4_b5", 44)):
        if name != "mid_b1_b5":
            assert not (ROOT / name / "capture").exists() and not (ROOT / name / "outer").exists()
        windows.append(dict(window=name, registered=slots, attempted=12 if name == "mid_b1_b5" else 0, raw_successful=11 if name == "mid_b1_b5" else 0,
                            failed=1 if name == "mid_b1_b5" else 0, unattempted=38 if name == "mid_b1_b5" else slots, formal_accepted=0))
    assert before == inventory()
    report = dict(schema="rfm-v8-formal-recovery-final-closure-1", status="closed_without_new_isolated_ranking", formal_timing_eligible=False, speed_claim_supported=False,
        recovery_cap_exhausted=True, further_gpu_or_formal_attempts=False, total_registered=160, total_attempted=12, raw_successful=11, failed_slots=[11], unattempted=148,
        numerical_archives_matching_prior_qualification=12, formal_accepted=0,
        independent_disqualifying_reasons=["Slot 011 remained failed after PermissionError during terminal-state journal update; exact terminal was not committed to identity ledger or EXIT event",
            "Active monitor recorded external Python CPU activity above the original budget, independently preventing isolated timing acceptance"], windows=windows, rows=results,
        registered_strict_verifier=record(OUT / "registered_window_verifier.json"),
        numerical_gate_scope=dict(physical_gate=qp["physical_gate"], implementation_gate=qp["implementation_gate"], serial_probe_gate=qp["serial_probe_relative_and_scaled_absolute_gate"],
            exact_qualified_summary=record(qroot / "qualification_summary.json"), original_qualified_rows=78, original_passed_policy_pairs=39,
            derivation="Full saved T72 fields at all 19 formal times, all 577 probe times, config and non-time metadata match exact prior qualified arrays; no new physical solve or rescoring was substituted"),
        failure_localization=dict(index=11, raw_status="failed", cleanup_failure_type="PermissionError", worker_return_solved=True, trajectory_serialized=True, worker_exit_code=0,
            independent_terminal_file=record(RAW / "outer_slot_terminal_011.json"), uncommitted_current_state_temporary=record(RAW / "outer_slot_state.json.tmp"), temporary_content_equals_terminal_record=True,
            current_state_phase=state["phase"], ledger_status=ledger["children"][11]["status"], monitor_exit_event_received=False,
            stage_supported_by_persisted_evidence="Saving terminal outer_slot_state.json after the immutable per-slot terminal file and before identity ledger/EXIT update",
            precise_OS_operation_or_lock_holder="Not established: no traceback, Win32 error code, or handle ownership record was persisted. No driver, GPU, app restart or particular reader attribution follows.",
            original_current_state_temp_omitted_from_coordinator_raw_hashes=("outer_slot_state.json.tmp" not in summary["raw_hashes"]), failed_slot_not_promoted=True),
        monitor=dict(raw_decision_recomputed_exactly=True, preflight_recomputed_exactly=True, preflight_passed=True, active=computed, external_cpu_budget_observations=external,
            scope="Observed Python lifetimes and sampled/endpoint activity only; unobserved short lifetimes and non-Python application load remain outside this scope"),
        limited_attribution=dict(ledger_files_scanned=len(paths), observations=attribution,
            conclusion="No listed external exact identity or its sampled direct parent matches an owned research ledger identity; none is a direct child of identified Python41960. More remote ancestry and source application remain unknown."),
        outer_cleanup=dict(normal_coordinator_exit=True, job_empty_confirmed=True, terminal=record(OUTER / "watchdog_terminal.json"), does_not_override_formal_rejection=True), environment_guard=record(RAW / "execution_environment_guard.json"),
        raw_evidence_unchanged_during_audit=True, raw_evidence_sha256=before, old_attempts_not_combined=True, no_pairwise_speed_ratios_computed=True, flow_solves_run=0, gpu_calls=0,
        auditor_sha256=sha(Path(__file__)), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = OUT / "final_closure.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = """Formal recovery closed without a new isolated ranking.

Final registered cohort: 160 slots; 12 attempted; 11 raw-successful; slot 011 failed; 148 not attempted. High windows were never started.
All 12 saved T72 numerical archives match the exact prior qualification, including 19 formal fields and all 577 probe times, metadata and cache checks. This does not promote failed slot 011 or establish timing eligibility.

Two independent reasons reject timing: (1) PermissionError left slot 011 terminal-state update uncommitted, without ledger EXIT; (2) active monitor recorded unknown external Python CPU over the original budget. Correcting the journal alone would not recover an isolated ranking.

The exact worker exit was zero; immutable terminal and uncommitted temporary state agree. The error stage is the terminal-state save before ledger/EXIT. The precise OS operation and lock holder are unknown. Outer Job cleanup and closing environment guard passed.

The predefined final-attempt cap is exhausted: no further GPU runs, no new formal attempt, no partial-row ranking, and no mixing earlier attempts. Original evidence remains unchanged.
"""
    (OUT / "final_closure.txt").write_text(text, encoding="utf-8")
    print(json.dumps(dict(status=report["status"], numerical_exact=12, raw_successful=11, formal_accepted=0, json_sha256=sha(path), attribution=attribution)))


if __name__ == "__main__":
    main()
