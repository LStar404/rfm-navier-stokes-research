"""Prepare/run/audit one explicitly diagnostic, externally protected T9 entry test."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"rank_fair_isolation_v2"))
import binding_v2 as binding
import smoke_launcher_v2 as launcher
import verify_transport_v2 as transport
import verify_rank_fair_cache_formal as cache_verify
from transport_v2 import read, require, key

T, WORK = binding.T, binding.WORK
SOURCE = Path(__file__).resolve()
WATCHDOG_ROOT = HERE/"rank_fair_cache_recovery_watchdog_fixture_v6_output"
BACKGROUND = HERE/"rank_fair_cache_recovery_identified_background.json"
INDICES = (11, 59, 76, 24, 20)


def record(path):
    return binding.qual.shared.record(path)


def prepare(root, cpu_certificate):
    cpu = read(cpu_certificate)
    require(cpu["status"] == "passed" and cpu["cpu_only"] and cpu["sources_sha256"] == binding.source_hashes(), "Matching CPU8/4 fixture required")
    qualification = HERE/"rank_fair_cache_qualification_output"
    qp, controls, _, qualified = binding.old.load_qualification(qualification)
    base = binding.old.make_window("mid_b1_b5", qualification)
    admin = read(HERE/"identified_background_recovery_admin.json")
    identity = {name:admin["identity"][name] for name in ("pid", "birth_filetime_100ns")}
    require(identity == dict(pid=41960, birth_filetime_100ns=134387561968139470), "Identified background changed")
    if not BACKGROUND.exists():
        T.save(BACKGROUND, dict(known_background=[identity], admin_evidence=record(HERE/"identified_background_recovery_admin.json"),
            status="prospectively_identified_original_budget_applies", known_individual_cpu_seconds_per_10_seconds=.5,
            known_total_cpu_seconds_per_10_seconds=1., new_unknown_python_rejected=True))
    require(read(BACKGROUND)["known_background"] == [identity], "Background binding changed")
    paths = (SOURCE, cpu_certificate, BACKGROUND, HERE/"rank_fair_cache_recovery_cpu_audit.json",
             HERE/"rank_fair_outer_watchdog.cs", WATCHDOG_ROOT/"rank_fair_outer_watchdog.exe", WATCHDOG_ROOT/"certificate.json",
             HERE/"rank_fair_cache_formal_mid_output/request_002.json")
    sources = dict(base["source_hashes"], **binding.source_hashes())
    sources.update({p.relative_to(WORK).as_posix():T.digest(p) for p in paths})
    snapshots = [v for v in qp["physical_times"] if v<=9]
    requests = []
    for index, qualified_index in enumerate(INDICES):
        control = qp["slots"][qualified_index]
        args = dict(control["args"], horizon=9, snapshots=snapshots)
        if "steps" in args:
            require(args["steps"] % 8 == 0, "T9 subset does not preserve integral time steps")
            args["steps"] //= 8
        fields = {k:control[k] for k in ("name", "arm", "seed", "policy", "family", "engine", "config", "ptx_artifact", "ptx_cache_key", "ptx_sha256", "runtime_id")}
        requests.append(dict(fields, index=index, qualification_index=qualified_index, block=1,
            schema=binding.WORKER_SCHEMA, affinity_mask=16, inherited_affinity_mask=256, effective_affinity_mask=16,
            trajectory=f"trajectory_{index:03d}.npz", imports="minimal", solve_args=args, compiler_identity=qp["compiler_identity"]))
    old_slot = read(HERE/"rank_fair_cache_formal_mid_output/request_002.json")
    require(all(requests[-1][k] == old_slot[k] for k in ("name", "arm", "seed", "policy", "qualification_index")), "Last T9 case is not exact old slot2 identity")
    p = dict(schema=binding.SCHEMA, status="frozen", mode="equivalence_only", study="entry_equivalence_T9_not_timing",
        coordinator_cpu=8, monitor_cpu=6, poll_interval_ns=1_000_000_000, max_poll_gap_ns=2_500_000_000,
        maximum_active_ns=150_000_000_000, budget_interval_ns=10_000_000_000, max_budget_lateness_ns=2_000_000_000,
        known_background=[identity], ready_timeout_seconds=base["ready_timeout_seconds"], return_timeout_seconds=base["return_timeout_seconds"],
        exit_timeout_seconds=base["exit_timeout_seconds"], slot_hard_timeout_seconds=45,
        qualification=qualified, controls=[controls[i] for i in INDICES], qualification_indices=list(INDICES),
        physical_times=snapshots, environment=base["environment"], sources_sha256=binding.source_hashes(),
        source_hashes=sources, cpu_fixture=record(cpu_certificate), identified_background=record(BACKGROUND),
        formal_timing_eligible=False, exact_T72_numerical_qualification_replaced=False,
        stated_scope="Five new entry/terminal T9 diagnostic cases; no speed comparison and no formal cohort completion.",
        outer_supervision=dict(observer_cpu=10, owner_cpu=12, whole_window_watchdog_seconds=210, confirmation_seconds=5,
                               executable=record(WATCHDOG_ROOT/"rank_fair_outer_watchdog.exe")))
    token = binding.old.rank.frozen.canonical_sha(dict(p, requests=requests))
    p.update(run_token=token, requests=[dict(j, run_token=token) for j in requests])
    root.mkdir()
    protocol = root/"protocol.json"
    T.save(protocol, p)
    output, outer = root/"capture", root/"outer"
    argv = [sys.executable, "-B", str(SOURCE), "--execute", str(protocol), "--output", str(output), "--outer-evidence", str(outer)]
    inputs = {str((WORK/name).resolve()):digest for name,digest in sources.items()}
    inputs[str(protocol)] = T.digest(protocol)
    request = dict(directory=str(outer), application=sys.executable, command_line=subprocess.list2cmdline(argv), working_directory=str(HERE),
        observer_cpu=10, owner_cpu=12, deadline_seconds=210, confirmation_seconds=5, inputs_sha256=inputs,
        slot_state_path=str(output/"outer_slot_state.json"), run_token=token, slot_count=5, slot_timeout_seconds=45)
    T.save(root/"watchdog_request.json", request)
    T.save(root/"prepared.json", dict(status="prepared_not_executed", gpu_contexts_created=0, flow_solves_run=0,
        cases=[{k:j[k] for k in ("index", "name", "arm", "seed", "policy", "qualification_index")} for j in requests],
        protocol=record(protocol), watchdog_request=record(root/"watchdog_request.json"),
        launch_argv=[str(WATCHDOG_ROOT/"rank_fair_outer_watchdog.exe"), "--observe", str(root/"watchdog_request.json")],
        audit_argv=[sys.executable, "-B", str(SOURCE), "--audit", str(root)]))
    return dict(status="prepared_not_executed", cases=5, directory=str(root))


def execute(protocol, output, outer):
    p = read(protocol)
    T.check_sources(p["source_hashes"])
    require(binding.old.environment() == p["environment"], "Current environment differs before diagnostic")
    before = time.perf_counter_ns()
    summary = launcher.execute(protocol, output, outer)
    after = time.perf_counter_ns()
    guard = dict(status="failed", protocol_sha256=T.digest(protocol), opening_environment=p["environment"],
        opening_guard_qpc_ns=before, closing_guard_qpc_ns=after, coordinator_summary_sha256=T.digest(output/"summary.json"))
    try:
        T.check_sources(p["source_hashes"])
        require(binding.old.environment() == p["environment"] and protocol.read_bytes() == (output/"protocol.json").read_bytes(), "Diagnostic runtime/protocol changed")
        guard.update(status="passed", closing_environment=p["environment"])
    except Exception as error:
        guard.update(failure_type=type(error).__name__)
    T.save(output/"execution_environment_guard.json", guard)
    return dict(status=summary["status"], environment_guard=guard["status"], formal_timing_eligible=False)


def audit(root):
    import numpy as np
    p = read(root/"protocol.json")
    T.check_sources(p["source_hashes"])
    require(p["sources_sha256"] == binding.source_hashes() and p["qualification_indices"] == list(INDICES), "Entry source/scope changed")
    outer, output = root/"outer", root/"capture"
    final, owner = read(outer/"watchdog_terminal.json"), read(outer/"owner_terminal.json")
    require(final["status"] == "owned_tree_terminal" and final["reason"] == "owner_exited" and final["termination_confirmed"] and
            final["owner_exited"] and final["owner_exit_code"] == 0 and final["closing_input_guard_passed"] and
            final["owner_terminal"] == owner and owner["status"] == "job_empty" and owner["reason"] == "coordinator_exited" and
            owner["job_empty_confirmed"] and owner["coordinator_exit_code"] == 0, "External bounded execution did not close normally")
    opening, inputs, request = read(outer/"watchdog_open.json"), read(outer/"watchdog_input_guard.json"), read(root/"watchdog_request.json")
    require(T.digest(root/"watchdog_request.json") == final["request_sha256"] == opening["request_sha256"] == inputs["request_sha256"] and
            T.digest(outer/"watchdog_input_guard.json") == final["input_guard_sha256"] and inputs["inputs_sha256"] == request["inputs_sha256"], "Outer input binding differs")
    for name,digest in p["source_hashes"].items():
        require(inputs["inputs_sha256"].get(str((WORK/name).resolve())) == digest, "Outer guard omitted a source")
    require(inputs["inputs_sha256"].get(str(root/"protocol.json")) == T.digest(root/"protocol.json"), "Outer protocol binding omitted")
    summary, rows, ledger = (read(output/name) for name in ("summary.json", "runs.json", "identity_ledger.json"))
    require(summary["status"] == "transport_complete" and summary["intended_slots"] == summary["successful_slots"] == 5 and
            not summary["unattempted_slots"] and len(rows) == len(ledger["children"]) == 5 and
            key(owner["coordinator_terminal"]) == key(ledger["coordinator"]), "Five diagnostic cases did not finish")
    for name,digest in summary["raw_hashes"].items():
        require(T.digest(output/name) == digest, "Diagnostic raw file changed")
    require(cache_verify.legacy.guard_accepted(output, p, summary)[0], "Opening/closing environment guard failed")
    results = []
    for rq,row,entry,control in zip(p["requests"], rows, ledger["children"], p["controls"], strict=True):
        path = transport.verify_transport(output,row,rq,entry,ledger,p,summary["open_qpc_ns"])
        reference = binding.qual.shared.bound(control["artifact"])
        with np.load(path, allow_pickle=False) as actual,np.load(reference, allow_pickle=False) as full:
            meta = json.loads(str(actual["metadata"]))
            require(json.loads(str(actual["config"])) == rq["config"] == json.loads(str(full["config"])), "Diagnostic config differs")
            states = [int(np.flatnonzero(full["times"] == t).item()) for t in p["physical_times"]]
            probes = full["probe_times"] <= 9
            expect = dict(states=full["states"][states], times=np.array(p["physical_times"]), probe_times=full["probe_times"][probes],
                          transverse=full["transverse"][probes], vorticity=full["vorticity"][probes])
            arrays = {name:actual[name].dtype == value.dtype and actual[name].shape == value.shape and
                np.isfinite(actual[name]).all() and actual[name].tobytes() == value.tobytes() for name,value in expect.items()}
            require(all(arrays.values()) and len(actual["probe_times"]) == 73 and row["marker"]["spatial_sha256"] == control["spatial_sha256"], "T9 subset not bitwise equal to exact T72 qualification")
            cache_verify.verify_cache(row, rq, meta, output)
        results.append(dict(index=rq["index"], name=rq["name"], arm=rq["arm"], policy=rq["policy"], equivalent=True,
                            arrays_bitwise=arrays, trajectory_sha256=T.digest(path), qualification_index=rq["qualification_index"]))
    result = dict(status="passed", all_cases_equivalent=True, formal_timing_eligible=False, sources_sha256=binding.source_hashes(),
        cases=results, solves_run_by_audit=0, protocol=record(root/"protocol.json"), capture_summary=record(output/"summary.json"),
        outer_terminal=record(outer/"watchdog_terminal.json"), environment_guard=record(output/"execution_environment_guard.json"),
        exact_T72_numerical_qualification_replaced=False, source_sha256=T.digest(SOURCE))
    T.save(root/"t9_equivalence_certificate.json", result)
    return dict(status=result["status"], all_cases_equivalent=True, cases=len(results), formal_timing_eligible=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--execute", type=Path)
    action.add_argument("--audit", type=Path)
    cli.add_argument("--cpu-certificate", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--outer-evidence", type=Path)
    args = cli.parse_args()
    if args.prepare:
        require(args.cpu_certificate, "Matching CPU fixture certificate required")
        result = prepare(args.prepare.resolve(), args.cpu_certificate.resolve())
    elif args.execute:
        require(args.output and args.outer_evidence, "Protected output required")
        result = execute(args.execute.resolve(), args.output.resolve(), args.outer_evidence.resolve())
    else:
        result = audit(args.audit.resolve())
    print(json.dumps(result))
