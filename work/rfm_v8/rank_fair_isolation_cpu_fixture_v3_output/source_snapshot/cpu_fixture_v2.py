"""Owned-Job CPU fixtures for CPU8 coordinator / CPU6 monitor / CPU4 worker."""
import argparse
import ctypes as C
from ctypes import wintypes as W
from hashlib import sha256
from pathlib import Path
import sys
import time

from worker_bootstrap_v2 import SCHEMA, admission, receive_control, affinity, scientific_modules
from transport_v2 import SECOND, TERMINATION_CODE, key, latch, publish, read, require

SOURCE = Path(__file__).resolve()
HERE = SOURCE.parent
WORK = HERE.parent.parent


def assert_outer_owner(evidence):
    from prepared_process_identity import current_process_identity
    proof = read(evidence / "owner_ready.json")
    require(proof["assigned_before_resume"] and proof["job_kill_on_close"] and not proof["job_handle_inherited"] and
            key(proof["coordinator_identity"]) == key(current_process_identity()), "Missing exact externally owned Job admission")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.IsProcessInJob.argtypes = (W.HANDLE, W.HANDLE, C.POINTER(W.BOOL))
    present = W.BOOL()
    require(bool(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, C.byref(present))) and present.value,
            "Refusing an unprotected coordinator subtree")


def above_normal():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.SetPriorityClass.argtypes = (W.HANDLE, W.DWORD)
    kernel.GetPriorityClass.argtypes = (W.HANDLE,)
    kernel.GetPriorityClass.restype = W.DWORD
    require(bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x8000)), "Cannot set fixture above-normal priority")
    require(kernel.GetPriorityClass(kernel.GetCurrentProcess()) == 0x8000, "Unexpected fixture priority")
    return 0x8000


def burn(output, index, stage):
    priority = above_normal()
    publish(output / f"busy_{index:03d}.json", dict(stage=stage, affinity_mask=affinity(), priority_class=priority,
                                                 scientific_modules=scientific_modules(), qpc_ns=time.perf_counter_ns()))
    stop = time.perf_counter_ns() + 60 * SECOND
    value = 0
    while time.perf_counter_ns() < stop:
        value = (value + 3) % 1000003
    raise RuntimeError("External and inner deadlines both failed to stop a CPU fixture")


def worker(path, digest):
    request, target, identity, parent, marker = admission(path, digest)
    mode, index, output = request["synthetic_mode"], request["index"], target.parent
    if mode == "before_ready":
        burn(output, index, mode)
    publish(latch(output, index, "READY"), dict(marker, event="READY", published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target, "GO", request, identity, parent, marker)
    if mode in ("busy_return", "outer_deadline"):
        burn(output, index, mode)
    started = time.perf_counter_ns()
    time.sleep(.035)
    solved = time.perf_counter_ns()
    before = time.perf_counter_ns()
    released = time.perf_counter_ns()
    core, release = (solved - started) / SECOND, (released - before) / SECOND
    publish(latch(output, index, "RETURN"), dict(marker, event="RETURN", status="solved", solver_seconds=core + release,
        solver_core_seconds=core, model_release_seconds=release, spatial_sha256="synthetic_no_flow", solve_started_ns=started,
        solve_completed_ns=solved, model_release_started_ns=before, model_released_ns=released, published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target, "SAVE", request, identity, parent, marker)
    if mode == "busy_exit":
        burn(output, index, mode)
    with target.with_suffix(".tmp.npz").open("xb") as handle:
        handle.write(b"synthetic CPU transport artifact; not numerical data\n")
    target.with_suffix(".tmp.npz").rename(target)
    return 0


def prepare(root, mode):
    import coordinator_v2 as coordinator
    prior = coordinator.prior
    root.mkdir()
    token = sha256(("affinity-v2 cpu fixture " + mode).encode("ascii")).hexdigest()
    old_sources = read(HERE.parent / "rank_fair_cache_formal_protocols" / "mid_b1_b5.json")["source_hashes"]
    names = prior.REQUIRED_SOURCES | set(old_sources) | {p.relative_to(WORK).as_posix() for p in HERE.glob("*.py")}
    plan = dict(schema=coordinator.SCHEMA, status="frozen", mode="transport_only", run_token=token,
                coordinator_cpu=8, monitor_cpu=6, poll_interval_ns=SECOND // 10, max_poll_gap_ns=SECOND // 4,
                maximum_active_ns=8 * SECOND, budget_interval_ns=10 * SECOND, max_budget_lateness_ns=2 * SECOND,
                ready_timeout_seconds=1.2, return_timeout_seconds=.8, exit_timeout_seconds=.8,
                slot_hard_timeout_seconds=2.5, known_background=[], source_hashes={n: prior.digest(WORK / n) for n in sorted(names)})
    plan["requests"] = [dict(schema=SCHEMA, index=i, run_token=token, affinity_mask=16, inherited_affinity_mask=256,
                             effective_affinity_mask=16, trajectory=f"trajectory_{i:03d}.npz", synthetic_mode=branch)
                        for i, branch in enumerate((mode, "success"))]
    prior.save(root / "protocol.json", plan)
    return plan


def execute(protocol, output, outer):
    assert_outer_owner(outer)
    import coordinator_v2 as coordinator
    plan = read(protocol)
    if plan["requests"][0]["synthetic_mode"] == "outer_deadline":
        # Fixture-only loss of coordinator responsiveness while the independent
        # non-Python watchdog must still enforce the active slot deadline.
        normal_receive = coordinator.ExactOwned.receive
        def stalled_receive(self, path, deadline_ns):
            if path.name.startswith("return_"):
                time.sleep(60)
            return normal_receive(self, path, deadline_ns)
        coordinator.ExactOwned.receive = stalled_receive
    summary = coordinator.run(protocol, output, plan, protocol.read_bytes(), worker=SOURCE)
    rows, ledger = read(output / "runs.json"), read(output / "identity_ledger.json")
    mode = plan["requests"][0]["synthetic_mode"]
    checks = dict(unpromoted=not summary["isolated_speed_claim_supported"], children_accounted=len(rows) == len(ledger["children"]))
    for row, entry in zip(rows, ledger["children"], strict=True):
        require(key(row["child_identity"]) == key(row["terminal_identity"]) and row["terminal_identity"] == entry["terminal"], "Exact terminal identity lost")
        proof = read(output / f"affinity_bootstrap_{row['index']:03d}.json")
        require(proof["inherited_affinity_mask"] == 256 and proof["effective_affinity_mask"] == 16 and not proof["scientific_modules_before_pin"] and not proof["scientific_modules_after_pin"], "CPU bootstrap invalid")
    if mode == "success":
        import verify_transport_v2 as verifier
        for row, entry in zip(rows, ledger["children"], strict=True):
            verifier.verify_transport(output, row, plan["requests"][row["index"]], entry, ledger, plan, summary["open_qpc_ns"])
        checks["complete_strict_transport"] = summary["status"] == "transport_complete" and len(rows) == 2
    else:
        checks["exact_timeout_terminal"] = summary["status"] == "transport_failed" and len(rows) == 1 and summary["unattempted_slots"] == [1] and rows[0]["status"] == "failed" and rows[0]["failure_type"] == "TimeoutError" and rows[0]["exit_code"] == TERMINATION_CODE and rows[0]["watchdog"]["terminate_process_ok"] and rows[0]["watchdog"]["post_terminate_wait_status"] == 0
        busy = read(output / "busy_000.json")
        checks["cpu4_above_normal_no_science"] = busy["affinity_mask"] == 16 and busy["priority_class"] == 0x8000 and not busy["scientific_modules"]
    require(all(checks.values()), "CPU isolation fixture failed")
    coordinator.prior.save(output.parent / "cpu_case_certificate.json", dict(status="passed", mode=mode, cpu_only=True,
        checks=checks, summary_sha256=coordinator.prior.digest(output / "summary.json"), coordinator_cpu=8, monitor_cpu=6, worker_cpu=4))


if __name__ == "__main__":
    # Worker receives exactly request-path/hash, preserving the real Popen path.
    if len(sys.argv) == 3 and not sys.argv[1].startswith("--"):
        raise SystemExit(worker(Path(sys.argv[1]).resolve(), sys.argv[2]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--mode", choices=("success", "before_ready", "busy_return", "busy_exit", "outer_deadline"))
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--outer-evidence", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve(), args.mode)
    else:
        require(args.execute and args.output and args.outer_evidence, "Execution requires an outer watchdog")
        execute(args.execute.resolve(), args.output.resolve(), args.outer_evidence.resolve())
