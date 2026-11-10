"""CPU-only real Windows Job fixtures, including a stalled termination helper."""

import argparse
import ctypes as C
from ctypes import wintypes as W
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


def read(path):
    return json.loads(path.read_bytes())


def run(root, exe):
    cases = (
        ("success", "success", {}, True, 0),
        ("coordinator_failure", "failure", {}, True, 17),
        ("hung_coordinator_and_descendant", "grandchild", {}, True, 772),
        ("failed_coordinator_with_descendant", "fail_with_child", {}, True, 19),
        ("stalled_termination_helper", "grandchild", {"fixture_termination_delay_ms": 3000}, False, None),
        ("owner_abrupt_exit", "hang", {"fixture_owner_abort_after_resume": True}, False, None),
        ("active_slot_deadline", "slot_deadline", {}, True, 772),
        ("slot_deadline_extension_rejected", "slot_extend", {}, True, 772),
        ("slot_wrong_identity_rejected", "slot_bad_identity", {}, True, 772),
    )
    source = HERE/"rank_fair_outer_watchdog.cs"
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (W.DWORD, W.BOOL, W.DWORD)
    kernel.OpenProcess.restype = W.HANDLE
    kernel.WaitForSingleObject.argtypes = (W.HANDLE, W.DWORD)
    kernel.WaitForSingleObject.restype = W.DWORD
    kernel.CloseHandle.argtypes = (W.HANDLE,)
    kernel.GetProcessTimes.argtypes = (W.HANDLE, *([C.POINTER(C.c_longlong)]*4))

    def process_times(handle, pid):
        values = [C.c_longlong() for _ in range(4)]
        assert kernel.GetProcessTimes(handle, *(C.byref(v) for v in values))
        return dict(pid=pid, **dict(zip(("birth_filetime_100ns", "exit_filetime_100ns", "kernel_cpu_100ns", "user_cpu_100ns"),
                                       (v.value for v in values), strict=True)))
    inputs = {str(p):digest(p) for p in (source, exe, Path(__file__).resolve(), HERE/"rank_fair_cache_recovery_cpu_audit.json")}
    sentinel = subprocess.Popen([str(exe), "--synthetic", "leaf", str(root)], creationflags=subprocess.CREATE_NO_WINDOW)
    records = []
    try:
        for name, mode, options, expect_confirmed, child_code in cases:
            folder = root/name
            folder.mkdir()
            request = dict(directory=str(folder), application=str(exe),
                           command_line=subprocess.list2cmdline([str(exe), "--synthetic", mode, str(folder)]),
                           working_directory=str(HERE), deadline_seconds=1.3, confirmation_seconds=.6,
                           inputs_sha256=inputs, fixture_only=True, **options)
            if mode.startswith("slot_"):
                request.update(slot_state_path=str(folder/"slot_state.json"), run_token="cpu-fixture-token", slot_count=1, slot_timeout_seconds=.45)
            path = root/(name+"_request.json")
            save(path, request)
            started = time.perf_counter()
            observer = subprocess.Popen([str(exe), "--observe", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            child_handle, exact_child = None, None
            if name == "owner_abrupt_exit":
                until = time.perf_counter()+2
                while not (folder/"owner_ready.json").exists() and time.perf_counter() < until:
                    time.sleep(.01)
                exact_child = read(folder/"owner_ready.json")["coordinator_identity"]
                child_handle = kernel.OpenProcess(0x00100000|0x1000, False, exact_child["pid"])
                assert child_handle
                assert process_times(child_handle, exact_child["pid"])["birth_filetime_100ns"] == exact_child["birth_filetime_100ns"]
            observer.communicate(timeout=5)
            elapsed = time.perf_counter()-started
            result = read(folder/"watchdog_terminal.json")
            assert result["termination_confirmed"] is expect_confirmed, result
            assert elapsed < 3 and result["wall_seconds"] < 2.5, (elapsed, result)
            assert result["next_window_allowed"] is False and sentinel.poll() is None
            assert read(folder/"owner_ready.json")["assigned_before_resume"] is True
            assert result["closing_input_guard_passed"] is True
            if expect_confirmed:
                assert observer.returncode == 0
                assert result["owner_terminal"]["job_empty_confirmed"] is True
                assert result["owner_terminal"]["coordinator_exit_code"] == child_code
            else:
                assert observer.returncode == 2 and result["status"] == "termination_pending"
            cleanup_proof = None
            if child_handle:
                try:
                    assert kernel.WaitForSingleObject(child_handle, 1000) == 0
                    cleanup_proof = process_times(child_handle, exact_child["pid"])
                    assert cleanup_proof["birth_filetime_100ns"] == exact_child["birth_filetime_100ns"]
                    assert cleanup_proof["exit_filetime_100ns"] > cleanup_proof["birth_filetime_100ns"]
                    save(folder/"independent_held_child_terminal.json", cleanup_proof)
                finally:
                    kernel.CloseHandle(child_handle)
            if mode.startswith("slot_"):
                expected = "active_slot_deadline_exceeded" if mode == "slot_deadline" else "active_slot_state_invalid"
                assert result["reason"] == expected and result["wall_seconds"] < 1.2, result
            if name == "stalled_termination_helper":
                assert result["owner_exited"] is False
                intent = read(folder/"owner_termination_intent.json")
                assert intent["termination_confirmed"] is False
                assert not (folder/"owner_termination_return.json").exists()
                cutoff = time.perf_counter()+6
                while not (folder/"owner_terminal.json").exists() and time.perf_counter() < cutoff:
                    time.sleep(.05)
                late = read(folder/"owner_terminal.json")
                assert late["job_empty_confirmed"] is True
                assert read(folder/"watchdog_terminal.json") == result  # Never rewrite a pending observation as success.
            records.append(dict(case=name, status="passed", observer_exit_code=observer.returncode,
                                observer_wall_seconds=elapsed, termination_confirmed=expect_confirmed,
                                terminal_sha256=digest(folder/"watchdog_terminal.json"),
                                independently_observed_child_terminal=cleanup_proof,
                                unrelated_sentinel_still_alive=True))
        # Corrupt binding rejection must happen before owner starts.
        folder = root/"changed_input_rejected"
        folder.mkdir()
        request = dict(directory=str(folder), application=str(exe), command_line="unused", working_directory=str(HERE),
                       deadline_seconds=1.3, confirmation_seconds=.6,
                       inputs_sha256={str(source):"0"*64}, fixture_only=True)
        path = root/"changed_input_rejected_request.json"
        save(path, request)
        bad = subprocess.run([str(exe), "--observe", str(path)], capture_output=True,
                             creationflags=subprocess.CREATE_NO_WINDOW, timeout=3)
        result = read(folder/"watchdog_terminal.json")
        assert bad.returncode == 3 and result["termination_confirmed"] is False
        assert not (folder/"owner_identity.json").exists() and sentinel.poll() is None
        records.append(dict(case="changed_input_rejected", status="passed", no_owner_launched=True,
                            terminal_sha256=digest(folder/"watchdog_terminal.json")))
        folder = root/"closing_input_change_rejected"
        folder.mkdir()
        mutable = folder/"fixture_input.txt"
        mutable.write_text("before", encoding="utf-8")
        request = dict(directory=str(folder), application=str(exe),
                       command_line=subprocess.list2cmdline([str(exe), "--synthetic", "hang", str(folder)]),
                       working_directory=str(HERE), deadline_seconds=1.3, confirmation_seconds=.6,
                       inputs_sha256=dict(inputs, **{str(mutable):digest(mutable)}), fixture_only=True)
        path = root/"closing_input_change_rejected_request.json"
        save(path, request)
        observer = subprocess.Popen([str(exe), "--observe", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        until = time.perf_counter()+2
        while not (folder/"owner_ready.json").exists() and time.perf_counter()<until:
            time.sleep(.01)
        assert (folder/"owner_ready.json").exists()
        mutable.write_text("after", encoding="utf-8")
        observer.communicate(timeout=4)
        result = read(folder/"watchdog_terminal.json")
        assert observer.returncode == 4 and result["status"] == "verification_failed"
        assert result["termination_confirmed"] is True and result["closing_input_guard_passed"] is False
        records.append(dict(case="closing_input_change_rejected", status="passed", observer_exit_code=4,
                            terminal_sha256=digest(folder/"watchdog_terminal.json")))
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=3)
    report = dict(status="passed", gpu_contexts_created=0, flow_solves_run=0, all_windows_restarted=False,
                  no_formal_experiment_started=True, source_sha256=digest(source), executable_sha256=digest(exe),
                  fixture_sha256=digest(Path(__file__).resolve()), cases=records,
                  pending_records_immutable=True, no_external_python_whitelist_added=True,
                  limitation="CPU fixtures prove finite observer return for tested coordinator/owner failures and delayed termination. Native file/hash/flush/process-creation calls remain synchronous; arbitrary OS failure is not a hard real-time guarantee. A stuck termination is pending, never proven cleanup.")
    save(root/"certificate.json", report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--executable", type=Path, required=True)
    args = cli.parse_args()
    result = run(args.output.resolve(), args.executable.resolve())
    print(json.dumps({k:result[k] for k in ("status", "gpu_contexts_created", "flow_solves_run", "no_formal_experiment_started")}))
