"""CPU-only real Windows Job fixtures, including a stalled termination helper."""

import argparse
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
    )
    source = HERE/"rank_fair_outer_watchdog.cs"
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
            path = root/(name+"_request.json")
            save(path, request)
            started = time.perf_counter()
            completed = subprocess.run([str(exe), "--observe", str(path)], capture_output=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=5)
            elapsed = time.perf_counter()-started
            result = read(folder/"watchdog_terminal.json")
            assert result["termination_confirmed"] is expect_confirmed, result
            assert elapsed < 3 and result["wall_seconds"] < 2.5, (elapsed, result)
            assert result["next_window_allowed"] is False and sentinel.poll() is None
            assert read(folder/"owner_ready.json")["assigned_before_resume"] is True
            assert result["closing_input_guard_passed"] is True
            if expect_confirmed:
                assert completed.returncode == 0
                assert result["owner_terminal"]["job_empty_confirmed"] is True
                assert result["owner_terminal"]["coordinator_exit_code"] == child_code
            else:
                assert completed.returncode == 2 and result["status"] == "termination_pending"
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
            records.append(dict(case=name, status="passed", observer_exit_code=completed.returncode,
                                observer_wall_seconds=elapsed, termination_confirmed=expect_confirmed,
                                terminal_sha256=digest(folder/"watchdog_terminal.json"),
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
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=3)
    report = dict(status="passed", gpu_contexts_created=0, flow_solves_run=0, all_windows_restarted=False,
                  no_formal_experiment_started=True, source_sha256=digest(source), executable_sha256=digest(exe),
                  fixture_sha256=digest(Path(__file__).resolve()), cases=records,
                  pending_records_immutable=True, no_external_python_whitelist_added=True,
                  limitation="Finite observer evidence is proved. A stuck OS termination is reported pending; it is not a guarantee that the driver or owned processes have terminated.")
    save(root/"certificate.json", report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--executable", type=Path, required=True)
    args = cli.parse_args()
    result = run(args.output.resolve(), args.executable.resolve())
    print(json.dumps({k:result[k] for k in ("status", "gpu_contexts_created", "flow_solves_run", "no_formal_experiment_started")}))
