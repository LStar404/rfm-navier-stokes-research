"""Windows isolation smoke and read-only audit of the two archived v7 windows."""

import argparse
from ctypes import wintypes as W
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
import time
from unittest.mock import patch
from uuid import uuid4

import win32_isolation_monitor as monitor


C = monitor.C
HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
SECOND = 1_000_000_000
CHILD = """
import sys, time
print('READY', flush=True)
if sys.stdin.readline() != 'GO\\n':
    raise RuntimeError('Missing owned test barrier')
deadline = time.process_time_ns() + int(sys.argv[1])
while time.process_time_ns() < deadline:
    pass
"""


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def windows_ticks(iso):
    match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,7}))?(Z|[+-]\d\d:\d\d)", iso)
    require(match is not None, "Archived wall timestamp is not ISO8601")
    main, fraction, zone = match.groups()
    seconds = datetime.fromisoformat(main+zone)-datetime(1601, 1, 1, tzinfo=timezone.utc)
    return (seconds.days*86400+seconds.seconds)*10_000_000+int((fraction or "").ljust(7, "0"))


def audit_v7():
    cohorts = {}
    for scale in ("mid", "high"):
        directory = V7/f"prepared_formal_timing_{scale}_output"
        names = ("protocol.json", "isolation.json", "execution.json", "identity_ledger.json", "timing.json", "summary.json")
        fingerprints = {name: digest(directory/name) for name in names}
        raw, ledger, timing, summary = (json.loads((directory/name).read_bytes()) for name in
                                        ("isolation.json", "identity_ledger.json", "timing.json", "summary.json"))
        require(all(summary[Path(name).stem+"_sha256"] == fingerprints[name] for name in names[:-1]),
                "Archived v7 timing summary no longer binds its raw files")
        started = windows_ticks(raw["start_utc"])
        first_birth = ledger["children"][0]["birth_filetime_100ns"]
        exit_time = raw["coordinator_exit"]["exit_filetime_100ns"]
        issues, activity, overlaps = [], [], []
        earlier = windows_ticks(raw["preflight"][-1]["snapshot"]["snapshot_end_utc"])
        for sample in raw["samples"]:
            snapshot = sample["snapshot"]
            left, right = map(windows_ticks, (snapshot["snapshot_start_utc"], snapshot["snapshot_end_utc"]))
            target = sample["target_elapsed_seconds"]
            for record in snapshot["unresolved_python"]:
                matches = [row for row in ledger["children"] if row["pid"] == record["pid"]]
                issues.append(dict(target_seconds=target, issue=record, after_owned_coordinator_exit=left > exit_time,
                                   ledger_pid_matches=len(matches), exact_issue_identity_present="birth_filetime_100ns" in record,
                                   matching_pid_exit_during_snapshot=any(left <= row["exit_filetime_100ns"] <= right for row in matches),
                                   matching_pid_exit_minus_snapshot_start_ms=[(row["exit_filetime_100ns"]-left)/10_000 for row in matches]))
            if sample["external_activity_detected"]:
                activity.append(target)
                if earlier < exit_time and right > first_birth:
                    overlaps.append(target)
            earlier = right
        require(not summary["isolated_speed_claim_supported"] and summary["rows"] == len(timing["rows"]),
                "v7 diagnostic verdict changed")
        cohorts[scale] = dict(source_sha256=fingerprints, status=summary["status"], rows=len(timing["rows"]),
                              fixed_seconds=raw["window_seconds"], owned_coordinator_exit_elapsed_seconds=(exit_time-started)/1e7,
                              unresolved=issues, external_sample_targets_seconds=activity,
                              external_intervals_overlapping_first_child_birth_to_coordinator_exit=overlaps)
    return cohorts


def etw_empty_capability():
    """Attempt an empty uniquely named session. Stop it only if we own it."""
    class Wnode(C.Structure):
        _fields_ = [("BufferSize", W.ULONG), ("ProviderId", W.ULONG), ("HistoricalContext", C.c_uint64),
                    ("TimeStamp", C.c_int64), ("Guid", C.c_ubyte*16), ("ClientContext", W.ULONG), ("Flags", W.ULONG)]

    class Properties(C.Structure):
        _fields_ = [("Wnode", Wnode)]+[(name, W.ULONG) for name in (
            "BufferSize", "MinimumBuffers", "MaximumBuffers", "MaximumFileSize", "LogFileMode", "FlushTimer",
            "EnableFlags", "AgeLimit", "NumberOfBuffers", "FreeBuffers", "EventsLost", "BuffersWritten",
            "LogBuffersLost", "RealTimeBuffersLost")]+[("LoggerThreadId", W.HANDLE),
            ("LogFileNameOffset", W.ULONG), ("LoggerNameOffset", W.ULONG)]

    storage = C.create_string_buffer(C.sizeof(Properties)+2048)
    properties = Properties.from_buffer(storage)
    properties.Wnode.BufferSize = C.sizeof(storage)
    properties.Wnode.Flags = 0x20000
    properties.Wnode.ClientContext = 1
    properties.LogFileMode = 0x100
    properties.LoggerNameOffset = C.sizeof(Properties)
    name = "RfmV8EmptyCapability-"+uuid4().hex
    advapi = C.WinDLL("advapi32")
    start = monitor.bind(advapi, "StartTraceW", W.ULONG, C.POINTER(C.c_uint64), W.LPCWSTR, C.c_void_p)
    control = monitor.bind(advapi, "ControlTraceW", W.ULONG, C.c_uint64, W.LPCWSTR, C.c_void_p, W.ULONG)
    handle = C.c_uint64()
    status = start(C.byref(handle), name, storage)
    stopped = control(handle, None, storage, 1) if status == 0 else None
    require(stopped in (None, 0), "Owned empty ETW session did not stop")
    return dict(start_win32_status=status, stop_win32_status=stopped, providers_enabled=0, trace_files=0)


def child(cpu_ns):
    p = subprocess.Popen([sys.executable, "-B", "-u", "-c", CHILD, str(cpu_ns)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    if p.stdout.readline().rstrip(b"\r\n") != b"READY":
        cleanup(p)
        raise RuntimeError("Owned smoke child failed before READY")
    return p


def release(p):
    p.stdin.write(b"GO\n")
    p.stdin.flush()


def finish(p, census):
    require(p.wait(timeout=10) == 0, "Owned smoke child failed")
    return census.times(p._handle)


def cleanup(p):
    if p.poll() is None:
        p.kill()
        p.wait(timeout=10)
    p.stdin.close()
    p.stdout.close()


def observed_lifetimes(census):
    with monitor.LifecycleSampler(census) as sampler:
        samples = [sampler.sample()]
        opened = time.perf_counter_ns()
        owned = child(80_000_000)
        try:
            external = child(330_000_000)
            try:
                owned_key, external_key = (monitor.process_key(census.times(p._handle)) for p in (owned, external))
                calibration = census.calibrate(owned._handle)
                samples.append(sampler.sample())
                release(owned)
                release(external)
                while owned.poll() is None or external.poll() is None:
                    samples.append(sampler.sample())
                    time.sleep(.02)
                expected = {monitor.process_key(row): row for row in (finish(owned, census), finish(external, census))}
                closed = time.perf_counter_ns()
                samples.append(sampler.sample())
            finally:
                cleanup(external)
        finally:
            cleanup(owned)
    sampled = {monitor.process_key(row): row for sample in samples for row in sample["terminals"]}
    require(all(all(sampled[key][field] == row[field] for field in (
            "exit_filetime_100ns", "kernel_cpu_100ns", "user_cpu_100ns")) for key, row in expected.items()),
            "Retained terminal identities or counters differ from Popen's owned handles")
    self_key = monitor.process_key(census.times(census.current_process()))
    result = monitor.evaluate_active_interval(samples, opened, closed, 10*SECOND, SECOND,
        owned_roles={owned_key:"cohort_child", self_key:"monitor"}, owned_terminal={owned_key:expected[owned_key]})
    selected = {tuple(row["identity"]): row for window in result["budget_windows"] for row in window["deltas"]}
    require(selected[owned_key]["role"] == "cohort_child" and selected[external_key]["role"] == "external_python" and
            selected[external_key]["total_delta_100ns"] > monitor.EXTERNAL_CPU_TICKS and
            result["external_activity_detected"] and not result["full_lifecycle_verified"],
            "Measured unknown process did not trip the unchanged activity threshold")
    return dict(owned_child_calibration=calibration, sampled_terminals_exact=True,
                owned=selected[owned_key], external=selected[external_key], boundary_verdict=result["status"],
                maximum_poll_gap_ns=result["max_poll_gap_ns"], controlled_raw_samples=[
                    {**{key: row[key] for key in ("qpc_start_ns", "qpc_end_ns", "start_filetime_100ns", "end_filetime_100ns")},
                     **{field:[process for process in row[field] if monitor.process_key(process) in expected]
                        for field in ("live", "terminals", "acquisition_issues")}} for row in samples])


def disappearing_after_census(census):
    with monitor.LifecycleSampler(census) as sampler:
        left = sampler.sample()
        opened = time.perf_counter_ns()
        p = child(30_000_000)
        try:
            live = census.live()
            key = monitor.process_key(census.times(p._handle))
            release(p)
            terminal = finish(p, census)
            p._handle.Close()
            # The system census is valid, but handle attachment happens after exit.
            with patch.object(census, "live", return_value=live):
                raced = sampler.sample()
            closed = time.perf_counter_ns()
            right = sampler.sample()
        finally:
            cleanup(p)
    self_key = monitor.process_key(census.times(census.current_process()))
    require(any(monitor.process_key(row) == key for row in raced["live"]), "The original race observation lost its exact identity")
    actual_issues = [row for row in raced["acquisition_issues"] if monitor.process_key(row) == key]
    real_terminal = next((row for row in right["terminals"] if monitor.process_key(row) == key), None)
    if not actual_issues and real_terminal is not None:
        require(all(terminal[field] == real_terminal[field] for field in (
                    "exit_filetime_100ns", "kernel_cpu_100ns", "user_cpu_100ns")), "Raced handle terminal counters differ")

    original_open = census.open_process

    def deny_owned_pid(access, inherited, pid):
        if pid == key[0]:
            C.set_last_error(87)
            return None
        return original_open(access, inherited, pid)

    # A retained OS object can remain openable after its owner's handle closes.
    # Inject the documented failing attach only for this controlled child.
    with monitor.LifecycleSampler(census) as denied:
        with (patch.object(census, "live", return_value=live),
              patch.object(census, "open_process", side_effect=deny_owned_pid)):
            injected = denied.sample()
    injected["collection_end_qpc_ns"] = raced["collection_end_qpc_ns"]
    require(any(monitor.process_key(row) == key and row["win32_error"] == 87 for row in injected["acquisition_issues"]),
            "The deterministic failing attach was not recorded")
    # The final observation for the injected branch has no retained handle for this identity.
    final_injected = dict(right, **{field:[row for row in right[field] if monitor.process_key(row) != key]
                                  for field in ("live", "terminals")})
    unresolved = monitor.evaluate_active_interval([left, injected, final_injected], opened, closed, 10*SECOND, SECOND,
                                                  owned_roles={self_key:"monitor"})
    require(any(row["identity"] == list(key) and row["stage"] == "missing_exact_terminal"
                for window in unresolved["budget_windows"] for row in window["issues"]),
            "An unregistered identity with no exact exit counter was incorrectly excused")
    reconciled = monitor.evaluate_active_interval([left, injected, final_injected], opened, closed, 10*SECOND, SECOND,
                                                  owned_roles={self_key:"monitor", key:"cohort_child"}, owned_terminal={key:terminal})
    require(not any(row["identity"] == list(key) for window in reconciled["budget_windows"] for row in window["issues"]),
            "The same exact owned terminal ledger did not reconcile the race")
    return dict(identity=list(key), actual_attach_issues=actual_issues, actual_retained_terminal_seen=real_terminal is not None,
                deterministic_injected_attach_failure_win32=87,
                unknown_coverage_failed=True, exact_owned_terminal_reconciled=True)


def wholly_between_samples(census):
    with monitor.LifecycleSampler(census) as sampler:
        before = sampler.sample()
        opened = time.perf_counter_ns()
        p = child(240_000_000)
        try:
            release(p)
            terminal = finish(p, census)
            p._handle.Close()
        finally:
            cleanup(p)
        closed = time.perf_counter_ns()
        after = sampler.sample()
    key = monitor.process_key(terminal)
    require(not any(monitor.process_key(row) == key for sample in (before, after) for field in ("live", "terminals") for row in sample[field]) and
            terminal["kernel_cpu_100ns"]+terminal["user_cpu_100ns"] > monitor.EXTERNAL_CPU_TICKS,
            "The deliberately blind test did not exercise a full above-threshold lifetime between polls")
    result = monitor.evaluate_active_interval([before, after], opened, closed, 10*SECOND, SECOND,
                        owned_roles={monitor.process_key(census.times(census.current_process())):"monitor"})
    require(not result["full_lifecycle_verified"], "Sampling must never imply a complete lifecycle guarantee")
    return dict(identity=list(key), owned_oracle_cpu_100ns=terminal["kernel_cpu_100ns"]+terminal["user_cpu_100ns"],
                not_seen_by_sampler=True, interval_poll_gap_ns=result["max_poll_gap_ns"], full_lifecycle_verified=False)


def synthetic_budget():
    def row(pid, birth, cpu, exit_time=0):
        return dict(pid=pid,birth_filetime_100ns=birth,kernel_cpu_100ns=0,user_cpu_100ns=cpu,exit_filetime_100ns=exit_time)

    def sample(start, live=(), terminals=()):
        return dict(qpc_start_ns=start,qpc_end_ns=start+1,collection_end_qpc_ns=start+2,
                    start_filetime_100ns=start,end_filetime_100ns=start+1,live=live,terminals=terminals,acquisition_issues=[])

    # The second incarnation uses the old PID but must remain external.
    original, reused = (41, 1), (41, 5)
    zero, end = sample(0, [row(*original, 100)]), sample(12*SECOND, [row(*reused, 2_000_001)])
    terminal = row(*original, 700_000, 4)
    inside = sample(10*SECOND+100, [row(*reused, 2_000_001)], [terminal])
    result = monitor.evaluate_active_interval([zero, inside, end], 3, 11*SECOND, 30*SECOND, 11*SECOND,
        owned_roles={original:"cohort_child"}, owned_terminal={original:terminal})
    delta = next(row for row in result["budget_windows"][0]["deltas"] if row["identity"] == list(reused))
    require(delta["role"] == "external_python" and delta["total_delta_100ns"] == 2_000_001 and
            result["external_activity_detected"], "A reused PID inherited cohort exemption")
    missing = monitor.evaluate_active_interval([zero, inside], 3, 11*SECOND, 30*SECOND, 11*SECOND)
    require(missing["reason"] == "missing_boundary_snapshot", "Missing end anchor was accepted")
    gap = monitor.evaluate_active_interval([zero, inside, end], 3, 11*SECOND, 30*SECOND, 1*SECOND)
    require("active_poll_gap" in gap["coverage_issues"], "Over-limit gap was accepted")
    known, self_key = (27, 1), (99, 2)
    pre = [sample(index*10*SECOND,[row(*known,index*5_000_000),row(*self_key,index*10_000_000)]) for index in range(3)]
    windows = [pre[:2], pre[1:]]
    require(monitor.evaluate_quiet_preflight(windows,[5,5],11*SECOND,known_background={known},monitor_key=self_key)["passed"],
            "Exact existing threshold boundary should pass preflight")
    require(not monitor.evaluate_quiet_preflight(windows,[5,6],11*SECOND,known_background={known},monitor_key=self_key)["passed"],
            "An elevated GPU preflight endpoint was accepted")
    noisy_anchor = dict(pre[1],live=[*pre[1]["live"],row(31, 8, 0)])
    require(not monitor.evaluate_quiet_preflight([[pre[0],noisy_anchor],[noisy_anchor,pre[2]]],[0,0],11*SECOND,
                known_background={known},monitor_key=self_key)["passed"], "Unknown zero-CPU identity was allowed during preflight")
    return dict(exact_pid_reuse_rejected=True, missing_end_anchor_rejected=True, over_limit_gap_rejected=True,
                preflight_exact_threshold_passed=True, preflight_gpu_and_unknown_identity_rejected=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("The audit output already exists")
    census = monitor.WindowsCensus()
    result = dict(status="passed", scope="No fluid solves or formal timing windows", windows_build=sys.getwindowsversion().build,
                  source_sha256={name:digest(HERE/name) for name in (Path(__file__).name,Path(monitor.__file__).name)},
                  v7_frozen_window_audit=audit_v7(), etw_permission=etw_empty_capability(), calibration=census.calibrate(),
                  observed_lifetimes=observed_lifetimes(census), post_census_race=disappearing_after_census(census),
                  entirely_between_polls=wholly_between_samples(census), synthetic=synthetic_budget())
    measurements, cpu = [], []
    with monitor.LifecycleSampler(census) as sampler:
        for _ in range(24):
            started_cpu = time.process_time_ns()
            row = sampler.sample()
            cpu.append(time.process_time_ns()-started_cpu)
            measurements.append(row["collection_end_qpc_ns"]-row["qpc_start_ns"])
    result["capture_overhead_ns"] = dict(count=len(measurements), median_wall=median(measurements), maximum_wall=max(measurements),
                                         median_monitor_cpu=median(cpu), maximum_monitor_cpu=max(cpu))
    with args.output.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print("passed")


if __name__ == "__main__":
    main()
