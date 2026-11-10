"""Prospective Windows Python-process monitor with active cohort boundaries.

Single-query live identity and CPU counts are guarded by an x64 ABI check.
Exact query handles provide terminal counts for observed lifetimes. Entirely
between-sample lifetimes remain outside the sampling guarantee.
"""

import ctypes as C
from ctypes import wintypes as W
import ntpath
import os
import time


TICK_NS = 100
BUDGET_INTERVAL_NS = 10_000_000_000
MAX_ANCHOR_LATENESS_NS = 2_000_000_000
EXTERNAL_CPU_TICKS = 2_000_000
KNOWN_INDIVIDUAL_CPU_TICKS = 5_000_000
KNOWN_TOTAL_CPU_TICKS = 10_000_000


class UnicodeString(C.Structure):
    _fields_ = [("length", W.USHORT), ("maximum_length", W.USHORT), ("buffer", C.c_void_p)]


class ProcessPrefix(C.Structure):
    # Windows x64 SystemProcessInformation prefix. The public header reserves
    # the 48 bytes after NumberOfThreads. Guard the three used values against
    # GetProcessTimes on an owned live process before arming this collector.
    _fields_ = [("next_entry", W.ULONG), ("threads", W.ULONG), ("reserved", C.c_byte*24),
                ("birth", C.c_int64), ("user", C.c_int64), ("kernel", C.c_int64),
                ("image", UnicodeString), ("priority", W.LONG), ("pid", W.HANDLE), ("parent_pid", W.HANDLE)]


def bind(library, name, result, *args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = result
    return function


def process_key(record):
    return record["pid"], record["birth_filetime_100ns"]


def filetime(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


class WindowsCensus:
    """Capture one process list without a name/identity OpenProcess race."""

    def __init__(self):
        if os.name != "nt" or C.sizeof(C.c_void_p) != 8 or (ProcessPrefix.birth.offset, ProcessPrefix.image.offset,
                ProcessPrefix.pid.offset, C.sizeof(ProcessPrefix)) != (32, 56, 80, 96):
            raise OSError("Unsupported SystemProcessInformation layout")
        kernel = C.WinDLL("kernel32", use_last_error=True)
        native = C.WinDLL("ntdll")
        self.query = bind(native, "NtQuerySystemInformation", W.LONG, W.ULONG, C.c_void_p, W.ULONG, C.POINTER(W.ULONG))
        self.get_times = bind(kernel, "GetProcessTimes", W.BOOL, W.HANDLE, *([C.POINTER(W.FILETIME)]*4))
        self.get_pid = bind(kernel, "GetProcessId", W.DWORD, W.HANDLE)
        self.current_process = bind(kernel, "GetCurrentProcess", W.HANDLE)
        self.open_process = bind(kernel, "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
        self.close_handle = bind(kernel, "CloseHandle", W.BOOL, W.HANDLE)
        self.precise_clock = bind(kernel, "GetSystemTimePreciseAsFileTime", None, C.POINTER(W.FILETIME))
        self.buffer = C.create_string_buffer(1024*1024)
        self.calibration = None

    def clock_filetime(self):
        clock = W.FILETIME()
        self.precise_clock(C.byref(clock))
        return filetime(clock)

    def times(self, handle):
        pid = self.get_pid(handle)
        if not pid:
            raise C.WinError(C.get_last_error())
        birth, exit_time, kernel, user = (W.FILETIME() for _ in range(4))
        if not self.get_times(handle, C.byref(birth), C.byref(exit_time), C.byref(kernel), C.byref(user)):
            raise C.WinError(C.get_last_error())
        return dict(pid=int(pid), birth_filetime_100ns=filetime(birth), exit_filetime_100ns=filetime(exit_time),
                    kernel_cpu_100ns=filetime(kernel), user_cpu_100ns=filetime(user))

    def live(self):
        length = W.ULONG()
        begin = time.perf_counter_ns()
        begin_filetime = self.clock_filetime()
        while True:
            status = self.query(5, self.buffer, C.sizeof(self.buffer), C.byref(length)) & 0xffffffff
            if status == 0:
                break
            new_size = max(2*C.sizeof(self.buffer), length.value)
            if status != 0xc0000004 or new_size > 64*1024*1024:
                raise OSError(f"SystemProcessInformation failed with NTSTATUS 0x{status:08x}")
            self.buffer = C.create_string_buffer(new_size)
        end_filetime = self.clock_filetime()
        end = time.perf_counter_ns()
        if not 96 <= length.value <= C.sizeof(self.buffer):
            raise RuntimeError("Native process census returned an invalid byte count")
        rows = {}
        base = C.addressof(self.buffer)
        offset = 0
        while True:
            if offset+96 > length.value:
                raise RuntimeError("Native process census escaped its record bounds")
            entry = ProcessPrefix.from_buffer(self.buffer, offset)
            image = entry.image
            if image.length:
                if (image.length % 2 or image.length > image.maximum_length or not image.buffer or
                        not base <= image.buffer <= base+length.value-image.length):
                    raise RuntimeError("Native process census contains an invalid image range")
                name = ntpath.basename(C.wstring_at(image.buffer, image.length//2)).casefold()
                if name.startswith("python"):
                    if not entry.pid or entry.birth <= 0 or min(entry.kernel, entry.user) < 0:
                        raise RuntimeError("Native Python process identity or counters are invalid")
                    row = dict(pid=entry.pid, parent_pid=entry.parent_pid or 0, birth_filetime_100ns=entry.birth,
                               kernel_cpu_100ns=entry.kernel, user_cpu_100ns=entry.user, exit_filetime_100ns=0)
                    key = process_key(row)
                    if key in rows:
                        raise RuntimeError("Duplicate exact identity in the native census")
                    rows[key] = row
            if not entry.next_entry:
                break
            if entry.next_entry < 96 or entry.next_entry % 8:
                raise RuntimeError("Native process census contains an invalid next offset")
            offset += entry.next_entry
        return dict(qpc_start_ns=begin, qpc_end_ns=end, start_filetime_100ns=begin_filetime,
                    end_filetime_100ns=end_filetime, live=sorted(rows.values(), key=process_key))

    def calibrate(self, handle=None):
        handle = self.current_process() if handle is None else handle
        before = self.times(handle)
        census = self.live()
        after = self.times(handle)
        matches = [row for row in census["live"] if process_key(row) == process_key(before)]
        counters = ("kernel_cpu_100ns", "user_cpu_100ns")
        if (len(matches) != 1 or process_key(after) != process_key(before) or before["exit_filetime_100ns"] or
                after["exit_filetime_100ns"] or not all(before[key] <= matches[0][key] <= after[key] for key in counters)):
            raise RuntimeError("Native process layout did not agree with the owned Win32 handle")
        self.calibration = dict(identity_and_counters_passed=True, before=before, native=matches[0], after=after)
        return self.calibration


class LifecycleSampler:
    """Retain exact query handles and record every unresolved acquisition."""

    def __init__(self, census):
        if census.calibration is None:
            raise ValueError("Calibrate the native census before sampling")
        self.census = census
        self.retained = {}
        self.exited = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle, _ in self.retained.values():
            self.census.close_handle(handle)
        self.retained.clear()

    def sample(self):
        snapshot = self.census.live()
        terminals, issues = [], []
        live = {process_key(row): row for row in snapshot["live"]}
        for key, row in live.items():
            if key in self.exited:
                issues.append(dict(pid=key[0], birth_filetime_100ns=key[1], stage="seen_live_after_terminal"))
                continue
            if key in self.retained:
                continue
            handle = self.census.open_process(0x1000, False, key[0])
            if not handle:
                issues.append(dict(pid=key[0], birth_filetime_100ns=key[1], stage="terminal_open", win32_error=C.get_last_error()))
                continue
            try:
                observed = self.census.times(handle)
                if process_key(observed) != key:
                    issues.append(dict(pid=key[0], birth_filetime_100ns=key[1], stage="identity_changed_before_handle"))
                    self.census.close_handle(handle)
                else:
                    self.retained[key] = (handle, row["parent_pid"])
            except OSError as error:
                self.census.close_handle(handle)
                issues.append(dict(pid=key[0], birth_filetime_100ns=key[1], stage="terminal_handle_query", win32_error=error.winerror))
        for key, (handle, parent) in list(self.retained.items()):
            if key in live:
                continue
            try:
                observed = self.census.times(handle)
                if process_key(observed) != key or not observed["exit_filetime_100ns"]:
                    issues.append(dict(pid=key[0], birth_filetime_100ns=key[1], stage="retained_not_terminal_after_census"))
                    continue
                terminals.append(dict(observed, parent_pid=parent))
                self.census.close_handle(handle)
                del self.retained[key]
                self.exited.add(key)
            except OSError as error:
                issues.append(dict(pid=key[0], birth_filetime_100ns=key[1], stage="retained_terminal_query", win32_error=error.winerror))
        return dict(snapshot, terminals=sorted(terminals, key=process_key), acquisition_issues=issues,
                    collection_end_qpc_ns=time.perf_counter_ns())


def classify_budget_window(samples, known_background, owned_roles, owned_terminal):
    """Assess one old-policy budget interval using exact cumulative counters."""
    left = {process_key(row): row for row in samples[0]["live"]}
    right = {process_key(row): row for row in samples[-1]["live"]}
    latest = {process_key(row): row for sample in samples for row in sample["live"]}
    terminals = {process_key(row): row for sample in samples[1:] for row in sample["terminals"]}
    issues, deltas = [], []
    for key in sorted(latest.keys() | terminals.keys()):
        owned = owned_terminal.get(key)
        terminal = terminals.get(key)
        if owned and terminal and any(owned[name] != terminal[name] for name in (
                "exit_filetime_100ns", "kernel_cpu_100ns", "user_cpu_100ns")):
            issues.append(dict(identity=list(key), stage="owned_terminal_disagrees"))
        if key not in right:
            final = terminal or owned
        else:
            final = right[key]
        lower_bound_only = final is None
        if lower_bound_only:
            issues.append(dict(identity=list(key), stage="missing_exact_terminal"))
            final = latest[key]
        first = left.get(key, {})
        kernel = final["kernel_cpu_100ns"]-first.get("kernel_cpu_100ns", 0)
        user = final["user_cpu_100ns"]-first.get("user_cpu_100ns", 0)
        role = owned_roles.get(key, "known_background" if key in known_background else "external_python")
        if min(kernel, user) < 0:
            issues.append(dict(identity=list(key), stage="cumulative_counter_regression"))
        deltas.append(dict(identity=list(key), role=role, kernel_delta_100ns=kernel, user_delta_100ns=user,
                           total_delta_100ns=kernel+user, terminal=key not in right and not lower_bound_only,
                           lower_bound_only=lower_bound_only))
    known = [row["total_delta_100ns"] for row in deltas if row["role"] == "known_background"]
    external = [row["total_delta_100ns"] for row in deltas if row["role"] == "external_python"]
    activity = (any(value > EXTERNAL_CPU_TICKS for value in external) or
                any(value > KNOWN_INDIVIDUAL_CPU_TICKS for value in known) or sum(known) > KNOWN_TOTAL_CPU_TICKS)
    return dict(deltas=deltas, issues=issues, external_activity_detected=activity)


def evaluate_quiet_preflight(windows, gpu_percent, max_poll_gap_ns, *, known_background, monitor_key):
    """Check both predeclared ten-second guards before any coordinator launch."""
    known = set(known_background)
    if len(windows) != 2 or len(gpu_percent) != 2 or monitor_key in known or max_poll_gap_ns <= 0:
        raise ValueError("Invalid preflight identity, cadence or guard count")
    if not all(windows) or windows[0][-1] != windows[1][0]:
        raise ValueError("Preflight guards must share their intervening anchor")
    verdicts = []
    for segment, gpu in zip(windows, gpu_percent, strict=True):
        sampled = {process_key(row) for sample in segment for field in ("live", "terminals") for row in sample[field]}
        gaps = [b["qpc_start_ns"]-a["qpc_start_ns"] for a, b in zip(segment, segment[1:])]
        budget = classify_budget_window(segment, known, {monitor_key:"monitor"}, {})
        elapsed = segment[-1]["qpc_start_ns"]-segment[0]["qpc_start_ns"]
        unknown = sorted(sampled - known - {monitor_key})
        coverage = (bool(gaps) and min(gaps) > 0 and max(gaps) <= max_poll_gap_ns and
                    BUDGET_INTERVAL_NS <= elapsed <= BUDGET_INTERVAL_NS+MAX_ANCHOR_LATENESS_NS and not budget["issues"] and
                    all(known | {monitor_key} <= {process_key(row) for row in sample["live"]} for sample in segment))
        verdicts.append(dict(coverage_pass=coverage, unknown_observed=[list(key) for key in unknown], gpu_percent=gpu,
                             budget=budget, passed=bool(coverage and not unknown and type(gpu) is int and 0 <= gpu <= 5 and
                                                        not budget["external_activity_detected"])))
    return dict(passed=all(row["passed"] for row in verdicts), windows=verdicts)


def evaluate_active_interval(samples, opened_ns, closed_ns, max_duration_ns, max_poll_gap_ns, *,
                             known_background=(), owned_roles=None, owned_terminal=None):
    """Bracket immutable first-launch and final-close markers without an idle tail.

    The caller must supply independently verified exact roles and terminal
    handle records. This routine can support a sampled verdict only.
    """
    owned_roles = {} if owned_roles is None else owned_roles
    owned_terminal = {} if owned_terminal is None else owned_terminal
    known_background = frozenset(known_background)
    if (not 0 < closed_ns-opened_ns <= max_duration_ns or max_poll_gap_ns <= 0 or
            known_background.intersection(owned_roles) or
            any(role not in ("cohort_coordinator", "cohort_child", "monitor") for role in owned_roles.values()) or
            any(key not in owned_roles or process_key(row) != key or row["exit_filetime_100ns"] <= key[1]
                for key, row in owned_terminal.items())):
        raise ValueError("Invalid prospective interval or owned identity ledger")
    left = [i for i, row in enumerate(samples) if row["collection_end_qpc_ns"] <= opened_ns]
    right = [i for i, row in enumerate(samples) if row["qpc_start_ns"] >= closed_ns]
    if not left or not right:
        return dict(status="coverage_incomplete", reason="missing_boundary_snapshot", full_lifecycle_verified=False)
    active = samples[left[-1]:right[0]+1]
    issues = []
    if any(not row["qpc_start_ns"] <= row["qpc_end_ns"] <= row["collection_end_qpc_ns"] for row in active):
        issues.append("invalid_sample_chronology")
    if any(a["collection_end_qpc_ns"] > b["qpc_start_ns"] for a, b in zip(active, active[1:])):
        issues.append("overlapping_sample_chronology")
    gaps = [later["qpc_start_ns"]-earlier["qpc_start_ns"] for earlier, later in zip(active, active[1:])]
    if not gaps or min(gaps) <= 0 or max(gaps) > max_poll_gap_ns:
        issues.append("active_poll_gap")
    if (any(row["end_filetime_100ns"] < row["start_filetime_100ns"] for row in active) or
            any(a["end_filetime_100ns"] > b["start_filetime_100ns"] for a, b in zip(active, active[1:]))):
        issues.append("wall_clock_regression")
    if any(issue["stage"] == "seen_live_after_terminal" for row in active for issue in row["acquisition_issues"]):
        issues.append("inconsistent_terminal_chronology")
    observed = {}
    for sample in active:
        identities = set()
        for row in (*sample["live"], *sample["terminals"]):
            key = process_key(row)
            previous = observed.get(key)
            if (key in identities or previous is not None and
                    (previous["exit_filetime_100ns"] or any(row[field] < previous[field] for field in (
                         "kernel_cpu_100ns", "user_cpu_100ns")))):
                issues.append("invalid_identity_counter_history")
            identities.add(key)
            observed[key] = row
    targets = range(opened_ns+BUDGET_INTERVAL_NS, closed_ns, BUDGET_INTERVAL_NS)
    anchors = [0]
    for target in targets:
        candidates = [i for i in range(anchors[-1]+1, len(active)) if active[i]["qpc_start_ns"] >= target]
        if not candidates or active[candidates[0]]["qpc_end_ns"]-target > MAX_ANCHOR_LATENESS_NS:
            issues.append("late_budget_anchor")
            break
        anchors.append(candidates[0])
    if anchors[-1] != len(active)-1:
        anchors.append(len(active)-1)
    windows = [classify_budget_window(active[a:b+1], known_background, owned_roles, owned_terminal)
               for a, b in zip(anchors, anchors[1:])]
    coverage = not issues and all(not window["issues"] for window in windows)
    activity = any(window["external_activity_detected"] for window in windows)
    status = "coverage_incomplete" if not coverage else "observed_external_activity" if activity else "no_sampled_external_activity"
    return dict(status=status, coverage_pass=coverage, external_activity_detected=activity, full_lifecycle_verified=False,
                active_samples=len(active), max_poll_gap_ns=max(gaps, default=0), budget_anchors=anchors,
                coverage_issues=issues, budget_windows=windows)
