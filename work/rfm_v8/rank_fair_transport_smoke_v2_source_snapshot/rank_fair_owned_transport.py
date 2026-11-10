"""Exact-handle deadline and file-latch protocol for fresh Windows workers."""

import ctypes as C
from ctypes import wintypes as W
import json
import time


SECOND = 1_000_000_000
POLL_SECONDS = .001
KILL_GRACE_MS = 4000
TERMINATION_CODE = 771
WAIT_SIGNALED, WAIT_TIMEOUT = 0, 258


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(record):
    return record["pid"], record["birth_filetime_100ns"]


def read(path):
    return json.loads(path.read_bytes())


def latch(output, index, event):
    return output/f"{event.lower()}_{index:03d}.json"


def publish(path, value):
    """Commit one immutable latch after closing its complete buffered write."""
    temporary = path.with_suffix(".pending")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n").encode("ascii"))
    temporary.rename(path)


def wait_latch(path, deadline_ns):
    while time.perf_counter_ns() < deadline_ns:
        try:
            value = read(path)
        except (FileNotFoundError, PermissionError):
            # Windows can deny opening a just-renamed file until the rename handle closes.
            time.sleep(POLL_SECONDS)
        else:
            return time.perf_counter_ns(), value
    raise TimeoutError(f"Latch deadline exceeded for {path.name}")


def require_identity(value, event, request, request_sha, identity):
    require(value["event"] == event and value["index"] == request["index"] and value["run_token"] == request["run_token"] and
            value["request_sha256"] == request_sha and key(value) == key(identity) and
            value["inherited_affinity_mask"] == request["affinity_mask"], "Latch does not authenticate the exact worker")


class ExactOwned:
    """Observe and, on failure, terminate only the original Popen handle."""

    def __init__(self, process, census):
        self.process = process
        self.census = census
        self.kernel = C.WinDLL("kernel32", use_last_error=True)
        self.wait = self.kernel.WaitForSingleObject
        self.wait.restype, self.wait.argtypes = W.DWORD, (W.HANDLE, W.DWORD)
        self.terminate = self.kernel.TerminateProcess
        self.terminate.restype, self.terminate.argtypes = W.BOOL, (W.HANDLE, W.UINT)

    def wait_ms(self, ms):
        status = self.wait(self.process._handle, ms)
        if status not in (WAIT_SIGNALED, WAIT_TIMEOUT):
            raise C.WinError(C.get_last_error())
        return status

    def wait_until(self, deadline_ns):
        remaining = deadline_ns-time.perf_counter_ns()
        ms = max(0, (remaining+999_999)//1_000_000)
        if self.wait_ms(ms) != WAIT_SIGNALED or time.perf_counter_ns() > deadline_ns:
            raise TimeoutError("Owned child did not exit before the hard deadline")

    def receive(self, path, deadline_ns):
        while time.perf_counter_ns() < deadline_ns:
            try:
                value = read(path)
            except (FileNotFoundError, PermissionError):
                require(self.wait_ms(0) != WAIT_SIGNALED, "Owned child exited before publishing its latch")
                time.sleep(POLL_SECONDS)
            else:
                stamp = time.perf_counter_ns()
                if stamp > deadline_ns:
                    raise TimeoutError("Worker latch arrived after the hard deadline")
                return stamp, value
        raise TimeoutError(f"Latch deadline exceeded for {path.name}")

    def finish(self, identity, watchdog):
        """Run before any inherited log/pipe cleanup, even after setup failures."""
        if self.wait_ms(0) != WAIT_SIGNALED:
            watchdog["termination_attempt_ns"] = time.perf_counter_ns()
            if identity is not None:
                require(key(self.census.times(self.process._handle)) == key(identity), "Owned handle identity changed")
            ok = bool(self.terminate(self.process._handle, TERMINATION_CODE))
            watchdog["terminate_process_ok"] = ok
            if not ok:
                watchdog["terminate_win32_error"] = C.get_last_error()
            watchdog["post_terminate_wait_status"] = self.wait_ms(KILL_GRACE_MS)
            if watchdog["post_terminate_wait_status"] != WAIT_SIGNALED:
                raise TimeoutError("Could not prove owned worker terminated")
        terminal = self.census.times(self.process._handle)
        require(terminal["pid"] == self.process.pid and terminal["exit_filetime_100ns"] > terminal["birth_filetime_100ns"] and
                (identity is None or key(terminal) == key(identity)), "Owned terminal identity differs")
        exit_code = self.process.poll()
        require(exit_code is not None, "Exact exit status is missing")
        return terminal, exit_code
