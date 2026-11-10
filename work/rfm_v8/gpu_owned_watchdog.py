"""Windows wall-clock guard for a single owned diagnostic subprocess."""

import ctypes as C
from ctypes import wintypes as W
import subprocess
import time

from win32_isolation_monitor import WindowsCensus, bind, process_key


WAIT_SIGNALED = 0
WAIT_TIMEOUT = 258
TERMINATION_CODE = 771


def run_owned(command, output, *, deadline_seconds, kill_wait_seconds=4):
    if not 0 < deadline_seconds <= 30 or not 0 < kill_wait_seconds <= 5:
        raise ValueError("GPU probes require a hard deadline of at most 30 seconds")
    kernel = C.WinDLL("kernel32",use_last_error=True)
    wait = bind(kernel,"WaitForSingleObject",W.DWORD,W.HANDLE,W.DWORD)
    terminate = bind(kernel,"TerminateProcess",W.BOOL,W.HANDLE,W.UINT)
    exit_code = bind(kernel,"GetExitCodeProcess",W.BOOL,W.HANDLE,C.POINTER(W.DWORD))
    census = WindowsCensus()
    output.mkdir()
    result = dict(status="failed",deadline_seconds=deadline_seconds,kill_wait_seconds=kill_wait_seconds)
    process = None
    with ((output/"stdout.log").open("xb") as stdout,(output/"stderr.log").open("xb") as stderr):
        begin = time.perf_counter_ns()
        cutoff = begin+round(deadline_seconds*1e9)
        try:
            process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            identity = census.times(process._handle)
            if identity["pid"] != process.pid:
                raise RuntimeError("Popen identity differs from its actual handle")
            result["identity"] = identity
            remaining_ms = max(0,(cutoff-time.perf_counter_ns()+999_999)//1_000_000)
            status = wait(process._handle,remaining_ms)
            result["first_wait_status"] = status
            if status not in (WAIT_SIGNALED,WAIT_TIMEOUT):
                raise C.WinError(C.get_last_error())
            if status==WAIT_SIGNALED and time.perf_counter_ns()<=cutoff:
                result["status"] = "complete"
            else:
                result["status"] = "deadline_exceeded"
                result["termination_attempt_ns"] = time.perf_counter_ns()
                now = census.times(process._handle)
                if process_key(now)!=process_key(identity):
                    raise RuntimeError("Owned handle identity changed")
                if not now["exit_filetime_100ns"]:
                    killed = bool(terminate(process._handle,TERMINATION_CODE))
                    result["terminate_process_ok"] = killed
                    if not killed:
                        result["terminate_win32_error"] = C.get_last_error()
                result["post_terminate_wait_status"] = wait(process._handle,round(kill_wait_seconds*1000))
                if result["post_terminate_wait_status"]!=WAIT_SIGNALED:
                    result["status"] = "termination_unconfirmed"
            if result["status"]!="termination_unconfirmed":
                terminal = census.times(process._handle)
                if process_key(terminal)!=process_key(identity) or not terminal["exit_filetime_100ns"]:
                    raise RuntimeError("Exact terminal handle identity is missing")
                result["terminal_identity"] = terminal
                code=W.DWORD()
                if not exit_code(process._handle,C.byref(code)):
                    raise C.WinError(C.get_last_error())
                process.returncode=code.value
                result["exit_code"]=code.value
                if code.value and result["status"]=="complete":
                    result["status"]="child_failed"
        except BaseException as error:
            result.update(status="watchdog_error",error_type=type(error).__name__)
            if process is not None:
                # The original Popen handle stays owned even when reading its identity fails.
                if wait(process._handle,0)!=WAIT_SIGNALED:
                    result["exception_terminate_ok"]=bool(terminate(process._handle,TERMINATION_CODE))
                    result["exception_post_terminate_wait_status"]=wait(process._handle,round(kill_wait_seconds*1000))
                    if result["exception_post_terminate_wait_status"]!=WAIT_SIGNALED:
                        result["status"]="termination_unconfirmed"
            if isinstance(error,(KeyboardInterrupt,SystemExit)):
                result["interrupted"]=True
        finally:
            if process is not None:
                if process.returncode is None:
                    process._child_created=False
                process._handle.Close()
            result["wall_seconds"]=(time.perf_counter_ns()-begin)/1e9
    return result
