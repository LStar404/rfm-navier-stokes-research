"""Exact Windows identity for the prepared timing coordinator and its children."""

import ctypes as C
from ctypes import wintypes as W


_kernel32 = C.WinDLL("kernel32", use_last_error=True)
_get_times = _kernel32.GetProcessTimes
_get_times.argtypes = [W.HANDLE, *([C.POINTER(W.FILETIME)]*4)]
_get_times.restype = W.BOOL
_get_pid = _kernel32.GetProcessId
_get_pid.argtypes = [W.HANDLE]
_get_pid.restype = W.DWORD
_get_current = _kernel32.GetCurrentProcess
_get_current.argtypes = []
_get_current.restype = W.HANDLE
_open_process = _kernel32.OpenProcess
_open_process.argtypes = [W.DWORD, W.BOOL, W.DWORD]
_open_process.restype = W.HANDLE
_close_handle = _kernel32.CloseHandle
_close_handle.argtypes = [W.HANDLE]
_close_handle.restype = W.BOOL


def _filetime_100ns(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def identity_from_handle(handle):
    """Read the birth and cumulative CPU counters from the same open handle."""
    handle = W.HANDLE(handle)
    pid = _get_pid(handle)
    if not pid:
        raise C.WinError(C.get_last_error())
    birth, exit_time, kernel, user = (W.FILETIME() for _ in range(4))
    if not _get_times(handle, C.byref(birth), C.byref(exit_time), C.byref(kernel), C.byref(user)):
        raise C.WinError(C.get_last_error())
    return dict(pid=int(pid), birth_filetime_100ns=_filetime_100ns(birth), exit_filetime_100ns=_filetime_100ns(exit_time),
                kernel_cpu_100ns=_filetime_100ns(kernel), user_cpu_100ns=_filetime_100ns(user))


def current_process_identity():
    return identity_from_handle(_get_current())


def identity_from_pid(pid):
    """Resolve a configured background PID to an exact, freezeable identity."""
    handle = _open_process(0x1000, False, pid)
    if not handle:
        raise C.WinError(C.get_last_error())
    try:
        return identity_from_handle(handle)
    finally:
        _close_handle(handle)
