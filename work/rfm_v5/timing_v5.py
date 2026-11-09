"""Pin benchmark processes to the same Windows logical CPU as v4."""
import ctypes


def pin():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    process=kernel.GetCurrentProcess()
    current,system=ctypes.c_size_t(),ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process,ctypes.byref(current),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    available=[i for i in range(current.value.bit_length()) if current.value&(1<<i)]
    cpu=available[min(4,len(available)-1)]
    if not kernel.SetProcessAffinityMask(process,1<<cpu):
        raise ctypes.WinError(ctypes.get_last_error())
    return cpu
