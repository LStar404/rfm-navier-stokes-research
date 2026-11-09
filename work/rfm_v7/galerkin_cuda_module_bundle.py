"""Load the existing fixed-step and bridge CUDA kernels as one NVRTC module."""

from contextlib import ExitStack
import ctypes as C
import os
from pathlib import Path

from cuda_environment_native import CHAR, DP, I, P, S, U, Driver, bind, call, nvrtc_load
from galerkin_cuda_native import ROOT, TOOLKIT_DEFAULT, Runtime


KERNEL_NAMES = ("spectrum", "products", "advance", "observe", "advance_bridge", "observe_absolute")


def bundled_source():
    # Independent C-linkage entry points with no translation-unit globals.
    # Preserve both original sources and their diagnostic file names.
    sources = ("galerkin_cuda_native.cu", "galerkin_cuda_variable.cu")
    return b"\n".join(f'#line 1 "{name}"\n'.encode("ascii")+(ROOT/name).read_bytes() for name in sources)


class _OwnedDriver(Driver):
    """Acquire each driver resource under the runtime's teardown stack."""

    def __init__(self, library, resources):
        # The existing Driver constructor does not expose its partially acquired
        # context and stream. Its buffer and graph methods are still reused.
        signatures = {
            "cuInit": [U],
            "cuDriverGetVersion": [C.POINTER(I)],
            "cuDeviceGet": [C.POINTER(I), I],
            "cuDevicePrimaryCtxRetain": [C.POINTER(P), I],
            "cuDevicePrimaryCtxRelease_v2": [I],
            "cuCtxGetCurrent": [C.POINTER(P)],
            "cuCtxSetCurrent": [P],
            "cuStreamCreate": [C.POINTER(P), U],
            "cuStreamDestroy_v2": [P],
            "cuStreamSynchronize": [P],
            "cuMemAlloc_v2": [C.POINTER(DP), S],
            "cuMemFree_v2": [DP],
            "cuMemcpyHtoD_v2": [DP, P, S],
            "cuMemcpyDtoH_v2": [P, DP, S],
            "cuModuleLoadData": [C.POINTER(P), P],
            "cuModuleUnload": [P],
            "cuModuleGetFunction": [C.POINTER(P), P, CHAR],
            "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, P, P],
            "cuStreamBeginCapture": [P, I],
            "cuStreamEndCapture": [P, C.POINTER(P)],
            "cuGraphInstantiateWithFlags": [C.POINTER(P), P, C.c_uint64],
            "cuGraphLaunch": [P, P],
            "cuGraphExecDestroy": [P],
            "cuGraphDestroy": [P],
        }
        for name, args in signatures.items():
            setattr(self, name, bind(library, name, args))
        call(self.cuInit, 0)
        device, version = I(), I()
        previous, context = P(), P()
        self.stream = P()
        call(self.cuDriverGetVersion, C.byref(version))
        self.version = version.value
        call(self.cuDeviceGet, C.byref(device), 0)
        call(self.cuCtxGetCurrent, C.byref(previous))
        call(self.cuDevicePrimaryCtxRetain, C.byref(context), device)
        resources.callback(call, self.cuDevicePrimaryCtxRelease_v2, device)
        resources.callback(call, self.cuCtxSetCurrent, previous)
        call(self.cuCtxSetCurrent, context)
        call(self.cuStreamCreate, C.byref(self.stream), 1)
        resources.callback(call, self.cuStreamDestroy_v2, self.stream)


class BundledSwitchRuntime(Runtime):
    """Runtime-compatible single-module setup with a fresh context retain per solve."""

    def __init__(self, toolkit=None):
        root = Path(toolkit or os.environ.get("CUDA_PATH", TOOLKIT_DEFAULT))
        source = bundled_source()
        with ExitStack() as resources:
            resources.enter_context(os.add_dll_directory(str(root/"bin")))
            self.driver_lib = C.WinDLL("nvcuda.dll")
            self.driver = _OwnedDriver(self.driver_lib, resources)
            self.module = nvrtc_load(root, self.driver, source, b"galerkin_cuda_module_bundle.cu")
            resources.callback(call, self.driver.cuModuleUnload, self.module)
            self.kernels = {}
            for name in KERNEL_NAMES:
                kernel = P()
                call(self.driver.cuModuleGetFunction, C.byref(kernel), self.module, name.encode("ascii"))
                self.kernels[name] = kernel

            self.fft = C.WinDLL(str(root/"bin"/"cufft64_10.dll"))
            self.plan_many = bind(self.fft, "cufftPlanMany",
                                  [C.POINTER(I), I, C.POINTER(I), P, I, I, P, I, I, I, I])
            self.set_stream = bind(self.fft, "cufftSetStream", [I, P])
            self.d2z = bind(self.fft, "cufftExecD2Z", [I, DP, DP])
            self.z2d = bind(self.fft, "cufftExecZ2D", [I, DP, DP])
            self.destroy_plan = bind(self.fft, "cufftDestroy", [I])
            self._resources = resources.pop_all()

    def __exit__(self, *exc):
        # A failed synchronization must still attempt every teardown step.
        with self._resources:
            self.sync()
