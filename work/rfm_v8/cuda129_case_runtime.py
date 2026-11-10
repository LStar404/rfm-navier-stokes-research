"""Use exactly the same pinned toolchain loader for Galerkin and FD."""

from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import os
from pathlib import Path
import time

from cuda129_runtime import (
    V7, NVRTC, BUILTINS, CUFFT, NVJIT, VARIANTS, BundledSwitchRuntime, _OwnedDriver, TOOLKIT_DEFAULT,
    I, P, DP, bind, call, nvrtc_load, modern_program, bundled_source, KERNEL_NAMES,
)


class PinnedCaseRuntime(BundledSwitchRuntime):
    """Keep the same driver/graph interface and change only compiler/FFT policy."""

    def __init__(self,toolkit=None,*,variant,source,filename,kernel_names):
        compiler,target,artifact,fft = VARIANTS[variant]
        old = Path(toolkit or os.environ.get("CUDA_PATH",TOOLKIT_DEFAULT))
        source = bytes(source)
        with ExitStack() as resources:
            resources.enter_context(os.add_dll_directory(str(old/"bin")))
            if compiler=="129":
                resources.enter_context(os.add_dll_directory(str(NVRTC.parent)))
                # NVRTC uses a plain name for its matching builtins. Retain an
                # absolute-path load so this process resolves the pinned copy.
                self._nvrtc_builtins = C.WinDLL(str(BUILTINS))
            if fft=="129":
                resources.enter_context(os.add_dll_directory(str(NVJIT.parent)))
                resources.enter_context(os.add_dll_directory(str(CUFFT.parent)))
            self.driver_lib = C.WinDLL("nvcuda.dll")
            self.driver = _OwnedDriver(self.driver_lib,resources)
            filename = bytes(filename)
            if compiler=="legacy":
                start = time.perf_counter()
                self.module = nvrtc_load(old,self.driver,source,filename)
                evidence = dict(nvrtc_version=[11,8],compiler_options=["--gpu-architecture="+target],artifact=artifact,
                                compile_and_module_load_seconds=time.perf_counter()-start)
            else:
                data,evidence = modern_program(source,filename,target,artifact)
                self.module = P()
                start = time.perf_counter()
                call(self.driver.cuModuleLoadData,C.byref(self.module),data)
                evidence["module_load_seconds"] = time.perf_counter()-start
            resources.callback(call,self.driver.cuModuleUnload,self.module)
            self.kernels = {}
            for name in kernel_names:
                kernel = P()
                call(self.driver.cuModuleGetFunction,C.byref(kernel),self.module,name.encode("ascii"))
                self.kernels[name] = kernel
            if fft=="129":
                self._nvjit = C.WinDLL(str(NVJIT))
            fft_file = CUFFT if fft=="129" else old/"bin"/"cufft64_10.dll"
            self.fft = C.WinDLL(str(fft_file))
            self.plan_many = bind(self.fft,"cufftPlanMany",[C.POINTER(I),I,C.POINTER(I),P,I,I,P,I,I,I,I])
            self.set_stream = bind(self.fft,"cufftSetStream",[I,P])
            self.d2z = bind(self.fft,"cufftExecD2Z",[I,DP,DP])
            self.z2d = bind(self.fft,"cufftExecZ2D",[I,DP,DP])
            self.destroy_plan = bind(self.fft,"cufftDestroy",[I])
            fft_version = I()
            call(bind(self.fft,"cufftGetVersion",[C.POINTER(I)]),C.byref(fft_version))
            self.evidence = dict(evidence,variant=variant,cufft_version=fft_version.value,cufft_filename=fft_file.name,
                                 driver_version=self.driver.version,kernel_source_sha256=sha256(source).hexdigest())
            self._resources = resources.pop_all()


class GalerkinCaseRuntime(PinnedCaseRuntime):
    def __init__(self, toolkit=None, *, variant):
        super().__init__(toolkit,variant=variant,source=bundled_source(),
                         filename=b"galerkin_cuda_module_bundle.cu",kernel_names=KERNEL_NAMES)


class FdCaseRuntime(PinnedCaseRuntime):
    def __init__(self, toolkit=None, *, variant):
        names = ("fd_fields","fd_products","fd_rhs","fd_band","fd_dense","fd_observe")
        super().__init__(toolkit,variant=variant,source=(V7/"fd_cuda_native.cu").read_bytes(),
                         filename=b"fd_cuda_native.cu",kernel_names=names)
