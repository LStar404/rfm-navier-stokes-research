"""Ablate the CUDA compiler target and FFT library without changing CUDA source."""

from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import os
from pathlib import Path
import time

from galerkin_minimal_model import V7
from cuda_environment_native import CHAR, DP, I, P, S, bind, call, nvrtc_load
from galerkin_cuda_module_bundle import BundledSwitchRuntime, KERNEL_NAMES, _OwnedDriver, bundled_source
from galerkin_cuda_native import TOOLKIT_DEFAULT


HERE = Path(__file__).resolve().parent
VENDOR = HERE/"vendor"/"cuda_12_9_1"
NVRTC = VENDOR/"nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll"
BUILTINS = NVRTC.with_name("nvrtc-builtins64_129.dll")
CUFFT = VENDOR/"nvidia/cufft/bin/cufft64_11.dll"
NVJIT = VENDOR/"nvidia/nvjitlink/bin/nvJitLink_120_0.dll"
VARIANTS = {
    "legacy":("legacy","compute_52","ptx","legacy"),
    "compiler129_compute52":("129","compute_52","ptx","legacy"),
    "compiler129_compute120":("129","compute_120","ptx","legacy"),
    "compiler129_sm120":("129","sm_120","cubin","legacy"),
    "fft129_only":("legacy","compute_52","ptx","129"),
    "compiler129_sm120_fft129":("129","sm_120","cubin","129"),
}


def modern_program(source, name, target, artifact):
    lib = C.WinDLL(str(NVRTC))
    ver,minor,count = I(),I(),I()
    call(bind(lib,"nvrtcVersion",[C.POINTER(I),C.POINTER(I)]),C.byref(ver),C.byref(minor))
    call(bind(lib,"nvrtcGetNumSupportedArchs",[C.POINTER(I)]),C.byref(count))
    arches = (I*count.value)()
    call(bind(lib,"nvrtcGetSupportedArchs",[C.POINTER(I)]),arches)
    create = bind(lib,"nvrtcCreateProgram",[C.POINTER(P),CHAR,CHAR,I,P,P])
    compile_ = bind(lib,"nvrtcCompileProgram",[P,I,C.POINTER(CHAR)])
    destroy = bind(lib,"nvrtcDestroyProgram",[C.POINTER(P)])
    log_size = bind(lib,"nvrtcGetProgramLogSize",[P,C.POINTER(S)])
    log_data = bind(lib,"nvrtcGetProgramLog",[P,P])
    kind = "PTX" if artifact=="ptx" else "CUBIN"
    size_for = bind(lib,"nvrtcGet"+kind+"Size",[P,C.POINTER(S)])
    data_for = bind(lib,"nvrtcGet"+kind,[P,P])
    options = ["--gpu-architecture="+target]
    option_array = (CHAR*len(options))(*(s.encode("ascii") for s in options))
    program = P()
    call(create,C.byref(program),source,name,0,None,None)
    try:
        t = time.perf_counter()
        status = compile_(program,len(options),option_array)
        elapsed = time.perf_counter()-t
        log_n = S()
        call(log_size,program,C.byref(log_n))
        log = C.create_string_buffer(log_n.value)
        call(log_data,program,log)
        if status:
            raise RuntimeError(f"NVRTC {target} returned {status}: {log.value.decode('utf-8','replace')}")
        n = S()
        call(size_for,program,C.byref(n))
        if n.value == 0:
            raise RuntimeError(f"Empty {artifact} for {target}")
        data = C.create_string_buffer(n.value)
        call(data_for,program,data)
    finally:
        call(destroy,C.byref(program))
    return data,dict(nvrtc_version=[ver.value,minor.value],nvrtc_architectures=list(arches),
                     compiler_options=options,artifact=artifact,artifact_bytes=n.value,
                     artifact_sha256=sha256(data.raw).hexdigest(),compile_call_seconds=elapsed,
                     compile_log=log.value.decode("utf-8","replace"))


class PinnedBundledRuntime(BundledSwitchRuntime):
    """Keep the same driver/graph interface and change only compiler/FFT policy."""

    def __init__(self,toolkit=None,*,variant):
        compiler,target,artifact,fft = VARIANTS[variant]
        old = Path(toolkit or os.environ.get("CUDA_PATH",TOOLKIT_DEFAULT))
        source = bundled_source()
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
            filename = b"galerkin_cuda_module_bundle.cu"
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
            for name in KERNEL_NAMES:
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
                                 driver_version=self.driver.version,bundled_source_sha256=sha256(source).hexdigest())
            self._resources = resources.pop_all()
