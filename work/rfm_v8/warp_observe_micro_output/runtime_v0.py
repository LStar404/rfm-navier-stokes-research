"""FP64 warp-per-probe observation kernels using the original solver interfaces."""

from contextlib import ExitStack
import ctypes as C
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
sys.path.insert(0,str(V7))

from cuda_environment_native import DP, I, P, bind, call, nvrtc_load
from galerkin_cuda_module_bundle import KERNEL_NAMES, _OwnedDriver, bundled_source
from galerkin_cuda_native import Launch, Runtime, TOOLKIT_DEFAULT, solve_native
from fd_cuda_native import solve_fd_native


UNIFORM = ("spectrum","products","advance","observe")
FD = ("fd_fields","fd_products","fd_rhs","fd_band","fd_dense","fd_observe")
OBSERVERS = {"observe","observe_absolute","fd_observe"}


def replace_observer(source, name):
    text = source.decode("utf-8")
    signature = f'extern "C" __global__ void {name}('
    if text.count(signature) != 1:
        raise ValueError(f"Expected exactly one {name} definition")
    absolute = name == "observe_absolute"
    predicate = "sample >= 0" if absolute else "n % stride == 0"
    index = "sample" if absolute else "n / stride"
    signature_start = text.index(signature)
    begin = text.index(f"    if ({predicate} && threadIdx.x < 6) {{",signature_start)
    end = text.index("    __syncthreads();",begin)
    old = text[begin:end]
    if old.count("value += weights[i] * source[i].x") != 1 or old.count("probe_lift[p]") != 1:
        raise ValueError("Frozen probe reduction no longer matches the replacement")
    replacement = f'''    const int p = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if ({predicate} && p < 6) {{
        const double *weights = probe_operator + p * frame;
        double value = 0.;
        for (int i = lane; i < frame; i += 32) value += weights[i] * source[i].x;
        for (int offset = 16; offset > 0; offset /= 2)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) probes[({index}) * 6 + p] = value + amplitudes[n].x * probe_lift[p];
    }}
'''
    return (text[:begin]+replacement+text[end:]).encode("utf-8")


def source_and_names(family, warp):
    if family == "uniform":
        source, names, observer = (V7/"galerkin_cuda_native.cu").read_bytes(),UNIFORM,"observe"
    elif family == "combined":
        source, names, observer = bundled_source(),KERNEL_NAMES,"observe_absolute"
    elif family == "fd":
        source, names, observer = (V7/"fd_cuda_native.cu").read_bytes(),FD,"fd_observe"
    else:
        raise ValueError("Unknown observation solver family")
    return (replace_observer(source,observer) if warp else source),names,observer


class WarpLaunch(Launch):
    def __call__(self):
        call(self.driver.cuLaunchKernel,self.kernel,1,1,1,192,1,1,0,self.driver.stream,self.params,None)


class ObservationRuntime(Runtime):
    def __init__(self,family,warp,toolkit=None):
        root = Path(toolkit or os.environ.get("CUDA_PATH",TOOLKIT_DEFAULT))
        source,names,self.observer = source_and_names(family,warp)
        self.warp = warp
        with ExitStack() as resources:
            resources.enter_context(os.add_dll_directory(str(root/"bin")))
            self.driver_lib = C.WinDLL("nvcuda.dll")
            self.driver = _OwnedDriver(self.driver_lib,resources)
            self.module = nvrtc_load(root,self.driver,source,b"observation_diagnostic.cu")
            resources.callback(call,self.driver.cuModuleUnload,self.module)
            self.kernels = {}
            for name in names:
                handle = P()
                call(self.driver.cuModuleGetFunction,C.byref(handle),self.module,name.encode("ascii"))
                self.kernels[name] = handle
            self.fft = C.WinDLL(str(root/"bin"/"cufft64_10.dll"))
            self.plan_many = bind(self.fft,"cufftPlanMany",[C.POINTER(I),I,C.POINTER(I),P,I,I,P,I,I,I,I])
            self.set_stream = bind(self.fft,"cufftSetStream",[I,P])
            self.d2z = bind(self.fft,"cufftExecD2Z",[I,DP,DP])
            self.z2d = bind(self.fft,"cufftExecZ2D",[I,DP,DP])
            self.destroy_plan = bind(self.fft,"cufftDestroy",[I])
            self.resources = resources.pop_all()

    def launch(self,name,count,*args):
        if self.warp and name == self.observer:
            if count != 1:
                raise ValueError("Observe must stay a single block")
            return WarpLaunch(self.driver,self.kernels[name],count,*args)
        return super().launch(name,count,*args)

    def __exit__(self,*exc):
        with self.resources:
            self.sync()


def solve_with_observation_runtime(family,warp,*,toolkit=None,**settings):
    """Uniform and FD adapters. Include runtime creation and teardown in the outer clock."""
    if family not in ("uniform","fd"):
        raise ValueError("Use the generated same-march adapter for combined Galerkin")
    started = time.perf_counter()
    with ObservationRuntime(family,warp,toolkit) as rt:
        initialized = time.perf_counter()-started
        solver = solve_native if family == "uniform" else solve_fd_native
        solution = solver(runtime=rt,toolkit=toolkit,**settings)
    solution.row.update(runtime_init_seconds=initialized,seconds=time.perf_counter()-started,
                        observation_reduction="warp" if warp else "serial")
    return solution
