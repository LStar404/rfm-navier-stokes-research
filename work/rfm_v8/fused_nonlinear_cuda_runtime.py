"""Device interface for the optional fused angular FFT node."""

import ctypes as C
import os
from pathlib import Path
import sys

from fused_nonlinear_fft import plan


HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/"rfm_v7"))
from cuda_environment_native import DP, I, P, bind, call, nvrtc_load
from galerkin_cuda_native import TOOLKIT_DEFAULT


SUPPORTED_LENGTHS = (200,225,243)
SOURCE = HERE/"fused_nonlinear_fft.cu"


class FusedNonlinearCUDA:
    """Replace the inverse FFT, product and forward FFT CUDA graph nodes.

    The unchanged input reconstruction writes field-major half spectra.
    The unchanged SBDF2 advance consumes raw, unnormalized flux and history.
    """

    def __init__(self, runtime, resources, *, n, k, rows, family, toolkit=None, module=None):
        if n not in SUPPORTED_LENGTHS or n <= 3*k or k < 1 or rows < 1 or family not in ("gal","fd"):
            raise ValueError("Unsupported fused FFT layout or de-aliasing width")
        fft = plan(n)
        self.runtime, self.rows, self.modes, self.family = runtime, rows, k+1, family
        self.shared = 2*n*16
        if module is None:
            root = Path(toolkit or os.environ.get("CUDA_PATH",TOOLKIT_DEFAULT))
            module = nvrtc_load(root,runtime.driver,SOURCE.read_bytes(),SOURCE.name.encode("ascii"))
            resources.callback(call,bind(runtime.driver_lib,"cuModuleUnload",[P]),module)
        self.kernel = P()
        call(runtime.driver.cuModuleGetFunction,C.byref(self.kernel),module,f"fused_{family}_{n}".encode("ascii"))
        self.permutation = runtime.upload(resources,fft.permutation)
        self.roots = runtime.upload(resources,fft.roots)

    def launch(self, spectrum, flux, history, *, mean_terms=None, step=None, mean=None):
        arrays = (spectrum,flux,history)
        if len({array.value for array in arrays}) != 3 or not all(array.value for array in arrays):
            raise ValueError("Input, output and history require distinct nonzero device addresses")
        if self.family == "fd" and any(value is None or not value.value for value in (mean_terms,step,mean)):
            raise ValueError("FD fusion requires the original mean buffers and time index")
        arguments = (*arrays,self.permutation,self.roots,
                     *(DP(0) if value is None else value for value in (mean_terms,step,mean)),I(self.rows),I(self.modes))
        params = (P*len(arguments))(*(C.addressof(arg) for arg in arguments))
        driver = self.runtime.driver

        def execute():
            call(driver.cuLaunchKernel,self.kernel,self.rows,1,1,128,1,1,self.shared,driver.stream,params,None)

        # Keep every ctypes argument alive for both eager calls and capture.
        execute.arguments = arguments
        return execute
