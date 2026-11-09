"""Small CUDA driver, NVRTC, batched cuFFT and capture compatibility probes."""

import argparse
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np


P = C.c_void_p
I = C.c_int
U = C.c_uint
S = C.c_size_t
DP = C.c_uint64
CHAR = C.c_char_p


def bind(lib, name, args):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = I
    return fn


def call(fn, *args):
    status = fn(*args)
    if status:
        raise RuntimeError(f"{fn.__name__}: status {status}")


class Driver:
    def __init__(self):
        lib = C.WinDLL("nvcuda.dll")
        signatures = {
            "cuInit": [U],
            "cuDriverGetVersion": [C.POINTER(I)],
            "cuDeviceGet": [C.POINTER(I), I],
            "cuDevicePrimaryCtxRetain": [C.POINTER(P), I],
            "cuCtxSetCurrent": [P],
            "cuStreamCreate": [C.POINTER(P), U],
            "cuStreamSynchronize": [P],
            "cuMemAlloc_v2": [C.POINTER(DP), S],
            "cuMemFree_v2": [DP],
            "cuMemcpyHtoD_v2": [DP, P, S],
            "cuMemcpyDtoH_v2": [P, DP, S],
            "cuModuleLoadData": [C.POINTER(P), P],
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
            setattr(self, name, bind(lib, name, args))
        call(self.cuInit, 0)
        device = I()
        context = P()
        version = I()
        self.stream = P()
        call(self.cuDriverGetVersion, C.byref(version))
        self.version = version.value
        call(self.cuDeviceGet, C.byref(device), 0)
        call(self.cuDevicePrimaryCtxRetain, C.byref(context), device)
        call(self.cuCtxSetCurrent, context)
        call(self.cuStreamCreate, C.byref(self.stream), 1)

    def alloc(self, nbytes):
        address = DP()
        call(self.cuMemAlloc_v2, C.byref(address), nbytes)
        return address

    def upload(self, address, values):
        call(self.cuMemcpyHtoD_v2, address, values.ctypes.data, values.nbytes)

    def download(self, address, values):
        call(self.cuStreamSynchronize, self.stream)
        call(self.cuMemcpyDtoH_v2, values.ctypes.data, address, values.nbytes)

    def graph(self, operation):
        graph, executable = P(), P()
        call(self.cuStreamBeginCapture, self.stream, 0)
        try:
            operation()
        except BaseException:
            self.cuStreamEndCapture(self.stream, C.byref(graph))
            raise
        call(self.cuStreamEndCapture, self.stream, C.byref(graph))
        try:
            call(self.cuGraphInstantiateWithFlags, C.byref(executable), graph, 0)
            try:
                for _ in range(2):
                    call(self.cuGraphLaunch, executable, self.stream)
                call(self.cuStreamSynchronize, self.stream)
            finally:
                call(self.cuGraphExecDestroy, executable)
        finally:
            call(self.cuGraphDestroy, graph)


def nvrtc_load(toolkit, driver, src, filename=b"probe.cu"):
    lib = C.WinDLL(str(toolkit / "bin" / "nvrtc64_112_0.dll"))
    create = bind(lib, "nvrtcCreateProgram", [C.POINTER(P), CHAR, CHAR, I, P, P])
    compile_program = bind(lib, "nvrtcCompileProgram", [P, I, C.POINTER(CHAR)])
    log_size = bind(lib, "nvrtcGetProgramLogSize", [P, C.POINTER(S)])
    get_log = bind(lib, "nvrtcGetProgramLog", [P, P])
    ptx_size = bind(lib, "nvrtcGetPTXSize", [P, C.POINTER(S)])
    get_ptx = bind(lib, "nvrtcGetPTX", [P, P])
    destroy = bind(lib, "nvrtcDestroyProgram", [C.POINTER(P)])
    program = P()
    call(create, C.byref(program), src, filename, 0, None, None)
    try:
        status = compile_program(program, 1, (CHAR * 1)(b"--gpu-architecture=compute_52"))
        if status:
            size = S()
            call(log_size, program, C.byref(size))
            log = C.create_string_buffer(size.value)
            call(get_log, program, log)
            raise RuntimeError(f"nvrtcCompileProgram: {status} {log.value.decode(errors='replace')}")
        size = S()
        call(ptx_size, program, C.byref(size))
        ptx = C.create_string_buffer(size.value)
        call(get_ptx, program, ptx)
    finally:
        call(destroy, C.byref(program))
    module = P()
    call(driver.cuModuleLoadData, C.byref(module), ptx)
    return module


def nvrtc_program(toolkit, driver):
    src = b'extern "C" __global__ void twice(double *x, int n) { int i=threadIdx.x + blockIdx.x*blockDim.x; if(i<n) x[i]*=2.0; }'
    module = nvrtc_load(toolkit, driver, src)
    kernel = P()
    call(driver.cuModuleGetFunction, C.byref(kernel), module, b"twice")
    return kernel


def check_kernel(toolkit, driver):
    kernel = nvrtc_program(toolkit, driver)
    values = np.arange(512, dtype=np.float64)
    result = np.empty_like(values)
    address = driver.alloc(values.nbytes)
    try:
        count = I(values.size)
        params = (P * 2)(C.addressof(address), C.addressof(count))

        def launch():
            call(driver.cuLaunchKernel, kernel, 2, 1, 1, 256, 1, 1, 0, driver.stream, params, None)

        driver.upload(address, values)
        launch()
        driver.download(address, result)
        out = {"kernel_max_abs": float(np.max(np.abs(result - 2 * values)))}
        try:
            driver.graph(launch)
            driver.download(address, result)
            out["graph_max_abs"] = float(np.max(np.abs(result - 8 * values)))
        except RuntimeError as exc:
            out["graph_error"] = str(exc)
        return out
    finally:
        call(driver.cuMemFree_v2, address)


def check_fft(toolkit, driver):
    lib = C.WinDLL(str(toolkit / "bin" / "cufft64_10.dll"))
    plan_many = bind(lib, "cufftPlanMany",
                     [C.POINTER(I), I, C.POINTER(I), P, I, I, P, I, I, I, I])
    set_stream = bind(lib, "cufftSetStream", [I, P])
    d2z = bind(lib, "cufftExecD2Z", [I, DP, DP])
    z2d = bind(lib, "cufftExecZ2D", [I, DP, DP])
    destroy = bind(lib, "cufftDestroy", [I])
    fft_version = I()
    call(bind(lib, "cufftGetVersion", [C.POINTER(I)]), C.byref(fft_version))
    batch, n = 4, 192
    width = n // 2 + 1
    values = np.random.default_rng(7331).normal(size=(batch, n))
    fvalues = np.empty((batch, width), np.complex128)
    result = np.empty_like(values)
    x = driver.alloc(values.nbytes)
    f = driver.alloc(fvalues.nbytes)
    y = driver.alloc(result.nbytes)
    plan_forward, plan_inverse = I(), I()
    try:
        length = (I * 1)(n)
        call(plan_many, C.byref(plan_forward), 1, length, None, 1, n, None, 1, width, 0x6A, batch)
        call(plan_many, C.byref(plan_inverse), 1, length, None, 1, width, None, 1, n, 0x6C, batch)
        for plan in (plan_forward, plan_inverse):
            call(set_stream, plan, driver.stream)
        driver.upload(x, values)

        def launch():
            call(d2z, plan_forward, x, f)
            call(z2d, plan_inverse, f, y)

        launch()
        driver.download(f, fvalues)
        driver.download(y, result)
        out = {"version": fft_version.value,
               "forward_max_abs": float(np.max(np.abs(fvalues - np.fft.rfft(values)))),
               "round_trip_max_abs": float(np.max(np.abs(result / n - values)))}
        try:
            driver.graph(launch)
            driver.download(y, result)
            out["graph_round_trip_max_abs"] = float(np.max(np.abs(result / n - values)))
        except RuntimeError as exc:
            out["graph_error"] = str(exc)
        return out
    finally:
        for plan in (plan_inverse, plan_forward):
            if plan.value:
                call(destroy, plan)
        for address in (y, f, x):
            call(driver.cuMemFree_v2, address)


def check_blas(toolkit, driver):
    lib = C.WinDLL(str(toolkit / "bin" / "cublas64_11.dll"))
    create = bind(lib, "cublasCreate_v2", [C.POINTER(P)])
    destroy = bind(lib, "cublasDestroy_v2", [P])
    set_stream = bind(lib, "cublasSetStream_v2", [P, P])
    scalar = C.POINTER(C.c_double)
    dgemm = bind(lib, "cublasDgemm_v2",
                 [P, I, I, I, I, I, scalar, DP, I, DP, I, scalar, DP, I])
    handle = P()
    call(create, C.byref(handle))
    try:
        call(set_stream, handle, driver.stream)
        version = I()
        call(bind(lib, "cublasGetVersion_v2", [P, C.POINTER(I)]), handle, C.byref(version))
        rng = np.random.default_rng(7331)
        m, n, k = 96, 128, 33
        a = np.asfortranarray(rng.normal(size=(m, k)))
        b = np.asfortranarray(rng.normal(size=(k, n)))
        c = np.empty((m, n), dtype=np.float64, order="F")
        x, y, z = (driver.alloc(p.nbytes) for p in (a, b, c))
        try:
            driver.upload(x, a)
            driver.upload(y, b)
            alpha, beta = C.c_double(1.0), C.c_double(0.0)

            def launch():
                call(dgemm, handle, 0, 0, m, n, k, C.byref(alpha),
                     x, m, y, k, C.byref(beta), z, m)

            launch()
            driver.download(z, c)
            reference = a @ b
            out = {"version": version.value,
                   "dgemm_max_abs": float(np.max(np.abs(c - reference)))}
            try:
                driver.graph(launch)
                driver.download(z, c)
                out["graph_dgemm_max_abs"] = float(np.max(np.abs(c - reference)))
            except RuntimeError as exc:
                out["graph_error"] = str(exc)
            return out
        finally:
            for address in (z, y, x):
                call(driver.cuMemFree_v2, address)
    finally:
        call(destroy, handle)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--toolkit", type=Path, default=Path(os.environ.get(
        "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")))
    args = cli.parse_args()
    driver = Driver()
    report = {"driver_api_version": driver.version}
    with os.add_dll_directory(str(args.toolkit / "bin")):
        for key, probe in (("nvrtc_compute52", check_kernel), ("batched_cufft_fp64", check_fft),
                           ("cublas_fp64", check_blas)):
            try:
                report[key] = probe(args.toolkit, driver)
            except (RuntimeError, OSError) as exc:
                report[key] = {"error": str(exc)}
    content = json.dumps(report, indent=2)
    print(content)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
