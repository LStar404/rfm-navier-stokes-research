"""Standalone fixed-state correctness check; intentionally contains no benchmark.

Requires NumPy. Optional GPU mode uses the CUDA driver and NVRTC directly;
it neither imports the Navier--Stokes solver nor consumes archived flow fields.
"""
from contextlib import ExitStack
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import model

HERE = Path(__file__).resolve().parent
SUPPORTED = (200, 225, 243)
BALANCE_FLOOR = 1e-8  # Conservative dispatch policy, not an error theorem.


def validate(spectrum, old_flux, n, k, mean_terms=None, allow_stress=False):
    fd = mean_terms is not None
    inputs, outputs = (4, 1) if fd else (3, 2)
    if n not in SUPPORTED or not 1 <= k < n / 3:
        raise ValueError("Require N in {200,225,243} and 1 <= K < N/3")
    if spectrum.dtype != np.complex128 or spectrum.ndim != 3:
        raise ValueError("Spectrum must be complex128 [field,row,half-mode]")
    rows, half = spectrum.shape[1:]
    if rows < 1 or spectrum.shape != (inputs, rows, n // 2 + 1):
        raise ValueError("Wrong spectrum shape")
    if old_flux.dtype != np.complex128 or old_flux.shape != (outputs, rows, half):
        raise ValueError("Wrong old flux shape or type")
    if not spectrum.flags.c_contiguous or not old_flux.flags.c_contiguous:
        raise ValueError("Use contiguous field-major arrays")
    if not np.isfinite(spectrum).all() or not np.isfinite(old_flux).all():
        raise ValueError("Nonfinite input")
    if np.any(spectrum[..., k + 1:] != 0) or np.any(spectrum[..., 0].imag != 0):
        raise ValueError("Require zero padding and a real DC coefficient")
    if n % 2 == 0 and np.any(spectrum[..., -1].imag != 0):
        raise ValueError("Require a real Nyquist coefficient")
    if fd and (mean_terms.dtype != np.float64 or mean_terms.shape != (rows, k + 1)
               or not mean_terms.flags.c_contiguous or not np.isfinite(mean_terms).all()):
        raise ValueError("Wrong FD mean term contract")
    pairs = [(0, 1)] + ([(2, 3)] if fd else [])
    risky = []
    for a, b in pairs:
        for row in range(rows):
            sa, sb = (float(np.max(np.abs(spectrum[x, row]))) for x in (a, b))
            if min(sa, sb) > 0 and min(sa, sb) / max(sa, sb) < BALANCE_FLOOR:
                risky.append({"fields": [a, b], "row": row,
                              "spectral_scale_ratio": min(sa, sb) / max(sa, sb)})
    if risky and not allow_stress:
        raise ValueError("Imbalanced packed fields: use an unfused path or explicitly request a diagnostic")
    return risky


def oracle(spectrum, n):
    # NumPy's inverse is normalized; undo its 1/N to match CUDA's contract.
    physical = np.fft.irfft(spectrum, n=n, axis=-1) * n
    if len(spectrum) == 3:
        products = np.stack((physical[0] * physical[2], physical[1] * physical[2]))
        scale = np.stack((np.max(np.abs(physical[0]), axis=-1) * np.max(np.abs(physical[2]), axis=-1),
                          np.max(np.abs(physical[1]), axis=-1) * np.max(np.abs(physical[2]), axis=-1))) * n
    else:
        products = (physical[0] * physical[2] + physical[1] * physical[3])[None]
        scale = (np.max(np.abs(physical[0]), axis=-1) * np.max(np.abs(physical[2]), axis=-1)
                 + np.max(np.abs(physical[1]), axis=-1) * np.max(np.abs(physical[3]), axis=-1))[None] * n
    return np.fft.rfft(products, axis=-1), scale


def convolution_oracle(spectrum, n, k):
    # Independent retained-band polynomial convolution, without any FFT.
    def product(a, b):
        out = np.zeros((a.shape[0], k + 1), np.complex128)
        def mode(x, j):
            return x[:, j] if j >= 0 else x[:, -j].conj()
        for j in range(k + 1):
            for p in range(-k, k + 1):
                if -k <= j - p <= k:
                    out[:, j] += mode(a, p) * mode(b, j - p)
        return n * out
    if len(spectrum) == 3:
        return np.stack((product(spectrum[0], spectrum[2]), product(spectrum[1], spectrum[2])))
    return (product(spectrum[0], spectrum[2]) + product(spectrum[1], spectrum[3]))[None]


def error_metrics(actual, expected, scale):
    diff = np.abs(actual - expected)
    magnitude = np.max(np.abs(expected), axis=(-2, -1))
    relative = [float(np.max(diff[i]) / m) if m > 0 else None
                for i, m in enumerate(magnitude)]
    # Packed strong fields can contaminate a weak output. This aggregate bound
    # is therefore reported alongside, never instead of, the per-channel errors.
    uncancelled = max(float(np.max(scale)), np.finfo(float).tiny)
    absolute_scaled = float(np.max(diff) / uncancelled)
    assert absolute_scaled < 5e-12, absolute_scaled
    return {"channel_relative_max": relative, "uncancelled_scale_error": absolute_scaled,
            "max_absolute": float(np.max(diff))}


def check_contract_rejections():
    n, k = 225, 74
    spectrum = np.ones((3, 4, 113), np.complex128)
    spectrum[..., k + 1:] = 0
    old = np.zeros((2, 4, 113), np.complex128)
    invalid = []
    invalid.append(("unsupported_length", spectrum, old, 224, k))
    invalid.append(("aliased_bandwidth", spectrum, old, n, 75))
    nonfinite = spectrum.copy(); nonfinite[0, 0, 1] = np.inf
    invalid.append(("nonfinite_spectrum", nonfinite, old, n, k))
    endpoint = spectrum.copy(); endpoint[0, 0, 0] = 1 + 1j
    invalid.append(("complex_dc", endpoint, old, n, k))
    padded = spectrum.copy(); padded[0, 0, k + 1] = 1
    invalid.append(("nonzero_padding", padded, old, n, k))
    invalid.append(("noncontiguous", spectrum[:, ::-1], old, n, k))
    invalid.append(("wrong_precision", spectrum.astype(np.complex64), old, n, k))
    invalid.append(("wrong_output_shape", spectrum, old[:, :, :-1], n, k))
    old_bad = old.copy(); old_bad[0, 0, 0] = np.nan
    invalid.append(("nonfinite_history", spectrum, old_bad, n, k))
    names = []
    for name, values, history, length, band in invalid:
        try:
            validate(values, history, length, band)
        except ValueError:
            names.append(name)
        else:
            raise AssertionError(f"Input contract accepted {name}")
    return names


class GPU:
    def __init__(self, cuda_path):
        self.resources = ExitStack()
        self.P, self.I, self.U64 = C.c_void_p, C.c_int, C.c_uint64
        root = Path(cuda_path)
        if os.name == "nt":
            self.resources.enter_context(os.add_dll_directory(str(root / "bin")))
            self.driver = C.WinDLL("nvcuda.dll")
            candidates = sorted((root / "bin").glob("nvrtc64_*.dll"))
            self.nvrtc = C.WinDLL(str(candidates[-1]))
        else:
            self.driver = C.CDLL("libcuda.so.1")
            candidates = sorted((root / "lib64").glob("libnvrtc.so*"))
            self.nvrtc = C.CDLL(str(candidates[0]))
        P, I, U64, S = self.P, self.I, self.U64, C.c_size_t
        signatures = {
            "cuInit": [C.c_uint], "cuDriverGetVersion": [C.POINTER(I)],
            "cuDeviceGet": [C.POINTER(I), I], "cuDeviceGetName": [P, I, I],
            "cuCtxCreate_v2": [C.POINTER(P), C.c_uint, I], "cuCtxDestroy_v2": [P],
            "cuCtxSynchronize": [], "cuMemAlloc_v2": [C.POINTER(U64), S],
            "cuMemFree_v2": [U64], "cuMemcpyHtoD_v2": [U64, P, S],
            "cuMemcpyDtoH_v2": [P, U64, S], "cuModuleLoadData": [C.POINTER(P), P],
            "cuModuleUnload": [P], "cuModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
            "cuLaunchKernel": [P, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                               C.c_uint, C.c_uint, P, P, P],
        }
        for name, args in signatures.items():
            self.bind(self.driver, name, args)
        self.call("cuInit", 0)
        device, context, version = I(), P(), I()
        self.call("cuDeviceGet", C.byref(device), 0)
        self.call("cuDriverGetVersion", C.byref(version))
        name = C.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), device)
        self.call("cuCtxCreate_v2", C.byref(context), 0, device)
        self.resources.callback(self.call, "cuCtxDestroy_v2", context)
        self.metadata = {"gpu": name.value.decode(), "driver_api_version": version.value,
                         "nvrtc_library": candidates[-1].name if os.name == "nt" else candidates[0].name,
                         "compile_options": ["--gpu-architecture=compute_75"]}
        self.module = self.compile((HERE / "fused_nonlinear_fft.cu").read_bytes())
        self.resources.callback(self.call, "cuModuleUnload", self.module)

    @staticmethod
    def bind(library, name, args):
        fn = getattr(library, name)
        fn.argtypes, fn.restype = args, C.c_int
        return fn

    def call(self, name, *args):
        status = getattr(self.driver, name)(*args)
        if status:
            raise RuntimeError(f"{name}: CUDA status {status}")

    def compile(self, source):
        P, I, S = self.P, self.I, C.c_size_t
        specs = {"nvrtcCreateProgram": [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P],
                 "nvrtcCompileProgram": [P, I, C.POINTER(C.c_char_p)],
                 "nvrtcGetProgramLogSize": [P, C.POINTER(S)], "nvrtcGetProgramLog": [P, P],
                 "nvrtcGetPTXSize": [P, C.POINTER(S)], "nvrtcGetPTX": [P, P],
                 "nvrtcDestroyProgram": [C.POINTER(P)]}
        for name, args in specs.items():
            self.bind(self.nvrtc, name, args)
        def nc(name, *args):
            status = getattr(self.nvrtc, name)(*args)
            if status:
                raise RuntimeError(f"{name}: NVRTC status {status}")
        program = P()
        nc("nvrtcCreateProgram", C.byref(program), source, b"fused_nonlinear_fft.cu", 0, None, None)
        try:
            options = (C.c_char_p * 1)(b"--gpu-architecture=compute_75")
            status = self.nvrtc.nvrtcCompileProgram(program, 1, options)
            if status:
                size = S()
                nc("nvrtcGetProgramLogSize", program, C.byref(size))
                log = C.create_string_buffer(size.value)
                nc("nvrtcGetProgramLog", program, log)
                raise RuntimeError(log.value.decode())
            size = S()
            nc("nvrtcGetPTXSize", program, C.byref(size))
            ptx = C.create_string_buffer(size.value)
            nc("nvrtcGetPTX", program, ptx)
        finally:
            nc("nvrtcDestroyProgram", C.byref(program))
        module = P()
        self.call("cuModuleLoadData", C.byref(module), ptx)
        return module

    def evaluate(self, spectrum, old_flux, fft, k, mean_terms=None, step=2, allow_stress=False):
        validate(spectrum, old_flux, fft.n, k, mean_terms, allow_stress)
        fd, rows = mean_terms is not None, spectrum.shape[1]
        initial_mean = np.full((2, rows), 1234567.25)
        arrays = [spectrum, old_flux.copy(), np.zeros_like(old_flux), fft.permutation,
                  fft.roots, mean_terms if fd else np.zeros(1), np.array([step], np.int32), initial_mean]
        with ExitStack() as allocations:
            addresses = []
            for array in arrays:
                pointer = self.U64()
                self.call("cuMemAlloc_v2", C.byref(pointer), array.nbytes)
                allocations.callback(self.call, "cuMemFree_v2", pointer)
                self.call("cuMemcpyHtoD_v2", pointer, array.ctypes.data, array.nbytes)
                addresses.append(pointer)
            kernel = self.P()
            name = f"fused_{'fd' if fd else 'gal'}_{fft.n}".encode()
            self.call("cuModuleGetFunction", C.byref(kernel), self.module, name)
            args = addresses + [self.I(rows), self.I(k + 1)]
            params = (self.P * len(args))(*(C.addressof(x) for x in args))
            self.call("cuLaunchKernel", kernel, rows, 1, 1, 128, 1, 1, 32 * fft.n, None, params, None)
            self.call("cuCtxSynchronize")
            outputs = []
            for i in (1, 2, 7):
                out = np.empty_like(arrays[i])
                self.call("cuMemcpyDtoH_v2", out.ctypes.data, addresses[i], out.nbytes)
                outputs.append(out)
        if fd:
            assert np.array_equal(outputs[2][1 - step % 2], initial_mean[1 - step % 2])
        return outputs[0], outputs[1], outputs[2][step % 2] if fd else None

    def close(self):
        self.resources.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--output", default="validation.json")
    args = parser.parse_args()
    gpu = None
    report = {"kind": "fixed_state_correctness_only", "python": sys.version.split()[0],
              "numpy": np.__version__, "platform": platform.platform(), "gpu": None,
              "kernel_sha256": hashlib.sha256((HERE / "fused_nonlinear_fft.cu").read_bytes()).hexdigest(),
              "cases": [], "latency_benchmark": "not_run", "full_NS_trajectory": "not_run",
              "cufftdx_comparison": "not_implemented",
              "invalid_input_rejections": check_contract_rejections()}
    rng = np.random.default_rng(7331)
    try:
        if args.gpu:
            gpu = GPU(args.cuda_path)
            report["gpu"] = gpu.metadata
        for n in SUPPORTED:
            fft, k, rows = model.plan(n), (n - 1) // 3, 4
            for fd in (False, True):
                inputs, outputs = (4, 1) if fd else (3, 2)
                fixtures = ("broadband", "zero", "constant", "retained_edge", "imbalanced",
                            "cancellation" if fd else "zero_velocity_channel")
                for index, kind in enumerate(fixtures):
                    spectrum = (rng.normal(size=(inputs, rows, n // 2 + 1))
                                + 1j * rng.normal(size=(inputs, rows, n // 2 + 1))) / n
                    spectrum[..., k + 1:] = 0
                    spectrum[..., 0] = spectrum[..., 0].real
                    if kind == "zero":
                        spectrum.fill(0)
                    elif kind == "constant":
                        spectrum[..., 1:] = 0
                    elif kind == "retained_edge":
                        spectrum[..., :k] = 0
                    elif kind == "imbalanced":
                        spectrum[0] *= 1e-10
                    elif kind == "cancellation" and fd:
                        spectrum[1] = spectrum[0]
                        spectrum[3] = -spectrum[2]
                    elif kind == "zero_velocity_channel":
                        spectrum[0] = 0
                    old = np.ascontiguousarray(rng.normal(size=(outputs, rows, n // 2 + 1)).astype(np.complex128))
                    mean_terms = np.ascontiguousarray(rng.normal(size=(rows, k + 1))) if fd else None
                    rejected = False
                    try:
                        validate(spectrum, old, n, k, mean_terms)
                    except ValueError:
                        if kind != "imbalanced":
                            raise
                        rejected = True
                    assert rejected == (kind == "imbalanced")
                    risks = validate(spectrum, old, n, k, mean_terms, allow_stress=True)
                    expected, scale = oracle(spectrum, n)
                    direct = convolution_oracle(spectrum, n, k)
                    direct_error = error_metrics(direct, expected[..., :k + 1], scale)
                    host, history, mean = model.nonlinear(spectrum, old, fft, k, mean_terms)
                    assert np.array_equal(history, old)
                    if fd:
                        assert np.array_equal(mean, model.fd_mean(mean_terms))
                    row = {"N": n, "K": k, "rows": rows, "family": "fd" if fd else "galerkin",
                           "fixture": kind, "balance_guard_rejected": rejected, "packing_risks": risks,
                           "convolution_vs_fft": direct_error, "cpu_model": error_metrics(host, expected, scale)}
                    if gpu:
                        device, history, mean = gpu.evaluate(spectrum, old, fft, k, mean_terms,
                                                             step=2 + index % 2, allow_stress=True)
                        assert np.array_equal(history, old)
                        if fd:
                            np.testing.assert_allclose(mean, model.fd_mean(mean_terms), rtol=1e-14, atol=1e-14)
                        row["cuda"] = error_metrics(device, expected, scale)
                    if kind not in ("imbalanced", "cancellation"):
                        for key in ("cpu_model", "cuda"):
                            if key in row:
                                assert all(v is None or v < 5e-12 for v in row[key]["channel_relative_max"])
                    report["cases"].append(row)
        report["status"] = "passed_scoped_checks_with_documented_packing_limitations"
    finally:
        if gpu:
            gpu.close()
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"{len(report['cases'])} fixed-state cases passed; GPU={bool(gpu)}; no timings collected.")
    print(f"Report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
