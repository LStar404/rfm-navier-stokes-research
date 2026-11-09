"""Check resource rollback and fixed-input equivalence of the two CUDA loaders."""

import argparse
from contextlib import ExitStack, contextmanager
import ctypes as C
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import galerkin_cuda_module_bundle as bundle
import galerkin_cuda_native as native
import galerkin_cuda_variable as variable


np = native.np
KERNEL_FILES = ("galerkin_cuda_native.cu", "galerkin_cuda_variable.cu")


def check_source_and_rollback():
    combined = bundle.bundled_source()
    original = [b'#line 1 "'+name.encode("ascii")+b'"\n'+(native.ROOT/name).read_bytes() for name in KERNEL_FILES]
    if combined != b"\n".join(original):
        raise AssertionError("A bundled CUDA kernel source changed")

    def run(failure, teardown):
        calls = []

        @contextmanager
        def directory(path):
            calls.append("directory_open")
            try:
                yield
            finally:
                calls.append("directory_close")

        def invoke(name, *args):
            label = name
            if name == "cuCtxSetCurrent":
                label = "restore_context" if args[0].value == 9 else "set_context"
            calls.append(label)
            if label == failure:
                raise RuntimeError(label)
            pointers = {"cuDriverGetVersion": 12030, "cuDeviceGet": 0, "cuCtxGetCurrent": 9,
                        "cuDevicePrimaryCtxRetain": 10, "cuStreamCreate": 11, "cuModuleGetFunction": 12}
            if name in pointers:
                args[0]._obj.value = pointers[name]
            return 0

        def bind(library, name, signature):
            def function(*args):
                return invoke(name, *args)
            return function

        def load(toolkit, driver, source, filename):
            if source != combined:
                raise AssertionError("Wrong source passed to NVRTC")
            invoke("module_load")
            return C.c_void_p(14)

        def dll(name):
            if str(name).endswith("cufft64_10.dll"):
                invoke("cufft_load")
            return name

        with (patch.object(bundle.os, "add_dll_directory", directory), patch.object(bundle.C, "WinDLL", dll),
              patch.object(bundle, "bind", bind), patch.object(bundle, "nvrtc_load", load)):
            try:
                with bundle.BundledSwitchRuntime("toolkit") as runtime:
                    if tuple(runtime.kernels) != bundle.KERNEL_NAMES:
                        raise AssertionError("Kernel lookup list changed")
            except RuntimeError as error:
                if str(error) != failure:
                    raise
            else:
                if failure is not None:
                    raise AssertionError(f"Failure injection did not run: {failure}")
        if calls[-len(teardown):] != teardown or calls.count("module_load") > 1:
            raise AssertionError(f"Unexpected teardown for {failure}: {calls}")

    full = ["cuModuleUnload", "cuStreamDestroy_v2", "restore_context", "cuDevicePrimaryCtxRelease_v2", "directory_close"]
    failures = {
        "cuDevicePrimaryCtxRetain": ["cuDevicePrimaryCtxRetain", "directory_close"],
        "set_context": ["set_context", *full[-3:]],
        "cuStreamCreate": ["cuStreamCreate", *full[-3:]],
        "module_load": ["module_load", *full[-4:]],
        "cuModuleGetFunction": ["cuModuleGetFunction", *full],
        "cufft_load": ["cufft_load", *full],
        "cuStreamSynchronize": ["cuStreamSynchronize", *full],
        "cuModuleUnload": full,
    }
    run(None, ["cuStreamSynchronize", *full])
    for failure, cleanup in failures.items():
        run(failure, cleanup)
    return dict(source_sha256=sha256(combined).hexdigest(), injected_failures=list(failures), normal_teardown=True)


def fixed_kernels(runtime):
    rng = np.random.default_rng(763217)

    def real(shape):
        return rng.integers(-31, 32, size=shape).astype(np.float64)/32

    def complex_(shape):
        return real(shape)+1j*real(shape)

    def execute(name, count, inputs, outputs, *arguments):
        host = [np.array(values, copy=True) for values in inputs]
        with ExitStack() as buffers:
            device = [runtime.upload(buffers, values) for values in host]
            runtime.launch(name, count, *device, *arguments)()
            for index in outputs:
                runtime.driver.download(device[index], host[index])
        return tuple(host[i] for i in outputs)

    w, k, q, n = 5, 4, 4, 16
    half, frame = n//2+1, w*(k+1)
    state, amplitudes = complex_((3, w, k+1)), complex_(6)
    step = np.array([2], dtype=np.int32)
    spectrum = complex_((3, q, half))
    transformed, previous = complex_((2, q, half)), complex_((2, q, half))
    physical, products = real((3, q, n)), real((2, q, n))
    linear, radial, angular = real((w, w, k+1)), real((w, q, k+1)), real((w, q, k+1))
    advance = (state, amplitudes, step, transformed, previous, linear, radial, angular, real(w), real(w))
    snapshots = np.array([-1, -1, -1, 0, -1, -1], dtype=np.int32)
    probe_slots = np.array([0, -1, -1, 1, -1, -1], dtype=np.int32)
    observation = (real((2, 3, frame)), real((2, 3)), complex_((2, w, k+1)), real((3, 2, 3)))
    coefficients = (15/7, -64/105, 15/7, -8/7)
    results = {
        "spectrum": execute("spectrum", q*half, (state, amplitudes, step, real((3, q, w)), real((2, q, w)),
                             np.linspace(1., 2., q), complex_((3, q)), spectrum), (7,), w, k, q, half),
        "products": execute("products", max(q*n, 2*q*half), (physical, transformed, products, previous),
                            (2, 3), q*n, 2*q*half),
        "advance": execute("advance", frame, advance, (0,), w, k, q, half),
        "observe": execute("observe", 1, (state, amplitudes, step, snapshots, *observation), (2, 6, 7), frame, 3, 1),
        "advance_bridge": execute("advance_bridge", frame, advance, (0,), w, k, q, half,
                                  *(C.c_double(c) for c in coefficients)),
        "observe_absolute": execute("observe_absolute", 1, (state, amplitudes, step, snapshots, probe_slots, *observation),
                                    (2, 7, 8), frame, 1),
    }
    for name in ("advance", "advance_bridge"):
        evolved, = results[name]
        if np.array_equal(evolved[0], state[0]) or not np.array_equal(evolved[1:], state[1:]) or evolved[0, :, 0].imag.any():
            raise AssertionError(f"Fixed state did not exercise circular-buffer or mean-mode behavior: {name}")
    for name in ("observe", "observe_absolute"):
        final, checkpoint, probe = results[name]
        if final.item() != 3 or not np.array_equal(checkpoint[0], state[0]) or not np.array_equal(probe[::2], observation[3][::2]):
            raise AssertionError(f"Fixed observation did not exercise absolute and guard slots: {name}")
    if not np.array_equal(results["products"][1], transformed):
        raise AssertionError("Fixed products did not exercise Fourier history copying")
    return results


def compare_modules():
    with (patch.object(native, "nvrtc_load", wraps=native.nvrtc_load) as normal,
          patch.object(variable, "nvrtc_load", wraps=variable.nvrtc_load) as bridge,
          patch.object(bundle, "nvrtc_load", wraps=bundle.nvrtc_load) as combined):
        with variable.SwitchRuntime() as runtime:
            legacy = fixed_kernels(runtime)
            legacy_distinct = runtime.module.value != runtime.switch_module.value
        with bundle.BundledSwitchRuntime() as runtime:
            candidate = fixed_kernels(runtime)
            driver_version = runtime.driver.version
    counts = dict(legacy=normal.call_count+bridge.call_count, bundled=combined.call_count)
    if not legacy_distinct or (normal.call_count, bridge.call_count, combined.call_count) != (1, 1, 1):
        raise AssertionError(f"Unexpected number of module load calls: {counts}")
    kernels = {}
    for name in bundle.KERNEL_NAMES:
        comparisons = []
        for old, new in zip(legacy[name], candidate[name], strict=True):
            a, b = old.tobytes(), new.tobytes()
            comparisons.append(dict(shape=list(old.shape), dtype=str(old.dtype), bitwise_equal=a == b,
                                    max_abs_difference=float(np.max(np.abs(old-new))), legacy_sha256=sha256(a).hexdigest(),
                                    bundled_sha256=sha256(b).hexdigest()))
        kernels[name] = comparisons
    passed = all(out["bitwise_equal"] for rows in kernels.values() for out in rows)
    return dict(passed=passed, nvrtc_load_calls=counts, driver_version=driver_version, kernel_outputs=kernels)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cpu-only", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.output is not None and args.output.exists():
        cli.error("Output already exists")
    sources = (*KERNEL_FILES, "cuda_environment_native.py", "galerkin_cuda_native.py", "galerkin_cuda_variable.py",
               Path(bundle.__file__).name, Path(__file__).name)
    result = dict(status="failed", source_sha256={name: sha256((native.ROOT/name).read_bytes()).hexdigest() for name in sources})
    try:
        result["rollback"] = check_source_and_rollback()
        if not args.cpu_only:
            result["fixed_gpu"] = compare_modules()
        if args.cpu_only or result["fixed_gpu"]["passed"]:
            result["status"] = "passed_cpu_only" if args.cpu_only else "passed"
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    output = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output is None:
        print(output, end="")
    else:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(output)
        print(result["status"])
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
