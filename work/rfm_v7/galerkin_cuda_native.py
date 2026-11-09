"""Native FP64 CUDA SBDF2 Galerkin, with resident state and batched cuFFT."""

import argparse
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import sys
import time

from cuda_environment_native import Driver, P, I, DP, bind, call, nvrtc_load

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "rfm_v6"))
import full_case as case
from full_galerkin_buffered import BufferedGalerkin

np = case.np
TOOLKIT_DEFAULT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")


class Launch:
    def __init__(self, driver, kernel, count, *args):
        self.driver = driver
        self.kernel = kernel
        self.grid = (count + 127) // 128
        self.args = [I(a) if isinstance(a, int) else a for a in args]
        self.params = (P * len(self.args))(*(C.addressof(a) for a in self.args))

    def __call__(self):
        call(self.driver.cuLaunchKernel, self.kernel, self.grid, 1, 1, 128, 1, 1,
             0, self.driver.stream, self.params, None)


class Runtime:
    def __init__(self, toolkit=None):
        root = Path(toolkit or os.environ.get("CUDA_PATH", TOOLKIT_DEFAULT))
        self.dll_directory = os.add_dll_directory(str(root / "bin"))
        self.driver = Driver()
        source = (ROOT / "galerkin_cuda_native.cu").read_bytes()
        self.module = nvrtc_load(root, self.driver, source, b"galerkin_cuda_native.cu")
        self.kernels = {}
        for name in ("spectrum", "products", "advance", "observe"):
            kernel = P()
            call(self.driver.cuModuleGetFunction, C.byref(kernel), self.module, name.encode())
            self.kernels[name] = kernel
        self.driver_lib = C.WinDLL("nvcuda.dll")
        self.fft = C.WinDLL(str(root / "bin" / "cufft64_10.dll"))
        self.plan_many = bind(self.fft, "cufftPlanMany",
                              [C.POINTER(I), I, C.POINTER(I), P, I, I, P, I, I, I, I])
        self.set_stream = bind(self.fft, "cufftSetStream", [I, P])
        self.d2z = bind(self.fft, "cufftExecD2Z", [I, DP, DP])
        self.z2d = bind(self.fft, "cufftExecZ2D", [I, DP, DP])
        self.destroy_plan = bind(self.fft, "cufftDestroy", [I])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.sync()
            call(bind(self.driver_lib, "cuModuleUnload", [P]), self.module)
            call(bind(self.driver_lib, "cuStreamDestroy_v2", [P]), self.driver.stream)
            device = I()
            call(self.driver.cuDeviceGet, C.byref(device), 0)
            call(bind(self.driver_lib, "cuDevicePrimaryCtxRelease_v2", [I]), device)
        finally:
            self.dll_directory.close()

    def sync(self):
        call(self.driver.cuStreamSynchronize, self.driver.stream)

    def upload(self, stack, host):
        host = np.ascontiguousarray(host)
        address = self.driver.alloc(host.nbytes)
        stack.callback(call, self.driver.cuMemFree_v2, address)
        self.driver.upload(address, host)
        return address

    def launch(self, name, count, *args):
        return Launch(self.driver, self.kernels[name], count, *args)

    def plan(self, stack, n, batch, inverse):
        plan = I()
        width = n // 2 + 1
        span_in, span_out, transform = (width, n, 0x6C) if inverse else (n, width, 0x6A)
        call(self.plan_many, C.byref(plan), 1, (I * 1)(n), None, 1, span_in, None, 1,
             span_out, transform, batch)
        stack.callback(call, self.destroy_plan, plan)
        call(self.set_stream, plan, self.driver.stream)
        return plan

    def capture(self, stack, operation, steps):
        driver = self.driver
        graph, executable = P(), P()
        call(driver.cuStreamBeginCapture, driver.stream, 0)
        try:
            for _ in range(steps):
                operation()
        except BaseException:
            driver.cuStreamEndCapture(driver.stream, C.byref(graph))
            if graph:
                driver.cuGraphDestroy(graph)
            raise
        call(driver.cuStreamEndCapture, driver.stream, C.byref(graph))
        try:
            call(driver.cuGraphInstantiateWithFlags, C.byref(executable), graph, 0)
        finally:
            call(driver.cuGraphDestroy, graph)
        stack.callback(call, driver.cuGraphExecDestroy, executable)
        return executable


def operators(model, dt):
    w, k, q = model.width, model.k, len(model.r)
    mean_w = model.mean_width
    linear = np.zeros((w, w, k + 1))
    radial = np.zeros((w, q, k + 1))
    angular = np.zeros_like(radial)
    mass = model.M0 + model.k2[:, None, None] * model.M1
    linear[..., 1:] = (model.implicit @ mass).transpose(1, 2, 0)
    linear[:mean_w, :mean_w, 0] = model.mean_inverse @ model.mean_mass
    scale = dt / model.kernel.n
    radial[..., 1:] = scale * (model.implicit @ model.test_s).transpose(1, 2, 0)
    radial[:mean_w, :, 0] = -scale * (model.mean_inverse @ model.mean_test)
    angular[..., 1:] = scale * model.modes * (model.implicit @ model.test).transpose(1, 2, 0)
    mean = np.zeros((2, q, w))
    mean[0, :, :mean_w] = model.mean_samples[0]
    mean[1, :, :mean_w] = model.mean_vort_samples
    jets, lift, radius, mean_jets = model.probe_values
    probe = np.zeros((2, len(radius), w, k + 1))
    probe[0, :, :, 1:] = (-2 * jets[1] / radius[:, None])[..., None]
    probe[1, :, :, 1:] = 2 * (jets[0][..., None] * model.k2 - jets[2][..., None]) / radius[:, None, None] ** 2
    probe[0, :, :mean_w, 0] = mean_jets[0]
    probe[1, :, :mean_w, 0] = (mean_jets[0] + mean_jets[1]) / radius[:, None]
    probe_lift = np.stack((-2 * lift[1] / radius, 2 * (lift[0] - lift[2]) / radius ** 2))
    return (linear, radial, angular, model.implicit[0] @ model.M_lift,
            model.implicit[0] @ model.bdf_lift, mean, probe, probe_lift)


def solve_native(*, steps, horizon=72., snapshots=case.SNAPSHOTS, probe_spacing=.125,
                 graph_steps=24, profile_steps=0, runtime=None, toolkit=None, **config):
    """Return a CPU-reconstructible solution from a complete CUDA time march."""
    started = time.perf_counter()
    if steps < 2 or graph_steps < 0 or not 0 <= profile_steps < steps:
        raise ValueError("Require at least two steps, nonnegative graph steps and a valid profile length")
    if config.get("backend", "fft") != "fft":
        raise ValueError("The native implementation requires the dealiased FFT backend")
    dt = horizon / steps
    times = np.asarray(snapshots, dtype=float)
    if not len(times) or times[0] <= 0 or np.any(np.diff(times) <= 0) or times[-1] > horizon:
        raise ValueError("Require positive ordered snapshots within the horizon")
    probe_times = np.arange(0., horizon + .01 * probe_spacing, probe_spacing)
    stride = int(case.step_indices(probe_times, dt)[1])
    if stride <= 0:
        raise ValueError("Require a probe interval of at least one step")
    amplitudes = case.speed(dt * np.arange(steps + 1))
    snap_slots = np.full(steps + 1, -1, dtype=np.int32)
    snap_slots[case.step_indices(times, dt)] = np.arange(len(times))
    model = BufferedGalerkin(dt=dt, **config)
    w, k, q, n = model.width, model.k, len(model.r), model.kernel.n
    half, frame = n // 2 + 1, w * (k + 1)
    states = np.zeros((3, w, k + 1), dtype=np.complex128)
    states[1, :, 1] = amplitudes[1] * model.start
    checkpoints = np.zeros((len(times), w, k + 1), dtype=np.complex128)
    probes = np.zeros((len(probe_times), 2, len(case.PROBES)))
    linear, radial, angular, mass_lift, bdf_lift, mean, probe, probe_lift = operators(model, dt)
    host_setup = time.perf_counter() - started
    with ExitStack() as resources:
        t = time.perf_counter()
        rt = resources.enter_context(Runtime(toolkit)) if runtime is None else runtime
        runtime_setup = time.perf_counter() - t
        driver = rt.driver
        t = time.perf_counter()
        uploaded = [rt.upload(resources, a) for a in (
            states, amplitudes, np.array([1], dtype=np.int32), snap_slots,
            model.time_jets, mean, model.r, model.lift_modes, linear, radial, angular,
            mass_lift, bdf_lift, probe, probe_lift, checkpoints, probes,
            np.zeros((3, q, half), dtype=np.complex128), np.zeros((3, q, n)),
            np.zeros((2, q, n)), np.zeros((2, q, half), dtype=np.complex128),
            np.zeros((2, q, half), dtype=np.complex128))]
        (state, amps, index, slots, jets, avg, radius, lift, lm, rm, am,
         ml, bl, proj, pl, save, pr, spectrum, physical, products, transformed,
         previous) = uploaded
        upload_time = time.perf_counter() - t
        t = time.perf_counter()
        inverse = rt.plan(resources, n, 3 * q, True)
        forward = rt.plan(resources, n, 2 * q, False)
        form = rt.launch("spectrum", q * half, state, amps, index, jets, avg, radius,
                         lift, spectrum, w, k, q, half)
        multiply = rt.launch("products", max(q * n, 2 * q * half), physical, transformed,
                             products, previous, q * n, 2 * q * half)
        advance = rt.launch("advance", frame, state, amps, index, transformed, previous,
                            lm, rm, am, ml, bl, w, k, q, half)
        observe = rt.launch("observe", 1, state, amps, index, slots, proj, pl, save, pr,
                            frame, stride, 1)
        first = rt.launch("observe", 1, state, amps, index, slots, proj, pl, save, pr,
                          frame, stride, 0)

        stage_names = ("radial_reconstruction", "inverse_fft", "products_and_history", "forward_fft",
                       "projection_and_advance", "probes_checkpoints_and_step")
        stages = (form, lambda: call(rt.z2d, inverse, spectrum, physical), multiply,
                  lambda: call(rt.d2z, forward, products, transformed), advance, observe)

        def body():
            for stage in stages:
                stage()

        # Materialize both cuFFT plans before graph capture. Zero products remain zero.
        call(rt.z2d, inverse, spectrum, physical)
        call(rt.d2z, forward, products, transformed)
        first()
        rt.sync()
        plan_time = time.perf_counter() - t
        t = time.perf_counter()
        ordinary_steps = steps - 1 - profile_steps
        chunk = min(ordinary_steps, graph_steps)
        executable = rt.capture(resources, body, chunk) if chunk else None
        graph_time = time.perf_counter() - t
        trace = []
        if profile_steps:
            create = bind(rt.driver_lib, "cuEventCreate", [C.POINTER(P), C.c_uint])
            destroy = bind(rt.driver_lib, "cuEventDestroy_v2", [P])
            record = bind(rt.driver_lib, "cuEventRecord", [P, P])
            event_elapsed = bind(rt.driver_lib, "cuEventElapsedTime", [C.POINTER(C.c_float), P, P])
            for _ in range(profile_steps):
                events = [P() for _ in range(len(stages) + 1)]
                for event in events:
                    call(create, C.byref(event), 0)
                    resources.callback(call, destroy, event)
                trace.append(events)
        t = time.perf_counter()
        if executable is None:
            tail = ordinary_steps
        else:
            blocks, tail = divmod(ordinary_steps, chunk)
            for _ in range(blocks):
                call(driver.cuGraphLaunch, executable, driver.stream)
        for _ in range(tail):
            body()
        for events in trace:
            call(record, events[0], driver.stream)
            for stage, event in zip(stages, events[1:]):
                stage()
                call(record, event, driver.stream)
        rt.sync()
        march_time = time.perf_counter() - t
        stage_times = []
        for events in trace:
            record_ms = []
            for a, b in zip(events, events[1:]):
                duration = C.c_float()
                call(event_elapsed, C.byref(duration), a, b)
                record_ms.append(duration.value)
            stage_times.append(record_ms)
        t = time.perf_counter()
        driver.download(save, checkpoints)
        driver.download(pr, probes)
        download_time = time.perf_counter() - t
    if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():
        raise FloatingPointError("Nonfinite native CUDA trajectory")
    elapsed = time.perf_counter() - started
    row = dict(steps=steps, horizon=horizon, dt=dt, seconds=elapsed, time_order=2,
               implementation="native-fp64-cufft-precontracted", graph_steps=graph_steps,
               host_setup_seconds=host_setup, runtime_init_seconds=runtime_setup,
               upload_seconds=upload_time, plan_seconds=plan_time, graph_setup_seconds=graph_time,
               march_seconds=march_time, download_seconds=download_time, **model.metadata)
    if stage_times:
        row["diagnostic_eager_tail_profile"] = dict(
            steps=profile_steps, median_device_ms=dict(zip(stage_names, np.median(stage_times, axis=0).tolist())))
    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--horizon", type=float, default=1.)
    cli.add_argument("--rate", type=int, default=128)
    cli.add_argument("--graph-steps", type=int, default=24)
    cli.add_argument("--family", choices=("rfm", "legendre"), default="rfm")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--cpu-solution", type=Path)
    cli.add_argument("--save-solution", type=Path)
    cli.add_argument("--reference", type=Path)
    args = cli.parse_args()
    horizon = args.horizon
    settings = dict(m=37, k=72, q=66, scale=62., mean_order=24,
                    layers=(16, 36, 80, 165, 340, 660), split=True,
                    family=args.family, seed=7331)
    snapshots = case.SNAPSHOTS if horizon == 72. else (horizon,)
    kwargs = dict(steps=round(args.rate * horizon), horizon=horizon, snapshots=snapshots, **settings)
    baseline = (case.load_solution(args.cpu_solution, BufferedGalerkin) if args.cpu_solution
                else case.solve(BufferedGalerkin, **kwargs))
    test = solve_native(graph_steps=args.graph_steps, **kwargs)
    if (baseline.row["steps"] != test.row["steps"] or not np.array_equal(baseline.times, test.times)
            or any(baseline.row[k] != test.row[k] for k in ("m", "k", "q", "method", "seed", "rank"))):
        raise ValueError("CPU comparison needs identical configuration and physical checkpoints")
    errors = {}
    for name in ("states", "transverse", "vorticity"):
        a, b = getattr(baseline, name), getattr(test, name)
        norm = np.linalg.norm(a)
        errors[name] = dict(max_abs=float(np.max(np.abs(a-b))),
                            relative_l2=float(np.linalg.norm(a-b)/norm) if norm > 1e-12 else None)
    result = dict(cpu=baseline.row, gpu=test.row, errors=errors)
    if horizon == 72.:
        from diagnostic_wake import analyze
        grids = case.evaluation_grids()
        grids["layer144"] = case.layer_scoring_grid()
        result["fields_vs_cpu"] = case.maxima(case.score(test, case.reference_fields(baseline, grids), grids))
        result["wake"] = analyze(test.probe_times, test.transverse, test.vorticity)
        if args.reference:
            from fd_full import FullFD
            reference = case.load_solution(args.reference, FullFD)
            result["fields_vs_fd"] = case.maxima(case.score(test, case.reference_fields(reference, grids), grids))
    if args.save_solution:
        case.store_solution(args.save_solution, test, settings)
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
