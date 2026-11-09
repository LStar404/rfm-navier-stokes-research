"""Resident FP64 CUDA full-circle FD4 with cached band or dense solves."""

import argparse
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import time

from galerkin_cuda_native import Runtime, Launch, ROOT, TOOLKIT_DEFAULT, case, np
from cuda_environment_native import Driver, P, I, DP, bind, call, nvrtc_load
from fd_full import FullFD
from scipy import sparse
from scipy.linalg.lapack import dgbtrs


class FDRuntime(Runtime):
    def __init__(self, toolkit=None):
        root = Path(toolkit or os.environ.get("CUDA_PATH", TOOLKIT_DEFAULT))
        self.dll_directory = os.add_dll_directory(str(root / "bin"))
        self.driver = Driver()
        source = (ROOT / "fd_cuda_native.cu").read_bytes()
        self.module = nvrtc_load(root, self.driver, source, b"fd_cuda_native.cu")
        self.kernels = {}
        for name in ("fd_fields", "fd_products", "fd_rhs", "fd_band", "fd_dense", "fd_observe"):
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


class BandLaunch(Launch):
    def __init__(self, driver, kernel, w, modes, *args):
        super().__init__(driver, kernel, modes, *args)
        self.modes = modes
        self.shared_bytes = 16 * w

    def __call__(self):
        call(self.driver.cuLaunchKernel, self.kernel, self.modes, 1, 1, 32, 1, 1,
             self.shared_bytes, self.driver.stream, self.params, None)


def packed_rows(matrices):
    matrices = [m.tocsr() for m in matrices]
    w = matrices[0].shape[0]
    width = max(int(np.diff(m.indptr).max()) for m in matrices)
    columns = np.zeros((len(matrices), w, width), dtype=np.int32)
    weights = np.zeros((len(matrices), w, width))
    for i, matrix in enumerate(matrices):
        for row in range(w):
            a, b = matrix.indptr[row:row + 2]
            columns[i, row, :b-a] = matrix.indices[a:b]
            weights[i, row, :b-a] = matrix.data[a:b]
    return columns, weights


def radial_operators(model):
    """Derive stencils from exactly the assembled CPU derivatives and boundary closures."""
    modes, w = model.rhs_shape
    inverse = model.inv_r[1:-1]
    zero = sparse.csr_matrix((1, w))
    interior = sparse.eye(w, format="csr")
    curl_base = -sparse.diags(inverse**2) @ model.d2
    wall = -(model.inv_r[0]**2) * model.wall_d2[0]
    homogeneous_curl = sparse.vstack((wall[:, 1:-1], curl_base[:, 1:-1], zero), format="csr")
    mode_curl = sparse.vstack((zero, sparse.diags(inverse**2), zero), format="csr")
    mean_outer = sparse.csr_matrix(
        (model.right_mean, (np.zeros(4, dtype=int), np.arange(w-4, w))), shape=(1, w))
    mean_extension = sparse.vstack((zero, interior, mean_outer), format="csr")
    mean_inner = (sparse.csr_matrix(model.left_d1[None, :]) @ mean_extension[:5]) * model.inv_r[0]
    mean_inside = sparse.diags(inverse) @ (model.d1 @ mean_extension + interior)
    mean_curl = sparse.vstack((mean_inner, mean_inside, zero), format="csr")
    columns, weights = packed_rows((
        model.d1[:, 1:-1], curl_base[:, 1:-1],
        model.d1 @ homogeneous_curl, model.d1 @ mode_curl, model.d1 @ mean_curl,
    ))
    boundary = np.zeros(w+2)
    boundary[-1] = model.r[-1]
    boundary_curl = np.r_[float((wall @ boundary)[0]), curl_base @ boundary, 0.]
    lift = np.stack((-inverse*(model.d1 @ boundary), boundary_curl[1:-1], model.d1 @ boundary_curl))
    probe = np.zeros((2, len(case.PROBES), w, modes))
    probe[0, :, :, 1:] = 2 * model.probe_azimuthal[:, 1:-1, None]
    probe[1, :, :, 1:] = 2 * (model.probe_vort[:, 1:-1, None] +
                              model.probe_psi[:, 1:-1, None] * model.k2[None, None, 1:])
    probe[0, :, :, 0] = model.probe_mean_velocity
    probe[1, :, :, 0] = model.probe_mean_vort
    probe_lift = 2 * model.r[-1] * np.stack((model.probe_azimuthal[:, -1], model.probe_vort[:, -1]))
    return columns, weights, lift, probe, probe_lift


def precompute_dense(model):
    """Apply cached band factors to the identity in double precision."""
    factors = model.bdf
    w = factors.width
    identity = np.eye(w, order="F")
    matrices = []
    for lu, piv in zip(factors.factors, factors.pivots):
        inverse, info = dgbtrs(np.asfortranarray(lu), factors.lower, factors.upper, identity, piv)
        if info:
            raise np.linalg.LinAlgError(f"DGBTRS inverse preparation failed ({info})")
        matrices.append(inverse)
    return np.ascontiguousarray(np.stack(matrices, axis=-1))


class EventTrace:
    def __init__(self, runtime, resources, count, stages):
        dll = runtime.driver_lib
        self.stream = runtime.driver.stream
        create = bind(dll, "cuEventCreate", [C.POINTER(P), C.c_uint])
        destroy = bind(dll, "cuEventDestroy_v2", [P])
        self.record = bind(dll, "cuEventRecord", [P, P])
        self.elapsed = bind(dll, "cuEventElapsedTime", [C.POINTER(C.c_float), P, P])
        self.events = []
        for _ in range(count):
            events = [P() for _ in range(stages+1)]
            for event in events:
                call(create, C.byref(event), 0)
                resources.callback(call, destroy, event)
            self.events.append(events)

    def run(self, operations):
        for events in self.events:
            call(self.record, events[0], self.stream)
            for operation, event in zip(operations, events[1:]):
                operation()
                call(self.record, event, self.stream)

    def medians(self, names):
        samples = []
        for events in self.events:
            row = []
            for a, b in zip(events, events[1:]):
                result = C.c_float()
                call(self.elapsed, C.byref(result), a, b)
                row.append(result.value)
            samples.append(row)
        return dict(zip(names, np.median(samples, axis=0).tolist()))


def solve_fd_native(*, steps, horizon=72., snapshots=case.SNAPSHOTS, probe_spacing=.125,
                    graph_steps=24, profile_steps=0, cuda_linear_solve="band",
                    runtime=None, toolkit=None, **config):
    """Run the identical FullFD scheme, with no stepwise host transfer."""
    start = time.perf_counter()
    if (steps < 2 or horizon < probe_spacing or probe_spacing <= 0 or graph_steps < 0
            or not 0 <= profile_steps < steps):
        raise ValueError("Invalid steps, graph chunk, probe interval or profiling tail")
    if config.get("angular", "fft") != "fft" or config.get("linear", "native") != "native":
        raise ValueError("Use the FullFD dealiased FFT and real cached CPU factors for setup")
    if cuda_linear_solve not in ("band", "dense"):
        raise ValueError("Choose a band or dense CUDA solve")
    dt = horizon / steps
    times = np.asarray(snapshots, dtype=float)
    if not len(times) or times[0] <= 0 or np.any(np.diff(times) <= 0) or times[-1] > horizon:
        raise ValueError("Require positive ordered snapshots within the horizon")
    probe_times = np.arange(0., horizon + .01 * probe_spacing, probe_spacing)
    stride = int(case.step_indices(probe_times, dt)[1])
    if stride <= 0:
        raise ValueError("Require a probe interval of at least one step")
    amplitudes = case.speed(dt * np.arange(steps+1))
    snap_slots = np.full(steps+1, -1, dtype=np.int32)
    snap_slots[case.step_indices(times, dt)] = np.arange(len(times))
    model = FullFD(dt=dt, **config)
    modes, w = model.rhs_shape
    if cuda_linear_solve == "band" and model.bdf.lower + model.bdf.upper >= 32:
        raise ValueError("The band update needs all active diagonals in one warp")
    n = model.plan.n
    frame, half = w*modes, n//2+1
    states = np.zeros((2, w, modes), dtype=np.complex128)
    states[1] = model.advance(np.zeros(model.rhs_shape, dtype=np.complex128), amplitudes[1], True).T
    checkpoints = np.zeros((len(times), modes, w), dtype=np.complex128)
    probes = np.zeros((len(probe_times), 2, len(case.PROBES)))
    columns, weights, lift, probe, probe_lift = radial_operators(model)
    host_setup = time.perf_counter() - start
    t = time.perf_counter()
    if cuda_linear_solve == "dense":
        linear_data = precompute_dense(model)
    else:
        linear_data = np.ascontiguousarray(model.bdf.factors.transpose(0, 2, 1))
    precompute_time = time.perf_counter() - t
    with ExitStack() as resources:
        t = time.perf_counter()
        rt = resources.enter_context(FDRuntime(toolkit)) if runtime is None else runtime
        runtime_time = time.perf_counter() - t
        driver = rt.driver
        t = time.perf_counter()
        arrays = (states, amplitudes, np.array([1], dtype=np.int32), snap_slots,
                  columns, weights, lift, model.inv_r[1:-1], model.lift_bdf,
                  probe, probe_lift, checkpoints, probes, linear_data,
                  np.zeros((2, w, modes), dtype=np.complex128), np.zeros((w, modes)),
                  np.zeros((2, w)), np.zeros((4, w, half), dtype=np.complex128),
                  np.zeros((4, w, n)), np.zeros((w, n)),
                  np.zeros((w, half), dtype=np.complex128), np.zeros((w, half), dtype=np.complex128),
                  np.zeros((w, modes), dtype=np.complex128))
        (state, amps, index, slots, col, weights_gpu, lift_gpu, inv_r, lift_bdf,
         p_op, p_lift, saved, p_saved, linear, mass, mean_terms, mean, spectrum, physical,
         products, transformed, previous, rhs) = [rt.upload(resources, a) for a in arrays]
        pivot = (rt.upload(resources, model.bdf.pivots) if cuda_linear_solve == "band" else None)
        upload_time = time.perf_counter() - t
        t = time.perf_counter()
        plan_inverse = rt.plan(resources, n, 4*w, True)
        plan_forward = rt.plan(resources, n, w, False)
        form = rt.launch("fd_fields", w*half, state, amps, index, col, weights_gpu,
                         lift_gpu, inv_r, mass, mean_terms, spectrum, w, modes, half, columns.shape[-1])
        product = rt.launch("fd_products", w*n, physical, transformed, mean_terms, index,
                            products, previous, mean, w, modes, n, half)
        assemble = rt.launch("fd_rhs", frame, amps, index, mass, mean, transformed, previous,
                             inv_r, lift_bdf, rhs, w, modes, n, half, C.c_double(dt))
        if cuda_linear_solve == "band":
            advance = BandLaunch(driver, rt.kernels["fd_band"], w, modes, state, index,
                                 rhs, linear, pivot, w, modes, model.bdf.lower, model.bdf.upper)
        else:
            advance = rt.launch("fd_dense", frame, state, index, rhs, linear, w, modes)
        observe = rt.launch("fd_observe", 1, state, amps, index, slots, p_op, p_lift,
                            saved, p_saved, w, modes, stride, 1)
        first = rt.launch("fd_observe", 1, state, amps, index, slots, p_op, p_lift,
                          saved, p_saved, w, modes, stride, 0)
        stage_names = ("radial_stencils", "inverse_fft", "products_history_mean", "forward_fft",
                       "rhs", "linear_solve", "probe_snapshot_step")
        stages = (form, lambda: call(rt.z2d, plan_inverse, spectrum, physical),
                  product, lambda: call(rt.d2z, plan_forward, products, transformed),
                  assemble, advance, observe)

        def body():
            for stage in stages:
                stage()

        # Prime cuFFT with zero fields outside capture. The first physical state
        # and zero-velocity histories remain unchanged.
        stages[1]()
        stages[3]()
        first()
        rt.sync()
        plan_time = time.perf_counter() - t
        t = time.perf_counter()
        ordinary_steps = steps - 1 - profile_steps
        chunk = min(ordinary_steps, graph_steps)
        executable = rt.capture(resources, body, chunk) if chunk else None
        graph_time = time.perf_counter() - t
        trace = EventTrace(rt, resources, profile_steps, len(stages)) if profile_steps else None
        t = time.perf_counter()
        if executable is None:
            tail = ordinary_steps
        else:
            blocks, tail = divmod(ordinary_steps, chunk)
            for _ in range(blocks):
                call(driver.cuGraphLaunch, executable, driver.stream)
        for _ in range(tail):
            body()
        if trace:
            trace.run(stages)
        rt.sync()
        march_time = time.perf_counter() - t
        profile = trace.medians(stage_names) if trace else None
        t = time.perf_counter()
        driver.download(saved, checkpoints)
        driver.download(p_saved, probes)
        download_time = time.perf_counter() - t
    if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():
        raise FloatingPointError("Nonfinite native CUDA FD trajectory")
    row = dict(steps=steps, horizon=horizon, dt=dt, seconds=time.perf_counter()-start,
               time_order=2, implementation="native-fp64-fd4-cufft", cuda_linear_solve=cuda_linear_solve,
               graph_steps=graph_steps, host_setup_seconds=host_setup,
               host_linear_precompute_seconds=precompute_time, runtime_init_seconds=runtime_time,
               upload_seconds=upload_time, plan_seconds=plan_time, graph_setup_seconds=graph_time,
               march_seconds=march_time, download_seconds=download_time,
               linear_bytes=linear_data.nbytes+(model.bdf.pivots.nbytes if pivot is not None else 0),
               **model.metadata)
    if profile:
        row["diagnostic_eager_tail_profile"] = dict(steps=profile_steps, median_device_ms=profile)
    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--horizon", type=float, default=1.)
    cli.add_argument("--rate", type=int, default=128)
    cli.add_argument("--radial", type=int, default=176)
    cli.add_argument("--modes", type=int, default=80)
    cli.add_argument("--graph-steps", type=int, default=24)
    cli.add_argument("--profile-steps", type=int, default=0)
    cli.add_argument("--linear-solve", choices=("band", "dense"), default="band")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--cpu-solution", type=Path)
    cli.add_argument("--save-solution", type=Path)
    cli.add_argument("--reference", type=Path)
    args = cli.parse_args()
    settings = dict(n=args.radial, k=args.modes, stretch=1., grid="outer", angular="fft", linear="native")
    times = case.SNAPSHOTS if args.horizon == 72. else (args.horizon,)
    parameters = dict(steps=round(args.horizon*args.rate), horizon=args.horizon, snapshots=times, **settings)
    if args.cpu_solution:
        baseline = case.load_solution(args.cpu_solution, FullFD)
    else:
        baseline = case.solve(FullFD, **parameters)
    result = solve_fd_native(**parameters, graph_steps=args.graph_steps,
                             profile_steps=args.profile_steps, cuda_linear_solve=args.linear_solve)
    if (not np.array_equal(result.times, baseline.times) or any(
            baseline.row[k] != result.row[k] for k in (
                "n", "k", "stretch", "grid", "steps", "horizon", "angular", "linear",
                "time_order", "ntheta"))):
        raise ValueError("CPU comparator requires identical grids, time steps and checkpoints")
    errors = {}
    for name in ("states", "transverse", "vorticity"):
        a, b = getattr(baseline, name), getattr(result, name)
        norm = np.linalg.norm(a)
        errors[name] = dict(max_abs=float(np.max(abs(a-b))),
                            relative_l2=float(np.linalg.norm(a-b)/norm) if norm > 1e-12 else None)
    grids = case.evaluation_grids()
    grids["layer144"] = case.layer_scoring_grid()
    record = dict(cpu=baseline.row, gpu=result.row, errors=errors,
                  fields_vs_cpu=case.maxima(case.score(result, case.reference_fields(baseline, grids), grids)))
    if args.horizon == 72.:
        from diagnostic_wake import analyze
        record["wake"] = analyze(result.probe_times, result.transverse, result.vorticity)
        if args.reference:
            ref = case.load_solution(args.reference, FullFD)
            record["fields_vs_reference"] = case.maxima(case.score(
                result, case.reference_fields(ref, grids), grids))
    if args.save_solution:
        case.store_solution(args.save_solution, result, settings)
    text = json.dumps(record, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
