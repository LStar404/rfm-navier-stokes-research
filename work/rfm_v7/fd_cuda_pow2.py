"""FP64 resident CUDA replay of the row-power-of-two FD4 reference scheme."""

from contextlib import ExitStack
import ctypes as C
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg.lapack import dgbtrs

from cuda_environment_native import P, bind, call, nvrtc_load
from fd_cuda_native import BandLaunch, FDRuntime, TOOLKIT_DEFAULT, case, packed_rows, radial_operators
from fd_pow2_long import Pow2EquilibratedBufferedFD


class Pow2Runtime(FDRuntime):
    def __init__(self, toolkit=None):
        super().__init__(toolkit)
        root = Path(toolkit or os.environ.get("CUDA_PATH", TOOLKIT_DEFAULT))
        source = Path(__file__).with_suffix(".cu")
        self.pow2_module = None
        try:
            self.pow2_module = nvrtc_load(root, self.driver, source.read_bytes(), b"fd_cuda_pow2.cu")
            for name in ("pow2_scale_rhs", "pow2_extend", "pow2_curl", "pow2_gradient"):
                kernel = P()
                call(self.driver.cuModuleGetFunction, C.byref(kernel), self.pow2_module, name.encode())
                self.kernels[name] = kernel
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        try:
            self.sync()
            if self.pow2_module is not None:
                call(bind(self.driver_lib, "cuModuleUnload", [P]), self.pow2_module)
        finally:
            super().__exit__(*exc)


def folded_mode(factors, mode):
    """Solve (D A) X = D, so a raw physical RHS is sufficient on device."""
    scale = factors.row_scale[mode]
    diagonal = np.zeros((scale.size, scale.size), order="F")
    np.fill_diagonal(diagonal, scale)
    solution, info = dgbtrs(np.asfortranarray(factors.factors[mode]), factors.lower,
                            factors.upper, diagonal, factors.pivots[mode], overwrite_b=True)
    if info:
        raise np.linalg.LinAlgError(f"DGBTRS folded inverse preparation failed ({info})")
    return solution


def folded_inverse(factors):
    width, modes = factors.width, factors.modes
    result = np.empty((width, width, modes))
    for mode in range(modes):
        result[:, :, mode] = folded_mode(factors, mode)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite row-equilibrated folded inverse")
    return result


def radial_pipeline(rt, resources, model, radial, state, amps, index, inv_r,
                    mass, mean_terms, spectrum, w, modes, half):
    if radial == "composed":
        columns, weights, lift, _, _ = radial_operators(model)
        col, val, boundary = [rt.upload(resources, a) for a in (columns, weights, lift)]
        form = rt.launch("fd_fields", w*half, state, amps, index, col, val, boundary,
                         inv_r, mass, mean_terms, spectrum, w, modes, half, columns.shape[-1])
        return (form,)

    columns, weights = packed_rows((model.d1, model.d2))
    wall = model.wall_d2[0].tocsr()
    arrays = (columns, weights, wall.indices.astype(np.int32), wall.data, model.left_d1,
              model.right_mean, model.inv_r, model.inv_r**2,
              np.zeros((w+2, modes), dtype=np.complex128), np.zeros((w+2, modes), dtype=np.complex128))
    col, val, wall_col, wall_val, left, right, inv, inv2, psi, omega = [
        rt.upload(resources, a) for a in arrays]
    taps = columns.shape[-1]
    extend = rt.launch("pow2_extend", (w+2)*modes, state, amps, index, right, psi,
                       C.c_double(model.r[-1]), w, modes)
    curl = rt.launch("pow2_curl", w*half+2*modes, psi, index, col, val, wall_col, wall_val,
                     left, inv, inv2, omega, mass, mean_terms, spectrum, w, modes, half,
                     taps, wall.nnz)
    gradient = rt.launch("pow2_gradient", w*half, omega, col, val, spectrum, w, modes, half, taps)
    return extend, curl, gradient


def solve_fd_cuda_pow2(*, steps, horizon=72., snapshots=case.SNAPSHOTS, probe_spacing=.125,
                       graph_steps=24, linear="dense", radial="staged", toolkit=None, **config):
    """Keep both SBDF2 histories resident and capture complete step sequences."""
    start = time.perf_counter()
    if steps < 2 or graph_steps < 0 or not 0 < probe_spacing <= horizon:
        raise ValueError("Invalid step count, graph chunk or probe interval")
    if linear not in ("dense", "band") or radial not in ("composed", "staged"):
        raise ValueError("Choose dense or band and composed or staged")
    if config.get("angular", "fft") != "fft":
        raise ValueError("Use the dealiased FFT reference angular discretization")
    dt = horizon/steps
    times = np.asarray(snapshots, dtype=float)
    if not len(times) or not np.isfinite(times).all() or not 0 < times[0] <= times[-1] <= horizon:
        raise ValueError("Require positive finite snapshots within the horizon")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Require strictly ordered snapshots")
    probe_times = np.arange(0., horizon+.01*probe_spacing, probe_spacing)
    stride = int(case.step_indices(probe_times, dt)[1])
    if stride <= 0:
        raise ValueError("Require a probe interval of at least one step")
    amplitudes = case.speed(dt*np.arange(steps+1))
    snap_slots = np.full(steps+1, -1, dtype=np.int32)
    snap_slots[case.step_indices(times, dt)] = np.arange(len(times))
    model = Pow2EquilibratedBufferedFD(dt=dt, linear="native", **config)
    modes, w = model.rhs_shape
    factors = model.bdf
    if linear == "band" and factors.lower+factors.upper >= 32:
        raise ValueError("The band update needs all active diagonals in one warp")
    n, frame = model.plan.n, w*modes
    half = n//2+1
    states = np.zeros((2, w, modes), dtype=np.complex128)
    states[1] = model.advance(np.zeros(model.rhs_shape, dtype=np.complex128), amplitudes[1], True).T
    checkpoints = np.zeros((len(times), modes, w), dtype=np.complex128)
    probes = np.zeros((len(probe_times), 2, len(case.PROBES)))
    _, _, _, probe, probe_lift = radial_operators(model)
    host_setup = time.perf_counter()-start
    t = time.perf_counter()
    if linear == "dense":
        linear_data = folded_inverse(factors)
        linear_bytes = linear_data.nbytes
    else:
        linear_data = np.ascontiguousarray(factors.factors.transpose(0, 2, 1))
        linear_bytes = linear_data.nbytes+factors.pivots.nbytes+factors.row_scale.nbytes
    precompute = time.perf_counter()-t

    with ExitStack() as resources:
        t = time.perf_counter()
        rt = resources.enter_context(Pow2Runtime(toolkit))
        runtime_time = time.perf_counter()-t
        driver = rt.driver
        t = time.perf_counter()
        arrays = (states, amplitudes, np.array([1], dtype=np.int32), snap_slots,
                  model.inv_r[1:-1], model.lift_bdf, probe, probe_lift, checkpoints, probes,
                  linear_data, np.zeros((2, w, modes), dtype=np.complex128), np.zeros((w, modes)),
                  np.zeros((2, w)), np.zeros((4, w, half), dtype=np.complex128),
                  np.zeros((4, w, n)), np.zeros((w, n)), np.zeros((w, half), dtype=np.complex128),
                  np.zeros((w, half), dtype=np.complex128), np.zeros((w, modes), dtype=np.complex128))
        (state, amps, index, slots, inv_r, lift_bdf, p_op, p_lift, saved, p_saved, linear_gpu,
         mass, mean_terms, mean, spectrum, physical, products, transformed, previous, rhs) = [
            rt.upload(resources, a) for a in arrays]
        radial_stages = radial_pipeline(rt, resources, model, radial, state, amps, index, inv_r,
                                       mass, mean_terms, spectrum, w, modes, half)
        if linear == "band":
            pivot = rt.upload(resources, factors.pivots)
            scale = rt.upload(resources, factors.row_scale.T)
            advance = (rt.launch("pow2_scale_rhs", frame, rhs, scale, frame),
                       BandLaunch(driver, rt.kernels["fd_band"], w, modes, state, index, rhs,
                                  linear_gpu, pivot, w, modes, factors.lower, factors.upper))
        else:
            advance = (rt.launch("fd_dense", frame, state, index, rhs, linear_gpu, w, modes),)
        upload = time.perf_counter()-t
        t = time.perf_counter()
        plan_inverse = rt.plan(resources, n, 4*w, True)
        plan_forward = rt.plan(resources, n, w, False)

        def inverse():
            call(rt.z2d, plan_inverse, spectrum, physical)

        def forward():
            call(rt.d2z, plan_forward, products, transformed)

        product = rt.launch("fd_products", w*n, physical, transformed, mean_terms, index,
                            products, previous, mean, w, modes, n, half)
        assemble = rt.launch("fd_rhs", frame, amps, index, mass, mean, transformed, previous,
                             inv_r, lift_bdf, rhs, w, modes, n, half, C.c_double(dt))
        observe = rt.launch("fd_observe", 1, state, amps, index, slots, p_op, p_lift,
                            saved, p_saved, w, modes, stride, 1)
        first = rt.launch("fd_observe", 1, state, amps, index, slots, p_op, p_lift,
                          saved, p_saved, w, modes, stride, 0)
        stages = (*radial_stages, inverse, product, forward, assemble, *advance, observe)

        def body():
            for stage in stages:
                stage()

        inverse()
        forward()
        first()
        rt.sync()
        plan = time.perf_counter()-t
        t = time.perf_counter()
        chunk = min(steps-1, graph_steps)
        executable = rt.capture(resources, body, chunk) if chunk else None
        graph = time.perf_counter()-t
        t = time.perf_counter()
        if executable is None:
            tail = steps-1
        else:
            blocks, tail = divmod(steps-1, chunk)
            for _ in range(blocks):
                call(driver.cuGraphLaunch, executable, driver.stream)
        for _ in range(tail):
            body()
        rt.sync()
        march = time.perf_counter()-t
        t = time.perf_counter()
        driver.download(saved, checkpoints)
        driver.download(p_saved, probes)
        download = time.perf_counter()-t
    if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():
        raise FloatingPointError("Nonfinite row-equilibrated CUDA FD trajectory")
    row = dict(model.metadata, steps=steps, horizon=horizon, dt=dt, time_order=2,
               seconds=time.perf_counter()-start, implementation="native-fp64-fd4-pow2-cufft",
               radial_pipeline=radial, cuda_linear_solve=linear, graph_steps=graph_steps,
               row_scale=model.scale_metadata(), host_setup_seconds=host_setup,
               host_linear_precompute_seconds=precompute, runtime_init_seconds=runtime_time,
               upload_seconds=upload, plan_seconds=plan, graph_setup_seconds=graph,
               march_seconds=march, download_seconds=download, linear_bytes=linear_bytes)
    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)
