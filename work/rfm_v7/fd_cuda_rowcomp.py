"""FP64 CUDA replay with the CPU FD4 k=1 row-moment arithmetic."""

from contextlib import ExitStack
import ctypes as C
import os
from pathlib import Path
import time

import numpy as np

from cuda_environment_native import P, bind, call, nvrtc_load
from fd_cpu_buffered import csr_arrays
from fd_cpu_row_moment import Pow2RowMomentFD, moment_metadata, moment_tables
from fd_cuda_native import BandLaunch, TOOLKIT_DEFAULT, case, packed_rows, radial_operators
from fd_cuda_pow2 import Pow2Runtime


class RowMomentRuntime(Pow2Runtime):
    def __init__(self, toolkit=None):
        self.moment_module = None
        super().__init__(toolkit)
        root = Path(toolkit or os.environ.get("CUDA_PATH", TOOLKIT_DEFAULT))
        source = Path(__file__).with_suffix(".cu")
        try:
            self.moment_module = nvrtc_load(root, self.driver, source.read_bytes(), b"fd_cuda_rowcomp.cu")
            for name in ("rowcomp_curl_k1", "rowcomp_gradient_k1"):
                kernel = P()
                call(self.driver.cuModuleGetFunction, C.byref(kernel), self.moment_module, name.encode())
                self.kernels[name] = kernel
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        try:
            self.sync()
            if self.moment_module is not None:
                call(bind(self.driver_lib, "cuModuleUnload", [P]), self.moment_module)
        finally:
            super().__exit__(*exc)


def radial_stages(rt, resources, model, state, amps, index, mass, mean_terms, spectrum):
    modes, w = model.rhs_shape
    half = model.plan.n//2+1
    columns, weights = packed_rows((model.d1, model.d2))
    wall = model.wall_d2[0].tocsr()
    moments = moment_tables(model)
    if moment_metadata(moments) != model.metadata["moment_tables"]:
        raise RuntimeError("CPU and CUDA radial moment tables differ")
    shape = (w+2, modes)
    arrays = (columns, weights, wall.indices.astype(np.int32), wall.data, model.left_d1,
              model.right_mean, model.inv_r, model.inv_r**2,
              np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128),
              *csr_arrays(model.d1), *csr_arrays(model.d2),
              *(moments[key] for key in ("d1_mu0", "d2_mu0", "d2_mu1", "d2_delta", "span")))
    (col, val, wall_col, wall_val, left, right, inv, inv2, psi, omega,
     d1_offsets, d1_columns, d1_values, d2_offsets, d2_columns, d2_values,
     d1_mu0, d2_mu0, d2_mu1, delta, span) = [rt.upload(resources, a) for a in arrays]
    taps = columns.shape[-1]
    extend = rt.launch("pow2_extend", (w+2)*modes, state, amps, index, right, psi,
                       C.c_double(model.r[-1]), w, modes)
    curl = rt.launch("pow2_curl", w*half+2*modes, psi, index, col, val, wall_col, wall_val,
                     left, inv, inv2, omega, mass, mean_terms, spectrum, w, modes, half,
                     taps, wall.nnz)
    curl_k1 = rt.launch("rowcomp_curl_k1", w, psi, index, d2_offsets, d2_columns, d2_values,
                        delta, span, d2_mu0, d2_mu1, inv2, omega, mass, mean_terms, spectrum,
                        w, modes, half)
    gradient = rt.launch("pow2_gradient", w*half, omega, col, val, spectrum, w, modes, half, taps)
    gradient_k1 = rt.launch("rowcomp_gradient_k1", w, omega, d1_offsets, d1_columns, d1_values,
                            d1_mu0, spectrum, w, modes, half)
    return (extend, curl, curl_k1, gradient, gradient_k1), (psi, omega)


def solve_fd_cuda_rowcomp(*, steps, horizon=72., snapshots=case.SNAPSHOTS, probe_spacing=.125,
                          graph_steps=24, toolkit=None, **config):
    """Execute the same SBDF2 problem with two selective k=1 device corrections."""
    start = time.perf_counter()
    if steps < 2 or graph_steps < 0 or not 0 < probe_spacing <= horizon:
        raise ValueError("Invalid step count, graph chunk or probe interval")
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
    model = Pow2RowMomentFD(dt=dt, linear="native", **config)
    modes, w = model.rhs_shape
    factors = model.bdf
    if modes < 2 or factors.lower+factors.upper >= 32:
        raise ValueError("Require k=1 and all active band diagonals in one warp")
    n, frame = model.plan.n, w*modes
    half = n//2+1
    states = np.zeros((2, w, modes), dtype=np.complex128)
    states[1] = model.advance(np.zeros(model.rhs_shape, dtype=np.complex128), amplitudes[1], True).T
    checkpoints = np.zeros((len(times), modes, w), dtype=np.complex128)
    probes = np.zeros((len(probe_times), 2, len(case.PROBES)))
    _, _, _, probe, probe_lift = radial_operators(model)
    linear_data = np.ascontiguousarray(factors.factors.transpose(0, 2, 1))
    linear_bytes = linear_data.nbytes+factors.pivots.nbytes+factors.row_scale.nbytes
    host_setup = time.perf_counter()-start

    with ExitStack() as resources:
        t = time.perf_counter()
        rt = resources.enter_context(RowMomentRuntime(toolkit))
        runtime_time = time.perf_counter()-t
        t = time.perf_counter()
        arrays = (states, amplitudes, np.array([1], dtype=np.int32), snap_slots,
                  model.inv_r[1:-1], model.lift_bdf, probe, probe_lift, checkpoints, probes,
                  linear_data, factors.pivots, factors.row_scale.T,
                  np.zeros((2, w, modes), dtype=np.complex128), np.zeros((w, modes)),
                  np.zeros((2, w)), np.zeros((4, w, half), dtype=np.complex128),
                  np.zeros((4, w, n)), np.zeros((w, n)), np.zeros((w, half), dtype=np.complex128),
                  np.zeros((w, half), dtype=np.complex128), np.zeros((w, modes), dtype=np.complex128))
        (state, amps, index, slots, inv_r, lift_bdf, p_op, p_lift, saved, p_saved, linear, pivot, scale,
         mass, mean_terms, mean, spectrum, physical, products, transformed, previous, rhs) = [
            rt.upload(resources, a) for a in arrays]
        forms, _ = radial_stages(rt, resources, model, state, amps, index, mass, mean_terms, spectrum)
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
        scale_rhs = rt.launch("pow2_scale_rhs", frame, rhs, scale, frame)
        advance = BandLaunch(rt.driver, rt.kernels["fd_band"], w, modes, state, index, rhs,
                             linear, pivot, w, modes, factors.lower, factors.upper)
        observe = rt.launch("fd_observe", 1, state, amps, index, slots, p_op, p_lift,
                            saved, p_saved, w, modes, stride, 1)
        first = rt.launch("fd_observe", 1, state, amps, index, slots, p_op, p_lift,
                          saved, p_saved, w, modes, stride, 0)
        stages = (*forms, inverse, product, forward, assemble, scale_rhs, advance, observe)

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
                call(rt.driver.cuGraphLaunch, executable, rt.driver.stream)
        for _ in range(tail):
            body()
        rt.sync()
        march = time.perf_counter()-t
        t = time.perf_counter()
        rt.driver.download(saved, checkpoints)
        rt.driver.download(p_saved, probes)
        download = time.perf_counter()-t
    if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():
        raise FloatingPointError("Nonfinite row-moment CUDA FD trajectory")
    row = dict(model.metadata, steps=steps, horizon=horizon, dt=dt, time_order=2,
               seconds=time.perf_counter()-start, implementation="native-fp64-fd4-pow2-rowmoment-cufft",
               radial_pipeline="staged-rowmoment-k1", cuda_linear_solve="band", graph_steps=graph_steps,
               row_scale=model.scale_metadata(), host_setup_seconds=host_setup,
               runtime_init_seconds=runtime_time, upload_seconds=upload, plan_seconds=plan,
               graph_setup_seconds=graph, march_seconds=march, download_seconds=download,
               linear_bytes=linear_bytes)
    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)
