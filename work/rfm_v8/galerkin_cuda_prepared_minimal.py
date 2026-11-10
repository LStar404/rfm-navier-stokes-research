"""Unchanged single-bridge CUDA march with explicit preparation ablations."""
from contextlib import ExitStack
import ctypes as C
import time

from sbdf2_schedule_minimal import SNAPSHOTS, bridge_weights, step_schedule
from cuda_environment_native import call
from galerkin_cuda_native import case, np
from galerkin_cuda_variable_minimal import SwitchRuntime, phase_operators, bridge_launch
from galerkin_cuda_module_bundle import BundledSwitchRuntime
from galerkin_phase_shared_minimal import phase_operators_shared

PREPARATIONS = {
    "module_only": (BundledSwitchRuntime, phase_operators),
    "shared_only": (SwitchRuntime, phase_operators_shared),
    "combined": (BundledSwitchRuntime, phase_operators_shared),
}

def solve_cuda_prepared(*, preparation="combined", startup_rate=128, late_rate=112, switch_time=8., horizon=72., snapshots=SNAPSHOTS,
                      probe_spacing=.125, graph_steps=24, toolkit=None, **config):
    start = time.perf_counter()
    if preparation not in PREPARATIONS:
        raise ValueError("Choose module_only, shared_only, or combined")
    runtime_type, phase_builder = PREPARATIONS[preparation]
    if graph_steps < 0 or config.get("backend", "fft") != "fft":
        raise ValueError("Require nonnegative graph size and the dealiased FFT backend")
    schedule = step_schedule(startup_rate=startup_rate, late_rate=late_rate, switch_time=switch_time,
                             horizon=horizon, snapshots=snapshots, probe_spacing=probe_spacing)
    split = int(schedule.switch_step)
    steps = len(schedule.times)-1
    if not 2 <= split < steps or len(case.PROBES) != 3:
        raise ValueError("Require at least two startup steps, one late step, and three probes")
    model, (mean, probe, probe_lift), packed, coefficients = phase_builder(startup_rate, late_rate, config)
    w, k, q, n = model.width, model.k, len(model.r), model.kernel.n
    half, frame = n//2+1, w*(k+1)
    amplitudes = case.speed(schedule.times)
    snapshots = schedule.snapshot_times
    snap_slots = np.full(steps+1, -1, dtype=np.int32)
    snap_slots[schedule.snapshot_steps] = np.arange(len(snapshots))
    probe_slots = np.full(steps+1, -1, dtype=np.int32)
    probe_slots[schedule.probe_steps] = np.arange(len(schedule.probe_times))
    states = np.zeros((3, w, k+1), dtype=np.complex128)
    states[1, :, 1] = amplitudes[1]*model.start
    checkpoints = np.zeros((len(snapshots), w, k+1), dtype=np.complex128)
    probes = np.zeros((len(schedule.probe_times), 2, len(case.PROBES)))
    setup = time.perf_counter()-start
    with ExitStack() as resources:
        t = time.perf_counter()
        rt = resources.enter_context(runtime_type(toolkit))
        runtime_setup = time.perf_counter()-t
        t = time.perf_counter()
        arrays = (states, amplitudes, np.array([1], dtype=np.int32), snap_slots, probe_slots,
                  model.time_jets, mean, model.r, model.lift_modes, probe, probe_lift, checkpoints, probes,
                  np.zeros((3, q, half), dtype=np.complex128), np.zeros((3, q, n)),
                  np.zeros((2, q, n)), np.zeros((2, q, half), dtype=np.complex128),
                  np.zeros((2, q, half), dtype=np.complex128))
        (state, amps, index, slots, p_slots, jets, avg, radius, lift, proj, pl, saved, p_saved,
         spectrum, physical, products, transformed, previous) = [rt.upload(resources, a) for a in arrays]
        if startup_rate == late_rate:
            shared = tuple(rt.upload(resources, a) for a in packed[0])
            old_pack, bridge_pack, late_pack = (shared,)*3
        else:
            old_pack, bridge_pack, late_pack = [tuple(rt.upload(resources, a) for a in pack) for pack in packed]
        upload = time.perf_counter()-t
        t = time.perf_counter()
        inverse = rt.plan(resources, n, 3*q, True)
        forward = rt.plan(resources, n, 2*q, False)
        form = rt.launch("spectrum", q*half, state, amps, index, jets, avg, radius, lift, spectrum, w, k, q, half)
        multiply = rt.launch("products", max(q*n, 2*q*half), physical, transformed, products, previous, q*n, 2*q*half)
        observe = rt.launch("observe_absolute", 1, state, amps, index, slots, p_slots, proj, pl,
                            saved, p_saved, frame, 1)
        first = rt.launch("observe_absolute", 1, state, amps, index, slots, p_slots, proj, pl,
                          saved, p_saved, frame, 0)
        old_advance = rt.launch("advance", frame, state, amps, index, transformed, previous, *old_pack, w, k, q, half)
        late_advance = rt.launch("advance", frame, state, amps, index, transformed, previous, *late_pack, w, k, q, half)
        bridge_advance = bridge_launch(rt, frame, state, amps, index, transformed, previous, bridge_pack,
                                       w, k, q, half, coefficients)

        def body(advance):
            form()
            call(rt.z2d, inverse, spectrum, physical)
            multiply()
            call(rt.d2z, forward, products, transformed)
            advance()
            observe()

        call(rt.z2d, inverse, spectrum, physical)
        call(rt.d2z, forward, products, transformed)
        first()
        rt.sync()
        plan = time.perf_counter()-t
        t = time.perf_counter()
        before_steps, after_steps = split-1, steps-split-1
        old_chunk, late_chunk = min(graph_steps, before_steps), min(graph_steps, after_steps)
        old_graph = rt.capture(resources, lambda: body(old_advance), old_chunk) if old_chunk else None
        late_graph = rt.capture(resources, lambda: body(late_advance), late_chunk) if late_chunk else None
        graph = time.perf_counter()-t

        def enqueue(advance, executable, chunk, count):
            blocks, tail = divmod(count, chunk) if chunk else (0, count)
            for _ in range(blocks):
                call(rt.driver.cuGraphLaunch, executable, rt.driver.stream)
            for _ in range(tail):
                body(advance)
            return blocks*chunk+tail

        t = time.perf_counter()
        executed = enqueue(old_advance, old_graph, old_chunk, before_steps)
        # transformed holds raw products from n-1. The next ordinary product
        # kernel copies them to previous, then cuFFT overwrites transformed.
        body(bridge_advance)
        executed += 1+enqueue(late_advance, late_graph, late_chunk, after_steps)
        rt.sync()
        march = time.perf_counter()-t
        t = time.perf_counter()
        final = np.empty(1, dtype=np.int32)
        rt.driver.download(index, final)
        rt.driver.download(saved, checkpoints)
        rt.driver.download(p_saved, probes)
        download = time.perf_counter()-t
    if int(final[0]) != steps or executed != steps-1:
        raise RuntimeError("Resident CUDA graph advanced a different number of steps")
    if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():
        raise FloatingPointError("Nonfinite variable-step Galerkin trajectory")
    a0, *_ = bridge_weights(startup_rate/late_rate)
    row = dict(model.metadata, steps=steps, horizon=horizon, time_order=2, dt=1/startup_rate,
               dt_role="spatial_reconstruction", step_schedule="sbdf2_single_bridge",
               startup_rate=startup_rate, late_rate=late_rate, switch_time=switch_time, switch_step=split,
               late_dt=1/late_rate, bridge_ratio=startup_rate/late_rate, bridge_a0=a0,
               bridge_effective_dt=(1.5/a0)/late_rate, implementation="native-fp64-cufft-precontracted-variable-sbdf2-prepared",
               preparation=preparation, phase_preparation_shared=(preparation != "module_only"),
               compiled_cuda_modules=(2 if preparation == "shared_only" else 1),
               reconstruction_class="basis_recurrence_model.RecurrenceGalerkin", graph_steps=graph_steps,
               nonlinear_evaluations=executed, marching_fft_batches=2*executed, priming_fft_batches=2,
               nonlinear_by_phase=dict(startup=before_steps+1, late=after_steps),
               inverse_field_transforms=3*executed, forward_field_transforms=2*executed,
               bridge_extra_fft_batches=0, fft_radial_lines=q, device_final_step=int(final[0]), host_setup_seconds=setup,
               runtime_init_seconds=runtime_setup, upload_seconds=upload, plan_seconds=plan,
               graph_setup_seconds=graph, march_seconds=march, download_seconds=download,
               seconds=time.perf_counter()-start, timing_scope="Diagnostic accuracy run without isolation")
    return case.Solution(model, checkpoints, snapshots, schedule.probe_times, probes[:, 0], probes[:, 1], row)
