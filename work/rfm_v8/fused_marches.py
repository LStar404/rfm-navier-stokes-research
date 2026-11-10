"""Generated matched warp-only and fused CUDA marches. See make_fused_marches.py."""
from contextlib import ExitStack
import ctypes as C
import time

from fused_march_runtime import FusedMarchRuntime, nonlinear_metadata, nonlinear_nodes
from cuda_environment_native import I, P, bind, call
from galerkin_cuda_native import case, np, operators
from fd_cuda_native import FullFD, BandLaunch, EventTrace, radial_operators, precompute_dense
from galerkin_minimal_model import RecurrenceGalerkin as BufferedGalerkin
from sbdf2_schedule_minimal import SNAPSHOTS, bridge_weights, step_schedule
from galerkin_cuda_variable_minimal import bridge_launch
from galerkin_phase_shared_minimal import phase_operators_shared


def solve_uniform_fused(*, engine, steps, horizon=72., snapshots=case.SNAPSHOTS, probe_spacing=.125,
                 graph_steps=24, profile_steps=0, toolkit=None, **config):
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
        rt = resources.enter_context(FusedMarchRuntime("uniform",engine,n,toolkit))
        runtime_setup = time.perf_counter() - t
        driver = rt.driver
        t = time.perf_counter()
        uploaded = [rt.upload(resources, a) if a is not None else None for a in (
            states, amplitudes, np.array([1], dtype=np.int32), snap_slots,
            model.time_jets, mean, model.r, model.lift_modes, linear, radial, angular,
            mass_lift, bdf_lift, probe, probe_lift, checkpoints, probes,
            np.zeros((3, q, half), dtype=np.complex128), None if rt.fused else np.zeros((3, q, n)),
            None if rt.fused else np.zeros((2, q, n)), np.zeros((2, q, half), dtype=np.complex128),
            np.zeros((2, q, half), dtype=np.complex128))]
        (state, amps, index, slots, jets, avg, radius, lift, lm, rm, am,
         ml, bl, proj, pl, save, pr, spectrum, physical, products, transformed,
         previous) = uploaded
        upload_time = time.perf_counter() - t
        t = time.perf_counter()
        angular_nodes, prime, angular_names = nonlinear_nodes(rt,resources,"gal",n,k,q,spectrum,physical,
                                                              products,transformed,previous)
        form = rt.launch("spectrum", q * half, state, amps, index, jets, avg, radius,
                         lift, spectrum, w, k, q, half)
        # Product and history run within the registered angular nodes.
        advance = rt.launch("advance", frame, state, amps, index, transformed, previous,
                            lm, rm, am, ml, bl, w, k, q, half)
        observe = rt.launch("observe", 1, state, amps, index, slots, proj, pl, save, pr,
                            frame, stride, 1)
        first = rt.launch("observe", 1, state, amps, index, slots, proj, pl, save, pr,
                          frame, stride, 0)

        stage_names = ("radial_reconstruction", *angular_names, "projection_and_advance", "probes_checkpoints_and_step")
        stages = (form, *angular_nodes, advance, observe)

        def body():
            for stage in stages:
                stage()

        for node in prime:
            node()
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
    row.update(nonlinear_metadata(rt,steps-1,q,"gal"))
    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)


def solve_combined_fused(*, engine, preparation="combined", startup_rate=128, late_rate=112, switch_time=8., horizon=72., snapshots=SNAPSHOTS,
                      probe_spacing=.125, graph_steps=24, toolkit=None, **config):
    start = time.perf_counter()
    if preparation != "combined":
        raise ValueError("Only the qualified combined preparation is registered")
    phase_builder = phase_operators_shared
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
        rt = resources.enter_context(FusedMarchRuntime("combined",engine,n,toolkit))
        runtime_setup = time.perf_counter()-t
        t = time.perf_counter()
        arrays = (states, amplitudes, np.array([1], dtype=np.int32), snap_slots, probe_slots,
                  model.time_jets, mean, model.r, model.lift_modes, probe, probe_lift, checkpoints, probes,
                  np.zeros((3, q, half), dtype=np.complex128), None if rt.fused else np.zeros((3, q, n)),
                  None if rt.fused else np.zeros((2, q, n)), np.zeros((2, q, half), dtype=np.complex128),
                  np.zeros((2, q, half), dtype=np.complex128))
        (state, amps, index, slots, p_slots, jets, avg, radius, lift, proj, pl, saved, p_saved,
         spectrum, physical, products, transformed, previous) = [rt.upload(resources, a) if a is not None else None for a in arrays]
        if startup_rate == late_rate:
            shared = tuple(rt.upload(resources, a) for a in packed[0])
            old_pack, bridge_pack, late_pack = (shared,)*3
        else:
            old_pack, bridge_pack, late_pack = [tuple(rt.upload(resources, a) for a in pack) for pack in packed]
        upload = time.perf_counter()-t
        t = time.perf_counter()
        angular_nodes, prime, _ = nonlinear_nodes(rt,resources,"gal",n,k,q,spectrum,physical,
                                                 products,transformed,previous)
        form = rt.launch("spectrum", q*half, state, amps, index, jets, avg, radius, lift, spectrum, w, k, q, half)
        # Product and history run within the registered angular nodes.
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
            for node in angular_nodes:
                node()
            advance()
            observe()

        for node in prime:
            node()
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
    row.update(nonlinear_metadata(rt,executed,q,"gal"))
    return case.Solution(model, checkpoints, snapshots, schedule.probe_times, probes[:, 0], probes[:, 1], row)


def solve_fd_fused(*, engine, steps, horizon=72., snapshots=case.SNAPSHOTS, probe_spacing=.125,
                    graph_steps=24, profile_steps=0, cuda_linear_solve="band",
                    toolkit=None, **config):
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
        rt = resources.enter_context(FusedMarchRuntime("fd",engine,n,toolkit))
        runtime_time = time.perf_counter() - t
        driver = rt.driver
        t = time.perf_counter()
        arrays = (states, amplitudes, np.array([1], dtype=np.int32), snap_slots,
                  columns, weights, lift, model.inv_r[1:-1], model.lift_bdf,
                  probe, probe_lift, checkpoints, probes, linear_data,
                  np.zeros((2, w, modes), dtype=np.complex128), np.zeros((w, modes)),
                  np.zeros((2, w)), np.zeros((4, w, half), dtype=np.complex128),
                  None if rt.fused else np.zeros((4, w, n)), None if rt.fused else np.zeros((w, n)),
                  np.zeros((w, half), dtype=np.complex128), np.zeros((w, half), dtype=np.complex128),
                  np.zeros((w, modes), dtype=np.complex128))
        (state, amps, index, slots, col, weights_gpu, lift_gpu, inv_r, lift_bdf,
         p_op, p_lift, saved, p_saved, linear, mass, mean_terms, mean, spectrum, physical,
         products, transformed, previous, rhs) = [rt.upload(resources, a) if a is not None else None for a in arrays]
        pivot = (rt.upload(resources, model.bdf.pivots) if cuda_linear_solve == "band" else None)
        upload_time = time.perf_counter() - t
        t = time.perf_counter()
        angular_nodes, prime, angular_names = nonlinear_nodes(rt,resources,"fd",n,modes-1,w,spectrum,physical,
            products,transformed,previous,mean_terms=mean_terms,step=index,mean=mean)
        form = rt.launch("fd_fields", w*half, state, amps, index, col, weights_gpu,
                         lift_gpu, inv_r, mass, mean_terms, spectrum, w, modes, half, columns.shape[-1])
        # Products, history and FD mean run within the registered angular nodes.
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
        stage_names = ("radial_stencils", *angular_names, "rhs", "linear_solve", "probe_snapshot_step")
        stages = (form, *angular_nodes, assemble, advance, observe)

        def body():
            for stage in stages:
                stage()

        for node in prime:
            node()
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
    row.update(nonlinear_metadata(rt,steps-1,w,"fd"))
    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)


