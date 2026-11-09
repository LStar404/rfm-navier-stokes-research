"""One variable-step SBDF2 bridge for the existing Galerkin CPU operators."""

from dataclasses import dataclass
import time

from cpu_precontracted_native import NativePrecontractedGalerkin, RecurrenceGalerkin, case, np


EARLY = (.5, 1., 2., 4., 6., 8.)
BRIDGE = (8.+1/112, 8.+2/112, 8.125, 8.5, 9., 10., 12., 16.)
LATE = case.SNAPSHOTS
SNAPSHOTS = EARLY+BRIDGE+LATE


@dataclass(frozen=True)
class StepSchedule:
    times: np.ndarray
    switch_step: int
    snapshot_times: np.ndarray
    snapshot_steps: np.ndarray
    probe_times: np.ndarray
    probe_steps: np.ndarray


def step_schedule(*, startup_rate=128, late_rate=112, switch_time=8., horizon=72.,
                  snapshots=SNAPSHOTS, probe_spacing=.125):
    """Integer phase indices preserve absolute observation times at the switch."""
    if min(startup_rate, late_rate, horizon, switch_time, probe_spacing) <= 0:
        raise ValueError("Require positive rates, horizon, switch time and probe spacing")
    h0, h1 = 1/startup_rate, 1/late_rate
    split = min(switch_time, horizon)
    n0 = int(case.step_indices([split], h0)[0])
    n1 = int(case.step_indices([horizon-split], h1)[0])
    times = h0*np.arange(n0+1)
    if n1:
        times = (h0*np.arange(n0+n1+1) if startup_rate == late_rate else
                 np.concatenate((times, split+h1*np.arange(1, n1+1))))
    snap = np.asarray(snapshots, dtype=float)
    if (snap.ndim != 1 or not len(snap) or not np.isfinite(snap).all() or snap[0] <= 0 or
            snap[-1] > horizon or np.any(np.diff(snap) <= 0)):
        raise ValueError("Require positive ordered snapshots within the horizon")
    probes = np.arange(0., horizon+.01*probe_spacing, probe_spacing)

    def locate(values):
        early = values <= split
        indices = np.empty(len(values), dtype=int)
        indices[early] = case.step_indices(values[early], h0)
        indices[~early] = n0+case.step_indices(values[~early]-split, h1)
        if not np.allclose(times[indices], values, atol=1e-10, rtol=0):
            raise ValueError("Observation is not on the mixed time grid")
        return indices

    return StepSchedule(times, n0, snap, locate(snap), probes, locate(probes))


def bridge_weights(ratio):
    a0 = (1+2*ratio)/(1+ratio)
    return a0, 1+ratio, ratio**2/(1+ratio), 1+ratio, ratio


def bridge_rhs(now, past, h, ratio):
    _, b0, b1, e0, e1 = bridge_weights(ratio)
    mass, flux = now
    old_mass, old_flux = past
    return b0*mass-b1*old_mass+h*(e0*flux-e1*old_flux)


def equal_spatial_models(models):
    arrays = ("transform", "mean_transform", "samples", "mean_samples", "M0", "M1", "M_lift",
              "mean_mass", "test", "test_s", "mean_test", "r", "lift", "k2")
    return all(np.array_equal(getattr(models[0], name), getattr(model, name))
               for model in models[1:] for name in arrays)


def raw_terms(model, state, amplitude):
    fields = model.fields_on(state, amplitude, model.samples, model.lift, model.r, model.mean_samples)
    radial, angular = model.kernel(*fields)
    # FullNonlinear produces independent transform outputs on each call.
    return state, amplitude, radial, angular


def contract_raw(model, raw):
    """Use the existing C projection with this stage's precontracted factors."""
    state, amplitude, radial, angular = raw
    arrays = state, radial, angular
    shapes = (model.rhs_shape, (len(model.r), model.k+1), (len(model.r), model.k+1))
    if any(x.shape != shape or x.dtype != np.complex128 or not x.flags.aligned or
           x.strides[1] != 16 or x.strides[0] % 16 or x.strides[0] <= 0
           for x, shape in zip(arrays, shapes)):
        raise ValueError("Require positive mode-contiguous complex bridge arrays")
    mass = np.empty(model.rhs_shape, dtype=np.complex128)
    flux = np.empty_like(mass)
    pointers = [x.ctypes.data for x in (*model.native_operators, *arrays, mass, flux)]
    strides = [x.strides[0]//16 for x in arrays]
    model._native(*pointers, model.width, len(model.r), model.k+1, *strides)
    mass[:, 1] += amplitude*model.pc_mass_lift
    return mass, flux


def solve_cpu_switch(model_class, *, startup_rate=128, late_rate=112, switch_time=8., horizon=72.,
                     snapshots=SNAPSHOTS, probe_spacing=.125, **config):
    """Advance from rest and bridge once, with no extra nonlinear evaluations."""
    started = time.perf_counter()
    if not issubclass(model_class, RecurrenceGalerkin) or config.get("backend", "fft") != "fft":
        raise ValueError("The switch path requires recurrence Galerkin and dealiased FFT")
    schedule = step_schedule(startup_rate=startup_rate, late_rate=late_rate, switch_time=switch_time,
                             horizon=horizon, snapshots=snapshots, probe_spacing=probe_spacing)
    steps = len(schedule.times)-1
    n0 = schedule.switch_step
    mixed = steps > n0 and startup_rate != late_rate
    native = issubclass(model_class, NativePrecontractedGalerkin)
    if mixed and n0 < 2:
        raise ValueError("The switch requires two completed startup steps")
    h0, h1 = 1/startup_rate, 1/late_rate
    ratio = h1/h0
    a0, *_ = bridge_weights(ratio)
    scale = 1.5/a0
    old = model_class(dt=h0, **config)
    if mixed:
        new = model_class(dt=h1, **config)
        # A_eff=(3/2)M+(scale*h1)K = scale*A_bridge, including its stored lift.
        bridge = model_class(dt=scale*h1, **config)
        if not equal_spatial_models((old, new, bridge)):
            raise ArithmeticError("Changing time factors changed the spatial coordinate system")
    else:
        new = bridge = old
    counts = {"startup": 0, "late": 0}

    def instrument(model, phase):
        original = model.kernel

        def counted(*fields):
            result = original(*fields)
            counts[phase] += 1
            return result

        model.kernel = counted
        return original

    old_kernel = instrument(old, "startup")
    new_kernel = instrument(new, "late") if mixed else None
    setup_seconds = time.perf_counter()-started
    amplitudes = case.speed(schedule.times)
    observed = dict(zip(schedule.probe_steps, range(len(schedule.probe_steps))))
    snapshots_set = set(schedule.snapshot_steps)
    probes = np.zeros((len(observed), len(case.PROBES)))
    transverse, vorticity = probes, np.zeros_like(probes)
    zero = np.zeros(old.rhs_shape, dtype=np.complex128)
    now = past = (zero, zero)
    states = []
    raw_previous = raw_switch = None
    try:
        for i in range(1, steps+1):
            active = new if mixed and i > n0 else old
            g = amplitudes[i]
            if mixed and i == n0+1:
                if native:
                    bridge_now, bridge_past = (contract_raw(bridge, x) for x in (raw_switch, raw_previous))
                    now = contract_raw(new, raw_switch)
                else:
                    bridge_now, bridge_past = now, past
                state = bridge.advance(scale*bridge_rhs(bridge_now, bridge_past, h1, ratio), g, False)
            else:
                mass, flux = now
                old_mass, old_flux = past
                h = h1 if mixed and i > n0 else h0
                rhs = mass+h*flux if i == 1 else 2*mass-.5*old_mass+h*(2*flux-old_flux)
                state = active.advance(rhs, g, i == 1)
            if i in snapshots_set:
                if not np.isfinite(state).all():
                    raise FloatingPointError(f"Nonfinite switched state at t={schedule.times[i]:.9g}")
                states.append(state.copy())
            if i in observed:
                j = observed[i]
                transverse[j], vorticity[j] = old.probes(state, g)
                if not np.isfinite(transverse[j]).all() or not np.isfinite(vorticity[j]).all():
                    raise FloatingPointError(f"Nonfinite switched probe at t={schedule.times[i]:.9g}")
            if i == steps:
                break
            if mixed and native and i == n0:
                raw_switch = raw_terms(old, state, g)
                if any(np.shares_memory(a, b) for a, b in zip(raw_previous[2:], raw_switch[2:])):
                    raise ArithmeticError("The raw product history was overwritten before the bridge")
            elif mixed and native and i == n0-1:
                raw_previous = raw_terms(old, state, g)
                past, now = now, contract_raw(old, raw_previous)
            else:
                past, now = now, active.terms(state, g)
    finally:
        old.kernel = old_kernel
        if mixed:
            new.kernel = new_kernel
    evaluations = sum(counts.values())
    if evaluations != steps-1 or (mixed and counts != dict(startup=n0, late=steps-n0-1)):
        raise ArithmeticError(f"Unexpected nonlinear work for {steps} steps: {counts}")
    row = dict(old.metadata, steps=steps, horizon=horizon, dt=h0, dt_role="spatial_reconstruction",
               setup_seconds=setup_seconds, seconds=time.perf_counter()-started, time_order=2,
               step_schedule="sbdf2_single_bridge" if mixed else "sbdf2_uniform",
               startup_rate=startup_rate, late_rate=late_rate if mixed else startup_rate,
               switch_time=switch_time if mixed else None, switch_step=n0 if mixed else None,
               late_dt=h1 if mixed else h0, bridge_ratio=ratio if mixed else None,
               bridge_a0=a0 if mixed else None, bridge_effective_dt=scale*h1 if mixed else None,
               nonlinear_evaluations=evaluations, nonlinear_by_phase=counts,
               marching_fft_batches=2*evaluations, inverse_field_transforms=3*evaluations,
               forward_field_transforms=2*evaluations, fft_radial_lines=len(old.r))
    return case.Solution(old, np.asarray(states), schedule.snapshot_times, schedule.probe_times,
                         transverse, vorticity, row)
