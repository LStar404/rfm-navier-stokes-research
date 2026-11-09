"""Audit and pilot the single t=8 SBDF2 rate transition on two CPU implementations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cpu_sbdf2_switch import (
    EARLY, SNAPSHOTS, NativePrecontractedGalerkin, RecurrenceGalerkin, bridge_rhs, bridge_weights,
    case, contract_raw, np, raw_terms, solve_cpu_switch, step_schedule,
)
import cpu_formal as base
from radial_layers import radial_quadrature


HERE = Path(__file__).resolve().parent
OUT = HERE/"cpu_sbdf2_switch_pilot_output"
FROZEN = OUT/"protocol.json"
R128 = HERE/"r128_operator_qualification_output"
REPEAT = HERE/"cpu_native_joint_repeat_output"/"protocol.json"
METHODS = {"rfm37_k72": 7331, "leg37_k72": 17}
MODELS = {"cpu_recurrence": RecurrenceGalerkin, "cpu_native_contract": NativePrecontractedGalerkin}
GATE = 1e-5


def relative(a, b):
    denominator = max(float(np.linalg.norm(b)), np.finfo(float).tiny)
    return float(np.linalg.norm(a-b))/denominator


def frozen():
    prior = base.read(REPEAT)
    control = base.read(R128/"runs.json")
    qualification = base.read(R128/"qualification.json")
    if control["status"] != "complete" or not qualification["passes"]:
        raise RuntimeError("Need the completed rate-128 control cohort")
    specs = {s["name"]: s for s in prior["inputs"]["solve_protocol"]["variants"] if s["name"] in METHODS}
    rows = []
    for name, seed in METHODS.items():
        for arm in MODELS:
            wanted = dict(name=name, seed=seed, arm=arm)
            row, = [r for r in control["rows"] if r["job"] == wanted]
            if row["status"] != "complete":
                raise RuntimeError(f"Missing completed archived control: {wanted}")
            path = R128/row["artifact"]
            if base.digest(path) != row["sha256"]:
                raise RuntimeError(f"Archived control hash changed: {path.name}")
            rows.append(dict(job=wanted, control=path.relative_to(HERE).as_posix(), sha256=row["sha256"]))
    sources = dict(prior["inputs"]["source_hashes"])
    for name in ("cpu_sbdf2_switch.py", Path(__file__).name):
        sources[f"rfm_v7/{name}"] = base.digest(HERE/name)
    for name, digest in sources.items():
        if base.digest(base.WORK/name) != digest:
            raise RuntimeError(f"Frozen solver dependency changed: {name}")
    return dict(status="frozen", rates=[128, 112], switch_time=8., horizon=72., snapshots=list(SNAPSHOTS),
                probe_spacing=.125, implementation_gate=GATE, original_protocol_sha256=base.digest(REPEAT),
                original_qualification_sha256=base.digest(R128/"qualification.json"),
                source_hashes=sources, specs=specs, jobs=rows, solve_protocol=prior["inputs"]["solve_protocol"],
                scope="Paired CPU pilot and work counts. Timing samples are diagnostic.")


def config_for(protocol, job):
    return base.config(protocol["specs"][job["name"]], protocol["solve_protocol"], job["seed"])[1]


def physical_blocks(model):
    """Assemble the diffusion Gram directly from each curl mode for the audit."""
    s, w = radial_quadrature(model.metadata["q"], model.metadata["split"])
    if not np.array_equal(case.radius(s), model.r):
        raise ArithmeticError("Audit quadrature differs from the time solver")
    f, _, d2 = model.samples
    lf, _, ld2 = model.lift
    scale = np.sqrt(w)/model.r
    curl = (d2[None] - model.k2[:, None, None]*f[None])*scale[None, :, None]
    diffusion = case.NU*(curl.transpose(0, 2, 1) @ curl)
    k_lift = case.NU*curl[0].T @ ((ld2-lf)*scale)
    strain = model.mean_samples.sum(axis=0)*np.sqrt(w)[:, None]
    return (model.M0+model.k2[:, None, None]*model.M1, diffusion, k_lift,
            model.mean_mass, case.NU*(strain.T @ strain))


def direct(model, blocks, rhs, g, a0, h):
    mass, diffusion, k_lift, mean_mass, mean_diffusion = blocks
    actual = rhs.copy()
    actual[:, 1] -= g*(a0*model.M_lift+h*k_lift)
    state = np.zeros_like(actual)
    matrix = a0*mass+h*diffusion
    state[:, 1:] = np.linalg.solve(matrix, actual[:, 1:].T[..., None])[..., 0].T
    count = model.mean_width
    mean_matrix = a0*mean_mass+h*mean_diffusion
    state[:count, 0] = np.linalg.solve(mean_matrix, actual[:count, 0].real)
    return state


def backward_error(model, blocks, state, rhs, g, a0, h):
    """Componentwise original-equation defect before any inverse application."""
    mass, diffusion, k_lift, mean_mass, mean_diffusion = blocks

    def measure(matrix, value, target):
        error = np.abs((matrix @ value[..., None])[..., 0]-target)
        scale = (np.abs(matrix) @ np.abs(value[..., None]))[..., 0]+np.abs(target)
        return float(np.max(error/np.maximum(scale, np.finfo(float).tiny)))

    forced = rhs[:, 1:].T.copy()
    forced[0] -= g*(a0*model.M_lift+h*k_lift)
    count = model.mean_width
    return max(measure(a0*mass+h*diffusion, state[:, 1:].T, forced),
               measure(a0*mean_mass+h*mean_diffusion, state[:count, 0].real, rhs[:count, 0].real))


def polynomial_audit(h, prior_h):
    r = h/prior_h
    a0, b0, b1, e0, e1 = bridge_weights(r)
    m = lambda t: .25+.73*t+.18*t*t
    f = lambda t: -.31+.27j+(.14-.11j)*t
    derivative = (a0*m(0.)-b0*m(-h)+b1*m(-h-prior_h))/h
    extrapolated = e0*f(-h)-e1*f(-h-prior_h)
    result = dict(derivative_error=abs(derivative-.73), extrapolation_error=abs(extrapolated-f(0.)))
    if max(result.values()) > 2e-13:
        raise ArithmeticError(f"Inconsistent bridge polynomial moments: {result}")
    return result


def fixed_audit(settings):
    h0, h = 1/128, 1/112
    r = h/h0
    a0, *_ = bridge_weights(r)
    scale = 1.5/a0
    old = RecurrenceGalerkin(dt=h0, **settings)
    bridge = RecurrenceGalerkin(dt=scale*h, **settings)
    native_old = NativePrecontractedGalerkin(dt=h0, **settings)
    native_bridge = NativePrecontractedGalerkin(dt=scale*h, **settings)
    native_next = NativePrecontractedGalerkin(dt=h, **settings)
    blocks = physical_blocks(old)
    rng = np.random.default_rng(819208)
    state_past, state_now = [(.002*(rng.standard_normal(old.rhs_shape)+1j*rng.standard_normal(old.rhs_shape)))
                             for _ in range(2)]
    for state in (state_past, state_now):
        state[:, 0] = state[:, 0].real
        state[old.mean_width:, 0] = 0.
    g_past, g_now, g_future, g_follow = .13-.41j, -.05-.48j, -.06-.5j, -.07-.49j
    physical = [old.terms(state, g) for state, g in ((state_now, g_now), (state_past, g_past))]
    captured = [raw_terms(native_old, state, g) for state, g in ((state_now, g_now), (state_past, g_past))]
    if not all(np.array_equal(a, b) for raw, state, g in zip(captured, (state_now, state_past), (g_now, g_past))
               for a, b in zip(contract_raw(native_old, raw), native_old.terms(state, g))):
        raise ArithmeticError("Raw native projection differs from the frozen operator")
    rhs = bridge_rhs(*physical, h, r)
    exact = direct(old, blocks, rhs, g_future, a0, h)
    rec = bridge.advance(scale*rhs, g_future, False)
    native_terms = [contract_raw(native_bridge, item) for item in captured]
    native = native_bridge.advance(scale*bridge_rhs(*native_terms, h, r), g_future, False)
    rhs_next = bridge_rhs(old.terms(native, g_future), physical[0], h, 1.)
    expected_next = direct(old, blocks, rhs_next, g_follow, 1.5, h)
    mapped = native_next.advance(bridge_rhs(native_next.terms(native, g_future), contract_raw(native_next, captured[0]), h, 1.),
                                 g_follow, False)
    zeros = np.zeros(old.rhs_shape, dtype=np.complex128)
    first = old.advance(zeros.copy(), g_future, True)
    expected_first = direct(old, blocks, zeros, g_future, 1., h0)
    uniform = 2*physical[0][0]-.5*physical[1][0]+h0*(2*physical[0][1]-physical[1][1])
    bitwise_uniform = np.array_equal(uniform, bridge_rhs(*physical, h0, 1.))
    baseline_uniform = old.advance(uniform.copy(), g_future, False)
    baseline_exact = direct(old, blocks, uniform, g_future, 1.5, h0)
    errors = dict(rec_bridge=relative(rec, exact), native_bridge=relative(native, exact),
                  rec_mean=relative(rec[:old.mean_width, 0], exact[:old.mean_width, 0]),
                  native_mean=relative(native[:old.mean_width, 0], exact[:old.mean_width, 0]),
                  native_first_uniform_after_bridge=relative(mapped, expected_next), euler=relative(first, expected_first))
    residuals = dict(rec_bridge=backward_error(old, blocks, rec, rhs, g_future, a0, h),
                     native_bridge=backward_error(old, blocks, native, rhs, g_future, a0, h),
                     native_first_uniform_after_bridge=backward_error(old, blocks, mapped, rhs_next, g_follow, 1.5, h),
                     euler=backward_error(old, blocks, first, zeros, g_future, 1., h0))
    if (max(*errors.values(), *residuals.values()) > 1e-8 or not bitwise_uniform or
            any(np.any(x[:, 0].imag) or np.any(x[old.mean_width:, 0]) for x in (rec, native, mapped))):
        raise ArithmeticError(f"Bridge, mean, or initial Euler audit failed: {errors}, {residuals}")
    return dict(errors=errors, original_equation_backward_error=residuals, raw_native_projection_exact=True,
                uniform_reference=dict(forward=relative(baseline_uniform, baseline_exact),
                                       backward=backward_error(old, blocks, baseline_uniform, uniform, g_future, 1.5, h0)),
                constant_rate_rhs_exact=bitwise_uniform, polynomial=polynomial_audit(h, h0))


def raw_equal(a, b):
    return all(np.array_equal(getattr(a, key), getattr(b, key)) for key in ("states", "transverse", "vorticity"))


def audit(protocol):
    path = OUT/"fixed_audit.json"
    if path.exists():
        result = base.read(path)
        if result["protocol_sha256"] != base.digest(FROZEN) or result["status"] != "passed":
            raise RuntimeError("Fixed bridge audit differs from the frozen pilot")
        return result
    rows = []
    expected = step_schedule()
    if (len(expected.times) != 8193 or expected.switch_step != 1024 or
            tuple(expected.snapshot_steps[6:8]) != (1025, 1026) or len(expected.probe_times) != 577):
        raise RuntimeError("Mixed absolute time grid differs from the fixed design")
    for name, seed in METHODS.items():
        settings = config_for(protocol, dict(name=name, seed=seed))
        fixed = fixed_audit(settings)
        uniform = {}
        for arm, cls in MODELS.items():
            old = case.solve(cls, steps=1024, horizon=8., snapshots=EARLY, **settings)
            prefix = solve_cpu_switch(cls, horizon=8., snapshots=EARLY, **settings)
            same_rate = solve_cpu_switch(cls, startup_rate=128, late_rate=128, horizon=8., switch_time=4.,
                                         snapshots=EARLY, **settings)
            uniform[arm] = dict(t8_prefix_exact=raw_equal(prefix, old), same_rate_split_exact=raw_equal(same_rate, old),
                                nonlinear_evaluations=prefix.row["nonlinear_evaluations"])
            if not all(uniform[arm][k] for k in ("t8_prefix_exact", "same_rate_split_exact")):
                raise ArithmeticError(f"Uniform-rate trajectory changed for {name}/{arm}")
        rows.append(dict(name=name, seed=seed, **fixed, uniform=uniform))
        print("FIXED", name, fixed["errors"], fixed["original_equation_backward_error"], uniform, flush=True)
    result = dict(status="passed", protocol_sha256=base.digest(FROZEN), rows=rows)
    base.save(path, result)
    return result


def subset(solution, times):
    indices = np.flatnonzero(np.isin(solution.times, times))
    if not np.array_equal(solution.times[indices], times):
        raise ValueError("Comparison snapshots are not shared at their absolute times")
    return case.Solution(solution.model, solution.states[indices], solution.times[indices], solution.probe_times,
                         solution.transverse, solution.vorticity, solution.row)


def run_one(item, protocol):
    job = item["job"]
    cls = MODELS[job["arm"]]
    target = OUT/f"{job['name']}_{job['seed']}_{job['arm']}.npz"
    report = target.with_suffix(".json")
    if report.exists():
        saved = base.read(report)
        if saved["job"] != job or ("artifact" in saved and base.digest(target) != saved["sha256"]):
            raise RuntimeError("Pilot result differs from the frozen job")
        return saved
    if target.exists():
        raise FileExistsError(target)
    print("START", job, flush=True)
    config = config_for(protocol, job)
    row = dict(job=job)
    try:
        solution = solve_cpu_switch(cls, **config)
        control_path = HERE/item["control"]
        if base.digest(control_path) != item["sha256"]:
            raise RuntimeError("Uniform rate128 control changed")
        control = case.load_solution(control_path, cls)
        match = subset(solution, control.times)
        probes_early = solution.probe_times <= 8.
        early_exact = np.array_equal(match.states[:len(EARLY)], control.states[:len(EARLY)]) and all(
            np.array_equal(getattr(solution, key)[probes_early], getattr(control, key)[probes_early])
            for key in ("transverse", "vorticity"))
        grids = base.grid_set()
        shared = case.score(match, case.reference_fields(control, grids), grids)
        if not early_exact or solution.row["nonlinear_evaluations"] != 8191:
            raise ArithmeticError("Startup rate128 prefix or actual nonlinear work changed")
        temp = target.with_suffix(".tmp.npz")
        case.store_solution(temp, solution, config)
        temp.replace(target)
        row.update(status="complete", artifact=target.name, sha256=base.digest(target), metadata=solution.row,
                   uniform128_shared_times=list(control.times), uniform128_shared_fields=shared,
                   uniform128_shared_worst=case.maxima(shared), uniform128_prefix_exact=early_exact,
                   uniform128_probe_relative={k:relative(getattr(solution, k), getattr(control, k))
                                              for k in ("transverse", "vorticity")}, wake=base.wake(solution))
    except (ArithmeticError, ValueError, OSError, RuntimeError) as exc:
        row.update(status="failed", failure=str(exc))
    base.save(report, row)
    print("DONE", job, row["status"], row.get("uniform128_shared_worst", row.get("failure")), flush=True)
    return row


def execute(protocol, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_one, item, protocol) for item in protocol["jobs"]]
        rows = [job.result() for job in as_completed(jobs)]
    comparisons = []
    for name, seed in METHODS.items():
        group = {row["job"]["arm"]: row for row in rows if row["job"]["name"] == name}
        if any(group[arm]["status"] != "complete" for arm in MODELS):
            comparisons.append(dict(name=name, seed=seed, passed=False, failure="Incomplete paired run"))
            continue
        old, new = [case.load_solution(OUT/group[arm]["artifact"], cls) for arm, cls in MODELS.items()]
        grids = base.grid_set()
        fields = case.score(new, case.reference_fields(old, grids), grids)
        worst = case.maxima(fields)
        probes = {k:relative(getattr(new, k), getattr(old, k)) for k in ("transverse", "vorticity")}
        comparisons.append(dict(name=name, seed=seed, fields=fields, worst=worst, probes=probes,
                                passed=max(*worst.values(), *probes.values()) <= GATE))
    result = dict(status="complete", protocol_sha256=base.digest(FROZEN), rows=rows, comparisons=comparisons)
    base.save(OUT/"pilot.json", result)
    for row in comparisons:
        print("PAIR", row["name"], row["passed"], row.get("worst"), row.get("probes"), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fixed-only", action="store_true")
    cli.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = cli.parse_args()
    protocol = frozen()
    if FROZEN.exists():
        if base.read(FROZEN) != protocol:
            raise RuntimeError("Frozen bridge pilot changed")
    else:
        base.save(FROZEN, protocol)
    audit(protocol)
    if not args.fixed_only:
        if (OUT/"pilot.json").exists():
            raise FileExistsError(OUT/"pilot.json")
        execute(protocol, args.workers)
