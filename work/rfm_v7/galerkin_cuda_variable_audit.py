"""Check the device bridge against the original equations and archive the mixed-rate march."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import cpu_formal as base
from basis_recurrence_model import native_model
from cpu_sbdf2_switch import SNAPSHOTS, bridge_weights, raw_terms, solve_cpu_switch
from fd_cuda_pow2_replay import write_json
from full_galerkin import radial_quadrature
from galerkin_cuda_native import solve_native
from galerkin_cuda_variable import RecurrenceGalerkin, SwitchRuntime, bridge_launch, case, np, phase_operators, solve_cuda_switch


HERE = Path(__file__).resolve().parent
OUT = HERE/"galerkin_cuda_variable_output"
METHODS = {"rfm37_k72": 7331, "leg37_k72": 17}
FIXED_GATE = 1e-9
PAIR_GATE = 1e-5
SOURCES = ("galerkin_cuda_variable.py", "galerkin_cuda_variable.cu", "galerkin_cuda_variable_audit.py",
           "cpu_sbdf2_switch.py", "galerkin_cuda_native.py", "galerkin_cuda_native.cu",
           "basis_recurrence_model.py")


def source_hashes():
    return {name: base.digest(HERE/name) for name in SOURCES}


def settings(name, seed):
    protocol = base.read(base.PROTOCOL)
    spec = next(s for s in protocol["variants"] if s["name"] == name)
    return base.config(spec, protocol, seed)[1]


def relative(actual, expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), np.finfo(float).tiny))


def direct_equations(model, config):
    s, weights = radial_quadrature(config["q"], config.get("split", True))
    if not np.array_equal(s, model.s):
        raise ArithmeticError("The reference radial quadrature differs")
    f, _, d2 = model.samples
    sw = np.sqrt(weights)
    l1 = (d2-f)*(sw/model.r)[:, None]
    fw = f*(sw/model.r)[:, None]
    cross = l1.T@fw
    shift = (model.k2-1)[:, None, None]
    mass = model.M0+model.k2[:, None, None]*model.M1
    diffusion = case.NU*((l1.T@l1)-shift*(cross+cross.T)+shift**2*(fw.T@fw))
    lift0, _, lift2 = model.lift
    lift_diffusion = case.NU*l1.T@((lift2-lift0)*sw/model.r)
    strain = model.mean_samples.sum(axis=0)*sw[:, None]
    return mass, diffusion, lift_diffusion, case.NU*(strain.T@strain)


def cpu_system(model, operators, states, speeds, ratio, h):
    mass, diffusion, lift_diffusion, mean_diffusion = operators
    a0, b0, b1, e0, e1 = bridge_weights(ratio)
    first, old = [model.terms(state, amp) for state, amp in zip(states, speeds)]
    rhs = b0*first[0]-b1*old[0]+h*(e0*first[1]-e1*old[1])
    rhs[:, 1] -= speeds[2]*(a0*model.M_lift+h*lift_diffusion)
    matrix = a0*mass+h*diffusion
    mean_matrix = a0*model.mean_mass+h*mean_diffusion
    pairs = np.stack((rhs[:, 1:].T.real, rhs[:, 1:].T.imag), axis=-1)
    solved = np.linalg.solve(matrix, pairs)
    answer = np.zeros(model.rhs_shape, dtype=np.complex128)
    answer[:, 1:] = (solved[..., 0]+1j*solved[..., 1]).T
    answer[:model.mean_width, 0] = np.linalg.solve(mean_matrix, rhs[:model.mean_width, 0].real)
    return answer, rhs, matrix, mean_matrix


def errors(actual, system, mean_width):
    expected, rhs, matrix, mean_matrix = system
    x = actual[:, 1:].T
    b = rhs[:, 1:].T
    residual = (matrix@x[..., None])[..., 0]-b
    denominator = (abs(matrix)@abs(x[..., None]))[..., 0]+abs(b)
    mean_x, mean_b = actual[:mean_width, 0], rhs[:mean_width, 0]
    mean_residual = mean_matrix@mean_x-mean_b
    mean_denominator = abs(mean_matrix)@abs(mean_x)+abs(mean_b)
    tiny = np.finfo(float).tiny
    return dict(state_relative=relative(actual, expected), mode1_relative=relative(actual[:, 1], expected[:, 1]),
                mean_relative=relative(actual[:, 0], expected[:, 0]),
                componentwise_backward=float(np.max(abs(residual)/np.maximum(denominator, tiny))),
                mean_componentwise_backward=float(np.max(abs(mean_residual)/np.maximum(mean_denominator, tiny))),
                mean_imag_max=float(np.max(abs(actual[:, 0].imag))))


def raw_gpu_products(model, state, amp):
    _, _, radial, angular = raw_terms(model, state, amp)
    n = model.kernel.n
    result = np.zeros((2, len(model.r), n//2+1), dtype=np.complex128)
    result[0, :, :model.k+1] = n*radial
    result[1, :, :model.k+1] = n*angular
    return result


def fixed_one(name, seed, rate):
    config = settings(name, seed)
    path = HERE/"r128_operator_qualification_output"/f"{name}_{seed}_cpu_recurrence.npz"
    archive = base.read(path.with_suffix(".json"))
    if base.digest(path) != archive["sha256"]:
        raise ArithmeticError("Held-state source digest changed")
    saved = case.load_solution(path, RecurrenceGalerkin)
    i6, = np.flatnonzero(saved.times == 6.)
    i8, = np.flatnonzero(saved.times == 8.)
    older, now = saved.states[i6], saved.states[i8]
    old, _, packed, coeff = phase_operators(128, rate, config)
    direct = direct_equations(old, config)
    h = 1/rate
    speeds = case.speed(np.array([8.-1/128, 8., 8.+h, 8.+2*h]))
    w, k, q = old.width, old.k, len(old.r)
    half, frame = old.kernel.n//2+1, w*(k+1)
    host_states = np.zeros((3, w, k+1), dtype=np.complex128)
    host_states[0], host_states[1] = older, now
    host_index = np.array([1], dtype=np.int32)
    raw_older, raw_now = [raw_gpu_products(old, x, g) for x, g in zip((older, now), speeds)]
    target = cpu_system(old, direct, (now, older), speeds[[1, 0, 2]], 128/rate, h)
    with ExitStack() as stack:
        runtime = stack.enter_context(SwitchRuntime())
        state, amps, index, product, prior = [runtime.upload(stack, a) for a in
            (host_states, speeds, host_index, raw_now, raw_older)]
        bridge, late = [tuple(runtime.upload(stack, a) for a in pack) for pack in packed[1:]]
        launch = bridge_launch(runtime, frame, state, amps, index, product, prior, bridge, w, k, q, half, coeff)
        launch()
        runtime.sync()
        runtime.driver.download(state, host_states)
        first = host_states[2].copy()
        bridge_errors = errors(first, target, old.mean_width)
        second_target = cpu_system(old, direct, (first, now), speeds[[2, 1, 3]], 1., h)
        host_index[0] = 2
        runtime.driver.upload(index, host_index)
        runtime.driver.upload(prior, raw_now)
        runtime.driver.upload(product, raw_gpu_products(old, first, speeds[2]))
        runtime.launch("advance", frame, state, amps, index, product, prior, *late, w, k, q, half)()
        runtime.sync()
        runtime.driver.download(state, host_states)
        second_errors = errors(host_states[0], second_target, old.mean_width)
    passed = all(np.isfinite(v) and v <= FIXED_GATE for row in (bridge_errors, second_errors) for v in row.values())
    return dict(name=name, seed=seed, rate=rate, passed=passed, bridge=bridge_errors, next=second_errors,
                held_input=path.relative_to(HERE).as_posix(), input_sha256=archive["sha256"])


def short_pair(name, seed, uniform):
    config = settings(name, seed)
    rate = 128 if uniform else 112
    times = tuple(t for t in SNAPSHOTS if t <= 9.)
    if uniform:
        times = times[:6]+(8.+1/128, 8.+2/128)+times[8:]
        with native_model():
            cpu_or_uniform = solve_native(steps=9*128, horizon=9., snapshots=times, **config)
    else:
        cpu_or_uniform = solve_cpu_switch(RecurrenceGalerkin, horizon=9., snapshots=times, **config)
    cuda = solve_cuda_switch(late_rate=rate, horizon=9., snapshots=times, **config)
    grids = base.grid_set()
    fields = case.score(cuda, case.reference_fields(cpu_or_uniform, grids), grids)
    peak = case.maxima(fields)
    probes = {key: relative(getattr(cuda, key), getattr(cpu_or_uniform, key)) for key in ("transverse", "vorticity")}
    equal = all(np.array_equal(getattr(cuda, k), getattr(cpu_or_uniform, k)) for k in ("states", "transverse", "vorticity"))
    passed = all(np.isfinite(v) and v <= (FIXED_GATE if uniform else PAIR_GATE) for v in (*peak.values(), *probes.values()))
    return dict(name=name, seed=seed, rate=rate, uniform=uniform, passed=passed, bitwise=equal,
                fields=fields, worst=peak, probes=probes, cuda=cuda.row)


def run_fixed(directory):
    out = directory/"fixed.json"
    if out.exists():
        raise FileExistsError(out)
    hashes = source_hashes()
    held = [fixed_one(name, seed, rate) for name, seed in METHODS.items() for rate in (112, 128)]
    short = [short_pair(name, seed, uniform) for name, seed in METHODS.items() for uniform in (False, True)]
    if source_hashes() != hashes:
        raise ArithmeticError("Bridge source changed during fixed tests")
    passed = all(r["passed"] for r in (*held, *short))
    write_json(out, dict(status="complete", passed=passed, source_hashes=hashes, fixed_gate=FIXED_GATE,
                         short_pair_gate=PAIR_GATE, held=held, short=short,
                         held_scope="Saved t=6 and t=8 states are arbitrary held coefficient vectors, with the physical "
                                    "lift amplitudes near the switch. The device histories use scaled CPU raw product spectra."))
    print("FIXED", passed, max(r[key]["state_relative"] for r in held for key in ("bridge", "next")),
          "same-rate bitwise", [r["bitwise"] for r in short if r["uniform"]], flush=True)
    return passed


def run_march(directory, all_seeds):
    fixed_path = directory/"fixed.json"
    fixed = base.read(fixed_path)
    hashes = source_hashes()
    if not fixed["passed"] or fixed["source_hashes"] != hashes:
        raise ValueError("The bridge source and fixed results must match before the full march")
    journal = directory/"march.json"
    if journal.exists():
        raise FileExistsError(journal)
    protocol = base.read(base.PROTOCOL)
    jobs = [(name, seed) for name, default in METHODS.items() for seed in
            (protocol["seeds"] if all_seeds and name == "rfm37_k72" else (default,))]
    result = dict(status="running", jobs=jobs, rows=[], source_hashes=hashes, fixed_sha256=base.digest(fixed_path),
                  scope="Concurrent accuracy-only full marches. No isolated timing claim.")
    for name, seed in jobs:
        config = settings(name, seed)
        print("START", name, seed, flush=True)
        sol = solve_cuda_switch(**config)
        row = sol.row
        if (row["steps"], row["nonlinear_evaluations"], row["marching_fft_batches"], row["priming_fft_batches"],
                row["bridge_extra_fft_batches"]) != (8192, 8191, 16382, 2, 0) or not np.array_equal(sol.times, SNAPSHOTS):
            raise ArithmeticError("Unexpected full CUDA schedule or nonlinear work counts")
        target = directory/f"{name}_{seed}_cuda_graph.npz"
        if target.exists():
            raise FileExistsError(target)
        tmp = target.with_suffix(".tmp.npz")
        case.store_solution(tmp, sol, config)
        tmp.replace(target)
        result["rows"].append(dict(name=name, seed=seed, artifact=target.name, sha256=base.digest(target),
                                   metadata=row, wake=base.wake(sol)))
        write_json(journal, result)
    if source_hashes() != hashes:
        raise ArithmeticError("Bridge sources changed during the full march")
    result["status"] = "complete"
    write_json(journal, result)
    print("COMPLETE", len(jobs), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixed", "march"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    if args.mode == "fixed":
        raise SystemExit(0 if run_fixed(args.output) else 1)
    run_march(args.output, args.all_seeds)
