"""Selected 65-digit solves of the same assembled CPU bridge equations."""

from decimal import localcontext
from pathlib import Path

from cpu_sbdf2_switch import (
    NativePrecontractedGalerkin, RecurrenceGalerkin, bridge_rhs, bridge_weights, case,
    contract_raw, np, raw_terms, solve_cpu_switch,
)
from cpu_sbdf2_switch_pilot import FROZEN, METHODS, OUT, base, config_for, direct, physical_blocks, relative
from fd_equilibration_precision import OracleBand, backward, complex_vector, decimal_rows, decimal_vector
from full_galerkin import inverse_spd
from radial_layers import radial_quadrature
from scipy.sparse import csr_matrix


H0, H1 = 1/128, 1/112
RATIO = H1/H0
A0, *_ = bridge_weights(RATIO)
SCALE = 1.5/A0
DT = SCALE*H1


def frozen_assembly(model):
    """Recover the exact binary64 assembly used in the existing constructor."""
    _, w = radial_quadrature(model.metadata["q"], model.metadata["split"])
    sw = np.sqrt(w)
    f, _, d2 = model.samples
    lf, _, ld2 = model.lift
    mass = model.M0+model.k2[:, None, None]*model.M1
    l1 = (d2-f)*(sw/model.r)[:, None]
    fw = f*(sw/model.r)[:, None]
    cross = l1.T@fw
    delta = (model.k2-1)[:, None, None]
    diffusion = case.NU*((l1.T@l1)-delta*(cross+cross.T)+delta**2*(fw.T@fw))
    k_lift = case.NU*l1.T@((ld2-lf)*sw/model.r)
    lift = 1.5*model.M_lift+DT*k_lift
    mv, md = model.mean_samples
    strain = (mv+md)*sw[:, None]
    mean = 1.5*model.mean_mass+DT*case.NU*(strain.T@strain)
    matrices = 1.5*mass+DT*diffusion
    if not np.array_equal(lift, model.bdf_lift):
        raise ArithmeticError("The reconstructed binary64 lift differs from the frozen one")
    return matrices, mean


def selected_modes(rec, alternate):
    err = np.linalg.norm(rec[:, 1:]-alternate[:, 1:], axis=0)
    norm = np.maximum(np.linalg.norm(alternate[:, 1:], axis=0), np.finfo(float).tiny)
    largest_rel = int(np.argmax(err/norm))+1
    largest_abs = int(np.argmax(err))+1
    return sorted({0, 1, largest_abs, largest_rel}), largest_abs, largest_rel


def check_state(rec_model, native_model, assembly, alternate_blocks, inputs):
    old_state, now_state, old_g, now_g, next_g = inputs
    physical = [rec_model.terms(state, g) for state, g in ((now_state, now_g), (old_state, old_g))]
    raw = [raw_terms(native_model, state, g) for state, g in ((now_state, now_g), (old_state, old_g))]
    contracted = [contract_raw(native_model, item) for item in raw]
    rhs = bridge_rhs(*physical, H1, RATIO)
    rec = rec_model.advance(SCALE*rhs, next_g, False)
    native = native_model.advance(SCALE*bridge_rhs(*contracted, H1, RATIO), next_g, False)
    alternate = direct(rec_model, alternate_blocks, rhs, next_g, A0, H1)
    effective_rhs = SCALE*rhs
    effective_rhs[:, 1] -= next_g*rec_model.bdf_lift
    modes, largest_abs, largest_rel = selected_modes(rec, alternate)
    matrices, mean = assembly
    mw = rec_model.mean_width
    report = []
    for mode in modes:
        if mode == 0:
            matrix, inverse = mean, rec_model.mean_inverse
            values = [x[:mw, 0].real for x in (effective_rhs, rec, native, alternate)]
        else:
            matrix, inverse = matrices[mode-1], rec_model.implicit[mode-1]
            values = [x[:, mode] for x in (effective_rhs, rec, native, alternate)]
        b, x_rec, x_native, x_alternate = values
        if not np.array_equal(inverse_spd(matrix), inverse):
            raise ArithmeticError(f"The reconstructed matrix differs from the frozen solve for mode {mode}")
        rows = decimal_rows(csr_matrix(matrix))
        truth = complex_vector(OracleBand(rows, len(matrix)-1).solve(decimal_vector(b)))
        candidates = dict(recurrence=x_rec, native=x_native, independent_curl_gram=x_alternate,
                          direct_same_binary64_matrix=np.linalg.solve(matrix, b))
        norms = np.sqrt(np.diag(matrix))
        report.append(dict(mode=mode, width=len(matrix), condition_2=float(np.linalg.cond(matrix)),
                           diagonally_scaled_condition_2=float(np.linalg.cond(matrix/norms[:, None]/norms)),
                           matrix_reconstruction_exact=True, candidates={name: dict(
                               forward_relative=relative(candidate, truth),
                               **backward(matrix, rows, candidate, b)) for name, candidate in candidates.items()}))
    return dict(largest_independent_assembly_absolute_mode=largest_abs,
                largest_independent_assembly_relative_mode=largest_rel, selected=report)


def main():
    path = OUT/"fixed_decimal_audit.json"
    if path.exists():
        raise FileExistsError(path)
    protocol = base.read(FROZEN)
    checks = []
    with localcontext() as context:
        context.prec = 65
        for name, seed in METHODS.items():
            config = config_for(protocol, dict(name=name, seed=seed))
            rec = RecurrenceGalerkin(dt=DT, **config)
            native = NativePrecontractedGalerkin(dt=DT, **config)
            assembly = frozen_assembly(rec)
            alternate = physical_blocks(rec)
            rng = np.random.default_rng(819208)
            old, now = [.002*(rng.standard_normal(rec.rhs_shape)+1j*rng.standard_normal(rec.rhs_shape))
                        for _ in range(2)]
            for state in (old, now):
                state[:, 0] = state[:, 0].real
                state[rec.mean_width:, 0] = 0.
            random = check_state(rec, native, assembly, alternate, (old, now, .13-.41j, -.05-.48j, -.06-.5j))
            prefix = solve_cpu_switch(RecurrenceGalerkin, horizon=8., snapshots=(8.-H0, 8.), **config)
            real = check_state(rec, native, assembly, alternate, (*prefix.states, case.speed(8.-H0),
                                                                 case.speed(8.), case.speed(8.+H1)))
            checks.append(dict(name=name, seed=seed, random=random, adjacent_t8=real))
            print(name, [(label, row["mode"], row["condition_2"], row["candidates"]["recurrence"]["forward_relative"],
                          row["candidates"]["native"]["forward_relative"])
                         for label, result in (("random", random), ("adjacent_t8", real))
                         for row in result["selected"]], flush=True)
    result = dict(status="complete", digits=65, protocol_sha256=base.digest(FROZEN),
                  source_sha256=base.digest(Path(__file__)), checks=checks,
                  scope="The Decimal truth uses exactly the reconstructed FP64 matrix and recurrence-assembled RHS. "
                        "Native projection rounds its own contraction and can differ against this common RHS. "
                        "Two rate128 prefixes supply adjacent held states. No additional T72 trajectories are run.")
    base.save(path, result)


if __name__ == "__main__":
    main()
