"""CPU-only residual audit for folded solves and high-grid curl gradients."""

import argparse
from decimal import localcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cpu_formal import digest
from fd_cuda_native import radial_operators
from fd_cuda_pow2 import folded_mode
from fd_cuda_pow2_replay import PROTOCOL, reference, write_json
from fd_equilibration_precision import (
    OracleBand, backward, complex_vector, d64, decimal_mv, decimal_rows, decimal_vector, dynamic_rhs, rel,
)
import fd_pow2_long
from fd_pow2_long import PowerOfTwoBand, case


class AuditedBand(PowerOfTwoBand):
    def __init__(self, matrices, backend):
        self.original = tuple(matrix.tocsr() for matrix in matrices)
        super().__init__(self.original, backend)


def solve_rows(solver, right_sides, modes):
    band = [solver(rhs) for _, rhs in right_sides]
    report = []
    for mode in modes:
        matrix = solver.original[mode]
        rows = decimal_rows(matrix)
        oracle = OracleBand(rows, solver.lower)
        inverse = folded_mode(solver, mode)
        for (label, rhs), x in zip(right_sides, band):
            b = rhs[mode]
            if not np.any(b):
                continue
            exact = complex_vector(oracle.solve(decimal_vector(b)))
            candidates = {"explicit_scaled_band": x[mode], "folded_dense": inverse@b}
            for method, candidate in candidates.items():
                report.append(dict(mode=mode, rhs=label, method=method, forward_relative=rel(candidate, exact),
                                   **backward(matrix, rows, candidate, b)))
    return report


def relative_gradient(actual, expected):
    diff = actual-expected
    return dict(relative=rel(actual, expected), max_abs=float(abs(diff).max()),
                outer8_relative=rel(actual[-8:], expected[-8:]), outer8_max_abs=float(abs(diff[-8:]).max()))


def gradient_rows(solution, checkpoints, modes):
    model = solution.model
    d1, d2 = map(decimal_rows, (model.d1, model.d2))
    wall = decimal_rows(model.wall_d2[0])
    columns, weights, lift, _, _ = radial_operators(model)
    report = []
    for t, index in checkpoints:
        speed = case.speed(t)
        state = solution.states[index]
        model._fill_radial(state, speed)
        for k in modes:
            psi = decimal_vector(model._psi[:, k])
            if k == 0:
                left = [dict(enumerate(map(d64, model.left_d1)))]
                edge = tuple(x*d64(model.inv_r[0]) for x in decimal_mv(left, psi)[0])
                interior = [tuple((a+b)*d64(r) for a, b in zip(y, z))
                            for y, z, r in zip(decimal_mv(d1, psi), psi[1:-1], model.inv_r[1:-1])]
                coeff = [(4, 1.)]
                boundary = np.zeros(state.shape[1], dtype=np.complex128)
            else:
                edge = tuple(-x*d64(model.inv_r[0]**2) for x in decimal_mv(wall, psi)[0])
                interior = [tuple(-(a-k*k*b)*d64(r) for a, b in zip(y, z))
                            for y, z, r in zip(decimal_mv(d2, psi), psi[1:-1], model.inv_r[1:-1]**2)]
                coeff = [(2, 1.), (3, float(k*k))]
                boundary = lift[2]*speed if k == 1 else np.zeros(state.shape[1], dtype=np.complex128)
            omega = [edge, *interior, (d64(0.), d64(0.))]
            exact = complex_vector(decimal_mv(d1, omega))
            staged = model._spectrum[2, :, k]
            terms = [factor*weights[op]*state[k, columns[op]] for op, factor in coeff]
            composed = sum(term.sum(axis=1) for term in terms)+boundary
            envelope = sum(abs(term).sum(axis=1) for term in terms)+abs(boundary)
            active = abs(exact) > np.finfo(float).tiny
            if not np.any(active):
                continue
            report.append(dict(time=t, mode=k, staged=relative_gradient(staged, exact),
                               composed=relative_gradient(composed, exact),
                               composed_cancellation_ratio=float(np.max(envelope[active]/abs(exact[active])))))
    return report


def run(protocol):
    with patch.object(fd_pow2_long, "PowerOfTwoBand", AuditedBand):
        solution = reference(protocol)
    model = solution.model
    checkpoints = [(float(t), int(np.flatnonzero(solution.times == t)[0])) for t in protocol["times_for_audit"]]
    modes = protocol["modes_for_audit"]
    rhs_first = np.zeros(model.rhs_shape, dtype=np.complex128)
    rhs_first[1] = -case.speed(solution.row["dt"])*model.lift_first
    rhs_bdf = [(f"held_history_start_at_{t:g}", dynamic_rhs(solution, index)) for t, index in checkpoints]
    with localcontext() as context:
        context.prec = 65
        solves = solve_rows(model.first, [("first_step_lift", rhs_first)], (1,))
        solves += solve_rows(model.bdf, rhs_bdf, modes)
        gradients = gradient_rows(solution, checkpoints, modes)
    return dict(reference_sha256=protocol["reference_sha256"], protocol_sha256=digest(PROTOCOL),
                original_unscaled_equation=solves, gradient=gradients,
                minimum_log_spacing=float(np.diff(model.s).min()),
                note="Decimal oracle treats assembled matrix and FD stencil coefficients as exact doubles. "
                     "BDF vectors come from saved flow and one held-history start, followed by the normal SBDF2 equation. "
                     "Folded dense application here uses CPU BLAS. It does not reproduce CUDA summation or FFT bitwise.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = run(protocol)
    write_json(args.output, result)
    for method in ("explicit_scaled_band", "folded_dense"):
        rows = [row for row in result["original_unscaled_equation"] if row["method"] == method]
        print(method, {key: max(row[key] for row in rows) for key in (
            "forward_relative", "componentwise_max", "componentwise_outer8", "residual_over_rhs")})
    for method in ("staged", "composed"):
        print(method, {key: max(row[method][key] for row in result["gradient"]) for key in (
            "relative", "outer8_relative", "outer8_max_abs")})


if __name__ == "__main__":
    main()
