"""Fixed-state FP64 FD4 row-centering and discrete Taylor-moment audit."""

import argparse
from decimal import localcontext
import json
from pathlib import Path

import numpy as np

from cpu_formal import digest
from fd_cuda_native import radial_operators
from fd_cuda_pow2_replay import PROTOCOL, reference, write_json
from fd_equilibration_precision import (
    ZERO, complex_vector, d64, decimal_mv, decimal_rows, decimal_vector,
)
from fd_pow2_long import case


HERE = Path(__file__).resolve().parent
KEYS = ("relative_l2", "max_abs", "outer8_relative_l2", "outer8_max_abs")


class MomentRows:
    """Rearrange a stored CSR action while restoring its measured moments."""

    def __init__(self, matrix, coordinates):
        self.rows = []
        for center, row in enumerate(decimal_rows(matrix), start=1):
            indices = np.fromiter(row, dtype=np.intp)
            delta = coordinates[indices] - coordinates[center]
            moment0 = sum(row.values(), ZERO)
            moment1 = sum((a*d64(dx) for a, dx in zip(row.values(), delta)), ZERO)
            self.rows.append((indices, tuple(map(float, row.values())), delta, moment0, moment1))
        self.spans = coordinates[2:] - coordinates[:-2]

    def apply(self, values, taylor=False):
        result = np.empty(len(self.rows), dtype=np.complex128)
        slopes = (values[2:] - values[:-2])/self.spans if taylor else None
        for i, (indices, weights, delta, moment0, moment1) in enumerate(self.rows):
            center = values[i+1]
            residual = values[indices] - center
            restoration = float(moment0)*center
            if taylor:
                residual -= slopes[i]*delta
                restoration += float(moment1)*slopes[i]
            result[i] = sum((a*f for a, f in zip(weights, residual)), 0j) + restoration
        return result

    def identity_error(self, values, original):
        """Verify both rearrangements with the same discrete moments in Decimal."""
        values64 = complex_vector(values)
        slopes = decimal_vector((values64[2:] - values64[:-2])/self.spans)
        worst = ZERO
        for i, (indices, weights, delta, moment0, moment1) in enumerate(self.rows):
            terms = [(int(j), d64(a), d64(dx)) for j, a, dx in zip(indices, weights, delta)]
            for component in (0, 1):
                center = values[i+1][component]
                slope = slopes[i][component]
                centered = sum((a*(values[j][component] - center) for j, a, _ in terms), ZERO)
                centered += moment0*center
                taylor = sum((a*(values[j][component] - center - slope*dx)
                              for j, a, dx in terms), ZERO)
                taylor += moment0*center + moment1*slope
                worst = max(worst, abs(centered-original[i][component]),
                            abs(taylor-original[i][component]))
        return float(worst)


def metrics(actual, expected):
    values = complex_vector(expected)
    error = complex_vector([(a-x, b-y) for (a, b), (x, y) in zip(decimal_vector(actual), expected)])

    def norms(subset):
        e, v = error[subset], values[subset]
        scale = float(np.linalg.norm(v))
        absolute = float(np.max(np.abs(e)))
        return (float(np.linalg.norm(e))/scale if scale else None, absolute)

    return dict(zip(KEYS, (*norms(slice(None)), *norms(slice(-8, None)))))


def quantization_envelope(matrix, values):
    """Magnitude bound for independent half-ULP perturbations of held inputs."""
    weights = abs(matrix)
    re = (weights @ np.abs(np.spacing(values.real)))*.5
    im = (weights @ np.abs(np.spacing(values.imag)))*.5
    envelope = np.hypot(re, im)
    return dict(full_max_abs=float(envelope.max()), outer8_max_abs=float(envelope[-8:].max()))


def exact_curl(model, d1, d2, wall, psi, k):
    if k == 0:
        left = [dict(enumerate(map(d64, model.left_d1)))]
        edge = tuple(x*d64(model.inv_r[0]) for x in decimal_mv(left, psi)[0])
        interior = [tuple((a+b)*d64(r) for a, b in zip(y, z))
                    for y, z, r in zip(decimal_mv(d1, psi), psi[1:-1], model.inv_r[1:-1])]
    else:
        edge = tuple(-x*d64(model.inv_r[0]**2) for x in decimal_mv(wall, psi)[0])
        interior = [tuple(-(a-k*k*b)*d64(r) for a, b in zip(y, z))
                    for y, z, r in zip(decimal_mv(d2, psi), psi[1:-1], model.inv_r[1:-1]**2)]
    return [edge, *interior, (ZERO, ZERO)]


def updated_curl(model, gradient, values, held, k):
    curl = held.copy()
    inv = model.inv_r[1:-1]
    if k == 0:
        curl[1:-1] = (gradient + values[1:-1])*inv
    else:
        curl[1:-1] = -(gradient - k*k*values[1:-1])*inv**2
    return curl


def summary(rows, category, baseline):
    methods = rows[0][category]
    output = {}
    for method in methods:
        output[method] = {}
        for metric in KEYS:
            pairs = [(r[category][method][metric], r[category][baseline][metric], r["time"], r["mode"])
                     for r in rows if r[category] is not None and r[category][method][metric] is not None]
            worst, _, time, mode = max(pairs)
            output[method][metric] = dict(worst=worst, time=time, mode=mode,
                                          better=sum(a < b for a, b, _, _ in pairs),
                                          tied=sum(a == b for a, b, _, _ in pairs),
                                          worse=sum(a > b for a, b, _, _ in pairs))
    return output


def audit(solution, times, modes):
    model = solution.model
    d1, d2, wall = map(decimal_rows, (model.d1, model.d2, model.wall_d2[0]))
    rows1, rows2 = (MomentRows(matrix, model.s) for matrix in (model.d1, model.d2))
    columns, weights, lift, _, _ = radial_operators(model)
    records = []
    for t in times:
        index, = np.flatnonzero(solution.times == t)
        speed = case.speed(t)
        state = solution.states[index]
        model._fill_radial(state, speed)
        for k in modes:
            values = model._psi[:, k]
            psi = decimal_vector(values)
            held = model._omega[:, k]
            held_dec = decimal_vector(held)
            fixed_exact = decimal_mv(d1, held_dec)
            omega_exact = exact_curl(model, d1, d2, wall, psi, k)
            chain_exact = decimal_mv(d1, omega_exact)
            staged = model._spectrum[2, :, k]
            coeff = [(4, 1.)] if k == 0 else [(2, 1.), (3, float(k*k))]
            boundary = lift[2]*speed if k == 1 else np.zeros(state.shape[1], dtype=np.complex128)
            terms = [factor*weights[op]*state[k, columns[op]] for op, factor in coeff]
            composed = sum(term.sum(axis=1) for term in terms)+boundary
            envelope = sum(np.abs(term).sum(axis=1) for term in terms)+np.abs(boundary)
            exact_magnitude = np.abs(complex_vector(chain_exact))
            active = exact_magnitude > np.finfo(float).tiny
            held_centered = rows1.apply(held)
            curl_d1 = k == 0
            raw_derivative = (model.d1 if curl_d1 else model.d2) @ values
            exact_derivative = decimal_mv(d1 if curl_d1 else d2, psi)
            curl_rows = rows1 if curl_d1 else rows2
            centered = curl_rows.apply(values)
            # The Taylor variant only changes the second derivative in nonzero modes.
            taylor = centered if curl_d1 else curl_rows.apply(values, taylor=True)
            omega_centered = updated_curl(model, centered, values, held, k)
            omega_taylor = updated_curl(model, taylor, values, held, k)
            chain = {
                "staged": staged,
                "composed": composed,
                "centered_d1_only": held_centered,
                "centered_both": rows1.apply(omega_centered),
                "taylor_d2_centered_d1": rows1.apply(omega_taylor),
                "exact_curl_rounded_once_then_exact_d1": complex_vector(
                    decimal_mv(d1, decimal_vector(complex_vector(omega_exact)))),
            }
            records.append(dict(
                time=t, mode=k,
                composed_cancellation_ratio=float(np.max(envelope[active]/exact_magnitude[active])),
                fixed_d1_omega={"staged": metrics(staged, fixed_exact),
                                "centered": metrics(held_centered, fixed_exact)},
                fixed_d2_psi=None if curl_d1 else {"raw": metrics(raw_derivative, exact_derivative),
                                                  "centered": metrics(centered, exact_derivative),
                                                  "taylor": metrics(taylor, exact_derivative)},
                curl={name: metrics(candidate, omega_exact) for name, candidate in
                      (("staged", held), ("centered", omega_centered), ("taylor", omega_taylor))},
                pipeline={name: metrics(candidate, chain_exact) for name, candidate in chain.items()},
                held_omega_half_ulp_envelope=quantization_envelope(model.d1, held),
                psi_half_ulp_curl_derivative_envelope=quantization_envelope(
                    model.d1 if curl_d1 else model.d2, values),
                decimal_identity_max_abs=dict(d1_held=rows1.identity_error(held_dec, fixed_exact),
                                              curl_derivative=curl_rows.identity_error(psi, exact_derivative)),
            ))
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    solution = reference(protocol)
    with localcontext() as context:
        context.prec = 65
        records = audit(solution, protocol["times_for_audit"], protocol["modes_for_audit"])
    summaries = {}
    for label, rows in (("all", records), ("k1", [r for r in records if r["mode"] == 1]),
                        ("controls", [r for r in records if r["mode"] != 1])):
        summaries[label] = {
            "fixed_d1_omega": summary(rows, "fixed_d1_omega", "staged"),
            "fixed_d2_psi": summary([r for r in rows if r["fixed_d2_psi"] is not None], "fixed_d2_psi", "raw"),
            "curl": summary(rows, "curl", "staged"),
            "pipeline": summary(rows, "pipeline", "staged"),
        }
    write_json(args.output, dict(
        config=protocol["config"], reference_sha256=protocol["reference_sha256"],
        protocol_sha256=digest(PROTOCOL), code_sha256=digest(Path(__file__)), decimal_digits=65,
        source_audit_sha256=digest(HERE/"fd_cuda_pow2_rhs_gradient_audit.json"),
        minimum_log_spacing=float(np.min(np.diff(solution.model.s))),
        notes={
            "truth": "Stored FP64 coefficients, metric factors and boundary-extended FP64 psi are exact inputs. "
                     "Fixed D1 uses stored stage omega. Pipeline applies both original stencils in Decimal. "
                     "Original composed FP64 includes coefficient rounding from sparse assembly.",
            "variants": "Center i+1 for interior output i. Use stored CSR order. Restore mu0 and mu1 from "
                        "65-digit sums rounded once. Mu1 uses the same FP64 dx as the residual. Slope is the "
                        "neighbor secant. Preserve existing FP64 wall and outer omega in modified stages. "
                        "The Taylor variant is identical to centered at k=0.",
            "metrics": "Errors subtract in Decimal before rounding each error component for norms. Relative L2 "
                       "uses the Decimal truth rounded for its norm. Full is all output rows, outer8 the last 8. "
                       "Curl includes wall and outer rows, unlike D1/D2 interior output.",
            "quantization": "Half-ULP envelopes bound independent input rounding perturbations up to measured "
                            "weights. They are sensitivity diagnostics, not observed input error. Rearranging "
                            "stencils cannot recover information already lost in stored psi or intermediate omega. "
                            "The rounded-once diagnostic also rounds its exact D1 result to FP64 for reporting.",
            "scope": "Fixed states and operators only. No trajectory, GPU work, timing, or altered discretization.",
        }, summaries=summaries, records=records,
    ))
    for category, variants in summaries["k1"].items():
        for name, values in variants.items():
            print(category, name, {k: v["worst"] for k, v in values.items()})


if __name__ == "__main__":
    main()
