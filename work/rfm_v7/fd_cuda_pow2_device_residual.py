"""Apply the actual CUDA BDF kernels to saved-flow RHS and audit the original rows."""

import argparse
from contextlib import ExitStack
from decimal import localcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cpu_formal import digest
from fd_cuda_native import BandLaunch
from fd_cuda_pow2 import Pow2Runtime, folded_inverse
from fd_cuda_pow2_audit import AuditedBand
from fd_cuda_pow2_replay import PROTOCOL, reference, write_json
from fd_equilibration_precision import OracleBand, backward, complex_vector, decimal_rows, decimal_vector, dynamic_rhs, rel
import fd_pow2_long


def run(protocol):
    with patch.object(fd_pow2_long, "PowerOfTwoBand", AuditedBand):
        solution = reference(protocol)
    factors = solution.model.bdf
    w, modes = factors.width, factors.modes
    if factors.lower+factors.upper >= 32:
        raise ValueError("The band update needs all active diagonals in one warp")
    moments = [(t, int(np.flatnonzero(solution.times == t)[0])) for t in protocol["times_for_audit"]]
    rhs_by_time = [(t, dynamic_rhs(solution, index)) for t, index in moments]
    inverse = folded_inverse(factors)
    states = np.zeros((2, w, modes), dtype=np.complex128)
    samples = []
    with ExitStack() as resources:
        runtime = resources.enter_context(Pow2Runtime())
        raw = np.zeros((w, modes), dtype=np.complex128)
        arrays = (states, np.zeros(1, dtype=np.int32), raw, inverse, factors.row_scale.T,
                  factors.factors.transpose(0, 2, 1), factors.pivots)
        state, step, rhs, dense, scales, band, pivot = [runtime.upload(resources, a) for a in arrays]
        dense_solve = runtime.launch("fd_dense", w*modes, state, step, rhs, dense, w, modes)
        scale_rhs = runtime.launch("pow2_scale_rhs", w*modes, rhs, scales, w*modes)
        band_solve = BandLaunch(runtime.driver, runtime.kernels["fd_band"], w, modes, state, step, rhs,
                                band, pivot, w, modes, factors.lower, factors.upper)
        for t, b in rhs_by_time:
            runtime.driver.upload(rhs, np.ascontiguousarray(b.T))
            dense_solve()
            runtime.sync()
            runtime.driver.download(state, states)
            dense_x = states[1].T.copy()
            scale_rhs()
            band_solve()
            runtime.sync()
            runtime.driver.download(state, states)
            samples.append((t, b, dense_x, states[1].T.copy()))
    rows = []
    with localcontext() as context:
        context.prec = 65
        for k in protocol["modes_for_audit"]:
            original = factors.original[k]
            matrix = decimal_rows(original)
            oracle = OracleBand(matrix, factors.lower)
            for t, b, dense_x, band_x in samples:
                exact = complex_vector(oracle.solve(decimal_vector(b[k])))
                for method, candidate in (("folded_dense_cuda", dense_x[k]), ("explicit_band_cuda", band_x[k])):
                    rows.append(dict(time=t, mode=k, method=method, forward_relative=rel(candidate, exact),
                                     **backward(original, matrix, candidate, b[k])))
    return dict(rows=rows, protocol_sha256=digest(PROTOCOL), reference_sha256=protocol["reference_sha256"],
                note="Resident kernel arithmetic measured on raw dynamic BDF right-hand sides. "
                     "The trajectory's first implicit update uses the original scaled CPU solve.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = run(protocol)
    write_json(args.output, result)
    for method in ("folded_dense_cuda", "explicit_band_cuda"):
        rows = [row for row in result["rows"] if row["method"] == method]
        print(method, {key: max(row[key] for row in rows) for key in (
            "forward_relative", "componentwise_max", "componentwise_outer8", "residual_over_rhs")})


if __name__ == "__main__":
    main()
