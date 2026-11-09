"""Short FP64 band-equilibration audit against unscaled high-precision equations."""
import argparse
from decimal import Decimal, localcontext
from pathlib import Path
import time
from unittest.mock import patch

import cpu_formal as base
import fd_full
from cpu_formal_fd_equilibrate import CaptureBandSolve, PROTOCOL
from scipy import sparse

np = base.case.np
HERE = Path(__file__).resolve().parent
VARIANTS = ("original", "row_max", "row_pow2", "row_col_pow2")
MODES = (0, 1, 8, 32, 64, 128)
ZERO = Decimal(0)


def scale_for(norm, power_two):
    if np.any(norm <= 0.) or not np.isfinite(norm).all():
        raise ValueError("Cannot equilibrate an empty or nonfinite row or column")
    return np.ldexp(np.ones_like(norm), 1-np.frexp(norm)[1]) if power_two else 1/norm


def maxima(matrix, axis):
    return np.asarray(abs(matrix).max(axis=axis).toarray()).ravel()


class ScaledLU:
    def __init__(self, matrices, variant):
        t0 = time.perf_counter()
        self.rows = np.ones((len(matrices), matrices[0].shape[0]))
        self.columns = np.ones_like(self.rows)
        scaled = []
        for i, matrix in enumerate(matrices):
            if variant == "original":
                scaled.append(matrix)
                continue
            powers = variant != "row_max"
            self.rows[i] = scale_for(maxima(matrix, 1), powers)
            a = sparse.diags(self.rows[i]) @ matrix
            if variant == "row_col_pow2":
                self.columns[i] = scale_for(maxima(a, 0), True)
                a = a @ sparse.diags(self.columns[i])
            scaled.append(a.tocsr())
        self.setup_scale_seconds = time.perf_counter()-t0
        t0 = time.perf_counter()
        self.band = fd_full.BandSolve(scaled, "native")
        self.setup_factor_seconds = time.perf_counter()-t0
        self.scaled = scaled

    def __call__(self, rhs):
        return self.columns * self.band(self.rows * rhs)


def d64(value):
    return Decimal.from_float(float(value))


def decimal_rows(matrix):
    return [{int(j):d64(v) for j,v in zip(matrix.indices[a:b],matrix.data[a:b]) if v != 0.}
            for a,b in zip(matrix.indptr[:-1],matrix.indptr[1:])]


def decimal_vector(values):
    return [(d64(z.real),d64(z.imag)) for z in values]


def decimal_mv(rows, x):
    return [(sum((a*x[j][0] for j,a in row.items()), ZERO),
             sum((a*x[j][1] for j,a in row.items()), ZERO)) for row in rows]


def complex_vector(values):
    return np.array([complex(float(r),float(i)) for r,i in values])


class OracleBand:
    """Decimal Gaussian elimination, original unscaled matrix, partial pivoting."""
    def __init__(self, matrix_rows, lower):
        self.upper = [r.copy() for r in matrix_rows]
        self.steps = []
        n = len(matrix_rows)
        for j in range(n):
            bound = min(j+lower+1,n)
            pivot = max(range(j,bound),key=lambda i:abs(self.upper[i].get(j,ZERO)))
            self.upper[j],self.upper[pivot] = self.upper[pivot],self.upper[j]
            top = self.upper[j]
            coefficient = top[j]
            after = [(k,v) for k,v in top.items() if k>j]
            modifiers = []
            for i in range(j+1,bound):
                lower_row = self.upper[i]
                value = lower_row.pop(j,ZERO)
                if not value:
                    continue
                factor = value/coefficient
                modifiers.append((i,factor))
                for k,v in after:
                    lower_row[k] = lower_row.get(k,ZERO)-factor*v
            self.steps.append((pivot,modifiers))

    def solve(self, rhs):
        x = rhs.copy()
        for j,(pivot,modifiers) in enumerate(self.steps):
            x[j],x[pivot] = x[pivot],x[j]
            a,b = x[j]
            for i,c in modifiers:
                r,t = x[i]
                x[i] = (r-c*a,t-c*b)
        for j in range(len(x)-1,-1,-1):
            row = self.upper[j]
            r,t = x[j]
            for k,a in row.items():
                if k>j:
                    r -= a*x[k][0]
                    t -= a*x[k][1]
            x[j] = (r/row[j],t/row[j])
        return x


def backward(matrix, rows, x, b):
    residual = decimal_mv(rows, decimal_vector(x))
    rhs = decimal_vector(b)
    difference = complex_vector([(r-u,i-v) for (r,i),(u,v) in zip(residual,rhs)])
    denom = abs(matrix)@abs(x)+abs(b)
    ratio = abs(difference)/np.maximum(denom,np.finfo(float).tiny)
    return dict(componentwise_max=float(ratio.max()),
                componentwise_outer8=float(ratio[-8:].max()),
                residual_over_rhs=float(np.linalg.norm(difference)/
                                        max(np.linalg.norm(b),np.finfo(float).tiny)))


def rel(x, expected):
    return float(np.linalg.norm(x-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny))


def smooth_state(model):
    s = model.s[1:-1]
    length = base.case.LENGTH
    x = s/length
    nonzero = x*x*(1-x)**3
    out = np.broadcast_to((1+.3j)*nonzero,model.rhs_shape).copy()
    alpha = (1+length)/(length*(2+length))
    out[0] = s*(1-alpha*s)
    return out


def dynamic_rhs(solution, index):
    """Use a checkpoint, one held-history start, then the actual SBDF2 RHS algebra."""
    model = solution.model
    t = solution.times[index]
    dt = solution.row["dt"]
    m0,f0 = model.terms(solution.states[index],base.case.speed(t))
    a1,a2 = base.case.speed(np.array([t+dt,t+2*dt]))
    next_state = model.advance(1.5*m0+dt*f0,a1,False)
    m1,f1 = model.terms(next_state,a1)
    b = 2*m1-.5*m0+dt*(2*f1-f0)
    b[1] -= a2*model.lift_bdf
    return b


def factor_record(solver, original, modes):
    fact = solver.band
    pivots = fact.pivots
    diagonal = fact.lower+fact.upper
    growth = {}
    for i in modes:
        max_a = np.max(abs(solver.scaled[i].data))
        growth[str(i)] = dict(upper_growth=float(np.max(abs(fact.factors[i,:diagonal+1]))/max_a),
                              changed_pivots=int(np.count_nonzero(pivots[i]!=original.band.pivots[i])))
    return dict(bandwidth=[fact.lower,fact.upper], scale_seconds=solver.setup_scale_seconds,
                factor_seconds=solver.setup_factor_seconds, scales_bytes=solver.rows.nbytes+
                solver.columns.nbytes, factor_mode_diagnostics=growth)


def audit(spec, protocol, directory):
    path = base.WORK/spec["source"]
    with patch.object(fd_full,"BandSolve",CaptureBandSolve):
        solution = base.open_reference(spec,protocol)
    matrices = solution.model.bdf.matrices
    modes = tuple(k for k in MODES if k<=spec["k"])
    dec = {k:decimal_rows(matrices[k]) for k in modes}
    solvers = {name:ScaledLU(matrices,name) for name in VARIANTS}
    lower = solvers["original"].band.lower
    with localcontext() as context:
        context.prec = 65
        oracles = {k:OracleBand(dec[k],lower) for k in modes}
        rng = np.random.default_rng(54187)
        random = rng.normal(size=solution.model.rhs_shape)+1j*rng.normal(size=solution.model.rhs_shape)
        random[0] = random[0].real
        manufactured = dict(random=random,smooth=smooth_state(solution.model),physical=solution.states[-1])
        right = {name:np.stack([a@x for a,x in zip(matrices,state)])
                 for name,state in manufactured.items()}
        for name,state in manufactured.items():
            for k in modes:
                right[name][k] = complex_vector(decimal_mv(dec[k],decimal_vector(state[k])))
        right.update(dynamic_t24=dynamic_rhs(solution,0),dynamic_t72=dynamic_rhs(solution,-1))
        rhs_path = directory/f"fd_equilibration_rhs_n{spec['n']}.npz"
        if rhs_path.exists():
            raise FileExistsError(rhs_path)
        np.savez_compressed(rhs_path,**right)
        results = []
        for name,b in right.items():
            candidate = {variant:solver(b) for variant,solver in solvers.items()}
            for k in modes:
                original = matrices[k]
                expected = complex_vector(oracles[k].solve(decimal_vector(b[k])))
                row = dict(test=name,mode=k,solves={})
                if name in manufactured:
                    row["rounded_rhs_oracle_vs_known"] = rel(expected,manufactured[name][k])
                for variant,x in candidate.items():
                    x = x[k]
                    record = backward(original,dec[k],x,b[k])
                    record["relative_to_decimal_solution"] = rel(x,expected)
                    if name in manufactured:
                        record["relative_to_known_solution"] = rel(x,manufactured[name][k])
                    row["solves"][variant] = record
                results.append(row)
    return dict(spec=spec, input_sha256=base.digest(path), rhs=rhs_path.name,
                rhs_sha256=base.digest(rhs_path), checks=results,
                original_row_max_range={str(k):float(maxima(matrices[k],1).max()/
                                                     maxima(matrices[k],1).min()) for k in modes},
                setup={name:factor_record(solver,solvers["original"],modes)
                       for name,solver in solvers.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
        default=HERE/"cpu_formal_point1_output"/"fd_equilibration_precision.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = base.read(PROTOCOL)
    rows = []
    for spec in protocol["reference_path"][:2]:
        row = audit(spec,protocol,args.output.parent)
        rows.append(row)
        for name in ("random","smooth","physical","dynamic_t24","dynamic_t72"):
            sample = [r for r in row["checks"] if r["test"]==name]
            print(spec["n"],name,{variant: max(r["solves"][variant]["relative_to_decimal_solution"]
                for r in sample) for variant in VARIANTS},flush=True)
    base.save(args.output,dict(rows=rows,decimal_precision=65,variants=VARIANTS,
        dynamic_definition="At saved t24/t72, use one held-history startup then actual SBDF2 right hand side.",
        scope="Fixed-matrix diagnostic only. All residuals use the original unscaled matrix, "
              "and Decimal factors/solutions are an audit only. Production variants are FP64 native band LU."))


if __name__ == "__main__":
    main()
