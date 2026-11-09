"""Investigate FD band-system scaling without changing the frozen reference ladder."""
import argparse
from decimal import Decimal, localcontext
from math import fsum
from pathlib import Path
from unittest.mock import patch

import cpu_formal as base
from cpu_formal_reference_point1 import PROTOCOL
import fd_full
from scipy import sparse

np = base.case.np
BandSolve = fd_full.BandSolve


class CaptureBandSolve(BandSolve):
    def __init__(self, matrices, backend):
        self.matrices = tuple(a.tocsr() for a in matrices)
        super().__init__(matrices,backend)


def row_norms(matrix):
    return np.asarray(abs(matrix).max(axis=1).toarray()).ravel()


class EquilibratedBandSolve(BandSolve):
    """Factor D A and apply the same diagonal row scaling to the RHS."""
    def __init__(self, matrices, backend):
        self.row_scale=np.stack([1/row_norms(a) for a in matrices])
        super().__init__([sparse.diags(scale)@matrix for scale,matrix in
                          zip(self.row_scale,matrices)],backend)

    def __call__(self, rhs):
        return super().__call__(rhs*self.row_scale)


class EquilibratedFullFD(base.FullFD):
    def __init__(self, *args, **settings):
        # The patched factory is used only during this instance's construction.
        with patch.object(fd_full,"BandSolve",EquilibratedBandSolve):
            super().__init__(*args,**settings)
        self.metadata["equilibration"]="row_max"


def stable_matvec(matrix, values):
    out=np.empty(matrix.shape[0],dtype=np.complex128)
    for i,(lo,hi) in enumerate(zip(matrix.indptr[:-1],matrix.indptr[1:])):
        products=matrix.data[lo:hi]*values[matrix.indices[lo:hi]]
        out[i]=complex(fsum(products.real),fsum(products.imag))
    return out


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))


def high_precision_rows(matrix, state, additions=None):
    """Reconstruct the real-matrix action accurately from its stored FP64 inputs."""
    exact=np.empty(matrix.shape[0],dtype=np.complex128)
    magnitudes=np.empty(matrix.shape[0])
    with localcontext() as context:
        context.prec=50
        for i,(lo,hi) in enumerate(zip(matrix.indptr[:-1],matrix.indptr[1:])):
            vals=state[matrix.indices[lo:hi]]
            coeff=matrix.data[lo:hi]
            extended=[(Decimal.from_float(float(a)),z) for a,z in zip(coeff,vals)]
            size=float(np.sum(abs(coeff)*abs(vals)))
            if additions is not None:
                value,vector=additions
                extended.append((Decimal.from_float(float(vector[i])),value))
                size+=float(abs(value)*abs(vector[i]))
            result=[sum((a*Decimal.from_float(float(getattr(z,part))) for a,z in extended),
                        Decimal(0)) for part in ("real","imag")]
            exact[i]=complex(*map(float,result))
            magnitudes[i]=size
    return exact,magnitudes


def cancellation_stats(ordinary, exact, magnitudes, s):
    ratio=magnitudes/np.maximum(abs(exact),np.finfo(float).tiny)
    worst=int(np.argmax(ratio))
    return dict(relative_l2=float(relative(ordinary,exact)),outer12_relative_l2=float(
        relative(ordinary[-12:],exact[-12:])),max_abs_arithmetic_change=float(abs(ordinary-exact).max()),
        max_cancellation_ratio=float(ratio[worst]),max_ratio_log_radius=float(s[worst]),
        outer12_max_ratio=float(max(ratio[-12:])))


def cancellation(spec, protocol):
    with patch.object(fd_full,"BandSolve",CaptureBandSolve):
        sol=base.open_reference(spec,protocol)
    model=sol.model
    rows=[]
    for j in (0,len(sol.times)-1):
        state=sol.states[j]
        amplitude=base.case.speed(sol.times[j])
        psi=np.r_[0.,state[1],model.r[-1]*amplitude]
        stencil=(model.d2-sparse.eye(len(model.s),format="csr")[1:-1]).tocsr()
        exact_mass,scale_mass=high_precision_rows(stencil,psi)
        computed_mass=(model.d2@psi)-psi[1:-1]
        metric_mass=cancellation_stats(computed_mass,exact_mass,scale_mass,model.s[1:-1])
        a=model.bdf.matrices[1]
        exact_rhs,scale_rhs=high_precision_rows(a,state[1],(amplitude,model.lift_bdf))
        computed_rhs=a@state[1]+amplitude*model.lift_bdf
        metric_rhs=cancellation_stats(computed_rhs,exact_rhs,scale_rhs,model.s[1:-1])
        rows.append(dict(time=float(sol.times[j]),mass_mode1=metric_mass,
                         reconstructed_bdf_rhs_mode1=metric_rhs))
    return dict(spec=spec,rows=rows)


def diagnose(spec, protocol):
    path=base.WORK/spec["source"]
    with patch.object(fd_full,"BandSolve",CaptureBandSolve):
        solution=base.open_reference(spec,protocol)
    original=solution.model.bdf
    matrices=original.matrices
    test=solution.states[-1]
    backends={"native":original,"lapack":BandSolve(matrices,"scipy"),
              "equilibrated_native":EquilibratedBandSolve(matrices,"native")}
    chosen=(0,1,8,32,64,96,spec["k"])
    rhs=np.stack([a@x for a,x in zip(matrices,test)])
    for i in chosen:
        rhs[i]=stable_matvec(matrices[i],test[i])
    output={name:solve(rhs) for name,solve in backends.items()}
    rows=[]
    for i in chosen:
        matrix=matrices[i]
        norm=row_norms(matrix)
        scale=1/norm
        dense=matrix.toarray()
        d=dict(mode=i,row_max_range=float(norm.max()/norm.min()),
               implicit_condition_infinity=float(np.linalg.cond(dense,p=np.inf)),
               equilibrated_condition_infinity=float(np.linalg.cond(scale[:,None]*dense,p=np.inf)),
               solve={})
        for name,actual in output.items():
            discrepancy=stable_matvec(matrix,actual[i])-rhs[i]
            denominator=abs(matrix)@abs(actual[i])+abs(rhs[i])
            d["solve"][name]=dict(relative_state=relative(actual[i],test[i]),
                max_componentwise_backward=float(np.max(abs(discrepancy)/
                                    np.maximum(denominator,np.finfo(float).tiny))))
        rows.append(d)
    return dict(spec=spec,time=float(solution.times[-1]),reference_sha256=base.digest(path),
                states_compared=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=base.HERE/"cpu_formal_point1_output"/
                        "fd_equilibration_diagnostic.json")
    parser.add_argument("--cancellation",action="store_true")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol=base.read(PROTOCOL)
    if args.cancellation:
        rows=[cancellation(s,protocol) for s in protocol["reference_path"][:2]]
        base.save(args.output,dict(rows=rows,note="50-digit sums using stored FP64 operator coefficients "
            "and saved state. The BDF right-hand side is reconstructed as A*x plus the outer lift. "
            "This does not recover the actual previous-step histories or operator coefficient errors."))
        for row in rows:
            print(row["spec"]["n"],row["rows"][-1],flush=True)
        return
    rows=[diagnose(s,protocol) for s in protocol["reference_path"][:2]]
    base.save(args.output,dict(rows=rows,note="Constructed A*x on stored physical states, not a new "
        "time trajectory. Equilibration is exact algebraically and tested here only as a numerical sensitivity."))
    for row in rows:
        print(row["spec"]["n"],max(r["row_max_range"] for r in row["states_compared"]),
              {name:max(x["solve"][name]["relative_state"] for x in row["states_compared"])
               for name in ("native","lapack","equilibrated_native")},flush=True)


if __name__=="__main__":
    main()
