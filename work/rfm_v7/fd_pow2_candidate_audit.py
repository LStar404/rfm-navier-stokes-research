"""Check the FP64 row scaling at the preregistered low-n FD candidate sizes."""
from decimal import localcontext
from pathlib import Path
from unittest.mock import patch

from fd_equilibration_precision import (OracleBand, backward, complex_vector, decimal_rows,
    decimal_vector, dynamic_rhs, fd_full, maxima, np, rel)
from fd_pow2_long import Pow2EquilibratedBufferedFD, base
from fd_cpu_buffered import BufferedFD
from cpu_formal_fd_equilibrate import CaptureBandSolve


OUT=Path(__file__).resolve().parent/"hardware_pow2_output"/"fd_candidate_equilibration_audit.json"
MODES=(0,1,8,32,64,80)


def audit(row):
    path=base.OUT/row["artifact"]
    if base.digest(path)!=row["sha256"]:
        raise RuntimeError(f"Qualification artifact changed: {path.name}")
    with patch.object(fd_full,"BandSolve",CaptureBandSolve):
        candidate=base.case.load_solution(path,BufferedFD)
    model=candidate.model
    scaled=base.case.load_solution(path,Pow2EquilibratedBufferedFD).model
    matrices=model.bdf.matrices
    if model.rhs_shape!=scaled.rhs_shape or not np.array_equal(model.lift_bdf,scaled.lift_bdf):
        raise RuntimeError("The comparison must keep the same FD system and lift")
    results=[]
    with localcontext() as context:
        context.prec=65
        dec={k:decimal_rows(matrices[k]) for k in MODES}
        oracles={k:OracleBand(dec[k],model.bdf.lower) for k in MODES}
        for index in (0,-1):
            rhs=dynamic_rhs(candidate,index)
            plain,balanced=model.bdf(rhs),scaled.bdf(rhs)
            for k in MODES:
                exact=complex_vector(oracles[k].solve(decimal_vector(rhs[k])))
                unscaled=backward(matrices[k],dec[k],plain[k],rhs[k])
                equilibrated=backward(matrices[k],dec[k],balanced[k],rhs[k])
                unscaled["relative_to_decimal_solution"]=rel(plain[k],exact)
                equilibrated["relative_to_decimal_solution"]=rel(balanced[k],exact)
                results.append(dict(time=float(candidate.times[index]),mode=k,unscaled=unscaled,
                                    power_of_two=equilibrated,relative_between_variants=rel(balanced[k],plain[k])))
    return dict(name=row["name"],source=path.relative_to(base.WORK).as_posix(),sha256=row["sha256"],
                n=model.metadata["n"],scale_setup_seconds=scaled.metadata["equilibration_setup_seconds"],
                stored_scale_bytes=scaled.first.row_scale.nbytes+scaled.bdf.row_scale.nbytes,
                original_row_max_range={str(k):float(maxima(matrices[k],1).max()/
                                                      maxima(matrices[k],1).min()) for k in MODES},rows=results)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    qualification=base.read(base.OUT/"qualification.json")
    candidates=[r for r in qualification["rows"] if r["method"]=="fd4"]
    if qualification["status"]!="complete" or len(candidates)!=3:
        raise RuntimeError("Expected all three frozen FD4 qualifications")
    results=[audit(row) for row in candidates]
    hashes={p.relative_to(base.WORK).as_posix():base.digest(p) for p in
            (Path(__file__),base.HERE/"fd_equilibration_precision.py",base.HERE/"fd_equilibration_pow2.py",
             base.HERE/"cpu_formal_fd_equilibrate.py",base.HERE/"fd_pow2_long.py")}
    base.save(OUT,dict(source_hashes=hashes,decimal_precision=65,modes=MODES,results=results,
        scope="Fixed original matrices and sampled dynamic right-hand sides at saved t24/t72. "
              "Setup time is diagnostic only. No long trajectory or isolated speed inference."))
    for row in results:
        for method in ("unscaled","power_of_two"):
            print(row["name"],method,{key:max(r[method][key] for r in row["rows"]) for key in
                ("relative_to_decimal_solution","componentwise_max")},flush=True)


if __name__=="__main__":
    main()
