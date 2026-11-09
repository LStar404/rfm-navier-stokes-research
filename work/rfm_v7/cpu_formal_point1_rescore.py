"""Score predeclared saved trajectories against the new independent 0.1% ladder."""
import argparse
from pathlib import Path

import cpu_formal as base
from cpu_formal_error_diagnostics import crossing_drift
from cpu_formal_layer_tangent import ROOT as LAYER_ROOT, TangentGalerkin
from cpu_formal_reference_point1 import ROOT, PROTOCOL
from cpu_formal_rfm_bandwidth import ROOT as BAND_ROOT

CASE = base.case


def candidates():
    experiments=[]
    bw=base.read(BAND_ROOT/"march.json")
    layers=base.read(LAYER_ROOT/"march.json")
    if bw["status"]!="complete" or layers["status"]!="complete":
        raise RuntimeError("Finish the preregistered bandwidth and layer trajectories first")
    for row in bw["rows"]:
        seed,scale=row["seed"],row["config"]["scale"]
        path=BAND_ROOT/f"t72_scale{scale}_seed{seed}.npz"
        experiments.append(("rfm49_bandwidth98",seed,path,row.get("sha256"),base.BufferedGalerkin,row))
    for row in layers["rows"]:
        family,variant=row["family"],row["variant"]
        path=LAYER_ROOT/f"{family}_{variant}.npz"
        model=TangentGalerkin if variant=="tangent" else base.BufferedGalerkin
        experiments.append((f"{family}49_{variant}",row["config"]["seed"],path,row.get("sha256"),model,row))
    legacy=base.read(base.OUT/"reference_audit.json")["reference_hashes"]
    for path,name in (
        ("rfm_v6/fd_full_n240_k120_rate320_outer1.npz","fd240_k120_rate320"),
        ("rfm_v6/parallel_refinement/fd280_ref.npz","fd280_k104_rate256"),
        ("rfm_v6/parallel_refinement/fd320_ref.npz","fd320_k104_rate256"),
        ("rfm_v6/parallel_refinement/fd320_ref_tk.npz","fd320_k112_rate320")):
        experiments.append((name,None,base.WORK/path,legacy[path],base.FullFD,{}))
    return experiments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    protocol=base.read(PROTOCOL)
    audit=base.read(ROOT/"reference_audit.json")
    target=ROOT/"saved_candidate_rescore.json"
    if target.exists():
        raise FileExistsError(target)
    for path,digest in audit["reference_hashes"].items():
        if base.digest(base.WORK/path)!=digest:
            raise RuntimeError("An independent reference artifact changed")
    fine=base.open_reference(protocol["reference_path"][-1],protocol)
    grids=base.grid_set()
    truth=CASE.reference_fields(fine,grids)
    rows=[]
    for name,seed,path,digest,model,saved in candidates():
        row=dict(name=name,seed=seed,source=path.relative_to(base.WORK).as_posix(),
                 reference_screen=audit["passes"],sha256=digest)
        if digest is None:
            row.update(numerical_pass=False,failure=saved.get("failure","No saved checkpoint"))
        else:
            if base.digest(path)!=digest:
                raise RuntimeError(f"Candidate artifact changed: {path.name}")
            solution=CASE.load_solution(path,model)
            if not CASE.np.array_equal(solution.times,fine.times):
                raise ValueError(f"Candidate scoring times differ: {path.name}")
            scores=CASE.score(solution,truth,grids)
            worst=CASE.maxima(scores)
            wake=base.wake(solution)
            row.update(metadata=solution.row,scores=scores,worst=worst,wake=wake,
                       absolute_velocity_crossing_lags=crossing_drift(wake,audit["wake"]),
                       numerical_pass=bool(max(worst.values())<=protocol["gate"] and
                                           wake["sustained_alternating_probe_signs"]))
        rows.append(row)
        print(name,seed,row["numerical_pass"],row.get("worst",row.get("failure")),flush=True)
    names=list(dict.fromkeys(r["name"] for r in rows))
    grouped=[]
    for name in names:
        cohort=[r for r in rows if r["name"]==name]
        expected=5 if name=="rfm49_bandwidth98" else 1
        ok=len(cohort)==expected and all(r["numerical_pass"] for r in cohort)
        grouped.append(dict(name=name,count=len(cohort),expected_count=expected,
            exploratory_only=name.startswith("rfm49_") and expected==1,
            numerical_pass=ok,reference_screen=audit["passes"],
            qualified_for_followup=bool(ok and audit["passes"] and
                                        not (name.startswith("rfm49_") and expected==1))))
    base.save(target,dict(protocol=protocol,reference_audit_sha256=base.digest(ROOT/"reference_audit.json"),
        rows=rows,summary=grouped,note="Saved accuracy diagnostics only; no isolated 0.1% timing."))


if __name__=="__main__":
    main()
