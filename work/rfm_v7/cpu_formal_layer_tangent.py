"""Matched outer-layer scale-tangent enrichment and provisional trajectory ablation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import cpu_formal as base
from cpu_formal_projection import projection
from radial_layers import layer_jets, radial_quadrature
from scipy.linalg import svdvals

case = base.case
np = case.np
ROOT = base.HERE / "cpu_formal_layer_output"
PROTOCOL = base.HERE / "protocol_layer_tangent.json"


def tangent_jets(s, rates, mean=False):
    """Jets of a rescaled derivative in the exponential layer-rate family."""
    d=(case.LENGTH-s)[:,None]
    z=(s/case.LENGTH)[:,None]
    beta=np.asarray(rates)[None,:]
    e=np.exp(-beta*d)
    if mean:
        h=d*d
        dh=beta*h-2*d
        return np.stack((z*h*e,(h/case.LENGTH+z*dh)*e))
    h=d**3
    dh=beta*h-3*d*d
    ddh=beta*beta*h-6*beta*d*d+6*d
    a=z*z
    da=2*z/case.LENGTH
    dda=2/case.LENGTH**2
    return np.stack((a*h*e,(da*h+a*dh)*e,(dda*h+2*da*dh+a*ddh)*e))


class TangentGalerkin(base.BufferedGalerkin):
    """Keep both v6 mode spaces and add homogeneous scale tangent jets."""

    def __init__(self, dt, m, k, q, *, tangent_rates, **settings):
        self.tangent_rates=tuple(tangent_rates)
        if not self.tangent_rates or any(not np.isfinite(b) or b<=0 for b in self.tangent_rates):
            raise ValueError("Require positive finite tangent rates")
        if m+len(settings.get("layers",()))+len(self.tangent_rates)>=q:
            raise ValueError("Require resolved nominal tangent space")
        super().__init__(dt,m,k,q,**settings)
        if self.mean_width>self.width:
            raise ValueError("The retained mean space must fit the state width")
        self.metadata["tangent_rates"]=self.tangent_rates

    def raw_jets(self, s):
        return np.concatenate((super().raw_jets(s),tangent_jets(s,self.tangent_rates)),axis=-1)

    def raw_mean(self, s):
        return np.concatenate((super().raw_mean(s),tangent_jets(s,self.tangent_rates,mean=True)),axis=-1)


def settings(protocol, family, variant):
    cfg={key:protocol[key] for key in ("m","k","q","scale","mean_order","seed")}
    cfg.update(family=family,split=True,backend="fft",layers=protocol["existing_rates"])
    cls=base.BufferedGalerkin
    if variant=="ordinary":
        cfg["layers"]=cfg["layers"]+protocol["extra_ordinary_rates"]
    elif variant=="tangent":
        cls=TangentGalerkin
        cfg["tangent_rates"]=protocol["tangent_rates"]
    return cls,cfg


def rank_record(model,cfg):
    s,w=radial_quadrature(cfg["q"],cfg["split"])
    raw=model.raw_jets(s)
    mean=model.raw_mean(s)
    cutoff=cfg.get("cutoff",1e-10)
    records={}
    for name,matrix,rank in (
        ("streamfunction",(raw[2]-raw[0])*(np.sqrt(w)/model.r)[:,None],model.width),
        ("mean",(mean[0]+mean[1])*np.sqrt(w)[:,None],model.mean_width)):
        scaled=matrix/np.linalg.norm(matrix,axis=0)
        values=svdvals(scaled,check_finite=True)
        records[name]=dict(nominal_columns=matrix.shape[1],actual_retained_rank=rank,
            condition_retained=float(values[0]/values[rank-1]),relative_cutoff=cutoff,
            scaled_singular_ratios=(values/values[0]).tolist(),
            rank_at_cutoff=int(np.count_nonzero(values>cutoff*values[0])))
    return records


def boundary_check(rates):
    beta=np.asarray(rates)
    interval=np.r_[0.,np.linspace(case.LENGTH-.16,case.LENGTH,123)]
    difference=beta*1e-5
    results={}
    for mean in (False,True):
        raw=tangent_jets(np.array([0.,case.LENGTH]),beta,mean=mean)
        traces=np.stack((raw[0,0],raw[0,1]+raw[1,1])) if mean else np.stack(
            (raw[0,0],raw[1,0],raw[0,1],raw[2,1]))
        exact=-((beta+1+1/case.LENGTH) if mean else (beta+2/case.LENGTH))*tangent_jets(
            interval,beta,mean=mean)
        finite_difference=(layer_jets(interval,beta+difference,mean=mean)-
                           layer_jets(interval,beta-difference,mean=mean))/(2*difference)
        err=float(np.linalg.norm(exact-finite_difference)/np.linalg.norm(exact))
        results["mean" if mean else "streamfunction"]=dict(max_boundary_residual=float(
            abs(traces).max()),relative_rate_derivative_check=err)
        if abs(traces).max()>1e-12 or err>1e-6:
            raise ArithmeticError("Tangent jets fail the homogeneous trace or rate-derivative check")
    return results


def reference(protocol):
    return case.load_solution(base.WORK/protocol["reference"],base.FullFD)


def oracle(protocol, jobs, path):
    ref=reference(protocol)
    grids=base.grid_set()
    target={n:tuple(np.stack(a,axis=-1) for a in zip(
        *(ref.modal(i,s) for i in range(len(ref.times)))))
            for n,(s,_,_) in grids.items()}
    result=dict(status="running",protocol=protocol,check=boundary_check(protocol["tangent_rates"]),
                reference_sha256=base.digest(base.WORK/protocol["reference"]),rows=[],
                note="The sampled numerical-span fit is a representation diagnostic. "
                     "Numerical rank truncation and fixed FD320 reference preclude a continuum bound.")
    base.save(path,result)
    for family,variant in jobs:
        cls,cfg=settings(protocol,family,variant)
        model=cls(dt=1/protocol["rate"],**cfg)
        rows=[]
        for n,(s,_,w) in grids.items():
            for region,weight in (("full",w),("near",w*(case.radius(s)<=2.5))):
                rows.append(dict(grid=n,region=region,**projection(model,s,weight,target[n],ref.times)))
        worst={f"{region}_{label}":max(max(r[label]) for r in rows if r["region"]==region)
               for region in ("full","near") for label in ("velocity","vorticity")}
        result["rows"].append(dict(family=family,variant=variant,rank=rank_record(model,cfg),
                                   worst=worst,rows=rows))
        base.save(path,result)
        print(family,variant,worst,flush=True)
    result["status"]="complete"
    base.save(path,result)


def march_one(protocol,family,variant):
    name=f"{family}_{variant}"
    path=ROOT/f"{name}.npz"
    details=ROOT/f"{name}.json"
    cls,cfg=settings(protocol,family,variant)
    if details.exists():
        old=base.read(details)
        if old["config"]!=cfg or ("sha256" in old and base.digest(path)!=old["sha256"]):
            raise RuntimeError(f"Changed saved tangent experiment {name}")
        return old
    if path.exists():
        path.rename(path.with_suffix(f".orphan_{time.time_ns()}.npz"))
    row=dict(family=family,variant=variant,config=cfg)
    try:
        ref=reference(protocol)
        grids=base.grid_set()
        truth=case.reference_fields(ref,grids)
        with np.errstate(over="ignore",invalid="ignore"):
            sol=case.solve(cls,steps=protocol["horizon"]*protocol["rate"],**cfg)
        scores=case.score(sol,truth,grids)
        temp=path.with_suffix(".tmp.npz")
        case.store_solution(temp,sol,cfg)
        temp.replace(path)
        row.update(status="complete",metadata=sol.row,worst=case.maxima(scores),scores=scores,
                   wake=base.wake(sol),rank=rank_record(sol.model,cfg),sha256=base.digest(path))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError) as exc:
        row.update(status="failed",failure=str(exc))
    base.save(details,row)
    return row


def march(protocol,jobs,path,workers):
    fixed=dict(protocol=protocol, jobs=[list(j) for j in jobs],
               source_sha256=base.digest(Path(__file__)),
               cpu_sources=base.source_hashes(),
               projection_source_sha256=base.digest(base.HERE/"cpu_formal_projection.py"),
               reference_sha256=base.digest(base.WORK/protocol["reference"]))
    if path.exists():
        result=base.read(path)
        if result["frozen"]!=fixed:
            raise RuntimeError("Changed frozen tangent trajectory experiment")
        for row in result["rows"]:
            if "sha256" in row and base.digest(ROOT/f"{row['family']}_{row['variant']}.npz")!=row["sha256"]:
                raise RuntimeError("Saved trajectory changed")
    else:
        result=dict(status="running",frozen=fixed,rows=[])
        base.save(path,result)
    known={(r["family"],r["variant"]) for r in result["rows"]}
    pending=[job for job in jobs if job not in known]
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(march_one,protocol,*job) for job in pending]):
                row=future.result()
                result["rows"].append(row)
                base.save(path,result)
                print(row["family"],row["variant"],row.get("worst",row.get("failure")),flush=True)
    result["status"]="complete"
    base.save(path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("oracle","march"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=2:
        parser.error("Use one or two workers for this correctness diagnostic")
    protocol=base.read(PROTOCOL)
    ROOT.mkdir(exist_ok=True)
    jobs=[(family,variant) for family in ("rfm","legendre") for variant in
          ("baseline","ordinary","tangent")]
    path=ROOT/f"{args.phase}.json"
    if args.phase=="oracle":
        if path.exists():
            raise FileExistsError(path)
        oracle(protocol,jobs,path)
    else:
        if base.read(ROOT/"oracle.json")["status"]!="complete":
            raise RuntimeError("Complete the representation check first")
        march(protocol,jobs,path,args.workers)


if __name__=="__main__":
    main()
