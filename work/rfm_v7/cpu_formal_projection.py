"""Oracle radial/angular fit to reference checkpoints, without time evolution."""
import argparse
from pathlib import Path

from cpu_formal import OUT, PROTOCOL, case, config, grid_set, open_reference, read, save
from scipy.linalg import svd

np = case.np


def power(x):
    return np.sum(abs(x)**2,axis=0)


def fit(matrix, target):
    """Fit all real/imaginary snapshot components against one real radial matrix."""
    scale=np.linalg.norm(matrix,axis=0)
    if (scale==0).any():
        raise ArithmeticError("A retained radial column is identically zero on the score grid")
    system=matrix/scale
    rhs=np.concatenate((target.real,target.imag),axis=1)
    vectors,sv,_=svd(system,full_matrices=False,lapack_driver="gesdd",check_finite=True)
    cutoff=max(system.shape)*np.finfo(system.dtype).eps
    rank=int(np.count_nonzero(sv>sv[0]*cutoff))
    orthogonal=vectors[:,:rank]
    residual=rhs-orthogonal@(orthogonal.T@rhs)
    first,second=np.hsplit(residual,2)
    condition=float(sv[0]/sv[rank-1]) if rank else None
    return power(first)+power(second),int(rank),condition,cutoff


def projection(model, s, weights, fields, times):
    samples,lift,radius,mean=model.precompute(s)
    active=weights>0
    f,d,d2=samples[:,active]
    lf,ld,ld2=lift[:,active]
    r=radius[active]
    w=np.sqrt(weights[active])
    mv,md=mean[:,active]
    # Each array has shape [active radial nodes, positive modes, checkpoint times].
    ur,ut,omega=[a[active]*w[:,None,None] for a in fields]
    velocity_power=power(ur)+power(ut)
    vorticity_power=power(omega)
    velocity_den=velocity_power[0]+2*velocity_power[1:].sum(axis=0)
    vorticity_den=vorticity_power[0]+2*vorticity_power[1:].sum(axis=0)
    v_res,v_rank,v_cond,v_cut=fit(mv*w[:,None],ut[:,0])
    o_res,o_rank,o_cond,o_cut=fit((mv+md)*(w/r)[:,None],omega[:,0])
    ranks=[(v_rank,o_rank)]
    conditions=[v_cond,o_cond]
    cutoffs=[(v_cut,o_cut)]
    v_res+=power(ur[:,0])
    amplitude=case.speed(times)[None,:]
    for k in range(1,model.k+1):
        u_r,u_t,vort=ur[:,k],ut[:,k],omega[:,k]
        if k==1:
            u_r=u_r-1j*(lf*w/r)[:,None]*amplitude
            u_t=u_t+(ld*w/r)[:,None]*amplitude
            vort=vort-((lf-ld2)*w/r**2)[:,None]*amplitude
        # Divide the radial component and its operator by i, preserving norm.
        v_matrix=np.concatenate((k*f*(w/r)[:,None],-d*(w/r)[:,None]))
        v_target=np.concatenate((-1j*u_r,u_t))
        o_matrix=(k*k*f-d2)*(w/r**2)[:,None]
        ve,vr,vc,v_cut=fit(v_matrix,v_target)
        oe,orr,oc,o_cut=fit(o_matrix,vort)
        v_res+=2*ve
        o_res+=2*oe
        ranks.append((vr,orr))
        conditions.extend((vc,oc))
        cutoffs.append((v_cut,o_cut))
    v_res+=2*velocity_power[model.k+1:].sum(axis=0)
    o_res+=2*vorticity_power[model.k+1:].sum(axis=0)
    return dict(velocity=np.sqrt(v_res/velocity_den).tolist(),
                vorticity=np.sqrt(o_res/vorticity_den).tolist(),
                max_scaled_condition=max(v for v in conditions if v is not None),
                ranks_by_mode_velocity_then_vorticity=ranks,
                relative_svd_cutoffs_by_mode_velocity_then_vorticity=cutoffs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs",nargs="+",default=["rfm37_k72","leg37_k72","rfm41_k80","leg39_k80"])
    parser.add_argument("--grids",nargs="+",choices=("gauss123","mid129","layer144"),
                        default=["gauss123","mid129","layer144"])
    parser.add_argument("--output",type=Path,default=OUT/"oracle_projection_svd.json")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol=read(PROTOCOL)
    variants={s["name"]:s for s in protocol["variants"] if s["method"]!="fd4"}
    reference=open_reference(protocol["reference_path"][-1],protocol)
    grids={n:g for n,g in grid_set().items() if n in args.grids}
    qualified=read(OUT/"qualification.json")
    target={n:tuple(np.stack(x,axis=-1) for x in zip(
        *(reference.modal(i,s) for i in range(len(reference.times)))))
            for n,(s,_,_) in grids.items()}
    result=dict(checkpoints=reference.times.tolist(),rows=[],
        note="Independent affine projection per velocity/curl norm, region, mode and time. "
             "The relative singular cutoff is max(rows,columns) times machine epsilon. "
             "Rank truncation changes the fitted numerical span, so these sampled-reference "
             "residuals are diagnostic estimates rather than guaranteed lower bounds for the original span.")
    for name in args.configs:
        spec=variants[name]
        cls,settings=config(spec,protocol,protocol["seeds"][0])
        model=cls(dt=1/spec["rate"],**settings)
        rows=[]
        for grid,(s,theta,weights) in grids.items():
            if len(theta)<=2*protocol["reference_path"][-1]["k"]:
                raise RuntimeError("Angular quadrature does not resolve all reference cross-terms")
            for region,w in (("full",weights),("near",weights*(case.radius(s)<=2.5))):
                rows.append(dict(grid=grid,region=region,**projection(
                    model,s,w,target[grid],reference.times)))
        worst={f"{region}_{label}":max(max(r[label]) for r in rows if r["region"]==region)
               for region in ("full","near") for label in ("velocity","vorticity")}
        witness=next(r for r in qualified["rows"] if r["name"]==name and (
            spec["method"]!="rfm" or r["seed"]==settings["seed"]))
        by_time={(r["grid"],r["time"]):r for r in witness["scores"]}
        excess=max(value-by_time[(r["grid"],t)][label if r["region"]=="full" else f"near_{label}"]
                   for r in rows for label in ("velocity","vorticity")
                   for value,t in zip(r[label],reference.times))
        result["rows"].append(dict(name=name,seed=settings["seed"],worst=worst,fits=rows,
            max_projection_minus_observed_error=float(excess),
            projection_consistent_with_witness=bool(excess<=5e-8)))
        save(args.output,result)
        print(name,worst,flush=True)


if __name__=="__main__":
    main()
