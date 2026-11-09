"""RFM experiments with shared analytic and CFD scoring grids."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from rfm_solver import Problem, solve, field_errors, np, interior_boundary, HardWeak

ROOT=Path(__file__).resolve().parent


def points(problem,n):
    return interior_boundary(problem,n)[0]


def boundary(problem,n=256):
    lo,L=problem.lo,problem.length
    z=np.linspace(lo,lo+L,n+1)
    return np.concatenate([np.column_stack((np.full_like(z,k),z)) for k in (lo,lo+L)]
        +[np.column_stack((z,np.full_like(z,k))) for k in (lo,lo+L)])


def cavity_truth(reynolds,xy):
    from cfd_reference import interpolate_solution
    candidates=list(ROOT.glob(f'cfd_reference4_cavity_Re{int(reynolds)}_n*.npz'))
    if not candidates:
        candidates=list(ROOT.glob(f'cfd_reference_cavity_Re{int(reynolds)}_n*.npz'))
    files=sorted(candidates,key=lambda p:int(p.stem.rsplit('n',1)[1]))
    if not files:
        return None,None
    path=files[-1]
    with np.load(path) as d:
        fields={k:d[k] for k in ('x','y','psi','omega','u','v')}
    return interpolate_solution(fields,xy[:,0],xy[:,1]),path.name


def evaluate(config,seed,save=None):
    p=Problem(config['case'],config['reynolds'])
    kwargs={k:v for k,v in config.items() if k not in ('case','reynolds','label')}
    op,c,row=solve(p,seed=seed,**kwargs)
    row['label']=config.get('label',row['method'])
    # Score away from the training nodes, using the curl of the actual velocity.
    xy=points(p,81)
    batched={}
    for b in np.array_split(xy,8):
        for name,a in op.fields(c,b).items():
            batched.setdefault(name,[]).append(a)
    pred={k:np.concatenate(v) for k,v in batched.items()}
    wall=boundary(p)
    wall_pred=op.fields(c,wall)
    bu,bv=p.boundary_velocity(wall)
    row['wall_velocity_max']=float(np.hypot(wall_pred['u']-bu,wall_pred['v']-bv).max())
    row['divergence_rms']=float(np.linalg.norm(pred['div'])/np.sqrt(len(xy)))
    if 'psi' in pred:
        row['psi_min']=float(pred['psi'].min())
    if 'omega_independent' in pred:
        row['vorticity_compatibility_rel_l2']=float(np.linalg.norm(pred['omega_independent']-pred['omega'])/np.linalg.norm(pred['omega']))
    if p.case=='kovasznay':
        ref=p.exact(xy);refname='analytical'
    else:
        ref,refname=cavity_truth(p.reynolds,xy)
        row['vorticity_integral_midpoint']=float(pred['omega'].mean())
        top=wall[:,1]==1.
        row['top_wall_vorticity_peak']=float(abs(wall_pred['omega'][top]).max())
    row['reference']=refname
    if ref is not None:
        row.update(field_errors(pred,ref))
    if save is not None:
        original_coefficients=op.transform@c if hasattr(op,'transform') else c
        np.savez_compressed(ROOT/(save+'.npz'),xy=xy,coefficients=original_coefficients,**pred,
            **({f'ref_{k}':v for k,v in ref.items()} if ref is not None else {}))
    print(p.case,p.reynolds,row['label'],row['m'],row['activation'],row['patches'],seed,
        round(row['seconds'],3),row.get('velocity_rel_l2'),row.get('vorticity_rel_l2'),
        'residual',row['final_residual'],'psi_min',row.get('psi_min'),flush=True)
    return row


def setting(case,reynolds,method,**kwargs):
    return dict(case=case,reynolds=reynolds,method=method,n=25,maxiter=9,**kwargs)


def pilot(case):
    for re in ([40] if case=='kovasznay' else [100,400]):
        for m in (96,160):
            for activation,patches,scale in [('tanh',1,5.),('tanh',2,3.),('cos',1,6.),('cos',2,3.)]:
                for method in ('picard','mixed','weak'):
                    yield setting(case,re,method,m=m,activation=activation,patches=patches,scale=scale,
                                  continuation=re>=100)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('pilot','formal'),required=True)
    ap.add_argument('--case',choices=('kovasznay','cavity'),default='kovasznay')
    args=ap.parse_args()
    rows=[]
    start=time.perf_counter()
    output=ROOT/f'{args.stage}_{args.case}.json'
    source=hashlib.sha256((ROOT/'rfm_solver.py').read_bytes()).hexdigest()
    if args.stage=='pilot':
        configs=list(pilot(args.case))
        seeds=[17]
    else:
        configs=json.loads((ROOT/f'frozen_{args.case}.json').read_text(encoding='utf8'))
        seeds=[101,211,307]
    for i,seed in enumerate(seeds):
        for j in np.random.default_rng(117+i).permutation(len(configs)):
            c=configs[j]
            stem=f"field_{args.case}_{c['reynolds']}_{c.get('label',c['method'])}" if i==0 and args.stage=='formal' else None
            rows.append(evaluate(c,seed,stem))
            result=dict(configs=configs,stage=args.stage,rows=rows,source_sha256=source,
                total_seconds=time.perf_counter()-start,
                timing='One CPU thread. Includes feature generation, matrix assembly, and all nonlinear iterations. Independent field scoring excluded.')
            output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    if args.stage=='formal':
        for c in configs:
            group=[r for r in rows if r['label']==c['label'] and r['reynolds']==c['reynolds']]
            med=statistics.median(r['seconds'] for r in group)
            print(c['label'],med,[(r.get('velocity_rel_l2'),r.get('vorticity_rel_l2')) for r in group],flush=True)


if __name__=='__main__':
    main()
