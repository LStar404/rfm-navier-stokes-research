"""Re1000 continuation of the prior RFM study and FD4 comparison."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'rfm_ns'))
import rfm_solver as base
from rfm_solver import np
from rfm_v2_benchmark import score_grids, weighted_errors
from rfm_v2_solver import solve as solve_rfm
from rfm_v2_warp_legendre import solve as solve_mapped_legendre
from cfd_reference import interpolate_solution, solve_cavity
from cfd_reference_fourth import CavityFourth
from cavity_reference_v3 import load

PROBLEM=base.Problem('cavity',1000)


class Score:
    def __init__(self):
        self.grids=score_grids(PROBLEM,113)
        fine=load(ROOT/'cavity1000_ref384.npz')
        previous=load(ROOT/'cavity1000_ref256.npz')
        self.truth={name:interpolate_solution(fine,xy[:,0],xy[:,1]) for name,(xy,_) in self.grids.items()}
        self.grid_change=self.evaluate(lambda xy:interpolate_solution(previous,xy[:,0],xy[:,1]),batch=False)

    def evaluate(self,fields,batch=True):
        rows={}
        for name,(xy,w) in self.grids.items():
            blocks=[fields(b) for b in np.array_split(xy,int(np.ceil(len(xy)/256)) if batch else 1)]
            result={key:np.concatenate([d[key] for d in blocks]) for key in ('u','v','omega')}
            rows[name]=weighted_errors(result,self.truth[name],w)
        return rows


def fd_two_level(n,coarse):
    rows=[];failed=None
    try:
        x,rec=solve_cavity(coarse,1000,_model_class=CavityFourth,maxiter=30)
        rows.append(rec)
    except (RuntimeError,FloatingPointError) as e:
        failed=str(e)
        x=None
        for re in (50,150,400,700,1000):
            x,rec=solve_cavity(coarse,re,initial=x,_model_class=CavityFourth,maxiter=30)
            rows.append(rec)
    if n!=coarse:
        x,rec=solve_cavity(n,1000,initial=x,_model_class=CavityFourth,maxiter=30)
        rows.append(rec)
    return x,dict(n=n,coarse=coarse,stages=rows,failed_direct=failed)


def errors(values):
    return {key:max(v[key] for v in values.values()) for key in ('velocity_rel_l2','vorticity_rel_l2')}


def setting(m,scale,p,q,warp=None,coarse=None,family='iid',reuse=2):
    return dict(m=m,scale=scale,order=p,q=q,family=family,coarse=coarse or int(m*.42),reuse=reuse,
                metric='h2_parity',**({'wall':dict(warp='sinh',strength=list(warp))} if warp else {}))


def pilot():
    import time
    score=Score()
    options=[('rfm',setting(m,s,p,q,warp)) for m,s,p,q in [(400,12,27,43),(496,14,29,47),
            (595,15,32,51),(666,16,34,54)] for warp in [(1.3,1.3),(1.9,1.9),(1.7,2.5)]]
    options.extend(('mapped_legendre',setting(m,1,p,q,w,c,'legendre')) for m,p,q,w,c in [
        (406,29,45,(1.3,1.3),171),(496,32,47,(1.3,1.3),210),
        (595,34,51,(1.9,1.9),253),(666,35,54,(1.7,2.5),276)])
    rows=[]
    for method,options in options:
        tick=time.perf_counter()
        try:
            fn=solve_rfm if method=='rfm' else solve_mapped_legendre
            op,c,meta=fn(PROBLEM,seed=17,**options)
            elapsed=time.perf_counter()-tick
            values=score.evaluate(lambda xy:op.fields(c,xy))
            row=dict(method=method,config=options,seconds=elapsed,errors=errors(values),scores=values,meta=meta)
            print(method,options['m'],options.get('wall'),round(elapsed,3),row['errors'],flush=True)
        except (ValueError,RuntimeError,ArithmeticError) as e:
            row=dict(method=method,config=options,error=str(e)); print(row,flush=True)
        rows.append(row)
        (ROOT/'cavity_pilot.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    pilot()
