"""Exploratory residual-metric pilots: seed17/33 only, never formal timings.

Preserves the trial feature family and implements a full H_0^2 test Gram
metric instead of the diagonal, individually normalized weak-test loss.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.linalg import cholesky,solve_triangular,eigh
import rfm_solver as s
from tensor_rfm import tensor_quadrature
from experiments import cavity_truth, points
from certify_rfm import raw_fields

ROOT=Path(__file__).resolve().parent
OriginalWeak=s.HardWeak


def normalized_test_gram(problem,order,q):
    z,w=s.leggauss(q);z=(z+1)/2;w=w*problem.length/2
    b=s.polynomial_jet(s.Poly([0,0,1,-2,1]),z,problem.length)
    modes=[]
    for degree in range(order):
        p=s.polynomial_jet(s.Legendre.basis(degree,domain=[0,1]),z,problem.length)
        modes.append((b[0]*p[0],b[1]*p[0]+b[0]*p[1],
            b[2]*p[0]+2*b[1]*p[1]+b[0]*p[2]))
    v,f,d=np.asarray(modes).transpose(1,0,2)
    M=(v*w)@v.T;D=(d*w)@d.T;C=(d*w)@v.T
    # <Delta(v_i v_j), Delta(v_k v_l)>; retain transpose explicitly.
    G=np.kron(D,M)+np.kron(M,D)+np.kron(C,C.T)+np.kron(C.T,C)
    diagonal=np.sqrt(np.diag(G));G=G/diagonal[:,None]/diagonal[None,:]
    return (G+G.T)/2


class FullGramWeak(OriginalWeak):
    def __init__(self,problem,basis,cells,order=3,q=9):
        if cells!=1:raise ValueError('Pilot supports the single global domain only')
        super().__init__(problem,basis,cells,order,q)
        G=normalized_test_gram(problem,order,q)
        self.metric_cholesky=cholesky(G,lower=True,check_finite=False)
        self.gram_reconstruction_relative=float(np.linalg.norm(self.metric_cholesky@self.metric_cholesky.T-G)/np.linalg.norm(G))

    def evaluate(self,c,alpha,jac=False,picard=False):
        original=super().evaluate(c,alpha,jac,picard)
        if jac:
            r,J=original
            return tuple(solve_triangular(self.metric_cholesky,a,lower=True,check_finite=False) for a in (r,J))
        return solve_triangular(self.metric_cholesky,original,lower=True,check_finite=False)


def gram_identity_check():
    p=s.Problem('cavity',400);order,q=5,11
    _,(_,_,L)=s.weak_quadrature(p,1,order,q)
    z,w=s.leggauss(q);weight=np.outer(w/2,w/2).ravel()
    # L includes quadrature weights and individual test normalization.
    direct=(L/weight)@L.T
    factored=normalized_test_gram(p,order,q)
    err=float(np.linalg.norm(factored-direct)/np.linalg.norm(direct))
    if err>2e-12:raise AssertionError(f'Gram factorization error {err}')
    C=cholesky(factored,lower=True)
    rng=np.random.default_rng(221);r=rng.normal(size=order*order)
    z=solve_triangular(C,r,lower=True)
    normcheck=float(abs(z@z-r@np.linalg.solve(factored,r))/abs(z@z))
    return dict(gram_factorization_relative_error=err,riesz_norm_identity_relative_error=normcheck)


def weighted_score(problem,basis,c,n=113):
    z,w=s.leggauss(n);z=problem.lo+(z+1)*problem.length/2
    xy=np.stack(np.meshgrid(z,z,indexing='ij'),-1).reshape(-1,2)
    weight=np.outer(w,w).ravel()*problem.length**2/4
    reference,refname=cavity_truth(400,xy)
    pred=raw_fields(problem,basis,c,'weak',xy)
    vden=np.sum(weight*(reference['u']**2+reference['v']**2))
    vnum=np.sum(weight*((pred['u']-reference['u'])**2+(pred['v']-reference['v'])**2))
    wden=np.sum(weight*reference['omega']**2)
    wnum=np.sum(weight*(pred['omega']-reference['omega'])**2)
    return dict(grid=f'Gauss{n}x{n}',velocity_rel_l2=float(np.sqrt(vnum/vden)),
        vorticity_rel_l2=float(np.sqrt(wnum/wden)),reference=refname)


def run(seed,metric,order,m=496,scale=14,family='cos'):
    p=s.Problem('cavity',400);q=max(43,int(np.ceil(1.5*order)))
    config=dict(method='weak',m=m,n=25,seed=seed,activation=family,patches=1,
        scale=scale,maxiter=20,continuation=False,cells=1,q=q,order=order)
    tick=time.perf_counter()
    klass=FullGramWeak if metric=='full_h2' else OriginalWeak
    with patch.object(s,'HardWeak',klass),patch.object(s,'weak_quadrature',tensor_quadrature):
        op,c,row=s.solve(p,**config)
    row.update(metric=metric,pilot_only=True,prototype_seconds=time.perf_counter()-tick)
    xy=points(p,81);reference,refname=cavity_truth(400,xy)
    pred=raw_fields(p,op.basis,c,'weak',xy)
    row.update(s.field_errors(pred,reference));row['reference']=refname
    row['gram_reconstruction_relative']=getattr(op,'gram_reconstruction_relative',None)
    row['coefficient_norm']=float(np.linalg.norm(c))
    row['gauss113']=weighted_score(p,op.basis,c)
    row['comment']='Exploratory seeds only. Require fresh-seed validation and independent-reference floor before acceptance.'
    tag='' if family=='cos' else '_'+family
    np.savez_compressed(ROOT/f'operator_opt_state_s{seed}_{metric}_p{order}_m{m}{tag}.npz',coefficients=c,xy=xy,**pred)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(17,33),default=17)
    ap.add_argument('--metric',choices=('diagonal','full_h2'),default='full_h2')
    ap.add_argument('--order',type=int,default=29);ap.add_argument('--m',type=int,default=496)
    ap.add_argument('--scale',type=float,default=14)
    ap.add_argument('--family',choices=('cos','legendre'),default='cos')
    args=ap.parse_args();checks=gram_identity_check()
    row=run(args.seed,args.metric,args.order,args.m,args.scale,args.family)
    result=dict(checks=checks,row=row)
    tag='' if args.family=='cos' else '_'+args.family
    name=f'operator_opt_s{args.seed}_{args.metric}_p{args.order}_m{args.m}{tag}.json'
    (ROOT/name).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
