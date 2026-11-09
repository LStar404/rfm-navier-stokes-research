"""Continuation-aware initialization of the immutable v2 final cavity operator.

Only the path to the final coefficients changes. All final residuals and fields
use TensorWeak at alpha=1. Failed attempts and restarts remain in the solve timer.
"""
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'rfm_ns'))
import rfm_v2_solver as old
from rfm_solver import np,scaled_lstsq
from rfm_v2_warp_legendre import MappedLegendre


class Homotopy:
    def __init__(self,op,alpha):self.op,self.alpha,self.size=op,alpha,op.size
    def evaluate(self,c,alpha,jac=False,picard=False):
        return self.op.evaluate(c,self.alpha*alpha,jac,picard)


def unresolved(history,budget):
    """A training-only flag; no independent reference or field-error threshold."""
    if len(history)<budget:return False
    if len(history)<2:return True
    previous,last=history[-2]['residual'],history[-1]['residual']
    return previous-last>max(1e-12,1e-7*previous)


def solve(problem,seed=17,*,m,scale,order,q,family='iid',coarse=0,reuse=2,maxiter=20,cond=1e-12,
          metric='h2_parity',wall=None,strategy='guard',alpha_ladder=(.25,.5,.75,1.)):
    begun=time.perf_counter()
    if strategy not in ('original','guard','ladder'):raise ValueError(strategy)
    if tuple(alpha_ladder)[-1]!=1. or any(not 0<a<=1 for a in alpha_ladder):
        raise ValueError('Continuation must end at the unchanged alpha=1 operator')
    if family=='legendre' and wall is not None:
        basis=MappedLegendre(problem,m,wall)
        perm=np.argsort(basis.ix+basis.iy,kind='stable')
        basis.ix,basis.iy=basis.ix[perm],basis.iy[perm]
    else:basis=old.trial_basis(problem,m,seed,family,scale,wall)
    op=old.TensorWeak(problem,basis,order,q,metric)
    setup=time.perf_counter()-begun
    small=op.prefix(coarse,min(order,int(np.ceil(np.sqrt(1.65*coarse))))) if coarse else op
    rhs=small.evaluate(np.zeros(small.size),0.)
    stokes,rank=scaled_lstsq(small.stokes,-rhs,True,cond)
    stages=[];triggers=[]
    def advance(operator,c,alpha,label):
        start=float(np.linalg.norm(operator.evaluate(c,alpha)))
        c,history=old.advance(Homotopy(operator,alpha),c,maxiter,cond,reuse)
        stages.append(dict(label=label,alpha=alpha,size=operator.size,history=history,
            initial_residual=start,final_residual=float(np.linalg.norm(operator.evaluate(c,alpha))),
            unresolved_budget=unresolved(history,maxiter)))
        return c,history
    def ladder(operator,c,label):
        history=[]
        for alpha in alpha_ladder:c,history=advance(operator,c,alpha,label)
        return c,history
    initial=stokes.copy();history=[]
    if coarse:
        if strategy=='ladder':initial,history=ladder(small,initial,'coarse_ladder')
        else:
            initial,history=advance(small,initial,1.,'coarse_direct')
            if strategy=='guard' and unresolved(history,maxiter):
                triggers.append('coarse_direct_exhausted_without_stationarity')
                initial,history=ladder(small,stokes.copy(),'coarse_restart_ladder')
    c=np.zeros(m);c[:small.size]=initial
    c,history=advance(op,c,1.,'fine_direct')
    if strategy!='original' and unresolved(history,maxiter):
        triggers.append('fine_direct_exhausted_without_stationarity')
        # Full-space Stokes restart is a declared last resort, never a reference
        # field warm start. Its assembly/linear solves are fully charged.
        rhs=op.evaluate(np.zeros(op.size),0.)
        initial,rank=scaled_lstsq(op.stokes,-rhs,True,cond)
        c,history=ladder(op,initial,'fine_restart_ladder')
    final=float(np.linalg.norm(op.evaluate(c,1.)))
    return op,c,dict(case=problem.case,reynolds=problem.reynolds,seed=seed,m=m,scale=scale,order=order,q=q,
        family=family,coarse=coarse,reuse=reuse,maxiter=maxiter,cond=cond,metric=metric,wall=wall,
        strategy=strategy,alpha_ladder=list(alpha_ladder),setup_seconds=setup,seconds=time.perf_counter()-begun,
        final_residual=final,stages=stages,fallback_triggers=triggers,
        final_budget_unresolved=unresolved(history,maxiter),rank=history[-1]['rank'] if history else rank,
        nonlinear_steps=sum(len(row['history']) for row in stages))

