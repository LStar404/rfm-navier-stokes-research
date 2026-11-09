"""Polynomial damping and inexact continuation for the existing cavity operator."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / 'rfm_v4'))
from cavity_solver import Homotopy, MappedLegendre, np, old, scaled_lstsq, unresolved


def polynomial_rate(r, linear, quadratic):
    """Minimize the quadratic residual's squared norm over [0, 1]."""
    derivative = np.array([
        r @ linear,
        linear @ linear + 2 * (r @ quadratic),
        3 * (linear @ quadratic),
        2 * (quadratic @ quadratic),
    ])
    scale = np.max(np.abs(derivative))
    if not np.isfinite(scale) or scale == 0:
        return 0.
    stationary = np.polynomial.polynomial.polyroots(derivative / scale)
    rates = [0., 1., *(float(x.real) for x in stationary
                         if abs(x.imag) <= 1e-10 and 0. < x.real < 1.)]
    return min(rates, key=lambda t: np.linalg.norm(r + t * (linear + t * quadratic)))


def polynomial_advance(op, c, maxiter, cond, reuse):
    """V4 Gauss–Newton, with polynomial damping tried before halving."""
    history = []
    factor = None
    budget = 0
    for _ in range(maxiter):
        fresh = budget == 0
        if fresh:
            r, jac = op.evaluate(c, 1., True)
            factor = old.PivotedStep(jac, cond) if reuse else None
            if factor is not None and factor.full_rank:
                delta, rank = factor.solve(-r), op.size
            else:
                delta, rank = scaled_lstsq(jac, -r, True, cond)
                factor = None
        else:
            r = op.evaluate(c, 1.)
            delta, rank = factor.solve(-r), op.size
            budget -= 1
        previous = np.linalg.norm(r)
        rate = 1.
        trial = op.evaluate(c + delta, 1.)
        residual = np.linalg.norm(trial)
        calls = 1
        used_polynomial = False
        if fresh and np.isfinite(residual) and residual > previous:
            linear = jac @ delta
            rate = polynomial_rate(r, linear, trial - r - linear)
            if 0. < rate < 1.:
                residual = np.linalg.norm(op.evaluate(c + rate * delta, 1.))
                calls += 1
                used_polynomial = bool(np.isfinite(residual) and residual <= previous)
            if not used_polynomial:
                rate, residual = 1., np.linalg.norm(trial)
        for _ in range(11 if fresh else 1):
            if np.isfinite(residual) and residual <= previous:
                break
            rate *= .5
            residual = np.linalg.norm(op.evaluate(c + rate * delta, 1.))
            calls += 1
        if not np.isfinite(residual) or residual > previous:
            budget = 0
            history.append(dict(fresh=fresh,residual=float(previous),step=0.,rank=rank,
                                trials=calls,polynomial=False))
            if fresh:
                break
            continue
        c = c + rate * delta
        history.append(dict(fresh=fresh,residual=float(residual),step=rate,rank=rank,
                            trials=calls,polynomial=used_polynomial))
        stalled = previous-residual <= max(1e-12,1e-7*previous)
        if stalled and fresh:
            break
        if stalled or residual > .7*previous or rate < 1.:
            budget = 0
        elif fresh and factor is not None:
            budget = reuse
    return c, history


def solve(problem, seed=17, *, m, scale, order, q, family='iid', coarse=0, reuse=2, maxiter=20,
          cond=1e-12, metric='h2_parity', wall=None, strategy='guard',
          alpha_ladder=(.25,.5,.75,1.), damping='polynomial', inner_budget=None,
          predict=False):
    """Change only the path used to solve the original v4 weak residual."""
    begun = time.perf_counter()
    if strategy not in ('guard','ladder') or damping not in ('polynomial','halving'):
        raise ValueError('Unknown strategy or damping')
    if (not alpha_ladder or alpha_ladder[-1] != 1.
            or any(b <= a for a,b in zip((0.,*alpha_ladder),alpha_ladder))):
        raise ValueError('Continuation must increase from zero and end at alpha=1')
    if inner_budget is not None and not 1 <= inner_budget <= maxiter:
        raise ValueError('Require 1 <= intermediate budget <= maxiter')
    step = polynomial_advance if damping == 'polynomial' else old.advance
    if family == 'legendre' and wall is not None:
        basis = MappedLegendre(problem,m,wall)
        perm = np.argsort(basis.ix+basis.iy,kind='stable')
        basis.ix,basis.iy = basis.ix[perm],basis.iy[perm]
    else:
        basis = old.trial_basis(problem,m,seed,family,scale,wall)
    op = old.TensorWeak(problem,basis,order,q,metric)
    setup = time.perf_counter()-begun
    small = op.prefix(coarse,min(order,int(np.ceil(np.sqrt(1.65*coarse))))) if coarse else op
    stages, triggers = [], []

    def stokes(operator):
        rhs = operator.evaluate(np.zeros(operator.size),0.)
        return scaled_lstsq(operator.stokes,-rhs,True,cond)[0]

    def advance(operator,c,alpha,label,limit,prediction=None):
        start = float(np.linalg.norm(operator.evaluate(c,alpha)))
        accepted_prediction = False
        if prediction is not None:
            proposed = np.linalg.norm(operator.evaluate(prediction,alpha))
            if proposed < start:
                c, start, accepted_prediction = prediction,float(proposed),True
        c,history = step(Homotopy(operator,alpha),c,limit,cond,reuse)
        stages.append(dict(label=label,alpha=alpha,size=operator.size,history=history,
                           initial_residual=start,final_residual=float(np.linalg.norm(operator.evaluate(c,alpha))),
                           unresolved_budget=unresolved(history,limit),prediction=accepted_prediction))
        return c,history

    def ladder(operator,initial,label):
        previous, c = None, initial
        before, now = None, 0.
        for alpha in alpha_ladder:
            candidate = c+(alpha-now)/(now-before)*(c-previous) if predict and previous is not None else None
            limit = inner_budget if inner_budget is not None and alpha < 1. else maxiter
            previous, (c, history) = c, advance(operator,c,alpha,label,limit,candidate)
            before, now = now,alpha
        return c,history

    initial = stokes(small)
    if coarse:
        if strategy == 'ladder':
            initial,history = ladder(small,initial,'coarse_ladder')
        else:
            direct,history = advance(small,initial,1.,'coarse_direct',maxiter)
            if unresolved(history,maxiter):
                triggers.append('coarse_direct_exhausted_without_stationarity')
                initial,history = ladder(small,initial,'coarse_restart_ladder')
            else:
                initial = direct
    c = np.zeros(m)
    c[:small.size] = initial
    c,history = advance(op,c,1.,'fine_direct',maxiter)
    if unresolved(history,maxiter):
        triggers.append('fine_direct_exhausted_without_stationarity')
        c,history = ladder(op,stokes(op),'fine_restart_ladder')
    final = float(np.linalg.norm(op.evaluate(c,1.)))
    return op,c,dict(seed=seed,m=m,strategy=strategy,damping=damping,inner_budget=inner_budget,
                     predict=predict,alpha_ladder=list(alpha_ladder),setup_seconds=setup,
                     seconds=time.perf_counter()-begun,final_residual=final,stages=stages,
                     fallback_triggers=triggers,final_budget_unresolved=unresolved(history,maxiter),
                     rank=history[-1]['rank'],
                     nonlinear_steps=sum(len(row['history']) for row in stages),
                     jacobians=sum(it['fresh'] for row in stages for it in row['history']))
