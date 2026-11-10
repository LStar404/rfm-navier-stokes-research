"""Global weak RFM with stratified frequency samples and reusable Jacobians."""
import time

from . import rfm_solver as base
from .rfm_solver import np
from scipy.linalg import cholesky, get_lapack_funcs, qr, solve_triangular
from scipy.stats import qmc
from .tensor_rfm import tensor_quadrature, TensorProjection


def trial_basis(problem, m, seed, family, scale, wall=None):
    if wall is not None:
        if family!='iid':
            raise ValueError('Wall maps currently use IID cosine samples')
        from .wall_trial import WallFeatures
        return WallFeatures(problem,m,seed,scale=scale,**wall)
    if family == 'legendre':
        basis = base.LegendreFeatures(problem,m)
        permutation = np.argsort(basis.ix+basis.iy,kind='stable')
        basis.ix,basis.iy = basis.ix[permutation],basis.iy[permutation]
        return basis
    sx,sy = np.broadcast_to(scale,(2,))
    basis = base.Features(problem,m,seed,'cos',1,1.)
    frequencies,phase,center = basis.blocks[0]
    if family == 'iid':
        basis.blocks[0] = (frequencies*[sx,sy],phase,center)
        return basis
    if family == 'sobol':
        n = m-1
        samples = qmc.Sobol(3,scramble=True,seed=seed).random_base2((n-1).bit_length())[:n]
        frequencies = (2*samples[:,:2]-1)*[3*sx,3*sy]
        phase = (2*samples[:,2]-1)*np.pi
    elif family == 'pairs':
        if m%2==0:
            raise ValueError('Paired features require an odd total count')
        n = (m-1)//2
        samples = qmc.Sobol(2,scramble=True,seed=seed).random_base2((n-1).bit_length())[:n]
        frequencies = np.repeat(np.column_stack((3*sx*samples[:,0],3*sy*(2*samples[:,1]-1))),2,axis=0)
        phase = np.tile([0.,-np.pi/2],n)
    else:
        raise ValueError(f'Unknown feature family {family!r}')
    basis.blocks[0] = (frequencies,phase,center)
    return basis


class GramMetric:
    """Biharmonic Riesz map, split by the two reflection parities."""
    def __init__(self,problem,order,q,kind):
        from .test_metric import normalized_test_gram
        gram = normalized_test_gram(problem,order,q)
        if kind=='h2_dense':
            self.blocks = [(slice(None),cholesky(gram,lower=True,check_finite=False))]
            return
        i,j = np.indices((order,order))
        parity = (2*(i%2)+j%2).ravel()
        self.blocks = []
        for label in np.unique(parity):
            idx = np.flatnonzero(parity==label)
            self.blocks.append((idx,cholesky(gram[np.ix_(idx,idx)],lower=True,check_finite=False)))

    def apply(self,array):
        result = np.empty_like(array)
        for idx,factor in self.blocks:
            result[idx] = solve_triangular(factor,array[idx],lower=True,check_finite=False)
        return result


class TensorWeak(base.HardWeak):
    def __init__(self,problem,basis,order,q,metric='diagonal'):
        self.problem,self.basis = problem,basis
        self.size = basis.count
        xy,(self.tx,self.ty,lap) = tensor_quadrature(problem,1,order,q)
        self.lift,env = base.lift_and_envelope(problem,xy)
        self.psi = base.multiply(basis.jet(xy),env)
        self.U,self.V,self.W = self.psi[2],-self.psi[1],-self.psi[3]
        self.nu_lap = problem.nu*lap
        self.linear = self.nu_lap@self.W
        self.configure_metric(metric,order,q)

    def configure_metric(self,kind,order,q):
        if kind not in ('diagonal','h2_dense','h2_parity'):
            raise ValueError(f'Unknown test metric {kind!r}')
        self.metric_kind,self.q = kind,q
        self.metric = None if kind=='diagonal' else GramMetric(self.problem,order,q,kind)
        self.stokes = self.linear if self.metric is None else self.metric.apply(self.linear)

    def evaluate(self,c,alpha,jac=False,picard=False):
        raw = super().evaluate(c,alpha,jac,picard)
        if self.metric is None:
            return raw
        if jac:
            r,j = raw
            return self.metric.apply(r),self.metric.apply(j)
        return self.metric.apply(raw)

    def prefix(self,count,order):
        """Reuse stored basis jets and quadrature in a smaller nonlinear start."""
        if not 1 <= count < self.size or order*order < count:
            raise ValueError('Coarse test/trial sizes must define an overdetermined prefix')
        coarse = object.__new__(TensorWeak)
        coarse.problem,coarse.basis,coarse.size = self.problem,self.basis,count
        coarse.lift,coarse.psi = self.lift,np.ascontiguousarray(self.psi[:,:,:count])
        coarse.U,coarse.V,coarse.W = coarse.psi[2],-coarse.psi[1],-coarse.psi[3]
        def restrict(op):
            return TensorProjection(tuple((a[:order],b[:order]) for a,b in op.factors),op.inverse_norm[:order,:order])
        coarse.tx,coarse.ty,coarse.nu_lap = (restrict(a) for a in (self.tx,self.ty,self.nu_lap))
        coarse.linear = coarse.nu_lap@coarse.W
        coarse.configure_metric(self.metric_kind,order,self.q)
        return coarse


class PivotedStep:
    """Store Householder reflectors to apply the same least-squares inverse again."""
    def __init__(self,jac,cond):
        self.scales = np.linalg.norm(jac,axis=0)
        self.scales = np.maximum(self.scales,max(self.scales.max()*1e-15,1e-30))
        (self.householder,self.tau),self.r,self.piv = qr(jac/self.scales,mode='raw',
            pivoting=True,check_finite=False)
        diag = abs(self.r.diagonal())
        self.full_rank = len(diag)==jac.shape[1] and diag.min()>diag.max()*cond*10
        self.ormqr = get_lapack_funcs('ormqr',(self.householder,))
        vector = np.empty((jac.shape[0],1))
        _,workspace,info = self.ormqr('L','T',self.householder,self.tau,vector,lwork=-1)
        if info:
            raise RuntimeError(f'ORMQR workspace query failed with info={info}')
        self.lwork = int(workspace[0])

    def solve(self,rhs):
        rotated,_,info = self.ormqr('L','T',self.householder,self.tau,rhs[:,None],lwork=self.lwork)
        if info:
            raise RuntimeError(f'ORMQR failed with info={info}')
        scaled = np.empty(len(self.piv))
        scaled[self.piv] = solve_triangular(self.r,rotated[:len(self.piv),0],check_finite=False)
        return scaled/self.scales


def advance(op,c,maxiter,cond,reuse):
    """Accept stale-Jacobian steps only on actual residual decrease."""
    history = []
    factor = None
    budget = 0
    for _ in range(maxiter):
        if budget:
            r = op.evaluate(c,1.)
            delta = factor.solve(-r)
            budget -= 1
            fresh = False
            rank = op.size
        else:
            r,j = op.evaluate(c,1.,True)
            fresh = True
            factor = PivotedStep(j,cond) if reuse else None
            if factor is not None and factor.full_rank:
                delta,rank = factor.solve(-r),op.size
            else:
                delta,rank = base.scaled_lstsq(j,-r,True,cond)
                factor = None
        old = np.linalg.norm(r)
        rate = 1.
        for _ in range(12 if fresh else 2):
            candidate = c+rate*delta
            residual = np.linalg.norm(op.evaluate(candidate,1.))
            if np.isfinite(residual) and residual<=old:
                break
            rate *= .5
        else:
            budget = 0
            history.append(dict(fresh=fresh,residual=float(old),step=0.,rank=rank))
            if fresh:
                break
            continue
        c = candidate
        history.append(dict(fresh=fresh,residual=float(residual),step=rate,rank=rank))
        stalled = old-residual<=max(1e-12,1e-7*old)
        if stalled and fresh:
            break
        if stalled or residual>.7*old or rate<1.:
            budget = 0
        elif fresh and factor is not None:
            budget = reuse
    return c,history


def solve(problem,seed=17,*,m,scale,order,q,family='iid',coarse=0,reuse=0,maxiter=20,cond=1e-12,
          metric='diagonal',wall=None,basis_factory=None):
    """Solve with a call-local factory; omission preserves the archived default.

    The optional factory has signature (problem, m, seed, family, scale, wall).
    It must return a basis with count=m and analytic second-order jets.  It is
    called once per solve.  No module-level factory is replaced.
    """
    begun = time.perf_counter()
    factory = trial_basis if basis_factory is None else basis_factory
    basis = factory(problem,m,seed,family,scale,wall)
    op = TensorWeak(problem,basis,order,q,metric)
    setup = time.perf_counter()-begun
    c = np.zeros(m)
    coarse_op = op.prefix(coarse,min(order,int(np.ceil(np.sqrt(1.65*coarse))))) if coarse else op
    r = coarse_op.evaluate(np.zeros(coarse_op.size),0.)
    initial,rank = base.scaled_lstsq(coarse_op.stokes,-r,True,cond)
    coarse_history = []
    if coarse:
        initial,coarse_history = advance(coarse_op,initial,maxiter,cond,reuse)
    c[:coarse_op.size] = initial
    c,history = advance(op,c,maxiter,cond,reuse)
    final = float(np.linalg.norm(op.evaluate(c,1.)))
    return op,c,dict(case=problem.case,reynolds=problem.reynolds,seed=seed,m=m,order=order,q=q,
        family=family,scale=scale,coarse=coarse,reuse=reuse,metric=metric,wall=wall,
        setup_seconds=setup,seconds=time.perf_counter()-begun,
        final_residual=final,history=history,coarse_history=coarse_history,rank=history[-1]['rank'] if history else rank)
