"""Clamped streamfunction RFM on the cylinder, with a matched Legendre trial."""
from dataclasses import dataclass

import cylinder_case as case
from cylinder_case import np,leggauss
from numpy.polynomial import Legendre
from scipy.linalg import cholesky,lstsq,solve_triangular


def mapped_coordinate(z,warp,mapping='inner'):
    if not warp:
        return z,np.ones_like(z),np.zeros_like(z)
    if mapping=='both':
        x=warp*(2*z-1)
        return .5+np.sinh(x)/(2*np.sinh(warp)),warp*np.cosh(x)/np.sinh(warp),2*warp**2*np.sinh(x)/np.sinh(warp)
    if mapping!='inner':
        raise ValueError(mapping)
    scale=-np.expm1(-warp)
    derivative=warp*np.exp(-warp*z)/scale
    return -np.expm1(-warp*z)/scale,derivative,-warp*derivative


def clamped(jet,z,outer,outer_jet=None):
    a,b,c=jet
    z=z[:,None]
    if outer=='dirichlet':
        v=16*z*z*(1-z)**2
        d=32*z*(1-z)*(1-2*z)
        dd=32*(1-6*z+6*z*z)
    elif outer=='farfield':
        v=16*z*z*(1-z)
        d=16*z*(2-3*z)
        dd=32*(1-3*z)
    else:
        raise ValueError(outer)
    values=np.stack((v*a,d*a+v*b,dd*a+2*d*b+v*c))
    if outer=='farfield' and outer_jet is not None:
        edge_value,edge_derivative=outer_jet
        outer_second=-64*edge_value-32*edge_derivative
        correction=np.stack((.5*z*z*(1-z)**2,z*(1-z)*(1-2*z),1-6*z+6*z*z))
        values-=correction*outer_second
    return values/np.array([1.,case.LENGTH,case.LENGTH**2])[:,None,None]


def polynomial(s,count,warp,mapping='inner',outer='farfield',trial=False):
    z=s/case.LENGTH
    x,d,dd=mapped_coordinate(z,warp,mapping)
    polys=[Legendre.basis(j,domain=[0,1]) for j in range(count)]
    value,first,second=(np.column_stack([p.deriv(j)(x) for p in polys]) for j in range(3))
    edge=None
    if trial and outer=='farfield':
        d1=mapped_coordinate(np.array([1.]),warp,mapping)[1][0]
        edge=(1.,d1*np.asarray([p.deriv()(1.) for p in polys]))
    return clamped((value,first*d[:,None],second*d[:,None]**2+first*dd[:,None]),z,outer,edge)


@dataclass
class Trial:
    m: int
    family: str
    seed: int
    scale: float
    warp: float
    mapping: str
    outer: str
    poly_prefix: int = 0
    sampling: str = 'iid'

    def __post_init__(self):
        if self.family not in ('rfm','legendre','hybrid'):
            raise ValueError(self.family)
        if self.family=='hybrid' and not 0<self.poly_prefix<self.m:
            raise ValueError('Hybrid requires a proper Legendre prefix')
        rng=np.random.default_rng(self.seed)
        if self.family=='hybrid':
            self.frequency=rng.uniform(-self.scale,self.scale,self.m-self.poly_prefix)
            self.phase=rng.uniform(-np.pi,np.pi,self.m-self.poly_prefix)
        elif self.sampling=='paired':
            if self.m%2!=1:
                raise ValueError('Paired sampling needs an odd trial size')
            count=(self.m-1)//2
            bins=(np.arange(count)+rng.uniform(.2,.8,count))*self.scale/count
            self.frequency=np.r_[0.,np.repeat(bins,2)]
            self.phase=np.r_[0.,np.tile([0.,-.5*np.pi],count)]
        elif self.sampling!='iid':
            raise ValueError(self.sampling)
        else:
            self.frequency=np.r_[0.,rng.uniform(-self.scale,self.scale,self.m-1)]
            self.phase=np.r_[0.,rng.uniform(-np.pi,np.pi,self.m-1)]

    def jets(self,s):
        if self.family=='legendre':
            return polynomial(s,self.m,self.warp,self.mapping,self.outer,trial=True)
        z=s/case.LENGTH
        x,d,dd=mapped_coordinate(z,self.warp,self.mapping)
        arg=(x[:,None]-.5)*self.frequency+self.phase
        dx=d[:,None]*self.frequency
        d2x=dd[:,None]*self.frequency
        v,b=np.cos(arg),-np.sin(arg)
        edge=None
        if self.outer=='farfield':
            d1=mapped_coordinate(np.array([1.]),self.warp,self.mapping)[1][0]
            phase=.5*self.frequency+self.phase
            edge=(np.cos(phase),-np.sin(phase)*d1*self.frequency)
        random=clamped((v,b*dx,-v*dx**2+b*d2x),z,self.outer,edge)
        if self.family=='hybrid':
            low=polynomial(s,self.poly_prefix,self.warp,self.mapping,self.outer,trial=True)
            return np.concatenate((low,random),axis=-1)
        return random


def weighted_inverse(a,g):
    d=np.sqrt(g.diagonal())
    factor=cholesky(g/d[:,None]/d,lower=True,check_finite=False)
    left=solve_triangular(factor,np.diag(1/d),lower=True,check_finite=False)
    weighted=left@a
    scale=np.linalg.norm(weighted,axis=0)
    x,_,rank,_=lstsq(weighted/scale,left,cond=1e-12,lapack_driver='gelsy',check_finite=False)
    return x/scale[:,None],int(rank)


class WeakTensor:
    def __init__(self,dt,m,k,p,q,scale=22.,warp=1.6,seed=17,family='rfm',mapping='inner',test_warp=0.,outer='farfield',poly_prefix=0,sampling='iid'):
        if not 1<=m<=p<q or k<1:
            raise ValueError('Require 1 <= trial size <= test size < quadrature order and k>=1')
        self.trial=Trial(m,family,seed,scale,warp,mapping,outer,poly_prefix,sampling)
        self.outer=outer
        self.k=np.arange(1,k+1)
        self.ntheta=case.angular_size(k)
        self.rhs_shape=(k,p)
        z,w=leggauss(q)
        self.s=(z+1)*(case.LENGTH/2)
        w=w*(case.LENGTH/2)
        r2=case.radius(self.s)**2
        self.r2=r2
        self.jets=self.trial.jets(self.s)
        self.lift=case.lift(self.s,outer)
        b,b_s,b_ss=polynomial(self.s,p,test_warp,mapping,outer)
        self.mass_test=b.T*(w*r2)
        self.test=b.T*w
        self.test_s=b_s.T*w
        invert, ranks = [],[]
        for mode in self.k:
            test_lap=(b_ss-mode**2*b).T
            omega=-(self.jets[2]-mode**2*self.jets[0])/r2[:,None]
            mass=self.mass_test@omega
            diffusion=(case.NU*test_lap*w)@omega
            gram=(test_lap*(w/r2))@test_lap.T
            matrix,rank=weighted_inverse(1.5*mass-dt*diffusion,gram)
            invert.append(matrix);ranks.append(rank)
            if mode==1:
                wb=-(self.lift[2]-self.lift[0])/r2
                mb=self.mass_test@wb
                db=(case.NU*test_lap*w)@wb
                self.bdf_lift=1.5*mb-dt*db
                self.start_lift=mb-dt*db
                self.start,start_rank=weighted_inverse(mass-dt*diffusion,gram)
        self.bdf=np.asarray(invert)
        self.metadata=dict(method=family,m=m,k=k,p=p,q=q,scale=scale,warp=warp,seed=seed,mapping=mapping,test_warp=test_warp,outer=outer,poly_prefix=poly_prefix,sampling=sampling,
            rank_min=min(ranks),rank_max=max(ranks),rank_start=start_rank,ntheta=self.ntheta)

    def advance(self,rhs,speed,first):
        if first:
            result=np.zeros((len(self.k),self.trial.m))
            result[0]=self.start@(rhs[0]-speed*self.start_lift)
            return result
        rhs[0]-=speed*self.bdf_lift
        return np.einsum('kmp,kp->km',self.bdf,rhs)

    def native(self,state,speed,jets,bs,r2):
        components=jets@state.T
        components[:,:,0]+=speed*bs
        psi,derivative,second=components
        omega=-(second-psi*self.k**2)/r2[:,None]
        return psi,derivative,omega

    def terms(self,state,speed):
        psi,derivative,omega=self.native(state,speed,self.jets,self.lift,self.r2)
        p=case.cosine_values(psi*self.k,self.ntheta)
        q=case.sine_values(derivative,self.ntheta)
        w=case.sine_values(omega,self.ntheta)
        s=case.sine_coeff(w*p,len(self.k))
        a=case.cosine_coeff(w*q,len(self.k))
        forcing=self.test_s@s-(self.test@a)*self.k
        return (self.mass_test@omega).T,forcing.T

    def modal(self,state,speed,s):
        return self.native(state,speed,self.trial.jets(s),case.lift(s,self.outer),case.radius(s)**2)
