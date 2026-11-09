"""Full-circle vorticity–streamfunction Galerkin with a mean momentum mode."""
from pathlib import Path
import sys

import full_case as case
from full_case import np
from full_nonlinear import FullNonlinear
from radial_layers import layer_jets, radial_quadrature
from scipy.linalg import cho_factor, cho_solve
from numpy.polynomial.legendre import legvander, legder, legval

ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'rfm_v3'))
from cylinder_rfm import Trial, case as legacy_case


def scaled_svd(values,cutoff):
    scale=np.linalg.norm(values,axis=0)
    _,singular,vh=np.linalg.svd(values/scale,full_matrices=False)
    keep=singular>cutoff*singular[0]
    return (vh[keep].T/scale[:,None])/singular[keep]


def inverse_spd(matrix):
    scale=np.sqrt(np.diag(matrix))
    factor=cho_factor(matrix/scale[:,None]/scale,check_finite=False)
    return cho_solve(factor,np.eye(len(scale)),check_finite=False)/scale[:,None]/scale


def real_apply(operator,values):
    """A real radial matrix acts on adjacent real and imaginary columns."""
    return (operator@values.view(float)).view(complex)


def mean_jets(s,m):
    z=s/case.LENGTH
    x=2*z-1
    v=legvander(x,m-2)
    d=np.column_stack([legval(x,legder(c)) for c in np.eye(m-1)])*2
    weight=z*(1-z)**2
    dw=(1-z)*(1-3*z)
    return np.stack((np.column_stack((z*((case.LENGTH+2)-(case.LENGTH+1)*z),weight[:,None]*v)),
        np.column_stack((case.LENGTH+2-2*(case.LENGTH+1)*z,dw[:,None]*v+weight[:,None]*d))/case.LENGTH))


class FullGalerkin:
    def __init__(self,dt,m,k,q,*,family='rfm',seed=17,scale=47.,warp=0.,cutoff=1e-10,
                 mean_order=24,backend='fft',mapping='inner',layers=(),split=False):
        if not 3<=m+len(layers)<q or not 3<=mean_order+len(layers)<q:
            raise ValueError('Require resolved basis sizes')
        self.k=k
        self.modes=np.arange(1,k+1)
        self.ik=1j*self.modes
        self.k2=self.modes**2
        self.mean_order=mean_order
        self.layers=tuple(layers)
        self.trial=Trial(m,family,seed,scale,warp,mapping,'farfield',
                         sampling='paired' if family=='rfm' else 'iid')
        self.s,w=radial_quadrature(q,split)
        r=case.radius(self.s)
        self.r,self.r2=r,r**2
        sw=np.sqrt(w)
        raw=self.raw_jets(self.s)
        self.transform=scaled_svd((raw[2]-raw[0])*(sw/r)[:,None],cutoff)
        self.samples=raw@self.transform
        f,d,d2=self.samples
        self.width=f.shape[1]
        if mean_order+len(layers)>self.width:
            raise ValueError('Mean order must not exceed the retained streamfunction rank')
        self.rhs_shape=(self.width,k+1)
        self.lift=case.lift(self.s)
        lf,ld,ld2=self.lift
        self.M0=(d.T*w)@d
        self.M1=(f.T*w)@f
        mass=self.M0+self.k2[:,None,None]*self.M1
        l1=(d2-f)*(sw/r)[:,None]
        fw=f*(sw/r)[:,None]
        cross=l1.T@fw
        delta=(self.k2-1)[:,None,None]
        diffusion=case.NU*((l1.T@l1)-delta*(cross+cross.T)+delta**2*(fw.T@fw))
        self.M_lift=(d.T*w)@ld+(f.T*w)@lf
        K_lift=case.NU*l1.T@((ld2-lf)*sw/r)
        self.implicit=np.stack([inverse_spd(a) for a in 1.5*mass+dt*diffusion])
        self.bdf_lift=1.5*self.M_lift+dt*K_lift
        self.start=-inverse_spd(mass[0]+dt*diffusion[0])@(self.M_lift+dt*K_lift)
        mean=self.raw_mean(self.s)
        self.mean_transform=scaled_svd((mean[0]+mean[1])*sw[:,None],cutoff)
        self.mean_samples=mean@self.mean_transform
        self.mean_width=self.mean_samples.shape[-1]
        mv,md=self.mean_samples
        test=mv.T*(w*self.r2)
        self.mean_test=test
        self.mean_mass=test@mv
        strain=(mv+md)*sw[:,None]
        self.mean_inverse=inverse_spd(1.5*self.mean_mass+dt*case.NU*(strain.T@strain))
        self.test=f.T*(w*r)
        self.test_s=d.T*(w*r)
        self.kernel=FullNonlinear(q,k,backend)
        self.probe_values=self.precompute(case.PROBE_S)
        self.metadata=dict(m=m,k=k,q=q,method=family,scale=scale,warp=warp,seed=seed,rank=self.width,
                           mean_order=mean_order,mean_rank=self.mean_width,backend=backend,
                           ntheta=self.kernel.n,mapping=mapping,layers=self.layers,split=split)

    def raw_jets(self,s):
        scale=legacy_case.LENGTH/case.LENGTH
        raw=self.trial.jets(s*scale)*np.array([1.,scale,scale**2])[:,None,None]
        return np.concatenate((raw,layer_jets(s,self.layers)),axis=-1) if self.layers else raw

    def raw_mean(self,s):
        raw=mean_jets(s,self.mean_order)
        return np.concatenate((raw,layer_jets(s,self.layers,mean=True)),axis=-1) if self.layers else raw

    def advance(self,rhs,amplitude,first):
        result=np.zeros_like(rhs)
        if first:
            result[:,1]=amplitude*self.start
            return result
        rhs[:,1]-=amplitude*self.bdf_lift
        real=np.ascontiguousarray(rhs[:,1:].T).view(float).reshape(self.k,self.width,2)
        transformed=self.implicit@real
        result[:,1:]=transformed.view(complex)[...,0].T
        result[:self.mean_width,0]=self.mean_inverse@rhs[:self.mean_width,0].real
        return result

    def precompute(self,s):
        return (self.raw_jets(s)@self.transform,case.lift(s),case.radius(s),
                self.raw_mean(s)@self.mean_transform)

    def fields_on(self,state,amplitude,samples,lift,r,mean):
        coefficients=np.ascontiguousarray(state[:,1:])
        size=len(r)
        psi,derivative,second=real_apply(samples.reshape(3*size,self.width),coefficients).reshape(3,size,self.k)
        psi[:,0]+=amplitude*lift[0]
        derivative[:,0]+=amplitude*lift[1]
        second[:,0]+=amplitude*lift[2]
        mstate=state[:self.mean_width,0].real
        v,dv=mean@mstate
        ur=np.zeros((len(r),self.k+1),dtype=complex)
        ut=np.empty_like(ur)
        omega=np.empty_like(ur)
        ur[:,1:]=psi*self.ik/r[:,None]
        ut[:,1:]=-derivative/r[:,None]
        omega[:,1:]=(psi*self.k2-second)/(r*r)[:,None]
        ut[:,0]=v
        omega[:,0]=(v+dv)/r
        return ur,ut,omega

    def terms(self,state,amplitude):
        ur,ut,omega=self.fields_on(state,amplitude,self.samples,self.lift,self.r,self.mean_samples)
        radial,angular=self.kernel(ur,ut,omega)
        mass=np.empty_like(state)
        flux=np.empty_like(state)
        x=np.ascontiguousarray(state[:,1:])
        mass[:,1:]=real_apply(self.M0,x)+real_apply(self.M1,x)*self.k2
        mass[:,1]+=amplitude*self.M_lift
        mass[:,0]=0.
        mass[:self.mean_width,0]=self.mean_mass@state[:self.mean_width,0].real
        flux[:,1:]=real_apply(self.test_s,np.ascontiguousarray(radial[:,1:]))-self.ik*real_apply(
            self.test,np.ascontiguousarray(angular[:,1:]))
        flux[:,0]=0.
        flux[:self.mean_width,0]=-self.mean_test@radial[:,0].real
        return mass,flux

    def modal(self,state,amplitude,s):
        return self.fields_on(state,amplitude,*self.precompute(s))

    def probes(self,state,amplitude):
        _,ut,omega=self.fields_on(state,amplitude,*self.probe_values)
        return ut[:,0].real+2*ut[:,1:].real.sum(axis=1),omega[:,0].real+2*omega[:,1:].real.sum(axis=1)
