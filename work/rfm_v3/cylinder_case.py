"""Finite-annulus, reflection-symmetric transient crossflow past a cylinder."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

from dataclasses import dataclass
import time
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from numpy.polynomial.legendre import leggauss

A, R, RE, T, RAMP = .5, 5., 100., 4., .8
LENGTH = np.log(R/A)
NU = 2*A/RE
TIMES = np.arange(1,9)*.5


def outer_speed(t):
    return -np.expm1(-(t/RAMP)**3)


def radius(s):
    return A*np.exp(s)


def lift(s,outer='farfield'):
    """Unit-speed lift with no slip on the cylinder."""
    z = s/LENGTH
    if outer=='farfield':
        # psi=R and psi_ss=R at the outer boundary for the unit sine mode.
        b=(LENGTH**2-2)/4
        a=1-b
        return R*np.stack((a*z*z+b*z**3,(2*a*z+3*b*z*z)/LENGTH,(2*a+6*b*z)/LENGTH**2))
    if outer!='dirichlet':
        raise ValueError(outer)
    return R*np.stack((3*z*z-2*z**3+LENGTH*(z**3-z*z),
        (6*z-6*z*z)/LENGTH+3*z*z-2*z,
        (6-12*z)/LENGTH**2+(6*z-2)/LENGTH))


def angular_size(k):
    # Fourier products have modes <=2*k; >3*k excludes alias to retained k.
    return next_fast_len(3*k+1,real=True)


def sine_values(a,n):
    shape = (*a.shape[:-1], n//2+1)
    c = np.zeros(shape,dtype=complex)
    c[...,1:a.shape[-1]+1] = (-.5j*n)*a
    return irfft(c,n=n,axis=-1,workers=1)


def cosine_values(a,n):
    shape = (*a.shape[:-1], n//2+1)
    c = np.zeros(shape,dtype=complex)
    c[...,1:a.shape[-1]+1] = (.5*n)*a
    return irfft(c,n=n,axis=-1,workers=1)


def sine_coeff(a,k):
    return (-2/a.shape[-1])*rfft(a,axis=-1,workers=1)[...,1:k+1].imag


def cosine_coeff(a,k):
    return (2/a.shape[-1])*rfft(a,axis=-1,workers=1)[...,1:k+1].real


def modal_fields(psi,psi_s,omega,s,theta):
    k = np.arange(1,psi.shape[1]+1)
    phase = k[:,None]*theta
    sin,cos = np.sin(phase),np.cos(phase)
    r = radius(s)[:,None]
    return dict(u=(psi*k)@cos/r,v=-(psi_s@sin)/r,omega=omega@sin,psi=psi@sin)


@dataclass
class Solution:
    model: object
    states: np.ndarray
    row: dict

    def fields(self,index,s,theta):
        psi,psi_s,omega = self.model.modal(self.states[index],outer_speed(TIMES[index]),s)
        return modal_fields(psi,psi_s,omega,s,theta)


def solve(model_class, steps, **settings):
    """Common first-order start plus SBDF2. Checkpoint states are included in time."""
    if steps<8 or steps%8:
        raise ValueError('Use a multiple of eight steps for the frozen output times')
    started=time.perf_counter()
    dt=T/steps
    model=model_class(dt=dt,**settings)
    setup=time.perf_counter()-started
    zero=np.zeros(model.rhs_shape)
    mass,nonlinear=zero,zero
    older_mass,older_nonlinear=zero,zero
    states=[]
    for step in range(1,steps+1):
        first=step==1
        rhs=mass+dt*nonlinear if first else 2*mass-.5*older_mass+dt*(2*nonlinear-older_nonlinear)
        speed=outer_speed(step*dt)
        state=model.advance(rhs,speed,first)
        if step%(steps//8)==0:
            if not np.isfinite(state).all():
                raise FloatingPointError(f'Nonfinite state at step {step}')
            states.append(state.copy())
        if step!=steps:
            older_mass,older_nonlinear=mass,nonlinear
            mass,nonlinear=model.terms(state,speed)
    checkpoints=np.asarray(states)
    elapsed=time.perf_counter()-started
    return Solution(model,checkpoints,dict(steps=steps,dt=dt,setup_seconds=setup,
        seconds=elapsed,**model.metadata))


def evaluation_grids():
    """Independent area-weighted quadrature and a midpoint grid with angular offset."""
    z,w=leggauss(113)
    s=(z+1)*(LENGTH/2)
    theta=2*np.pi*(np.arange(193)+.37)/193
    radial=radius(s)**2*w*(LENGTH/2)
    midpoint=(np.arange(117)+.5)*(R-A)/117+A
    return {'gauss113':(s,theta,radial[:,None]*np.full((1,len(theta)),2*np.pi/len(theta))),
            'midpoint117':(np.log(midpoint/A),theta+.23/193,
                midpoint[:,None]*np.full((1,len(theta)),(R-A)/117*2*np.pi/len(theta)))}


def score(solution,truth,grids=None):
    grids=evaluation_grids() if grids is None else grids
    rows=[]
    for name,(s,theta,w) in grids.items():
        for i,t in enumerate(TIMES):
            a,b=solution.fields(i,s,theta),truth.fields(i,s,theta)
            e2=(a['u']-b['u'])**2+(a['v']-b['v'])**2
            v2=b['u']**2+b['v']**2
            w2=(a['omega']-b['omega'])**2
            wt=b['omega']**2
            def rel(delta,ref,weights):
                return float(np.sqrt(np.sum(weights*delta)/np.sum(weights*ref)))
            near=w*(radius(s)[:,None]<=2.)
            rows.append(dict(grid=name,t=float(t),velocity=rel(e2,v2,w),vorticity=rel(w2,wt,w),
                near_velocity=rel(e2,v2,near),near_vorticity=rel(w2,wt,near)))
    return rows


def maxima(rows):
    return {key:max(r[key] for r in rows) for key in ('velocity','vorticity','near_velocity','near_vorticity')}
