"""Re=100 full-annulus cylinder. No reflection or circulation constraint."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

from dataclasses import dataclass
import time
import numpy as np
from numpy.polynomial.legendre import leggauss

A, R, RE, NU = .5, 6., 100., .01
LENGTH = np.log(R/A)
PROBES = np.array([1.5,2.5,3.5])
PROBE_S = np.log(PROBES/A)
SNAPSHOTS = (24.,36.,48.,54.,60.,66.,72.)


def radius(s):
    return A*np.exp(s)


def speed(t):
    t=np.asarray(t)
    phase=np.pi*(t-4)/8
    uy=np.where((t>4)&(t<12),.12*np.sin(phase)**4,0.)
    return -.5*(uy+1j*(-np.expm1(-(t/.8)**3)))


def lift(s):
    z=s/LENGTH
    b=(LENGTH**2-2)/4
    a=1-b
    return R*np.stack((a*z*z+b*z**3,(2*a*z+3*b*z*z)/LENGTH,(2*a+6*b*z)/LENGTH**2))


def evaluation_grids():
    x,w=leggauss(123)
    s=(x+1)*(LENGTH/2)
    t=2*np.pi*(np.arange(433)+.23)/433
    mid=(np.arange(129)+.5)*(LENGTH/129)
    shifted=2*np.pi*(np.arange(449)+.43)/449
    return {'gauss123':(s,t,w*(LENGTH/2)*radius(s)**2),
            'mid129':(mid,shifted,(LENGTH/129)*radius(mid)**2)}


def layer_scoring_grid():
    points,weights=[],[]
    edges=(0.,LENGTH-.12,LENGTH-.018,LENGTH)
    for left,right,count in zip(edges,edges[1:],(96,24,24)):
        x,w=leggauss(count)
        s=left+(x+1)*((right-left)/2)
        points.append(s)
        weights.append(w*((right-left)/2)*radius(s)**2)
    t=2*np.pi*(np.arange(457)+.31)/457
    return np.concatenate(points),t,np.concatenate(weights)


def reconstruct(coefficients, theta):
    phase=np.exp(1j*np.arange(1,coefficients.shape[-1])[:,None]*theta)
    return coefficients[:,:1].real+2*(coefficients[:,1:]@phase).real


@dataclass
class Solution:
    model: object
    states: np.ndarray
    times: np.ndarray
    probe_times: np.ndarray
    transverse: np.ndarray
    vorticity: np.ndarray
    row: dict

    def modal(self,index,s):
        return self.model.modal(self.states[index],speed(self.times[index]),s)

    def fields(self,index,s,theta):
        return tuple(reconstruct(a,theta) for a in self.modal(index,s))


def step_indices(times,dt):
    index=np.round(np.asarray(times)/dt).astype(int)
    if not np.allclose(index*dt,times,atol=1e-10,rtol=0):
        raise ValueError('Checkpoint and probe times must be integer multiples of dt')
    return index


def solve(model_class, *, steps, horizon=72., snapshots=SNAPSHOTS, probe_spacing=.125, time_order=2, **config):
    started=time.perf_counter()
    dt=horizon/steps
    snaps=np.asarray(snapshots,dtype=float)
    if not len(snaps) or snaps[0]<=0 or np.any(np.diff(snaps)<=0) or snaps[-1]>horizon:
        raise ValueError('Checkpoints must be positive, ordered and within the horizon')
    snapshot_steps=set(step_indices(snaps,dt))
    probe_times=np.arange(0.,horizon+.01*probe_spacing,probe_spacing)
    probe_steps=step_indices(probe_times,dt)
    probe_stride=int(probe_steps[1])
    speeds=speed(dt*np.arange(steps+1))
    if time_order not in (2,3):
        raise ValueError('Expected SBDF2 or SBDF3')
    ratio=11/9 if time_order==3 else 1.
    model=model_class(dt=dt/ratio,**config)
    setup=time.perf_counter()-started
    probe=getattr(model,'probes',None)
    transverse=np.zeros((len(probe_times),len(PROBES)))
    vorticity=np.zeros_like(transverse)
    zero=np.zeros(model.rhs_shape,dtype=np.complex128)
    mass=flux=old_mass=old_flux=older_mass=older_flux=zero
    states=[]
    for i in range(1,steps+1):
        first=i==1 and time_order==2
        if time_order==3:
            # Rest and the cubic ramp give third-order consistent zero histories.
            rhs=(3*mass-1.5*old_mass+older_mass/3+dt*(3*flux-3*old_flux+older_flux))/ratio
        else:
            rhs=mass+dt*flux if first else 2*mass-.5*old_mass+dt*(2*flux-old_flux)
        state=model.advance(rhs,speeds[i],first)
        if i in snapshot_steps:
            if not np.isfinite(state).all():
                raise FloatingPointError(f'Nonfinite state at t={i*dt:.5g}')
            states.append(state.copy())
        if i%probe_stride==0:
            if probe is None:
                _,ut,omega=model.modal(state,speeds[i],PROBE_S)
                value=ut[:,0].real+2*ut[:,1:].real.sum(axis=1)
                vort=omega[:,0].real+2*omega[:,1:].real.sum(axis=1)
            else:
                value,vort=probe(state,speeds[i])
            if not np.isfinite(value).all() or not np.isfinite(vort).all():
                raise FloatingPointError(f'Nonfinite probe at t={i*dt:.5g}')
            j=i//probe_stride
            transverse[j],vorticity[j]=value,vort
        if i!=steps:
            older_mass,older_flux=old_mass,old_flux
            old_mass,old_flux=mass,flux
            mass,flux=model.terms(state,speeds[i])
    row=dict(steps=steps,horizon=horizon,dt=dt,setup_seconds=setup,
             seconds=time.perf_counter()-started,time_order=time_order,**model.metadata)
    return Solution(model,np.asarray(states),snaps,probe_times,transverse,vorticity,row)


def reference_fields(solution,grids):
    return {name:[solution.fields(i,s,theta) for i in range(len(solution.times))]
            for name,(s,theta,w) in grids.items()}


def score(solution,truth,grids):
    rows=[]
    for name,(s,theta,w) in grids.items():
        near=w*(radius(s)<=2.5)
        for i,t in enumerate(solution.times):
            u,v,o=solution.fields(i,s,theta)
            ur,vr,orr=truth[name][i]
            def norm(delta,base,weight):
                return float(np.sqrt(weight@delta.sum(axis=1)/(weight@base.sum(axis=1))))
            e=(u-ur)**2+(v-vr)**2
            werr=(o-orr)**2
            vref=ur**2+vr**2
            oref=orr**2
            rows.append(dict(grid=name,time=float(t),velocity=norm(e,vref,w),vorticity=norm(werr,oref,w),
                near_velocity=norm(e,vref,near),near_vorticity=norm(werr,oref,near)))
    return rows


def maxima(rows):
    return {key:max(row[key] for row in rows) for key in (
        'velocity','vorticity','near_velocity','near_vorticity')}


def store_solution(path,solution,config):
    import json
    np.savez_compressed(path,states=solution.states,times=solution.times,probe_times=solution.probe_times,
        transverse=solution.transverse,vorticity=solution.vorticity,config=json.dumps(config),
        metadata=json.dumps(solution.row))


def load_solution(path,model_class):
    import json
    with np.load(path,allow_pickle=False) as data:
        row=json.loads(str(data['metadata']))
        config=json.loads(str(data['config']))
        ratio=11/9 if row.get('time_order',2)==3 else 1.
        return Solution(model_class(dt=row['dt']/ratio,**config),data['states'],data['times'],data['probe_times'],
                        data['transverse'],data['vorticity'],row)
