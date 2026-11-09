"""CPU counterpart of the existing CUDA Galerkin precontraction.

This is an opt-in candidate, not a change to the frozen CPU implementation.
The generic SBDF2 driver now receives A^{-1} times both histories; the first
step remains the original implicit-Euler start from rest.
"""
from pathlib import Path
import json
import time

from basis_recurrence_model import RecurrenceGalerkin
import cpu_formal as base

np, case = base.case.np, base.case
HERE = Path(__file__).resolve().parent


class PrecontractedGalerkin(RecurrenceGalerkin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        start = time.perf_counter()
        mass = self.M0 + self.k2[:,None,None]*self.M1
        self.pc_mass = self.implicit @ mass
        self.pc_radial = self.implicit @ self.test_s
        self.pc_angular = self.implicit @ self.test
        self.pc_mass_lift = self.implicit[0] @ self.M_lift
        self.pc_bdf_lift = self.implicit[0] @ self.bdf_lift
        self.pc_mean_mass = self.mean_inverse @ self.mean_mass
        self.pc_mean_radial = -(self.mean_inverse @ self.mean_test)
        self._pc_angular_work = np.empty(self.rhs_shape, dtype=np.complex128)
        self.metadata.update(implementation='cpu-galerkin-precontracted-strided',
                             precontraction_seconds=time.perf_counter()-start,
                             reconstruction_class='cpu_precontracted.PrecontractedGalerkin')

    def _apply_modes(self, matrix, values, out):
        rows = values.shape[0]
        source = np.ascontiguousarray(values).view(float).reshape(rows,self.k+1,2)[:,1:].transpose(1,0,2)
        target = out.view(float).reshape(self.width,self.k+1,2)[:,1:].transpose(1,0,2)
        np.matmul(matrix,source,out=target)

    def terms(self, state, amplitude):
        ur,ut,omega = self.fields_on(state,amplitude,self.samples,self.lift,self.r,self.mean_samples)
        radial,angular = self.kernel(ur,ut,omega)
        mass,flux = np.empty_like(state),np.empty_like(state)
        self._apply_modes(self.pc_mass,state,mass)
        mass[:,1] += amplitude*self.pc_mass_lift
        mass[:,0] = 0.
        mass[:self.mean_width,0] = self.pc_mean_mass @ state[:self.mean_width,0].real
        self._apply_modes(self.pc_radial,radial,flux)
        self._apply_modes(self.pc_angular,angular,self._pc_angular_work)
        flux[:,1:] -= self.ik*self._pc_angular_work[:,1:]
        flux[:,0] = 0.
        flux[:self.mean_width,0] = self.pc_mean_radial @ radial[:,0].real
        return mass,flux

    def advance(self, rhs, amplitude, first):
        if first:
            return super().advance(rhs,amplitude,first)
        rhs[:,1] -= amplitude*self.pc_bdf_lift
        rhs[self.mean_width:,0] = 0.
        return rhs


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),np.finfo(float).tiny))


def audit_one(spec, protocol):
    seed=protocol['seeds'][0] if spec['method']=='rfm' else protocol['deterministic_seed']
    _,config=base.config(spec,protocol,seed)
    model=PrecontractedGalerkin(dt=1/spec['rate'],**config)
    old=RecurrenceGalerkin(dt=1/spec['rate'],**config)
    source=base.OUT/f'qualify_00_{spec["name"]}_{seed}.npz'
    with np.load(source,allow_pickle=False) as a:
        states=a['states']
    checks=[]
    for i in (0,len(states)-1):
        state=states[i].copy();amplitude=case.speed(protocol['checkpoints'][i])
        a=old.terms(state,amplitude)
        b=model.terms(state,amplitude)
        # A^{-1} applied to physical mass/flux histories, without a new-step lift.
        expected=[]
        for val in a:
            mapped=old.advance(val.copy(),0.,False)
            expected.append(mapped)
        errors=[relative(v,w) for v,w in zip(b,expected)]
        combined_old=old.advance(1.5*a[0]+(.5/spec['rate'])*a[1],amplitude,False)
        combined_new=model.advance(1.5*b[0]+(.5/spec['rate'])*b[1],amplitude,False)
        errors.append(relative(combined_new,combined_old))
        if max(errors)>1e-9 or not np.isfinite(errors).all():
            raise ArithmeticError(f'Precontraction fixed-state mismatch: {errors}')
        checks.append(dict(time=protocol['checkpoints'][i],relative=errors))
    outputs=[];solutions=[]
    for cls in (RecurrenceGalerkin,PrecontractedGalerkin):
        sol=case.solve(cls,steps=8*spec['rate'],horizon=8.,snapshots=(4.,8.),**config)
        solutions.append(sol)
        outputs.append(dict(name=cls.__name__,seconds=sol.row['seconds'],setup=sol.row['setup_seconds']))
    truth=case.reference_fields(solutions[0],base.grid_set())
    errors=case.maxima(case.score(solutions[1],truth,base.grid_set()))
    probes={key:relative(getattr(solutions[1],key),getattr(solutions[0],key))
            for key in ('transverse','vorticity')}
    if max(*errors.values(),*probes.values())>1e-7:
        raise ArithmeticError(f'Short trajectory mismatch: {errors}, {probes}')
    return dict(name=spec['name'],seed=seed,fixed_checks=checks,short_field_errors=errors,
                short_probe_errors=probes,diagnostic_times=outputs)


if __name__=='__main__':
    destination=HERE/'cpu_precontracted_pilot.json'
    if destination.exists():
        raise FileExistsError('Preserve the previous precontraction pilot')
    protocol=base.read(base.PROTOCOL)
    rows=[]
    for spec in protocol['variants']:
        if spec['name'] in ('rfm37_k72','leg37_k72'):
            rows.append(audit_one(spec,protocol))
            base.save(destination,dict(status='running',timing_scope='non-isolated pilot only',rows=rows))
            print(json.dumps(rows[-1]),flush=True)
    base.save(destination,dict(status='complete',timing_scope='non-isolated pilot only',rows=rows))
