"""Check the full Galerkin nonlinear work cancellation on stored fields."""
import json
from pathlib import Path

import full_case as case
from full_case import np
from full_galerkin import FullGalerkin
from full_galerkin_buffered import BufferedGalerkin

ROOT=Path(__file__).resolve().parent


def weighted_pair(state,load):
    return np.vdot(state[:,0],load[:,0]).real+2*np.vdot(state[:,1:],load[:,1:]).real


def main():
    path=ROOT/'full_state_rfm35_k64.npz'
    solution=case.load_solution(path,BufferedGalerkin)
    s,w=case.leggauss(101)
    s=(s+1)*case.LENGTH/2
    w*=case.LENGTH/2
    data=[]
    for i,(t,state) in enumerate(zip(solution.times,solution.states)):
        _,self_flux=solution.model.terms(state,0.)
        _,all_flux=solution.model.terms(state,case.speed(t))
        work=weighted_pair(state,self_flux)
        scale=np.sqrt(weighted_pair(state,state)*weighted_pair(self_flux,self_flux))
        ur,ut,omega=solution.modal(i,s)
        mean_fraction=np.sqrt((w@((case.radius(s)*ut[:,0].real)**2))/
            (w@((case.radius(s)**2)*(abs(ut[:,0])**2+2*np.sum(
                abs(ur[:,1:])**2+abs(ut[:,1:])**2,axis=1)))))
        data.append(dict(time=float(t),relative_homogeneous_work=float(abs(work)/scale),
                         lift_work=float(weighted_pair(state,all_flux)-work),
                         mean_velocity_fraction=float(mean_fraction),
                         max_mean_omega=float(abs(omega[:,0]).max())))
    baseline=case.load_solution(ROOT/'full_state_rfm35_k64_basic.npz',FullGalerkin)
    grids=case.evaluation_grids()
    grids['layer144']=case.layer_scoring_grid()
    difference=case.score(solution,case.reference_fields(baseline,grids),grids)
    result=dict(energy=data,trajectory_difference=case.maxima(difference),scores=difference)
    (ROOT/'full_energy_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
