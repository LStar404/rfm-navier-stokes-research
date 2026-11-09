"""Restart the fine FD run for late diagnostic frames, outside accuracy/timing."""
import argparse
import json
from pathlib import Path

from fd_full import FullFD, ROOT, case, np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,
        default=ROOT/'fd_full_n200_k144_rate256_outer1.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'fd_full_dense_from54.npz')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = case.load_solution(args.source,FullFD)
    with np.load(args.source,allow_pickle=False) as saved:
        config=json.loads(str(saved['config']))
    dt=source.row['dt']
    model=source.model
    start=54.
    times=np.arange(60.,66.01,.25)
    checkpoints=set(case.step_indices(times-start,dt))
    state=source.states[np.flatnonzero(source.times==start)[0]].copy()
    mass,flux=model.terms(state,case.speed(start))
    states=[]
    for i in range(1,max(checkpoints)+1):
        first=i==1
        rhs=mass+dt*flux if first else 2*mass-.5*previous_mass+dt*(2*flux-previous_flux)
        speed=case.speed(start+i*dt)
        state=model.advance(rhs,speed,first)
        if i in checkpoints:
            if not np.isfinite(state).all():
                raise FloatingPointError(f'Nonfinite restart at {start+i*dt}')
            states.append(state.copy())
        previous_mass,previous_flux=mass,flux
        mass,flux=model.terms(state,speed)
    states=np.asarray(states)
    diagnostics={}
    for t in (60.,66.):
        original=source.states[np.flatnonzero(source.times==t)[0]]
        recent=states[np.flatnonzero(times==t)[0]]
        diagnostics[str(t)]=float(np.linalg.norm(recent-original)/np.linalg.norm(original))
    row=dict(source.row,diagnostic_only=True,restart_time=start,
             source=args.source.name,checkpoint_state_relative=diagnostics)
    result=case.Solution(model,states,times,source.probe_times,source.transverse,source.vorticity,row)
    case.store_solution(args.output,result,config)
    print(json.dumps(row,indent=2),flush=True)


if __name__=='__main__':
    main()
