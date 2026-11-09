"""Pinned cold solves, frozen checkpoints and independent full-wake gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import full_case as case
from full_case import np
from full_galerkin import FullGalerkin
from full_galerkin_buffered import BufferedGalerkin
from fd_full import FullFD
from diagnostic_wake import analyze

ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'rfm_v5'))
from timing_v5 import pin

SEEDS=(7331,7687,8011)
GATES=(.02,.01,.005,.001,.0005)
ENGINES={'gal':FullGalerkin,'buffered':BufferedGalerkin,'fd':FullFD}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def boundary_errors(sol):
    s=np.array([0.,case.LENGTH])
    errors=np.zeros(4)
    for i,t in enumerate(sol.times):
        ur,ut,omega=sol.modal(i,s)
        target=np.zeros_like(ur[1])
        target[1]=1j*case.speed(t)
        errors=np.maximum(errors,[np.linalg.norm(ur[0]),np.linalg.norm(ut[0]),
                                  np.linalg.norm(ur[1]-target),np.linalg.norm(omega[1])])
    return dict(zip(('wall_ur','wall_ut','outer_ur_error','outer_vorticity'),map(float,errors)))


def summarize(specs,rows,gate_screens):
    result=[]
    for spec in specs:
        group=[r for r in rows if r['name']==spec['name'] and 'errors' in r]
        durations=[r['seconds'] for r in group]
        worst={key:max(r['errors'][key] for r in group) for key in group[0]['errors']} if group else None
        shedding=len(group)==len(SEEDS) and all(r['wake']['sustained_alternating_probe_signs'] for r in group)
        matches={str(g):bool(shedding and worst and max(worst.values())<=g) for g in GATES}
        result.append(dict(**spec,completed=len(group),times=durations,worst=worst,
            seconds=statistics.median(durations) if durations else None,shedding=shedding,
            numerical_pass=matches,passes={g:passed and gate_screens[g] for g,passed in matches.items()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--specs',type=Path,default=ROOT/'full_specs.json')
    p.add_argument('--output',type=Path,default=ROOT/'full_formal.json')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    specs=json.loads(a.specs.read_text(encoding='utf8'))
    if not specs or len({s['name'] for s in specs})!=len(specs):
        raise ValueError('Require unique configurations')
    sources=[a.specs,a.reference,a.predecessor,
             *[ROOT/n for n in ('full_formal.py','full_case.py','full_galerkin.py',
                'full_galerkin_buffered.py','radial_layers.py','full_nonlinear.py','fd_full.py',
                'fd_full_band.c','fd_full_band.dll','diagnostic_wake.py')],
             ROOT.parent/'rfm_v5'/'timing_v5.py',
             *[ROOT.parent/'rfm_v3'/n for n in ('cylinder_case.py','cylinder_rfm.py','cylinder_fd.py')]]
    workspace=ROOT.parents[1]
    hashes={f.resolve().relative_to(workspace).as_posix():digest(f) for f in sources}
    cpu=pin()
    reference=case.load_solution(a.reference,FullFD)
    predecessor=case.load_solution(a.predecessor,FullFD)
    if not np.array_equal(reference.times,predecessor.times):
        raise ValueError('Expected matching reference checkpoints')
    grids=case.evaluation_grids()
    grids['layer144']=case.layer_scoring_grid()
    truth=case.reference_fields(reference,grids)
    reference_scores=case.score(predecessor,truth,grids)
    change=case.maxima(reference_scores)
    screens={str(g):max(change.values())<=.25*g for g in GATES}
    wake_ref=analyze(reference.probe_times,reference.transverse,reference.vorticity)
    if not wake_ref['sustained_alternating_probe_signs']:
        raise ValueError('Reference does not exhibit persistent alternating shedding')
    np.linalg.svd(np.eye(10))
    rng=np.random.default_rng(800613)
    schedule=[(seed,int(i)) for seed in SEEDS for i in rng.permutation(len(specs))]
    result=dict(status='running',specs=specs,hashes=hashes,seeds=SEEDS,gates=GATES,schedule=schedule,
        rows=[],cpu=cpu,reference=a.reference.name,predecessor=a.predecessor.name,
        reference_metadata=reference.row,predecessor_metadata=predecessor.row,
        reference_change=change,reference_scores=reference_scores,gate_screens=screens,
        reference_wake=wake_ref,thread_environment={key:os.environ[key] for key in (
            'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        policy='Cold t=0–72. Absolute checkpoint times, no phase shift. Whole field and near cylinder, three independent grids, velocity and curl. Gates require all runs plus reference refinement change <=25% of gate. Setup, march, saved probes included.')
    dump(a.output,result)
    for seed,i in schedule:
        spec=specs[i]
        config=dict(spec['config'])
        if spec['kind']!='fd':
            config['seed']=seed
        start=time.perf_counter()
        try:
            with np.errstate(over='ignore',invalid='ignore'):
                sol=case.solve(ENGINES[spec['engine']],steps=72*spec['rate'],**config)
            elapsed=time.perf_counter()-start
            scores=case.score(sol,truth,grids)
            row=dict(name=spec['name'],seed=seed,seconds=elapsed,errors=case.maxima(scores),
                     metadata=sol.row,scores=scores,boundary_errors=boundary_errors(sol),
                     wake=analyze(sol.probe_times,sol.transverse,sol.vorticity))
            if seed==SEEDS[0]:
                target=a.output.parent/f'full_state_{spec["name"]}.npz'
                case.store_solution(target,sol,config)
                row['artifact']=dict(file=target.name,sha256=digest(target))
        except (ArithmeticError,ValueError,np.linalg.LinAlgError) as exc:
            row=dict(name=spec['name'],seed=seed,seconds=time.perf_counter()-start,failure=str(exc))
        result['rows'].append(row)
        dump(a.output,result)
        print(spec['name'],seed,row['seconds'],row.get('errors',row.get('failure')),flush=True)
    if any(digest(workspace/f)!=sha for f,sha in hashes.items()):
        raise RuntimeError('Frozen source or reference changed')
    result.update(status='complete',summary=summarize(specs,result['rows'],screens))
    dump(a.output,result)


if __name__=='__main__':
    main()
