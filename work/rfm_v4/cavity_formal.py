"""Five fresh-seed Re1000 cavity comparisons; run only after CPU timing GO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import statistics
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'rfm_v3'))
from cavity_study import Score,PROBLEM,errors,setting,interpolate_solution,np
from cavity_solver import solve
from rfm_v2_benchmark import write_json,one_thread,machine_info
from scipy.linalg import lstsq

loader=importlib.util.spec_from_file_location('v3_cavity_frozen_fd',ROOT.parent/'rfm_v3'/'cavity_formal.py')
legacy=importlib.util.module_from_spec(loader);loader.loader.exec_module(legacy)
SEEDS=(3203,3511,3821,4127,4457)
TIERS=(.001,.0005)
OUTPUT=ROOT/'cavity_formal.json'
FREEZE=ROOT/'cavity_formal_freeze.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs():
    def build(name,m,scale,p,q,strategy='guard',family='iid',strength=(1.9,1.9)):
        config=setting(m,scale,p,q,strength,family=family)
        config.update(strategy=strategy,alpha_ladder=[.25,.5,.75,1.])
        return dict(name=name,kind='legendre' if family=='legendre' else 'rfm',config=config)
    return [build('rfm496_guard',496,14,29,47),build('rfm496_enriched_guard',496,14,33,53),
        build('rfm595_guard',595,15,32,51),build('rfm595_enriched_guard',595,15,36,57),
        build('rfm666_ladder',666,16,34,54,'ladder'),
        build('matched_leg595_p32',595,1,32,51,'ladder','legendre'),
        build('matched_leg595_p36',595,1,36,57,'ladder','legendre'),
        build('strong_leg666_mild',666,1,39,58,'guard','legendre',(1.3,1.3)),
        dict(name='fd152_c32',kind='fd',config=dict(n=152,coarse=32)),
        dict(name='fd180_c32',kind='fd',config=dict(n=180,coarse=32))]


def run(spec,seed,score):
    row=dict(name=spec['name'],kind=spec['kind'],seed=seed,config=spec['config'])
    tick=time.perf_counter()
    try:
        if spec['kind']=='fd':
            native,metadata=legacy.fd_cold(**spec['config']);op=c=None
            elapsed=time.perf_counter()-tick
            evaluator=lambda xy:interpolate_solution(native,xy[:,0],xy[:,1])
        else:
            op,c,metadata=solve(PROBLEM,seed=seed,**spec['config']);elapsed=time.perf_counter()-tick
            evaluator=lambda xy:op.fields(c,xy)
        row.update(seconds=elapsed,metadata=metadata)
        scores=score.evaluate(evaluator,batch=spec['kind']!='fd');worst=errors(scores)
        if any(not np.isfinite(value) for value in worst.values()):raise ArithmeticError('Non-finite field score')
        row.update(status='completed',scores=scores,errors=worst,
                   passes={str(t):max(worst.values())<=t for t in TIERS})
        if seed==SEEDS[0]:
            arrays={}
            for name,(xy,w) in score.grids.items():
                blocks=[evaluator(b) for b in np.array_split(xy,int(np.ceil(len(xy)/256)))]
                arrays.update({f'{name}_{key}':np.concatenate([b[key] for b in blocks]) for key in ('u','v','omega')})
                arrays[f'{name}_xy'],arrays[f'{name}_weights']=xy,w
            if c is not None:arrays['coefficients']=c
            artifact=ROOT/f'cavity_state_{spec["name"]}_seed{seed}.npz'
            np.savez_compressed(artifact,**arrays);row['artifact']=dict(file=artifact.name,sha256=digest(artifact))
    except Exception as exc:
        row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        row.setdefault('seconds',time.perf_counter()-tick)
    print(row['name'],seed,row['status'],round(row['seconds'],5),row.get('errors',row.get('error')),flush=True)
    return row


def source_hashes():
    files=[ROOT/name for name in ('cavity_formal.py','cavity_solver.py','cavity_pilot.py','cavity_pilot.json',
                                 'cavity_control_pilot.py','cavity_control_pilot.json')]
    files += [ROOT.parent/'rfm_v3'/name for name in ('cavity_study.py','cavity_formal.py','cavity_reference_v3.py',
                                                    'cavity1000_ref256.npz','cavity1000_ref384.npz')]
    files += [ROOT.parent/'rfm_ns'/name for name in ('rfm_solver.py','rfm_v2_solver.py','rfm_v2_warp_legendre.py',
        'wall_trial.py','tensor_rfm.py','operator_opt.py','rfm_v2_benchmark.py','cfd_reference.py','cfd_reference_fourth.py')]
    return {str(path.resolve()):digest(path) for path in files}


def summarize(result):
    summary=[]
    for spec in result['specs']:
        group=[row for row in result['rows'] if row['name']==spec['name']]
        valid=[row for row in group if row['status']=='completed']
        worst={name:max((row['errors'][key] for row in valid),default=None)
               for name,key in (('velocity','velocity_rel_l2'),('vorticity','vorticity_rel_l2'))}
        passed={str(t):bool(len(valid)==len(SEEDS) and max(worst.values())<=t) for t in TIERS}
        summary.append(dict(**spec,runs=len(group),completed=len(valid),worst=worst,passes=passed,
            seconds=statistics.median(row['seconds'] for row in valid) if valid else None,
            times=[row['seconds'] for row in valid],
            fallback_runs=sum(bool(row['metadata'].get('fallback_triggers')) for row in valid),
            final_budget_unresolved_runs=sum(bool(row['metadata'].get('final_budget_unresolved')) for row in valid)))
    comparisons=[]
    ref=max(errors(result['reference_grid_change']).values())
    for tier in TIERS:
        passing=[row for row in summary if row['passes'][str(tier)]]
        def best(kind):
            rows=[row for row in passing if row['kind']==kind]
            return min(rows,key=lambda row:row['seconds']) if rows else None
        fd,rfm,poly=best('fd'),best('rfm'),best('legendre')
        comparisons.append(dict(tier=tier,reference_screen_passed=ref<=.2*tier,
            fastest_tested_passing_fd=fd['name'] if fd else None,
            fastest_tested_passing_rfm=rfm['name'] if rfm else None,
            fastest_tested_passing_legendre=poly['name'] if poly else None,
            fd_over_rfm=fd['seconds']/rfm['seconds'] if fd and rfm else None,
            legendre_over_rfm=poly['seconds']/rfm['seconds'] if poly and rfm else None))
    return summary,comparisons


def main():
    if OUTPUT.exists() or FREEZE.exists():raise FileExistsError('Preserve existing v4 formal evidence')
    with one_thread():
        score=Score();configurations=specs();hashes=source_hashes()
        schedule=[]
        for repeat,seed in enumerate(SEEDS):
            schedule.extend(dict(seed=seed,index=int(i)) for i in np.random.default_rng(64017+repeat).permutation(len(configurations)))
        frozen=dict(seeds=SEEDS,thresholds=TIERS,specs=configurations,schedule=schedule,source_sha256=hashes,
            machine=machine_info(),reference_grid_change=score.grid_change,
            final_operator='Unmodified rfm_ns.rfm_v2_solver.TensorWeak at alpha=1. Continuation changes only the initialization/iteration path.',
            timing='One CPU thread; entire fresh solve, setup, coarse solve, every failed attempt/restart and final field state. Reference preparation, independent scoring and serialization excluded.',
            protocol='All five fresh RFM seeds; five deterministic/FD repeats. Both velocity and vorticity on midpoint81 and area-weighted Gauss113 must pass. Reference-grid change <=20% of tier is empirical, not certified.',
            guard='Restart if maxiter exhausted with the last residual improvement above the existing stationarity tolerance; no field/reference error enters guard.')
        write_json(FREEZE,frozen)
        result=dict(status='running',freeze_sha256=digest(FREEZE),**frozen,rows=[],summary=[],comparisons=[])
        write_json(OUTPUT,result)
        rng=np.random.default_rng(338);lstsq(rng.normal(size=(24,12)),rng.normal(size=24),lapack_driver='gelsy')
        started=time.perf_counter()
        for job in schedule:
            result['rows'].append(run(configurations[job['index']],job['seed'],score))
            result['total_wall_seconds']=time.perf_counter()-started;write_json(OUTPUT,result)
        if any(digest(path)!=sha for path,sha in hashes.items()):
            result['status']='source_changed';write_json(OUTPUT,result);raise RuntimeError('A frozen source changed')
        result['summary'],result['comparisons']=summarize(result)
        result['status']='complete';write_json(OUTPUT,result)
        print('FORMAL_COMPLETE',str(OUTPUT),flush=True)


if __name__=='__main__':main()
