"""Frozen Re1000 smooth-lid comparison, including full cold FD initialization.

Run --stage all only in an isolated CPU window. Pilot selection never uses the
formal seeds. Both velocity and vorticity must pass on midpoint81 and Gauss113.
Reference generation and independent scoring are outside all solver timings.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from cavity_study import (ROOT,PROBLEM,Score,setting,errors,solve_rfm,
    solve_mapped_legendre,solve_cavity,CavityFourth,interpolate_solution,np)
from rfm_v2_benchmark import one_thread,write_json,machine_info
from scipy.linalg import lstsq

SEEDS=(1201,1301,1423)
TIERS=(.001,.0005)
OUTPUT=ROOT/'cavity_formal.json'
PILOT=ROOT/'cavity_formal_pilot.json'
FREEZE=ROOT/'cavity_formal_freeze.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(name,m,scale,p,q,warp=(1.9,1.9),method='rfm'):
    return dict(name=name,method=method,config=setting(m,scale,p,q,warp,
        family='legendre' if method!='rfm' else 'iid'))


def fd_cold(n,coarse):
    """No reference files; all stages start from a newly computed zero-field solve."""
    stages=[];failures=[];state=None;previous=0
    for re in (100,400,700,1000):
        started=time.perf_counter()
        try:
            state,row=solve_cavity(coarse,re,initial=state,_model_class=CavityFourth,maxiter=35)
            stages.append(row)
        except (RuntimeError,FloatingPointError) as exc:
            failures.append(dict(Re=re,error=str(exc),seconds=time.perf_counter()-started))
            midpoint=(previous+re)/2
            state,row=solve_cavity(coarse,midpoint,initial=state,_model_class=CavityFourth,maxiter=35);stages.append(row)
            state,row=solve_cavity(coarse,re,initial=state,_model_class=CavityFourth,maxiter=35);stages.append(row)
        previous=re
    state,row=solve_cavity(n,1000,initial=state,_model_class=CavityFourth,maxiter=35);stages.append(row)
    return state,dict(n=n,coarse=coarse,stages=stages,failed_attempts=failures,
                      success=all(row['success'] for row in stages))


def run(specification,seed,score,save=False):
    tick=time.perf_counter();row=dict(name=specification['name'],method=specification['method'],seed=seed,
                                     config=specification['config'])
    try:
        if specification['method']=='fd4':
            native,metadata=fd_cold(**specification['config']);op=c=None
            elapsed=time.perf_counter()-tick
            evaluator=lambda xy:interpolate_solution(native,xy[:,0],xy[:,1])
        else:
            fn=solve_mapped_legendre if specification['method']=='mapped_legendre' else solve_rfm
            op,c,metadata=fn(PROBLEM,seed=seed,**specification['config'])
            elapsed=time.perf_counter()-tick
            evaluator=lambda xy:op.fields(c,xy)
        row.update(seconds=elapsed,metadata=metadata)
        scores=score.evaluate(evaluator,batch=specification['method']!='fd4')
        worst=errors(scores)
        if not all(np.isfinite(value) for value in worst.values()):raise ArithmeticError('Non-finite score')
        row.update(status='completed',scores=scores,errors=worst,
                   passes={str(t):max(worst.values())<=t for t in TIERS})
        if save:
            artifact=ROOT/f'cavity_formal__{specification["name"]}_seed{seed}.npz'
            arrays={}
            for name,(xy,w) in score.grids.items():
                blocks=[evaluator(b) for b in np.array_split(xy,int(np.ceil(len(xy)/256)))]
                arrays.update({f'{name}_{key}':np.concatenate([b[key] for b in blocks]) for key in ('u','v','omega')})
                arrays[f'{name}_xy'],arrays[f'{name}_weights']=xy,w
            if c is not None:arrays['coefficients']=c
            np.savez_compressed(artifact,**arrays)
            row['artifact']=dict(path=artifact.name,sha256=sha(artifact))
    except Exception as exc:
        row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        row.setdefault('seconds',time.perf_counter()-tick)
    print(specification['name'],seed,row['status'],round(row['seconds'],4),row.get('errors',row.get('error')),flush=True)
    return row


def pilot(score):
    strong=[spec(f'leg{m}_{"mapped" if warp else "plain"}_enriched',m,1,p,q,warp,
                 'mapped_legendre' if warp else 'legendre')
            for m,p,q,warp in [(666,39,58,(1.3,1.3)),(780,41,61,(1.3,1.3)),
                               (861,43,64,(1.3,1.3)),(861,43,64,None)]]
    data=dict(protocol=__doc__,exploratory_seed=17,predeclared_strong_controls=strong,
              predeclared_fd_coarse=[32,48],predeclared_fd_candidates=[152,180,160,192],rows=[])
    for candidate in strong:
        data['rows'].append(run(candidate,17,score));write_json(PILOT,data)
    controls=[row for row in data['rows'] if row['status']=='completed']
    if not controls:raise RuntimeError('No deterministic control converged')
    selected={min(controls,key=lambda r:max(r['errors'].values()))['name']}
    for tier in TIERS:
        passed=[row for row in controls if max(row['errors'].values())<=tier]
        if passed:selected.add(min(passed,key=lambda r:r['seconds'])['name'])
    strong_selected=[candidate for candidate in strong if candidate['name'] in selected]
    # Compare two independently cold coarse strategies at the first target.
    fd_first=[]
    for coarse in (32,48):
        candidate=dict(name=f'fd4_n152_c{coarse}',method='fd4',config=dict(n=152,coarse=coarse))
        row=run(candidate,17,score);data['rows'].append(row);fd_first.append(row);write_json(PILOT,data)
    viable=[r for r in fd_first if r['status']=='completed']
    if not viable:raise RuntimeError('No cold FD strategy converged')
    coarse=min(viable,key=lambda r:r['seconds'])['config']['coarse']
    fd_selected=[]
    for target,fallback,tier in ((152,160,.001),(180,192,.0005)):
        row=next((r for r in fd_first if r['config']==dict(n=target,coarse=coarse)),None)
        if row is None:
            candidate=dict(name=f'fd4_n{target}_c{coarse}',method='fd4',config=dict(n=target,coarse=coarse))
            row=run(candidate,17,score);data['rows'].append(row);write_json(PILOT,data)
        if row['status']!='completed' or max(row['errors'].values())>tier:
            candidate=dict(name=f'fd4_n{fallback}_c{coarse}',method='fd4',config=dict(n=fallback,coarse=coarse))
            row=run(candidate,17,score);data['rows'].append(row);write_json(PILOT,data)
        fd_selected.append(dict(name=row['name'],method='fd4',config=row['config']))
    configs=[spec('rfm496',496,14,29,47),spec('rfm595',595,15,32,51),spec('rfm666',666,16,34,54),
             spec('matched_leg496',496,1,29,47,method='mapped_legendre'),
             spec('matched_leg666',666,1,34,54,method='mapped_legendre'),*strong_selected,*fd_selected]
    data['selected_configs']=configs;write_json(PILOT,data)
    return configs


def source_hashes():
    here=['cavity_formal.py','cavity_study.py','cavity_reference_v3.py',
          'cavity1000_ref256.npz','cavity1000_ref384.npz','cavity_pilot.json','cavity_fd_pilot.json']
    earlier=['rfm_solver.py','rfm_v2_solver.py','rfm_v2_warp_legendre.py','wall_trial.py',
             'tensor_rfm.py','operator_opt.py','rfm_v2_benchmark.py','cfd_reference.py','cfd_reference_fourth.py']
    paths=[ROOT/name for name in here]+[ROOT.parent/'rfm_ns'/name for name in earlier]
    return {str(path.resolve()):sha(path) for path in paths}


def summarize(data):
    summaries=[]
    for config in data['freeze']['configs']:
        group=[r for r in data['rows'] if r['name']==config['name']]
        valid=[r for r in group if r['status']=='completed']
        worst={key:max((r['errors'][key] for r in valid),default=None) for key in ('velocity_rel_l2','vorticity_rel_l2')}
        passed={str(t):bool(len(valid)==len(SEEDS) and max(worst.values())<=t) for t in TIERS}
        summaries.append(dict(name=config['name'],method=config['method'],config=config['config'],
            runs=len(group),completed=len(valid),worst_errors=worst,passes=passed,
            median_seconds=statistics.median(r['seconds'] for r in valid) if valid else None,
            min_seconds=min((r['seconds'] for r in valid),default=None),max_seconds=max((r['seconds'] for r in valid),default=None)))
    comparisons=[]
    ref=max(errors(data['freeze']['reference_grid_change']).values())
    for t in TIERS:
        passed=[r for r in summaries if r['passes'][str(t)]]
        fd=[r for r in passed if r['method']=='fd4']
        best=min(fd,key=lambda r:r['median_seconds']) if fd else None
        comparisons.append(dict(tier=t,reference_change=ref,reference_screen_passed=ref<=.2*t,
            fastest_tested_passing_fd=best['name'] if best else None,
            rows=[dict(name=r['name'],seconds=r['median_seconds'],
                  fd_over_method_time=best['median_seconds']/r['median_seconds'] if best else None)
                  for r in passed if r['method']!='fd4']))
    return summaries,comparisons


def formal(configs,score):
    if OUTPUT.exists() or FREEZE.exists():raise FileExistsError('Formal evidence already exists; preserve it')
    hashes=source_hashes()
    schedule=[]
    for repeat,seed in enumerate(SEEDS):
        for index in np.random.default_rng(10037+repeat).permutation(len(configs)):
            schedule.append(dict(seed=seed,index=int(index)))
    freeze=dict(case='smooth_lid_cavity',reynolds=1000,lid='16*x^2*(1-x)^2',
        seeds=SEEDS,tiers=TIERS,configs=configs,schedule=schedule,source_sha256=hashes,machine=machine_info(),
        reference_grid_change=score.grid_change,pilot_sha256=sha(PILOT),
        timing='One CPU thread; entire cold solver including initialization, failed continuation attempts, setup and native fields. Reference preparation, independent scoring and serialization excluded.',
        acceptance='All fresh seeds/repeats, both area-weighted midpoint81 and Gauss113 grids, both velocity and actual-curl vorticity. FD primary omega is the native discrete unknown. Empirical reference change must be <=20% of a tier; not a certified bound.')
    write_json(FREEZE,freeze)
    data=dict(status='running',freeze=freeze,freeze_sha256=sha(FREEZE),rows=[],summaries=[],comparisons=[])
    write_json(OUTPUT,data)
    rng=np.random.default_rng(553);lstsq(rng.normal(size=(24,12)),rng.normal(size=24),lapack_driver='gelsy')
    started=time.perf_counter()
    for job in schedule:
        data['rows'].append(run(configs[job['index']],job['seed'],score,save=job['seed']==SEEDS[0]))
        data['total_wall_seconds']=time.perf_counter()-started;write_json(OUTPUT,data)
    if any(sha(path)!=value for path,value in hashes.items()):
        data['status']='source_changed';write_json(OUTPUT,data);raise RuntimeError('A frozen source changed')
    data['summaries'],data['comparisons']=summarize(data)
    data['status']='complete';write_json(OUTPUT,data)
    print('FORMAL_COMPLETE',str(OUTPUT),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=('pilot','formal','all'),default='all')
    args=ap.parse_args()
    with one_thread():
        score=Score()
        configs=pilot(score) if args.stage in ('pilot','all') else json.loads(PILOT.read_text())['selected_configs']
        if args.stage in ('formal','all'):formal(configs,score)


if __name__=='__main__':main()
