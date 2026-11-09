"""Fair FD4 comparison on the exact RFM score grid.

--stage score only interpolates already saved fields; it performs no solves.
--stage timing must be run in an agreed isolated CPU window. It tests direct
zero-state initialization, then times three independent solves of each selected
grid. No reference solution or saved CFD state enters a timed solve. Reference
generation, strategy pilot and independent field scoring are excluded, just as
RFM development/tuning and scoring are excluded from its formal solve timings.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
import argparse,json,time,hashlib,statistics
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from cfd_reference import solve_cavity,interpolate_solution
from cfd_reference_fourth import CavityFourth
from experiments import points,cavity_truth
from rfm_solver import Problem,field_errors

ROOT=Path(__file__).resolve().parent
TIERS=[.01,.003,.001]
CANDIDATE_GRIDS=[32,48,64,96,128]
PRIMARY_TARGETS=[(100,.003),(100,.001),(400,.01),(400,.003)]
def load_fields(path):
    with np.load(path) as data:return {key:data[key] for key in ('x','y','psi','omega','u','v')}
def signature(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def candidate_path(Re,n):
    original=ROOT/f'cfd_reference4_cavity_Re{Re}_n{n}.npz'
    return original if original.exists() else ROOT/f'cfd_reference_compare_candidate_Re{Re}_n{n}.npz'

def score():
    rows=[];selections=[];references={};refinement={}
    for Re in [100,400]:
        xy=points(Problem('cavity',Re),81);truth,refname=cavity_truth(Re,xy);references[str(Re)]=refname
        for n in CANDIDATE_GRIDS:
            path=candidate_path(Re,n)
            if not path.exists():continue
            fields=load_fields(path)
            pred=interpolate_solution(fields,xy[:,0],xy[:,1]);errs=field_errors(pred,truth)
            row=dict(Re=Re,n=n,**errs);row['passes']={str(t):max(errs.values())<=t for t in TIERS};rows.append(row)
        previous_n=128 if Re==100 else 256
        prior=load_fields(ROOT/f'cfd_reference4_cavity_Re{Re}_n{previous_n}.npz')
        refinement[str(Re)]=dict(coarser_n=previous_n,reference=refname,
            **field_errors(interpolate_solution(prior,xy[:,0],xy[:,1]),truth))
        for tier in TIERS:
            passing=[r['n'] for r in rows if r['Re']==Re and max(r['velocity_rel_l2'],r['vorticity_rel_l2'])<=tier]
            selections.append(dict(Re=Re,tier=tier,smallest_tested_passing_n=min(passing) if passing else None))
    result=dict(score_grid='Exactly experiments.points(Problem(cavity,Re),81); 81x81 interior midpoints, equal weights',
        points_sha256=hashlib.sha256(xy.tobytes()).hexdigest(),points_count=len(xy),tiers=TIERS,
        predeclared_candidate_grids=CANDIDATE_GRIDS,primary_targets=PRIMARY_TARGETS,rows=rows,
        selections=selections,references=references,reference_grid_change_on_same_score_points=refinement,
        vorticity_definition='FD4 omega=-Delta_h psi, with its compatibility residual solved to tolerance; native omega field is bicubically interpolated, as in the CFD reference.',
        source_sha256={name:signature(ROOT/name) for name in ['cfd_reference.py','cfd_reference_fourth.py','cfd_reference_compare.py','experiments.py','rfm_solver.py']})
    save(ROOT/'cfd_fd4_accuracy.json',result)
    print(json.dumps(result,indent=2))
    return result

def compute(n,Re,strategy):
    """All states are generated inside this function; no data file reads."""
    started=time.perf_counter();stages=[]
    if strategy=='direct':
        fields,row=solve_cavity(n,Re,initial=None,_model_class=CavityFourth);stages.append(row)
    elif strategy=='hierarchy':
        fields=None
        for rr in ([10,25,50,100] if Re==100 else [10,25,50,100,200,400]):
            fields,row=solve_cavity(32,rr,initial=fields,_model_class=CavityFourth);stages.append(row)
        levels=[];grid=64
        while grid<n:levels.append(grid);grid*=2
        if n>32:levels.append(n)
        for grid in levels:
            fields,row=solve_cavity(grid,Re,initial=fields,_model_class=CavityFourth);stages.append(row)
        if n not in CANDIDATE_GRIDS:raise ValueError('Only predeclared comparison grids are allowed')
    else:raise ValueError(strategy)
    wall=time.perf_counter()-started
    return fields,dict(strategy=strategy,wall_seconds=wall,
        sum_inner_solve_seconds=sum(r['total_seconds'] for r in stages),stages=stages,
        finite=all(np.isfinite(fields[key]).all() for key in ('psi','omega','u','v')))

def timing():
    overall_started=time.perf_counter()
    candidate_pilots=[]
    # No data from saved fields is passed to these new candidate solves.
    for Re in [100,400]:
        for n in CANDIDATE_GRIDS:
            if candidate_path(Re,n).exists():continue
            for strategy in ['direct','hierarchy']:
                attempt=time.perf_counter()
                try:
                    fields,record=compute(n,Re,strategy)
                    if not record['finite'] or not all(s['success'] for s in record['stages']):raise RuntimeError('Nonconverged candidate')
                    record.update(Re=Re,n=n,passed=True);candidate_pilots.append(record)
                    np.savez_compressed(candidate_path(Re,n),**fields,record=json.dumps(record))
                    break
                except Exception as exc:
                    candidate_pilots.append(dict(Re=Re,n=n,strategy=strategy,passed=False,
                        wall_seconds=time.perf_counter()-attempt,error=str(exc)))
            else:raise RuntimeError(f'Both cold strategies failed at Re={Re},n={n}')
    accuracy=score()
    configs=sorted({(s['Re'],s['smallest_tested_passing_n']) for s in accuracy['selections']
                    if s['smallest_tested_passing_n'] is not None and (s['Re'],s['tier']) in PRIMARY_TARGETS})
    # Load reference and point data before any timed region.
    truth={};xy={}
    for Re,_ in configs:
        xy[Re]=points(Problem('cavity',Re),81);truth[Re]=cavity_truth(Re,xy[Re])[0]
    # A small common BLAS/LAPACK warmup; no CFD mesh/solution is reused.
    rng=np.random.default_rng(9);aa=rng.normal(size=(24,12));lstsq(aa,rng.normal(size=24),lapack_driver='gelsy')
    strategies={};pilots=[]
    for Re,n in configs:
        target=min(s['tier'] for s in accuracy['selections'] if s['Re']==Re and s['smallest_tested_passing_n']==n)
        for strategy in ['direct','hierarchy']:
            try:
                fields,row=compute(n,Re,strategy)
                row.update(field_errors(interpolate_solution(fields,xy[Re][:,0],xy[Re][:,1]),truth[Re]))
                passed=row['finite'] and all(s['success'] for s in row['stages']) and max(row['velocity_rel_l2'],row['vorticity_rel_l2'])<=target
                row.update(Re=Re,n=n,selection_target=target,passed=bool(passed));pilots.append(row)
                if passed:strategies[(Re,n)]=strategy;break
            except Exception as exc:pilots.append(dict(Re=Re,n=n,strategy=strategy,passed=False,error=str(exc)))
        if (Re,n) not in strategies:raise RuntimeError(f'No robust reference-free CFD strategy for Re={Re},n={n}')
    result=dict(protocol=__doc__,accuracy=accuracy,candidate_pilots=candidate_pilots,strategy_pilots=pilots,
        frozen_strategy=[dict(Re=k[0],n=k[1],strategy=v) for k,v in strategies.items()],rows=[],summary=[])
    save(ROOT/'cfd_fd4_formal.json',result)
    for repeat in range(3):
        for index in np.random.default_rng(701+repeat).permutation(len(configs)):
            Re,n=configs[index];fields,row=compute(n,Re,strategies[(Re,n)])
            row.update(field_errors(interpolate_solution(fields,xy[Re][:,0],xy[Re][:,1]),truth[Re]))
            row.update(Re=Re,n=n,repeat=repeat)
            row['passes']={str(t):row['finite'] and max(row['velocity_rel_l2'],row['vorticity_rel_l2'])<=t for t in TIERS}
            result['rows'].append(row);save(ROOT/'cfd_fd4_formal.json',result)
            print('FORMAL',Re,n,repeat,row['strategy'],row['wall_seconds'],row['velocity_rel_l2'],row['vorticity_rel_l2'],flush=True)
    for Re,n in configs:
        group=[r for r in result['rows'] if r['Re']==Re and r['n']==n]
        result['summary'].append(dict(Re=Re,n=n,strategy=strategies[(Re,n)],
            median_wall_seconds=statistics.median(r['wall_seconds'] for r in group),
            min_wall_seconds=min(r['wall_seconds'] for r in group),max_wall_seconds=max(r['wall_seconds'] for r in group),
            worst_velocity_rel_l2=max(r['velocity_rel_l2'] for r in group),worst_vorticity_rel_l2=max(r['vorticity_rel_l2'] for r in group)))
    result['total_comparison_wall_seconds']=time.perf_counter()-overall_started
    result['candidate_and_strategy_pilot_solve_seconds']=sum(r.get('wall_seconds',0) for r in candidate_pilots+pilots)
    result['source_sha256']={name:signature(ROOT/name) for name in ['cfd_reference.py','cfd_reference_fourth.py','cfd_reference_compare.py','experiments.py','rfm_solver.py']}
    save(ROOT/'cfd_fd4_formal.json',result);print('DONE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['score','timing'],required=True);args=parser.parse_args()
    if args.stage=='score':score()
    else:timing()
