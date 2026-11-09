"""Parallel accuracy qualification. Concurrent wall times are not benchmarks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'parallel_refinement'


def specifications():
    from layer_candidates import SPECS
    from full_formal import SEEDS

    def fd(name,n,k,rate,kind='fd'):
        return dict(name=name,kind=kind,rate=rate,config=dict(
            n=n,k=k,stretch=1.,grid='outer',angular='fft',linear='native'))
    jobs=[fd('fd280_ref',280,104,256,'reference'),
          fd('fd320_ref',320,104,256,'reference'),
          fd('fd320_ref_tk',320,112,320,'reference')]
    jobs += [fd(f'fd{n}_k80',n,80,rate) for n,rate in (
        (176,128),(184,128),(192,128),(200,128),(208,144))]
    for name in ('rfm37_q66_k72','rfm41_k80','leg37_q66_k72','leg39_k80'):
        rate,config=SPECS[name]
        leg=name.startswith('leg')
        for seed in ((17,) if leg else SEEDS):
            jobs.append(dict(name=f'{name}_{seed}',kind='legendre' if leg else 'rfm',
                             rate=rate,config=dict(config,seed=seed)))
    return jobs


def run(spec):
    import full_case as case
    from fd_full import FullFD
    from full_galerkin_buffered import BufferedGalerkin
    from diagnostic_wake import analyze

    name=spec['name']
    target=OUT/f'{name}.npz'
    if target.is_file():
        return dict(spec,file=target.name,reused=True)
    solver=FullFD if spec['kind'] in ('fd','reference') else BufferedGalerkin
    try:
        with case.np.errstate(over='ignore',invalid='ignore'):
            solution=case.solve(solver,steps=72*spec['rate'],**spec['config'])
        case.store_solution(target,solution,spec['config'])
        wake=analyze(solution.probe_times,solution.transverse,solution.vorticity)
        return dict(spec,file=target.name,diagnostic_seconds=solution.row['seconds'],wake=wake)
    except (ArithmeticError,ValueError,case.np.linalg.LinAlgError) as exc:
        return dict(spec,failure=str(exc))


def score(jobs):
    import full_case as case
    from fd_full import FullFD
    from full_galerkin_buffered import BufferedGalerkin
    from diagnostic_wake import analyze

    ref=case.load_solution(OUT/'fd320_ref_tk.npz',FullFD)
    grids=case.evaluation_grids()
    grids['layer144']=case.layer_scoring_grid()
    truth=case.reference_fields(ref,grids)
    records=[]
    for job in jobs:
        if 'failure' in job:
            records.append(job)
            continue
        cls=FullFD if job['kind'] in ('fd','reference') else BufferedGalerkin
        sol=case.load_solution(OUT/job['file'],cls)
        errors=case.score(sol,truth,grids)
        wake=job.get('wake') or analyze(sol.probe_times,sol.transverse,sol.vorticity)
        records.append(dict(job,worst=case.maxima(errors),scores=errors,wake=wake))
    base=case.load_solution(ROOT/'fd_full_n240_k120_rate320_outer1.npz',FullFD)
    step=case.load_solution(OUT/'fd320_ref.npz',FullFD)
    coarse=case.load_solution(OUT/'fd280_ref.npz',FullFD)
    fixed_spatial=case.maxima(case.score(coarse,case.reference_fields(step,grids),grids))
    frozen_change=case.maxima(case.score(base,truth,grids))
    return dict(reference='fd320_ref_tk.npz',frozen_to_new_change=frozen_change,
                paired_radial_change=fixed_spatial,rows=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    if not 1<=args.workers<=32:
        raise ValueError('Choose 1–32 one-thread workers')
    jobs=specifications()
    workers=min(len(jobs),args.workers)
    OUT.mkdir(exist_ok=True)
    manifest=OUT/'parallel_refinement.json'
    rows=[]
    started=time.perf_counter()
    print(f'START {len(jobs)} accuracy jobs using up to {workers} CPU processes',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running=[pool.submit(run,job) for job in jobs]
        for pending in as_completed(running):
            row=pending.result()
            rows.append(row)
            print('DONE',row['name'],row.get('diagnostic_seconds'),row.get('failure',''),flush=True)
            manifest.write_text(json.dumps(dict(status='running',workers=workers,rows=rows),indent=2),encoding='utf8')
    result=score(rows)
    result.update(status='complete',workers=workers,batch_elapsed=time.perf_counter()-started)
    manifest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print('COMPLETE',result['batch_elapsed'],flush=True)


if __name__=='__main__':
    main()
