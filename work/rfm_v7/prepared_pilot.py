"""Frozen short marches for preparation ablations; never a performance ranking."""
import gc
import json
from pathlib import Path

import cpu_formal as base
from galerkin_cuda_native import case, np
from galerkin_cuda_prepared import solve_cuda_prepared
from galerkin_cuda_variable import solve_cuda_switch

HERE = Path(__file__).resolve().parent
OUT = HERE/'prepared_pilot_output'
INPUTS = ('prepared_pilot.py','galerkin_cuda_prepared.py','cuda_prepared_solver_generation.json',
          'galerkin_cuda_module_bundle.py','galerkin_cuda_module_bundle_audit.json',
          'galerkin_phase_shared.py','galerkin_phase_shared_audit.json',
          'galerkin_cuda_variable.py','galerkin_cuda_variable.cu','galerkin_cuda_native.py',
          'galerkin_cuda_native.cu','cpu_sbdf2_switch.py','basis_recurrence_model.py',
          'cuda_environment_native.py')


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve every pilot attempt')
    OUT.mkdir()
    sources={name:base.digest(HERE/name) for name in INPUTS}
    for name in ('galerkin_cuda_module_bundle_audit.json','galerkin_phase_shared_audit.json'):
        audit=base.read(HERE/name)
        if audit['status'] != 'passed':
            raise ValueError('Fixed preparation audit has not passed')
    jobs=[]
    for scale,dirname,names,rates in (
        ('mid','switch_qualification_output',('rfm37_k72','leg37_k72'),(128,112)),
        ('high','multiscale_switch_qualification_output',('rfm41_k80','leg39_k80'),(152,136))):
        path=HERE/dirname/'protocol.json'
        sources[f'{dirname}/protocol.json']=base.digest(path)
        protocol=base.read(path)
        for name in names:
            seed=7331 if name.startswith('rfm') else 17
            _,config=base.config(protocol['specs'][name],protocol['solve_protocol'],seed)
            for late in ((rates[1],rates[0]) if scale=='mid' else (rates[1],)):
                snapshots=[.5,1.,2.,4.,6.,8.,8.+1/late,8.+2/late,8.125,8.5,9.]
                jobs.append(dict(scale=scale,name=name,seed=seed,config=config,startup_rate=rates[0],
                                 late_rate=late,snapshots=snapshots))
    protocol=dict(status='frozen',source_hashes=sources,jobs=jobs,horizon=9.,probe_spacing=.125,
                  preparations=['legacy','module_only','shared_only','combined'],
                  gate='All saved field and probe arrays, observation times and work counts must agree exactly.',
                  timing_scope='Sequential in-process diagnostic; not isolated or fresh-process timing.')
    save(OUT/'protocol.json',protocol)
    rows=[]
    for job in jobs:
        reference=None
        for preparation in protocol['preparations']:
            label=f"{job['name']}_{job['seed']}_r{job['startup_rate']}_{job['late_rate']}_{preparation}"
            kwargs=dict(startup_rate=job['startup_rate'],late_rate=job['late_rate'],horizon=9.,
                        snapshots=job['snapshots'],probe_spacing=.125,graph_steps=24,**job['config'])
            result=solve_cuda_switch(**kwargs) if preparation=='legacy' else solve_cuda_prepared(preparation=preparation,**kwargs)
            arrays={key:getattr(result,key).copy() for key in ('states','times','probe_times','transverse','vorticity')}
            if reference is None:
                reference=(arrays,result.row.copy())
            comparisons={key:dict(bitwise=value.tobytes()==reference[0][key].tobytes(),
                                   max_abs=float(np.max(abs(value-reference[0][key])))) for key,value in arrays.items()}
            count_keys=('steps','nonlinear_evaluations','marching_fft_batches','priming_fft_batches',
                        'bridge_extra_fft_batches','device_final_step')
            work={key:result.row[key] for key in count_keys}
            passed=all(value['bitwise'] for value in comparisons.values()) and all(
                value==reference[1][key] for key,value in work.items())
            artifact=OUT/(label+'.npz')
            with artifact.open('xb') as handle:
                case.store_solution(handle,result,job['config'])
            row=dict(name=job['name'],seed=job['seed'],startup_rate=job['startup_rate'],late_rate=job['late_rate'],
                     preparation=preparation,artifact=artifact.name,sha256=base.digest(artifact),
                     comparisons=comparisons,work=work,metadata=result.row,passed=passed)
            save(OUT/(label+'.json'),row)
            rows.append(row)
            save(OUT/'runs.json',dict(status='running',rows=rows))
            print(len(rows),label,passed,flush=True)
            del result
            gc.collect()
    if sources!={name:base.digest(HERE/name) for name in sources}:
        raise ValueError('Pilot sources changed during execution')
    save(OUT/'runs.json',dict(status='complete',rows=rows))
    summary=dict(status='passed' if all(row['passed'] for row in rows) else 'failed',rows=len(rows),
                 protocol_sha256=base.digest(OUT/'protocol.json'),runs_sha256=base.digest(OUT/'runs.json'),
                 passed=sum(row['passed'] for row in rows),scope=protocol['timing_scope'])
    save(OUT/'summary.json',summary)
    if summary['status']!='passed':
        raise ArithmeticError('Prepared short trajectory differs from the frozen march')


if __name__=='__main__':
    main()
