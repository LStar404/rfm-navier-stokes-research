"""Frozen accuracy qualification of one SBDF2 bridge, with independent references."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path

import cpu_sbdf2_switch as switched
from galerkin_cuda_variable import solve_cuda_switch
import early_reference_continuation as early
import r128_operator_qualification as original
import joint_hardware_pow2 as old
from fd_pow2_long import Pow2EquilibratedBufferedFD


base, case, np = old.base, old.case, old.np
HERE = base.HERE
TEMPLATE = HERE/'protocol_switch_qualification.json'
OUT = HERE/'switch_qualification_output'
FROZEN, RUNS, SUMMARY = (OUT/name for name in ('protocol.json','runs.json','qualification.json'))
MODELS = original.MODELS
METRICS = old.METRICS


def fingerprint(path):
    return path.relative_to(base.WORK).as_posix(), base.digest(path)


def merge_hashes(target, entries):
    for name, digest in entries.items():
        if name in target and target[name]!=digest:
            raise RuntimeError('Conflicting frozen source fingerprint: '+name)
        old.sha_matches(base.WORK/name,digest)
        target[name]=digest


def freeze():
    p = base.read(TEMPLATE)
    prior = base.read(original.FROZEN)
    ec, er, ea = base.read(early.FROZEN), base.read(early.RUNS), base.read(early.REFERENCE)
    lr = base.read(original.chain.JOURNAL)
    la = base.read(original.chain.OUT/lr['reference_audit'])
    physical = p['early_checkpoints']+p['transition_physical_checkpoints']+p['late_checkpoints']
    snapshots = sorted(physical+p['bridge_only_checkpoints'])
    if (len(physical)!=19 or len(snapshots)!=21 or tuple(snapshots)!=switched.SNAPSHOTS or
            (p['startup_rate'],p['late_rate'],p['switch_time'],p['horizon'])!=(128,112,8,72) or
            p['arms']!=list(MODELS) or p['rfm_seeds']!=prior['template']['rfm_seeds'] or
            p['deterministic_seed']!=17 or p['physical_gate']!=.005 or p['implementation_gate']!=1e-5 or
            (p['probe_spacing'],p['graph_steps'],p['cpu_workers'])!=(.125,24,3) or
            p['references']!=['angular_fine','time_fine'] or p['reference_budget_fraction']!=.25 or
            physical[:12]!=ec['protocol']['checkpoints'] or
            physical[12:]!=lr['frozen']['protocol']['checkpoints'] or
            physical[12:]!=prior['template']['late_checkpoints'] or
            p['work_counts']!=dict(time_steps=8192,nonlinear_evaluations=8191,marching_fft_batches=16382,
                                  cuda_priming_fft_batches=2,bridge_extra_fft_batches=0)):
        raise ValueError('Unexpected switch qualification schedule or acceptance criteria')
    budget = p['physical_gate']*p['reference_budget_fraction']
    if (er['status']!='complete' or not er['bitwise_t8_pass'] or not er['reference_pass'] or
            ea['status']!='complete' or not ea['passes'] or max(ea['worst'].values())>budget or
            lr['status']!='complete' or la['composed_difference']>budget or ea['budget']!=budget or
            er['protocol_sha256']!=base.digest(early.FROZEN) or ea['protocol_sha256']!=er['protocol_sha256'] or
            ea['arm_hashes']!={r['artifact']:r['sha256'] for r in er['arms']} or len(er['arms'])!=4):
        raise RuntimeError('Independent early/late reference qualification is incomplete')
    old.sha_matches(early.REFERENCE,er['reference_audit_sha256'])
    cp=HERE/'cpu_sbdf2_switch_pilot_output'
    gp=HERE/'galerkin_cuda_variable_output'
    cpu_protocol=base.read(cp/'protocol.json')
    cpu_fixed=base.read(cp/'fixed_audit.json')
    cpu_decimal=base.read(cp/'fixed_decimal_audit.json')
    gpu_fixed=base.read(gp/'fixed.json')
    pairing=base.read(gp/'comparison.json')
    if (cpu_fixed['status']!='passed' or cpu_fixed['protocol_sha256']!=base.digest(cp/'protocol.json') or
            cpu_decimal['status']!='complete' or cpu_decimal['digits']!=65 or
            cpu_decimal['protocol_sha256']!=cpu_fixed['protocol_sha256'] or
            not gpu_fixed['passed'] or not pairing['passed'] or not pairing['paired_checked'] or
            pairing['implementation_gate']!=p['implementation_gate'] or pairing['physical_gate']!=p['physical_gate']):
        raise RuntimeError('Required fixed-equation or paired pilot evidence is incomplete')
    deps = (TEMPLATE,Path(__file__),HERE/'cpu_sbdf2_switch.py',HERE/'galerkin_cuda_variable.py',
            HERE/'galerkin_cuda_variable.cu',early.FROZEN,early.RUNS,early.REFERENCE,
            original.FROZEN,original.JOURNAL,original.SUMMARY,original.chain.JOURNAL,
            original.chain.OUT/lr['reference_audit'])
    sources = {}
    merge_hashes(sources,prior['source_hashes'])
    merge_hashes(sources,cpu_protocol['source_hashes'])
    merge_hashes(sources,dict(fingerprint(path) for path in deps))
    for folder,names in ((cp,('protocol.json','fixed_audit.json','fixed_decimal_audit.json','fixed_screen_rejected.json',
                             'protocol_fixed_backward_floor_rejected.json','protocol_fixed_forward_floor_rejected.json','pilot.json')),
                         (gp,('fixed.json','march.json','comparison.json','pilot_score.json'))):
        merge_hashes(sources,dict(fingerprint(folder/name) for name in names))
    for name,digest in gpu_fixed['source_hashes'].items():
        merge_hashes(sources,{(HERE/name).relative_to(base.WORK).as_posix():digest})
    old.sha_matches(HERE/'cpu_sbdf2_switch_decimal.py',cpu_decimal['source_sha256'])
    merge_hashes(sources,dict(fingerprint(HERE/name) for name in ('cpu_sbdf2_switch_pilot.py','cpu_sbdf2_switch_decimal.py')))
    merge_hashes(sources,ec['source_hashes'])
    for key in ('original_protocol','original_runs','original_audit'):
        r=ec[key]; merge_hashes(sources,{r['source']:r['sha256']})
    for arm in ec['arms']:
        for key in ('old_arm','old_history'):
            r=arm[key]; merge_hashes(sources,{r['source']:r['sha256']})
    refs = {}
    for row in er['arms']:
        if not row['bitwise_t8_pass']:
            raise RuntimeError('A continued reference did not reproduce its original T8 history')
        path=early.OUT/row['artifact']
        old.sha_matches(path,row['sha256'])
        refs[row['spec']['name']]=dict(path=path.relative_to(base.WORK).as_posix(),sha256=row['sha256'])
        r=row['restart']; path=early.OUT/r['artifact']
        merge_hashes(sources,{path.relative_to(base.WORK).as_posix():r['sha256']})
    for group in ('source_hashes','files','reference_hashes'):
        merge_hashes(sources,prior['reference'][group])
    for path,digest in sources.items():
        old.sha_matches(base.WORK/path,digest)
    previous, prior_runs = base.read(original.SUMMARY),base.read(original.JOURNAL)
    if (previous['status']!='complete' or not previous['passes'] or prior_runs['status']!='complete' or
            not prior_runs['passes'] or previous['runs_sha256']!=base.digest(original.JOURNAL) or
            previous['protocol_sha256']!=base.digest(original.FROZEN) or
            prior_runs['protocol_sha256']!=base.digest(original.FROZEN)):
        raise RuntimeError('Previous rate128 control qualification is incomplete or changed')
    controls=[]
    for row in previous['rows']:
        if row['job']['arm']=='cpu_recurrence':
            path=original.OUT/row['artifact']
            old.sha_matches(path,row['sha256'])
            controls.append(dict(job=row['job'],path=path.relative_to(base.WORK).as_posix(),sha256=row['sha256']))
    expected_controls={(name,seed) for name in p['methods'] for seed in
                       (p['rfm_seeds'] if name.startswith('rfm') else [17])}
    if len(controls)!=6 or {(r['job']['name'],r['job']['seed']) for r in controls}!=expected_controls:
        raise ValueError('Need all six previously frozen rate128 CPU controls')
    jobs=[]
    for name in p['methods']:
        for seed in (p['rfm_seeds'] if name.startswith('rfm') else [p['deterministic_seed']]):
            jobs.extend(dict(name=name,seed=seed,arm=arm,role='candidate') for arm in p['arms'])
            jobs.append(dict(name=name,seed=seed,arm='cpu_recurrence',role='anchor'))
    result=dict(status='frozen',template=p,physical_times=physical,checkpoints=snapshots,jobs=jobs,
                solve_protocol=prior['protocol'],specs=prior['specs'],source_hashes=sources,
                early_references=refs,late_reference=prior['reference'],late_controls=controls)
    if FROZEN.exists():
        if base.read(FROZEN)!=result:
            raise RuntimeError('Frozen switch qualification inputs changed')
    else:
        base.save(FROZEN,result)
    return result


def stem(job):
    return '_'.join(str(job[k]) for k in ('role','name','seed','arm'))


def validate(sol, frozen, job):
    p=frozen['template']
    anchor=job['role']=='anchor'
    times=frozen['physical_times'][:12] if anchor else frozen['checkpoints']
    horizon=16 if anchor else 72
    steps=2048 if anchor else 8192
    probes=np.arange(0.,horizon+.001,.125)
    if (sol.row['steps']!=steps or sol.row['horizon']!=horizon or not np.array_equal(sol.times,times) or
            not np.array_equal(sol.probe_times,probes) or
            sol.states.shape!=(len(times),*sol.model.rhs_shape) or
            sol.transverse.shape!=(len(probes),3) or sol.vorticity.shape!=(len(probes),3) or
            not all(np.isfinite(getattr(sol,k)).all() for k in ('states','transverse','vorticity'))):
        raise ArithmeticError('Switch trajectory differs from its frozen shape, times or finite-state requirements')
    if not anchor:
        if (sol.row['nonlinear_evaluations']!=8191 or sol.row['marching_fft_batches']!=16382 or
                sol.row['startup_rate']!=128 or sol.row['late_rate']!=112 or sol.row['switch_step']!=1024 or
                sol.row['nonlinear_by_phase']!=dict(startup=1024,late=7167)):
            raise ArithmeticError('Measured step schedule or nonlinear work differs from qualification')
        if job['arm']=='cuda_graph' and any(sol.row[k]!=v for k,v in dict(
                device_final_step=8192,graph_steps=24,priming_fft_batches=2,bridge_extra_fft_batches=0).items()):
            raise ArithmeticError('CUDA graph accounting differs from its frozen schedule')


def run(job, frozen):
    path=OUT/(stem(job)+'.npz'); record=path.with_suffix('.json')
    if record.exists():
        row=base.read(record)
        if row['job']!=job:
            raise RuntimeError('Existing run belongs to another switch job')
        if row['status']=='complete':
            old.sha_matches(path,row['sha256'])
        return row
    if path.exists():
        raise FileExistsError('Inspect uncommitted switch trajectory '+path.name)
    _,cfg=base.config(frozen['specs'][job['name']],frozen['solve_protocol'],job['seed'])
    row=dict(job=job)
    print('START',job,flush=True)
    try:
        if job['role']=='anchor':
            sol=case.solve(switched.RecurrenceGalerkin,steps=2048,horizon=16.,
                           snapshots=frozen['physical_times'][:12],**cfg)
        else:
            kwargs=dict(startup_rate=128,late_rate=112,switch_time=8.,horizon=72.,
                        snapshots=frozen['checkpoints'],probe_spacing=.125,**cfg)
            if job['arm']=='cuda_graph':
                sol=solve_cuda_switch(graph_steps=24,**kwargs)
            else:
                sol=switched.solve_cpu_switch(MODELS[job['arm']],**kwargs)
        validate(sol,frozen,job)
        temporary=path.with_suffix('.tmp.npz')
        case.store_solution(temporary,sol,cfg)
        temporary.replace(path)
        row.update(status='complete',artifact=path.name,sha256=base.digest(path),metadata=sol.row)
    except (ArithmeticError,RuntimeError,ValueError,OSError,np.linalg.LinAlgError) as exc:
        row.update(status='failed',failure=str(exc))
    base.save(record,row)
    print('DONE',job,row['status'],flush=True)
    return row


def select(sol, times):
    indices=[]
    for t in times:
        matches=np.flatnonzero(sol.times==t)
        if len(matches)!=1:
            raise ValueError('No unique exact stored snapshot for '+str(t))
        indices.append(int(matches[0]))
    return case.Solution(sol.model,sol.states[indices],np.asarray(times),sol.probe_times,
                         sol.transverse,sol.vorticity,sol.row)


def load(row):
    path=OUT/row['artifact']
    old.sha_matches(path,row['sha256'])
    return case.load_solution(path,MODELS[row['job']['arm']])


def score_all(frozen, rows):
    grids=base.grid_set(); p=frozen['template']; times=frozen['physical_times']
    early_times,late_times=times[:12],times[12:]
    late=base.read(original.chain.JOURNAL)
    refs=original.chain.references(late['frozen']['protocol'],late['rows'])
    late_by_name={s['name']:sol for s,sol in zip(late['frozen']['protocol']['reference_path'],refs,strict=True)}
    truth={}
    for ref in p['references']:
        spec=frozen['early_references'][ref]
        path=base.WORK/spec['path']; old.sha_matches(path,spec['sha256'])
        e=case.load_solution(path,Pow2EquilibratedBufferedFD)
        if not np.array_equal(e.times,early_times):
            raise ValueError('Early reference times differ')
        ef=case.reference_fields(e,grids); lf=case.reference_fields(late_by_name[ref],grids)
        truth[ref]={grid:ef[grid]+lf[grid] for grid in grids}
    raw={stem(r['job']):r for r in rows}
    if len(raw)!=24 or len(rows)!=24 or set(raw)!={stem(j) for j in frozen['jobs']}:
        raise ValueError('Expected eighteen switched runs and six fixed-rate anchors')
    prior={(r['job']['name'],r['job']['seed']):r for r in frozen['late_controls']}
    results=[]; anchors=[]
    physical_keys={(g,t) for g in grids for t in times}
    implementation_keys={(g,t) for g in grids for t in frozen['checkpoints']}
    for name in p['methods']:
        for seed in (p['rfm_seeds'] if name.startswith('rfm') else [17]):
            ajob=dict(name=name,seed=seed,arm='cpu_recurrence',role='anchor')
            arow=raw[stem(ajob)]
            cjob=dict(name=name,seed=seed,arm='cpu_recurrence',role='candidate')
            crow=raw[stem(cjob)]
            if arow['status']!='complete' or crow['status']!='complete':
                results.extend(dict(job=dict(name=name,seed=seed,arm=arm,role='candidate'),qualified=False,
                                    failure='Missing complete original CPU or anchor trajectory') for arm in p['arms'])
                continue
            anchor,control=load(arow),load(crow)
            prior_row=prior[name,seed]; prior_path=base.WORK/prior_row['path']
            old.sha_matches(prior_path,prior_row['sha256'])
            late_control=select(case.load_solution(prior_path,switched.RecurrenceGalerkin),late_times)
            if (anchor.states.shape[1:]!=late_control.states.shape[1:] or
                    not switched.equal_spatial_models((anchor.model,late_control.model))):
                raise ArithmeticError('The early and late rate128 controls use different spatial coordinates')
            baseline=case.Solution(anchor.model,np.concatenate((anchor.states,late_control.states)),np.asarray(times),
                                   late_control.probe_times,late_control.transverse,late_control.vorticity,late_control.row)
            if not np.array_equal(select(control,p['early_checkpoints']).states,anchor.states[:6]):
                results.extend(dict(job=dict(name=name,seed=seed,arm=arm,role='candidate'),qualified=False,
                    failure='The switched CPU solution changed its pre-switch states') for arm in p['arms'])
                continue
            baseline_fields=case.reference_fields(baseline,grids)
            baseline_scores={ref:case.score(baseline,fields,grids) for ref,fields in truth.items()}
            for scores in baseline_scores.values():
                old.score_rows(scores,physical_keys)
            anchors.append(dict(name=name,seed=seed,artifact=arow['artifact'],sha256=arow['sha256'],
                                late_source=prior_row,physical_scores=baseline_scores))
            implementation_fields=case.reference_fields(control,grids)
            for arm in p['arms']:
                job=dict(name=name,seed=seed,arm=arm,role='candidate'); row=raw[stem(job)]
                if row['status']!='complete':
                    results.append(dict(job=job,qualified=False,failure=row['failure'])); continue
                sol=load(row); validate(sol,frozen,job)
                impl=case.score(sol,implementation_fields,grids)
                impl_max=old.score_rows(impl,implementation_keys)
                probe={k:float(np.linalg.norm(getattr(sol,k)-getattr(control,k))/np.linalg.norm(getattr(control,k)))
                       for k in ('transverse','vorticity')}
                physical_sol=select(sol,times)
                scored={ref:case.score(physical_sol,fields,grids) for ref,fields in truth.items()}
                maximum={ref:old.score_rows(scores,physical_keys) for ref,scores in scored.items()}
                delta=case.score(physical_sol,baseline_fields,grids)
                old.score_rows(delta,physical_keys)
                guard={}
                for ref in p['references']:
                    bs={(r['grid'],r['time']):r for r in baseline_scores[ref]}
                    guard[ref]=[dict(grid=d['grid'],time=d['time'],**{
                        k:d[k]*(1+bs[d['grid'],d['time']][k])+bs[d['grid'],d['time']][k] for k in METRICS}) for d in delta]
                guard_max={ref:old.score_rows(scores,physical_keys) for ref,scores in guard.items()}
                wake=base.wake(sol)
                imp=all(math.isfinite(v) and v<=p['implementation_gate'] for v in (*impl_max.values(),*probe.values()))
                phy=all(max(m.values())<=p['physical_gate'] for m in maximum.values())
                guarded=all(max(m.values())<=p['physical_gate'] for m in guard_max.values())
                passed=bool(imp and phy and guarded and wake['sustained_alternating_probe_signs'])
                results.append(dict(job=job,artifact=row['artifact'],sha256=row['sha256'],implementation_scores=impl,
                    implementation_worst=impl_max,full_probe_error=probe,physical_scores=scored,physical_worst=maximum,
                    baseline_difference=delta,guard_scores=guard,guard_worst=guard_max,wake=wake,
                    implementation_pass=imp,physical_pass=phy,guard_pass=guarded,qualified=passed))
                print('SCORE',name,seed,arm,passed,maximum,flush=True)
    return results,anchors


def execute(frozen):
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    sha=base.digest(FROZEN)
    if RUNS.exists() and base.read(RUNS)['protocol_sha256']!=sha:
        raise RuntimeError('Switch journal has another frozen protocol')
    journal=dict(status='running',protocol_sha256=sha,rows=[])
    base.save(RUNS,journal)
    with ProcessPoolExecutor(max_workers=3) as pool:
        cpu=[pool.submit(run,j,frozen) for j in frozen['jobs'] if j['arm']!='cuda_graph']
        for job in (j for j in frozen['jobs'] if j['arm']=='cuda_graph'):
            journal['rows'].append(run(job,frozen)); base.save(RUNS,journal)
        for future in as_completed(cpu):
            journal['rows'].append(future.result()); base.save(RUNS,journal)
    if freeze()!=frozen:
        raise RuntimeError('Frozen sources changed while running switch qualification')
    scored,anchors=score_all(frozen,journal['rows'])
    passed=len(scored)==18 and all(r['qualified'] for r in scored)
    journal.update(status='complete',passes=passed); base.save(RUNS,journal)
    base.save(SUMMARY,dict(status='complete',passes=passed,protocol_sha256=sha,runs_sha256=base.digest(RUNS),
                           rows=scored,anchors=anchors,isolated_speed_claim_supported=False,scope=frozen['template']['scope']))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--execute',action='store_true')
    args=cli.parse_args()
    frozen=freeze()
    print('FROZEN',base.digest(FROZEN),len(frozen['jobs']),flush=True)
    if args.execute:
        execute(frozen)
