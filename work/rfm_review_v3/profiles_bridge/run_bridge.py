"""CPU T72 sensitivity at switch=12, preserving frozen rates and physical checkpoints.

Run: python work/rfm_review_v3/profiles_bridge/run_bridge.py
No elapsed-time speed inference: the four runs execute concurrently.
"""
from review_common import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import platform

def run(name):
    j=job(name,'combined'); args=j['args']; cfg=j['config']
    output=HERE/(name+'_switch12.npz')
    if output.exists():
        row=read(output.with_suffix('.json'))
        if sha(output)!=row['sha256']: raise ValueError('New trajectory changed')
        return row
    sol=switch.solve_cpu_switch(switch.RecurrenceGalerkin,
        startup_rate=args['startup_rate'],late_rate=args['late_rate'],switch_time=12.,
        horizon=72.,snapshots=TIMES,probe_spacing=.125,**cfg)
    assert sol.row['switch_step']==12*args['startup_rate']
    assert sol.row['nonlinear_evaluations']==12*args['startup_rate']+60*args['late_rate']-1
    case.store_solution(output,sol,cfg)
    row=dict(name=name,status='complete',config=cfg,artifact=output.name,sha256=sha(output),
             metadata=sol.row,wake=base.wake(sol),timing_scope='Concurrent CPU diagnostic, not isolated timing')
    write(output.with_suffix('.json'),row)
    print('COMPLETE',name,sol.row['nonlinear_evaluations'],flush=True)
    return row

def analyze(rows):
    refs=references(); grids=base.grid_set()
    truth={ref:case.reference_fields(sol,grids) for ref,sol in refs.items()}
    records=[]
    for name in NAMES:
        uniform=control(name,'uniform'); ufields=case.reference_fields(uniform,grids)
        uscores={ref:case.score(uniform,fields,grids) for ref,fields in truth.items()}
        for arm,sol in [('uniform',uniform),('switch8',control(name,'combined')),
                        ('switch12',case.load_solution(HERE/(name+'_switch12.npz'),switch.RecurrenceGalerkin))]:
            scores={ref:case.score(sol,fields,grids) for ref,fields in truth.items()}
            delta=case.score(sol,ufields,grids)
            guards={}
            for ref in refs:
                indexed={(r['grid'],r['time']):r for r in uscores[ref]}
                guards[ref]=[dict(grid=d['grid'],time=d['time'],**{
                    k:d[k]*(1+indexed[d['grid'],d['time']][k])+indexed[d['grid'],d['time']][k]
                    for k in METRICS}) for d in delta]
            worst=maxima(scores); guard=maxima(guards)
            rate=job(name,'combined')['args']['startup_rate']; late=job(name,'combined')['args']['late_rate']
            ns=72*rate if arm=='uniform' else (8 if arm=='switch8' else 12)*rate+(64 if arm=='switch8' else 60)*late
            wake=base.wake(sol)
            record=dict(name=name,arm=arm,nonlinear_evaluations=ns-1,marching_fft_batches=2*(ns-1),
                nonlinear_reduction_from_uniform=1-(ns-1)/(72*rate-1),retained_rank=sol.row.get('rank'),
                physical_scores=scores,worst=worst,same_space_uniform_delta=delta,
                guard_scores=guards,guard_worst=guard,wake=wake,
                physical_pass=all(x['value']<=.005 for x in worst.values()),
                uniform_guard_pass=all(x['value']<=.005 for x in guard.values()))
            if arm=='switch12':
                # All t<=12 must reproduce the fixed-rate CPU control: no switch has happened yet.
                early=sol.times<=12
                record['pre_switch_state_max_abs_difference']=float(np.max(abs(sol.states[early]-uniform.states[early])))
            records.append(record)
            print('SCORE',name,arm,{k:round(v['value']*100,6) for k,v in worst.items()},record['uniform_guard_pass'],flush=True)
    result=dict(status='complete',scope='CPU accuracy/work-count sensitivity, not GPU validation or isolated timing',
        physical_gate=.005,physical_times=TIMES,references=['angular_fine','time_fine'],grids=list(grids),
        rows=records,new_runs=rows,order_claim='No time-refinement series was run; no order-reduction claim is made.',
        pulse_regularness='The sin^4 transverse pulse is C3 at t=12 with a possible fourth-derivative jump. This is a pulse-junction sensitivity test, not removal of startup transient order reduction.')
    write(HERE/'bridge_summary.json',result)

if __name__=='__main__':
    protocol=dict(status='frozen_before_execution',names=NAMES,switch_time=12,physical_times=TIMES,
        startup_late_rates={name:[job(name,'combined')['args'][key] for key in ('startup_rate','late_rate')] for name in NAMES},
        workers=4,blas_threads=1,python=platform.python_version(),numpy=np.__version__,gpu=False,
        sources={str(p.relative_to(WORK)):sha(p) for p in [Path(__file__),HERE/'review_common.py',WORK/'rfm_v7/cpu_sbdf2_switch.py',WORK/'rfm_v7/basis_recurrence_model.py']},
        frozen_protocols={str(p.relative_to(WORK)):sha(p) for p in [WORK/'rfm_v7/switch_qualification_output/protocol.json',WORK/'rfm_v8/rank_fair_cache_qualification_output/protocol.json']},
        reference_inputs=FROZEN['early_references']|{'late':FROZEN['late_reference']['reference_hashes']},
        control_inputs={name:{arm:job(name,arm)['control'] for arm in ('uniform','combined')} for name in NAMES},
        note='Generic CPU phase clock audited: switch_time=12 explicitly; default t8 observation list overridden. No old source modified.')
    protocol_path=HERE/'bridge_protocol.json'
    if protocol_path.exists():
        assert read(protocol_path)==protocol
    else: write(protocol_path,protocol)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[future.result() for future in as_completed([pool.submit(run,name) for name in NAMES])]
    analyze(rows)
