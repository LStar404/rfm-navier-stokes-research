"""Isolated, reference-free two-level FD4 timings; all coarse work is charged.

The tested starting grids are predeclared: n32 for targets48/64, n32 or48 for
target96, and n32 or64 for target128. A coarse solve starts from zero fields at
the target Re. If that fails, its cost plus the full low-Re continuation is
charged. Only independent post-solve field scoring is outside the timer.
"""
from cfd_reference_compare import *
from cfd_reference import solve_cavity
from cfd_reference_fourth import CavityFourth

def two_level(n,Re,coarse):
    tick=time.perf_counter();stages=[];failures=[]
    try:
        fields,row=solve_cavity(coarse,Re,initial=None,_model_class=CavityFourth);stages.append(row)
    except Exception as exc:
        failures.append(dict(stage='coarse direct',wall_seconds=time.perf_counter()-tick,error=str(exc)))
        fields=None
        for rr in ([10,25,50,100] if Re==100 else [10,25,50,100,200,400]):
            fields,row=solve_cavity(coarse,rr,initial=fields,_model_class=CavityFourth);stages.append(row)
    fields,row=solve_cavity(n,Re,initial=fields,_model_class=CavityFourth);stages.append(row)
    wall=time.perf_counter()-tick
    return fields,dict(strategy=f'two_level_{coarse}',wall_seconds=wall,stages=stages,failed_attempts=failures,
        finite=all(np.isfinite(fields[key]).all() for key in ('psi','omega','u','v')))

def main():
    started=time.perf_counter();accuracy=json.loads((ROOT/'cfd_fd4_accuracy.json').read_text())
    prior=json.loads((ROOT/'cfd_fd4_formal.json').read_text())
    configs=sorted({(s['Re'],s['smallest_tested_passing_n']) for s in accuracy['selections'] if s['smallest_tested_passing_n'] is not None})
    data={Re:(points(Problem('cavity',Re),81),None) for Re,_ in configs}
    data={Re:(xy,cavity_truth(Re,xy)[0]) for Re,(xy,_) in data.items()}
    cold_grids={32:[],48:[32],64:[32],96:[32,48],128:[32,64]}
    rng=np.random.default_rng(921);aa=rng.normal(size=(24,12));lstsq(aa,rng.normal(size=24),lapack_driver='gelsy')
    out=dict(protocol=__doc__,accuracy=accuracy,predeclared_coarse_grids=cold_grids,pilots=[],rows=[],summaries=[],selected_by_tier=[],
        prior_direct_formal_source='cfd_fd4_formal.json',prior_direct_formal_sha256=signature(ROOT/'cfd_fd4_formal.json'))
    def run(Re,n,strategy):
        if strategy=='direct':fields,row=compute(n,Re,'direct')
        else:fields,row=two_level(n,Re,int(strategy.rsplit('_',1)[-1]))
        xy,truth=data[Re];row.update(field_errors(interpolate_solution(fields,xy[:,0],xy[:,1]),truth))
        target=min(s['tier'] for s in accuracy['selections'] if s['Re']==Re and s['smallest_tested_passing_n']==n)
        row.update(Re=Re,n=n,selection_target=target,
            passed=bool(row['finite'] and all(s['success'] for s in row['stages']) and max(row['velocity_rel_l2'],row['vorticity_rel_l2'])<=target))
        return row
    repeat_variants=[]
    for Re,n in configs:
        old=[r for r in prior['rows'] if r['Re']==Re and r['n']==n]
        if old:
            for r in old:out['rows'].append(dict(**r,reused_prior_isolated_direct=True))
            direct_time=statistics.median(r['wall_seconds'] for r in old)
        else:
            row=run(Re,n,'direct');out['pilots'].append(row);direct_time=row['wall_seconds'] if row['passed'] else float('inf')
            if row['passed']:repeat_variants.append((Re,n,'direct'))
        candidates=[]
        for coarse in cold_grids[n]:
            row=run(Re,n,f'two_level_{coarse}');out['pilots'].append(row)
            print('PILOT_TWO',Re,n,coarse,row['wall_seconds'],row['passed'],flush=True)
            if row['passed']:candidates.append(row)
        if candidates:
            best=min(r['wall_seconds'] for r in candidates)
            # Repeat every coarse-start variant within 20% of the pilot best;
            # report the rest as pilots, not as three-repeat evidence.
            for row in candidates:
                if row['wall_seconds']<=1.2*best and row['wall_seconds']<direct_time:
                    repeat_variants.append((Re,n,row['strategy']))
        save(ROOT/'cfd_fd4_multilevel.json',out)
    for repeat in range(3):
        for ii in np.random.default_rng(819+repeat).permutation(len(repeat_variants)):
            Re,n,strategy=repeat_variants[ii];row=run(Re,n,strategy);row['repeat']=repeat;out['rows'].append(row)
            print('FORMAL_TWO',Re,n,strategy,repeat,row['wall_seconds'],row['passed'],flush=True)
            save(ROOT/'cfd_fd4_multilevel.json',out)
    for Re,n in configs:
        strategies=sorted({r['strategy'] for r in out['rows'] if r['Re']==Re and r['n']==n})
        for strategy in strategies:
            group=[r for r in out['rows'] if r['Re']==Re and r['n']==n and r['strategy']==strategy]
            passed=all(r.get('passed',r['finite'] and all(s['success'] for s in r['stages'])) for r in group)
            out['summaries'].append(dict(Re=Re,n=n,strategy=strategy,repeats=len(group),passed=passed,
                median_wall_seconds=statistics.median(r['wall_seconds'] for r in group),
                min_wall_seconds=min(r['wall_seconds'] for r in group),max_wall_seconds=max(r['wall_seconds'] for r in group),
                worst_velocity_rel_l2=max(r['velocity_rel_l2'] for r in group),worst_vorticity_rel_l2=max(r['vorticity_rel_l2'] for r in group)))
    for selection in accuracy['selections']:
        eligible=[r for r in out['summaries'] if r['Re']==selection['Re'] and r['n']==selection['smallest_tested_passing_n']
                  and r['passed'] and r['repeats']==3 and max(r['worst_velocity_rel_l2'],r['worst_vorticity_rel_l2'])<=selection['tier']]
        if eligible:out['selected_by_tier'].append(dict(**selection,**{'best':min(eligible,key=lambda r:r['median_wall_seconds'])}))
    out['total_comparison_wall_seconds']=time.perf_counter()-started
    out['source_sha256']={name:signature(ROOT/name) for name in ['cfd_reference.py','cfd_reference_fourth.py','cfd_reference_compare.py','cfd_reference_multilevel.py']}
    save(ROOT/'cfd_fd4_multilevel.json',out);print('DONE',flush=True)

if __name__=='__main__':main()
