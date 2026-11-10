"""Independent conservation-of-error-energy and new-trajectory checks."""
from review_common import *
import csv

if __name__=='__main__':
    profile=read(HERE/'profiles_summary.json'); bridge=read(HERE/'bridge_summary.json')
    with (HERE/'physical_errors_all_times.csv').open(encoding='utf-8',newline='') as f:
        physical=list(csv.DictReader(f))
    indexed={(r['name'],r['arm'],r['reference'],float(r['time'])):r for r in physical if r['grid']=='layer144'}
    diffs=[]
    for r in profile['energy_summary']:
        expected=float(indexed[r['name'],r['arm'],r['reference'],r['time']][r['field']])
        diffs.append(abs(r['relative_error']-expected))
    assert max(diffs)<1e-12, max(diffs)
    for row in bridge['new_runs']:
        assert sha(HERE/row['artifact'])==row['sha256']
    fresh=[r for r in bridge['rows'] if r['arm']=='switch12']
    assert len(fresh)==4
    assert all(r['physical_pass'] and r['uniform_guard_pass'] and r['wake']['sustained_alternating_probe_signs']
               and r['pre_switch_state_max_abs_difference']==0 for r in fresh)
    assert sorted(r['nonlinear_evaluations'] for r in fresh)==[8255,8255,9983,9983]
    result=dict(status='passed',parseval_vs_original_layer144_score_max_abs_difference=max(diffs),
        number_of_parseval_checks=len(diffs),new_trajectories=4,new_trajectories_sha_verified=True,
        pre_switch_uniform_states_exact=True,physical_and_uniform_guard_pass=True,
        sustained_alternating_wake=True,nonlinear_work_counts_verified=True,
        timing_claim=False,source_sha256=sha(Path(__file__)))
    write(HERE/'verification.json',result)
    print(json.dumps(result,indent=2))
