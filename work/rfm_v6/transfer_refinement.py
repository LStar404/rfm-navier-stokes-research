"""Requalify identical v6 isolated timing configurations against the finer reference."""
import hashlib
import json
from pathlib import Path

import full_case as case
from fd_full import FullFD
from diagnostic_wake import analyze

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'parallel_refinement'
TIER=.005


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def common_config(config):
    return {k:v for k,v in config.items() if k!='seed' and not (k=='backend' and v=='fft')}


def main():
    formal=read(ROOT/'full_formal.json')
    parallel=read(OUT/'parallel_refinement.json')
    if formal['status']!='complete' or parallel['status']!='complete':
        raise RuntimeError('Need both completed datasets')
    for name,value in formal['hashes'].items():
        if hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest()!=value:
            raise RuntimeError(f'Frozen v6 input changed: {name}')
    indexed={r['name']:r for r in parallel['rows']}
    radial=max(parallel['paired_radial_change'].values())
    joint=max(indexed['fd320_ref']['worst'].values())
    screening=max(radial*(1+joint)+joint,max(parallel['frozen_to_new_change'].values()))
    screen=screening<=.25*TIER
    result=[]
    ref=case.load_solution(OUT/parallel['reference'],FullFD)
    grids=case.evaluation_grids()
    grids['layer144']=case.layer_scoring_grid()
    truth=case.reference_fields(ref,grids)
    mapping={
        'rfm37_k72':[f'rfm37_q66_k72_{seed}' for seed in formal['seeds']],
        'rfm41_k80':[f'rfm41_k80_{seed}' for seed in formal['seeds']],
        'leg37_k72':['leg37_q66_k72_17'],
    }
    for r in formal['summary']:
        if r['kind']=='fd':
            path=ROOT/f'full_state_{r["name"]}.npz'
            saved=next(a['artifact'] for a in formal['rows'] if
                       a['name']==r['name'] and 'artifact' in a)
            if saved['file']!=path.name or hashlib.sha256(path.read_bytes()).hexdigest()!=saved['sha256']:
                raise RuntimeError(f'Frozen state changed: {path.name}')
            solution=case.load_solution(path,FullFD)
            checks=[dict(worst=case.maxima(case.score(solution,truth,grids)),
                         wake=analyze(solution.probe_times,solution.transverse,solution.vorticity))]
        elif r['name'] in mapping:
            checks=[indexed[name] for name in mapping[r['name']]]
            for check in checks:
                if 'worst' not in check or (check['rate']!=r['rate'] or
                        common_config(check['config'])!=common_config(r['config'])):
                    raise RuntimeError(f'Expected identical accuracy configuration for {r["name"]}')
        else:
            continue
        worst={key:max(check['worst'][key] for check in checks) for key in checks[0]['worst']}
        raw=(max(worst.values())<=TIER and all(
            check['wake']['sustained_alternating_probe_signs'] for check in checks))
        result.append(dict(name=r['name'],kind=r['kind'],worst=worst,
                           old_isolated_seconds=r['seconds'],score_pass=raw,
                           qualified=bool(raw and screen)))
    (OUT/'transfer_refinement.json').write_text(json.dumps(dict(gate=TIER,reference_screen=screen,
        bound_radial_and_joint=radial*(1+joint)+joint,radial_change=radial,
        joint_time_angular_change=joint,old_to_new=max(parallel['frozen_to_new_change'].values()),
        rows=result),indent=2),encoding='utf8')


if __name__=='__main__':
    main()
