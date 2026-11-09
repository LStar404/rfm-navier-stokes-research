"""Frozen paired comparison of v4 and compact v5 continuation on fresh seeds."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import hashlib
import json
import statistics
import sys
import time

from cavity_v5 import ROOT, np, solve as solve_v5
from cavity_solver import solve as solve_v4
from timing_v5 import pin
sys.path.append(str(ROOT.parent/'rfm_v3'))
from cavity_study import Score, PROBLEM, errors


SEEDS = (5101,5419,5737,6047,6353)
GATES = {496:.001,595:.0005}
V5 = dict(strategy='ladder',damping='halving',inner_budget=3,predict=True)
RUNS = ('v4','v5')
V3_DIR, V4_DIR = ROOT.parent/'rfm_v3',ROOT.parent/'rfm_v4'
OUTPUT = ROOT/'cavity_paired.json'
FREEZE = ROOT/'cavity_paired_freeze.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,result):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pending.replace(path)


def run(m,seed,method,config,score):
    solver = solve_v4 if method == 'v4' else solve_v5
    tick = time.perf_counter()
    row = dict(m=m,seed=seed,method=method)
    try:
        op,c,meta = solver(PROBLEM,seed=seed,**config)
        row['seconds'] = time.perf_counter()-tick
        scores = score.evaluate(lambda xy: op.fields(c,xy))
        worst = errors(scores)
        if not np.all(np.isfinite(list(worst.values()))):
            raise ArithmeticError('Non-finite field errors')
        row.update(status='completed',worst=worst,scores=scores,metadata=meta,
                   passed=max(worst.values()) <= GATES[m],
                   jacobians=sum(it['fresh'] for s in meta['stages'] for it in s['history']))
    except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
        row.setdefault('seconds',time.perf_counter()-tick)
        row.update(status='failed',error=str(exc),passed=False)
    print(m,seed,method,row['seconds'],row.get('worst',row.get('error')),
          'jacobians',row.get('jacobians'),flush=True)
    return row


def summarize(rows):
    summary = {}
    for m in GATES:
        by_method = {method:[r for r in rows if r['method']==method and r['m']==m] for method in RUNS}
        paired = [{method:next(r['seconds'] for r in group if r['seed']==seed)
                   for method,group in by_method.items()} for seed in SEEDS]
        eligible = all(row['passed'] for group in by_method.values() for row in group)
        summary[m] = dict(gate=GATES[m],paired=paired,
                          same_accuracy=eligible,
                          v5_wins=sum(r['v5'] < r['v4'] for r in paired) if eligible else None,
                          median_paired_ratio=statistics.median(r['v4']/r['v5'] for r in paired) if eligible else None,
                          methods={})
        for method,group in by_method.items():
            valid = [r for r in group if r['status']=='completed']
            summary[m]['methods'][method] = dict(
                seconds=statistics.median(r['seconds'] for r in valid) if valid else None,
                worst={key:max(r['worst'][key] for r in valid) for key in valid[0]['worst']} if valid else None,
                jacobians=statistics.median(r['jacobians'] for r in valid) if valid else None,
                passes=sum(r['passed'] for r in group),
                completed=len(valid),
                guard_restarts=sum(bool(r['metadata']['fallback_triggers']) for r in valid),
                unresolved=sum(r['metadata']['final_budget_unresolved'] for r in valid))
    return summary


def main():
    if OUTPUT.exists() or FREEZE.exists():
        raise FileExistsError('Preserve the frozen evidence')
    archive = json.loads((V4_DIR/'cavity_formal.json').read_text(encoding='utf-8'))
    v4 = {r['config']['m']:r['config'] for r in archive['specs']
          if r['name'] in ('rfm496_guard','rfm595_guard')}
    configs = {m:dict(v4=v4[m],v5=dict(v4[m],**V5)) for m in GATES}
    rng = np.random.default_rng(532383)
    choices = [(m,method) for m in GATES for method in RUNS]
    schedule = [(seed,*choices[i]) for seed in SEEDS for i in rng.permutation(len(choices))]
    tree = ROOT.parents[1]
    sources = (
        ROOT/'cavity_paired.py',ROOT/'cavity_v5.py',ROOT/'timing_v5.py',
        V4_DIR/'cavity_solver.py',V4_DIR/'cavity_formal.json',
        V3_DIR/'cavity_study.py',V3_DIR/'cavity1000_ref256.npz',V3_DIR/'cavity1000_ref384.npz',
        ROOT.parent/'rfm_ns'/'rfm_v2_solver.py',ROOT.parent/'rfm_ns'/'rfm_solver.py')
    hashes = {p.relative_to(tree).as_posix():digest(p) for p in sources}
    freeze = dict(seeds=SEEDS,gates=GATES,configs=configs,schedule=schedule,hashes=hashes,
                  cpu=pin(),timing='Cold single-thread solve. Reference and scoring excluded.')
    write(FREEZE,freeze)
    score = Score()
    np.linalg.svd(np.eye(10))
    result = dict(freeze=freeze,freeze_sha256=digest(FREEZE),status='running',rows=[])
    for seed,m,method in schedule:
        result['rows'].append(run(m,seed,method,configs[m][method],score))
        write(OUTPUT,result)
    if any(digest(tree/p)!=value for p,value in hashes.items()):
        raise RuntimeError('Source changed during the frozen run')
    result.update(status='complete',summary=summarize(result['rows']))
    write(OUTPUT,result)


if __name__=='__main__':
    main()
