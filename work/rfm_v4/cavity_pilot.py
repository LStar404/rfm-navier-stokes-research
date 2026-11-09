"""New cavity pilots; formal seeds are excluded from exploratory selection."""
from pathlib import Path
import sys
import json
import time
import hashlib

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'rfm_v3'))
from cavity_study import Score,PROBLEM,errors,setting
from cavity_solver import solve
from rfm_v2_benchmark import write_json,one_thread


def config(m,scale,order,q,**more):
    return dict(setting(m,scale,order,q,(1.9,1.9)),**more)


def evaluate(name,settings,seed,score):
    start=time.perf_counter();op,c,meta=solve(PROBLEM,seed=seed,**settings)
    elapsed=time.perf_counter()-start
    values=score.evaluate(lambda xy:op.fields(c,xy))
    row=dict(name=name,config=settings,seed=seed,seconds=elapsed,errors=errors(values),scores=values,metadata=meta)
    print(name,seed,round(elapsed,3),row['errors'],'fallback',meta['fallback_triggers'],flush=True)
    return row


def main():
    score=Score();result=dict(rows=[],formal_seeds_excluded=[3203,3511,3821,4127,4457],
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('cavity_solver.py','cavity_pilot.py')})
    trials=[(f'old_failure666_{strategy}',config(666,16,34,54,strategy=strategy),1423) for strategy in ('original','ladder','guard')]
    for seed in (17,1907):
        trials.extend((name,settings,seed) for name,settings in [
            ('rfm496_guard',config(496,14,29,47)),('rfm595_guard',config(595,15,32,51)),
            ('rfm496_enriched_guard',config(496,14,33,53)),
            ('rfm595_enriched_guard',config(595,15,36,57))])
    with one_thread():
        for name,settings,seed in trials:
            result['rows'].append(evaluate(name,settings,seed,score))
            write_json(ROOT/'cavity_pilot.json',result)


if __name__=='__main__':main()
