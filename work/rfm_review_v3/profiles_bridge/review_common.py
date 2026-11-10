"""Read-only access to hash-bound archived trajectories; no GPU calls."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import sys, json, hashlib
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path[:0] = [str(WORK/'rfm_v7'), str(WORK/'rfm_v6'), str(WORK/'rfm_v5')]
import cpu_formal as base
import cpu_sbdf2_switch as switch
from fd_cpu_buffered import BufferedFD
from fd_pow2_long import Pow2EquilibratedBufferedFD
import numpy as np
case = base.case
METRICS = ('velocity','vorticity','near_velocity','near_vorticity')
NAMES = ('rfm37_k72','leg35_k72_q66','rfm41_k80','leg36_k80_q76')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

def bound(spec):
    path = WORK/spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError('Frozen input changed: '+str(path))
    return path

PLAN = read(WORK/'rfm_v8/rank_fair_cache_qualification_output/protocol.json')
FROZEN = read(WORK/'rfm_v7/switch_qualification_output/protocol.json')
TIMES = FROZEN['physical_times']

def job(name, arm):
    return next(j for j in PLAN['slots'] if j['name']==name and j['arm']==arm
                and j['seed'] in (7331,17) and j['policy']=='empty_first')

def select(sol,times=TIMES):
    indices = []
    for t in times:
        m = np.flatnonzero(sol.times==t)
        if len(m)!=1: raise ValueError('Missing exact physical time '+str(t))
        indices.append(int(m[0]))
    return case.Solution(sol.model,sol.states[indices],np.asarray(times),sol.probe_times,
                         sol.transverse,sol.vorticity,sol.row)

def combine(early,late):
    # Each component is a full-history run from rest. Only observations are joined.
    e,l=select(early,TIMES[:12]),select(late,TIMES[12:])
    return case.Solution(e.model,np.concatenate((e.states,l.states)),np.asarray(TIMES),
                         l.probe_times,l.transverse,l.vorticity,l.row)

def control(name,arm):
    j=job(name,arm); specs=j['control']
    cls=BufferedFD if j['family']=='fd' else switch.RecurrenceGalerkin
    sols={key:case.load_solution(bound(spec),cls) for key,spec in specs.items()}
    if 'single' in sols: return select(sols['single'])
    if not switch.equal_spatial_models((sols['early'].model,sols['late'].model)):
        raise ValueError('Archived coordinate system differs')
    return combine(sols['early'],sols['late'])

def references():
    result={}
    for name in ('angular_fine','time_fine'):
        early=case.load_solution(bound(FROZEN['early_references'][name]),Pow2EquilibratedBufferedFD)
        rate=384 if name=='angular_fine' else 512
        path=WORK/f'rfm_v7/cpu_formal_point1_pow2_chain_output/fd448_k144_rate{rate}_pow2.npz'
        expected=FROZEN['late_reference']['reference_hashes'][path.relative_to(WORK).as_posix()]
        if sha(path)!=expected: raise ValueError('Late reference hash differs')
        late=case.load_solution(path,Pow2EquilibratedBufferedFD)
        result[name]=combine(early,late)
    return result

def flatten_scores(scores):
    return [dict(reference=ref,**row) for ref,rows in scores.items() for row in rows]

def maxima(scores):
    rows=flatten_scores(scores)
    return {key:dict(value=max(rows,key=lambda r:r[key])[key],
                     **{k:max(rows,key=lambda r:r[key])[k] for k in ('reference','grid','time')})
            for key in METRICS}
