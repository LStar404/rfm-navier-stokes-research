"""Read-only algebra/FD4 audit; no time stepping, GPU, or performance test."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
from math import factorial
import os
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'work/rfm_v6'),str(ROOT/'work/rfm_v3')]
import numpy as np
from cylinder_fd import wall_second
from fd_full import FullFD

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rational_solve(matrix,rhs):
    a=[[F(v) for v in row]+[F(y)] for row,y in zip(matrix,rhs)]
    for j in range(len(a)):
        pivot=next(i for i in range(j,len(a)) if a[i][j])
        a[j],a[pivot]=a[pivot],a[j]
        fac=a[j][j]
        a[j]=[x/fac for x in a[j]]
        for i in range(len(a)):
            if i!=j:
                fac=a[i][j]
                a[i]=[x-fac*y for x,y in zip(a[i],a[j])]
    return [row[-1] for row in a]

matrix=[[F(j)**m for j in range(5)]+[F(m==1)] for m in range(6)]
weights=rational_solve(matrix,[0,0,2,0,0,0])
expected=[-F(415,72),F(8),-F(3),F(8,9),-F(1,8),-F(25,6)]
assert weights==expected
moments=[]
for m in range(8):
    actual=sum(weights[j]*F(j)**m for j in range(5))+weights[5]*F(m==1)
    target=F(2 if m==2 else 0)
    moments.append(dict(degree=m,moment=str(actual),target=str(target),
                        residual=str(actual-target),taylor_coefficient=str((actual-target)/factorial(m))))
assert all(F(x['residual'])==0 for x in moments[:6])
assert F(moments[6]['taylor_coefficient'])==-F(1,15)

uniform_check=[]
for h in (F(1,4),F(1,8),F(1,16),F(1,32)):
    approx=sum(weights[j]*(j*h)**6 for j in range(5))/h**2
    uniform_check.append(dict(h=str(h),degree_6_error=str(approx),error_over_h4=str(approx/h**4)))
assert all(F(x['error_over_h4'])==-48 for x in uniform_check)

nonuniform=[]
L=np.log(12.)
for n in (176,184,384,448):
    s=L*np.sin(.5*np.pi*np.linspace(0,1,n+1))
    s[-1]=L
    wall,slopes=wall_second(s)
    h=s[1]
    z=s[:5]/h
    a=wall.getrow(0).toarray()[0,:5]*h*h
    b=slopes[0]*h
    residuals=[float(a@(z**m)+b*(m==1)-(2 if m==2 else 0)) for m in range(6)]
    assert max(abs(x) for x in residuals)<5e-11
    omitted=float(a@(z**6)/factorial(6))
    nonuniform.append(dict(n=n,coordinate='s=log(r/a)',grid='outer',stretch=1.,
        first_spacing=float(h),neighbor_offsets_scaled=z.tolist(),scaled_value_weights=a.tolist(),
        scaled_slope_weight=float(b),polynomial_moment_residuals=residuals,
        first_error_coefficient_over_h4=omitted,
        max_difference_from_uniform_scaled_weights=float(np.max(abs(a-np.asarray([float(v) for v in weights[:5]]))))))

# Construct only, using the actual FullFD class and a SciPy solve backend.
# This checks the wall-vorticity row in the production spatial operator.
model=FullFD(1/128,184,2,stretch=1.,grid='outer',angular='fft',linear='scipy')
assert np.array_equal(model.s,L*np.sin(.5*np.pi*np.linspace(0,1,185)))
state=np.zeros(model.rhs_shape,complex)
state[1]=model.s[1:-1]**2*(1-model.s[1:-1]/L)**3
_,_,omega=model.native(state,0j,True)
polynomial_wall_error=float(abs(omega[0,1]+2/.5**2))
assert polynomial_wall_error<1e-9

# Bind FD configurations to their actual final-cohort records.
cohort=[]
for p in sorted((ROOT/'work/rfm_v8/rank_fair_cache_qualification_output').glob('slot_*/result.json')):
    record=json.loads(p.read_text())
    if record['job']['name'].startswith('fd'):
        cohort.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),
                           name=record['job']['name'],config=record['job']['config'],
                           steps=record['job']['args']['steps']))
assert {row['name'] for row in cohort}=={'fd176_k80','fd184_k80'}
assert all(row['config']['grid']=='outer' and row['config']['stretch']==1. for row in cohort)
references=[]
for label in ('angular_fine_n448_k144_r384','time_fine_n448_k144_r512'):
    p=ROOT/'work/rfm_v7/early_reference_continuation_output'/f'{label}.npz'
    with np.load(p,allow_pickle=False) as arrays:
        config=json.loads(str(arrays['config']))
    assert config['n']==448 and config['grid']=='outer' and config['stretch']==1.
    references.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),config=config))
sources=['work/rfm_v3/cylinder_fd.py','work/rfm_v6/fd_full.py',
         'work/rfm_v6/full_case.py','work/rfm_v7/fd_pow2_long.py',
         'work/rfm_ns/cfd_reference_fourth.py']
result=dict(status='passed',scope='Symbolic rational moments plus spatial-only construction; no PDE trajectory',
    uniform_weights=[str(v) for v in weights],uniform_moments=moments,
    uniform_degree_6_refinement=uniform_check,nonuniform_annular_checks=nonuniform,
    actual_FD184_wall_vorticity_degree5_polynomial_error=polynomial_wall_error,
    final_cohort=cohort,reference_inputs=references,source_sha256={p:sha(ROOT/p) for p in sources},
    verification_source_sha256=sha(Path(__file__)),
    conclusions=[
        'The displayed constant coefficients are the uniform-s special case, not the final nonuniform annular coefficients.',
        'Dropping the wall value coefficient requires psi(0)=0. General value coefficient is -415/72.',
        'D_h psi - psi_ss(0) = -h^4 psi^(6)(0)/15 + O(h^5) in the uniform case.',
        'The canonical annulus uses s_j=log(12)*sin(pi*j/(2*n)), fourth-order Hermite weights from wall value/slope and four neighbors.',
        'For nonzero angular modes psi_k(0)=psi_ks(0)=0, hence omega_k(0)=-a^(-2)*D_s^2 psi_k(0); the mean tangential block is separate.',
        'cfd_reference_fourth.py is an earlier Cartesian cavity implementation, not the final annular reference solver.'])
(HERE/'fd_closure_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=result['status'],uniform_weights=result['uniform_weights'],
    nonuniform_moment_max=max(max(abs(x) for x in row['polynomial_moment_residuals']) for row in nonuniform),
    polynomial_wall_error=polynomial_wall_error,cohort_records=len(cohort)),indent=2))
