"""Pilot audit against direct convolution, including the mean and mixed parity."""
import argparse
import json
from pathlib import Path
import time

from full_nonlinear import FullNonlinear, np, irfft, rfft


def convolution(left, right):
    """Return positive retained coefficients of the exact Hermitian convolution."""
    k=left.shape[-1]-1
    bilateral=lambda values: np.concatenate((values[:,1:][:,::-1].conj(),values),axis=1)
    a,b=bilateral(left),bilateral(right)
    return np.stack([np.convolve(x,y)[2*k:3*k+1] for x,y in zip(a,b)])


def random_modes(rng, q, k, count):
    modes=rng.normal(size=(count,q,k+1))+1j*rng.normal(size=(count,q,k+1))
    modes[...,0]=modes[...,0].real
    return modes


def rel(result, reference):
    return float(np.linalg.norm(result-reference)/np.linalg.norm(reference))


def audit():
    rng=np.random.default_rng(51222)
    rows=[]
    for k in (1,4,17,31,47,64,93,128):
        q=7
        ur,ut,w,ws=random_modes(rng,q,k,4)
        r=np.exp(np.linspace(-.69,2,q))
        urw=convolution(ur,w)
        utw=convolution(ut,w)
        exact_adv=-(convolution(ur,ws)+convolution(ut,w*(1j*np.arange(k+1))))/r[:,None]
        for backend in ('basic','fft','dense','packed'):
            plan=FullNonlinear(q,k,backend)
            a,b=plan(ur,ut,w)
            adv,mean=plan.advect(ur,ut,w,ws,r)
            errors=dict(radial=rel(a,urw),angular=rel(b,utw),advective=rel(adv,exact_adv),
                        mean=rel(mean,-urw[:,0].real),conservative_mean=rel(a[:,0],urw[:,0]))
            if max(errors.values())>5e-13:
                raise ArithmeticError((k,backend,errors))
            rows.append(dict(k=k,q=q,backend=backend,n=plan.n,errors=errors))
    k=4
    a=np.zeros(k+1,complex)
    b=a.copy()
    a[k]=.5
    b[k]=-.5j
    n=3*k
    bad=rfft(irfft(a,n=n,norm='forward')*irfft(b,n=n,norm='forward'),norm='forward')[k]
    ok=FullNonlinear(1,k)(a[None,:],a[None,:],b[None,:])[0][0,k]
    alias=dict(k=k,n_equal_3k=n,true_k_coefficient=0.,aliased_amplitude=float(abs(bad)),
               safe_amplitude=float(abs(ok)))
    return dict(errors=rows,equality_counterexample=alias)


def pilot_timing():
    rng=np.random.default_rng(51224)
    rows=[]
    for k,q in ((24,32),(40,46),(56,64),(72,80),(96,104),(128,144)):
        args=random_modes(rng,q,k,4)
        ur,ut,w,ws=args
        r=np.exp(np.linspace(-.69,2,q))
        for operation,settings in (('weak',(ur,ut,w)),('advective',(ur,ut,w,ws,r))):
            plans={b:FullNonlinear(q,k,b) for b in ('basic','fft','dense','packed')}
            def function(plan):
                return plan if operation=='weak' else plan.advect
            for plan in plans.values():
                function(plan)(*settings)
            elapsed={b:[] for b in plans}
            for _ in range(5):
                for backend in rng.permutation(list(plans)):
                    action=function(plans[backend])
                    start=time.perf_counter()
                    for _ in range(28):
                        action(*settings)
                    elapsed[backend].append((time.perf_counter()-start)/28)
            rows.append(dict(k=k,q=q,operation=operation,n=next(iter(plans.values())).n,
                             seconds={k:float(np.median(v)) for k,v in elapsed.items()}))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing',action='store_true',help='Run local kernel timing pilots, not full solver benchmarks.')
    args=parser.parse_args()
    data=audit()
    if args.timing:
        data['pilot_timing']=pilot_timing()
    destination=Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
    print('Max operator error',max(max(r['errors'].values()) for r in data['errors']),flush=True)
    print(data['equality_counterexample'],flush=True)
    for row in data.get('pilot_timing',[]):
        print(row['k'],row['q'],row['operation'],
              {key:round(1e6*v,1) for key,v in row['seconds'].items()},flush=True)


if __name__=='__main__':
    main()
