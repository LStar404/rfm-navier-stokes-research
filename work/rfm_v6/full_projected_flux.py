"""Pilot the exact commutation of radial test contractions and Fourier transforms."""
import argparse
import json
from pathlib import Path
import time

from full_nonlinear import FullNonlinear, np, rfft
from full_nonlinear_audit import random_modes, rel


class ProjectedFlux(FullNonlinear):
    """Pre-contract real angular samples before Fourier analysis."""
    def __init__(self,k,radial_test,angular_test,mean_test):
        rank,q=radial_test.shape
        super().__init__(q,k,'fft')
        self.radial_test=radial_test
        self.angular_test=angular_test
        self.mean_test=mean_test
        self.projected=np.empty((2,rank,self.n))

    def __call__(self,ur,ut,omega):
        values=self._values((ur,ut,omega))
        np.multiply(values[:2],values[2],out=values[:2])
        np.matmul(self.radial_test,values[0],out=self.projected[0])
        np.matmul(self.angular_test,values[1],out=self.projected[1])
        radial,angular=rfft(self.projected,axis=-1,norm='forward',workers=1)[...,1:self.count]
        mean=-self.mean_test@values[0].mean(axis=-1)
        return radial-self.ik[1:]*angular,mean


class FourierFirst:
    """Matched transform-first contraction using the solver's real BLAS views."""
    def __init__(self,k,radial_test,angular_test,mean_test):
        self.kernel=FullNonlinear(radial_test.shape[1],k)
        self.radial_test=radial_test
        self.angular_test=angular_test
        self.mean_test=mean_test

    def __call__(self,ur,ut,omega):
        radial,angular=self.kernel(ur,ut,omega)
        first=(self.radial_test@np.ascontiguousarray(radial[:,1:]).view(float)).view(complex)
        second=(self.angular_test@np.ascontiguousarray(angular[:,1:]).view(float)).view(complex)
        return first-self.kernel.ik[1:]*second,-self.mean_test@radial[:,0].real


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=16)
    args=parser.parse_args()
    rng=np.random.default_rng(7014)
    rows=[]
    for q,rank,k in ((38,24,48),(50,27,64),(64,34,72),(84,41,96),(100,36,96)):
        settings=random_modes(rng,q,k,3)
        tests=rng.normal(size=(2,rank,q))
        mean=rng.normal(size=(min(24,rank),q))
        plans=FourierFirst(k,*tests,mean),ProjectedFlux(k,*tests,mean)
        errors=[rel(a,b) for a,b in zip(plans[1](*settings),plans[0](*settings))]
        if max(errors)>5e-13:
            raise ArithmeticError(errors)
        times=[[],[]]
        for _ in range(5):
            for j in rng.permutation(2):
                start=time.perf_counter()
                for _ in range(args.repeats):
                    plans[j](*settings)
                times[j].append((time.perf_counter()-start)/args.repeats)
        row=dict(q=q,rank=rank,k=k,errors=errors,
                 microseconds={b:1e6*float(np.median(t)) for b,t in zip(('fourier_first','radial_first'),times)})
        rows.append(row)
        print(row,flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
