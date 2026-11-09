"""Pilot of reusable NumPy FFT output arrays for the exact full-circle kernel."""
import argparse
import json
from pathlib import Path
import time

from full_nonlinear import FullNonlinear, np
from full_nonlinear_audit import convolution, random_modes, rel


class BufferedProducts(FullNonlinear):
    def __init__(self,q,k):
        super().__init__(q,k,'fft')
        self.real_values=np.empty((3,q,self.n))
        self.output_modes=np.empty((2,q,self.n//2+1),dtype=complex)

    def __call__(self,ur,ut,omega):
        spectrum=self._workspace(3)
        spectrum[0,:,:self.count]=ur
        spectrum[1,:,:self.count]=ut
        spectrum[2,:,:self.count]=omega
        values=np.fft.irfft(spectrum,n=self.n,norm='forward',out=self.real_values)
        np.multiply(values[:2],values[2],out=values[:2])
        coefficients=np.fft.rfft(values[:2],norm='forward',out=self.output_modes)
        # No caller can observe subsequent work-array writes.
        a,b=coefficients[:,:,:self.count].copy()
        return a,b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args()
    rng=np.random.default_rng(61109)
    rows=[]
    for q,k in ((32,48),(40,64),(48,80),(50,96)):
        ur,ut,omega=random_modes(rng,q,k,3)
        scipy,numpy=FullNonlinear(q,k),BufferedProducts(q,k)
        expected=convolution(ur,omega),convolution(ut,omega)
        backends=(scipy,numpy)
        error=[max(rel(a,b) for a,b in zip(f(ur,ut,omega),expected)) for f in backends]
        if max(error)>5e-13:
            raise ArithmeticError((q,k,error))
        times=[[],[]]
        for _ in range(5):
            for backend in rng.permutation(2):
                start=time.perf_counter()
                for _ in range(args.repeats):
                    backends[backend](ur,ut,omega)
                times[backend].append((time.perf_counter()-start)/args.repeats)
        row=dict(q=q,k=k,n=scipy.n,relative_errors=dict(zip(('scipy','numpy_out'),error)),
                 microseconds={b:1e6*float(np.median(t)) for b,t in zip(('scipy','numpy_out'),times)})
        rows.append(row)
        print(row,flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
