"""Exact pre-scaled jet assembly for the full-domain Galerkin hot path."""
import argparse
import json
from pathlib import Path
import time

from full_galerkin import FullGalerkin, np


class BufferedGalerkin(FullGalerkin):
    """Reuse velocity/curl work arrays only for each nonlinear update.

    Rescaling jet rows by fixed powers of r commutes with the coefficient
    matrix product. Evaluation at independent output grids is unchanged.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        reciprocal=1/self.r
        f,d,d2=self.samples
        self.time_jets=np.stack((f*reciprocal[:,None]**2,
                                 -d*reciprocal[:,None],
                                 -d2*reciprocal[:,None]**2))
        self.time_fields=np.empty((3,len(self.r),self.k+1),dtype=complex)
        self.radial_scale=self.r[:,None]*self.ik
        self.lift_modes=np.stack((1j*self.lift[0]*reciprocal,
                                  -self.lift[1]*reciprocal,
                                  (self.lift[0]-self.lift[2])*reciprocal**2))
        self.mean_vort_samples=(self.mean_samples[0]+self.mean_samples[1])*reciprocal[:,None]
        self.metadata['jet_backend']='buffered'

    def fields_on(self,state,amplitude,samples,lift,r,mean):
        if samples is not self.samples:
            return super().fields_on(state,amplitude,samples,lift,r,mean)
        x=np.ascontiguousarray(state).view(float)
        np.matmul(self.time_jets,x,out=self.time_fields.view(float))
        ur,ut,omega=self.time_fields
        omega[:,1:]+=ur[:,1:]*self.k2
        ur[:,1:]*=self.radial_scale
        self.time_fields[:,:,1]+=amplitude*self.lift_modes
        coefficient=state[:self.mean_width,0].real
        ur[:,0]=0.
        ut[:,0]=self.mean_samples[0]@coefficient
        omega[:,0]=self.mean_vort_samples@coefficient
        return ur,ut,omega


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=24)
    args=parser.parse_args()
    rng=np.random.default_rng(42207)
    output=[]
    for m,q,k,scale in ((29,48,56,55.),(43,64,72,67.),(51,78,88,78.)):
        config=dict(dt=1/160,m=m,q=q,k=k,scale=scale,mean_order=23)
        models=FullGalerkin(**config),BufferedGalerkin(**config)
        state=(rng.normal(size=models[0].rhs_shape)+1j*rng.normal(size=models[0].rhs_shape))*.002
        state[:,0]=state[:,0].real
        lift=-.5j
        a,b=[model.terms(state,lift) for model in models]
        differences=[float(np.linalg.norm(x-y)/np.linalg.norm(x)) for x,y in zip(a,b)]
        if max(differences)>5e-13:
            raise ArithmeticError(differences)
        samples=[[],[]]
        for _ in range(5):
            for j in rng.permutation(2):
                start=time.perf_counter()
                for _ in range(args.repeats):
                    models[j].terms(state,lift)
                samples[j].append((time.perf_counter()-start)/args.repeats)
        row=dict(m=m,q=q,k=k,relative_differences=differences,
                 microseconds=[1e6*float(np.median(v)) for v in samples])
        print(row,flush=True)
        output.append(row)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
