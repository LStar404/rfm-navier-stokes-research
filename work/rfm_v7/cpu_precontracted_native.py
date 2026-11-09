"""Mode-contiguous native CPU application, preserving the existing SciPy FFT."""
import ctypes as C
from pathlib import Path
import subprocess
import time

from cpu_precontracted import PrecontractedGalerkin, RecurrenceGalerkin, base, case, np

HERE=Path(__file__).resolve().parent
LIBRARY=HERE/'galerkin_cpu_contract.dll'


def compile_library():
    if LIBRARY.exists():
        raise FileExistsError('Preserve the existing native binary')
    subprocess.run(['C:/mingw64/bin/gcc.exe','-O3','-std=c99','-shared','-mavx2',
                    '-ffp-contract=off','-ffreestanding','-nostdlib','-nodefaultlibs',
                    '-Wl,--entry,0',str(HERE/'galerkin_cpu_contract.c'),'-o',str(LIBRARY)],check=True)


class NativePrecontractedGalerkin(PrecontractedGalerkin):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        start=time.perf_counter()
        w,q,m=self.width,len(self.r),self.k+1
        mw=self.mean_width
        linear=np.zeros((w,w,m))
        radial=np.zeros((w,q,m))
        angular=np.zeros_like(radial)
        linear[:,:,1:]=self.pc_mass.transpose(1,2,0)
        radial[:,:,1:]=self.pc_radial.transpose(1,2,0)
        angular[:,:,1:]=(self.pc_angular*self.modes[:,None,None]).transpose(1,2,0)
        linear[:mw,:mw,0]=self.pc_mean_mass
        radial[:mw,:,0]=self.pc_mean_radial
        self.native_operators=linear,radial,angular
        self._native_library=C.CDLL(str(LIBRARY))
        self._native=self._native_library.galerkin_contract
        self._native.argtypes=[C.c_void_p]*8+[C.c_int]*6
        self._native.restype=None
        self.metadata.update(implementation='cpu-galerkin-precontracted-mode-contiguous-avx2',
                             native_setup_seconds=time.perf_counter()-start,
                             reconstruction_class='cpu_precontracted_native.NativePrecontractedGalerkin')

    def terms(self,state,amplitude):
        if (state.shape!=self.rhs_shape or state.dtype!=np.complex128
                or not state.flags.aligned or state.strides[1]!=16 or state.strides[0]<=0):
            raise ValueError('Require an aligned, mode-contiguous state with the registered shape')
        ur,ut,omega=self.fields_on(state,amplitude,self.samples,self.lift,self.r,self.mean_samples)
        radial,angular=self.kernel(ur,ut,omega)
        arrays=(state,radial,angular)
        expected_shapes=(self.rhs_shape,(len(self.r),self.k+1),(len(self.r),self.k+1))
        if any(x.shape!=shape or x.dtype!=np.complex128 or not x.flags.aligned
               or x.strides[1]!=16 or x.strides[0]%16 or x.strides[0]<=0
               for x,shape in zip(arrays,expected_shapes)):
            raise ValueError('Require positive mode-contiguous complex arrays')
        mass=np.empty(self.rhs_shape,dtype=np.complex128)
        flux=np.empty_like(mass)
        pointers=[x.ctypes.data for x in (*self.native_operators,*arrays,mass,flux)]
        strides=[x.strides[0]//16 for x in arrays]
        self._native(*pointers,self.width,len(self.r),self.k+1,*strides)
        mass[:,1]+=amplitude*self.pc_mass_lift
        return mass,flux


def run_pilot():
    destination=HERE/'cpu_precontracted_native_pilot.json'
    if destination.exists():
        raise FileExistsError(destination)
    protocol=base.read(base.PROTOCOL)
    rows=[]
    for spec in protocol['variants']:
        if spec['name'] not in ('rfm37_k72','leg37_k72'):
            continue
        seed=protocol['seeds'][0] if spec['method']=='rfm' else protocol['deterministic_seed']
        _,config=base.config(spec,protocol,seed)
        models=[cls(dt=1/spec['rate'],**config) for cls in (PrecontractedGalerkin,NativePrecontractedGalerkin)]
        with np.load(base.OUT/f'qualify_00_{spec["name"]}_{seed}.npz',allow_pickle=False) as a:
            states=a['states']
        checks=[]
        for i in (0,len(states)-1):
            outputs=[m.terms(states[i],case.speed(protocol['checkpoints'][i])) for m in models]
            err=[float(np.linalg.norm(b-a)/np.linalg.norm(a)) for a,b in zip(*outputs)]
            assert np.isfinite(err).all() and max(err)<1e-10,err
            checks.append(err)
        pairs=[]
        for repeat in range(3):
            results={}
            order=(RecurrenceGalerkin,NativePrecontractedGalerkin) if repeat%2==0 else (NativePrecontractedGalerkin,RecurrenceGalerkin)
            for cls in order:
                results[cls.__name__]=case.solve(cls,steps=8*spec['rate'],horizon=8.,snapshots=(4.,8.),**config)
            original,actual=(results[c.__name__] for c in (RecurrenceGalerkin,NativePrecontractedGalerkin))
            errors=case.maxima(case.score(actual,case.reference_fields(original,base.grid_set()),base.grid_set()))
            probes={k:float(np.linalg.norm(getattr(actual,k)-getattr(original,k))/np.linalg.norm(getattr(original,k)))
                    for k in ('transverse','vorticity')}
            assert max(*errors.values(),*probes.values())<1e-7
            pairs.append(dict(times={k:s.row['seconds'] for k,s in results.items()},fields=errors,probes=probes))
        rows.append(dict(name=spec['name'],seed=seed,fixed_state_errors=checks,pairs=pairs))
        base.save(destination,dict(status='running',timing_scope='paired but non-isolated diagnostic',rows=rows))
        print(spec['name'],[r['times'] for r in pairs],flush=True)
    base.save(destination,dict(status='complete',timing_scope='paired but non-isolated diagnostic',rows=rows))


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument('--compile',action='store_true')
    args=cli.parse_args()
    if args.compile:
        compile_library()
    run_pilot()
