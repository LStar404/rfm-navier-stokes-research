"""Replay v7 CPU and optional CUDA paths from a self-contained release."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cuda',action='store_true')
    args = cli.parse_args()
    sys.path.insert(0,str(ROOT/'work/rfm_v7'))
    import cpu_formal as base
    from basis_recurrence_model import RecurrenceGalerkin, native_model
    from fd_cpu_buffered import BufferedFD
    from galerkin_cuda_native import solve_native
    from fd_cuda_native import solve_fd_native
    protocol = base.read(base.PROTOCOL)
    specs = {s['name']:s for s in protocol['variants']}
    grids = base.grid_set()
    for name in ('rfm37_k72','leg37_k72','fd176_k80'):
        spec = specs[name]
        seed = protocol['seeds'][0] if spec['method']=='rfm' else protocol['deterministic_seed']
        cls,settings = base.config(spec,protocol,seed)
        kwargs = dict(steps=8*spec['rate'],horizon=8.,snapshots=(8.,),**settings)
        original = base.case.solve(cls,**kwargs)
        improved = base.case.solve(BufferedFD if cls is base.FullFD else RecurrenceGalerkin,**kwargs)
        truth = base.case.reference_fields(original,grids)
        error = max(base.case.maxima(base.case.score(improved,truth,grids)).values())
        if not base.case.np.isfinite(error) or error>1e-8:
            raise RuntimeError(f'CPU replay failed for {name}: {error}')
        print(f'{name} CPU field replay passed: {error:.3g}',flush=True)
        if args.cuda:
            if cls is base.FullFD:
                gpu = solve_fd_native(**kwargs,cuda_linear_solve='dense',graph_steps=24)
            else:
                with native_model():
                    gpu = solve_native(**kwargs,graph_steps=24)
            error = max(base.case.maxima(base.case.score(gpu,truth,grids)).values())
            if not base.case.np.isfinite(error) or error>1e-5:
                raise RuntimeError(f'CUDA replay failed for {name}: {error}')
            print(f'{name} CUDA field replay passed: {error:.3g}',flush=True)
    print('Eight-time-unit replay exercises the nonzero transverse boundary pulse. '
          'It is an implementation smoke test, not a replacement for the t=72 accuracy study.',flush=True)


if __name__=='__main__':
    main()
