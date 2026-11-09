"""CPU pilot for experimental throughput, using real solver work and fixed FFT threading."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from itertools import product
from pathlib import Path
import sys
import time

from cpu_formal import OUT, PROTOCOL, case, config, read, save


_blas = None
_threads = None


def tune_windows_blas(threads):
    """Use the active NumPy OpenBLAS DLL's exported runtime thread control."""
    global _blas, _threads
    if sys.platform != "win32":
        raise RuntimeError("This machine-specific pilot expects the Windows NumPy OpenBLAS build")
    directory = Path(case.np.__file__).parent.parent / "numpy.libs"
    files = list(directory.glob("libscipy_openblas64_*.dll"))
    if len(files)!=1:
        raise RuntimeError("Expected a single NumPy OpenBLAS DLL")
    _blas = ctypes.CDLL(str(files[0]))
    setter = _blas.scipy_openblas_set_num_threads64_
    setter.argtypes = (ctypes.c_int,)
    setter.restype = None
    getter = _blas.scipy_openblas_get_num_threads64_
    getter.restype = ctypes.c_int
    setter(threads)
    _threads = getter()
    if _threads!=threads:
        raise RuntimeError(f"Requested BLAS threads {threads}, read {_threads}")


def trial(job):
    spec,protocol,horizon = job
    cls,settings = config(spec,protocol,protocol["seeds"][0])
    t0 = time.perf_counter()
    sol = case.solve(cls,steps=horizon*spec["rate"],horizon=horizon,
                     snapshots=(horizon,),**settings)
    elapsed = time.perf_counter()-t0
    if not case.np.isfinite(sol.states).all():
        raise FloatingPointError(f"Nonfinite pilot state for {spec['name']}")
    return dict(name=spec["name"],seconds=elapsed,observed_blas_threads=_threads,
                steps=sol.row["steps"],state_norm=float(case.np.linalg.norm(sol.states[-1])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,nargs="+",default=[1,4,8,12])
    parser.add_argument("--threads",type=int,nargs="+",default=[1,2])
    parser.add_argument("--scenario",nargs="+",metavar="WORKERS:BLAS_THREADS",
                        help="Use these exact worker/thread pairs rather than the Cartesian grid.")
    parser.add_argument("--horizon",type=int,default=4)
    parser.add_argument("--jobs-per-family",type=int,default=4)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--output",type=Path,default=OUT/"throughput.json")
    args=parser.parse_args()
    if min(*args.workers,*args.threads,args.jobs_per_family,args.horizon,args.repeats)<1:
        parser.error("Use positive worker, thread, horizon and repetition counts")
    protocol=read(PROTOCOL)
    variants={s["name"]:s for s in protocol["variants"]}
    names=("rfm37_k72","leg37_k72","fd176_k80")
    specs=[variants[n] for n in names]
    jobs=[(s,protocol,args.horizon) for _ in range(args.jobs_per_family) for s in specs]
    try:
        pairs=[tuple(map(int,s.split(":"))) for s in args.scenario] if args.scenario else list(
            product(args.workers,args.threads))
        if any(len(pair)!=2 or min(pair)<1 for pair in pairs) or len(set(pairs))!=len(pairs):
            raise ValueError("Invalid or duplicate pair")
    except ValueError as exc:
        parser.error(str(exc))
    scenarios=[(w,t,i) for w,t in pairs for i in range(1,args.repeats+1)]
    rng=case.np.random.default_rng(64812)
    ordered=[scenarios[i] for i in rng.permutation(len(scenarios))]
    frozen=dict(horizon=args.horizon,scenarios=ordered,specification=specs,jobs_per_family=args.jobs_per_family,
                note="Processes start from scratch, FFT/native band solve stay single-threaded. "
                     "Short-window mixed-family throughput only, separate from isolated timings.")
    if args.output.exists():
        result=read(args.output)
        if result["frozen"]!={**frozen,"scenarios":[list(s) for s in ordered]}:
            raise RuntimeError("Throughput workload or settings changed")
    else:
        result=dict(status="running",frozen=frozen,samples=[])
        save(args.output,result)
    known={(r["workers"],r["blas_threads"],r["repetition"]) for r in result["samples"]}
    for workers,threads,repetition in ordered:
        if (workers,threads,repetition) in known:
            continue
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers,initializer=tune_windows_blas,
                                 initargs=(threads,)) as pool:
            rows=list(pool.map(trial,jobs))
        wall=time.perf_counter()-start
        result["samples"].append(dict(workers=workers,blas_threads=threads,repetition=repetition,
            wall_seconds=wall,completed_jobs=len(rows),jobs_per_second=len(rows)/wall,rows=rows))
        save(args.output,result)
        print(workers,threads,repetition,round(len(rows)/wall,4),flush=True)
    result["status"]="complete"
    save(args.output,result)


if __name__=="__main__":
    main()
