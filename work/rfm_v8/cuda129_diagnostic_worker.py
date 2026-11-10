"""One modern-toolchain diagnostic solve with fresh-process RETURN and SAVE."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from solver_worker import THREADS, affinity_mask, emit, model_signature, require
from prepared_process_identity import current_process_identity


def worker(path,expected):
    payload = path.read_bytes()
    require(sha256(payload).hexdigest()==expected,"Toolchain request hash differs")
    job = json.loads(payload)
    require(job["schema"]=="rfm-v8-cuda129-diagnostic-1" and
            all(os.environ.get(key)=="1" for key in THREADS) and
            os.environ.get("CUDA_CACHE_DISABLE")=="1" and "CUDA_PATH" not in os.environ,
            "Unregistered diagnostic process or cache policy")
    target = path.parent/job["trajectory"]
    require(target.resolve().parent==path.parent and not target.exists(),"Invalid toolchain output")
    from cuda129_case_runtime import FdCaseRuntime, GalerkinCaseRuntime, VARIANTS
    require(job["variant"] in VARIANTS,"Unregistered toolchain variant")
    if job["family"]=="fd":
        from fd_cuda_native_cuda129 import solve_fd_native as solve
        runtime = FdCaseRuntime
    else:
        from galerkin_cuda_prepared_cuda129 import solve_cuda_prepared as solve
        runtime = GalerkinCaseRuntime
    ident = current_process_identity()
    marker = dict(index=job["index"],run_token=job["run_token"],request_sha256=expected,pid=ident["pid"],
                  birth_filetime_100ns=ident["birth_filetime_100ns"],inherited_affinity_mask=affinity_mask())
    require(marker["inherited_affinity_mask"]==job["affinity_mask"],"Worker CPU mask differs")
    evidence = {}

    def factory(toolkit):
        rt = runtime(toolkit,variant=job["variant"])
        evidence.update(rt.evidence)
        return rt

    emit("READY",marker)
    require(sys.stdin.buffer.readline()==b"GO\n","Missing solver permission")
    started = perf_counter()
    try:
        solution = solve(runtime_factory=factory,**job["solve_args"])
        core = perf_counter()-started
        spatial = model_signature(solution.model,job["family"])
        t = perf_counter()
        solution.model = None
        released = perf_counter()-t
    except Exception as exc:
        detail = str(exc).replace(str(Path(__file__).resolve().parents[2]),"<workspace>").replace(str(Path.home()),"<user>")
        emit("RETURN",dict(marker,status="failed",failure_type=type(exc).__name__,failure=detail))
        return 1
    emit("RETURN",dict(marker,status="solved",solver_seconds=core+released,solver_core_seconds=core,
                        model_release_seconds=released,spatial_sha256=spatial,toolchain=evidence))
    require(sys.stdin.buffer.readline()==b"SAVE\n","Missing save permission")
    from full_case import store_solution
    temp = target.with_suffix(".tmp.npz")
    with temp.open("xb") as stream:
        store_solution(stream,solution,job["config"])
    temp.rename(target)
    return 0


if __name__=="__main__":
    require(len(sys.argv)==3,"Expected frozen request and SHA-256")
    raise SystemExit(worker(Path(sys.argv[1]).resolve(),sys.argv[2]))
