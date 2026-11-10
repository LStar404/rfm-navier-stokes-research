"""Fresh private-context timed worker, with the existing exact bounded handshake."""

from pathlib import Path
import sys
import time

from rank_fair_cache_worker import select
from rank_fair_owned_transport import SECOND, latch, publish, require
from rank_fair_solver_worker import admission, receive_control
from solver_worker import model_signature


CACHE_FIELDS = ("ptx_cache_policy","ptx_cache_observed_mode","ptx_cache_key","ptx_artifact_sha256","ptx_sha256",
                "cuda_context_mode","cuda_driver_api","private_context_finally_unbound","ptx_nvrtc_seconds")


def run(path,digest):
    request,target,identity,parent,marker = admission(path,digest)
    index,output = request["index"],target.parent
    empty = output/f"explicit_cache_{index:03d}"
    require(not empty.exists(),"Each first-use timed slot requires its own new namespace")
    solve = select(request,request["compiler_identity"],empty)
    publish(latch(output,index,"READY"),dict(marker,event="READY",published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target,"GO",request,identity,parent,marker)
    start = time.perf_counter_ns()
    try:
        solution = solve(**request["solve_args"])
        solved = time.perf_counter_ns()
        signature = model_signature(solution.model,request["family"])
        metadata = {field:solution.row[field] for field in CACHE_FIELDS}
        before = time.perf_counter_ns()
        solution.model = None
        released = time.perf_counter_ns()
        core,release = (solved-start)/SECOND,(released-before)/SECOND
        event = dict(status="solved",solver_seconds=core+release,solver_core_seconds=core,model_release_seconds=release,
                     spatial_sha256=signature,solve_started_ns=start,solve_completed_ns=solved,model_release_started_ns=before,
                     model_released_ns=released,cache=metadata)
    except Exception as error:
        publish(latch(output,index,"RETURN"),dict(marker,event="RETURN",status="failed",failure_type=type(error).__name__,
                                                 published_monotonic_ns=time.perf_counter_ns()))
        return 1
    publish(latch(output,index,"RETURN"),dict(marker,**event,event="RETURN",published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target,"SAVE",request,identity,parent,marker)
    from full_case import store_solution

    temporary = target.with_suffix(".tmp.npz")
    with temporary.open("xb") as handle:
        store_solution(handle,solution,request["config"])
    temporary.rename(target)
    return 0


if __name__ == "__main__":
    require(len(sys.argv) == 3,"Expected a registered request and hash")
    raise SystemExit(run(Path(sys.argv[1]).resolve(),sys.argv[2]))
