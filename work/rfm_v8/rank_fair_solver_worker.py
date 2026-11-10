"""Fresh rank-fair GPU solver with no scoring imports and a bounded file handshake."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

from rank_fair_owned_transport import SECOND, key, latch, publish, read, require, require_identity, wait_latch
from solver_worker import THREADS, affinity_mask, model_signature
from prepared_process_identity import current_process_identity, identity_from_pid


SCHEMA = "rfm-v8-rank-fair-worker-1"


def select_solver(request):
    require(request["imports"] == "minimal" and request["engine"] in ("warp_cufft", "warp_fused"),
            "Unknown solver engine or import policy")
    from fused_marches import solve_combined_fused, solve_fd_fused, solve_uniform_fused
    return {"uniform":solve_uniform_fused,"combined":solve_combined_fused,"fd":solve_fd_fused}[request["arm"]]


def admission(request_path, expected_sha):
    raw = request_path.read_bytes()
    require(sha256(raw).hexdigest() == expected_sha, "Frozen request hash differs")
    request = json.loads(raw)
    require(request["schema"] == SCHEMA and all(os.environ.get(name) == "1" for name in THREADS),
            "Unexpected worker schema or thread policy")
    index, output = request["index"], request_path.parent
    target = output/request["trajectory"]
    require(target.name == f"trajectory_{index:03d}.npz" and target.resolve().parent == output and not target.exists(),
            "The registered trajectory target is invalid")
    identity, parent = current_process_identity(), identity_from_pid(os.getppid())
    marker = dict(index=index, run_token=request["run_token"], request_sha256=expected_sha,
                  pid=identity["pid"], birth_filetime_100ns=identity["birth_filetime_100ns"],
                  inherited_affinity_mask=affinity_mask())
    require(marker["inherited_affinity_mask"] == request["affinity_mask"], "Worker inherited different CPU affinity")
    return request, target, identity, parent, marker


def receive_control(target, event, request, identity, parent, marker):
    stamp, value = wait_latch(latch(target.parent,request["index"],event), time.perf_counter_ns()+60*SECOND)
    require_identity(value,event,request,marker["request_sha256"],identity)
    require(key(value["coordinator"]) == key(parent) == key(identity_from_pid(os.getppid())), "Exact parent identity differs")
    if event == "GO":
        ledger = read(target.parent/"identity_ledger.json")
        entries = [entry for entry in ledger["children"] if entry["index"] == request["index"]]
        require(len(entries) == 1 and ledger["run_token"] == request["run_token"] and key(ledger["coordinator"]) == key(parent),
                "Registered worker admission is missing")
        entry = entries[0]
        require(entry["status"] == "monitor_registered" and key(entry["identity"]) == key(identity) and
                entry["request_sha256"] == marker["request_sha256"] and
                entry["launch_monotonic_ns"] <= entry["monitor_registration_qpc_ns"] <= value["sent_monotonic_ns"] <= stamp,
                "Monitor admission was not committed before GO")


def worker(request_path, expected_sha):
    request,target,identity,parent,marker = admission(request_path,expected_sha)
    solve = select_solver(request)
    index, output = request["index"], target.parent
    publish(latch(output,index,"READY"),dict(marker,event="READY",published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target,"GO",request,identity,parent,marker)
    started = time.perf_counter_ns()
    try:
        solution = solve(engine=request["engine"],**request["solve_args"])
        solved = time.perf_counter_ns()
        signature = model_signature(solution.model,request["family"])
        release_start = time.perf_counter_ns()
        solution.model = None
        released = time.perf_counter_ns()
        core,release = (solved-started)/SECOND,(released-release_start)/SECOND
        returned = dict(status="solved",solver_seconds=core+release,solver_core_seconds=core,model_release_seconds=release,
                        spatial_sha256=signature,solve_started_ns=started,solve_completed_ns=solved,
                        model_release_started_ns=release_start,model_released_ns=released)
    except Exception as error:
        publish(latch(output,index,"RETURN"),dict(marker,event="RETURN",status="failed",failure_type=type(error).__name__,
                                                 published_monotonic_ns=time.perf_counter_ns()))
        return 1
    publish(latch(output,index,"RETURN"),dict(marker,**returned,event="RETURN",published_monotonic_ns=time.perf_counter_ns()))
    receive_control(target,"SAVE",request,identity,parent,marker)
    import full_case as case
    temporary = target.with_suffix(".tmp.npz")
    with temporary.open("xb") as handle:
        case.store_solution(handle,solution,request["config"])
    temporary.rename(target)
    return 0


if __name__ == "__main__":
    require(len(sys.argv) == 3, "Expected a frozen request and hash")
    raise SystemExit(worker(Path(sys.argv[1]).resolve(),sys.argv[2]))
