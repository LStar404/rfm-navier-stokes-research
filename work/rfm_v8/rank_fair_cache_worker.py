"""Fresh solver-only worker for private-context, stock cuFFT cache-policy pairs."""

from functools import partial
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

from rank_fair_owned_transport import read, require
from solver_worker import THREADS, affinity_mask, digest, model_signature
from prepared_process_identity import current_process_identity, identity_from_pid


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
OWNER_TOKEN = "RFM_RANK_FAIR_OWNER"


def select(job,compiler,empty_root):
    """Select the verified thin runtime without touching any PTX or cache directory."""
    from cuda_private_scope_solver import solve_scoped

    policy = job["policy"]
    require(policy in ("empty_first","strict_hit") and job["engine"] == "warp_cufft","Unregistered cache policy or engine")
    record = job["ptx_artifact"]
    artifact = PurePosixPath(record["path"])
    require(not artifact.is_absolute() and artifact.as_posix() == record["path"] and ".." not in artifact.parts and
            artifact.name == job["ptx_cache_key"]+".rfmptx","Invalid fixed PTX artifact identity")
    directory = empty_root if policy == "empty_first" else WORK/artifact.parent
    return partial(solve_scoped,job["arm"],engine=job["engine"],policy=policy,artifact_directory=directory,
                   expected_artifact_sha256=record["sha256"],compiler_identity=compiler)


def capture_worker(root,index):
    from fresh_worker_diagnostic import pin

    pin(4)
    protocol = root/"protocol.json"
    p = read(protocol)
    h = digest(protocol)
    parent = identity_from_pid(os.getppid())
    require(p["status"] == "frozen" and p["exact_runtime_adapter_selected"] and all(os.environ[name] == "1" for name in THREADS) and
            os.environ[OWNER_TOKEN] == f"{h}:{parent['pid']}:{parent['birth_filetime_100ns']}:{index}" and affinity_mask() == 16,
            "Unexpected capture parent, frozen protocol or BLAS policy")
    slot = root/f"slot_{index:03d}"
    empty = slot/"explicit_cache"
    require(slot.is_dir() and not empty.exists(),"Explicit first-build directory has already been used")
    job = p["slots"][index]
    own = current_process_identity()
    with (slot/"phases.jsonl").open("x",encoding="ascii",newline="\n") as phases:
        def phase(event):
            phases.write(json.dumps(dict(event=event,index=index,qpc_ns=time.perf_counter_ns(),identity=own),separators=(",", ":"))+"\n")
            phases.flush()

        phase("python_ready")
        solve = select(job,p["compiler_identity"],empty)
        phase("solver_enter")
        solution = solve(**job["args"])
        phase("solver_return_after_context_cleanup")
    signature = model_signature(solution.model,job["family"])
    from full_case import store_solution
    from rank_fair_owned_transport import publish

    raw = slot/"trajectory.npz"
    temporary = slot/"trajectory.tmp.npz"
    with temporary.open("xb") as file:
        store_solution(file,solution,job["config"])
    temporary.rename(raw)
    publish(slot/"result.json",dict(status="complete",index=index,protocol_sha256=h,job=job,metadata=solution.row,
            spatial_sha256=signature,trajectory_sha256=digest(raw),affinity_mask=affinity_mask(),identity=current_process_identity()))


if __name__ == "__main__":
    require(len(sys.argv) == 3,"Expected output and registered slot")
    capture_worker(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
