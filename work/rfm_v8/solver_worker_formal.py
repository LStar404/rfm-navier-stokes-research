"""The thin solver transport with exact registered parent and monitor admission."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

from solver_worker import THREADS, affinity_mask, emit, model_signature, require, select_solver
from prepared_process_identity import current_process_identity, identity_from_pid


def key(record):
    return record["pid"], record["birth_filetime_100ns"]


def worker(request_path, expected_sha):
    raw = request_path.read_bytes()
    require(sha256(raw).hexdigest() == expected_sha, "Request hash mismatch")
    request = json.loads(raw)
    require(request["schema"] == "rfm-v8-worker-1" and all(os.environ.get(name) == "1" for name in THREADS),
            "Unexpected worker schema or BLAS thread policy")
    target = request_path.parent/request["trajectory"]
    require(target.resolve().parent == request_path.parent.resolve() and not target.exists(), "Invalid trajectory target")
    ident = current_process_identity()
    parent = identity_from_pid(os.getppid())
    solve, context = select_solver(request)
    marker = dict(index=request["index"], run_token=request["run_token"], request_sha256=expected_sha,
                  pid=ident["pid"], birth_filetime_100ns=ident["birth_filetime_100ns"], inherited_affinity_mask=affinity_mask())
    require(marker["inherited_affinity_mask"] == request["affinity_mask"], "Affinity differs from frozen request")
    emit("READY", marker)
    require(sys.stdin.buffer.readline() == b"GO\n", "Expected GO")
    ledger = json.loads((request_path.parent/"identity_ledger.json").read_bytes())
    entries = [row for row in ledger["children"] if row["index"] == request["index"]]
    require(len(entries) == 1 and ledger["run_token"] == request["run_token"] and
            key(ledger["coordinator"]) == key(parent) == key(identity_from_pid(os.getppid())), "Owned parent ledger differs")
    entry = entries[0]
    require(entry["status"] == "monitor_registered" and entry["request_sha256"] == expected_sha and
            key(entry["identity"]) == key(ident) == key(current_process_identity()) and
            entry["launch_monotonic_ns"] <= entry["monitor_registration_qpc_ns"] <= time.perf_counter_ns(),
            "Worker or monitor admission was not committed before GO")
    started = time.perf_counter()
    try:
        with context:
            solution = solve(**request["solve_args"])
        core = time.perf_counter()-started
        signature = model_signature(solution.model, request["family"])
        before_release = time.perf_counter()
        solution.model = None
        release = time.perf_counter()-before_release
    except Exception as error:
        emit("RETURN", dict(marker, status="failed", failure_type=type(error).__name__))
        return 1
    emit("RETURN", dict(marker, status="solved", solver_seconds=core+release, solver_core_seconds=core,
                        model_release_seconds=release, spatial_sha256=signature))
    require(sys.stdin.buffer.readline() == b"SAVE\n", "Expected SAVE")
    import full_case as case
    temporary = target.with_suffix(".tmp.npz")
    require(not temporary.exists(), "Temporary trajectory already exists")
    with temporary.open("xb") as handle:
        case.store_solution(handle, solution, request["config"])
    temporary.rename(target)
    return 0


if __name__ == "__main__":
    require(len(sys.argv) == 3, "Expected request path and hash")
    raise SystemExit(worker(Path(sys.argv[1]).resolve(), sys.argv[2]))
