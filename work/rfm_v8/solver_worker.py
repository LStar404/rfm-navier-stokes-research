"""Fresh CUDA worker with no qualification or report imports.

The coordinator supplies a frozen, hashed request. Imports and READY/GO are
included in its launch-to-RETURN clock. Solver setup, march, teardown and model
release are included in the internal clock. Serialization follows SAVE.
"""
from contextlib import nullcontext
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "rfm_v7"))
MARKER = b"V8_SOLVER "
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(event, values):
    sys.stdout.buffer.write(MARKER + json.dumps(dict(values, event=event), separators=(",", ":"),
                                               allow_nan=False).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()


def affinity_mask():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = C.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    process, system = C.c_size_t(), C.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), C.byref(process), C.byref(system)):
        raise C.WinError(C.get_last_error())
    return process.value


def model_signature(model, family):
    # Identical field order and byte encoding to v7 switch_formal_timing.
    # The registered family replaces an import of the FD class solely for isinstance.
    if family == "fd":
        values = [model.s, model.r, model.right_mean, model.left_d1, model.right_d1]
        for matrix in (model.d1, model.d2, model.wall_d2[0]):
            values.extend((matrix.data, matrix.indices, matrix.indptr))
    else:
        names = ("transform", "mean_transform", "samples", "mean_samples", "M0", "M1", "M_lift",
                 "mean_mass", "test", "test_s", "mean_test", "r", "lift", "k2")
        values = [getattr(model, name) for name in names]
    result = hashlib.sha256()
    for value in values:
        result.update(repr((value.shape, value.dtype.str)).encode("ascii"))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def select_solver(request):
    if request["imports"] == "formal":
        __import__("prepared_formal_timing")
    minimal = request["imports"] == "minimal"
    require(request["imports"] in ("formal", "selected", "minimal"), "Unknown import policy")
    arm = request["arm"]
    context = nullcontext()
    if arm == "fd":
        from fd_cuda_native import solve_fd_native as solve
    elif arm == "uniform":
        from galerkin_cuda_native import solve_native as solve
        if minimal:
            from galerkin_minimal_model import native_model
        else:
            from basis_recurrence_model import native_model
        context = native_model()
    elif arm == "switch_old":
        if minimal:
            from galerkin_cuda_variable_minimal import solve_cuda_switch as solve
        else:
            from galerkin_cuda_variable import solve_cuda_switch as solve
    elif arm in ("module_only", "shared_only", "combined"):
        if minimal:
            from galerkin_cuda_prepared_minimal import solve_cuda_prepared as solve
        else:
            from galerkin_cuda_prepared import solve_cuda_prepared as solve
    else:
        raise ValueError("Unregistered solver arm")
    return solve, context


def worker(request_path, expected_sha):
    require(digest(request_path) == expected_sha, "Request hash mismatch")
    request = json.loads(request_path.read_text(encoding="utf8"))
    require(request["schema"] == "rfm-v8-worker-1", "Unknown worker schema")
    require(all(os.environ.get(key) == "1" for key in THREADS), "Expected one BLAS thread")
    target = request_path.parent / request["trajectory"]
    require(target.parent.resolve() == request_path.parent.resolve() and not target.exists(), "Invalid trajectory target")
    from prepared_process_identity import current_process_identity
    solve, context = select_solver(request)
    ident = current_process_identity()
    marker = dict(index=request["index"], run_token=request["run_token"], request_sha256=expected_sha,
                  pid=ident["pid"], birth_filetime_100ns=ident["birth_filetime_100ns"],
                  inherited_affinity_mask=affinity_mask())
    require(marker["inherited_affinity_mask"] == request["affinity_mask"], "Affinity differs from frozen request")
    emit("READY", marker)
    require(sys.stdin.buffer.readline() == b"GO\n", "Expected GO")
    started = time.perf_counter()
    try:
        with context:
            solution = solve(**request["solve_args"])
        core = time.perf_counter() - started
        signature = model_signature(solution.model, request["family"])
        before_release = time.perf_counter()
        solution.model = None
        release = time.perf_counter() - before_release
    except Exception as exc:
        emit("RETURN", dict(marker, status="failed", failure_type=type(exc).__name__))
        return 1
    emit("RETURN", dict(marker, status="solved", solver_seconds=core + release, solver_core_seconds=core,
                        model_release_seconds=release, spatial_sha256=signature))
    require(sys.stdin.buffer.readline() == b"SAVE\n", "Expected SAVE")
    import full_case as case
    temporary = target.with_suffix(".tmp.npz")
    require(not temporary.exists(), "Temporary trajectory already exists")
    case.store_solution(temporary, solution, request["config"])
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    require(len(sys.argv) == 3, "Expected request path and hash")
    raise SystemExit(worker(Path(sys.argv[1]).resolve(), sys.argv[2]))
