"""Standard-library-only admission. Pin CPU 4 before any scientific/solver import."""
import ctypes as C
from ctypes import wintypes as W
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "rfm_v7"))
from prepared_process_identity import current_process_identity, identity_from_pid
from transport_v2 import SECOND, key, latch, publish, read, require, require_identity, wait_latch

SCHEMA = "rfm-v8-rank-fair-worker-affinity-2"
BOOTSTRAP_SCHEMA = "rfm-v8-affinity-before-science-2"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
SCIENTIFIC_PREFIXES = ("numpy", "scipy", "cupy", "torch", "numba", "pycuda", "cuda", "cufft", "nvrtc", "warp", "galerkin", "fused_marches", "full_case", "full_galerkin", "fd_cuda")


def scientific_modules():
    return sorted(n for n in sys.modules if n.split(".")[0].startswith(SCIENTIFIC_PREFIXES))


def affinity():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.GetProcessAffinityMask.argtypes = (W.HANDLE, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t))
    kernel.GetProcessAffinityMask.restype = W.BOOL
    process, system = C.c_size_t(), C.c_size_t()
    require(bool(kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), C.byref(process), C.byref(system))), "Cannot read worker affinity")
    return process.value


def pin(mask):
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.SetProcessAffinityMask.argtypes = (W.HANDLE, C.c_size_t)
    kernel.SetProcessAffinityMask.restype = W.BOOL
    require(bool(kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask)), "Cannot pin effective worker CPU")
    require(affinity() == mask, "Worker effective CPU was not applied")


def admission(request_path, expected_sha):
    entered = time.perf_counter_ns()
    raw = request_path.read_bytes()
    require(sha256(raw).hexdigest() == expected_sha, "Frozen request hash differs")
    request = json.loads(raw)
    require(request["schema"] == SCHEMA and all(os.environ.get(n) == "1" for n in THREADS), "Unexpected v2 schema or BLAS policy")
    require(request["inherited_affinity_mask"] == 256 and request["effective_affinity_mask"] == request["affinity_mask"] == 16,
            "V2 requires coordinator CPU 8 and solver CPU 4")
    before_modules = scientific_modules()
    require(not before_modules, "Scientific modules loaded before effective CPU pin")
    inherited = affinity()
    require(inherited == request["inherited_affinity_mask"], "Worker inherited a different coordinator CPU")
    pin(request["effective_affinity_mask"])
    pinned = time.perf_counter_ns()
    require(not scientific_modules(), "Scientific modules loaded during standard-library bootstrap")
    identity, parent = current_process_identity(), identity_from_pid(os.getppid())
    index, output = request["index"], request_path.parent
    target = output / request["trajectory"]
    require(target.name == f"trajectory_{index:03d}.npz" and target.resolve().parent == output and not target.exists(), "Invalid trajectory target")
    proof = dict(schema=BOOTSTRAP_SCHEMA, index=index, run_token=request["run_token"], request_sha256=expected_sha,
                 identity=identity, parent_identity=parent, inherited_affinity_mask=inherited, effective_affinity_mask=affinity(),
                 admission_entered_ns=entered, effective_pin_completed_ns=pinned,
                 before_solver_import_ns=time.perf_counter_ns(), scientific_modules_before_pin=before_modules,
                 scientific_modules_after_pin=scientific_modules(), thread_policy={n: os.environ[n] for n in THREADS})
    path = output / f"affinity_bootstrap_{index:03d}.json"
    publish(path, proof)
    marker = dict(index=index, run_token=request["run_token"], request_sha256=expected_sha,
                  pid=identity["pid"], birth_filetime_100ns=identity["birth_filetime_100ns"], inherited_affinity_mask=inherited,
                  effective_affinity_mask=affinity(), affinity_bootstrap_sha256=sha256(path.read_bytes()).hexdigest())
    return request, target, identity, parent, marker


def receive_control(target, event, request, identity, parent, marker):
    stamp, value = wait_latch(latch(target.parent, request["index"], event), time.perf_counter_ns() + 60 * SECOND)
    require_identity(value, event, request, marker["request_sha256"], identity)
    require(value["affinity_bootstrap_sha256"] == marker["affinity_bootstrap_sha256"] and affinity() == 16,
            "Worker affinity or bootstrap binding changed")
    require(key(value["coordinator"]) == key(parent) == key(identity_from_pid(os.getppid())), "Exact parent identity differs")
    if event == "GO":
        ledger = read(target.parent / "identity_ledger.json")
        entries = [r for r in ledger["children"] if r["index"] == request["index"]]
        require(len(entries) == 1 and ledger["run_token"] == request["run_token"] and key(ledger["coordinator"]) == key(parent), "Missing v2 admission")
        entry = entries[0]
        require(entry["status"] == "monitor_registered" and key(entry["identity"]) == key(identity) and entry["request_sha256"] == marker["request_sha256"] and
                entry["launch_monotonic_ns"] <= entry["monitor_registration_qpc_ns"] <= value["sent_monotonic_ns"] <= stamp,
                "Monitor admission was not committed before GO")
