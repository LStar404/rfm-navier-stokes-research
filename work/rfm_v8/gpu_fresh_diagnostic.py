"""Short same-transport CUDA phase and fresh-child diagnostics. No isolation ranking."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
V7 = HERE.parent / "rfm_v7"
OUT = HERE / "gpu_fresh_diagnostic_output"
MARKER = b"V8_GPU_DIAGNOSTIC "
ARMS = ("uniform", "switch_old", "module_only", "combined", "fd176_dense")
MODES = {"full24":("full",24), "thin24":("thin",24), "thin64":("thin",64)}
TIMES = (.5,2.,4.,6.,8.,8.125,8.5,9.)
ARRAYS = ("states", "times", "probe_times", "transverse", "vorticity")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def worker():
    entry = time.perf_counter()
    job = json.loads(sys.stdin.buffer.readline())
    sys.path.insert(0,str(V7))
    if job["imports"] == "full":
        # Import the frozen formal entry point, without running its orchestration.
        __import__("prepared_formal_timing")
    arm = job["arm"]
    graph = job["graph_steps"]
    args = dict(horizon=9.,snapshots=TIMES,probe_spacing=.125,graph_steps=graph,**job["config"])
    context = nullcontext()
    if arm == "uniform":
        from basis_recurrence_model import native_model
        from galerkin_cuda_native import solve_native as solve
        context = native_model()
        args.update(steps=1152,profile_steps=job["profile_steps"])
    elif arm == "fd176_dense":
        from fd_cuda_native import solve_fd_native as solve
        args.update(steps=1152,profile_steps=job["profile_steps"],cuda_linear_solve="dense")
    else:
        args.update(startup_rate=128,late_rate=112,switch_time=8.)
        if arm == "switch_old":
            from galerkin_cuda_variable import solve_cuda_switch as solve
        else:
            from galerkin_cuda_prepared import solve_cuda_prepared as solve
            args["preparation"] = arm
    imported = time.perf_counter()
    with context:
        solution = solve(**args)
    solved = time.perf_counter()
    solution.model = None
    released = time.perf_counter()
    arrays = {}
    for name in ARRAYS:
        value = getattr(solution,name)
        arrays[name] = dict(shape=value.shape,dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    phases = {k:v for k,v in solution.row.items() if k.endswith("_seconds")}
    n = 1135 if arm not in ("uniform","fd176_dense") else 1151
    require(solution.row["steps"] == n+1 and all(solution.row.get(k,v)==v for k,v in (
        ("nonlinear_evaluations",n),("marching_fft_batches",2*n),("priming_fft_batches",2))), "Unexpected nonlinear work")
    setup = sum(phases[k] for k in ("host_setup_seconds","runtime_init_seconds","upload_seconds","plan_seconds","graph_setup_seconds"))
    result = dict(job=job,entry_to_import_seconds=imported-entry,solver_seconds=released-imported,
                  release_seconds=released-solved,phases=phases,setup_seconds=setup,
                  nonlinear_evaluations=n,marching_fft_batches=2*n,priming_fft_batches=2,
                  inverse_transforms_per_evaluation=4 if arm=="fd176_dense" else 3,
                  forward_transforms_per_evaluation=1 if arm=="fd176_dense" else 2,
                  array_fingerprints=arrays,device_eager_tail=solution.row.get("diagnostic_eager_tail_profile"))
    print(MARKER.decode()+json.dumps(result,separators=(",",":"),allow_nan=False),flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def launch(job):
    payload = (json.dumps(job,separators=(",",":"))+"\n").encode()
    env = dict(os.environ,**{k:"1" for k in THREADS},PYTHONUNBUFFERED="1")
    started = time.perf_counter_ns()
    with subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--worker"],stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env) as child:
        child.stdin.write(payload)
        child.stdin.close()
        child.stdin = None
        while line := child.stdout.readline():
            received = time.perf_counter_ns()
            if line.startswith(MARKER):
                record = json.loads(line[len(MARKER):])
                break
        else:
            child.wait(timeout=30)
            raise RuntimeError(f"Diagnostic worker exited before its marker ({child.returncode})")
        child.communicate(timeout=30)
        require(child.returncode == 0 and record["job"] == job,"Diagnostic worker changed configuration or failed")
    record["fresh_process_seconds"] = (received-started)/1e9
    record["outside_solver_seconds"] = record["fresh_process_seconds"]-record["solver_seconds"]
    require(record["outside_solver_seconds"] >= 0,"Unexpected fresh-clock relation")
    return record


def run():
    require(not OUT.exists(),"Preserve the existing diagnostic attempt")
    sys.path.insert(0,str(V7))
    import cpu_formal as base
    q = V7 / "switch_qualification_output" / "protocol.json"
    d = V7 / "fd_early_multiscale_output" / "protocol.json"
    gal = read(q)
    fd = read(d)
    config = base.config(gal["specs"]["rfm37_k72"],gal["solve_protocol"],7331)[1]
    from fd_early_multiscale import config as fd_config
    fd_job = next(j for j in fd["jobs"] if j["name"] == "fd176_k80" and j["arm"] == "cuda_graph")
    configs = {arm:config for arm in ARMS} | {"fd176_dense":fd_config(fd_job,fd["template"])}
    inputs = [Path(__file__),q,d,*(V7/name for name in ("prepared_formal_timing.py","galerkin_cuda_native.py",
        "galerkin_cuda_native.cu","galerkin_cuda_variable.py","galerkin_cuda_variable.cu","galerkin_cuda_prepared.py",
        "galerkin_cuda_module_bundle.py","galerkin_phase_shared.py","cuda_environment_native.py","fd_cuda_native.py","fd_cuda_native.cu"))]
    sources = {p.relative_to(HERE.parent).as_posix():digest(p) for p in inputs}
    rng = random.Random(437007)
    jobs = []
    for repeat in (1,2):
        block = [dict(arm=arm,mode=mode,repeat=repeat,imports=imports,graph_steps=graph,config=configs[arm],profile_steps=0)
                 for arm in ARMS for mode,(imports,graph) in MODES.items()]
        rng.shuffle(block)
        jobs.extend(block)
    jobs.extend(dict(arm=arm,mode="profile_only",repeat=0,imports="thin",graph_steps=24,config=configs[arm],profile_steps=24)
                for arm in ("uniform","fd176_dense"))
    OUT.mkdir()
    save(OUT/"protocol.json",dict(status="registered_diagnostic",scope="Sequential fresh children, no isolation monitoring or rankings",
        order_seed=437007,horizon=9.,snapshots=TIMES,jobs=jobs,source_hashes=sources))
    rows = []
    for job in jobs:
        try:
            row = dict(launch(job),status="complete")
        except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
            row = dict(job=job,status="failed",failure_type=type(exc).__name__)
        rows.append(row)
        save(OUT/"runs.json",dict(status="running",rows=rows))
        print(job["arm"],job["mode"],row["status"],row.get("fresh_process_seconds"),flush=True)
    save(OUT/"runs.json",dict(status="complete",rows=rows))
    require(sources == {p.relative_to(HERE.parent).as_posix():digest(p) for p in inputs},"Diagnostic inputs changed")
    require(all(r["status"] == "complete" for r in rows),"Preserved failed diagnostic rows")
    arm_rows = {a:[r for r in rows if r["job"]["arm"]==a] for a in ARMS}
    exact = {a:all(r["array_fingerprints"] == rs[0]["array_fingerprints"] for r in rs) for a,rs in arm_rows.items()}
    require(all(exact.values()),"Import, graph size or profiling changed a saved trajectory")
    groups = {}
    for arm, group in arm_rows.items():
        base_rows = {r["job"]["repeat"]:r for r in group if r["job"]["mode"]=="full24"}
        thin_rows = {r["job"]["repeat"]:r for r in group if r["job"]["mode"]=="thin24"}
        groups[arm] = dict()
        for mode in MODES:
            sample = [r for r in group if r["job"]["mode"]==mode]
            match = base_rows if mode=="thin24" else thin_rows
            keys = ("fresh_process_seconds","outside_solver_seconds","entry_to_import_seconds","solver_seconds","setup_seconds")
            measures = {k:[r[k] for r in sample] for k in keys} | {k:[r["phases"][k] for r in sample] for k in sample[0]["phases"]}
            delta = {k:[r[k]-match[r["job"]["repeat"]][k] for r in sample] for k in keys} if mode!="full24" else {}
            groups[arm][mode] = dict(samples=measures,medians={k:median(v) for k,v in measures.items()},
                                     paired_delta_seconds=delta)
    save(OUT/"summary.json",dict(status="complete",scope="Diagnostic only, two repeats and no external-load isolation",
        protocol_sha256=digest(OUT/"protocol.json"),runs_sha256=digest(OUT/"runs.json"),rows=len(rows),
        exact_saved_trajectories=exact,groups=groups,device_profiles={r["job"]["arm"]:r["device_eager_tail"]
            for r in rows if r["job"]["mode"]=="profile_only"}))


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--worker"] else run()
