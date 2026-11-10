"""Draft CUDA 11.8 full-trajectory qualification of the three rank controls."""

import argparse
import json
import os
from pathlib import Path
import sys

import legendre_rank_uniform as uniform
from galerkin_cuda_prepared_observe import solve_cuda_prepared_observe
from galerkin_minimal_model import native_model
from warp_observe_full_runtime import solve_with_observation
from gpu_owned_watchdog import run_owned
from prepared_process_identity import current_process_identity, identity_from_pid
from watchdog_transport_smoke import pin


base,case,np,old = uniform.base,uniform.case,uniform.np,uniform.old
require,fields_at,measured,maxima = uniform.require,uniform.fields_at,uniform.measured,uniform.maxima
HERE = uniform.HERE
TEMPLATE = HERE/"protocol_legendre_rank_gpu_draft.json"
OUT = HERE/"legendre_rank_gpu_qualification_output"
TOOLKIT_DEFAULT = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
TOOLKIT_FILES = ("nvrtc64_112_0.dll","nvrtc-builtins64_118.dll","cufft64_10.dll")
PROBES = uniform.PROBES
TOKEN = "RFM_LEGENDRE_GPU_QUALIFICATION_OWNER"
THREADS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")


def toolkit_identity(toolkit):
    require(toolkit.name == "v11.8","Register the original CUDA 11.8 toolkit")
    return {name:base.digest(toolkit/"bin"/name) for name in TOOLKIT_FILES}


def record(path,expected=None):
    sha = base.digest(path)
    require(expected is None or sha == expected,"A CPU or GPU control artifact changed")
    return dict(path=path.relative_to(HERE.parent).as_posix(),sha256=sha)


def freeze(toolkit):
    require(not OUT.exists(),"Preserve the first registered GPU attempt")
    p = base.read(TEMPLATE)
    sw_dir,eq_dir = HERE/"legendre_rank_switch_output",HERE/"legendre_rank_match_high_output"
    up,us,sp,ss,xp,xs = (d/name for d,name in ((uniform.OUT,"protocol.json"),(uniform.OUT,"summary.json"),
        (sw_dir,"protocol.json"),(sw_dir,"summary.json"),(eq_dir,"protocol.json"),(eq_dir,"summary.json")))
    archive,switched,exact = (base.read(x) for x in (us,ss,xs))
    require(archive["protocol_sha256"] == base.digest(up) and switched["protocol_sha256"] == base.digest(sp) and
            exact["protocol_sha256"] == base.digest(xp),"CPU qualification journal identity differs")
    cpu_protocol = base.read(sp)
    physical = cpu_protocol["physical_protocol"]
    require(all(p[k] == physical[k] for k in ("physical_times","horizon","probe_spacing","physical_gate","reference_budget_fraction")) and
            p["physical_references"] == physical["references"],"GPU and CPU qualification criteria differ")
    inputs = [Path(__file__),TEMPLATE,up,us,sp,ss,xp,xs,HERE/"warp_observe_cuda.py",HERE/"warp_observe_full_runtime.py",
              HERE/"galerkin_cuda_prepared_observe.py",HERE/"cuda_observe_solver_generation.json",
              *(HERE/name for name in ("gpu_owned_watchdog.py","win32_isolation_monitor.py","watchdog_transport_smoke.py")),
              *(uniform.V7/name for name in ("galerkin_cuda_native.cu","galerkin_cuda_variable.cu","galerkin_cuda_native.py",
                 "galerkin_cuda_prepared.py","galerkin_cuda_module_bundle.py","cuda_environment_native.py","prepared_process_identity.py"))]
    jobs = []
    cpu = {r["job"]["name"]:(r,uniform.OUT) for r in archive["rows"]} | {exact["job"]["name"]:(exact,eq_dir)}
    bridges = {r["job"]["name"]:r for r in switched["rows"]}
    for arm in p["arms"]:
        name = arm["name"]
        anchor,directory = cpu[name]
        bridge = bridges[name]
        require(anchor["status"] == bridge["status"] == "complete" and anchor["physical_pass"] and
                anchor["wake"]["sustained_alternating_probe_signs"] and bridge["qualified"] and
                anchor["actual_rank"] == bridge["actual_rank"] == arm["rank"] and anchor["mean_rank"] == arm["mean_rank"] and
                (bridge["job"]["startup_rate"],bridge["job"]["late_rate"]) == tuple(arm["rates"]),"Expected qualified same-rank CPU identity")
        cfg = anchor["job"]["config"]
        require(cfg == bridge["job"]["config"] and all(cfg[k] == arm[k] for k in ("m","k","q","seed")) and
                cfg["family"] == "legendre","CPU same-space configuration differs")
        uniform_path,switched_path = directory/anchor["artifact"],sw_dir/bridge["artifact"]
        controls = {"uniform":record(uniform_path,anchor["artifact_sha256"]),
                    "combined":record(switched_path,bridge["artifact_sha256"])}
        inputs.extend((uniform_path,switched_path))
        for schedule in p["schedules"]:
            jobs += [dict(arm,config=cfg,schedule=schedule,observer=observer,cpu_control=controls[schedule],
                          uniform_anchor=controls["uniform"]) for observer in p["observers"]]
    require(len(jobs) == p["registered_trajectories"] == 12,"Unexpected GPU qualification inventory")
    sources = dict(cpu_protocol["source_hashes"])
    sources.update({path.relative_to(HERE.parent).as_posix():base.digest(path) for path in inputs})
    require(all(base.digest(HERE.parent/name)==sha for name,sha in sources.items()),"Frozen source or archive changed")
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="frozen",template=p,jobs=jobs,toolkit_sha256=toolkit_identity(toolkit),source_hashes=sources))
    print("FROZEN",base.digest(OUT/"protocol.json"),len(jobs),flush=True)


def stem(job):
    return "_".join(job[k] for k in ("name","schedule","observer"))


def load_control(ref):
    path = HERE.parent/ref["path"]
    record(path,ref["sha256"])
    return case.load_solution(path,uniform.RecurrenceGalerkin)


def probe_errors(candidate,control):
    differences = {}
    for key in PROBES:
        a,b = getattr(candidate,key),getattr(control,key)
        norm = np.linalg.norm(b)
        require(a.shape == b.shape == (577,3) and np.isfinite(a).all() and norm > 1e-12,"Undefined full-horizon probe comparison")
        delta = a-b
        differences[key] = dict(relative=float(np.linalg.norm(delta)/norm),
                                scaled_absolute=float(np.max(abs(delta)/np.maximum(1.,abs(b)))))
    return differences


def run_job(job,p,toolkit,grids,truth,budget,row):
    physical = p["physical_times"]
    mixed,warp = job["schedule"] == "combined",job["observer"] == "warp"
    h0,h1 = job["rates"]
    t0 = p["switch_time"]
    observations = sorted(physical+[t0+i/h1 for i in (1,2)]) if mixed else physical
    args = dict(job["config"],horizon=p["horizon"],snapshots=observations,probe_spacing=p["probe_spacing"],graph_steps=p["graph_steps"])
    if mixed:
        solution = solve_cuda_prepared_observe(warp=warp,startup_rate=h0,late_rate=h1,switch_time=t0,toolkit=toolkit,**args)
    else:
        with native_model():
            solution = solve_with_observation("uniform",warp,steps=h0*p["horizon"],toolkit=toolkit,**args)
    steps = t0*h0+(p["horizon"]-t0)*h1 if mixed else h0*p["horizon"]
    require(np.array_equal(solution.times,observations) and np.array_equal(solution.probe_times,np.arange(577)/8) and
            solution.states.shape == (len(observations),job["rank"],job["k"]+1) and np.isfinite(solution.states).all() and
            solution.model.mean_width == job["mean_rank"] and solution.row["steps"] == steps and
            solution.row["graph_steps"] == p["graph_steps"] and solution.row["observation_reduction"] == job["observer"],
            "CUDA trajectory differs from its registered rank, observations or steps")
    if mixed:
        require(solution.row["marching_fft_batches"] == 2*(steps-1) and solution.row["bridge_extra_fft_batches"] == 0,
                "CUDA bridge work changed")
    artifact = OUT/(stem(job)+".npz")
    with artifact.open("xb") as handle:
        case.store_solution(handle,solution,job["config"])
    row.update(artifact=artifact.name,artifact_sha256=base.digest(artifact))
    cpu = load_control(job["cpu_control"])
    anchor = load_control(job["uniform_anchor"])
    require(np.array_equal(cpu.times,observations) and np.array_equal(cpu.probe_times,solution.probe_times) and
            old.switched.equal_spatial_models((solution.model,cpu.model,anchor.model)),"GPU and CPU spatial coordinates differ")
    raw_fields = fields_at(solution,observations,grids)
    impl = measured(raw_fields,fields_at(cpu,observations,grids),grids,observations)
    max_impl,probes = maxima(impl),probe_errors(solution,cpu)
    if mixed:
        direct,guard,_,tables,_ = uniform.scores(fields_at(solution,physical,grids),fields_at(anchor,physical,grids),truth,grids,physical)
    else:
        direct,guard,tables = {},{},{}
        for name,target in truth.items():
            values = measured(raw_fields,target,grids,physical)
            direct[name] = maxima(values)
            tables[name] = {"direct":[dict(grid=g,time=t,**v) for (g,t),v in values.items()],"guard":[]}
    physical_ok = all(max(v.values()) <= p["physical_gate"] for v in direct.values())
    guard_ok = all(max(v.values()) <= p["physical_gate"] for v in guard.values())
    impl_ok = all(v <= p["implementation_gate"] for v in (*max_impl.values(),*(r["relative"] for r in probes.values())))
    same_serial = None
    if warp:
        serial_job = dict(job,observer="serial")
        serial = base.read(OUT/(stem(serial_job)+".json"))
        require(serial["job"] == serial_job and "artifact" in serial,"Matched serial GPU trajectory is missing")
        control = load_control(record(OUT/serial["artifact"],serial["artifact_sha256"]))
        exact = {k:getattr(solution,k).tobytes()==getattr(control,k).tobytes() for k in ("states","times","probe_times")}
        errors = probe_errors(solution,control)
        same_serial = dict(bitwise=exact,probe_differences=errors,paired_serial_qualified=serial["qualified"],passed=bool(
            serial["qualified"] and all(exact.values()) and all(v["relative"] <= p["warp_probe_relative_gate"] and
                v["scaled_absolute"] <= p["warp_probe_scaled_absolute_gate"] for v in errors.values())))
    wake = base.wake(solution)
    row.update(status="complete",qualified=bool(physical_ok and guard_ok and impl_ok and wake["sustained_alternating_probe_signs"] and
            (same_serial is None or same_serial["passed"])),actual_rank=solution.model.width,mean_rank=solution.model.mean_width,
        steps=steps,nonlinear_evaluations=steps-1,marching_fft_batches=2*(steps-1),reference_budget=budget,
        physical_pass=physical_ok,guard_pass=guard_ok,implementation_pass=impl_ok,physical_worst=direct,guard_worst=guard,
        scores=tables,cpu_implementation_worst=max_impl,cpu_probe_errors=probes,wake=wake,serial_observer_control=same_serial,
        implementation_rows=[dict(grid=g,time=t,**v) for (g,t),v in impl.items()])


def worker(index,toolkit):
    protocol = OUT/"protocol.json"
    frozen = base.read(protocol)
    sha,p = base.digest(protocol),frozen["template"]
    parent = identity_from_pid(os.getppid())
    key = f"{sha}:{parent['pid']}:{parent['birth_filetime_100ns']}:{index}"
    require(os.environ.get(TOKEN) == key and all(os.environ.get(k)=="1" for k in THREADS),
            "Launch GPU work through its watchdog owner")
    affinity = pin(p["watchdog"]["worker_cpu"])
    identity = current_process_identity()
    job = frozen["jobs"][index]
    grids = base.grid_set()
    require(list(grids) == p["physical_grids"],"Frozen scoring grids differ")
    truth,budget = uniform.absolute_reference(base.read(old.FROZEN),grids)
    row = dict(job=job,protocol_sha256=sha,qualified=False,worker_identity=identity,worker_affinity_mask=affinity)
    try:
        run_job(job,p,toolkit,grids,truth,budget,row)
    except (ArithmeticError,RuntimeError,ValueError,TypeError,OSError) as exc:
        row.update(status="failed",failure_type=type(exc).__name__,failure=str(exc))
    base.save(OUT/(stem(job)+".json"),row)
    print(stem(job),row["status"],row["qualified"],flush=True)
    return 0 if row["qualified"] else 2


def run(toolkit):
    protocol = OUT/"protocol.json"
    frozen = base.read(protocol)
    sha,p = base.digest(protocol),frozen["template"]
    require(frozen["status"] == "frozen" and toolkit_identity(toolkit) == frozen["toolkit_sha256"] and
            all(base.digest(HERE.parent/name)==expected for name,expected in frozen["source_hashes"].items()),"Frozen GPU inputs changed")
    with (OUT/"started.json").open("x",encoding="utf-8") as handle:
        json.dump(dict(protocol_sha256=sha),handle)
    policy = p["watchdog"]
    owner_mask = pin(policy["owner_cpu"])
    owner = current_process_identity()
    for key in THREADS:
        os.environ[key] = "1"
    rows = []
    attempts = []
    for index,job in enumerate(frozen["jobs"]):
        os.environ[TOKEN] = f"{sha}:{owner['pid']}:{owner['birth_filetime_100ns']}:{index}"
        guard_dir = OUT/f"watchdog_{index:02d}"
        command = [sys.executable,"-B","-u",str(Path(__file__).resolve()),"--worker",str(index),"--toolkit",str(toolkit)]
        attempt = run_owned(command,guard_dir,deadline_seconds=policy["hard_deadline_seconds"],kill_wait_seconds=policy["kill_wait_seconds"])
        attempt.update(index=index,job=job,owner_affinity_mask=owner_mask)
        base.save(guard_dir/"watchdog.json",attempt)
        attempts.append(attempt)
        sidecar = OUT/(stem(job)+".json")
        row = base.read(sidecar) if sidecar.exists() else None
        if row:
            require(row["job"] == job and row["protocol_sha256"] == sha,"Worker result does not match its frozen slot")
            ident = attempt.get("identity",{})
            require(all(row["worker_identity"][k] == ident.get(k) for k in ("pid","birth_filetime_100ns")) and
                    row["worker_affinity_mask"] == 1<<policy["worker_cpu"],"Exact owned worker identity or CPU differs")
            rows.append(row)
        print(stem(job),attempt["status"],row["qualified"] if row else False,flush=True)
        if attempt["status"] != "complete" or row is None or not row["qualified"]:
            break
    require(toolkit_identity(toolkit) == frozen["toolkit_sha256"] and
            all(base.digest(HERE.parent/name)==expected for name,expected in frozen["source_hashes"].items()),"GPU inputs changed in flight")
    groups = {f"{schedule}/{observer}":sum(r["qualified"] for r in rows if (r["job"]["schedule"],r["job"]["observer"])==(schedule,observer))
              for schedule in p["schedules"] for observer in p["observers"]}
    passed = len(rows)==len(frozen["jobs"]) and all(r["qualified"] for r in rows) and all(a["status"]=="complete" for a in attempts)
    with (OUT/"summary.json").open("x",encoding="utf-8") as handle:
        json.dump(dict(status="complete" if passed else "stopped",protocol_sha256=sha,rows=rows,
                       watchdog_attempts=attempts,unattempted=frozen["jobs"][len(attempts):],
                       qualified_by_schedule_and_observer=groups,isolated_speed_claim_supported=False),handle,indent=2,allow_nan=False)
    return 0 if passed else 2


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--worker",type=int,choices=range(12),help=argparse.SUPPRESS)
    cli.add_argument("--toolkit",type=Path,default=TOOLKIT_DEFAULT)
    args = cli.parse_args()
    if args.freeze:
        freeze(args.toolkit)
    elif args.run:
        sys.exit(run(args.toolkit))
    else:
        sys.exit(worker(args.worker,args.toolkit))
