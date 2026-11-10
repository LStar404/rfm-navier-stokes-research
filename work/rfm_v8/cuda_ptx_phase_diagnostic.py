"""Stop-on-failure phase diagnostics for the original CUDA 11.8 stock control."""

import os
import sys
import time


def mark(event,**data):
    stamp=time.time_ns()
    cpu=os.times()
    value=dict(phase=event,pid=os.getpid(),utc_unix_ns=stamp,monotonic_ns=time.perf_counter_ns(),
               cpu_process_ns=time.process_time_ns(),cpu_user_seconds=cpu.user,cpu_kernel_seconds=cpu.system,**data)
    # A single OS write preserves each marker even when the watchdog terminates Python.
    os.write(1,("RFM_PHASE "+__import__("json").dumps(value,separators=(",",":"),allow_nan=False)+"\n").encode("ascii"))


if "--worker" in sys.argv:
    mark("python_entry")
    mark("standard_imports_begin")

import argparse
from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import inspect
import json
from pathlib import Path
import textwrap

if "--worker" in sys.argv:
    mark("standard_imports_end")


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
V7=WORK/"rfm_v7"
FROZEN=HERE/"cuda_ptx_march_t9_output"
ARCHIVE=HERE/"cuda_ptx_artifact_output"
MODES=("cpu_import_and_source","tiny_device_health","stock_t9")
DEADLINES=(30,20,30)
THREADS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")
ENV_KEYS=(*THREADS,"CUDA_CACHE_DISABLE","CUDA_LAUNCH_BLOCKING","CUDA_MODULE_LOADING","BLIS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","OMP_WAIT_POLICY","MKL_THREADING_LAYER")
ARTIFACT_FILENAME="rfm_v8_warp_march.cu"


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path,data):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")


def environment():
    return {name:os.environ.get(name) for name in ENV_KEYS}


def affinity(cpu=None):
    kernel=C.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=C.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[C.c_void_p,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes=[C.c_void_p,C.c_size_t]
    handle=kernel.GetCurrentProcess()
    if cpu is not None and not kernel.SetProcessAffinityMask(handle,1<<cpu):
        raise C.WinError(C.get_last_error())
    actual,system=C.c_size_t(),C.c_size_t()
    if not kernel.GetProcessAffinityMask(handle,C.byref(actual),C.byref(system)):
        raise C.WinError(C.get_last_error())
    require(cpu is None or actual.value==1<<cpu,"The requested CPU affinity is unavailable")
    return actual.value


def clone(original,markers,**bindings):
    """Insert only the verified markers into a private copy of an original function."""
    source=textwrap.dedent(inspect.getsource(original))
    lines=source.splitlines(keepends=True)
    output=[]
    seen=set()
    for line in lines:
        stripped=line.strip()
        if stripped in markers:
            require(stripped not in seen,"A marker anchor is ambiguous")
            seen.add(stripped)
            prefix=line[:len(line)-len(line.lstrip())]
            output.append(f"{prefix}phase_mark({markers[stripped]!r})\n")
        output.append(line)
    require(seen==set(markers),"Original source no longer matches its phase marker anchors")
    globals_copy=dict(original.__globals__,phase_mark=mark,**bindings)
    exec(compile("".join(output),"<phase-only copy of "+original.__qualname__+">","exec"),globals_copy)
    return globals_copy[original.__name__]


def observed_cuda():
    import cuda_environment_native as native
    import fused_march_runtime as runtime
    import fused_marches as marches
    from warp_observe_cuda import _OwnedDriver

    driver_init=clone(_OwnedDriver.__init__,{
        "call(self.cuInit, 0)":"cuInit_begin",
        "device, version = I(), I()":"cuInit_end",
        "call(self.cuDevicePrimaryCtxRetain, C.byref(context), device)":"context_retain_begin",
        "resources.callback(call, self.cuDevicePrimaryCtxRelease_v2, device)":"context_retain_end",
        "call(self.cuCtxSetCurrent, context)":"context_current_begin",
        "call(self.cuStreamCreate, C.byref(self.stream), 1)":"stream_create_begin",
        "resources.callback(call, self.cuStreamDestroy_v2, self.stream)":"stream_create_end",
    })
    tagged_driver=type("PhaseOwnedDriver",(_OwnedDriver,),{"__init__":driver_init})
    tagged_load=clone(native.nvrtc_load,{
        'lib = C.WinDLL(str(toolkit / "bin" / "nvrtc64_112_0.dll"))':"nvrtc_dll_load_begin",
        'create = bind(lib, "nvrtcCreateProgram", [C.POINTER(P), CHAR, CHAR, I, P, P])':"nvrtc_dll_load_end",
        "call(create, C.byref(program), src, filename, 0, None, None)":"nvrtc_program_begin",
        'status = compile_program(program, 1, (CHAR * 1)(b"--gpu-architecture=compute_52"))':"source_compile_begin",
        "if status:":"source_compile_end",
        "module = P()":"module_load_begin",
        "return module":"module_load_end",
    })
    runtime_init=clone(runtime.FusedMarchRuntime.__init__,{
        "source,names,self.observer=march_source(family,n,self.fused)":"source_assembly_begin",
        "with ExitStack() as resources:":"source_assembly_end",
        'self.driver_lib=C.WinDLL("nvcuda.dll")':"driver_dll_load_begin",
        "self.driver=_OwnedDriver(self.driver_lib,resources)":"driver_dll_load_end",
        "self.kernels={}":"kernel_binding_begin",
        'self.fft=C.WinDLL(str(root/"bin"/"cufft64_10.dll"))':"cufft_dll_load_begin",
        'self.plan_many=bind(self.fft,"cufftPlanMany",[C.POINTER(I),I,C.POINTER(I),P,I,I,P,I,I,I,I])':"cufft_dll_load_end",
        "self.resources=resources.pop_all()":"runtime_setup_end",
    },_OwnedDriver=tagged_driver,nvrtc_load=tagged_load)
    tagged_runtime=type("PhaseStockRuntime",(runtime.FusedMarchRuntime,),{"__init__":runtime_init})
    solver=clone(marches.solve_uniform_fused,{
        "started = time.perf_counter()":"host_setup_begin",
        "host_setup = time.perf_counter() - started":"host_setup_end",
        "runtime_setup = time.perf_counter() - t":"runtime_return",
        "upload_time = time.perf_counter() - t":"uploads_end",
        'angular_nodes, prime, angular_names = nonlinear_nodes(rt,resources,"gal",n,k,q,spectrum,physical,':"plan_begin",
        "for node in prime:":"prime_begin",
        "plan_time = time.perf_counter() - t":"prime_end",
        "executable = rt.capture(resources, body, chunk) if chunk else None":"graph_capture_begin",
        "graph_time = time.perf_counter() - t":"graph_capture_end",
        "if executable is None:":"march_begin",
        "march_time = time.perf_counter() - t":"march_end_after_sync",
        "driver.download(save, checkpoints)":"download_begin",
        "download_time = time.perf_counter() - t":"download_end",
        "if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():":"teardown_end",
    },FusedMarchRuntime=tagged_runtime)
    return native,tagged_driver,tagged_load,solver


def tiny_health(native,tagged_driver,tagged_load,toolkit):
    src=b'extern "C" __global__ void twice(double *x) { *x *= 2.0; }'
    with ExitStack() as resources:
        resources.enter_context(os.add_dll_directory(str(toolkit/"bin")))
        mark("driver_dll_load_begin")
        library=C.WinDLL("nvcuda.dll")
        mark("driver_dll_load_end")
        driver=tagged_driver(library,resources)
        module=tagged_load(toolkit,driver,src,b"phase_tiny_health.cu")
        resources.callback(native.call,driver.cuModuleUnload,module)
        kernel=native.P()
        native.call(driver.cuModuleGetFunction,C.byref(kernel),module,b"twice")
        value=native.np.array([3.125],dtype=native.np.float64)
        pointer=driver.alloc(value.nbytes)
        resources.callback(native.call,driver.cuMemFree_v2,pointer)
        driver.upload(pointer,value)
        args=(native.P*1)(C.addressof(pointer))
        mark("tiny_kernel_begin")
        native.call(driver.cuLaunchKernel,kernel,1,1,1,1,1,1,0,driver.stream,args,None)
        driver.download(pointer,value)
        mark("tiny_kernel_end",fp64=float(value[0]),driver_version=driver.version)
        require(float(value[0])==6.25,"Tiny FP64 kernel failed")
    mark("teardown_end")
    return {"fp64_observed":6.25,"driver_version":driver.version}


def worker(root,mode):
    mark("affinity_begin",inherited_affinity=affinity(),environment=environment())
    mark("affinity_end",affinity=affinity(4))
    mark("workspace_path_begin")
    sys.path.append(str(V7))
    plan=read(root/"protocol.json")
    psha=digest(root/"protocol.json")
    mark("workspace_path_end")
    mark("driverless_imports_begin")
    from prepared_process_identity import identity_from_pid
    from win32_isolation_monitor import WindowsCensus

    parent=identity_from_pid(os.getppid())
    key=f"{psha}:{parent['pid']}:{parent['birth_filetime_100ns']}:{mode}"
    require(os.environ.get("RFM_PHASE_OWNER")==key and environment()==plan["worker_environment"],
            "Phase worker owner or environment differs")
    mark("driverless_imports_end")
    mark("frozen_diagnostic_import_begin")
    import cuda_ptx_march_diagnostic as frozen_diagnostic
    mark("frozen_diagnostic_import_end")
    mark("stock_solver_import_begin")
    from build_cuda_ptx_artifacts import TOOLKIT, recipe
    from fused_marches import case
    from solver_worker import model_signature
    mark("stock_solver_import_end")
    mark("phase_binding_begin")
    native,tagged_driver,tagged_load,solver=observed_cuda()
    mark("phase_binding_end")
    job=read(FROZEN/"protocol.json")["slots"][0]
    require(job["name"]=="rfm37_k72" and job["arm"]=="uniform" and job["engine"]=="warp_cufft" and
            job["policy"]=="disabled","Wrong frozen stock control")
    module=next(item for item in read(ARCHIVE/"manifest.json")["modules"] if item["manifest"]["key"]==job["artifact_key"])
    mark("source_reconstruction_begin")
    source,spec=recipe("uniform","warp_cufft",job["angular_length"],module["manifest"]["contract"]["compiler"])
    require(spec==module["manifest"]["contract"] and
            sha256(source).hexdigest()==spec["source_sha256"],"Original translation unit was not restored")
    mark("source_reconstruction_end",source_sha256=spec["source_sha256"],source_bytes=len(source))
    output=root/mode
    if mode=="cpu_import_and_source":
        result={"source_sha256":spec["source_sha256"],"source_bytes":len(source),"device_context_created":False,
                "numpy":native.np.__version__,"frozen_entrypoint_imported":frozen_diagnostic.__name__}
    elif mode=="tiny_device_health":
        result=tiny_health(native,tagged_driver,tagged_load,TOOLKIT)
    else:
        mark("solve_call_begin")
        solved=solver(engine=job["engine"],toolkit=TOOLKIT,**job["args"])
        mark("solve_return")
        path=output/"trajectory.npz"
        with path.open("xb") as stream:
            case.store_solution(stream,solved,job["config"])
        reference=WORK/job["original_t9"]["path"]
        require(digest(reference)==job["original_t9"]["sha256"],"Frozen original trajectory changed")
        comparison=frozen_diagnostic.equal_arrays(path,reference)
        require(all(comparison.values()),"The phase-instrumented stock control differs from the original trajectory")
        result=dict(metadata=solved.row,spatial_sha256=model_signature(solved.model,job["family"]),
                    trajectory_sha256=digest(path),original_t9_bitwise=comparison)
        mark("archive_verified")
    census=WindowsCensus()
    save(output/"result.json",dict(result,status="passed",identity=census.times(census.current_process()),protocol_sha256=psha))
    mark("python_worker_end")


def parse_markers(path,pid):
    prefix=b"RFM_PHASE "
    rows=[]
    for line in path.read_bytes().splitlines():
        if not line.startswith(prefix):
            continue
        row=json.loads(line[len(prefix):])
        require(row["pid"]==pid and (not rows or row["monotonic_ns"]>=rows[-1]["monotonic_ns"]),
                "Phase marker PID or clock ordering differs")
        rows.append(row)
    return rows


def run(root):
    sys.path.append(str(V7))
    from gpu_owned_watchdog import run_owned
    from win32_isolation_monitor import WindowsCensus, process_key

    inherited_affinity=affinity()
    affinity(6)
    inherited_env=environment()
    for key in THREADS:
        os.environ[key]="1"
    bound=[Path(__file__).resolve(),HERE/"gpu_owned_watchdog.py",HERE/"win32_isolation_monitor.py",
           HERE/"fused_march_runtime.py",HERE/"fused_marches.py",V7/"cuda_environment_native.py",
           V7/"galerkin_cuda_module_bundle.py",FROZEN/"protocol.json",FROZEN/"summary.json",
           ARCHIVE/"manifest.json",HERE/"cuda_ptx_march_diagnostic.py"]
    frozen=read(FROZEN/"protocol.json")
    frozen_stop=read(FROZEN/"summary.json")
    require(frozen_stop["protocol_sha256"]==digest(FROZEN/"protocol.json") and frozen_stop["status"]=="stopped",
            "Frozen timeout evidence does not match")
    sources={path.relative_to(WORK).as_posix():digest(path) for path in bound}
    sources.update(frozen["sources_sha256"])
    sources[Path(__file__).resolve().relative_to(WORK).as_posix()]=digest(Path(__file__))
    root.mkdir()
    p=root/"protocol.json"
    save(p,dict(status="frozen",scope="Three bounded stages, sequential and stopped on any failure",
                original_protocol_sha256=digest(FROZEN/"protocol.json"),sources_sha256=sources,
                parent_inherited_affinity=inherited_affinity,parent_affinity=affinity(),child_affinity=16,
                inherited_environment=inherited_env,worker_environment=environment(),
                stages=[dict(mode=mode,deadline_seconds=seconds) for mode,seconds in zip(MODES,DEADLINES)]))
    psha=digest(p)
    require(all(digest(WORK/name)==sha for name,sha in sources.items()),"Phase diagnostic source changed")
    census=WindowsCensus()
    parent=census.times(census.current_process())
    outcomes=[]
    for mode,deadline in zip(MODES,DEADLINES):
        output=root/mode
        os.environ["RFM_PHASE_OWNER"]=f"{psha}:{parent['pid']}:{parent['birth_filetime_100ns']}:{mode}"
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker",mode,"--output",str(root)]
        watchdog=run_owned(command,output,deadline_seconds=deadline)
        save(output/"watchdog.json",watchdog)
        markers=parse_markers(output/"stdout.log",watchdog.get("identity",{}).get("pid"))
        complete=watchdog["status"]=="complete" and bool(markers) and markers[-1]["phase"]=="python_worker_end"
        result=read(output/"result.json") if complete else None
        if complete:
            require(result["protocol_sha256"]==psha and process_key(result["identity"])==process_key(watchdog["identity"]),
                    "Recorded worker identity differs")
        outcome=dict(mode=mode,status="passed" if complete else "stopped",watchdog=watchdog,markers=markers,result=result)
        outcomes.append(outcome)
        print(mode,watchdog["status"],markers[-1]["phase"] if markers else "no_python_marker",flush=True)
        if not complete:
            break
    require(digest(p)==psha and all(digest(WORK/name)==sha for name,sha in sources.items()),
            "Phase diagnostic freeze changed")
    passed=len(outcomes)==len(MODES) and all(row["status"]=="passed" for row in outcomes)
    save(root/"summary.json",dict(status="passed" if passed else "stopped",protocol_sha256=psha,rows=outcomes,
                                 unattempted=list(MODES[len(outcomes):]),timing_claim_supported=False))
    return passed


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--worker",choices=MODES,help=argparse.SUPPRESS)
    args=cli.parse_args()
    if args.worker:
        worker(args.output.resolve(),args.worker)
    else:
        raise SystemExit(0 if run(args.output.resolve()) else 2)
