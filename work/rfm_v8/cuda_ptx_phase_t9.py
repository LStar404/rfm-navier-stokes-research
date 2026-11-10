"""Repeat the full frozen PTX T9 matrix with early affinity and phase evidence."""

import os
import sys
import time


def mark(event,**data):
    utc=time.time_ns()
    cpu=os.times()
    value=dict(phase=event,pid=os.getpid(),utc_unix_ns=utc,monotonic_ns=time.perf_counter_ns(),
               cpu_process_ns=time.process_time_ns(),cpu_user_seconds=cpu.user,cpu_kernel_seconds=cpu.system,**data)
    os.write(1,("RFM_PHASE "+__import__("json").dumps(value,separators=(",",":"),allow_nan=False)+"\n").encode("ascii"))


if "--phase-worker" in sys.argv:
    mark("python_entry")
    mark("standard_imports_begin")

import argparse
from pathlib import Path
from types import FunctionType

from cuda_ptx_phase_diagnostic import affinity, clone, environment, observed_cuda, parse_markers

if "--phase-worker" in sys.argv:
    mark("standard_imports_end")


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
V7=WORK/"rfm_v7"
PREVIOUS=HERE/"cuda_ptx_march_t9_output"
HEALTH=HERE/"cuda_ptx_phase_output"
EXPECTED_DRIVER_VERSION=13020
THREADS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")


def observed_solver(arm,policy,settings):
    from cuda_ptx_cache import Nvrtc, artifact, load_device
    from cuda_ptx_march_runtime import ExplicitPtxMarchRuntime
    from fused_marches import solve_combined_fused, solve_fd_fused, solve_uniform_fused

    _,owned,_,uniform=observed_cuda()
    base=uniform.__globals__["FusedMarchRuntime"]

    class CheckedDriver(owned):
        def __init__(self,*args):
            super().__init__(*args)
            if self.version!=EXPECTED_DRIVER_VERSION:
                raise RuntimeError("CUDA driver API version differs from the frozen health check")
            mark("driver_api_verified",driver_version=self.version)

    if policy=="disabled":
        original=base.__init__
        runtime_class=type("TaggedStockRuntime",(base,),{})
        runtime_class.__init__=FunctionType(original.__code__,dict(original.__globals__,_OwnedDriver=CheckedDriver),
                                            original.__name__,original.__defaults__,original.__closure__)
    else:
        class TaggedNvrtc(Nvrtc):
            def compile(self,*args,**kwargs):
                mark("source_compile_begin")
                value=super().compile(*args,**kwargs)
                mark("source_compile_end")
                return value

        def tagged_artifact(*args,**kwargs):
            mark("ptx_artifact_begin")
            value=artifact(*args,**kwargs)
            mark("ptx_artifact_end",mode=value[2]["mode"])
            return value

        def tagged_load(*args,**kwargs):
            mark("module_load_begin")
            value=load_device(*args,**kwargs)
            mark("module_load_end")
            return value

        runtime_init=clone(ExplicitPtxMarchRuntime.__init__,{
            "source,spec=recipe(family,engine,n,compiler_identity)":"source_assembly_begin",
            "recipe_seconds=time.perf_counter()-begin":"source_assembly_end",
            'self.driver_lib=C.WinDLL("nvcuda.dll")':"driver_dll_load_begin",
            "self.driver=_OwnedDriver(self.driver_lib,resources)":"driver_dll_load_end",
            "compiler=Nvrtc(root) if policy==\"empty_first\" else None":"compiler_identity_begin",
            "compiler_identity_seconds=time.perf_counter()-begin if compiler is not None else 0.":"compiler_identity_end",
            "self.kernels={}":"kernel_binding_begin",
            'self.fft=C.WinDLL(str(root/"bin"/"cufft64_10.dll"))':"cufft_dll_load_begin",
            'self.plan_many=bind(self.fft,"cufftPlanMany",[C.POINTER(I),I,C.POINTER(I),P,I,I,P,I,I,I,I])':"cufft_dll_load_end",
            "self.resources=resources.pop_all()":"runtime_setup_end",
        },_OwnedDriver=CheckedDriver,Nvrtc=TaggedNvrtc,artifact=tagged_artifact,load_device=tagged_load)
        runtime_class=type("TaggedArtifactRuntime",(ExplicitPtxMarchRuntime,),{"__init__":runtime_init})

    created=[]

    def runtime(*args):
        value=runtime_class(*args,**settings)
        created.append(value)
        return value

    if arm=="uniform":
        fn=FunctionType(uniform.__code__,dict(uniform.__globals__,FusedMarchRuntime=runtime),
                        uniform.__name__,uniform.__defaults__,uniform.__closure__)
        fn.__kwdefaults__=uniform.__kwdefaults__
    else:
        common={"start = time.perf_counter()":"host_setup_begin",
                "for node in prime:":"prime_begin",
                "if not np.isfinite(checkpoints).all() or not np.isfinite(probes).all():":"teardown_end"}
        combined={
            "setup = time.perf_counter()-start":"host_setup_end",
            "runtime_setup = time.perf_counter()-t":"runtime_return",
            "upload = time.perf_counter()-t":"uploads_end",
            'angular_nodes, prime, _ = nonlinear_nodes(rt,resources,"gal",n,k,q,spectrum,physical,':"plan_begin",
            "plan = time.perf_counter()-t":"prime_end",
            "old_graph = rt.capture(resources, lambda: body(old_advance), old_chunk) if old_chunk else None":"graph_capture_begin",
            "graph = time.perf_counter()-t":"graph_capture_end",
            "executed = enqueue(old_advance, old_graph, old_chunk, before_steps)":"march_begin",
            "march = time.perf_counter()-t":"march_end_after_sync",
            "final = np.empty(1, dtype=np.int32)":"download_begin",
            "download = time.perf_counter()-t":"download_end",
        }
        fd={
            "host_setup = time.perf_counter() - start":"host_setup_end",
            'if cuda_linear_solve == "dense":':"host_linear_precompute_begin",
            "precompute_time = time.perf_counter() - t":"host_linear_precompute_end",
            "runtime_time = time.perf_counter() - t":"runtime_return",
            "upload_time = time.perf_counter() - t":"uploads_end",
            'angular_nodes, prime, angular_names = nonlinear_nodes(rt,resources,"fd",n,modes-1,w,spectrum,physical,':"plan_begin",
            "plan_time = time.perf_counter() - t":"prime_end",
            "executable = rt.capture(resources, body, chunk) if chunk else None":"graph_capture_begin",
            "graph_time = time.perf_counter() - t":"graph_capture_end",
            "if executable is None:":"march_begin",
            "march_time = time.perf_counter() - t":"march_end_after_sync",
            "driver.download(saved, checkpoints)":"download_begin",
            "download_time = time.perf_counter() - t":"download_end",
        }
        original,markers=(solve_combined_fused,combined) if arm=="combined" else (solve_fd_fused,fd)
        fn=clone(original,dict(common,**markers),FusedMarchRuntime=runtime)
    return fn,created


def worker(root,index):
    mark("affinity_begin",inherited_affinity=affinity(),environment=environment())
    mark("affinity_end",affinity=affinity(4))
    mark("workspace_path_begin")
    sys.path.append(str(V7))
    mark("workspace_path_end")
    mark("scientific_imports_begin")
    import cuda_ptx_march_diagnostic as original
    from fused_marches import case
    from prepared_process_identity import identity_from_pid
    mark("scientific_imports_end")
    protocol=root/"protocol.json"
    psha=original.sha(protocol)
    plan=original.read(protocol)
    job=plan["slots"][index]
    original.require(environment()==plan["phase_worker_environment"] and original.affinity_mask()==16 and
        os.environ["RFM_PTX_OWNER"]==original.identity_tag(psha,identity_from_pid(os.getppid()),index),
        "Phase worker differs from the frozen identity or environment")
    policy=job["policy"]
    output=root/f"slot_{index:03d}"
    directory=output/"empty" if policy=="empty_first" else original.ARCHIVE/"artifacts"
    original.require(policy!="empty_first" or not directory.exists(),"Cold namespace is already present")
    settings={} if policy=="disabled" else dict(policy=policy,artifact_directory=directory,
        expected_artifact_sha256=job["artifact_sha256"],compiler_identity=plan["compiler"])
    mark("phase_binding_begin",index=index,policy=policy,engine=job["engine"])
    solve,created=observed_solver(job["arm"],policy,settings)
    mark("phase_binding_end")
    mark("solve_call_begin")
    result=solve(engine=job["engine"],toolkit=original.TOOLKIT,**job["args"])
    returned=time.perf_counter_ns()
    mark("solve_return")
    original.require(len(created)==1,"A phase solve must own one runtime")
    result.row.update(ptx_cache_policy=policy)
    if settings:
        result.row.update(created[0].cache_row)
    signature=original.model_signature(result.model,job["family"])
    raw=output/"trajectory.npz"
    mark("serialization_begin")
    with raw.open("xb") as handle:
        case.store_solution(handle,result,job["config"])
    census=original.WindowsCensus()
    original.save(output/"result.json",dict(status="complete",index=index,protocol_sha256=psha,job=job,metadata=result.row,
         spatial_sha256=signature,trajectory_sha256=original.sha(raw),affinity_mask=original.affinity_mask(),
         solve_return_monotonic_ns=returned,identity=census.times(census.current_process())))
    mark("serialization_end")
    mark("python_worker_end")


def run(root):
    affinity(6)
    for key in THREADS:
        os.environ[key]="1"
    sys.path.append(str(V7))
    import cuda_ptx_march_diagnostic as original

    old=original.read(PREVIOUS/"protocol.json")
    previous=original.read(PREVIOUS/"summary.json")
    health=original.read(HEALTH/"summary.json")
    original.validate(old)
    original.require(previous["status"]=="stopped" and len(previous["unattempted"])==23 and
        previous["rows"][0]["watchdog"]["status"]=="deadline_exceeded" and
        previous["protocol_sha256"]==original.sha(PREVIOUS/"protocol.json") and
        health["status"]=="passed" and health["rows"][1]["result"]["driver_version"]==EXPECTED_DRIVER_VERSION and
        health["protocol_sha256"]==original.sha(HEALTH/"protocol.json"),"Prior timeout and fresh health evidence differ")
    files=(Path(__file__).resolve(),HERE/"cuda_ptx_phase_diagnostic.py",PREVIOUS/"protocol.json",
           PREVIOUS/"summary.json",HEALTH/"protocol.json",HEALTH/"summary.json")
    sources=dict(old["sources_sha256"],**{path.relative_to(WORK).as_posix():original.sha(path) for path in files})
    plan=dict(old,sources_sha256=sources,phase_worker_environment=environment(),phase_expected_driver_api=EXPECTED_DRIVER_VERSION,
              phase_affinity=dict(parent=64,child=16,pin_before_scientific_imports=True),
              timing_scope="Sequential phased fresh processes only; do not mix with original uninstrumented clocks")
    root.mkdir()
    original.save(root/"protocol.json",plan)

    def run_phased(command,output,*,deadline_seconds):
        command=["--phase-worker" if part=="--worker" else part for part in command]
        watchdog=original.run_owned(command,output,deadline_seconds=deadline_seconds)
        identity=watchdog.get("identity",{})
        markers=parse_markers(output/"stdout.log",identity.get("pid"))
        ended=bool(markers) and markers[-1]["phase"]=="python_worker_end"
        driver_version=[row["driver_version"] for row in markers if row["phase"]=="driver_api_verified"]
        if watchdog["status"]=="complete" and (not ended or driver_version!=[EXPECTED_DRIVER_VERSION]):
            watchdog=dict(watchdog,status="phase_incomplete")
        logfile=output/"phase_log.json"
        original.save(logfile,dict(status=watchdog["status"],markers=markers,
            raw_stdout_sha256=original.sha(output/"stdout.log"),driver_api_versions=driver_version))
        return dict(watchdog,phase_log_sha256=original.sha(logfile),phase_markers=len(markers),
                    last_phase=markers[-1]["phase"] if markers else None)

    scoped=FunctionType(original.run.__code__,dict(original.run.__globals__,__file__=__file__,run_owned=run_phased),
                        original.run.__name__,original.run.__defaults__,original.run.__closure__)
    return scoped(root)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--phase-worker",type=int,help=argparse.SUPPRESS)
    args=cli.parse_args()
    if args.phase_worker is None:
        raise SystemExit(0 if run(args.output.resolve()) else 2)
    worker(args.output.resolve(),args.phase_worker)
