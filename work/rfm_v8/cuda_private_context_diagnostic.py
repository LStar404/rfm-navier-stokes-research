"""Audit and qualify a private driver context without editing the primary-context cohorts."""

import os
import sys

if "--health-worker" in sys.argv or "--private-worker" in sys.argv:
    from cuda_ptx_phase_t9 import mark

    mark("private_python_entry")

import argparse
from contextlib import ExitStack
import ctypes as C
from pathlib import Path
import re
from types import FunctionType

import cuda_ptx_phase_t9 as phased
from cuda_ptx_phase_diagnostic import affinity, environment, observed_cuda, parse_markers, require
from cuda_ptx_phase_t9 import HERE, HEALTH, PREVIOUS, THREADS, V7, WORK, mark


PRIMARY_PHASE=HERE/"cuda_ptx_march_t9_phase_output"
MODE="private_cuCtxCreate_v2"
HEALTH_REPEATS=4


def imports():
    sys.path.append(str(V7))
    import cuda_ptx_march_diagnostic as base
    from cuda_private_context import PrivateDriver, SIGNATURES

    return base,PrivateDriver,SIGNATURES


def static_audit(base,driver_type,signatures):
    header=base.TOOLKIT/"include"/"cuda.h"
    lines=header.read_bytes().decode("utf-8")
    patterns=(r"#define\s+cuCtxCreate\s+cuCtxCreate_v2",r"#define\s+cuCtxDestroy\s+cuCtxDestroy_v2",
              r"CUresult CUDAAPI cuCtxCreate\(CUcontext \*pctx, unsigned int flags, CUdevice dev\);",
              r"CUresult CUDAAPI cuCtxDestroy\(CUcontext ctx\);",
              r"CUresult CUDAAPI cuCtxGetCurrent\(CUcontext \*pctx\);",
              r"typedef int CUdevice_v1;",r"typedef CUdevice_v1 CUdevice;",r"typedef struct CUctx_st \*CUcontext;")
    require(all(re.search(pattern,lines) for pattern in patterns),"Installed CUDA header ABI differs")
    require(signatures["cuCtxCreate_v2"]==[C.POINTER(C.c_void_p),C.c_uint,C.c_int] and
            signatures["cuCtxDestroy_v2"]==[C.c_void_p] and signatures["cuCtxGetCurrent"]==[C.POINTER(C.c_void_p)],
            "ctypes CUDA context ABI differs")
    require(not any("PrimaryCtx" in name or "CtxSetCurrent" in name for name in signatures),
            "Unexpected primary-context/current replacement binding")

    def scenario(stream_failure,prior):
        events=[]
        context_stack=[prior] if prior else []
        owned=0x1234
        stream=0x4321
        def output(pointer,value,kind):
            C.cast(pointer,C.POINTER(kind))[0]=value
        def handler(name,*args):
            events.append(name)
            if name=="cuDriverGetVersion":
                output(args[0],phased.EXPECTED_DRIVER_VERSION,C.c_int)
            elif name=="cuCtxGetCurrent":
                output(args[0],context_stack[-1] if context_stack else None,C.c_void_p)
            elif name=="cuCtxCreate_v2":
                require(args[1]==0,"Private context flags differ")
                context_stack.append(owned)
                output(args[0],owned,C.c_void_p)
            elif name=="cuCtxDestroy_v2":
                require(args[0].value==owned and context_stack.pop()==owned,"Destroy did not own current context")
            elif name=="cuStreamCreate":
                if stream_failure:
                    return 719
                require(args[1]==1,"Unexpected stream flags")
                output(args[0],stream,C.c_void_p)
            elif name=="cuStreamDestroy_v2":
                require(context_stack[-1]==owned and args[0].value==stream,"Stream teardown order differs")
            return 0
        def fake(name):
            def function(*args):
                return handler(name,*args)
            function.__name__=name
            return function
        library=type("FakeLibrary",(),{})()
        for name in signatures:
            setattr(library,name,fake(name))
        failure_seen=False
        try:
            with ExitStack() as resources:
                instance=driver_type(library,resources)
                instance.assert_current()
        except RuntimeError as error:
            require(stream_failure and str(error)=="cuStreamCreate: status 719","Unexpected mocked failure")
            failure_seen=True
        require(failure_seen==stream_failure and context_stack==([prior] if prior else []) and
                events.count("cuCtxCreate_v2")==events.count("cuCtxDestroy_v2")==1,
                "Mocked CUDA ownership leaked or restored the wrong stack")
        require(events.count("cuStreamDestroy_v2")==int(not stream_failure),"Mocked stream ownership differs")
        if not stream_failure:
            require(events.index("cuStreamDestroy_v2")<events.index("cuCtxDestroy_v2"),"Destroy ordering differs")
        return dict(injected_stream_failure=stream_failure,predecessor_present=prior is not None,events=events)

    return dict(status="passed",scope="Local headers and fake call-order fixture, no CUDA device call",
                cuda_header_sha256=base.sha(header),cases=[scenario(f,p) for f in (False,True) for p in (None,0xAAAA)])


def health_worker(root,index):
    mark("affinity_begin",inherited_affinity=affinity(),environment=environment())
    mark("affinity_end",affinity=affinity(4))
    mark("scientific_imports_begin")
    base,driver_type,_=imports()
    from cuda_environment_native import DP, I, P, bind, call, np
    from galerkin_cuda_native import Runtime
    from prepared_process_identity import identity_from_pid
    mark("scientific_imports_end")
    plan=base.read(root/"protocol.json")
    psha=base.sha(root/"protocol.json")
    require(plan["context_mode"]==MODE and environment()==plan["worker_environment"] and
            os.environ["RFM_PRIVATE_HEALTH_OWNER"]==base.identity_tag(psha,identity_from_pid(os.getppid()),index),
            "Private health owner or environment differs")
    _,_,load,_=observed_cuda()
    source=b'extern "C" __global__ void twice(double *x) { *x *= 2.0; }'
    with ExitStack() as resources:
        resources.enter_context(os.add_dll_directory(str(base.TOOLKIT/"bin")))
        mark("driver_dll_load_begin")
        driver=driver_type(C.WinDLL("nvcuda.dll"),resources,mark=mark)
        require(driver.version==phased.EXPECTED_DRIVER_VERSION,"Driver API version differs")
        mark("driver_api_verified",driver_version=driver.version)
        module=load(base.TOOLKIT,driver,source,b"rfm_private_context_health.cu")
        resources.callback(call,driver.cuModuleUnload,module)
        kernel=P()
        call(driver.cuModuleGetFunction,C.byref(kernel),module,b"twice")
        scalar=np.array([3.125],dtype=np.float64)
        pointer=driver.alloc(scalar.nbytes)
        resources.callback(call,driver.cuMemFree_v2,pointer)
        driver.upload(pointer,scalar)
        args=(P*1)(C.addressof(pointer))
        mark("fp64_kernel_begin")
        call(driver.cuLaunchKernel,kernel,1,1,1,1,1,1,0,driver.stream,args,None)
        driver.download(pointer,scalar)
        mark("fp64_kernel_end",observed=float(scalar[0]))
        require(float(scalar[0])==6.25,"FP64 kernel failed in the private context")
        n,batch=15,2
        x=((np.arange(n*batch,dtype=np.float64)%11)-5).reshape(batch,n)/8
        frequencies=np.empty((batch,n//2+1),dtype=np.complex128)
        reconstructed=np.empty_like(x)
        pointers=[]
        for host in (x,frequencies,reconstructed):
            address=driver.alloc(host.nbytes)
            resources.callback(call,driver.cuMemFree_v2,address)
            pointers.append(address)
        driver.upload(pointers[0],x)
        rt=Runtime.__new__(Runtime)
        rt.driver=driver
        mark("cufft_dll_load_begin")
        rt.fft=C.WinDLL(str(base.TOOLKIT/"bin"/"cufft64_10.dll"))
        mark("cufft_dll_load_end")
        rt.plan_many=bind(rt.fft,"cufftPlanMany",[C.POINTER(I),I,C.POINTER(I),P,I,I,P,I,I,I,I])
        rt.set_stream=bind(rt.fft,"cufftSetStream",[I,P])
        rt.d2z=bind(rt.fft,"cufftExecD2Z",[I,DP,DP])
        rt.z2d=bind(rt.fft,"cufftExecZ2D",[I,DP,DP])
        rt.destroy_plan=bind(rt.fft,"cufftDestroy",[I])
        mark("cufft_plan_begin")
        forward=rt.plan(resources,n,batch,False)
        inverse=rt.plan(resources,n,batch,True)
        driver.assert_current()
        mark("cufft_plan_end")

        def transforms():
            call(rt.d2z,forward,pointers[0],pointers[1])
            call(rt.z2d,inverse,pointers[1],pointers[2])

        mark("cufft_prime_begin")
        transforms()
        rt.sync()
        mark("cufft_prime_end")
        mark("cufft_graph_capture_begin")
        graph=rt.capture(resources,transforms,1)
        mark("cufft_graph_capture_end")
        call(driver.cuGraphLaunch,graph,driver.stream)
        driver.download(pointers[1],frequencies)
        driver.download(pointers[2],reconstructed)
        driver.assert_current()
        spectral_error=float(np.max(np.abs(frequencies-np.fft.rfft(x))))
        physical_error=float(np.max(np.abs(reconstructed-n*x)))
        mark("cufft_graph_verified",spectral_error=spectral_error,physical_error=physical_error)
        require(spectral_error<2e-13 and physical_error<2e-13,"Private-context cuFFT oracle failed")
    require(driver.current()==driver.previous,"Prior context was not restored")
    census=base.WindowsCensus()
    base.save(root/f"health_{index:02d}"/"result.json",dict(status="passed",index=index,protocol_sha256=psha,
        identity=census.times(census.current_process()),context_mode=MODE,driver_api=driver.version,
        fp64=float(scalar[0]),cufft_spectral_max_abs=spectral_error,cufft_round_trip_max_abs=physical_error,
        cleanup_verified=True))
    mark("python_worker_end")


def setup_parent():
    affinity(6)
    for key in THREADS:
        os.environ[key]="1"
    return imports()


def read_phases(base,output,watchdog,*,tag):
    markers=parse_markers(output/"stdout.log",watchdog.get("identity",{}).get("pid"))
    phase=[row["phase"] for row in markers]
    versions=[row["driver_version"] for row in markers if row["phase"]=="driver_api_verified"]
    valid=bool(phase) and phase[-1]=="python_worker_end" and phase.count("private_context_create_end")==1 and \
        phase.count("private_context_destroy_end")==1 and "context_retain_begin" not in phase and \
        versions==[phased.EXPECTED_DRIVER_VERSION]
    if watchdog["status"]=="complete" and not valid:
        watchdog=dict(watchdog,status="phase_incomplete")
    path=output/"phase_log.json"
    base.save(path,dict(status=watchdog["status"],markers=markers,raw_stdout_sha256=base.sha(output/"stdout.log"),kind=tag))
    return dict(watchdog,phase_log_sha256=base.sha(path),last_phase=phase[-1] if phase else None)


def health(root):
    base,driver_type,signatures=setup_parent()
    audit=static_audit(base,driver_type,signatures)
    previous=base.read(PRIMARY_PHASE/"summary.json")
    require(previous["status"]=="stopped" and previous["rows"][-1]["watchdog"]["last_phase"]=="context_retain_begin",
            "The instrumented primary-context failure is not registered")
    root.mkdir()
    base.save(root/"static_abi.json",audit)
    sources=source_hashes(base)
    sources.update({path.relative_to(WORK).as_posix():base.sha(path) for path in
                   (PRIMARY_PHASE/"protocol.json",PRIMARY_PHASE/"summary.json")})
    plan=dict(status="frozen",context_mode=MODE,worker_environment=environment(),sources_sha256=sources,
              abi_audit_sha256=base.sha(root/"static_abi.json"),driver_dll_sha256=base.sha(base.DRIVER),
              expected_driver_api=phased.EXPECTED_DRIVER_VERSION,health_repeats=HEALTH_REPEATS,deadline_seconds=20)
    protocol=root/"protocol.json"
    base.save(protocol,plan)
    psha=base.sha(protocol)
    census=base.WindowsCensus()
    parent=census.times(census.current_process())
    rows=[]
    for index in range(HEALTH_REPEATS):
        output=root/f"health_{index:02d}"
        os.environ["RFM_PRIVATE_HEALTH_OWNER"]=base.identity_tag(psha,parent,index)
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--health-worker",str(index),"--output",str(root)]
        watchdog=read_phases(base,output,base.run_owned(command,output,deadline_seconds=20),tag="private_health")
        base.save(output/"watchdog.json",watchdog)
        success=watchdog["status"]=="complete"
        result=base.read(output/"result.json") if success else None
        if success:
            require(result["protocol_sha256"]==psha and result["index"]==index and
                    base.process_key(result["identity"])==base.process_key(watchdog["identity"]),
                    "Private health result identity differs")
        rows.append(dict(index=index,watchdog=watchdog,result=result))
        print("private health",index,watchdog["status"],watchdog["last_phase"],flush=True)
        if not success:
            break
    require(base.sha(protocol)==psha and base.sha(base.DRIVER)==plan["driver_dll_sha256"] and
            all(base.sha(WORK/name)==value for name,value in sources.items()),"Private health inputs changed")
    success=len(rows)==HEALTH_REPEATS and all(row["watchdog"]["status"]=="complete" for row in rows)
    base.save(root/"summary.json",dict(status="passed" if success else "stopped",protocol_sha256=psha,
        context_mode=MODE,rows=rows,unattempted=list(range(len(rows),HEALTH_REPEATS)),timing_claim_supported=False))
    return success


def source_hashes(base):
    return {path.relative_to(WORK).as_posix():base.sha(path) for path in (
        Path(__file__).resolve(),HERE/"cuda_private_context.py",HERE/"cuda_ptx_phase_t9.py",
        HERE/"cuda_ptx_phase_diagnostic.py",HERE/"cuda_ptx_march_diagnostic.py",HERE/"gpu_owned_watchdog.py",
        HERE/"win32_isolation_monitor.py",V7/"cuda_environment_native.py",
        V7/"galerkin_cuda_module_bundle.py",V7/"galerkin_cuda_native.py")}


def private_worker(root,index):
    def private_observed_cuda():
        _,driver_type,_=imports()
        native,_,loader,solver=observed_cuda()
        class TaggedPrivateDriver(driver_type):
            def __init__(self,*args):
                super().__init__(*args,mark=mark)
        return native,TaggedPrivateDriver,loader,solver

    old=phased.observed_solver
    adapted=FunctionType(old.__code__,dict(old.__globals__,observed_cuda=private_observed_cuda),
                         old.__name__,old.__defaults__,old.__closure__)
    old=phased.worker
    worker=FunctionType(old.__code__,dict(old.__globals__,observed_solver=adapted),
                        old.__name__,old.__defaults__,old.__closure__)
    return worker(root,index)


def qualify(root,health_root):
    base,_,_=setup_parent()
    archive=base.read(health_root/"summary.json")
    hplan=base.read(health_root/"protocol.json")
    require(archive["status"]=="passed" and archive["context_mode"]==MODE and len(archive["rows"])==HEALTH_REPEATS and
            all(row["result"]["cleanup_verified"] and row["result"]["driver_api"]==phased.EXPECTED_DRIVER_VERSION
                for row in archive["rows"]) and archive["protocol_sha256"]==base.sha(health_root/"protocol.json") and
            hplan["driver_dll_sha256"]==base.sha(base.DRIVER) and
            all(base.sha(WORK/name)==value for name,value in hplan["sources_sha256"].items()),
            "All four private-context health checks must pass before T9")
    old=base.read(PREVIOUS/"protocol.json")
    base.validate(old)
    files=(health_root/"protocol.json",health_root/"summary.json",health_root/"static_abi.json",
           PRIMARY_PHASE/"protocol.json",PRIMARY_PHASE/"summary.json",PREVIOUS/"protocol.json",PREVIOUS/"summary.json")
    sources=dict(old["sources_sha256"],**source_hashes(base))
    sources.update({path.relative_to(WORK).as_posix():base.sha(path) for path in files})
    plan=dict(old,sources_sha256=sources,phase_worker_environment=environment(),
              phase_expected_driver_api=phased.EXPECTED_DRIVER_VERSION,
              phase_affinity=dict(parent=64,child=16,pin_before_scientific_imports=True),
              context_mode=MODE,health_protocol_sha256=archive["protocol_sha256"],
              timing_scope="Sequential private-context phased fresh processes only; primary-context clocks are separate")
    root.mkdir()
    base.save(root/"protocol.json",plan)

    def private_guard(command,output,*,deadline_seconds):
        command=["--private-worker" if part=="--worker" else part for part in command]
        watchdog=base.run_owned(command,output,deadline_seconds=deadline_seconds)
        return read_phases(base,output,watchdog,tag="private_t9")

    scope=FunctionType(base.run.__code__,dict(base.run.__globals__,__file__=__file__,run_owned=private_guard),
                       base.run.__name__,base.run.__defaults__,base.run.__closure__)
    return scope(root)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--static",action="store_true")
    action.add_argument("--health",action="store_true")
    action.add_argument("--qualify",action="store_true")
    action.add_argument("--health-worker",type=int,help=argparse.SUPPRESS)
    action.add_argument("--private-worker",type=int,help=argparse.SUPPRESS)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--health-root",type=Path)
    args=cli.parse_args()
    root=args.output.resolve()
    if args.static:
        base,driver_type,signatures=setup_parent()
        base.save(root,static_audit(base,driver_type,signatures))
        print("private context static ABI and lifecycle passed",flush=True)
    elif args.health:
        raise SystemExit(0 if health(root) else 2)
    elif args.qualify:
        require(args.health_root is not None,"Provide the independent health directory")
        raise SystemExit(0 if qualify(root,args.health_root.resolve()) else 2)
    elif args.health_worker is not None:
        health_worker(root,args.health_worker)
    else:
        private_worker(root,args.private_worker)
