"""Run one legacy CUDA probe per owned subprocess under a hard Windows watchdog."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys
import time

from fresh_worker_diagnostic import pin
from gpu_owned_watchdog import run_owned
from solver_worker import affinity_mask
from win32_isolation_monitor import WindowsCensus, process_key


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
TOOLKIT=Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
LAYOUTS=(("gal",200,64,60),("gal",225,72,66),("gal",243,80,76),
         ("fd",200,64,60),("fd",225,72,66),("fd",243,80,175),("fd",243,80,183))
REGISTERED_MICRO=(1,2,5,6)


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def save(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def health():
    import ctypes as C
    from cuda_environment_native import Driver, I, P, bind, call, check_kernel

    with os.add_dll_directory(str(TOOLKIT/"bin")):
        driver=Driver()
        try:
            result=check_kernel(TOOLKIT,driver)
            require(result=={"kernel_max_abs":0.,"graph_max_abs":0.},"Original CUDA FP64 eager/graph health check differed")
            return dict(result,driver_version=driver.version,toolchain="CUDA 11.8",eager_graph_exact=True)
        finally:
            dll=C.WinDLL("nvcuda.dll")
            try:
                call(driver.cuStreamSynchronize,driver.stream)
            finally:
                try:
                    call(bind(dll,"cuStreamDestroy_v2",[P]),driver.stream)
                finally:
                    device=I()
                    call(driver.cuDeviceGet,C.byref(device),0)
                    call(bind(dll,"cuDevicePrimaryCtxRelease_v2",[I]),device)


def micro(rt,family,n,k,rows):
    """Matched diagnostic CUDA-event samples with both graph boundaries inside capture."""
    from contextlib import ExitStack
    import ctypes as C
    from statistics import median

    import numpy as np
    from cuda_environment_native import P, U, bind, call
    from fused_nonlinear_cuda_runtime import FusedNonlinearCUDA
    from verify_fused_nonlinear_cuda import download, stock_node
    from verify_fused_nonlinear_fft import make_spectra

    rng=np.random.default_rng(661229)
    fd=family=="fd"
    inputs,outputs=(4,1) if fd else (3,2)
    spectral=make_spectra(rng,inputs,rows,k,n,"smooth")
    initial=np.zeros((outputs,rows,n//2+1),dtype=np.complex128)
    terms=rng.normal(size=(rows,k+1)) if fd else None
    evaluations,repeats=64,9
    with ExitStack() as stack:
        step=rt.upload(stack,np.array([1],np.int32))
        device_terms=rt.upload(stack,terms) if fd else None
        operations,graph,events,inputs_on_device={},{},{},{}
        create=bind(rt.driver_lib,"cuEventCreate",[C.POINTER(P),U])
        destroy=bind(rt.driver_lib,"cuEventDestroy_v2",[P])
        record=bind(rt.driver_lib,"cuEventRecord",[P,P])
        external=bind(rt.driver_lib,"cuEventRecordWithFlags",[P,P,U])
        elapsed=bind(rt.driver_lib,"cuEventElapsedTime",[C.POINTER(C.c_float),P,P])
        for variant in ("stock","fused"):
            spectrum=rt.upload(stack,spectral)
            inputs_on_device[variant]=spectrum
            flux=rt.upload(stack,initial)
            history=rt.upload(stack,initial)
            mean=rt.upload(stack,np.zeros((2,rows))) if fd else None
            if variant=="stock":
                operation=stock_node(rt,stack,family,n,rows,k,flux,history,mean)(spectrum,device_terms,step)
            else:
                fused=FusedNonlinearCUDA(rt,stack,n=n,k=k,rows=rows,family=family,toolkit=TOOLKIT)
                operation=fused.launch(spectrum,flux,history,mean_terms=device_terms,step=step,mean=mean)
            operations[variant]=operation
            operation()
            require(download(rt,spectrum,spectral).tobytes()==spectral.tobytes(),"Repeated microprobe would mutate its fixed input spectrum")
            markers=(P(),P())
            for marker in markers:
                call(create,C.byref(marker),0)
                stack.callback(call,destroy,marker)
            events[variant]=markers

            def inside_graph(operation=operation,markers=markers):
                call(external,markers[0],rt.driver.stream,1)
                for _ in range(evaluations):
                    operation()
                call(external,markers[1],rt.driver.stream,1)

            graph[variant]=rt.capture(stack,inside_graph,1)
            call(rt.driver.cuGraphLaunch,graph[variant],rt.driver.stream)
            rt.sync()

        samples={mode:{variant:[] for variant in operations} for mode in ("eager","graph")}
        for repeat in range(repeats):
            for mode in ("eager","graph"):
                variants=("stock","fused") if repeat%2==0 else ("fused","stock")
                for variant in variants:
                    start,end=events[variant]
                    if mode=="graph":
                        call(rt.driver.cuGraphLaunch,graph[variant],rt.driver.stream)
                    else:
                        call(record,start,rt.driver.stream)
                        for _ in range(evaluations):
                            operations[variant]()
                        call(record,end,rt.driver.stream)
                    rt.sync()
                    duration=C.c_float()
                    call(elapsed,C.byref(duration),start,end)
                    require(duration.value>0,"CUDA event span is nonpositive")
                    samples[mode][variant].append(duration.value*1000/evaluations)
        for address in inputs_on_device.values():
            require(download(rt,address,spectral).tobytes()==spectral.tobytes(),"Repeated microprobe changed its fixed data")
    return dict(family=family,ntheta=n,k=k,radial_rows=rows,evaluations_per_sample=evaluations,repeats=repeats,
                units="microseconds per nonlinear evaluation",logical_field_transforms_per_row=5,
                same_input_and_evaluation_counts=True,stock_nodes_per_evaluation=3,fused_nodes_per_evaluation=1,
                scope="CUDA events after setup. Graph events are inside the graph. Eager events may include host issue gaps.",
                samples=samples,medians={mode:{name:median(values) for name,values in arms.items()} for mode,arms in samples.items()},
                stock_over_fused_paired={mode:[a/b for a,b in zip(arms["stock"],arms["fused"],strict=True)] for mode,arms in samples.items()})


def child(kind,index,output):
    pin(4)
    census=WindowsCensus()
    identity=census.times(census.current_process())
    ready=dict(affinity_mask=affinity_mask(),identity=identity)
    require(ready["affinity_mask"]==16,"CUDA child is not pinned to logical CPU 4")
    save(output/"child_ready.json",ready)
    if kind=="hang":
        time.sleep(60)
        raise RuntimeError("Watchdog failed to stop the owned fixture")
    if kind=="trivial":
        result=dict(passed=True)
    elif kind=="health":
        result=health()
    else:
        import numpy as np
        from verify_fused_nonlinear_cuda import BundledSwitchRuntime, FDRuntime, one_case
        family,n,k,rows=LAYOUTS[index]
        runtime=BundledSwitchRuntime if family=="gal" else FDRuntime
        with runtime(TOOLKIT) as rt:
            result=(one_case(rt,family,n,k,rows,TOOLKIT,np.random.default_rng(287835+index))
                    if kind=="fixed" else micro(rt,family,n,k,rows))
    save(output/"result.json",dict(status="passed",kind=kind,layout_index=index,**result))


def owned_probe(root,kind,index=0,deadline=30):
    output=root/(kind if kind in ("hang","trivial","health") else f"{kind}_{index:02d}")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child",kind,"--index",str(index),"--output",str(output)]
    monitor=run_owned(command,output,deadline_seconds=deadline)
    ready=output/"child_ready.json"
    result=output/"result.json"
    try:
        if ready.is_file():
            facts=json.loads(ready.read_bytes())
            require(facts["affinity_mask"]==16 and process_key(facts["identity"])==process_key(monitor["identity"]),
                    "Exact watchdog and child startup identities differ")
            monitor["ready_sha256"]=sha(ready)
        if result.is_file():
            monitor["result_sha256"]=sha(result)
    except Exception as error:
        monitor.update(status="invalid_probe_evidence",evidence_failure_type=type(error).__name__)
    save(output/"watchdog.json",monitor)
    return monitor


def audit(root):
    root.mkdir()
    pin(6)
    require(affinity_mask()==64,"Watchdog is not pinned to logical CPU 6")
    normal=owned_probe(root,"trivial",deadline=4)
    hung=owned_probe(root,"hang",deadline=1.25)
    require(normal["status"]=="complete" and normal["exit_code"]==0 and hung["status"]=="deadline_exceeded" and
            hung["terminate_process_ok"] and hung["post_terminate_wait_status"]==0 and hung["exit_code"]==771 and
            process_key(hung["identity"])==process_key(hung["terminal_identity"]) and
            "ready_sha256" in hung and "result_sha256" not in hung and hung["wall_seconds"]<=5.25,
            "Owned hard-wall-clock termination fixture failed")
    result=dict(status="passed",scope="No CUDA or GPU access",normal=normal,hung=hung,
                parent_affinity_mask=affinity_mask(),sources_sha256={name:sha(HERE/name) for name in
                ("run_fused_device_isolated.py","gpu_owned_watchdog.py")})
    save(root/"certificate.json",result)
    return result


def device(root,watchdog_audit):
    certificate=json.loads(watchdog_audit.read_bytes())
    require(certificate["status"]=="passed" and all(sha(HERE/name)==value for name,value in certificate["sources_sha256"].items()),
            "The exact watchdog must pass its no-GPU fixture before opening a context")
    root.mkdir()
    pin(6)
    require(affinity_mask()==64,"Watchdog is not pinned to logical CPU 6")
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name]="1"
    paths=(Path(__file__),HERE/"gpu_owned_watchdog.py",HERE/"verify_fused_nonlinear_cuda.py",
           HERE/"fused_nonlinear_cuda_runtime.py",HERE/"fused_nonlinear_fft.cu",HERE/"fused_nonlinear_fft.py",
           HERE/"verify_fused_nonlinear_fft.py",WORK/"rfm_v7"/"cuda_environment_native.py",
           WORK/"rfm_v7"/"galerkin_cuda_native.cu",WORK/"rfm_v7"/"fd_cuda_native.cu",
           WORK/"rfm_v7"/"galerkin_cuda_module_bundle.py",WORK/"rfm_v7"/"fd_cuda_native.py")
    plan=[("health",0,15),*(("fixed",i,30) for i in range(len(LAYOUTS))),*(("micro",i,30) for i in REGISTERED_MICRO)]
    save(root/"protocol.json",dict(layouts=LAYOUTS,attempts=plan,scope="Old-runtime health and fixed device diagnostics only",
        sources_sha256={path.relative_to(WORK).as_posix():sha(path) for path in paths},watchdog_audit_sha256=sha(watchdog_audit)))
    rows=[]
    for kind,index,deadline in plan:
        row=owned_probe(root,kind,index,deadline)
        rows.append(dict(kind=kind,layout_index=index,**row))
        print(kind,index,row["status"],flush=True)
        if row["status"]!="complete" or "result_sha256" not in row:
            break
    complete=len(rows)==len(plan) and all(row["status"]=="complete" and "result_sha256" in row for row in rows)
    result=dict(status="passed" if complete else "stopped_on_first_anomaly",attempts=rows,
                protocol_sha256=sha(root/"protocol.json"),no_driver_reset=True,formal_solves_run=0)
    save(root/"summary.json",result)
    return result


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    actions=cli.add_mutually_exclusive_group(required=True)
    actions.add_argument("--watchdog-audit",action="store_true")
    actions.add_argument("--device",type=Path,help="Passed no-GPU watchdog certificate")
    actions.add_argument("--child",choices=("hang","trivial","health","fixed","micro"),help=argparse.SUPPRESS)
    cli.add_argument("--index",type=int,default=0)
    args=cli.parse_args()
    if args.child:
        child(args.child,args.index,args.output)
    else:
        result=audit(args.output) if args.watchdog_audit else device(args.output,args.device)
        print(result["status"])
        raise SystemExit(result["status"]!="passed")
