"""Run bounded fresh-process CUDA PTX diagnostics with unchanged numerical gates."""

import argparse
import os
from pathlib import Path
import sys
import time

from build_cuda_ptx_artifacts import hits, recipe
from cuda_ptx_cache import key_for
from fused_march_qualification import (
    ENGINES, HERE, THIN, TOOLKIT, WORK, assessment, build, read, record, require, save, sha,
)
from fresh_worker_diagnostic import pin
from gpu_owned_watchdog import run_owned
from solver_worker import THREADS, affinity_mask, model_signature
from win32_isolation_monitor import WindowsCensus, process_key


ARCHIVE=HERE/"cuda_ptx_artifact_output"
ORIGINAL=HERE/"fused_march_t9_output"
DRIVER=Path(os.environ["SystemRoot"])/"System32"/"nvcuda.dll"
POLICIES=("disabled","strict_hit")
ENVIRONMENT=("CUDA_CACHE_DISABLE","CUDA_CACHE_PATH","CUDA_CACHE_MAXSIZE","CUDA_FORCE_PTX_JIT","CUDA_MODULE_LOADING")
ARRAYS=("times","states","probe_times","transverse","vorticity","config")


def identity_tag(protocol_hash,parent,index):
    return f"{protocol_hash}:{parent['pid']}:{parent['birth_filetime_100ns']}:{index}"


def env_policy():
    return {name:("default" if name not in os.environ else
                  "nondefault_path" if name=="CUDA_CACHE_PATH" else os.environ[name]) for name in ENVIRONMENT}


def freeze(stage,prior,root):
    from scipy.fft import next_fast_len

    archive=read(ARCHIVE/"manifest.json")
    require(hits(ARCHIVE)["status"]=="passed","Prebuilt artifacts failed strict verification")
    artifacts={row["manifest"]["key"]:row for row in archive["modules"]}
    template=build(stage,ORIGINAL if stage=="t72" else None)
    if stage=="t72":
        require(prior is not None,"T72 requires the registered T9 promotion result")
        decision=read(prior/"summary.json")
        require(decision["promotion_eligible"] and decision["protocol_sha256"]==sha(prior/"protocol.json"),
                "T9 did not permit promotion")
    old=read(ORIGINAL/"protocol.json")
    old_summ=read(ORIGINAL/"summary.json")
    require(old_summ["protocol_sha256"]==sha(ORIGINAL/"protocol.json") and old_summ["status"]=="passed",
            "Original T9 accuracy archive did not pass")
    def tag(slot):
        return tuple(slot[name] for name in ("name","arm","seed","engine"))

    prior_raw={tag(slot):record(ORIGINAL/f"slot_{i:03d}"/"trajectory.npz",
                               read(ORIGINAL/f"slot_{i:03d}"/"result.json")["trajectory_sha256"])
               for i,slot in enumerate(old["slots"])} if stage=="t9" else {}
    sources=dict(template["sources_sha256"])
    files=[Path(__file__),HERE/"cuda_ptx_cache.py",HERE/"cuda_ptx_march_runtime.py",
           HERE/"build_cuda_ptx_artifacts.py",HERE/"fused_march_qualification.py",
           ARCHIVE/"manifest.json",ARCHIVE/"fresh_cpu_hit.json",ARCHIVE/"rebuild_verification.json",
           ORIGINAL/"protocol.json",ORIGINAL/"summary.json"]
    if prior is not None:
        files.extend((prior/"protocol.json",prior/"summary.json"))
    sources.update({path.relative_to(WORK).as_posix():sha(path) for path in files})
    sources.update({item["path"]:item["sha256"] for item in prior_raw.values()})
    for row in artifacts.values():
        artifact_path=ARCHIVE/"artifacts"/row["cpu_cold_build"]["artifact"]
        sources[artifact_path.relative_to(WORK).as_posix()]=row["cpu_cold_build"]["artifact_sha256"]
    def register(slot,policy):
        n=next_fast_len(3*slot["args"]["k"]+1,real=True)
        _,spec=recipe(slot["arm"],slot["engine"],n,archive["compiler"])
        key=key_for(spec)
        expected=artifacts[key]["cpu_cold_build"]["artifact_sha256"]
        return dict(slot,policy=policy,artifact_key=key,artifact_sha256=expected,angular_length=n,
                    original_t9=prior_raw.get(tag(slot)))

    slots=[]
    for index,slot in enumerate(template["slots"]):
        order=POLICIES if index%2==0 else POLICIES[::-1]
        slots.extend(register(slot,policy) for policy in order)
    if stage=="t9":
        cold={}
        for slot in slots:
            cold.setdefault(slot["artifact_key"],dict(slot,policy="empty_first"))
        from fused_march_qualification import thin_jobs
        for name,arm in (("rfm37_k72","combined"),("rfm41_k80","uniform")):
            job=thin_jobs()[0][name,arm,7331]
            args=dict(job["args"],horizon=9)
            times=template["physical_times"]
            if arm=="combined":
                times=sorted(times+[8+i/args["late_rate"] for i in (1,2)])
            else:
                args["steps"]=args["steps"]//72*9
            slot=register(dict(job,args=dict(args,snapshots=times),engine="warp_fused",pair=None),"empty_first")
            cold.setdefault(slot["artifact_key"],slot)
        require(set(cold)==set(artifacts),"T9 needs one fresh empty-cache solve per registered module key")
        slots.extend(cold.values())
    for slot in slots:
        for item in (*slot["control"].values(),*slot["anchor"].values()):
            sources[item["path"]]=item["sha256"]
    protocol=dict(template,slots=slots,sources_sha256=sources,compiler=archive["compiler"],
        driver_dll_sha256=sha(DRIVER),driver_environment=env_policy(),
        stage=stage,experiment="explicit_ptx_march",timing_scope="Sequential bounded diagnostic; OS/driver cache is uncontrolled.",
        promotion_rule=("All runs meet existing numerical gates; strict-hit fused solver and fresh clocks "
                        "each show majority faster and no >10% regression against strict-hit stock; "
                        "every core strict-hit solve is faster than its disabled solve on both clocks."))
    validate(protocol)
    root.mkdir()
    save(root/"protocol.json",protocol)


def validate(plan):
    require(plan["status"]=="frozen" and all(sha(WORK/name)==value for name,value in plan["sources_sha256"].items()) and
            all(sha(TOOLKIT/"bin"/name)==value for name,value in plan["toolkit_sha256"].items()) and
            sha(DRIVER)==plan["driver_dll_sha256"] and env_policy()==plan["driver_environment"],"PTX diagnostic inputs changed")


def worker(root,index):
    from cuda_ptx_march_runtime import solve_with_ptx_policy
    from fused_marches import case
    from prepared_process_identity import identity_from_pid

    pin(4)
    protocol=root/"protocol.json"
    protocol_hash=sha(protocol)
    require(os.environ["RFM_PTX_OWNER"]==identity_tag(protocol_hash,identity_from_pid(os.getppid()),index) and
            affinity_mask()==16 and all(os.environ[name]=="1" for name in THREADS),"Unregistered PTX worker launch")
    plan=read(protocol)
    job=plan["slots"][index]
    output=root/f"slot_{index:03d}"
    policy=job["policy"]
    directory=output/"empty" if policy=="empty_first" else ARCHIVE/"artifacts"
    require(policy!="empty_first" or not directory.exists(),"A first solve needs a fresh empty namespace")
    result=solve_with_ptx_policy(job["arm"],engine=job["engine"],policy=policy,artifact_directory=directory,
        expected_artifact_sha256=job["artifact_sha256"],compiler_identity=plan["compiler"],toolkit=TOOLKIT,**job["args"])
    returned=time.perf_counter_ns()
    signature=model_signature(result.model,job["family"])
    raw=output/"trajectory.npz"
    with raw.open("xb") as handle:
        case.store_solution(handle,result,job["config"])
    census=WindowsCensus()
    save(output/"result.json",dict(status="complete",index=index,protocol_sha256=protocol_hash,job=job,metadata=result.row,
         spatial_sha256=signature,trajectory_sha256=sha(raw),affinity_mask=affinity_mask(),solve_return_monotonic_ns=returned,
         identity=census.times(census.current_process())))


def equal_arrays(a,b):
    import numpy as np

    with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
        return {name:left[name].dtype==right[name].dtype and left[name].shape==right[name].shape and
                left[name].tobytes()==right[name].tobytes() for name in ARRAYS}


def run(root):
    protocol=root/"protocol.json"
    psha=sha(protocol)
    plan=read(protocol)
    validate(plan)
    require(not (root/"execution.json").exists(),"A diagnostic may only run once")
    pin(6)
    census=WindowsCensus()
    parent=census.times(census.current_process())
    save(root/"execution.json",dict(status="started",protocol_sha256=psha,identity=parent))
    for name in THREADS:
        os.environ[name]="1"
    evaluate,pair_evaluate=assessment(root,plan)
    outcomes=[]
    for index,job in enumerate(plan["slots"]):
        output=root/f"slot_{index:03d}"
        os.environ["RFM_PTX_OWNER"]=identity_tag(psha,parent,index)
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(index),"--output",str(root)]
        started=time.perf_counter_ns()
        watchdog=run_owned(command,output,deadline_seconds=plan["hard_deadline_seconds"])
        save(output/"watchdog.json",watchdog)
        sidecar=dict(index=index,job=job,protocol_sha256=psha,watchdog=watchdog,qualified=False)
        if watchdog["status"]=="complete":
            try:
                result=read(output/"result.json")
                require(result["index"]==index and result["job"]==job and result["protocol_sha256"]==psha and
                        process_key(result["identity"])==process_key(watchdog["identity"]) and result["affinity_mask"]==16,
                        "PTX worker identity or frozen slot differs")
                outer=(result["solve_return_monotonic_ns"]-started)/1e9
                row=result["metadata"]
                require(row["ptx_cache_policy"]==job["policy"] and 0<row["seconds"]<=outer<=watchdog["wall_seconds"]+.02,
                        "Return clocks or PTX policy differ")
                if job["policy"]!="disabled":
                    require(row["ptx_cache_key"]==job["artifact_key"] and
                            row["ptx_artifact_sha256"]==job["artifact_sha256"],"Expected PTX artifact was not used")
                sidecar.update(evaluate(index,result),metadata=row,fresh_dispatch_to_return_seconds=outer,
                               result_sha256=sha(output/"result.json"))
                if job["original_t9"] is not None:
                    comparison=equal_arrays(output/"trajectory.npz",WORK/job["original_t9"]["path"])
                    sidecar.update(original_t9_bitwise=comparison)
                    require(all(comparison.values()),"PTX replay differs from the original same-engine arrays")
            except Exception as error:
                sidecar.update(qualified=False,failure_type=type(error).__name__,
                               failure=str(error).replace(str(WORK.parent),"<workspace>"))
        save(output/"qualification.json",sidecar)
        outcomes.append(sidecar)
        print(index,job["name"],job["arm"],job["engine"],job["policy"],watchdog["status"],sidecar["qualified"],flush=True)
        if not sidecar["qualified"]:
            break
    validate(plan)
    require(sha(protocol)==psha,"PTX diagnostic protocol changed during execution")
    matched=[]
    engine_pairs=[]
    complete=len(outcomes)==len(plan["slots"]) and all(row["qualified"] for row in outcomes)
    if complete:
        rows={(r["job"]["pair"],r["job"]["engine"],r["job"]["policy"]):r for r in outcomes if r["job"]["policy"] in POLICIES}
        pairs=sorted({pair for pair,_,_ in rows})
        for pair in pairs:
            for engine in ENGINES:
                a,b=(rows[pair,engine,policy] for policy in POLICIES)
                bitwise=equal_arrays(root/f"slot_{a['index']:03d}"/"trajectory.npz",root/f"slot_{b['index']:03d}"/"trajectory.npz")
                matched.append(dict(pair=pair,name=a["job"]["name"],arm=a["job"]["arm"],engine=engine,bitwise=bitwise,
                    disabled_over_hit_solver=a["metadata"]["seconds"]/b["metadata"]["seconds"],
                    disabled_over_hit_fresh=a["fresh_dispatch_to_return_seconds"]/b["fresh_dispatch_to_return_seconds"]))
            a,b=(rows[pair,engine,"strict_hit"] for engine in ENGINES)
            engine_pairs.append(dict(pair=pair,name=a["job"]["name"],arm=a["job"]["arm"],
                numerics=pair_evaluate(a["index"],b["index"]),stock_over_fused_solver=a["metadata"]["seconds"]/b["metadata"]["seconds"],
                stock_over_fused_fresh=a["fresh_dispatch_to_return_seconds"]/b["fresh_dispatch_to_return_seconds"],
                stock_over_fused_march=a["metadata"]["march_seconds"]/b["metadata"]["march_seconds"]))
    numerical=complete and all(all(row["bitwise"].values()) for row in matched) and all(row["numerics"]["passed"] for row in engine_pairs)
    cache_gain=numerical and all(min(row["disabled_over_hit_solver"],row["disabled_over_hit_fresh"])>1 for row in matched)
    fused_gain=numerical and all(all(row[k]>=1/1.1 for row in engine_pairs) and
        sum(row[k]>1 for row in engine_pairs)>len(engine_pairs)//2 for k in ("stock_over_fused_solver","stock_over_fused_fresh"))
    save(root/"summary.json",dict(status="passed" if numerical else "stopped",stage=plan["stage"],protocol_sha256=psha,
        rows=outcomes,matched_cache=matched,paired_strict_hit_engines=engine_pairs,cache_diagnostic_signal=bool(cache_gain),
        fused_diagnostic_signal=bool(fused_gain),promotion_eligible=bool(cache_gain and fused_gain),
        unattempted=plan["slots"][len(outcomes):],isolated_speed_claim_supported=False))
    return numerical


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze",choices=("t9","t72"))
    action.add_argument("--run",action="store_true")
    action.add_argument("--worker",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prior",type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    if args.freeze:
        freeze(args.freeze,args.prior.resolve() if args.prior else None,root)
    elif args.run:
        raise SystemExit(0 if run(root) else 2)
    else:
        worker(root,args.worker)
