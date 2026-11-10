"""Bounded paired accuracy and phase diagnostics for the fused CUDA march."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys

from fresh_worker_diagnostic import pin
from gpu_owned_watchdog import run_owned
from solver_worker import THREADS, affinity_mask, model_signature
from win32_isolation_monitor import WindowsCensus, process_key


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
V7=WORK/"rfm_v7"
TOOLKIT=Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
THIN=HERE/"thin_t72_qualification_output"
LEG=HERE/"legendre_rank_gpu_qualification_output"
REFERENCE=V7/"switch_qualification_output"/"protocol.json"
HISTORY=("transverse","vorticity")
ENGINES=("warp_cufft","warp_fused")
RFM_SEEDS=(7331,7687,8011,8353,9041)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def save(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def record(path,expected=None):
    value=sha(path)
    require(expected is None or expected==value,"A frozen artifact changed")
    return dict(path=path.relative_to(WORK).as_posix(),sha256=value)


def bound(spec):
    path=WORK/spec["path"]
    record(path,spec["sha256"])
    return path


def thin_jobs():
    plan=read(THIN/"protocol.json")
    summary=read(THIN/"qualification.json")
    require(summary["status"]=="passed" and summary["protocol_sha256"]==sha(THIN/"protocol.json"),"Thin controls are not frozen")
    original={scale:read(bound(value["archive"]["protocol"])) for scale,value in plan["archive_contexts"].items()}
    results={}
    for item in plan["jobs"]:
        if item["family"] not in ("rfm","fd"):
            continue
        name,arm,seed,scale=(item[key] for key in ("name","arm","seed","scale"))
        old=original[scale]
        if arm=="fd":
            control={"single":old["fd_controls"][name]["cpu"]}
            anchor=control
        else:
            match=next(c for c in old["gal_controls"] if (c["name"],c["seed"])==(name,seed))
            anchor={key:match[key] for key in ("early","late")}
            control=anchor if arm=="uniform" else {"single":match["cpu_recurrence"]}
        results[name,arm,seed]=dict(name=name,arm=arm,seed=seed,scale=scale,family=item["family"],
            config=item["config"],args=item["solve_args"],control=control,anchor=anchor)
    return results,plan


def strong_jobs(physical):
    plan=read(LEG/"protocol.json")
    summary=read(LEG/"summary.json")
    require(summary["status"]=="complete" and summary["protocol_sha256"]==sha(LEG/"protocol.json") and
            len(summary["rows"])==12 and all(row["qualified"] for row in summary["rows"]),"Stronger Legendre GPU controls did not qualify")
    results=[]
    for job in plan["jobs"]:
        if job["observer"]!="warp":
            continue
        name,arm,cfg=(job[key] for key in ("name","schedule","config"))
        start,late=job["rates"]
        args=dict(cfg,horizon=72,snapshots=physical,probe_spacing=.125,graph_steps=24)
        args.update(steps=72*start) if arm=="uniform" else args.update(startup_rate=start,late_rate=late,switch_time=8,preparation="combined")
        results.append(dict(name=name,arm=arm,seed=17,scale="mid" if job["k"]==72 else "high",family="legendre",
            config=cfg,args=args,control={"single":job["cpu_control"]},anchor={"single":job["uniform_anchor"]}))
    return results,plan


def build(stage,prior):
    template=read(REFERENCE)
    physical=template["physical_times"]
    thin,thin_plan=thin_jobs()
    reps=[thin[name,arm,seed] for name,arm,seed in (("rfm37_k72","uniform",7331),("rfm41_k80","combined",7331),
                                                ("fd176_k80","fd",17),("fd184_k80","fd",17))]
    files=[Path(__file__),HERE/"fused_marches.py",HERE/"make_fused_marches.py",HERE/"fused_marches_generation.json",
           HERE/"fused_march_runtime.py",HERE/"fused_nonlinear_cuda_runtime.py",HERE/"fused_nonlinear_fft.py",
           HERE/"fused_nonlinear_fft.cu",HERE/"warp_observe_cuda.py",HERE/"gpu_owned_watchdog.py",
           HERE/"fused_device_watchdog_audit_output"/"certificate.json",HERE/"fused_nonlinear_fft_device_output"/"summary.json",
           HERE/"galerkin_cuda_prepared_minimal.py",THIN/"protocol.json",THIN/"qualification.json",REFERENCE]
    sources={}
    for item in thin_plan["archive_contexts"].values():
        sources.update(item["old_source_hashes"])
    sources.update(thin_plan["source_hashes"])
    if stage=="t9":
        jobs,horizon=reps,9
    else:
        require(prior is not None,"A later stage must bind the prior qualification")
        previous=read(prior/"summary.json")
        expected="t9" if stage=="t72" else "t72"
        require(previous["status"]=="passed" and previous["stage"]==expected and
                previous["protocol_sha256"]==sha(prior/"protocol.json") and
                (stage=="t72" or previous["timing_diagnostic_signal"]),"Prior stage did not permit promotion")
        files.extend((prior/"summary.json",prior/"protocol.json"))
        horizon=72
        if stage=="t72":
            stronger,strong_plan=strong_jobs(physical)
            sources.update(strong_plan["source_hashes"])
            files.extend((LEG/"protocol.json",LEG/"summary.json"))
            jobs=[thin[name,arm,7331] for name in ("rfm37_k72","rfm41_k80") for arm in ("uniform","combined")]
            jobs+=reps[2:]+stronger
        else:
            jobs=[thin[name,arm,seed] for name in ("rfm37_k72","rfm41_k80") for arm in ("uniform","combined") for seed in RFM_SEEDS[1:]]
    for job in jobs:
        for spec in (*job["control"].values(),*job["anchor"].values()):
            sources[spec["path"]]=spec["sha256"]
    sources.update({path.relative_to(WORK).as_posix():sha(path) for path in files})
    slots=[]
    for i,job in enumerate(jobs):
        args=dict(job["args"])
        args["horizon"]=horizon
        times=[t for t in physical if t<=horizon]
        if job["arm"]=="combined":
            times=sorted(times+[8+i/args["late_rate"] for i in (1,2)])
        else:
            args["steps"]=args["steps"]//72*horizon
        args["snapshots"]=times
        order=ENGINES if i%2==0 else ENGINES[::-1]
        # Alternate engine order across pairs while keeping both matched outputs.
        for engine in order:
            slots.append(dict(job,args=args,engine=engine,pair=i))
    return dict(status="frozen",stage=stage,horizon=horizon,physical_times=[t for t in physical if t<=horizon],
        slots=slots,sources_sha256=sources,toolkit_sha256={name:sha(TOOLKIT/"bin"/name) for name in
        ("nvrtc64_112_0.dll","nvrtc-builtins64_118.dll","cufft64_10.dll")},
        hard_deadline_seconds=30,implementation_gate=1e-5,physical_gate=.005,
        timing_scope="Sequential bounded diagnostic only, no isolated ranking. >10% fused total regression blocks promotion.")


def validate(plan):
    require(plan["status"]=="frozen" and all(sha(WORK/name)==value for name,value in plan["sources_sha256"].items()) and
            all(sha(TOOLKIT/"bin"/name)==value for name,value in plan["toolkit_sha256"].items()),"Registered fused march inputs changed")


def worker(root,index):
    from fused_marches import solve_uniform_fused, solve_combined_fused, solve_fd_fused, case
    from prepared_process_identity import identity_from_pid

    pin(4)
    parent=identity_from_pid(os.getppid())
    protocol=root/"protocol.json"
    protocol_hash=sha(protocol)
    require(os.environ["RFM_FUSED_OWNER"]==f"{protocol_hash}:{parent['pid']}:{parent['birth_filetime_100ns']}:{index}" and
            affinity_mask()==16 and all(os.environ[name]=="1" for name in THREADS),"Unregistered worker launch")
    job=read(protocol)["slots"][index]
    solver={"uniform":solve_uniform_fused,"combined":solve_combined_fused,"fd":solve_fd_fused}[job["arm"]]
    result=solver(engine=job["engine"],toolkit=TOOLKIT,**job["args"])
    signature=model_signature(result.model,job["family"])
    output=root/f"slot_{index:03d}"
    raw=output/"trajectory.npz"
    with raw.open("xb") as handle:
        case.store_solution(handle,result,job["config"])
    census=WindowsCensus()
    save(output/"result.json",dict(status="complete",index=index,protocol_sha256=protocol_hash,job=job,metadata=result.row,
         spatial_sha256=signature,trajectory_sha256=sha(raw),affinity_mask=affinity_mask(),identity=census.times(census.current_process())))


def assessment(root,plan):
    import numpy as np
    import cpu_formal as base
    from fd_cuda_native import FullFD
    from galerkin_minimal_model import RecurrenceGalerkin, case
    from legendre_rank_uniform import scores
    from switch_qualification import select
    from verify_switch_qualification import absolute_reference, fields_at, maxima, measured

    grids=base.grid_set()
    truth,budget=absolute_reference(read(REFERENCE),grids)
    full=read(REFERENCE)["physical_times"]
    physical=plan["physical_times"]
    truth={name:{g:[values[full.index(t)] for t in physical] for g,values in fields.items()} for name,fields in truth.items()}

    def load_control(spec,cls):
        if "single" in spec:
            return case.load_solution(bound(spec["single"]),cls)
        early=case.load_solution(bound(spec["early"]),cls)
        late=case.load_solution(bound(spec["late"]),cls)
        right=select(late,full[12:])
        return case.Solution(late.model,np.concatenate((early.states,right.states)),np.asarray(full),
                             late.probe_times,late.transverse,late.vorticity,late.row)

    def evaluate(index,recorded):
        job=plan["slots"][index]
        path=root/f"slot_{index:03d}"/"trajectory.npz"
        require(recorded["trajectory_sha256"]==sha(path),"Saved timed trajectory differs")
        cls=FullFD if job["arm"]=="fd" else RecurrenceGalerkin
        solution=case.load_solution(path,cls)
        matched=load_control(job["control"],cls)
        model_id=model_signature(matched.model,job["family"])
        require(model_id==recorded["spatial_sha256"]==model_signature(solution.model,job["family"]),
                "Candidate and CPU reconstruction coordinates differ")
        with np.load(path,allow_pickle=False) as values:
            require(json.loads(str(values["config"]))==job["config"],"Saved GPU configuration changed")
        times=job["args"]["snapshots"]
        probes=np.arange(8*plan["horizon"]+1)/8
        require(np.array_equal(solution.times,times) and np.array_equal(solution.probe_times,probes) and
                np.array_equal(matched.probe_times[:len(probes)],probes),"Saved snapshot or full probe schedule differs")
        row=solution.row
        steps=(8*job["args"]["startup_rate"]+(plan["horizon"]-8)*job["args"]["late_rate"] if job["arm"]=="combined" else job["args"]["steps"])
        work=steps-1
        require(row==recorded["metadata"] and row["steps"]==steps and row["nonlinear_evaluations"]==work and
                row["logical_fft_groups"]==2*work and row["inverse_field_transforms"]+row["forward_field_transforms"]==5*work and
                row["logical_field_transforms_all_radial_rows"]==5*work*row["fft_radial_lines"] and
                row["priming_logical_field_transforms"]==5*row["fft_radial_lines"] and row["compiled_cuda_modules"]==1 and
                row["nonlinear_engine"]==job["engine"] and row["observation_reduction"]=="warp" and row["graph_steps"]==24,
                "Nonlinear work, runtime or observations differ from the registration")
        values=fields_at(solution,times,grids)
        impl=measured(values,fields_at(matched,times,grids),grids,times)
        impl_max=maxima(impl)
        probe_errors={}
        for key in HISTORY:
            a,b=getattr(solution,key),getattr(matched,key)[:len(probes)]
            require(a.shape==b.shape==(len(probes),3) and np.isfinite(a).all() and np.linalg.norm(b)>1e-12,"Undefined full probe error")
            probe_errors[key]=float(np.linalg.norm(a-b)/np.linalg.norm(b))
        fields=fields_at(solution,physical,grids)
        if job["arm"]=="combined":
            anchor=load_control(job["anchor"],cls)
            direct,guard,_,tables,_=scores(fields,fields_at(anchor,physical,grids),truth,grids,physical)
        else:
            raw={name:measured(fields,target,grids,physical) for name,target in truth.items()}
            direct={name:maxima(table) for name,table in raw.items()}
            guard={}
            tables={name:dict(direct=[dict(grid=g,time=t,**v) for (g,t),v in table.items()],guard=[]) for name,table in raw.items()}
        wake=base.wake(solution) if plan["horizon"]==72 else None
        qualified=(max((*impl_max.values(),*probe_errors.values()))<=plan["implementation_gate"] and
                   all(max(row.values())<=plan["physical_gate"] for row in (*direct.values(),*guard.values())) and
                   (wake is None or wake["sustained_alternating_probe_signs"]))
        return dict(qualified=bool(qualified),physical_worst=direct,guard_worst=guard,implementation_worst=impl_max,
            probes=probe_errors,wake=wake,tables=tables,reference_budget=budget,
            implementation_rows=[dict(grid=g,time=t,**v) for (g,t),v in impl.items()])
    def evaluate_pair(first,second):
        pair={plan["slots"][index]["engine"]:index for index in (first,second)}
        job=plan["slots"][first]
        cls=FullFD if job["arm"]=="fd" else RecurrenceGalerkin
        old,new=(case.load_solution(root/f"slot_{pair[engine]:03d}"/"trajectory.npz",cls) for engine in ENGINES)
        times=job["args"]["snapshots"]
        fields=maxima(measured(fields_at(new,times,grids),fields_at(old,times,grids),grids,times))
        probes={key:float(np.linalg.norm(getattr(new,key)-getattr(old,key))/np.linalg.norm(getattr(old,key))) for key in HISTORY}
        passed=max((*fields.values(),*probes.values()))<=plan["implementation_gate"]
        return dict(passed=bool(passed),fields=fields,probes=probes)
    return evaluate,evaluate_pair


def run(stage,root,prior):
    plan=build(stage,prior)
    validate(plan)
    root.mkdir()
    save(root/"protocol.json",plan)
    psha=sha(root/"protocol.json")
    pin(6)
    census=WindowsCensus()
    parent=census.times(census.current_process())
    for name in THREADS:
        os.environ[name]="1"
    # Import scoring only on the CPU parent, before launching any GPU solve.
    import fused_marches
    evaluate,evaluate_pair=assessment(root,plan)
    outcomes=[]
    for index,job in enumerate(plan["slots"]):
        output=root/f"slot_{index:03d}"
        os.environ["RFM_FUSED_OWNER"]=f"{psha}:{parent['pid']}:{parent['birth_filetime_100ns']}:{index}"
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(index),"--output",str(root)]
        watchdog=run_owned(command,output,deadline_seconds=plan["hard_deadline_seconds"])
        save(output/"watchdog.json",watchdog)
        sidecar=dict(index=index,job=job,protocol_sha256=psha,watchdog=watchdog,qualified=False)
        if watchdog["status"]=="complete":
            try:
                recorded=read(output/"result.json")
                require(recorded["index"]==index and recorded["job"]==job and recorded["protocol_sha256"]==psha and
                        process_key(recorded["identity"])==process_key(watchdog["identity"]) and recorded["affinity_mask"]==16,
                        "Actual worker identity or frozen slot differs")
                sidecar.update(evaluate(index,recorded),result_sha256=sha(output/"result.json"),metadata=recorded["metadata"])
                if index%2==1 and sidecar["qualified"]:
                    matched=evaluate_pair(index-1,index)
                    sidecar.update(paired_warp_engine=matched,qualified=matched["passed"])
            except Exception as error:
                sidecar.update(failure_type=type(error).__name__,failure=str(error).replace(str(WORK.parent),"<workspace>"))
        save(output/"qualification.json",sidecar)
        outcomes.append(sidecar)
        print(index,job["name"],job["arm"],job["engine"],watchdog["status"],sidecar["qualified"],flush=True)
        if not sidecar["qualified"]:
            break
    validate(plan)
    require(sha(root/"protocol.json")==psha,"Frozen qualification protocol changed")
    pairs=[]
    for i in range(len(outcomes)//2):
        rows={r["job"]["engine"]:r for r in outcomes[2*i:2*i+2]}
        if all(row["qualified"] for row in rows.values()):
            old,new=(rows[engine]["metadata"] for engine in ENGINES)
            pairs.append(dict(pair=i,name=rows[ENGINES[0]]["job"]["name"],arm=rows[ENGINES[0]]["job"]["arm"],
                old_over_fused_total=old["seconds"]/new["seconds"],old_over_fused_march=old["march_seconds"]/new["march_seconds"],
                old_total_seconds=old["seconds"],fused_total_seconds=new["seconds"]))
    complete=len(outcomes)==len(plan["slots"]) and all(row["qualified"] for row in outcomes)
    signal=complete and all(row["old_over_fused_total"]>=1/1.1 for row in pairs) and sum(
        row["old_over_fused_total"]>1 and row["old_over_fused_march"]>1 for row in pairs)>len(pairs)//2
    save(root/"summary.json",dict(status="passed" if complete else "stopped",stage=stage,protocol_sha256=psha,
        rows=outcomes,paired_diagnostic=pairs,timing_diagnostic_signal=bool(signal),
        unattempted=plan["slots"][len(outcomes):],isolated_speed_claim_supported=False))
    return complete


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--stage",choices=("t9","t72","seeds"))
    actions.add_argument("--worker",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prior",type=Path)
    args=parser.parse_args()
    if args.worker is not None:
        worker(args.output,args.worker)
    else:
        raise SystemExit(0 if run(args.stage,args.output,args.prior) else 2)
