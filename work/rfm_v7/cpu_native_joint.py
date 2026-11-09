"""Freeze or run the selected-rate ten-arm isolated hardware comparison."""
import argparse
import math
import os
from pathlib import Path
import platform
from statistics import median
import time

import cpu_native_qualification as qual


base,case,np,joint = qual.base,qual.case,qual.np,qual.joint
HERE=base.HERE
TEMPLATE=HERE/"protocol_cpu_native_joint_draft.json"
OUT=HERE/"cpu_native_joint_output"
FROZEN=OUT/"protocol.json"
TIMING=OUT/"timing.json"
THREAD_KEYS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")


def read_frozen():
    template=base.read(TEMPLATE)
    result=base.read(qual.JOURNAL)
    build=qual.OUT/"native_build_provenance.json"
    build_record=base.read(build)
    inputs=qual.manifest()
    protocol=inputs["solve_protocol"]
    specs={s["name"]:s for s in protocol["variants"]}
    required={(j["name"],j["seed"]) for j in inputs["jobs"]}
    named={(r["job"]["name"],r["job"]["seed"]):r for r in result["rows"]}
    if (result["status"]!="complete" or not result["passes"] or result["frozen"]!=inputs or
            base.digest(qual.FIXED)!=result["fixed_state_sha256"] or len(named)!=len(result["rows"]) or
            set(named)!=required or not all(r["qualified"] for r in named.values())):
        raise RuntimeError("All six new CPU qualification trajectories are required")
    if (build_record["status"]!="passed" or build_record["pe_machine"]!="0x8664" or
            not build_record["files"] or any(inputs["source_hashes"].get(n)!=sha for n,sha in build_record["files"].items())):
        raise RuntimeError("Native CPU build provenance differs from qualification")
    if (template["horizon"]!=protocol["horizon"] or template["gate"]!=protocol["gate"] or
            template["reference_budget_fraction"]!=protocol["reference_budget_fraction"] or
            template["galerkin_rate"]!=inputs["protocol"]["rate"] or
            template["reference_names"]!=inputs["protocol"]["reference_names"] or
            template["implementation_tolerance"]!=inputs["protocol"]["implementation_tolerance"] or
            template["registered_rfm_seeds"]!=protocol["seeds"] or
            template["deterministic_seed"]!=protocol["deterministic_seed"] or
            template["cuda_graph_steps"]!=joint.kernels.GRAPH_STEPS or template["cuda_fd_linear"]!="dense"):
        raise RuntimeError("Timing template differs from qualification")
    trajectories={}
    for r in result["rows"]:
        artifact=qual.OUT/r["artifact"]
        joint.sha_matches(artifact,r["sha256"])
        trajectories[artifact.relative_to(base.WORK).as_posix()]=r["sha256"]
    if (template["galerkin_names"]!=inputs["protocol"]["methods"] or
            template["galerkin_arms"]!=["cpu_recurrence","cpu_native_contract","cuda_graph","cuda_graph_rate128"] or
            template["fd_arms"]!=["cpu_buffered","cuda_graph"] or
            template["galerkin_cuda_control_rate"]!=128 or template["fd_rate"]!=128):
        raise RuntimeError("Unexpected rate or backend comparison inventory")
    rates={(n,a):template["galerkin_rate"] for n in template["galerkin_names"] for a in template["galerkin_arms"]}
    rates.update({(n,"cuda_graph_rate128"):template["galerkin_cuda_control_rate"] for n in template["galerkin_names"]})
    fd=template["fd_name"]
    rates.update({(fd,a):template["fd_rate"] for a in template["fd_arms"]})
    gpu_evidence=[]
    step_evidence=base.read(qual.step.OUT/"qualification.json")
    required_gpu={(name,seed,rate) for name,seed in required for rate in
                  (template["galerkin_rate"],template["galerkin_cuda_control_rate"])}
    required_gpu.add((fd,template["deterministic_seed"],template["fd_rate"]))
    for row in step_evidence["gpu"]:
        name,seed,rate=(row["job"][k] for k in ("name","seed","rate"))
        if (name,seed,rate) in required_gpu:
            if not row["direct_pass"]:
                raise RuntimeError("A selected GPU trajectory failed qualification")
            path=qual.path_and_hash(row)
            trajectories[path.relative_to(base.WORK).as_posix()]=row["sha256"]
            gpu_evidence.append((name,seed,rate))
    if len(gpu_evidence)!=len(required_gpu) or set(gpu_evidence)!=required_gpu:
        raise RuntimeError("Selected GPU qualification inventory is incomplete")
    fd_job=dict(name=fd,seed=template["deterministic_seed"],rate=template["fd_rate"])
    fd_record=qual.step.OUT/f"{qual.step.Screen.tag(fd_job,'cpu')}.json"
    fd_row=base.read(fd_record)
    if fd_row["job"]!=fd_job or not fd_row["direct_pass"]:
        raise RuntimeError("Selected FD CPU archive failed qualification")
    fd_path=qual.path_and_hash(fd_row)
    trajectories[fd_path.relative_to(base.WORK).as_posix()]=fd_row["sha256"]
    arms=[(n,a) for n in template["galerkin_names"] for a in template["galerkin_arms"]]
    arms.extend((fd,a) for a in template["fd_arms"])
    if len(arms)!=10 or template["blocks"]!=len(protocol["seeds"]):
        raise RuntimeError("Need five randomized ten-arm blocks")
    schedule=[]
    rng=np.random.default_rng(template["order_seed"])
    for block,seed in enumerate(protocol["seeds"],1):
        for index in rng.permutation(len(arms)):
            name,arm=arms[index]
            schedule.append(dict(block=block,name=name,arm=arm,rate=rates[name,arm],
                                 seed=seed if specs[name]["method"]=="rfm" else protocol["deterministic_seed"]))
    files=(Path(__file__),TEMPLATE,qual.JOURNAL,qual.FIXED,build,fd_record)
    manifest=dict(status="frozen",template=template,inputs=inputs,schedule=schedule,trajectories=trajectories,
                  files={p.relative_to(base.WORK).as_posix():base.digest(p) for p in files})
    if FROZEN.exists():
        if base.read(FROZEN)!=manifest:
            raise RuntimeError("Frozen native hardware comparison differs")
    else:
        base.save(FROZEN,manifest)
    return manifest,specs,fd_path


def solver(job,spec,protocol):
    spec=dict(spec,rate=job["rate"])
    _,cfg=base.config(spec,protocol,job["seed"])
    steps=protocol["horizon"]*spec["rate"]
    if job["arm"] in ("cpu_native_contract","cpu_recurrence"):
        cls=qual.NativePrecontractedGalerkin if job["arm"]=="cpu_native_contract" else qual.RecurrenceGalerkin
        return case.solve(cls,steps=steps,**cfg),cfg
    return joint.kernels.solve(spec,protocol,job["seed"],"cuda_graph" if job["arm"].startswith("cuda_graph") else "cpu")


def run_job(job,spec,protocol,template,control,fields):
    path=OUT/f"b{job['block']}_{job['name']}_{job['arm']}_{job['seed']}.npz"
    details=path.with_suffix(".json")
    if details.exists():
        row=base.read(details)
        if any(row[k]!=v for k,v in job.items()) or (
                "artifact" in row and base.digest(path)!=row["sha256"]):
            raise RuntimeError("Checkpointed timing observation differs")
        return row
    if path.exists():
        raise FileExistsError(f"Inspect orphaned timed trajectory {path.name}")
    row=dict(job)
    try:
        started=time.perf_counter()
        sol,cfg=solver(job,spec,protocol)
        seconds=time.perf_counter()-started
        if not np.array_equal(sol.times,control.times) or not np.array_equal(sol.probe_times,control.probe_times):
            raise RuntimeError("Timed and qualified observation times differ")
        keys={(g,t) for g in qual._grids for t in protocol["checkpoints"]}
        scores={n:case.score(sol,truth,qual._grids) for n,truth in qual._truth.items()}
        worst={n:joint.score_rows(values,keys) for n,values in scores.items()}
        field=joint.score_rows(case.score(sol,fields,qual._grids),keys)
        probes={k:qual.relative(getattr(sol,k),getattr(control,k)) for k in ("transverse","vorticity")}
        wake=base.wake(sol)
        implementation=all(math.isfinite(x) and x<=template["implementation_tolerance"]
                           for x in (*field.values(),*probes.values()))
        numerical=all(max(values.values())<=template["gate"] for values in worst.values()) and wake["sustained_alternating_probe_signs"]
        temp=path.with_suffix(".tmp.npz")
        case.store_solution(temp,sol,cfg)
        temp.replace(path)
        row.update(seconds=seconds,metadata=sol.row,scores=scores,worst=worst,wake=wake,
                   boundaries=base.boundary_errors(sol),
                   implementation_field=field,implementation_probe=probes,implementation_pass=implementation,
                   numerical_pass=numerical,artifact=path.name,sha256=base.digest(path))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError,OSError,RuntimeError) as exc:
        row.update(failure=str(exc),implementation_pass=False,numerical_pass=False)
    base.save(details,row)
    return row


def summary(rows,template):
    keyed={(r["name"],r["arm"],r["block"]):r for r in rows}
    blocks=range(1,template["blocks"]+1)
    groups=[]
    for name,arm in dict.fromkeys((r["name"],r["arm"]) for r in rows):
        group=[keyed[name,arm,b] for b in blocks]
        times=[r["seconds"] for r in group if "seconds" in r]
        scored=[r for r in group if "worst" in r]
        worst={ref:{k:max(r["worst"][ref][k] for r in scored) for k in joint.METRICS}
               for ref in template["reference_names"]} if scored else None
        groups.append(dict(name=name,arm=arm,count=len(group),successes=len(times),
            passes=all(r["implementation_pass"] and r["numerical_pass"] for r in group),
            seconds=times,median=median(times) if times else None,
            minimum=min(times) if times else None,maximum=max(times) if times else None,worst=worst))
    comparisons=[]
    fd=template["fd_name"]
    for name in template["galerkin_names"]:
        comparisons.extend(((name,"cpu_recurrence",name,"cpu_native_contract"),
                            (name,"cuda_graph_rate128",name,"cuda_graph"),
                            (fd,"cpu_buffered",name,"cpu_native_contract"),
                            (fd,"cuda_graph",name,"cuda_graph"),
                            (name,"cpu_native_contract",name,"cuda_graph")))
    rfm,legendre=template["galerkin_names"]
    comparisons.extend((legendre,arm,rfm,arm) for arm in template["galerkin_arms"])
    comparisons.append((fd,"cpu_buffered",fd,"cuda_graph"))
    ratios=[]
    for name,arm,other,other_arm in comparisons:
        pairs=[]
        for block in blocks:
            a,b=keyed[name,arm,block],keyed[other,other_arm,block]
            passed=all(r["implementation_pass"] and r["numerical_pass"] for r in (a,b))
            pairs.append(dict(block=block,ratio=a["seconds"]/b["seconds"] if passed else None))
        samples=[p["ratio"] for p in pairs if p["ratio"] is not None]
        aggregate=dict(median=median(samples),minimum=min(samples),maximum=max(samples),
                       denominator_wins=sum(v>1 for v in samples)) if len(samples)==len(pairs) else None
        ratios.append(dict(numerator=[name,arm],denominator=[other,other_arm],pairs=pairs,aggregate=aggregate))
    return groups,ratios


def time_cohort(frozen,specs,fd_path):
    p=frozen["template"]
    protocol=frozen["inputs"]["solve_protocol"]
    cpu=base.pin()
    qual.start_worker(frozen["inputs"])
    controls={}
    for job in frozen["inputs"]["jobs"]:
        for i,rate in enumerate((p["galerkin_rate"],p["galerkin_cuda_control_rate"])):
            _,sol=qual.control(job,i)
            controls[job["name"],job["seed"],rate]=sol,case.reference_fields(sol,qual._grids)
    sol=case.load_solution(fd_path,base.FullFD)
    controls[p["fd_name"],p["deterministic_seed"],p["fd_rate"]]=sol,case.reference_fields(sol,qual._grids)
    result=dict(status="running",protocol_sha256=base.digest(FROZEN),cpu=cpu,
                gpu=joint.old_joint.nvidia(),platform=platform.platform(),parent_pid=os.getpid(),
                environment={k:os.environ[k] for k in THREAD_KEYS},rows=[])
    if TIMING.exists():
        old=base.read(TIMING)
        if old["cpu"]!=cpu or old["protocol_sha256"]!=result["protocol_sha256"]:
            raise RuntimeError("Existing timing journal differs in affinity or freeze")
        if old["status"]=="complete":
            return
    np.linalg.svd(np.eye(10))
    for job in frozen["schedule"]:
        ctrl,fields=controls[job["name"],job["seed"],job["rate"]]
        row=run_job(job,specs[job["name"]],protocol,p,ctrl,fields)
        result["rows"].append(row)
        base.save(TIMING,result)
        print(job["block"],job["name"],job["arm"],row.get("seconds",row.get("failure")),flush=True)
    if read_frozen()[0]!=frozen:
        raise RuntimeError("Frozen timing inputs changed")
    groups,ratios=summary(result["rows"],p)
    result.update(status="complete",groups=groups,ratios=ratios)
    base.save(TIMING,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only",action="store_true")
    mode.add_argument("--confirm-isolated",action="store_true")
    args=parser.parse_args()
    if args.confirm_isolated and any(os.environ.get(k)!="1" for k in THREAD_KEYS):
        parser.error("Run isolated timing with one BLAS thread")
    frozen,specs,fd_path=read_frozen()
    print("FROZEN",len(frozen["schedule"]),flush=True)
    if args.confirm_isolated:
        time_cohort(frozen,specs,fd_path)


if __name__=="__main__":
    main()
