"""Qualify CPU and CUDA Galerkin at the original rate over early and late snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

from cpu_precontracted_native import NativePrecontractedGalerkin, RecurrenceGalerkin
from basis_recurrence_model import native_model
from galerkin_cuda_native import solve_native
import cpu_formal as base
import fd_pow2_reference_chain as chain
import joint_hardware_pow2 as old


HERE=base.HERE
TEMPLATE=HERE/"protocol_r128_operator_qualification.json"
OUT=HERE/"r128_operator_qualification_output"
FROZEN=OUT/"protocol.json"
JOURNAL=OUT/"runs.json"
SUMMARY=OUT/"qualification.json"
case,np=base.case,base.case.np
MODELS={"cpu_recurrence":RecurrenceGalerkin,"cpu_native_contract":NativePrecontractedGalerkin,
        "cuda_graph":RecurrenceGalerkin}


def freeze():
    p=base.read(TEMPLATE)
    previous=base.read(old.FROZEN)
    protocol=previous["protocol"]
    old.approved_inputs(previous)
    if (p["horizon"]!=protocol["horizon"] or p["late_checkpoints"]!=protocol["checkpoints"] or
            p["rfm_seeds"]!=protocol["seeds"] or len(set(p["rfm_seeds"]))!=5 or
            p["deterministic_seed"]!=protocol["deterministic_seed"] or p["arms"]!=list(MODELS) or
            p["late_physical_gate"]!=protocol["gate"] or p["graph_steps"]!=old.kernels.GRAPH_STEPS):
        raise RuntimeError("Qualification differs from the registered late protocol")
    specs={r["name"]:r for r in protocol["variants"] if r["name"] in p["methods"]}
    if len(specs)!=2 or any(r["rate"]!=p["rate"] for r in specs.values()):
        raise RuntimeError("Both Galerkin controls must use the original rate128")
    checkpoints=p["early_checkpoints"]+p["late_checkpoints"]
    if len(checkpoints)!=13 or checkpoints!=sorted(set(checkpoints)):
        raise RuntimeError("Need 13 distinct chronological checkpoints")
    jobs=[]
    for name in p["methods"]:
        seeds=p["rfm_seeds"] if specs[name]["method"]=="rfm" else [p["deterministic_seed"]]
        jobs.extend(dict(name=name,seed=seed,arm=arm) for seed in seeds for arm in p["arms"])
    if len(jobs)!=18:
        raise RuntimeError("Need eighteen registered trajectory solves")
    files=(Path(__file__),TEMPLATE,HERE/"cpu_precontracted.py",HERE/"cpu_precontracted_native.py",
           HERE/"galerkin_cpu_contract.c",HERE/"galerkin_cpu_contract.dll")
    sources=old.source_hashes()|chain.fingerprint()|{f"rfm_v7/{path.name}":base.digest(path) for path in files}
    reference=previous["reference"]
    result=dict(status="frozen",template=p,original_protocol_sha256=base.digest(old.FROZEN),
        protocol=protocol,specs=specs,checkpoints=checkpoints,jobs=jobs,source_hashes=sources,reference=reference)
    if FROZEN.exists():
        if base.read(FROZEN)!=result:
            raise RuntimeError("Frozen qualification inputs differ")
    else:
        base.save(FROZEN,result)
    return result


def artifact(job):
    return OUT/f"{job['name']}_{job['seed']}_{job['arm']}.npz"


def check_solution(sol, frozen):
    p=frozen["template"]
    probes=np.arange(0.,p["horizon"]+.01*p["probe_spacing"],p["probe_spacing"])
    if (sol.row["steps"]!=p["rate"]*p["horizon"] or sol.row["dt"]!=1/p["rate"] or
            not np.array_equal(sol.times,frozen["checkpoints"]) or not np.array_equal(sol.probe_times,probes) or
            sol.states.shape[0]!=len(frozen["checkpoints"]) or
            not all(np.isfinite(getattr(sol,k)).all() for k in ("states","transverse","vorticity"))):
        raise RuntimeError("Trajectory shape, time or finite-state check failed")


def run(job,frozen):
    path=artifact(job)
    record=path.with_suffix(".json")
    if record.exists():
        row=base.read(record)
        if row["job"]!=job or ("artifact" in row and base.digest(path)!=row["sha256"]):
            raise RuntimeError("Existing qualification trajectory differs")
        return row
    if path.exists():
        raise FileExistsError(f"Inspect orphaned trajectory {path.name}")
    p=frozen["template"]
    _,cfg=base.config(frozen["specs"][job["name"]],frozen["protocol"],job["seed"])
    args=dict(steps=p["rate"]*p["horizon"],horizon=p["horizon"],snapshots=frozen["checkpoints"],
              probe_spacing=p["probe_spacing"],**cfg)
    row=dict(job=job)
    print("START",job,flush=True)
    try:
        if job["arm"]=="cuda_graph":
            with native_model():
                sol=solve_native(graph_steps=p["graph_steps"],**args)
        else:
            sol=case.solve(MODELS[job["arm"]],**args)
        check_solution(sol,frozen)
        temp=path.with_suffix(".tmp.npz")
        case.store_solution(temp,sol,cfg)
        temp.replace(path)
        row.update(status="complete",artifact=path.name,sha256=base.digest(path),metadata=sol.row)
    except (ArithmeticError,ValueError,RuntimeError,OSError,np.linalg.LinAlgError) as exc:
        row.update(status="failed",failure=str(exc))
    base.save(record,row)
    print("DONE",job,row["status"],flush=True)
    return row


def score_all(frozen,rows):
    p=frozen["template"]
    journal=base.read(chain.JOURNAL)
    rp=journal["frozen"]["protocol"]
    if rp["checkpoints"]!=p["late_checkpoints"] or journal["status"]!="complete":
        raise RuntimeError("Reference arm protocol differs")
    reference=chain.references(rp,journal["rows"])
    refs={spec["name"]:sol for spec,sol in zip(rp["reference_path"],reference,strict=True)}
    grids=base.grid_set()
    truth={name:case.reference_fields(refs[name],grids) for name in p["references"]}
    early={(g,t) for g in grids for t in p["early_checkpoints"]}
    late={(g,t) for g in grids for t in p["late_checkpoints"]}
    keys={(r["job"]["name"],r["job"]["seed"],r["job"]["arm"]):r for r in rows}
    if len(rows)!=18 or len(keys)!=18:
        raise RuntimeError("Expected complete distinct raw trajectory records")
    results=[]
    controls={}
    for job in frozen["jobs"]:
        name,seed,arm=(job[k] for k in ("name","seed","arm"))
        row=keys[name,seed,arm]
        if row["status"]!="complete" or (arm!="cpu_recurrence" and (name,seed) not in controls):
            results.append(dict(job=job,qualified=False,failure=row.get("failure","Missing CPU control")))
            continue
        path=OUT/row["artifact"]
        old.sha_matches(path,row["sha256"])
        sol=case.load_solution(path,MODELS[arm])
        check_solution(sol,frozen)
        if arm=="cpu_recurrence":
            controls[name,seed]=sol,case.reference_fields(sol,grids)
        control,fields=controls[name,seed]
        scores=case.score(sol,fields,grids)
        full_max=old.score_rows(scores,early|late)
        early_max=old.score_rows([r for r in scores if (r["grid"],r["time"]) in early],early)
        late_max=old.score_rows([r for r in scores if (r["grid"],r["time"]) in late],late)
        probe={}
        for k in ("transverse","vorticity"):
            x,y=getattr(sol,k),getattr(control,k)
            probe[k]=float(np.linalg.norm(x-y)/np.linalg.norm(y))
        late_start=len(p["early_checkpoints"])
        late_sol=case.Solution(sol.model,sol.states[late_start:],sol.times[late_start:],sol.probe_times,
                               sol.transverse,sol.vorticity,sol.row)
        physical={ref:case.score(late_sol,values,grids) for ref,values in truth.items()}
        physical_max={ref:old.score_rows(values,late) for ref,values in physical.items()}
        wake=base.wake(sol)
        implementation=all(math.isfinite(v) and v<=p["implementation_tolerance"]
                           for v in (*full_max.values(),*probe.values()))
        numerical=all(max(metrics.values())<=p["late_physical_gate"] for metrics in physical_max.values()) and wake[
            "sustained_alternating_probe_signs"]
        results.append(dict(job=job,artifact=row["artifact"],sha256=row["sha256"],implementation_scores=scores,
            implementation_worst=full_max,early_implementation_worst=early_max,late_implementation_worst=late_max,
            full_horizon_probe_error=probe,late_physical_scores=physical,late_physical_worst=physical_max,wake=wake,
            implementation_pass=implementation,late_physical_pass=numerical,qualified=bool(implementation and numerical)))
        print("SCORE",job,results[-1]["qualified"],early_max,flush=True)
    return results


def execute(frozen):
    expected=base.digest(FROZEN)
    if JOURNAL.exists() and base.read(JOURNAL)["protocol_sha256"]!=expected:
        raise RuntimeError("Qualification run journal differs from its freeze")
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    result=dict(status="running",protocol_sha256=expected,rows=[])
    base.save(JOURNAL,result)
    jobs=frozen["jobs"]
    with ProcessPoolExecutor(max_workers=frozen["template"]["cpu_workers"]) as pool:
        cpu=[pool.submit(run,job,frozen) for job in jobs if job["arm"]!="cuda_graph"]
        for job in (j for j in jobs if j["arm"]=="cuda_graph"):
            result["rows"].append(run(job,frozen))
            base.save(JOURNAL,result)
        for task in cpu:
            result["rows"].append(task.result())
            base.save(JOURNAL,result)
    if freeze()!=frozen or base.digest(FROZEN)!=expected:
        raise RuntimeError("Frozen qualification inputs changed")
    scored=score_all(frozen,result["rows"])
    passed=len(scored)==18 and all(r["qualified"] for r in scored)
    result.update(status="complete",passes=passed)
    base.save(JOURNAL,result)
    base.save(SUMMARY,dict(status="complete",passes=passed,protocol_sha256=expected,
        runs_sha256=base.digest(JOURNAL),rows=scored,scope=frozen["template"]["scope"]))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    mode=cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only",action="store_true")
    mode.add_argument("--execute",action="store_true")
    args=cli.parse_args()
    protocol=freeze()
    print("FROZEN",base.digest(FROZEN),len(protocol["jobs"]),flush=True)
    if args.execute:
        execute(protocol)
