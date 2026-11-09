"""Build an independently refined T=8 FD reference and rescore existing fields."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
from unittest.mock import patch

from basis_recurrence_model import RecurrenceGalerkin
from cpu_precontracted_native import NativePrecontractedGalerkin
import fd_pow2_reference_chain as late
import joint_hardware_pow2 as hardware


base,case,np=late.base,late.case,late.np
HERE=base.HERE
OUT=HERE/"early_reference_output"
TEMPLATE=HERE/"protocol_early_reference.json"
FROZEN=OUT/"protocol.json"
RUNS=OUT/"runs.json"
REFERENCE=OUT/"reference_audit.json"
RESCORE=OUT/"candidate_rescore.json"
CHECKPOINT=HERE/"operator_round_checkpoint.json"
METRICS=hardware.METRICS
MODELS={"cpu_recurrence":RecurrenceGalerkin,"cpu_native_contract":NativePrecontractedGalerkin,
        "cuda_graph":RecurrenceGalerkin}


def candidates():
    startup=HERE/"startup_replay_output"
    full=HERE/"r128_operator_qualification_output"
    sa,fa=base.read(startup/"audit.json"),base.read(full/"qualification.json")
    if (sa["status"]!="complete" or fa["status"]!="complete" or not fa["passes"] or
            len(sa["trajectories"])!=12 or len(fa["rows"])!=18):
        raise RuntimeError("Expected twelve saved T8 and eighteen qualified r128 T72 trajectories")
    entries=[]
    for r in sa["trajectories"]:
        path=startup/r["artifact"]
        entries.append(dict(cohort="startup_t8",name=r["name"],seed=r["seed"],rate=r["rate"],
                            arm=r["arm"],source=path.relative_to(base.WORK).as_posix(),sha256=r["sha256"]))
    for r in fa["rows"]:
        path=full/r["artifact"]
        entries.append(dict(cohort="qualified_r128_t72_early_subset",rate=128,**r["job"],
                            source=path.relative_to(base.WORK).as_posix(),sha256=r["sha256"]))
    if len({r["source"] for r in entries})!=30 or any(r["arm"] not in MODELS for r in entries):
        raise RuntimeError("Saved candidate inventory differs")
    for r in entries:
        hardware.sha_matches(base.WORK/r["source"],r["sha256"])
    return entries,(startup/"protocol.json",startup/"audit.json",full/"protocol.json",full/"qualification.json")


def freeze():
    p=base.read(TEMPLATE)
    old=base.read(late.PROTOCOL)
    if (p["reference_path"]!=[{k:r[k] for k in ("name","n","k","rate")} for r in old["reference_path"]] or
            p["horizon"]!=8 or len(p["checkpoints"])!=6 or p["gate"]!=0.005 or p["cpu_workers"]!=3):
        raise RuntimeError("Startup experiment differs from the authorized four independent arms")
    for a,b,axis in zip(p["reference_path"],p["reference_path"][1:],("n","k","rate")):
        if [k for k in ("n","k","rate") if a[k]!=b[k]]!=[axis] or a[axis]>=b[axis]:
            raise RuntimeError("A reference edge changes more than one independent axis")
    archived=base.read(CHECKPOINT)["release"]
    if archived["members"]!=937 or archived["commit"]!="818ad3ad641d16a19d25ac5302a36a172b611177":
        raise RuntimeError("Unexpected preceding release checkpoint")
    release=base.WORK.parent/"release"/"rfm-navier-stokes"/"MANIFEST.json"
    if release.exists():
        hardware.sha_matches(release,archived["manifest_sha256"])
    entries,inputs=candidates()
    sources=hardware.source_hashes()|late.fingerprint()
    for path in (Path(__file__),TEMPLATE,HERE/"cpu_precontracted.py",HERE/"cpu_precontracted_native.py",
                 HERE/"galerkin_cpu_contract.c",HERE/"galerkin_cpu_contract.dll"):
        sources[path.relative_to(base.WORK).as_posix()]=base.digest(path)
    evidence={p.relative_to(base.WORK).as_posix():base.digest(p) for p in (*inputs,CHECKPOINT)}
    result=dict(status="frozen",protocol=p,source_hashes=sources,inputs=evidence,candidates=entries,
                preceding_release=dict(commit=archived["commit"],members=archived["members"],
                                       manifest_sha256=archived["manifest_sha256"]))
    if FROZEN.exists():
        if base.read(FROZEN)!=result:
            raise RuntimeError("Frozen startup reference inputs changed")
    else:
        base.save(FROZEN,result)
    return result


def arm_file(spec):
    return OUT/f"{spec['name']}_n{spec['n']}_k{spec['k']}_r{spec['rate']}.npz"


def run_arm(spec,p):
    path=arm_file(spec)
    record=path.with_suffix(".json")
    if record.exists():
        saved=base.read(record)
        if saved["spec"]!=spec:
            raise RuntimeError("Stored arm differs from freeze")
        hardware.sha_matches(path,saved["sha256"])
        return saved
    if path.exists():
        raise FileExistsError(f"Inspect orphaned trajectory: {path.name}")
    resume=OUT/spec["name"]
    resume.mkdir(exist_ok=True)
    driver=late.radial_driver
    with patch.multiple(driver,OUT=resume,RATE=spec["rate"],HORIZON=p["horizon"],
                        STEPS=p["horizon"]*spec["rate"],GRID=late.grid(spec)), patch.object(case,"SNAPSHOTS",p["checkpoints"]):
        sol,cfg=driver.march(spec["n"])
    temp=path.with_suffix(".tmp.npz")
    case.store_solution(temp,sol,cfg)
    temp.replace(path)
    row=dict(status="complete",spec=spec,artifact=path.name,sha256=base.digest(path),configuration=cfg,metadata=sol.row)
    base.save(record,row)
    print("COMPLETE",spec["name"],sol.row["seconds"],flush=True)
    return row


def norm_entries(candidate,reference,grids,times,floors=None):
    entries=[]
    for name,(s,_,weight) in grids.items():
        near=weight*(case.radius(s)<=2.5)
        for i,t in enumerate(times):
            u,v,o=candidate[name][i]
            U,V,O=reference[name][i]
            velocity=(u-U)**2+(v-V)**2
            curl=(o-O)**2
            velocity_ref=U**2+V**2
            curl_ref=O**2
            items={}
            for key,diff,truth,w in zip(METRICS,(velocity,curl,velocity,curl),
                                       (velocity_ref,curl_ref,velocity_ref,curl_ref),(weight,weight,near,near),strict=True):
                numerator=float(np.sqrt(w@diff.sum(axis=1)))
                denominator=float(np.sqrt(w@truth.sum(axis=1)))
                if not (math.isfinite(numerator) and math.isfinite(denominator)):
                    raise FloatingPointError("Nonfinite weighted field norm")
                if floors is None:
                    items[key]=denominator
                else:
                    floor=floors[name,key]
                    items[key]=dict(absolute_l2=numerator,reference_l2=denominator,denominator_floor=floor,
                                    relative_l2=numerator/denominator if denominator>floor else None)
            entries.append(dict(grid=name,time=t,metrics=items))
    return entries


def summarize(rows):
    metrics={k:[r["metrics"][k]["relative_l2"] for r in rows] for k in METRICS}
    undefined=sum(v is None for values in metrics.values() for v in values)
    worst={k:max((v for v in values if v is not None),default=None) for k,values in metrics.items()}
    return worst,undefined


def reference_audit(frozen,arms):
    p=frozen["protocol"]
    grids=base.grid_set()
    times=p["checkpoints"]
    indexed={r["spec"]["name"]:r for r in arms}
    fields=[]
    for spec in p["reference_path"]:
        row=indexed[spec["name"]]
        path=OUT/row["artifact"]
        hardware.sha_matches(path,row["sha256"])
        sol=case.load_solution(path,late.MODEL)
        if (row["spec"]!=spec or not np.array_equal(sol.times,times) or sol.row["steps"]!=p["horizon"]*spec["rate"] or
                any(sol.row[k]!=v for k,v in dict(late.grid(spec),n=spec["n"]).items()) or
                sol.row["row_scale"]!=sol.model.scale_metadata()):
            raise RuntimeError("Reference reconstruction differs from its frozen parameters")
        fields.append(case.reference_fields(sol,grids))
    finest=fields[-1]
    denominators=norm_entries(finest,finest,grids,times)
    floors={(g,k):max(np.finfo(float).tiny,p["near_zero_relative_floor"]*max(
        r["metrics"][k] for r in denominators if r["grid"]==g)) for g in grids for k in METRICS}
    edges=[]
    for axis,a,b in zip(("radial","angular","time"),fields,fields[1:]):
        rows=norm_entries(a,b,grids,times,floors)
        worst,undefined=summarize(rows)
        edges.append(dict(axis=axis,rows=rows,worst=worst,undefined=undefined))
    direct=norm_entries(fields[0],finest,grids,times,floors)
    combined=[]
    for a,b,c,d in zip(*(e["rows"] for e in edges),direct,strict=True):
        if len({(r["grid"],r["time"]) for r in (a,b,c,d)})!=1:
            raise RuntimeError("Cannot compose different physical times or grids")
        values={}
        for k in METRICS:
            e=[r["metrics"][k]["relative_l2"] for r in (a,b,c)]
            upper=None if None in e else e[0]+(1+e[0])*(e[1]+(1+e[1])*e[2])
            measured=d["metrics"][k]["relative_l2"]
            if upper is not None and measured is not None and measured>upper+1e-14:
                raise ArithmeticError("Direct coarse/fine discrepancy exceeds its paired triangle bound")
            values[k]=dict(edge_relative_l2=e,relative_l2=upper,direct_relative_l2=measured)
        combined.append(dict(grid=d["grid"],time=d["time"],metrics=values))
    worst,undefined=summarize(combined)
    limit=p["gate"]*p["reference_budget_fraction"]
    passed=undefined==0 and max(worst.values())<=limit
    base.save(REFERENCE,dict(status="complete",passes=passed,budget=limit,worst=worst,undefined=undefined,
        edges=edges,composition=combined,direct=direct,minimum_fine_reference_norm=min(
            v for r in denominators for v in r["metrics"].values()),arm_hashes={r["artifact"]:r["sha256"] for r in arms},
        protocol_sha256=base.digest(FROZEN)))
    print("REFERENCE",passed,worst,"undefined",undefined,flush=True)
    return grids,finest,floors,passed


def rescore(frozen,grids,truth,floors,reference_pass):
    p=frozen["protocol"]
    times=p["checkpoints"]
    rows=[]
    for r in frozen["candidates"]:
        path=base.WORK/r["source"]
        hardware.sha_matches(path,r["sha256"])
        sol=case.load_solution(path,MODELS[r["arm"]])
        if not np.array_equal(sol.times[:len(times)],times) or sol.row["dt"]!=1/r["rate"]:
            raise RuntimeError("Saved candidate rate or early snapshots differ")
        early=case.Solution(sol.model,sol.states[:len(times)],sol.times[:len(times)],sol.probe_times,
                            sol.transverse,sol.vorticity,sol.row)
        scores=norm_entries(case.reference_fields(early,grids),truth,grids,times,floors)
        worst,undefined=summarize(scores)
        direct=undefined==0 and max(worst.values())<=p["gate"]
        rows.append(dict(**r,scores=scores,worst=worst,undefined=undefined,direct_pass=direct,
                         qualified=bool(direct and reference_pass)))
        print("RESCORE",r["cohort"],r["name"],r["seed"],r["rate"],r["arm"],rows[-1]["qualified"],worst,flush=True)
    groups=[]
    for key in dict.fromkeys((r["cohort"],r["name"],r["rate"],r["arm"]) for r in rows):
        cohort=[r for r in rows if (r["cohort"],r["name"],r["rate"],r["arm"])==key]
        values={k:max((r["worst"][k] for r in cohort if r["worst"][k] is not None),default=None) for k in METRICS}
        groups.append(dict(zip(("cohort","name","rate","arm"),key),count=len(cohort),
                           qualified_count=sum(r["qualified"] for r in cohort),worst=values))
    base.save(RESCORE,dict(status="complete",reference_pass=reference_pass,reference_audit_sha256=base.digest(REFERENCE),
        protocol_sha256=base.digest(FROZEN),rows=rows,groups=groups,
        scope="Physical field checks at six early times. Saved cohorts remain separate and are not independent repeated seeds."))


def execute(frozen):
    p=frozen["protocol"]
    result=dict(status="running",protocol_sha256=base.digest(FROZEN),arms=[])
    if RUNS.exists() and base.read(RUNS)["protocol_sha256"]!=result["protocol_sha256"]:
        raise RuntimeError("Early reference journal differs from its freeze")
    if RESCORE.exists():
        raise FileExistsError(RESCORE)
    base.save(RUNS,result)
    with ProcessPoolExecutor(max_workers=p["cpu_workers"]) as pool:
        futures=[pool.submit(run_arm,spec,p) for spec in p["reference_path"]]
        for task in as_completed(futures):
            result["arms"].append(task.result())
            base.save(RUNS,result)
    if freeze()!=frozen:
        raise RuntimeError("Startup reference inputs changed")
    rescore(frozen,*reference_audit(frozen,result["arms"]))
    result.update(status="complete",reference_audit_sha256=base.digest(REFERENCE),candidate_rescore_sha256=base.digest(RESCORE))
    base.save(RUNS,result)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-only",action="store_true")
    action.add_argument("--execute",action="store_true")
    args=cli.parse_args()
    fixed=freeze()
    print("FROZEN",base.digest(FROZEN),flush=True)
    if args.execute:
        execute(fixed)
