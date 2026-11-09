"""Qualify the bridge at the two outer spatial resolutions without timing claims."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path

import switch_qualification as mid
from verify_switch_qualification import absolute_reference


base,case,np=mid.base,mid.case,mid.np
HERE=mid.HERE
OUT=HERE/"multiscale_switch_qualification_output"
TEMPLATE=HERE/"protocol_multiscale_switch_qualification.json"
FROZEN=OUT/"protocol.json"
RUNS=OUT/"runs.json"
SUMMARY=OUT/"qualification.json"
SCALES=HERE/"cuda_multiscale_pow2_output"
MODELS=mid.MODELS
METRICS=mid.METRICS


def lookup_key(job):
    return job["resolution"],job["name"],job["seed"],job["arm"],job["role"]


def fixed_input(path):
    name,sha=mid.fingerprint(path)
    return dict(path=name,sha256=sha)


def uniform_matches(row,spec,seed,solve_protocol,alias):
    if (row["backend"]!="cpu" or row["seed"]!=seed or row["spec"]["name"]!=alias or
            any(row["spec"][k]!=spec[k] for k in ("method","m","k","q","rate")) or
            not row["implementation_pass"] or row["metadata"].get("basis_evaluation")!="differentiated-recurrence"):
        return False
    path=SCALES/row["artifact"]
    mid.old.sha_matches(path,row["sha256"])
    with np.load(path,allow_pickle=False) as data:
        config=json.loads(str(data["config"]))
        metadata=json.loads(str(data["metadata"]))
        _,expected=base.config(spec,solve_protocol,seed)
        return (config==expected and metadata==row["metadata"] and metadata["dt"]==1/spec["rate"] and
                metadata["horizon"]==72 and metadata["steps"]==72*spec["rate"] and metadata["time_order"]==2 and
                np.array_equal(data["times"],solve_protocol["checkpoints"]))


def freeze():
    p=base.read(TEMPLATE)
    previous=base.read(mid.FROZEN)
    prior=base.read(mid.SUMMARY)
    verified=base.read(HERE/"switch_qualification_verification.json")
    scale_protocol=base.read(SCALES/"protocol.json")
    scale=base.read(SCALES/"timing.json")
    if (not prior["passes"] or prior["protocol_sha256"]!=base.digest(mid.FROZEN) or
            not verified["qualification_passes"] or verified["qualification_sha256"]!=base.digest(mid.SUMMARY) or
            scale["status"]!="complete" or scale["protocol_sha256"]!=base.digest(SCALES/"protocol.json")):
        raise RuntimeError("Middle qualification or exact uniform archive is incomplete")
    old=p["resolutions"]
    seeds=previous["template"]["rfm_seeds"]
    expected=[("low",128,112,{"rfm35_k64":[seeds[0]],"leg35_k64":[17]}),
              ("high",152,136,{"rfm41_k80":seeds,"leg39_k80":[17]})]
    if ([(r["name"],r["startup_rate"],r["late_rate"],r["methods"]) for r in old]!=expected or
            any(p[k]!=previous["template"][k] for k in ("switch_time","horizon","probe_spacing","graph_steps",
                "physical_gate","implementation_gate","reference_budget_fraction","arms")) or p["cpu_workers_max"]!=3):
        raise ValueError("Unexpected outer-resolution configuration")
    solve_protocol=previous["solve_protocol"]
    specs={s["name"]:s for s in solve_protocol["variants"] if s["name"] in {n for r in old for n in r["methods"]}}
    physical=previous["physical_times"]
    sources={}
    mid.merge_hashes(sources,previous["source_hashes"])
    mid.merge_hashes(sources,scale_protocol["source_hashes"])
    mid.merge_hashes(sources,dict(mid.fingerprint(path) for path in (Path(__file__),TEMPLATE,mid.FROZEN,mid.RUNS,
                    mid.SUMMARY,HERE/"switch_qualification_verification.json",HERE/"verify_switch_qualification.py",
                    SCALES/"protocol.json",SCALES/"timing.json")))
    jobs=[]
    reused=[]
    schedules={}
    for resolution in old:
        group=resolution["name"]
        h0,h1=resolution["startup_rate"],resolution["late_rate"]
        offgrid=[p["switch_time"]+i/h1 for i in (1,2)]
        snaps=sorted(physical+offgrid)
        if len(set(snaps))!=21 or len(physical)!=19:
            raise ValueError("Unexpected mixed observation grid")
        count=p["switch_time"]*h0+(p["horizon"]-p["switch_time"])*h1
        schedules[group]=dict(physical_times=physical,implementation_times=snaps,bridge_only_times=offgrid,
                              startup_rate=h0,late_rate=h1,steps=count,nonlinear_evaluations=count-1,
                              marching_fft_batches=2*(count-1),uniform_startup_steps=p["horizon"]*h0)
        for name,cohort in resolution["methods"].items():
            spec=specs[name]
            if spec["rate"]!=h0:
                raise ValueError("Uniform anchor step rate differs from the original spatial ladder")
            alias="matched_leg35_k64" if name=="leg35_k64" else name
            for seed in cohort:
                group_key=dict(resolution=group,name=name,seed=seed)
                jobs.extend(dict(group_key,arm=arm,role="candidate") for arm in p["arms"])
                jobs.append(dict(group_key,arm="cpu_recurrence",role="anchor_early"))
                matching=next((row for row in scale["rows"] if uniform_matches(row,spec,seed,solve_protocol,alias)),None)
                if matching is None:
                    jobs.append(dict(group_key,arm="cpu_recurrence",role="anchor_late"))
                else:
                    path=SCALES/matching["artifact"]
                    record=fixed_input(path)
                    mid.merge_hashes(sources,{record["path"]:record["sha256"]})
                    reused.append(dict(group_key,archive_spec=matching["spec"],archive_block=matching["block"],
                                       archived_time_fine_worst=matching["reference_errors"],**record))
    if len(jobs)!=35 or len(reused)!=5 or len({lookup_key(j) for j in jobs})!=35:
        raise RuntimeError("Need 24 mixed solves, eight T16 anchors and three missing T72 anchors")
    result=dict(status="frozen",template=p,schedules=schedules,solve_protocol=solve_protocol,specs=specs,
                jobs=jobs,reused_late=reused,source_hashes=sources,middle_protocol=fixed_input(mid.FROZEN))
    if FROZEN.exists():
        if base.read(FROZEN)!=result:
            raise RuntimeError("Outer-resolution freeze changed")
    else:
        base.save(FROZEN,result)
    return result


def stem(job):
    return "_".join(str(x) for x in lookup_key(job))


def validate(sol,job,frozen):
    s=frozen["schedules"][job["resolution"]]
    role=job["role"]
    horizon=16 if role=="anchor_early" else 72
    times=s["implementation_times"] if role=="candidate" else (
          s["physical_times"][:12] if role=="anchor_early" else s["physical_times"][12:])
    steps=s["steps"] if role=="candidate" else s["startup_rate"]*horizon
    probes=np.arange(0.,horizon+.001,.125)
    if (sol.row["steps"]!=steps or sol.row["horizon"]!=horizon or sol.row["dt"]!=1/s["startup_rate"] or
            sol.row["time_order"]!=2 or not np.array_equal(sol.times,times) or
            not np.array_equal(sol.probe_times,probes) or sol.states.shape!=(len(times),*sol.model.rhs_shape) or
            sol.transverse.shape!=sol.vorticity.shape or sol.transverse.shape!=(len(probes),3) or
            not all(np.isfinite(getattr(sol,k)).all() for k in ("states","transverse","vorticity"))):
        raise ArithmeticError("Outer trajectory differs from frozen observation or finite-state schedule")
    if role=="candidate":
        expected=dict(startup_rate=s["startup_rate"],late_rate=s["late_rate"],switch_step=8*s["startup_rate"],
                      nonlinear_evaluations=s["nonlinear_evaluations"],marching_fft_batches=s["marching_fft_batches"],
                      nonlinear_by_phase=dict(startup=8*s["startup_rate"],late=steps-8*s["startup_rate"]-1))
        if job["arm"]=="cuda_graph":
            expected.update(device_final_step=steps,graph_steps=24,priming_fft_batches=2,bridge_extra_fft_batches=0)
        if any(sol.row[k]!=v for k,v in expected.items()):
            raise ArithmeticError("Outer nonlinear work differs from frozen schedule")


def run(job,frozen):
    path=OUT/(stem(job)+".npz")
    record=path.with_suffix(".json")
    if record.exists():
        row=base.read(record)
        if row["job"]!=job:
            raise RuntimeError("Saved outer record differs")
        if row["status"]=="complete":
            mid.old.sha_matches(path,row["sha256"])
        return row
    if path.exists():
        raise FileExistsError(path)
    p,s=frozen["template"],frozen["schedules"][job["resolution"]]
    _,cfg=base.config(frozen["specs"][job["name"]],frozen["solve_protocol"],job["seed"])
    row=dict(job=job)
    print("START",job,flush=True)
    try:
        if job["role"]=="candidate":
            args=dict(startup_rate=s["startup_rate"],late_rate=s["late_rate"],switch_time=p["switch_time"],
                      horizon=p["horizon"],snapshots=s["implementation_times"],probe_spacing=p["probe_spacing"],**cfg)
            sol=(mid.solve_cuda_switch(graph_steps=p["graph_steps"],**args) if job["arm"]=="cuda_graph" else
                 mid.switched.solve_cpu_switch(MODELS[job["arm"]],**args))
        else:
            early=job["role"]=="anchor_early"
            horizon=16 if early else p["horizon"]
            times=s["physical_times"][:12] if early else s["physical_times"][12:]
            sol=case.solve(MODELS["cpu_recurrence"],steps=horizon*s["startup_rate"],horizon=horizon,snapshots=times,**cfg)
        validate(sol,job,frozen)
        temp=path.with_suffix(".tmp.npz")
        case.store_solution(temp,sol,cfg)
        temp.replace(path)
        row.update(status="complete",artifact=path.name,sha256=base.digest(path),metadata=sol.row)
    except (ArithmeticError,RuntimeError,ValueError,OSError,np.linalg.LinAlgError) as exc:
        row.update(status="failed",failure=str(exc))
    base.save(record,row)
    print("DONE",job,row["status"],flush=True)
    return row


def load(row,frozen):
    path=OUT/row["artifact"]
    mid.old.sha_matches(path,row["sha256"])
    sol=case.load_solution(path,MODELS[row["job"]["arm"]])
    validate(sol,row["job"],frozen)
    return sol


def score(frozen):
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    paths=[OUT/(stem(j)+".json") for j in frozen["jobs"]]
    rows=[base.read(path) for path in paths]
    if [r["job"] for r in rows]!=frozen["jobs"]:
        raise RuntimeError("Raw outer job records differ from freeze")
    base.save(RUNS,dict(status="complete",protocol_sha256=base.digest(FROZEN),rows=rows))
    raw={lookup_key(r["job"]):r for r in rows}
    reused={(r["resolution"],r["name"],r["seed"]):r for r in frozen["reused_late"]}
    p=frozen["template"]
    mid.old.sha_matches(base.WORK/frozen["middle_protocol"]["path"],frozen["middle_protocol"]["sha256"])
    grids=base.grid_set()
    truth,reference=absolute_reference(base.read(mid.FROZEN),grids)
    results=[]
    anchors=[]
    for g in p["resolutions"]:
        group=g["name"]
        s=frozen["schedules"][group]
        times=s["physical_times"]
        physical_keys={(key,t) for key in grids for t in times}
        implementation_keys={(key,t) for key in grids for t in s["implementation_times"]}
        for name,seeds in g["methods"].items():
            for seed in seeds:
                group_key=(group,name,seed)
                anchor=raw[*group_key,"cpu_recurrence","anchor_early"]
                ctrl=raw[*group_key,"cpu_recurrence","candidate"]
                late=raw.get((*group_key,"cpu_recurrence","anchor_late"))
                group_rows=[raw[*group_key,arm,"candidate"] for arm in p["arms"]]
                failure=next((r["failure"] for r in (anchor,ctrl,late) if r is not None and r["status"]!="complete"),None)
                if failure is None:
                    first,control=load(anchor,frozen),load(ctrl,frozen)
                    if late is None:
                        prior=reused[group_key]
                        path=base.WORK/prior["path"]
                        mid.old.sha_matches(path,prior["sha256"])
                        last=case.load_solution(path,MODELS["cpu_recurrence"])
                    else:
                        prior=dict(artifact=late["artifact"],sha256=late["sha256"])
                        last=load(late,frozen)
                    if (not np.array_equal(last.times,times[12:]) or not mid.switched.equal_spatial_models((first.model,last.model)) or
                            not np.array_equal(mid.select(control,times[:6]).states,first.states[:6])):
                        failure="Uniform baseline or mixed pre-switch prefix differs"
                if failure is not None:
                    results.extend(dict(job=r["job"],qualified=False,failure=failure) for r in group_rows)
                    continue
                baseline=case.Solution(first.model,np.concatenate((first.states,last.states)),np.asarray(times),
                                       last.probe_times,last.transverse,last.vorticity,last.row)
                control_fields=case.reference_fields(control,grids)
                baseline_fields=case.reference_fields(baseline,grids)
                baseline_scores={ref:case.score(baseline,fields,grids) for ref,fields in truth.items()}
                baseline_worst={ref:mid.old.score_rows(scores,physical_keys) for ref,scores in baseline_scores.items()}
                anchors.append(dict(resolution=group,name=name,seed=seed,early_artifact=anchor["artifact"],
                    early_sha256=anchor["sha256"],late=prior,physical_scores=baseline_scores,physical_worst=baseline_worst))
                for raw_row in group_rows:
                    job=raw_row["job"]
                    if raw_row["status"]!="complete":
                        results.append(dict(job=job,qualified=False,failure=raw_row["failure"]))
                        continue
                    sol=load(raw_row,frozen)
                    impl=case.score(sol,control_fields,grids)
                    iw=mid.old.score_rows(impl,implementation_keys)
                    probes={k:float(np.linalg.norm(getattr(sol,k)-getattr(control,k))/np.linalg.norm(getattr(control,k)))
                            for k in ("transverse","vorticity")}
                    physical=mid.select(sol,times)
                    direct={ref:case.score(physical,fields,grids) for ref,fields in truth.items()}
                    dw={ref:mid.old.score_rows(scores,physical_keys) for ref,scores in direct.items()}
                    delta=case.score(physical,baseline_fields,grids)
                    mid.old.score_rows(delta,physical_keys)
                    guard={}
                    for ref,values in baseline_scores.items():
                        index={(r["grid"],r["time"]):r for r in values}
                        guard[ref]=[dict(grid=d["grid"],time=d["time"],**{
                            k:d[k]*(1+index[d["grid"],d["time"]][k])+index[d["grid"],d["time"]][k] for k in METRICS})
                            for d in delta]
                    gw={ref:mid.old.score_rows(scores,physical_keys) for ref,scores in guard.items()}
                    wake=base.wake(sol)
                    ip=all(math.isfinite(x) and x<=p["implementation_gate"] for x in (*iw.values(),*probes.values()))
                    dp=all(max(x.values())<=p["physical_gate"] for x in dw.values())
                    gp=all(max(x.values())<=p["physical_gate"] for x in gw.values())
                    result=dict(job=job,artifact=raw_row["artifact"],sha256=raw_row["sha256"],implementation_scores=impl,
                        implementation_worst=iw,full_probe_error=probes,physical_scores=direct,physical_worst=dw,
                        baseline_difference=delta,guard_scores=guard,guard_worst=gw,wake=wake,
                        implementation_pass=ip,physical_pass=dp,guard_pass=gp,
                        qualified=bool(ip and dp and gp and wake["sustained_alternating_probe_signs"]))
                    results.append(result)
                    print("SCORE",job,result["qualified"],dw,flush=True)
    if len(results)!=24 or len({lookup_key(r["job"]) for r in results})!=24:
        raise RuntimeError("Outer qualification scores omitted an arm")
    summary={group:dict(qualified=sum(r["qualified"] for r in results if r["job"]["resolution"]==group),
                        count=sum(r["job"]["resolution"]==group for r in results)) for group in frozen["schedules"]}
    base.save(SUMMARY,dict(status="complete",protocol_sha256=base.digest(FROZEN),reference=reference,
        runs_sha256=base.digest(RUNS),
        raw_records={path.name:base.digest(path) for path in paths},rows=results,anchors=anchors,summary=summary,
        isolated_speed_claim_supported=False,scope=p["scope"]))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-only",action="store_true")
    action.add_argument("--cpu",action="store_true")
    action.add_argument("--gpu",action="store_true")
    action.add_argument("--score",action="store_true")
    cli.add_argument("--workers",type=int,default=3,choices=(1,2,3))
    args=cli.parse_args()
    fixed=freeze()
    print("FROZEN",base.digest(FROZEN),len(fixed["jobs"]),flush=True)
    if args.cpu:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cpu=[pool.submit(run,j,fixed) for j in fixed["jobs"] if j["arm"]!="cuda_graph"]
            for task in as_completed(cpu):
                task.result()
    elif args.gpu:
        for job in (j for j in fixed["jobs"] if j["arm"]=="cuda_graph"):
            run(job,fixed)
    elif args.score:
        score(fixed)
