"""Independently verify saved switch qualification fields without a time solve."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import switch_qualification as source


base,case,np=source.base,source.case,source.np
HERE=source.HERE
OUTPUT=HERE/"switch_qualification_verification.json"
METRICS=("velocity","vorticity","near_velocity","near_vorticity")
IDENTITY=("name","seed","arm","role")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def same(a,b):
    return math.isclose(a,b,rel_tol=2e-8,abs_tol=2e-13)


def key(job):
    return tuple(job[k] for k in IDENTITY)


def indexed(rows,expected):
    keys=[(r["grid"],r["time"]) for r in rows]
    require(len(keys)==len(expected) and set(keys)==expected,"Missing, duplicate or shifted grid/time score")
    require(all(set(r)=={"grid","time",*METRICS} and all(type(r[k]) in (float,int) and
                 math.isfinite(r[k]) and r[k]>=0 for k in METRICS) for r in rows),"Invalid score vector")
    return dict(zip(keys,rows,strict=True))


def maxima(rows):
    return {k:max(r[k] for r in rows.values()) for k in METRICS}


def compare_scores(observed,expected):
    require(set(observed)==set(expected) and all(same(row[k],expected[gt][k])
                 for gt,row in observed.items() for k in METRICS),"Recomputed physical or implementation scores differ")


def checksums(paths):
    for path,sha in paths.items():
        require(base.digest(base.WORK/path)==sha,f"Changed input hash: {path}")


def raw_arrays(row,frozen):
    job=row["job"]
    expected_stem="_".join(str(job[k]) for k in ("role","name","seed","arm"))
    require(base.read(source.OUT/(expected_stem+".json"))==row,"Raw run sidecar differs from the journal")
    if row["status"]!="complete":
        require(row["status"]=="failed" and bool(row.get("failure")),"Run failure is missing its evidence")
        return None
    path=source.OUT/row["artifact"]
    require(path.name==expected_stem+".npz" and base.digest(path)==row["sha256"],"Raw trajectory hash or name differs")
    expected_times=frozen["physical_times"][:12] if job["role"]=="anchor" else frozen["checkpoints"]
    horizon=16 if job["role"]=="anchor" else 72
    with np.load(path,allow_pickle=False) as npz:
        info=json.loads(str(npz["metadata"]))
        config=json.loads(str(npz["config"]))
        _,correct=base.config(frozen["specs"][job["name"]],frozen["solve_protocol"],job["seed"])
        values=SimpleNamespace(**{k:npz[k] for k in ("times","probe_times","transverse","vorticity","states")})
    probes=np.arange(horizon*8+1)/8
    require(info==row["metadata"] and config==correct and info["horizon"]==horizon and
            info["steps"]==(2048 if job["role"]=="anchor" else 8192) and
            np.array_equal(values.times,expected_times) and np.array_equal(values.probe_times,probes) and
            values.states.shape==(len(expected_times),info["rank"],info["k"]+1) and
            values.transverse.shape==values.vorticity.shape==(len(probes),3) and
            all(np.isfinite(getattr(values,k)).all() for k in ("states","transverse","vorticity")),
            "Stored configuration, observation times or trajectory shape differs")
    if job["role"]=="candidate":
        require(info["startup_rate"]==128 and info["late_rate"]==112 and info["switch_step"]==1024 and
                info["nonlinear_evaluations"]==8191 and info["marching_fft_batches"]==16382 and
                info["nonlinear_by_phase"]=={"startup":1024,"late":7167},"Work schedule changed")
        if job["arm"]=="cuda_graph":
            require(info["priming_fft_batches"]==2 and info["bridge_extra_fft_batches"]==0 and
                    info["device_final_step"]==8192 and info["graph_steps"]==24,"CUDA schedule changed")
    return values


def absolute_reference(frozen,grids):
    p=frozen["template"]
    ec=base.read(source.early.FROZEN)
    er=base.read(source.early.RUNS)
    ea=base.read(source.early.REFERENCE)
    late=source.original.chain
    lr=base.read(late.JOURNAL)
    la=base.read(late.OUT/lr["reference_audit"])
    early_times,late_times=frozen["physical_times"][:12],frozen["physical_times"][12:]
    require(ec["protocol"]["checkpoints"]==early_times and lr["frozen"]["protocol"]["checkpoints"]==late_times and
            er["reference_pass"] and er["bitwise_t8_pass"] and ea["passes"] and ea["undefined"]==0 and
            er["protocol_sha256"]==ea["protocol_sha256"]==base.digest(source.early.FROZEN) and
            er["reference_audit_sha256"]==base.digest(source.early.REFERENCE),"Early or late reference identity differs")
    require(len(er["arms"])==4 and all(r["bitwise_t8_pass"] and all(r["replays_original_t8"].values()) for r in er["arms"])
            and ea["arm_hashes"]=={r["artifact"]:r["sha256"] for r in er["arms"]},"T16 history evidence differs")
    early_arms={r["spec"]["name"]:(source.early.OUT/r["artifact"],r["sha256"]) for r in er["arms"]}
    frozen_arms={k:(base.WORK/r["path"],r["sha256"]) for k,r in frozen["early_references"].items()}
    require(early_arms==frozen_arms and len(early_arms)==4,"Frozen T16 arm inventory differs")
    for path,sha in early_arms.values():
        require(base.digest(path)==sha,"T16 reference artifact changed")
    targets={(grid,t) for grid in grids for t in early_times}
    require([(e["axis"],e["undefined"]) for e in ea["edges"]]==[("radial",0),("angular",0),("time",0)] and
            len(ea["composition"])==len(targets)==36,"Early axis scores missing")
    require({(r["grid"],r["time"]) for r in ea["composition"]}==targets,"Early composition times differ")
    axes=[]
    for edge in ea["edges"]:
        values={(r["grid"],r["time"]):r["metrics"] for r in edge["rows"]}
        require(len(edge["rows"])==len(values)==len(targets) and set(values)==targets,"Reference edge coverage differs")
        axes.append(values)
    for r in ea["composition"]:
        gt=(r["grid"],r["time"])
        for k in METRICS:
            e=[a[gt][k]["relative_l2"] for a in axes]
            require(r["metrics"][k]["edge_relative_l2"]==e and same(r["metrics"][k]["relative_l2"],
                    e[0]+(1+e[0])*(e[1]+(1+e[1])*e[2])),"Early triangle composition differs")
    early_max=max(v["relative_l2"] for row in ea["composition"] for v in row["metrics"].values())
    late_targets={(grid,t) for grid in grids for t in late_times}
    require([e["axis"] for e in la["edges"]]==["radial","angle","time"] and la["independent_axes"],"Late axes differ")
    late_edge_max=[max(maxima(indexed(e["scores"],late_targets)).values()) for e in la["edges"]]
    late_bound=math.prod(1+x for x in late_edge_max)-1
    budget=p["physical_gate"]*p["reference_budget_fraction"]
    require(early_max==max(ea["worst"].values()) and same(late_bound,la["composed_difference"]) and
            max(early_max,late_bound)<=budget,"An independent reference did not meet the 0.5% budget")
    late_by_name={r["name"]:r for r in lr["frozen"]["protocol"]["reference_path"]}
    truth={}
    for name in ("angular_fine","time_fine"):
        old=late_by_name[name]
        e=case.load_solution(early_arms[name][0],source.Pow2EquilibratedBufferedFD)
        path=base.WORK/old["source"]
        require(base.digest(path)==frozen["late_reference"]["reference_hashes"][old["source"]],
                "Selected late fine reference differs")
        l=case.load_solution(path,source.Pow2EquilibratedBufferedFD)
        for sol,times in ((e,early_times),(l,late_times)):
            require(np.array_equal(sol.times,times) and sol.row["row_scale"]==sol.model.scale_metadata(),
                    "Fine reference reconstruction or times differ")
        first,last=case.reference_fields(e,grids),case.reference_fields(l,grids)
        truth[name]={g:first[g]+last[g] for g in grids}
    return truth,dict(early_bound=early_max,late_bound=late_bound,budget=budget)


def measured(fields,truth,grids,times):
    rows={}
    for g,(s,_,weight) in grids.items():
        near=weight*(case.radius(s)<=2.5)
        for i,t in enumerate(times):
            a,b=fields[g][i],truth[g][i]
            diff=[(x-y)**2 for x,y in zip(a,b,strict=True)]
            denominator=[x*x for x in b]
            energy=(diff[0]+diff[1],diff[2])
            ref=(denominator[0]+denominator[1],denominator[2])
            values={}
            for k,index,w in zip(METRICS,(0,1,0,1),(weight,weight,near,near),strict=True):
                num=np.sum(energy[index]*w[:,None])
                denom=np.sum(ref[index]*w[:,None])
                require(np.isfinite(num) and np.isfinite(denom) and num>=0 and denom>0,"Invalid saved-field norms")
                values[k]=float(np.sqrt(num/denom))
            rows[g,t]=values
    return rows


def fields_at(sol,times,grids):
    indices={t:i for i,t in enumerate(sol.times)}
    require(len(indices)==len(sol.times) and all(t in indices for t in times),"Missing or duplicate physical checkpoint")
    return {g:[sol.fields(indices[t],s,theta) for t in times] for g,(s,theta,_) in grids.items()}


def verify():
    frozen=base.read(source.FROZEN)
    runs=base.read(source.RUNS)
    scored=base.read(source.SUMMARY)
    require(runs["status"]==scored["status"]=="complete" and not scored["isolated_speed_claim_supported"] and
            runs["protocol_sha256"]==scored["protocol_sha256"]==base.digest(source.FROZEN) and
            scored["runs_sha256"]==base.digest(source.RUNS),"Qualification manifest links differ")
    checksums(frozen["source_hashes"])
    for r in frozen["late_controls"]:
        checksums({r["path"]:r["sha256"]})
    p=frozen["template"]
    physical,implemented=frozen["physical_times"],frozen["checkpoints"]
    require(len(physical)==len(set(physical))==19 and len(implemented)==len(set(implemented))==21 and
            set(implemented)-set(physical)==set(p["bridge_only_checkpoints"]),"Bridge times were scored as physical times")
    roster={(name,seed,arm,"candidate") for name in p["methods"] for seed in
            (p["rfm_seeds"] if name.startswith("rfm") else [p["deterministic_seed"]]) for arm in p["arms"]}
    expected=roster|{(name,seed,"cpu_recurrence","anchor") for name,seed,_,_ in roster}
    raw={key(r["job"]):r for r in runs["rows"]}
    rows={key(r["job"]):r for r in scored["rows"]}
    require(len(runs["rows"])==len(raw)==len(frozen["jobs"])==24 and set(raw)==expected=={key(j) for j in frozen["jobs"]} and
            len(scored["rows"])==len(rows)==18 and set(rows)==roster,"Missing or duplicate switched identities")
    arrays={k:raw_arrays(r,frozen) for k,r in raw.items()}
    anchors={(r["name"],r["seed"]):r for r in scored["anchors"]}
    require(len(anchors)==len(scored["anchors"]),"Repeated anchor scores")
    prior={(r["job"]["name"],r["job"]["seed"]):r for r in frozen["late_controls"]}
    require(len(prior)==len(frozen["late_controls"])==6 and set(prior)=={(k[0],k[1]) for k in roster},
            "Missing or duplicate uniform late controls")
    grids=base.grid_set()
    truth,reference=absolute_reference(frozen,grids)
    physical_keys={(g,t) for g in grids for t in physical}
    implementation_keys={(g,t) for g in grids for t in implemented}
    failures=[]
    representatives=[]
    checked_anchors=set()
    early_positions=[implemented.index(t) for t in p["early_checkpoints"]]
    for k in sorted(roster):
        name,seed,arm,_=k
        akey=(name,seed,"cpu_recurrence","anchor")
        ckey=(name,seed,"cpu_recurrence","candidate")
        r=rows[k]
        if "failure" in r:
            require(not r["qualified"] and bool(r["failure"]),"A failed job was marked qualified")
            if arrays[k] is None:
                require(raw[k]["failure"]==r["failure"] or arrays[akey] is None or arrays[ckey] is None,
                        "A raw failure was replaced by an unrelated outcome")
            elif arrays[akey] is not None and arrays[ckey] is not None:
                require(not np.array_equal(arrays[akey].states[:6],arrays[ckey].states[early_positions]),
                        "Complete trajectory marked as a failure without a failed pre-switch prefix")
            failures.append(dict(job=r["job"],failure=r["failure"]))
            continue
        require(arrays[k] is not None and arrays[akey] is not None and arrays[ckey] is not None and
                r["artifact"]==raw[k]["artifact"] and r["sha256"]==raw[k]["sha256"],"Scored artifact differs from raw record")
        require(np.array_equal(arrays[akey].states[:6],arrays[ckey].states[early_positions]),
                "Scored group changed its pre-switch CPU prefix")
        anchor=anchors[name,seed]
        checked_anchors.add((name,seed))
        require(anchor["artifact"]==raw[akey]["artifact"] and anchor["sha256"]==raw[akey]["sha256"] and
                anchor["late_source"]==prior[name,seed],"Anchor score provenance differs")
        impl=indexed(r["implementation_scores"],implementation_keys)
        delta=indexed(r["baseline_difference"],physical_keys)
        require(r["implementation_worst"]==maxima(impl),"Stored implementation maxima differ")
        probe={k_:float(np.linalg.norm(getattr(arrays[k],k_)-getattr(arrays[ckey],k_))/np.linalg.norm(getattr(arrays[ckey],k_)))
               for k_ in ("transverse","vorticity")}
        require(all(same(v,r["full_probe_error"][k_]) for k_,v in probe.items()),"Stored full-horizon probe discrepancy differs")
        actual,guard={},{}
        for ref in ("angular_fine","time_fine"):
            actual[ref]=indexed(r["physical_scores"][ref],physical_keys)
            guard[ref]=indexed(r["guard_scores"][ref],physical_keys)
            anchors_rows=indexed(anchor["physical_scores"][ref],physical_keys)
            for gt in physical_keys:
                for metric in METRICS:
                    e=anchors_rows[gt][metric]
                    bound=delta[gt][metric]*(1+e)+e
                    require(same(bound,guard[ref][gt][metric]) and actual[ref][gt][metric]<=bound+2e-13,
                            "A grid/time/norm guard is inconsistent with the normalized triangle inequality")
            require(r["physical_worst"][ref]==maxima(actual[ref]) and r["guard_worst"][ref]==maxima(guard[ref]),
                    "Stored physical or guard maxima differ")
        wake=base.wake(arrays[k])
        imp=max(*r["implementation_worst"].values(),*probe.values())<=p["implementation_gate"]
        phy=all(max(x.values())<=p["physical_gate"] for x in r["physical_worst"].values())
        guard_pass=all(max(x.values())<=p["physical_gate"] for x in r["guard_worst"].values())
        accepted=bool(imp and phy and guard_pass and wake["sustained_alternating_probe_signs"])
        require(wake==r["wake"] and (imp,phy,guard_pass,accepted)==(r["implementation_pass"],r["physical_pass"],
                r["guard_pass"],r["qualified"]),"Saved physical/implementation qualification flags differ")
        if not accepted:
            failures.append(dict(job=r["job"],implementation_pass=imp,physical_pass=phy,guard_pass=guard_pass))
        if (name,seed) not in {("rfm37_k72",7331),("leg37_k72",17)}:
            continue
        sol=case.load_solution(source.OUT/r["artifact"],source.MODELS[arm])
        control=case.load_solution(source.OUT/raw[ckey]["artifact"],source.MODELS["cpu_recurrence"])
        old_early=case.load_solution(source.OUT/anchor["artifact"],source.MODELS["cpu_recurrence"])
        old_late=case.load_solution(base.WORK/prior[name,seed]["path"],source.MODELS["cpu_recurrence"])
        fields=fields_at(sol,physical,grids)
        compare_scores(measured(fields_at(sol,implemented,grids),fields_at(control,implemented,grids),grids,implemented),impl)
        left,right=fields_at(old_early,physical[:12],grids),fields_at(old_late,physical[12:],grids)
        anchor_fields={g:left[g]+right[g] for g in grids}
        recomputed_delta=measured(fields,anchor_fields,grids,physical)
        compare_scores(recomputed_delta,delta)
        for ref,truth_fields in truth.items():
            compare_scores(measured(fields,truth_fields,grids,physical),actual[ref])
            anchored=measured(anchor_fields,truth_fields,grids,physical)
            compare_scores(anchored,indexed(anchor["physical_scores"][ref],physical_keys))
            independent={gt:{k_:recomputed_delta[gt][k_]*(1+anchored[gt][k_])+anchored[gt][k_] for k_ in METRICS}
                         for gt in physical_keys}
            compare_scores(independent,guard[ref])
        representatives.append(r["job"])
    require(set(anchors)==checked_anchors,"Anchor score inventory differs")
    require(scored["passes"]==runs["passes"]==(not failures),"Cohort pass flag differs from preserved failures")
    return dict(status="passed",qualification_passes=scored["passes"],raw_jobs=len(raw),candidate_identities=len(rows),
                failures=failures,representative_fields_recomputed=representatives,reference=reference,
                implementation_grid_time_rows=len(implementation_keys),physical_grid_time_rows=len(physical_keys),
                source_hashes_verified=len(frozen["source_hashes"]),protocol_sha256=base.digest(source.FROZEN),
                runs_sha256=base.digest(source.RUNS),qualification_sha256=base.digest(source.SUMMARY),
                verifier_source_sha256=base.digest(Path(__file__)),time_solves_run=0)


if __name__=="__main__":
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result=verify()
    base.save(OUTPUT,result)
    print(result,flush=True)
