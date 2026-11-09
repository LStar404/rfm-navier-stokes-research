"""Audit the new reference and run the freshly selected six-method comparison."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import cpu_formal as base
import cpu_formal_joint as old_joint
import fd_pow2_reference_chain as chain
import joint_hardware as kernels

case, np = base.case, base.case.np
HERE = base.HERE
OUT = HERE/"hardware_pow2_output"
TEMPLATE = HERE/"protocol_joint_hardware_pow2.json"
FROZEN = OUT/"protocol.json"
PREFLIGHT = OUT/"preflight.json"
TIMING = OUT/"timing.json"
POST_AUDIT = OUT/"post_audit_manifest.json"
METRICS = ("velocity","vorticity","near_velocity","near_vorticity")
EXTRA_SOURCE_NAMES = ("cpu_formal_reference_point1.py", "cpu_formal_projection.py", "protocol_point1_ref.json")


def sha_matches(path, expected):
    if base.digest(path)!=expected:
        raise RuntimeError(f"Fingerprint changed: {path.name}")


def score_rows(rows, expected):
    keys=[(r["grid"],r["time"]) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError("Need one score row for each frozen grid and time")
    if any(set(r)!={"grid","time",*METRICS} for r in rows):
        raise ValueError("Expected exactly the four registered relative norms")
    if any(type(r[k]) not in (float,int) or not math.isfinite(r[k]) or r[k]<0 for r in rows for k in METRICS):
        raise ValueError("Invalid relative field error")
    return {k:max(r[k] for r in rows) for k in METRICS}


def cohort_seeds(protocol):
    rfm=set(protocol["seeds"])
    if len(rfm)!=5 or len(protocol["seeds"])!=5:
        raise RuntimeError("Expected five distinct preregistered seeds")
    groups={s["name"]:rfm if s["method"]=="rfm" else {protocol["deterministic_seed"]}
            for s in protocol["variants"]}
    if len(groups)!=9:
        raise RuntimeError("Expected nine preregistered hardware candidate variants")
    groups["rfm49_bandwidth98"]=rfm
    for suffix in ("baseline", "ordinary", "tangent"):
        groups[f"legendre49_{suffix}"]={7331}
    for name in ("fd240_k120_rate320","fd280_k104_rate256","fd320_k104_rate256","fd320_k112_rate320"):
        groups[name]={None}
    return groups


def audit_inputs():
    journal=base.read(chain.JOURNAL)
    if journal["status"]!="complete":
        raise RuntimeError("The live reference chain must finish first")
    inputs=journal["frozen"]
    for name,sha in inputs["source_hashes"].items():
        sha_matches(base.WORK/name,sha)
    if inputs["protocol"]!=base.read(chain.PROTOCOL):
        raise RuntimeError("Reference protocol differs from the frozen chain")
    protocol=inputs["protocol"]
    check=set((g,t) for g in base.grid_set() for t in protocol["checkpoints"])
    if len(check)!=21 or protocol["checkpoints"]!=list(case.SNAPSHOTS):
        raise RuntimeError("Expected the three original grids and seven times")
    paths=protocol["reference_path"]
    if len(paths)!=4 or len({s["name"] for s in paths})!=4 or len({s["source"] for s in paths})!=4:
        raise RuntimeError("Expected four distinct independent reference artifacts")
    for a,b,key in zip(paths,paths[1:],("n","k","rate")):
        if [k for k in ("n","k","rate") if a[k]!=b[k]]!=[key] or a[key]>=b[key]:
            raise RuntimeError("Expected a single independently refined axis on each edge")
    audit_path=chain.OUT/journal["reference_audit"]
    audit=base.read(audit_path)
    edges=audit["edges"]
    if [e["axis"] for e in edges]!=["radial","angle","time"] or not audit["independent_axes"]:
        raise RuntimeError("Reference audit axes changed")
    maxima=[]
    for e in edges:
        worst=score_rows(e["scores"],check)
        if e["worst"]!=worst:
            raise ValueError("Reference edge maxima do not match raw values")
        maxima.append(max(worst.values()))
    if audit["direct"]["worst"]!=score_rows(audit["direct"]["scores"],check):
        raise ValueError("Direct reference difference does not match raw values")
    composed=math.prod(1+e for e in maxima)-1
    limit=protocol["gate"]*protocol["reference_budget_fraction"]
    if (composed!=audit["composed_difference"] or composed!=journal["composed_difference"]
            or audit["threshold"]!=limit or audit["passes"]!=(composed<=limit)
            or journal["reference_screen"]!=audit["passes"]):
        raise RuntimeError("Recorded reference composition or decision changed")
    if (len(journal["rows"])!=2 or any(not any(r["status"]=="complete" and r["spec"]==s
            for r in journal["rows"]) for s in paths[2:])):
        raise RuntimeError("Expected one complete frozen journal record per new reference arm")
    references=chain.references(protocol,journal["rows"])
    fine=references[-1]
    if not np.array_equal(fine.times,protocol["checkpoints"]):
        raise RuntimeError("Fine reference checkpoint times changed")
    source_inputs={(r["name"],r["seed"]):(r["path"],r["sha256"]) for r in inputs["candidate_inputs"]}
    if len(source_inputs)!=33 or len(inputs["candidate_inputs"])!=33:
        raise RuntimeError("Expected 33 distinct frozen candidates")
    rescore_path=chain.OUT/journal["candidate_rescore"]
    rescored=base.read(rescore_path)
    rows={}
    for row in rescored["rows"]:
        key=(row["name"],row["seed"])
        if key in rows or source_inputs.get(key)!=(row["source"],row["sha256"]):
            raise RuntimeError("Rescore did not cover the frozen candidate artifacts")
        sha_matches(base.WORK/row["source"],row["sha256"])
        worst=score_rows(row["scores"],check)
        ok=max(worst.values())<=protocol["gate"] and row["wake"]["sustained_alternating_probe_signs"]
        if (row["worst"]!=worst or row["numerical_pass"]!=ok or
                row["reference_screen"]!=audit["passes"]):
            raise RuntimeError("Rescored maxima or 0.1% decision changed")
        rows[key]=row
    if set(rows)!=set(source_inputs):
        raise RuntimeError("Missing or extra rescore rows")
    expected_groups=cohort_seeds(base.read(base.PROTOCOL))
    if set(rows)!={(name,seed) for name,seeds in expected_groups.items() for seed in seeds}:
        raise RuntimeError("Rescore candidate or seed inventory changed")
    groups={r["name"]:r for r in rescored["summary"]}
    if len(groups)!=len(rescored["summary"]) or set(groups)!=set(expected_groups):
        raise RuntimeError("Missing or duplicate groups")
    for name,seeds in expected_groups.items():
        group=[r for (n,_),r in rows.items() if n==name]
        expected=len(seeds)
        ok=len(group)==expected and all(r["numerical_pass"] for r in group)
        worst={k:max(r["worst"][k] for r in group) for k in METRICS}
        info=groups[name]
        if (info["count"]!=len(group) or info["expected_count"]!=expected or info["worst"]!=worst or
            info["reference_screen"]!=audit["passes"] or
            info["numerical_pass"]!=ok or info["qualified"]!=(ok and audit["passes"])):
            raise RuntimeError("Recorded rescore group summary does not match raw records")
    ref_hashes={r["source"]:base.digest(base.WORK/r["source"]) for r in paths}
    files={p.relative_to(base.WORK).as_posix():base.digest(p)
           for p in (chain.JOURNAL,audit_path,rescore_path)}
    return fine,rows,dict(composed_difference=composed,reference_hashes=ref_hashes,
        files=files,audit_gate=protocol["gate"],audit_passes=audit["passes"],
        source_hashes=inputs["source_hashes"],
        qualified_at_chain_gate=[g["name"] for g in rescored["summary"] if g["qualified"]])


def post_audit_manifest(reference):
    extra={f"rfm_v7/{name}":base.digest(HERE/name) for name in EXTRA_SOURCE_NAMES}
    journal=chain.JOURNAL.relative_to(base.WORK).as_posix()
    fixed=dict(chain_journal=journal,chain_journal_sha256=reference["files"][journal],
        verified_live_source_hashes=reference["source_hashes"],
        transitive_source_hashes_observed_after_run=extra,
        scope="These supplemental hashes were observed after chain completion and do not amend the original live freeze.")
    if POST_AUDIT.exists():
        saved=base.read(POST_AUDIT)
        if any(saved[k]!=v for k,v in fixed.items()):
            raise RuntimeError("Post-audit manifest no longer matches")
    else:
        base.save(POST_AUDIT,dict(fixed,observed_utc=datetime.now(timezone.utc).isoformat()))
    return base.digest(POST_AUDIT)


def select(protocol,template,rows):
    choices=[]
    evidence=[]
    specs={s["name"]:s for s in protocol["variants"]}
    for spec in specs.values():
        name,method=spec["name"],spec["method"]
        required=protocol["seeds"] if method=="rfm" else [protocol["deterministic_seed"]]
        if {seed for n,seed in rows if n==name}!=set(required):
            raise RuntimeError(f"Frozen candidate seed set changed for {name}")
        cohort=[rows[(name,seed)] for seed in required]
        worst={key:max(r["worst"][key] for r in cohort) for key in METRICS}
        qualifies=max(worst.values())<=template["gate"] and all(
            r["wake"]["sustained_alternating_probe_signs"] for r in cohort)
        evidence.append(dict(name=name,method=method,seeds=required,worst=worst,
                             qualifies=qualifies,artifacts=[r["sha256"] for r in cohort]))
    for method in template["family_order"]:
        choices.append(next((r for r in evidence if r["method"]==method and r["qualifies"]),None))
    if any(c is None for c in choices):
        raise RuntimeError("The complete three-family 0.5% experiment needs an eligible variant per family")
    return choices,evidence,specs


def source_hashes():
    names=(Path(__file__),TEMPLATE,*(HERE/name for name in EXTRA_SOURCE_NAMES))
    return kernels.hashes()|{p.relative_to(base.WORK).as_posix():base.digest(p) for p in names}


def freeze():
    template=base.read(TEMPLATE)
    protocol=dict(base.read(base.PROTOCOL),gate=template["gate"])
    if template["graph_steps"]!=kernels.GRAPH_STEPS or template["cuda_fd_linear"]!="dense":
        raise RuntimeError("Hardware kernel settings differ from the frozen template")
    fine,rows,reference=audit_inputs()
    reference["files"][POST_AUDIT.relative_to(base.WORK).as_posix()]=post_audit_manifest(reference)
    reference_limit=template["gate"]*template["reference_budget_fraction"]
    if reference["composed_difference"]>reference_limit:
        raise RuntimeError("The new independent reference failed the 0.5% screen")
    choices,evidence,specs=select(protocol,template,rows)
    rng=np.random.default_rng(template["order_seed"])
    methods=[(backend,c) for backend in template["backends"] for c in choices]
    schedule=[]
    for block,seed in enumerate(protocol["seeds"],1):
        for i in rng.permutation(len(methods)):
            backend,c=methods[i]
            schedule.append(dict(block=block,backend=backend,name=c["name"],
                seed=seed if c["method"]=="rfm" else protocol["deterministic_seed"]))
    fixed=dict(template=template,protocol=protocol,reference=reference,reference_limit=reference_limit,
               choices=choices,all_candidate_decisions=evidence,schedule=schedule,source_hashes=source_hashes())
    if FROZEN.exists():
        if base.read(FROZEN)!=fixed:
            raise RuntimeError("Frozen 0.5% hardware selection or implementation changed")
    else:
        base.save(FROZEN,fixed)
    return fixed,fine,rows,specs


def control(row,spec,protocol,grids):
    path=base.WORK/row["source"]
    sha_matches(path,row["sha256"])
    cls,settings=base.config(spec,protocol,row["seed"])
    with np.load(path,allow_pickle=False) as data:
        archived=json.loads(str(data["config"]))
    if settings!=archived:
        raise RuntimeError(f"Qualification parameters do not match selected spec: {path.name}")
    sol=case.load_solution(path,cls)
    if (sol.row["steps"]!=protocol["horizon"]*spec["rate"] or
            not np.array_equal(sol.times,protocol["checkpoints"])):
        raise RuntimeError("Qualification time discretization differs")
    return sol,case.reference_fields(sol,grids)


def run_one(job,spec,protocol,template,grids,truth,old,archive=None):
    item=dict(job)
    started=time.perf_counter()
    try:
        sol,settings=kernels.solve(spec,protocol,job["seed"],job["backend"])
        elapsed=time.perf_counter()-started
        previous,previous_fields=old
        if not np.array_equal(sol.times,previous.times) or not np.array_equal(
                sol.probe_times,previous.probe_times):
            raise RuntimeError("Timed solver changed the registered observation times")
        scores=case.score(sol,truth,grids)
        expected={(g,t) for g in grids for t in protocol["checkpoints"]}
        worst=score_rows(scores,expected)
        implementation=score_rows(case.score(sol,previous_fields,grids),expected)
        probes={k:float(np.linalg.norm(getattr(sol,k)-getattr(previous,k))/
                        np.linalg.norm(getattr(previous,k))) for k in ("transverse","vorticity")}
        if not all(math.isfinite(v) for v in probes.values()):
            raise ValueError("Nonfinite implementation probe discrepancy")
        wake=base.wake(sol)
        item.update(metadata=sol.row,scores=scores,worst=worst,wake=wake,boundaries=base.boundary_errors(sol),
            implementation_field_error=implementation,implementation_probe_error=probes,
            implementation_pass=max(*implementation.values(),*probes.values())<=template["implementation_tolerance"],
            numerical_pass=max(worst.values())<=template["gate"] and wake["sustained_alternating_probe_signs"],
            reconstruction_class="FullFD" if spec["method"]=="fd4" else "RecurrenceGalerkin")
        if archive is not None:
            temp=archive.with_suffix(".tmp.npz")
            case.store_solution(temp,sol,settings)
            temp.replace(archive)
            item.update(seconds=elapsed,artifact=archive.name,sha256=base.digest(archive))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError,RuntimeError,OSError) as exc:
        item.update(failure=str(exc),numerical_pass=False,implementation_pass=False)
    return item


def approved_inputs(fixed):
    if source_hashes()!=fixed["source_hashes"]:
        raise RuntimeError("Timed implementation source changed")
    for key in ("reference_hashes","files","source_hashes"):
        for name,sha in fixed["reference"][key].items():
            sha_matches(base.WORK/name,sha)


def preflight(fixed,rows,specs,grids,truth):
    protocol,template=fixed["protocol"],fixed["template"]
    jobs=fixed["schedule"][:len(fixed["choices"])*len(template["backends"])]
    protocol_hash=base.digest(FROZEN)
    if PREFLIGHT.exists():
        result=base.read(PREFLIGHT)
        if (result["protocol_sha256"]!=protocol_hash or len(result["rows"])>len(jobs) or not all(
                all(r[k]==v for k,v in job.items()) for r,job in zip(result["rows"],jobs))):
            raise RuntimeError("Stored preflight does not match the frozen schedule")
    else:
        result=dict(protocol_sha256=protocol_hash,status="running",rows=[])
        base.save(PREFLIGHT,result)
    for job in jobs[len(result["rows"]):]:
        spec=specs[job["name"]]
        old=control(rows[job["name"],job["seed"]],spec,protocol,grids)
        row=run_one(job,spec,protocol,template,grids,truth,old)
        result["rows"].append(row)
        base.save(PREFLIGHT,result)
        print("PREFLIGHT",job["backend"],job["name"],row.get("worst",row.get("failure")),flush=True)
    approved_inputs(fixed)
    result.update(status="complete",passes=len(result["rows"])==len(jobs) and all(
        r["implementation_pass"] and r["numerical_pass"] for r in result["rows"]))
    base.save(PREFLIGHT,result)


def aggregate(rows,fixed):
    output=[]
    for choice in fixed["choices"]:
        for backend in fixed["template"]["backends"]:
            group=[r for r in rows if r["name"]==choice["name"] and r["backend"]==backend]
            good=[r for r in group if "seconds" in r]
            expected=len(fixed["protocol"]["seeds"])
            passed=len(group)==expected and len(good)==expected and all(
                r["implementation_pass"] and r["numerical_pass"] for r in good)
            values=[r["seconds"] for r in good]
            output.append(dict(name=choice["name"],backend=backend,passes=passed,count=len(group),
                failures=sum("failure" in r for r in group),samples_seconds=values,
                median_seconds=statistics.median(values) if values else None,
                min_seconds=min(values) if values else None,max_seconds=max(values) if values else None,
                worst={k:max(r["worst"][k] for r in good) for k in METRICS} if good else None))
    return output


def timing(fixed,rows,specs,grids,truth):
    checked=base.read(PREFLIGHT)
    if (checked["status"]!="complete" or not checked["passes"] or checked["protocol_sha256"]!=base.digest(FROZEN)
            or len(checked["rows"])!=6 or not all(
                all(r[k]==v for k,v in j.items()) and r["implementation_pass"] and r["numerical_pass"]
                for r,j in zip(checked["rows"],fixed["schedule"][:6]))):
        raise RuntimeError("Run the matched implementation preflight for all six methods first")
    cpu=base.pin()
    protocol,template=fixed["protocol"],fixed["template"]
    if TIMING.exists():
        result=base.read(TIMING)
        if result["frozen"]!=fixed or result["cpu"]!=cpu:
            raise RuntimeError("Frozen experiment or pinned CPU differs")
        for row in result["rows"]:
            if "artifact" in row:
                sha_matches(OUT/row["artifact"],row["sha256"])
    else:
        result=dict(status="running",frozen=fixed,rows=[],cpu=cpu,gpu=old_joint.nvidia(),
                    platform=platform.platform(),environment={
                        k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
        base.save(TIMING,result)
    key=lambda r:(r["block"],r["name"],r["backend"])
    known={key(r) for r in result["rows"]}
    jobs={key(j):j for j in fixed["schedule"]}
    if len(known)!=len(result["rows"]) or any(key(r) not in jobs or
            any(r[k]!=v for k,v in jobs[key(r)].items()) for r in result["rows"]):
        raise RuntimeError("Stored timing rows differ from the frozen schedule")
    previous={}
    np.linalg.svd(np.eye(10))
    for job in fixed["schedule"]:
        if key(job) in known:
            continue
        identity=job["name"],job["seed"]
        spec=specs[job["name"]]
        if identity not in previous:
            previous[identity]=control(rows[identity],spec,protocol,grids)
        path=OUT/f"block{job['block']}_{job['backend']}_{job['name']}_{job['seed']}.npz"
        record=path.with_suffix(".json")
        if record.exists():
            row=base.read(record)
            if any(row[k]!=v for k,v in job.items()):
                raise RuntimeError("Saved job does not match the frozen block")
            if "artifact" in row:
                sha_matches(path,row["sha256"])
        else:
            if path.exists():
                raise FileExistsError(f"Inspect the orphaned artifact {path.name}")
            row=run_one(job,spec,protocol,template,grids,truth,previous[identity],archive=path)
            base.save(record,row)
        result["rows"].append(row)
        base.save(TIMING,result)
        print(job["block"],job["backend"],job["name"],row.get("seconds"),
              row.get("worst",row.get("failure")),flush=True)
    approved_inputs(fixed)
    if {key(r) for r in result["rows"]}!=set(jobs):
        raise RuntimeError("The completed block inventory differs")
    result.update(status="complete",summary=aggregate(result["rows"],fixed))
    base.save(TIMING,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","preflight","timing"))
    parser.add_argument("--confirm-isolated",action="store_true")
    args=parser.parse_args()
    if args.phase=="timing" and not args.confirm_isolated:
        parser.error("Stop competing experiments and confirm isolation")
    fixed,fine,rows,specs=freeze()
    print("CHOICES",[(c["name"],c["worst"]) for c in fixed["choices"]],flush=True)
    if args.phase!="freeze":
        grids=base.grid_set()
        truth=case.reference_fields(fine,grids)
        if args.phase=="preflight":
            preflight(fixed,rows,specs,grids,truth)
        else:
            timing(fixed,rows,specs,grids,truth)


if __name__=="__main__":
    main()
