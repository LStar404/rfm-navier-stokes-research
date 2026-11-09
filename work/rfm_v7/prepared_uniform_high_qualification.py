"""Qualify the original uniform high-resolution CUDA arm on all six identities."""

import argparse
import json
import os
from pathlib import Path

from basis_recurrence_model import native_model
from galerkin_cuda_native import solve_native
import prepared_qualification as prepared

base, case, np, mid = prepared.base,prepared.case,prepared.np,prepared.mid
require, record = prepared.require,prepared.record
HERE = prepared.HERE
OUT = HERE / "prepared_uniform_high_qualification_output"
TEMPLATE = HERE / "protocol_prepared_uniform_high_qualification.json"
FROZEN,RUNS,SUMMARY = (OUT / name for name in ("protocol.json","runs.json","qualification.json"))


def freeze():
    require(not FROZEN.exists(),"Preserve the existing high uniform qualification freeze")
    p = base.read(TEMPLATE)
    parent, parent_result = base.read(prepared.FROZEN),base.read(prepared.SUMMARY)
    q = parent["template"]
    require(parent_result["status"] == "complete" and parent_result["protocol_sha256"] == base.digest(prepared.FROZEN) and
            parent_result["runs_sha256"] == base.digest(prepared.RUNS) and "high" in parent_result["timing_eligible_scales"],
            "Prepared high switch qualification is incomplete")
    require(all(p[k] == q[k] for k in ("physical_times","horizon","probe_spacing","graph_steps","physical_gate",
                                      "implementation_gate","reference_budget_fraction","references")) and
            p["rate"] == parent["schedules"]["high"]["startup_rate"] == 152 and p["resolution"] == "high" and
            p["arm"] == "uniform", "High uniform rate, observations or accuracy criteria differ")
    jobs = [dict(resolution="high",name=name,seed=seed,arm=p["arm"]) for name,seeds in p["methods"].items() for seed in seeds]
    controls = {prepared.key(c):c for c in parent["controls"] if c["resolution"] == "high"}
    require(len(controls) == len(jobs) == 6 and set(controls) == {prepared.key(j) for j in jobs} and
            len(set(p["physical_times"])) == len(p["physical_times"]) == 19, "High uniform seed or observation coverage differs")
    anchors = [{k:controls[prepared.key(job)][k] for k in ("resolution","name","seed","early","late")} for job in jobs]
    sources = dict(parent["source_hashes"])
    deps = (Path(__file__),TEMPLATE,prepared.FROZEN,prepared.RUNS,prepared.SUMMARY,HERE / "galerkin_cuda_native.py",
            HERE / "galerkin_cuda_native.cu",HERE / "basis_recurrence_model.py")
    mid.merge_hashes(sources,dict(mid.fingerprint(path) for path in deps))
    prepared.verify_inputs(sources)
    base.save(FROZEN,dict(status="frozen",template=p,source_hashes=sources,jobs=jobs,controls=anchors,
        specs={name:parent["specs"][name] for name in p["methods"]},solve_protocol=parent["solve_protocol"],
        reference_protocol=record(mid.FROZEN)))


def stem(job):
    return "_".join(str(job[k]) for k in ("resolution","name","seed","arm"))


def config(job,frozen):
    return base.config(frozen["specs"][job["name"]],frozen["solve_protocol"],job["seed"])[1]


def validate(sol,p):
    probes = np.arange(0,p["horizon"]+p["probe_spacing"],p["probe_spacing"])
    steps = p["rate"]*p["horizon"]
    expected = dict(horizon=p["horizon"],dt=1/p["rate"],steps=steps,time_order=2,graph_steps=p["graph_steps"],
                    implementation="native-fp64-cufft-precontracted")
    require(all(sol.row[k] == v for k,v in expected.items()) and np.array_equal(sol.times,p["physical_times"]) and
            np.array_equal(sol.probe_times,probes) and sol.states.shape == (len(p["physical_times"]),*sol.model.rhs_shape) and
            all(getattr(sol,k).shape == (len(probes),3) for k in prepared.PROBES) and
            all(np.isfinite(getattr(sol,k)).all() for k in ("states",*prepared.PROBES)), "High uniform schedule or raw shape differs")


def execute(frozen):
    prepared.verify_inputs(frozen["source_hashes"])
    require(all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")),
            "Qualification requires one host BLAS thread")
    p = frozen["template"]
    for job in frozen["jobs"]:
        path, sidecar = (OUT / (stem(job)+ext) for ext in (".npz",".json"))
        if sidecar.exists():
            prior = base.read(sidecar)
            require(prior["job"] == job,"High uniform resume job differs")
            if prior["status"] == "complete":
                record(path,prior["sha256"])
            continue
        require(not path.exists(),"High uniform raw trajectory has no sidecar")
        print("START",job,flush=True)
        cfg = config(job,frozen)
        row = dict(job=job)
        try:
            with native_model():
                sol = solve_native(steps=p["rate"]*p["horizon"],horizon=p["horizon"],snapshots=p["physical_times"],
                                   probe_spacing=p["probe_spacing"],graph_steps=p["graph_steps"],**cfg)
            validate(sol,p)
            temporary = path.with_suffix(".tmp.npz")
            case.store_solution(temporary,sol,cfg)
            temporary.replace(path)
            row.update(status="complete",artifact=path.name,sha256=base.digest(path),metadata=sol.row)
        except (ArithmeticError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            row.update(status="failed",failure_type=type(exc).__name__,failure=str(exc))
        base.save(sidecar,row)
        print("DONE",job,row["status"],flush=True)


def score(frozen):
    require(not RUNS.exists() and not SUMMARY.exists(),"Preserve the existing high uniform qualification")
    prepared.verify_inputs(frozen["source_hashes"])
    p = frozen["template"]
    times,grids = p["physical_times"],base.grid_set()
    ref = frozen["reference_protocol"]
    ref_path = base.WORK / ref["path"]
    record(ref_path,ref["sha256"])
    truth,reference = prepared.absolute_reference(base.read(ref_path),grids)
    keys = {(g,t) for g in grids for t in times}
    rows,raw_records = [],{}
    for job,control in zip(frozen["jobs"],frozen["controls"],strict=True):
        sidecar = OUT / (stem(job)+".json")
        raw = base.read(sidecar)
        raw_records[sidecar.name] = base.digest(sidecar)
        require(raw["job"] == job and prepared.key(control) == prepared.key(job),"High uniform raw job differs")
        if raw["status"] != "complete":
            rows.append(dict(job=job,qualified=False,raw=raw))
            continue
        path = OUT / (stem(job)+".npz")
        require(raw["artifact"] == path.name,"High uniform raw artifact name differs")
        sol = prepared.load_saved(record(path,raw["sha256"]))
        first,last = (prepared.load_saved(control[k]) for k in ("early","late"))
        require(sol.row == raw["metadata"] and np.array_equal(first.times,times[:12]) and
                mid.switched.equal_spatial_models((first.model,last.model,sol.model)),"High uniform spatial coordinates or observations differ")
        with np.load(path,allow_pickle=False) as saved:
            require(json.loads(str(saved["config"])) == json.loads(json.dumps(config(job,frozen))),"High uniform config differs")
        last = mid.select(last,times[12:])
        cpu = case.Solution(last.model,np.concatenate((first.states,last.states)),np.asarray(times),
                            last.probe_times,last.transverse,last.vorticity,last.row)
        validate(sol,p)
        impl = prepared.compare(sol,cpu,case.reference_fields(cpu,grids),grids,keys,p["implementation_gate"])
        scores,worst = {},{}
        for name,fields in truth.items():
            scores[name],worst[name] = prepared.scores(sol,fields,grids,keys)
        phy = all(max(v.values()) <= p["physical_gate"] for v in worst.values())
        wake = base.wake(sol)
        qualified = impl["passed"] and phy and wake["sustained_alternating_probe_signs"]
        rows.append(dict(job=job,artifact=raw["artifact"],sha256=raw["sha256"],qualified=qualified,implementation=impl,
                         implementation_pass=impl["passed"],physical_scores=scores,physical_worst=worst,physical_pass=phy,wake=wake))
        print("SCORE",job,qualified,flush=True)
    sha = base.digest(FROZEN)
    base.save(RUNS,dict(status="complete",protocol_sha256=sha,raw_records=raw_records))
    base.save(SUMMARY,dict(status="complete",protocol_sha256=sha,runs_sha256=base.digest(RUNS),reference=reference,
                         passes=all(r["qualified"] for r in rows),rows=rows,isolated_speed_claim_supported=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ("freeze-only","gpu","score"):
        action.add_argument("--"+flag,action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        freeze()
        print("FROZEN",base.digest(FROZEN),flush=True)
    elif args.gpu:
        execute(base.read(FROZEN))
    else:
        score(base.read(FROZEN))
