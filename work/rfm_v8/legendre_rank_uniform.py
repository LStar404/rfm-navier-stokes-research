"""Preserve four full deterministic Legendre trajectories and their dual-reference gates."""

import argparse
import json
from pathlib import Path

from galerkin_minimal_model import RecurrenceGalerkin, case, np
import cpu_formal as base
import switch_qualification as old
from verify_switch_qualification import absolute_reference, fields_at, maxima, measured, require


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
TEMPLATE = HERE/"protocol_legendre_rank_uniform.json"
OUT = HERE/"legendre_rank_uniform_output"
PROBES = ("transverse","vorticity")
METRICS = ("velocity","vorticity","near_velocity","near_vorticity")


def freeze():
    require(not OUT.exists(),"Preserve the registered Legendre attempt")
    p = base.read(TEMPLATE)
    q = V7/"prepared_qualification_output"/"protocol.json"
    archived,ref_protocol = base.read(q),base.read(old.FROZEN)
    require(p["physical_times"] == ref_protocol["physical_times"] and all(p[k] == ref_protocol["template"][k]
            for k in ("horizon","probe_spacing","references","physical_gate","reference_budget_fraction")),
            "Legendre reference observations or gates changed")
    source = {name:sha for name,sha in archived["source_hashes"].items() if name.endswith((".py",".c",".dll"))}
    references = {r["path"]:r["sha256"] for r in ref_protocol["early_references"].values()}
    references.update(ref_protocol["late_reference"]["reference_hashes"])
    source.update(references)
    late = old.original.chain
    prior = base.read(late.JOURNAL)
    files = [Path(__file__),TEMPLATE,HERE/"galerkin_minimal_model.py",q,old.FROZEN,old.early.FROZEN,old.early.RUNS,old.early.REFERENCE,
             late.JOURNAL,late.OUT/prior["reference_audit"]]
    jobs = []
    for arm in p["arms"]:
        spec = archived["specs"][arm["anchor"]]
        require(spec["method"] == "legendre" and all(arm[k] == spec[k] for k in ("k","q","rate")),"Registered grid or rate differs from anchor")
        cfg = base.config(dict(spec,m=arm["m"]),archived["solve_protocol"],p["seed"])[1]
        selected = next(c for c in archived["controls"] if c["name"] == arm["anchor"] and c["seed"] == p["seed"])
        anchor = {k:selected[k] for k in ("early","late")}
        source.update({r["path"]:r["sha256"] for r in anchor.values()})
        jobs.append(dict(arm,config=cfg,seed=p["seed"],control=anchor))
    source.update({x.relative_to(HERE.parent).as_posix():base.digest(x) for x in files})
    require(all(base.digest(HERE.parent/name) == sha for name,sha in source.items()),"Frozen Legendre source or archive changed")
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="frozen",template=p,source_hashes=source,jobs=jobs))
    print("FROZEN",base.digest(OUT/"protocol.json"),len(jobs),flush=True)


def load_control(job, times, grids):
    values = []
    for name,subset in (("early",times[:12]),("late",times[12:])):
        spec = job["control"][name]
        path = HERE.parent/spec["path"]
        require(base.digest(path) == spec["sha256"],"The same-rate high-rank control changed")
        raw = case.load_solution(path,RecurrenceGalerkin)
        require(raw.row["dt"] == 1/job["rate"],"High-rank control time step differs")
        values.append(fields_at(raw,subset,grids))
    return {g:values[0][g]+values[1][g] for g in grids}


def scores(candidate,anchor,truth,grids,times):
    delta = measured(candidate,anchor,grids,times)
    direct,triangle,anchor_max,rows = {},{},{},{}
    for ref,fields in truth.items():
        errors = measured(candidate,fields,grids,times)
        baseline = measured(anchor,fields,grids,times)
        guard = {gt:{k:baseline[gt][k]+(1+baseline[gt][k])*delta[gt][k] for k in METRICS} for gt in errors}
        require(all(errors[gt][k] <= guard[gt][k]+2e-13 for gt in errors for k in METRICS),"Pointwise triangle does not bound the physical norm")
        direct[ref],triangle[ref],anchor_max[ref] = maxima(errors),maxima(guard),maxima(baseline)
        rows[ref] = dict(direct=[dict(grid=g,time=t,**v) for (g,t),v in errors.items()],
                         guard=[dict(grid=g,time=t,**v) for (g,t),v in guard.items()])
    return direct,triangle,anchor_max,rows,maxima(delta)


def run(index):
    path = OUT/"protocol.json"
    frozen = base.read(path)
    p,job = frozen["template"],frozen["jobs"][index]
    require(all(base.digest(HERE.parent/name) == sha for name,sha in frozen["source_hashes"].items()),"Frozen Legendre input changed")
    base_name = OUT/job["name"]
    raw,sidecar = base_name.with_suffix(".npz"),base_name.with_suffix(".json")
    require(not raw.exists() and not sidecar.exists(),"A registered Legendre case already has results")
    times = p["physical_times"]
    row = dict(job=job,protocol_sha256=base.digest(path),qualified=False)
    try:
        solution = case.solve(RecurrenceGalerkin,steps=p["horizon"]*job["rate"],horizon=p["horizon"],
                              snapshots=times,probe_spacing=p["probe_spacing"],**job["config"])
        require(np.array_equal(solution.times,times) and np.array_equal(solution.probe_times,np.arange(577)/8) and
                all(getattr(solution,k).shape == (577,3) for k in PROBES) and
                solution.states.shape == (19,*solution.model.rhs_shape) and
                all(np.isfinite(getattr(solution,k)).all() for k in ("states",*PROBES)),"Full physical observations differ")
        with raw.open("xb") as handle:
            case.store_solution(handle,solution,job["config"])
        row.update(artifact=raw.name,artifact_sha256=base.digest(raw))
        grids = base.grid_set()
        reference,budget = absolute_reference(base.read(old.FROZEN),grids)
        candidate = fields_at(solution,times,grids)
        anchor = load_control(job,times,grids)
        direct,guard,control,rows,difference = scores(candidate,anchor,reference,grids,times)
        wake = base.wake(solution)
        gate = p["physical_gate"]
        direct_ok = all(max(v.values()) <= gate for v in direct.values())
        guard_ok = all(max(v.values()) <= gate for v in guard.values())
        anchor_ok = all(max(v.values()) <= gate for v in control.values())
        width,mean = solution.model.width,solution.model.mean_width
        row.update(status="complete",qualified=bool(direct_ok and guard_ok and anchor_ok and wake["sustained_alternating_probe_signs"]),
                   physical_pass=direct_ok,guard_pass=guard_ok,anchor_pass=anchor_ok,wake=wake,reference_budget=budget,
                   actual_rank=width,mean_rank=mean,rank_matches_target=(width==job["rank_target"]),
                   real_state_dofs=2*job["k"]*width+mean,physical_worst=direct,guard_worst=guard,anchor_worst=control,
                   difference_from_large_rank_anchor=difference,scores=rows,
                   scope="Numerical results only; no ranking by runtime")
    except (ArithmeticError,RuntimeError,ValueError,OSError) as exc:
        row.update(status="failed",failure_type=type(exc).__name__,failure=str(exc))
    with sidecar.open("x",encoding="utf-8") as handle:
        json.dump(row,handle,indent=2,allow_nan=False)
    print(job["name"],row["status"],row["qualified"],row.get("actual_rank"),flush=True)


def summarize():
    protocol = OUT/"protocol.json"
    p = base.read(protocol)
    rows = [base.read(OUT/(j["name"]+".json")) for j in p["jobs"]]
    require([r["job"] for r in rows] == p["jobs"] and all(r["protocol_sha256"] == base.digest(protocol) for r in rows),
            "Registered Legendre results omitted or changed")
    for row in rows:
        if "artifact" in row:
            require(base.digest(OUT/row["artifact"]) == row["artifact_sha256"],"Saved Legendre artifact changed")
    result = dict(status="complete",protocol_sha256=base.digest(protocol),scope=p["template"]["scope"],rows=rows,
                  switch_candidates=[r["job"]["name"] for r in rows if r["qualified"]],
                  timing_claim_supported=False)
    with (OUT/"summary.json").open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print("SUMMARY",base.digest(OUT/"summary.json"),result["switch_candidates"],flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze",action="store_true")
    actions.add_argument("--job",type=int,choices=range(4))
    actions.add_argument("--summarize",action="store_true")
    args = cli.parse_args()
    if args.freeze:
        freeze()
    elif args.summarize:
        summarize()
    else:
        run(args.job)
