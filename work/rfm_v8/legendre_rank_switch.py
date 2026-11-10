"""Score all eligible lower-rank Legendre switched trajectories in a separate freeze."""

import argparse
import json
from pathlib import Path

import legendre_rank_uniform as uniform


base, case, np, old = uniform.base,uniform.case,uniform.np,uniform.old
require, fields_at, absolute_reference = uniform.require,uniform.fields_at,uniform.absolute_reference
HERE = uniform.HERE
OUT = HERE/"legendre_rank_switch_output"
TEMPLATE = HERE/"protocol_legendre_rank_switch.json"
EXACT = HERE/"legendre_rank_match_high_output"
MODEL = old.switched.RecurrenceGalerkin


def freeze():
    require(not OUT.exists(),"Preserve the registered switched Legendre attempt")
    up,us = (uniform.OUT/name for name in ("protocol.json","summary.json"))
    p,summary,s = (base.read(path) for path in (up,us,TEMPLATE))
    require(summary["protocol_sha256"] == base.digest(up) and [r["job"] for r in summary["rows"]] == p["jobs"],
            "Uniform candidate inventory differs")
    xp,xs = (EXACT/name for name in ("protocol.json","summary.json"))
    x_protocol,x_row = base.read(xp),base.read(xs)
    require(x_row["protocol_sha256"] == base.digest(xp) and x_row["job"] == x_protocol["job"] and
            x_protocol["physical"] == p["template"],"Supplemental exact-rank evidence differs")
    inputs = [Path(__file__),TEMPLATE,up,us,xp,xs,HERE/"summarize_legendre_rank_uniform.py"]
    jobs = []
    inventory = [(row,uniform.OUT) for row in summary["rows"]]+[(x_row,EXACT)]
    for row,folder in inventory:
        if folder == uniform.OUT:
            sidecar = folder/(row["job"]["name"]+".json")
            require(base.digest(sidecar) == summary["raw_records"][sidecar.name],"Original uniform sidecar changed")
            inputs.append(sidecar)
        if "artifact" in row:
            path = folder/row["artifact"]
            require(base.digest(path) == row["artifact_sha256"],"Uniform source artifact changed")
            inputs.append(path)
        if row.get("physical_pass") and row.get("wake",{}).get("sustained_alternating_probe_signs"):
            require("artifact" in row,"A physically eligible uniform trajectory has no raw state")
            j = row["job"]
            jobs.append(dict(name=j["name"],seed=j["seed"],config=j["config"],startup_rate=j["rate"],
                             late_rate=s["rate_map"][str(j["rate"])],uniform_rank=row["actual_rank"],
                             anchor=dict(path=path.relative_to(HERE.parent).as_posix(),sha256=row["artifact_sha256"])))
    require(jobs,"No preregistered uniform trajectory passed the shared gates")
    hashes = dict(p["source_hashes"]) | x_protocol["source_hashes"]
    hashes.update({x.relative_to(HERE.parent).as_posix():base.digest(x) for x in inputs})
    require(all(base.digest(HERE.parent/name) == sha for name,sha in hashes.items()),"Frozen switched input changed")
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="frozen",template=s,physical_protocol=p["template"],source_hashes=hashes,jobs=jobs))
    print("FROZEN",base.digest(OUT/"protocol.json"),len(jobs),flush=True)


def run(index):
    protocol_path = OUT/"protocol.json"
    protocol = base.read(protocol_path)
    p,s,job = protocol["physical_protocol"],protocol["template"],protocol["jobs"][index]
    require(all(base.digest(HERE.parent/name) == sha for name,sha in protocol["source_hashes"].items()),"Frozen switched source changed")
    dest = OUT/job["name"]
    raw,sidecar = dest.with_suffix(".npz"),dest.with_suffix(".json")
    require(not raw.exists() and not sidecar.exists(),"A registered switched result already exists")
    physical = p["physical_times"]
    t0 = s["switch_time"]
    times = sorted(physical+[t0+i/job["late_rate"] for i in (1,2)])
    row = dict(job=job,protocol_sha256=base.digest(protocol_path),qualified=False)
    try:
        solution = old.switched.solve_cpu_switch(MODEL,startup_rate=job["startup_rate"],late_rate=job["late_rate"],
            switch_time=t0,horizon=p["horizon"],snapshots=times,probe_spacing=p["probe_spacing"],**job["config"])
        control = case.load_solution(HERE.parent/job["anchor"]["path"],uniform.RecurrenceGalerkin)
        require(np.array_equal(solution.times,times) and np.array_equal(solution.probe_times,control.probe_times) and
                old.switched.equal_spatial_models((solution.model,control.model)) and solution.model.width == job["uniform_rank"],
                "Switched trajectory changed spatial coordinates or absolute times")
        with raw.open("xb") as handle:
            case.store_solution(handle,solution,job["config"])
        row.update(artifact=raw.name,artifact_sha256=base.digest(raw))
        prefix = solution.probe_times <= t0
        early = (solution.states[solution.times<=t0].tobytes()==control.states[control.times<=t0].tobytes() and
                 all(getattr(solution,k)[prefix].tobytes()==getattr(control,k)[prefix].tobytes() for k in uniform.PROBES))
        grids = base.grid_set()
        truth,budget = absolute_reference(base.read(old.FROZEN),grids)
        direct,guard,anchor,raw_scores,delta = uniform.scores(fields_at(solution,physical,grids),fields_at(control,physical,grids),truth,grids,physical)
        wake = base.wake(solution)
        g = p["physical_gate"]
        direct_ok,guard_ok,anchor_ok = (all(max(v.values()) <= g for v in x.values()) for x in (direct,guard,anchor))
        row.update(status="complete",qualified=bool(early and direct_ok and guard_ok and anchor_ok and wake["sustained_alternating_probe_signs"]),
                   prefix_bitwise=early,physical_pass=direct_ok,guard_pass=guard_ok,anchor_pass=anchor_ok,wake=wake,reference_budget=budget,
                   physical_worst=direct,guard_worst=guard,anchor_worst=anchor,difference_from_same_rank_uniform=delta,scores=raw_scores,
                   steps=solution.row["steps"],nonlinear_evaluations=solution.row["nonlinear_evaluations"],actual_rank=solution.model.width)
    except (ArithmeticError,RuntimeError,ValueError,OSError) as exc:
        row.update(status="failed",failure_type=type(exc).__name__,failure=str(exc))
    with sidecar.open("x",encoding="utf-8") as handle:
        json.dump(row,handle,indent=2,allow_nan=False)
    print(job["name"],row["status"],row["qualified"],flush=True)


def summarize():
    path = OUT/"protocol.json"
    protocol = base.read(path)
    sha = base.digest(path)
    rows = [base.read(OUT/(job["name"]+".json")) for job in protocol["jobs"]]
    require([r["job"] for r in rows] == protocol["jobs"] and all(r["protocol_sha256"] == sha for r in rows),"Switched evidence differs")
    for r in rows:
        if "artifact" in r:
            require(base.digest(OUT/r["artifact"]) == r["artifact_sha256"],"Saved switched trajectory changed")
    with (OUT/"summary.json").open("x",encoding="utf-8") as handle:
        json.dump(dict(status="complete",protocol_sha256=sha,rows=rows,qualified=sum(r["qualified"] for r in rows),
                       scope=protocol["template"]["scope"],timing_claim_supported=False),handle,indent=2,allow_nan=False)
    print("SUMMARY",base.digest(OUT/"summary.json"),flush=True)


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
