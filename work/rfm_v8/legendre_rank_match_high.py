"""Qualify the exact retained-rank high Legendre candidate without changing the first freeze."""

import argparse
import json
from pathlib import Path

import legendre_rank_uniform as prior


base,case,np,old = prior.base,prior.case,prior.np,prior.old
require,fields_at,absolute_reference = prior.require,prior.fields_at,prior.absolute_reference
HERE = prior.HERE
OUT = HERE/"legendre_rank_match_high_output"
TEMPLATE = HERE/"protocol_legendre_rank_match_high.json"


def freeze():
    require(not OUT.exists(),"Preserve the registered exact-rank attempt")
    prior_protocol = prior.OUT/"protocol.json"
    p,t = base.read(prior_protocol),base.read(TEMPLATE)
    arm = t["candidate"]
    neighbor = next(j for j in p["jobs"] if j["name"] == "leg35_k80_q76")
    require(all(neighbor[k] == arm[k] for k in ("k","q","rate","seed")),"Exact-rank candidate changed the physical grid")
    references, sources = {},dict(p["source_hashes"])
    old_dir = prior.V7/"prepared_qualification_output"
    for scale,name in (("mid","rfm37_k72"),("high","rfm41_k80")):
        refs = []
        for seed in (7331,7687,8011,8353,9041):
            path = old_dir/f"{scale}_{name}_{seed}_combined.json"
            row = base.read(path)
            require(row["status"] == "complete" and row["job"]["seed"] == seed and row["job"]["name"] == name,
                    "Archived RFM rank identity differs")
            refs.append(dict(seed=seed,rank=row["metadata"]["rank"],mean_rank=row["metadata"]["mean_rank"],
                             path=path.relative_to(HERE.parent).as_posix(),sha256=base.digest(path)))
        require({x["rank"] for x in refs} == {39 if scale == "mid" else 40},"RFM retained ranks changed")
        references[scale] = refs
    high_leg = old_dir/"high_leg39_k80_17_combined.json"
    reference = base.read(high_leg)
    require(reference["metadata"]["rank"] == 43,"Archived Legendre39 rank differs")
    sources.update({r["path"]:r["sha256"] for refs in references.values() for r in refs})
    files = [Path(__file__),TEMPLATE,prior_protocol,high_leg]
    sources.update({x.relative_to(HERE.parent).as_posix():base.digest(x) for x in files})
    require(all(base.digest(HERE.parent/name) == sha for name,sha in sources.items()),"Exact-rank frozen input changed")
    job = dict(arm,config=dict(neighbor["config"],m=arm["m"]),control=neighbor["control"])
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="frozen",template=t,physical=p["template"],job=job,source_hashes=sources,
        archived_rfm_retained_ranks=references,archived_high_legendre39_rank=43))
    print("FROZEN",base.digest(OUT/"protocol.json"),flush=True)


def run():
    path = OUT/"protocol.json"
    frozen = base.read(path)
    p,job = frozen["physical"],frozen["job"]
    require(all(base.digest(HERE.parent/name)==sha for name,sha in frozen["source_hashes"].items()),"Exact-rank source changed")
    raw,summary = OUT/(job["name"]+".npz"),OUT/"summary.json"
    require(not raw.exists() and not summary.exists(),"Preserve the existing exact-rank result")
    times = p["physical_times"]
    row = dict(job=job,qualified=False,protocol_sha256=base.digest(path))
    try:
        sol = case.solve(prior.RecurrenceGalerkin,steps=p["horizon"]*job["rate"],horizon=p["horizon"],snapshots=times,
                         probe_spacing=p["probe_spacing"],**job["config"])
        require(np.array_equal(sol.times,times) and np.array_equal(sol.probe_times,np.arange(577)/8) and
                sol.states.shape == (19,*sol.model.rhs_shape) and all(np.isfinite(getattr(sol,k)).all()
                for k in ("states",*prior.PROBES)),"Exact-rank physical observations differ")
        with raw.open("xb") as handle:
            case.store_solution(handle,sol,job["config"])
        row.update(artifact=raw.name,artifact_sha256=base.digest(raw))
        grids = base.grid_set()
        truth,budget = absolute_reference(base.read(old.FROZEN),grids)
        anchor = prior.load_control(job,times,grids)
        direct,guard,baseline,raw_scores,difference = prior.scores(fields_at(sol,times,grids),anchor,truth,grids,times)
        wake = base.wake(sol)
        g = p["physical_gate"]
        physical,diagnostic = (all(max(v.values()) <= g for v in a.values()) for a in (direct,guard))
        row.update(status="complete",qualified=bool(physical and wake["sustained_alternating_probe_signs"]),
                   physical_pass=physical,cross_rank_guard_diagnostic_pass=diagnostic,wake=wake,
                   actual_rank=sol.model.width,mean_rank=sol.model.mean_width,rank_matches_target=sol.model.width==job["rank_target"],
                   real_state_dofs=2*job["k"]*sol.model.width+sol.model.mean_width,reference_budget=budget,
                   physical_worst=direct,cross_rank_guard_worst=guard,large_rank_anchor_worst=baseline,
                   difference_from_large_rank_anchor=difference,scores=raw_scores)
    except (ArithmeticError,RuntimeError,ValueError,OSError) as exc:
        row.update(status="failed",failure_type=type(exc).__name__,failure=str(exc))
    with summary.open("x",encoding="utf-8") as handle:
        json.dump(row,handle,indent=2,allow_nan=False)
    print(job["name"],row["status"],row["qualified"],row.get("actual_rank"),base.digest(summary),flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    args = cli.parse_args()
    freeze() if args.freeze else run()
