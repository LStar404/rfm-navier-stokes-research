"""Compare all saved scoring fields and estimate prospective monitored duration."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import prepared_formal_scoring_audit as previous
import prepared_formal_timing as timing


base, case, np, require = timing.base, timing.case, timing.np, timing.require
HERE = base.HERE
SCORE_INPUT = HERE/"prepared_formal_scoring_audit.json"
TRANSPORT_INPUT = timing.previous.OUT/"timing.json"
PHASES = ("seconds","host_setup_seconds","host_linear_precompute_seconds","runtime_init_seconds","upload_seconds",
          "plan_seconds","graph_setup_seconds","march_seconds","download_seconds")


def score(target, job, protocol, scoring, spatial_sha, *, cached):
    row = dict(job=job,marker=dict(spatial_sha256=spatial_sha))
    reconstruct = case.Solution.fields
    evaluations = 0

    def counted(solution,index,s,theta):
        nonlocal evaluations
        evaluations += 1
        return reconstruct(solution,index,s,theta)

    view = timing.CachedScoringTrajectory if cached else lambda sol,_:sol
    with patch.object(timing,"CachedScoringTrajectory",view), patch.object(case.Solution,"fields",counted):
        started = perf_counter()
        row.update(timing.score_solution(target,row,job,protocol,scoring))
        scoring_seconds = perf_counter()-started
    timing.verify_numerical_row(row,protocol)
    payload = json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return row,payload,scoring_seconds,evaluations


def store_observations(source,target,job,p):
    measured = case.load_solution(source,base.FullFD if job["arm"] == "fd" else timing.MODEL)
    with np.load(source,allow_pickle=False) as raw:
        cfg = json.loads(str(raw["config"]))
    archived = len(measured.times)
    selected = timing.switch.select(measured,p["design"]["physical_times"]) if archived != 19 else measured
    require(archived in (19,21) and len(selected.times) == 19 and selected.row == measured.row,
            "Scoring audit changed archived observation metadata")
    started = perf_counter()
    temp = target.with_suffix(".tmp.npz")
    case.store_solution(temp,selected,cfg)
    temp.replace(target)
    base.digest(target)
    return measured, archived, perf_counter()-started


def journal_time(output,p,replays):
    journal = dict(status="running",rows=[])
    started = perf_counter()
    for i,job in enumerate(p["schedule"]):
        source = replays[job["name"],job["arm"]]
        row = dict(source,job=job,index=i,status="qualified")
        base.save(output/("synthetic-"+timing.previous.stem(job)+".json"),row)
        journal["rows"].append(row)
        base.save(output/"synthetic-timing.json",journal)
    journal["status"] = "complete"
    base.save(output/"synthetic-timing.json",journal)
    return perf_counter()-started


def source_phases(job,p):
    item = previous.archive(job,p)
    source = (base.WORK/item["path"]).with_suffix(".json")
    raw = base.read(source)
    require(raw["sha256"] == item["sha256"], "Diagnostic timing row and trajectory identities differ")
    metadata = raw["metadata"]
    phases = {k:metadata[k] for k in PHASES if k in metadata}
    require("seconds" in phases and "march_seconds" in phases and
            all(v >= 0 for v in phases.values()), "Archived solver timing inventory differs")
    return dict(phases=phases,archive=item,raw_sha256=base.digest(source))


def prospective(p,rows,phases,context_seconds,hash_seconds,journal_seconds,overheads):
    groups = []
    for name,arm in p["groups"]:
        observed = [r for r in rows if (r["job"]["name"],r["job"]["arm"]) == (name,arm)]
        solver = [r["phases"]["seconds"] for job,r in phases if (job["name"],job["arm"]) == (name,arm)]
        require(len(observed) == 1 and len(solver) == 5, "Prospective duration lacks a registered group")
        row = observed[0]
        groups.append(dict(name=name,arm=arm,slots=5,solver_sample_seconds=solver,
                           cached_scoring_sample_seconds=row["cached_scoring_seconds"],
                           synthetic_save_and_hash_sample_seconds=row["serialization_and_hash_seconds"]))
    count = len(p["schedule"])
    solver_base = sum(sum(r["solver_sample_seconds"]) for r in groups)
    solver_ceiling = sum(r["slots"]*max(r["solver_sample_seconds"]) for r in groups)
    scoring_base = sum(r["slots"]*r["cached_scoring_sample_seconds"] for r in groups)
    serialize_base = sum(r["slots"]*r["synthetic_save_and_hash_sample_seconds"] for r in groups)
    middle = timing.previous.statistics(overheads)["median"]
    assumptions = dict(maximum_prior_transport_seconds=max(overheads),median_prior_transport_seconds=middle,
                       coordinator_scoring_multiplier=1.25,serialization_multiplier=1.25,
                       journal_multiplier=1.25,context_and_hash_multiplier=1.25,
                       unobserved_cleanup_and_release_per_slot_seconds=.30,fixed_window_reserve_seconds=30.)
    base_seconds = (solver_base + count*middle + scoring_base + serialize_base +
                    context_seconds + sum(hash_seconds) + journal_seconds)
    conservative = (solver_ceiling + count*(max(overheads)+assumptions["unobserved_cleanup_and_release_per_slot_seconds"]) +
                    1.25*(scoring_base + serialize_base + context_seconds + sum(hash_seconds) + journal_seconds) +
                    assumptions["fixed_window_reserve_seconds"])
    window = p["isolation"]["window_seconds"]
    return dict(rows=count,window_seconds=window,window_margin_conservative_seconds=window-conservative,
                estimated_seconds=base_seconds,conservative_planning_seconds=conservative,
                current_window_supported_by_estimate=bool(conservative <= window),assumptions=assumptions,groups=groups,
                components_seconds=dict(archived_solver_sum=solver_base,solver_group_maxima_sum=solver_ceiling,
                    cached_coordinator_scoring_projection=scoring_base,serialization_and_hash_projection=serialize_base,
                    scoring_context=context_seconds,frozen_hash_passes=hash_seconds,journal_full_cohort=journal_seconds))


def run():
    cpu = base.pin()
    prior = base.read(TRANSPORT_INPUT)
    overheads = [row["fresh_process_return_seconds"]-row["solver_seconds"]
                 for row in prior["rows"] if row["qualified"] and row["arm"].startswith("cuda_")]
    require(len(overheads) == 25 and min(overheads) > 0, "Prior diagnostic CUDA transport inventory differs")
    original = base.read(SCORE_INPUT)
    require(original["status"] == "passed" and len(original["rows"]) == 19,
            "Original uncached archive audit is missing")
    old_rows = {(r["scale"],r["job"]["name"],r["job"]["arm"]):r for r in original["rows"]}
    settings = base.read(timing.eligibility.TEMPLATE)
    background = timing.identity_from_pid(settings["background_server_pid"])
    fingerprints = {name:base.digest(HERE/name) for name in (
        "prepared_formal_timing.py","prepared_formal_eligibility.py","prepared_formal_timing_template.json",
        "prepared_formal_scoring_cache_audit.py")}
    report = dict(status="running",logical_cpu=cpu,source_sha256=fingerprints,uncached_original_sha256=base.digest(SCORE_INPUT),
                  scope="Read-only archive replay with bitwise JSON comparison of every raw score, gate, probe, wake and work item. "
                  "Files and archived metadata are unchanged. No new solve or CUDA context.",rows=[])
    budget = dict(status="prospective_only",source_sha256=fingerprints,prior_diagnostic_timing_sha256=base.digest(TRANSPORT_INPUT),
                  scope="Estimates from archived non-isolated solver phases and transport overhead, one scoring/serialization "
                  "sample per registered scale/family/arm, full-cohort synthetic journal writes, and measured coordinator setup. "
                  "The arithmetic budget is neither an isolation observation nor a guarantee.",
                  thresholds_unchanged=True,scales={})
    with TemporaryDirectory(prefix="prepared_scoring_cache_audit_") as tmp:
        directory = Path(tmp)
        for scale in ("mid","high"):
            p = timing.eligibility.make_protocol(scale,background)
            started = perf_counter()
            scoring = timing.context(p)
            context_seconds = perf_counter()-started
            replays, rows = {}, []
            for job in (r for r in p["schedule"] if r["block"] == 1):
                item = previous.archive(job,p)
                path = base.WORK/item["path"]
                target = directory/f"{scale}_{path.name}"
                measured,archived,serialize = store_observations(path,target,job,p)
                spatial_sha = timing.previous.model_signature(measured.model)
                order = (False,True) if len(rows)%2 else (True,False)
                both = {cached:score(target,job,p,scoring,spatial_sha,cached=cached) for cached in order}
                old,old_bytes,old_seconds,old_count = both[False]
                new,new_bytes,new_seconds,new_count = both[True]
                reference = old_rows[scale,job["name"],job["arm"]]
                valid = (old_bytes == new_bytes and new["qualified"] and new["metadata"] == measured.row and
                         archived == reference["archived_observations"] and item == reference["archived_input"] and
                         {k:r["worst"] for k,r in new["implementation"].items()} == reference["implementation_worst"] and
                         new["physical_worst"] == reference["physical_worst"] and new["guard_worst"] == reference["guard_worst"] and
                         new["work"] == reference["work"] and old_count > new_count == 57)
                rows.append(dict(job=job,scale=scale,archived_input=item,archived_observations=archived,replayed_observations=19,
                    passed=bool(valid),uncached_scoring_seconds=old_seconds,cached_scoring_seconds=new_seconds,
                    uncached_reconstructions=old_count,cached_reconstructions=new_count,
                    serialization_and_hash_seconds=serialize,uncached_score_sha256=sha256(old_bytes).hexdigest(),
                    cached_score_sha256=sha256(new_bytes).hexdigest()))
                replays[job["name"],job["arm"]] = new
                print(scale,job["name"],job["arm"],valid,round(old_seconds,3),round(new_seconds,3),flush=True)
            hashes = []
            for _ in range(2):
                started = perf_counter()
                timing.previous.check_hashes(p["source_hashes"])
                hashes.append(perf_counter()-started)
            journal = journal_time(directory,p,replays)
            phases = [(job,source_phases(job,p)) for job in p["schedule"]]
            budget["scales"][scale] = prospective(p,rows,phases,context_seconds,hashes,journal,overheads)
            budget["scales"][scale]["diagnostic_phase_sources"] = [dict(job=job,**record) for job,record in phases]
            report["rows"].extend(rows)
    require(len(report["rows"]) == 19, "Cached scoring audit did not cover all registered branches")
    report["status"] = "passed" if all(r["passed"] for r in report["rows"]) else "failed"
    base.save(HERE/"prepared_formal_scoring_cache_audit.json",report)
    base.save(HERE/"prepared_formal_window_budget.json",budget)
    return report["status"] == "passed"


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
