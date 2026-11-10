"""Independently rescore only completed T72 slots from the interrupted cohort."""

from pathlib import Path

import cuda129_t72_diagnostic as diagnostic
import thin_t72_qualification as thin
import prepared_formal_timing as archived


HERE = Path(__file__).resolve().parent
OUT = HERE/"cuda129_t72_diagnostic_output"
INCIDENT = HERE/"cuda129_t72_interruption.json"
RESULT = HERE/"cuda129_t72_partial_rescore.json"
read,digest,require = thin.read,thin.digest,thin.require


def context(protocol, jobs):
    selected = {(job["name"],job["seed"]) for job in jobs}
    subset = dict(protocol,gal_controls=[r for r in protocol["gal_controls"] if (r["name"],r["seed"]) in selected],
                  fd_controls={k:r for k,r in protocol["fd_controls"].items() if (k,r["seed"]) in selected})
    return archived.context(subset)


def score(job,entry,plan,original,scoring):
    index = entry["index"]
    request = job["request"]
    raw = OUT/f"run_{index:03d}.json"
    req = OUT/f"request_{index:03d}.json"
    target = OUT/request["trajectory"]
    row = read(raw)
    require(row["status"]=="complete" and row["index"]==request["index"]==index and
            row["request_sha256"]==digest(req)==entry["request_sha256"] and read(req)==request and
            digest(raw)==entry["run_sha256"] and row["trajectory_sha256"]==digest(target)==entry["trajectory_sha256"],
            "Completed raw slot or request differs")
    thin.verify_transport(row,request)
    old = read(thin.check_record(job["archive_sidecar"]))
    require(row["marker"]["spatial_sha256"]==job["expected_spatial_sha256"]==old["marker"]["spatial_sha256"] and
            {key:old[key] for key in thin.JOB_FIELDS}==job["archive_job"], "Old spatial signature or job differs")
    thin.check_record(job["archive_artifact"])
    out = dict(job=job["archive_job"],marker=row["marker"])
    out.update(archived.score_solution(target,out,job["archive_job"],original,scoring))
    archived.verify_numerical_row(out,original)
    require(thin.non_time_metadata(out["metadata"])==thin.non_time_metadata(old["metadata"]),
            "Raw complete trajectory metadata differs")
    exact = all(entry["archive_byte_exact"].values())
    if exact:
        for field in ("physical_scores","physical_worst","guard_scores","guard_worst","implementation","wake"):
            require(out[field]==old[field],"Recomputed scores for identical trajectories differ from original")
    return dict(index=index,name=job["name"],scale=job["scale"],seed=job["seed"],variant=job["variant"],block=job["block"],
                input=entry,score=out,qualified=out["qualified"],raw_phase_clocks=diagnostic.phases(OUT,job,row,plan))


def paired(rows):
    identities = {(r["name"],r["block"],r["variant"]):r for r in rows if r["qualified"]}
    pairs,missing = [],[]
    for name,block in sorted({(r["name"],r["block"]) for r in rows}):
        for contrast,(a,b) in diagnostic.CONTRASTS.items():
            baseline = identities.get((name,block,a))
            candidate = identities.get((name,block,b))
            header = dict(name=name,block=block,contrast=contrast,baseline=a,candidate=b)
            if baseline is None or candidate is None:
                missing.append(header)
                continue
            left,right = (r["raw_phase_clocks"] for r in (baseline,candidate))
            deltas = {k:right[k]-left[k] for k in sorted(left.keys() & right.keys())}
            pairs.append(dict(header,paired_phase_delta_seconds=deltas,slots=[baseline["index"],candidate["index"]],
                              scope="One qualified within-configuration pair from an interrupted non-isolated cohort."))
    return pairs,missing


def main():
    require(not RESULT.exists(),"Preserve existing partial rescoring")
    p,incident = read(OUT/"protocol.json"),read(INCIDENT)
    require(digest(OUT/"protocol.json")==incident["protocol_sha256"] and
            incident["status"]=="incomplete_failed_no_performance_cohort","Interrupted cohort identity differs")
    diagnostic.verify_environment(p)
    archive = read(thin.check_record(p["thin_archive"]["protocol.json"]))
    plans = {scale:read(thin.check_record(info["archive"]["protocol"]))
             for scale,info in archive["archive_contexts"].items()}
    complete = incident["completed_slots"]
    jobs = [p["jobs"][r["index"]] for r in complete]
    results = []
    for scale,orig in plans.items():
        view = context(orig,[j for j in jobs if j["scale"]==scale])
        for job,entry in zip(jobs,complete,strict=True):
            if job["scale"]!=scale:
                continue
            result = score(job,entry,p,orig,view)
            results.append(result)
            print(result["index"],result["name"],result["variant"],result["qualified"],flush=True)
        del view
    require(len(results)==len(complete)==16,"Only the 16 actually completed slots are eligible for rescoring")
    results.sort(key=lambda r:r["index"])
    diagnostic.verify_environment(p)
    pairs,missing = paired(results)
    thin.save_new(RESULT,dict(status="complete_partial_rescore",source_sha256=digest(Path(__file__)),
        incident_sha256=digest(INCIDENT),protocol_sha256=digest(OUT/"protocol.json"),rows=results,
        qualified=sum(r["qualified"] for r in results),pairwise_phase_diagnostics=pairs,missing_or_unqualified_pairs=missing,
        failed_slots=incident["failed_slots"],interrupted_slots=incident["interrupted_without_completed_row"],
        unattempted_slots=incident["unattempted_slots"],overall_speed_ratios=None,isolated_speed_claim_supported=False,
        scope="Recomputed dual-reference physical and implementation scores, probes and wake from saved states. "
              "One block, no completed legacy baseline, no inference about relative CUDA 11.8 or all-case speed."))
    print("PARTIAL",len(results),sum(r["qualified"] for r in results),digest(RESULT),flush=True)


if __name__=="__main__":
    main()
