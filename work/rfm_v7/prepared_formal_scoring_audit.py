"""Replay registered scoring on already saved GPU trajectories without a solve."""

from pathlib import Path
import json
from tempfile import TemporaryDirectory

import prepared_formal_timing as timing


base, case, np, require = timing.base, timing.case, timing.np, timing.require
HERE = base.HERE


def archive(job,p):
    arm = job["arm"]
    if arm == "fd":
        return p["fd_controls"][job["name"]]["gpu"]
    c = next(row for row in p["gal_controls"] if (row["name"],row["seed"]) == (job["name"],job["seed"]))
    if arm == "uniform":
        return c["uniform_gpu"]
    if arm == "switch_old":
        return c["cuda_graph"]
    match = p["template"]["prepared_arms"][arm]
    return next(row["artifact"] for row in p["prepared_evidence"] if
                (row["job"]["name"],row["job"]["seed"],row["job"]["preparation"]) == (job["name"],job["seed"],match))


def run():
    settings = base.read(timing.eligibility.TEMPLATE)
    background = timing.identity_from_pid(settings["background_server_pid"])
    report = dict(status="running",scope="Read-only scoring replay of one archived input for each scale/family/arm. "
                  "Switched inputs are subset in a temporary copy. No new solve or CUDA context.",rows=[])
    with TemporaryDirectory(prefix="prepared_scoring_audit_") as directory:
        for scale in ("mid","high"):
            p = timing.eligibility.make_protocol(scale,background)
            scoring = timing.context(p)
            for job in p["schedule"]:
                if job["block"] != 1:
                    continue
                item = archive(job,p)
                path = base.WORK/item["path"]
                measured = case.load_solution(path,base.FullFD if job["arm"] == "fd" else timing.MODEL)
                n_saved = len(measured.times)
                with np.load(path,allow_pickle=False) as raw:
                    cfg = json.loads(str(raw["config"]))
                target = path
                if n_saved != 19:
                    require(n_saved == 21 and job["arm"].startswith("switch_"), "Archived observation inventory differs")
                    selected = timing.switch.select(measured,p["design"]["physical_times"])
                    require(selected.row == measured.row, "Selecting common observations altered original metadata")
                    target = Path(directory)/f"{scale}_{path.name}"
                    case.store_solution(target,selected,cfg)
                row = {"job":job,"marker":{"spatial_sha256":timing.previous.model_signature(measured.model)}}
                try:
                    row.update(timing.score_solution(target,row,job,p,scoring))
                    timing.verify_numerical_row(row,p)
                except Exception as exc:
                    row.update(qualified=False,failure_type=type(exc).__name__,failure=timing.public_failure(exc))
                report["rows"].append(dict(scale=scale,job=job,archived_input=item,archived_observations=n_saved,
                    replayed_observations=19,qualified=row["qualified"],metadata_preserved=row.get("metadata") == measured.row,
                    implementation_worst={name:r["worst"] for name,r in row.get("implementation",{}).items()},
                    physical_worst=row.get("physical_worst"),guard_worst=row.get("guard_worst"),work=row.get("work"),
                    **({k:row[k] for k in ("failure_type","failure")} if "failure" in row else {})))
                print(scale,job["name"],job["arm"],row["qualified"],row.get("failure",""),flush=True)
    require(len(report["rows"]) == 19,"Scoring test did not exercise every scale/family/arm")
    report["status"] = "passed" if all(r["qualified"] and r["metadata_preserved"] for r in report["rows"]) else "failed"
    report["source_sha256"] = {name:base.digest(HERE/name) for name in (
        "prepared_formal_timing.py","prepared_formal_eligibility.py","prepared_formal_timing_template.json",
        "prepared_formal_scoring_audit.py")}
    base.save(HERE/"prepared_formal_scoring_audit.json",report)
    return report["status"] == "passed"


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
