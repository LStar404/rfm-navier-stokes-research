"""Recheck published cavity scores and sampled fields using only release files."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0,str(HERE.parent/"rfm_v3"))
from cavity_study import Score, np
from rfm_v2_benchmark import weighted_errors


NORMS=("velocity_rel_l2", "vorticity_rel_l2")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(name, expected):
    path=(TREE/name).resolve()
    require(path.relative_to(TREE.resolve()).as_posix()==name, "Unexpected cavity evidence path")
    require(digest(path)==expected,f"Cavity hash mismatch: {name}")
    return path


def close_scores(observed,expected,rtol):
    require(observed.keys()==expected.keys(),"Cavity score schema differs")
    require(all(math.isclose(value,expected[k],rel_tol=rtol,abs_tol=1e-12)
                for k,value in observed.items()),"Cavity scores differ")


def verify():
    audit=read(HERE/"supplement_cavity_audit.json")
    derived=read(HERE/"cavity_release_metadata.json")
    require(audit["status"]=="passed" and derived["status"]=="sanitized_derived_metadata",
            "Expected passed audit and sanitized archive metadata")
    require(derived["source_sha256"]==audit["inputs"]["work/rfm_v4/cavity_formal.json"],
            "The source archive provenance differs")
    expected_sources={r["path"]:r["sha256"] for r in audit["sources"] if r["collection"]=="v4"}
    require(derived["workspace_relative_dependency_hashes"]==expected_sources,
            "Derived dependency map differs from the historical audit")
    for row in audit["sources"]:
        if row["path"]!="work/rfm_v4/cavity_formal.json":
            artifact(row["path"],row["sha256"])
    v5_name="work/rfm_v5/cavity_paired.json"
    v5_freeze_name="work/rfm_v5/cavity_paired_freeze.json"
    v5=read(artifact(v5_name,audit["inputs"][v5_name]))
    freeze5=read(artifact(v5_freeze_name,audit["inputs"][v5_freeze_name]))
    require(v5["freeze_sha256"]==audit["inputs"][v5_freeze_name],"V5 freeze hash differs")
    rows={(r["collection"],r["name"],r["seed"]):r for r in audit["checked_rows"]}
    expected4={("v4",g["name"],seed) for g in derived["summary"] for seed in derived["seeds"]}
    expected5={("v5",f"{method}_m{m}",seed) for m in freeze5["gates"] for method in ("v4","v5")
               for seed in freeze5["seeds"]}
    require(len(rows)==len(audit["checked_rows"])==70 and len(expected4)==50 and len(expected5)==20
            and set(rows)==expected4|expected5,"Cavity cohort inventory differs")
    for r in rows.values():
        require(math.isfinite(r["seconds"]) and r["seconds"]>0 and set(r["worst"])==set(NORMS)
                and all(math.isfinite(v) and v>=0 for v in r["worst"].values()),"Invalid cavity timing or errors")
        gates=derived["thresholds"] if r["collection"]=="v4" else [next(iter(r["gate_checks"]))]
        require(r["gate_checks"]=={str(g):max(r["worst"].values())<=float(g) for g in gates},
                "Cavity gate calculation differs")
    for group in derived["summary"]:
        cohort=[rows["v4",group["name"],seed] for seed in derived["seeds"]]
        require(group["runs"]==group["completed"]==len(cohort) and
                group["seconds"]==median(r["seconds"] for r in cohort),"V4 cohort timing aggregate differs")
        require(group["worst"]=={name:max(r["worst"][key] for r in cohort)
                for name,key in zip(("velocity","vorticity"),NORMS)},"V4 cohort worst error differs")
    require(v5["status"]=="complete" and len(v5["rows"])==20,"V5 raw records incomplete")
    for r in v5["rows"]:
        observed=rows["v5",f"{r['method']}_m{r['m']}",r["seed"]]
        require(r["seconds"]==observed["seconds"] and r["worst"]==observed["worst"] and
                observed["gate_checks"]=={str(freeze5["gates"][str(r["m"])]):r["passed"]},
                "V5 raw and audited records differ")

    reference=Score()
    for grid in reference.grids:
        close_scores(reference.grid_change[grid],audit["reference_grid_change"][grid],1e-10)
    replay=audit["sampled_field_replay"]
    require(len(replay)==len(derived["summary"])==10 and
            {(r["name"],r["seed"]) for r in replay}=={
                (r["name"],derived["seeds"][0]) for r in derived["summary"]},"V4 saved-field inventory differs")
    for r in replay:
        scores={}
        with np.load(artifact(r["artifact"],r["sha256"]),allow_pickle=False) as data:
            for grid,(xy,w) in reference.grids.items():
                require(np.array_equal(data[f"{grid}_xy"],xy) and np.array_equal(data[f"{grid}_weights"],w),
                        "Saved field grid or weights differ")
                fields={k:data[f"{grid}_{k}"] for k in ("u","v","omega")}
                scores[grid]=weighted_errors(fields,reference.truth[grid],w)
                close_scores(scores[grid],r["scores"][grid],1e-8)
        worst={k:max(g[k] for g in scores.values()) for k in NORMS}
        close_scores(worst,rows["v4",r["name"],r["seed"]]["worst"],1e-8)
    return dict(status="passed",checked_recorded_rows=len(rows),rescored_saved_fields=len(replay),
                scope="The original v4 freeze is private. Its archived SHA-256 is provenance only. "
                      "Only the ten published first-seed field arrays were reconstructed.")


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
