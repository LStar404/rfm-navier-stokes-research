"""Resume the two missing, same-equilibration reference axes and rescore saved cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from unittest.mock import patch

import fd_pow2_long as radial_driver
import cpu_formal_point1_rescore as saved

base,case,np = radial_driver.base,radial_driver.case,radial_driver.np
MODEL = radial_driver.Pow2EquilibratedBufferedFD
HERE = Path(__file__).resolve().parent
OUT = HERE/"cpu_formal_point1_pow2_chain_output"
PROTOCOL = HERE/"protocol_fd_pow2_reference.json"
JOURNAL = OUT/"runs.json"


def grid(spec):
    return dict(radial_driver.GRID,k=spec["k"])


def run(spec, protocol):
    path = base.WORK/spec["source"]
    details = path.with_suffix(".json")
    if details.exists():
        row = base.read(details)
        if row["spec"]!=spec or base.digest(path)!=row["sha256"]:
            raise RuntimeError("Completed reference arm or specification changed")
        return row
    if path.exists():
        raise FileExistsError(path)
    resume = OUT/spec["name"]
    resume.mkdir(exist_ok=True)
    # Each spawned worker owns this process-local configuration. Reuse the
    # already verified SBDF2 and history-checkpoint order exactly.
    with patch.multiple(radial_driver, OUT=resume, RATE=spec["rate"],
                        HORIZON=protocol["horizon"],
                        STEPS=protocol["horizon"]*spec["rate"], GRID=grid(spec)):
        print("ARM",spec["name"],"k",spec["k"],"rate",spec["rate"],"PID",os.getpid(),flush=True)
        solution,config = radial_driver.march(spec["n"])
    temp = path.with_suffix(".tmp.npz")
    case.store_solution(temp,solution,config)
    temp.replace(path)
    row = dict(spec=spec,configuration=config,metadata=solution.row,wake=base.wake(solution),
               sha256=base.digest(path),status="complete",resume_directory=resume.name)
    base.save(details,row)
    print("COMPLETE",spec["name"],row["metadata"]["seconds"],row["wake"]["strouhal"],flush=True)
    return row


def references(protocol, rows):
    arms = {row["spec"]["name"]:row for row in rows}
    result = []
    for spec in protocol["reference_path"]:
        path = base.WORK/spec["source"]
        expected = spec["sha256"] if "sha256" in spec else arms[spec["name"]]["sha256"]
        if base.digest(path)!=expected:
            raise RuntimeError(f"Reference trajectory hash changed: {path.name}")
        sol = case.load_solution(path,MODEL)
        info = sol.row
        if (info["steps"]!=spec["rate"]*protocol["horizon"] or
                not np.array_equal(sol.times,protocol["checkpoints"]) or
                any(info[key]!=value for key,value in dict(grid(spec),n=spec["n"]).items()) or
                info["reconstruction_class"]!=radial_driver.CLASS or
                info["equilibration"]!="row_power_of_two" or
                info["row_scale"]!=sol.model.scale_metadata() or
                not base.wake(sol)["sustained_alternating_probe_signs"]):
            raise RuntimeError(f"Reference reconstruction check failed: {path.name}")
        result.append(sol)
    return result


def axis_audit(protocol, solutions):
    edges = []
    multiplier = 1.
    for name,left,right in zip(("radial","angle","time"),solutions,solutions[1:]):
        row = radial_driver.compare(left,right)
        multiplier *= 1+max(row["worst"].values())
        edges.append(dict(axis=name,**row))
    composed = multiplier-1
    threshold = protocol["gate"]*protocol["reference_budget_fraction"]
    return dict(edges=edges,composed_difference=composed,threshold=threshold,
                passes=bool(composed<=threshold),independent_axes=True,
                wake=base.wake(solutions[-1]),direct=radial_driver.compare(solutions[0],solutions[-1]))


def selected_candidates():
    qualification = base.read(base.OUT/"qualification.json")
    if qualification["status"]!="complete" or len(qualification["rows"])!=21:
        raise RuntimeError("Expected all 21 frozen CPU qualification candidates")
    result = []
    for row in qualification["rows"]:
        cls = base.FullFD if row["method"]=="fd4" else base.BufferedGalerkin
        result.append((row["name"],row["seed"],base.OUT/row["artifact"],row["sha256"],cls))
    result.extend((name,seed,path,digest,cls)
                  for name,seed,path,digest,cls,_ in saved.candidates()
                  if name=="rfm49_bandwidth98" or name.startswith(("legendre49","fd")))
    if len(result)!=33:
        raise RuntimeError("Expected 21 original, 5 wide RFM, 3 Legendre49 and 4 saved FD controls")
    return result


def rescore(protocol, audit, ref, candidates):
    grids = base.grid_set()
    truth = case.reference_fields(ref,grids)
    rows = []
    for name,seed,path,digest,cls in candidates:
        if base.digest(path)!=digest:
            raise RuntimeError(f"Saved candidate changed: {path.name}")
        sol = case.load_solution(path,cls)
        if not np.array_equal(sol.times,ref.times):
            raise RuntimeError("Scoring times differ")
        scores = case.score(sol,truth,grids)
        worst = case.maxima(scores)
        wake = base.wake(sol)
        passes = bool(max(worst.values())<=protocol["gate"] and
                      wake["sustained_alternating_probe_signs"])
        rows.append(dict(name=name,seed=seed,source=path.relative_to(base.WORK).as_posix(),
                         sha256=digest,metadata=sol.row,scores=scores,worst=worst,
                         wake=wake,absolute_velocity_crossing_lags=saved.crossing_drift(wake,audit["wake"]),
                         numerical_pass=passes,reference_screen=audit["passes"]))
        print("RESCORE",name,seed,passes,worst,flush=True)
    groups = []
    for name in dict.fromkeys(row["name"] for row in rows):
        part = [row for row in rows if row["name"]==name]
        expected = 5 if name.startswith("rfm") else 1
        qualifies = len(part)==expected and all(row["numerical_pass"] for row in part)
        groups.append(dict(name=name,count=len(part),expected_count=expected,
            numerical_pass=qualifies,reference_screen=audit["passes"],
            qualified=bool(qualifies and audit["passes"]),
            worst={key:max(row["worst"][key] for row in part) for key in part[0]["worst"]}))
    return dict(rows=rows,summary=groups,note="Saved candidate fields only. No 0.1% timing claim.")


def fingerprint():
    names=("fd_pow2_reference_chain.py","protocol_fd_pow2_reference.json",
           "cpu_formal_point1_rescore.py","cpu_formal_layer_tangent.py",
           "cpu_formal_rfm_bandwidth.py","cpu_formal_error_diagnostics.py")
    return radial_driver.source_manifest() | {f"rfm_v7/{n}":base.digest(HERE/n) for n in names}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers",type=int,default=2,choices=(2,))
    args=cli.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol=base.read(PROTOCOL)
    for spec in protocol["reference_path"][:2]:
        if base.digest(base.WORK/spec["source"])!=spec["sha256"]:
            raise RuntimeError("Original scaled radial edge no longer matches the frozen hashes")
    candidates=selected_candidates()
    fixed=dict(protocol=protocol,source_hashes=fingerprint(),
        candidate_inputs=[dict(name=name,seed=seed,path=path.relative_to(base.WORK).as_posix(),
                                sha256=digest) for name,seed,path,digest,_ in candidates])
    if JOURNAL.exists():
        journal=base.read(JOURNAL)
        if journal["frozen"]!=fixed:
            raise RuntimeError("Frozen chain or saved candidate list changed")
    else:
        journal=dict(status="running",parent_pid=os.getpid(),rows=[],frozen=fixed)
        base.save(JOURNAL,journal)
    if journal["status"]=="complete":
        return
    complete={row["spec"]["name"] for row in journal["rows"]}
    todo=[spec for spec in protocol["reference_path"][2:] if spec["name"] not in complete]
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run,spec,protocol) for spec in todo]):
                journal["rows"].append(future.result())
                base.save(JOURNAL,journal)
    if fingerprint()!=fixed["source_hashes"]:
        raise RuntimeError("Source changed during the independent chain")
    all_refs=references(protocol,journal["rows"])
    audit=axis_audit(protocol,all_refs)
    base.save(OUT/"reference_audit.json",audit)
    print("REFERENCE",audit["passes"],audit["composed_difference"],
          [row["worst"] for row in audit["edges"]],flush=True)
    summary=rescore(protocol,audit,all_refs[-1],candidates)
    base.save(OUT/"saved_candidate_rescore.json",summary)
    journal.update(status="complete",reference_screen=audit["passes"],
                   composed_difference=audit["composed_difference"],
                   reference_audit="reference_audit.json",candidate_rescore="saved_candidate_rescore.json")
    base.save(JOURNAL,journal)
    print("RESULT",summary["summary"],flush=True)


if __name__=="__main__":
    main()
