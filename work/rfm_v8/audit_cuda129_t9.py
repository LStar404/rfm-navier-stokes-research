"""T9 compiler/FFT ablation with archived controls and dual physical references."""

import argparse
from functools import partial
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from cuda129_runtime import HERE, V7, VENDOR, PinnedBundledRuntime, VARIANTS


OUT = HERE/"cuda129_t9_output"
PREFLIGHT = HERE/"cuda129_preflight_builtins.json"
CONTROL = HERE/"fresh_worker_mid_t9_output"
SWITCH = V7/"switch_qualification_output"/"protocol.json"
ARRAYS = ("states","times","probe_times","transverse","vorticity")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path,content):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(content,stream,indent=2,allow_nan=False)
        stream.write("\n")


def selected_arrays(solved,reference):
    import numpy as np
    result = {}
    for key in ARRAYS:
        a,b = solved[key],reference[key]
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.isfinite(a).all():
            raise ArithmeticError("T9 array layout or finite check differs")
        result[key] = dict(byte_exact=a.tobytes()==b.tobytes(),maximum_absolute=float(np.max(abs(a-b))),
                           relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny)),
                           candidate_sha256=sha256(a.tobytes()).hexdigest(),control_sha256=sha256(b.tobytes()).hexdigest())
    return result


def frozen_plan():
    before = read(PREFLIGHT)
    if (before["runtime_source_sha256"] != digest(HERE/"cuda129_runtime.py") or
            before["vendor_manifest_sha256"] != digest(VENDOR/"manifest.json")):
        raise ValueError("T9 runtime or vendor changed after FP64 graph preflight")
    original,raw = read(CONTROL/"protocol.json"),read(CONTROL/"runs.json")
    reference = read(SWITCH)
    cases = []
    for request,row in zip(original["requests"],raw,strict=True):
        if request["arm"]!="combined" or request["block"]!=0 or request["imports"]!="minimal":
            continue
        target = CONTROL/request["trajectory"]
        if row["trajectory_sha256"]!=digest(target):
            raise ValueError("T9 archive trajectory differs")
        cases.append(dict(name=request["name"],config=request["config"],solve_args=request["solve_args"],
                          control=target.name,control_sha256=row["trajectory_sha256"],
                          spatial_sha256=row["marker"]["spatial_sha256"]))
    if len(cases)!=2:
        raise ValueError("Expected both registered Galerkin families")
    source = {p.resolve().relative_to(HERE.parent).as_posix():digest(p)
              for p in (Path(__file__),HERE/"cuda129_runtime.py",HERE/"galerkin_cuda_prepared_cuda129.py",
                                      HERE/"cuda129_solver_source_audit.json",PREFLIGHT,CONTROL/"protocol.json",CONTROL/"runs.json",
                                      SWITCH,VENDOR/"manifest.json",V7/"galerkin_cuda_native.cu",V7/"galerkin_cuda_variable.cu",
                                      HERE/"galerkin_cuda_prepared_minimal.py",HERE/"galerkin_phase_shared_minimal.py",
                                      HERE/"galerkin_minimal_model.py",HERE/"galerkin_cuda_variable_minimal.py",
                                      HERE/"sbdf2_schedule_minimal.py")}
    return dict(status="frozen",cases=cases,variants=[r["variant"] for r in before["rows"] if r["passed"]],
                preflight_rejections=[r["variant"] for r in before["rows"] if not r["passed"]],
                physical_gate=reference["template"]["physical_gate"],implementation_gate=reference["template"]["implementation_gate"],
                source_sha256=source,scope="T9 correctness only. Inherited CUDA_CACHE_DISABLE=1; no new timing claim.")


def child(protocol,index,variant):
    from galerkin_cuda_prepared_cuda129 import solve_cuda_prepared
    from galerkin_minimal_model import case
    from solver_worker import model_signature

    plan = read(protocol)
    job = plan["cases"][index]
    if variant not in plan["variants"] or variant not in VARIANTS:
        raise ValueError("Unregistered T9 variant")
    runtime = {}

    def factory(toolkit):
        rt = PinnedBundledRuntime(toolkit,variant=variant)
        runtime.update(rt.evidence)
        return rt

    solution = solve_cuda_prepared(runtime_factory=factory,**job["solve_args"])
    spatial = model_signature(solution.model,job["config"]["family"])
    stem = job["name"]+"_"+variant
    artifact = protocol.parent/(stem+".npz")
    with artifact.open("xb") as stream:
        case.store_solution(stream,solution,job["config"])
    save(protocol.parent/(stem+".json"),dict(status="complete",artifact=artifact.name,sha256=digest(artifact),
         runtime=runtime,spatial_sha256=spatial,metadata=solution.row))


def calculate_scores(protocol, rows):
    import numpy as np
    from galerkin_minimal_model import case, RecurrenceGalerkin
    from cpu_formal import grid_set
    from verify_switch_qualification import absolute_reference

    plan = read(protocol)
    original = read(SWITCH)
    grids = grid_set()
    truth,_ = absolute_reference(original,grids)
    for job in plan["cases"]:
        target = CONTROL/job["control"]
        if digest(target)!=job["control_sha256"]:
            raise ValueError("T9 comparator changed during execution")
        base = case.load_solution(target,RecurrenceGalerkin)
        indexes = [original["physical_times"].index(t) for t in job["solve_args"]["snapshots"]]
        references = {name:{grid:[values[i] for i in indexes] for grid,values in all_grids.items()}
                      for name,all_grids in truth.items()}
        baseline_fields = case.reference_fields(base,grids)
        for row in rows:
            if row["name"]!=job["name"] or row["status"]!="complete":
                continue
            try:
                artifact = protocol.parent/row["artifact"]
                if digest(artifact)!=row["sha256"] or row["spatial_sha256"]!=job["spatial_sha256"]:
                    raise ValueError("T9 result archive or spatial signature differs")
                with np.load(artifact,allow_pickle=False) as a,np.load(target,allow_pickle=False) as b:
                    if not all(json.loads(str(s["config"]))==job["config"] for s in (a,b)):
                        raise ValueError("T9 configuration differs")
                    arrays = selected_arrays(a,b)
                if not all(arrays[k]["byte_exact"] for k in ("times","probe_times")):
                    raise ValueError("T9 absolute times differ")
                solved = case.load_solution(artifact,RecurrenceGalerkin)
                implementation = case.maxima(case.score(solved,baseline_fields,grids))
                physical = {name:case.maxima(case.score(solved,fields,grids)) for name,fields in references.items()}
                probe_error = {k:arrays[k]["relative_l2"] for k in ("transverse","vorticity")}
                ipass = max(*implementation.values(),*probe_error.values())<=plan["implementation_gate"]
                ppass = max(v for group in physical.values() for v in group.values())<=plan["physical_gate"]
                row.update(qualified=bool(ipass and ppass),arrays=arrays,implementation_fields=implementation,
                           implementation_probes=probe_error,physical=physical)
            except Exception as exc:
                row.update(qualified=False,failure_type=type(exc).__name__,failure=str(exc))


def main():
    if os.environ.get("CUDA_CACHE_DISABLE")!="1" or OUT.exists():
        raise ValueError("Need fresh output and process-local CUDA_CACHE_DISABLE=1")
    p = frozen_plan()
    OUT.mkdir()
    protocol = OUT/"protocol.json"
    save(protocol,p)
    rows = []
    for index,job in enumerate(p["cases"]):
        for variant in p["variants"]:
            proc = subprocess.run([sys.executable,"-B",str(Path(__file__)), "--child",str(index),"--variant",variant,
                                   "--protocol",str(protocol)],capture_output=True,text=True,timeout=90,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            stem = job["name"]+"_"+variant
            if proc.returncode:
                error = proc.stderr.replace(str(HERE.parent.parent),"<workspace>").replace(str(Path.home()),"<user>")
                row = dict(status="failed",returncode=proc.returncode,failure=error)
                save(OUT/(stem+".json"),row)
            else:
                row = read(OUT/(stem+".json"))
            rows.append(dict(name=job["name"],variant=variant,**row))
            print(stem,row["status"],flush=True)
    calculate_scores(protocol,rows)
    if any(digest(HERE.parent/name)!=sha for name,sha in p["source_sha256"].items()):
        raise ValueError("Frozen T9 source changed")
    result = dict(status="passed" if rows and all(row.get("qualified",False) for row in rows) else "failed",
                  protocol_sha256=digest(protocol),preflight_rejections=p["preflight_rejections"],rows=rows,
                  scope="T9 nine physical observations on three grids. Independent fields and paired probe norms, no wake qualification.")
    save(OUT/"qualification.json",result)
    print(result["status"],len(rows),flush=True)
    return result["status"]=="passed"


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",type=int)
    parser.add_argument("--variant")
    parser.add_argument("--protocol",type=Path)
    args = parser.parse_args()
    if args.child is None:
        sys.exit(0 if main() else 1)
    child(args.protocol,args.child,args.variant)
