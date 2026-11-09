"""Interleave three qualified CPU methods and native CUDA Galerkin in isolation."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import cpu_formal as base
from cpu_implicit_stride import StridedGalerkin
from galerkin_cuda_native import solve_native

np = base.case.np
OUT = base.OUT
VALIDATION = OUT / "joint_cpu_validation.json"
TIMING = OUT / "joint_isolated.json"
SELECTED = ("rfm37_k72", "leg37_k72", "fd176_k80")
BACKENDS = (("cpu",SELECTED[0]), ("cpu",SELECTED[1]), ("cpu",SELECTED[2]),
            ("cuda_graph",SELECTED[0]), ("cuda_graph",SELECTED[1]))
GRAPH_STEPS = 24


def hash_set():
    paths=(Path(__file__), base.HERE/"cpu_implicit_stride.py", base.HERE/"cpu_profile.py",
           base.HERE/"galerkin_cuda_native.py", base.HERE/"galerkin_cuda_native.cu",
           base.HERE/"cuda_environment_native.py")
    return base.source_hashes() | {p.relative_to(base.WORK).as_posix():base.digest(p) for p in paths}


def frozen_qualification(protocol, reference):
    qual=base.read(OUT/"qualification.json")
    if qual["status"]!="complete" or qual["frozen"]["protocol"]!=protocol:
        raise RuntimeError("Complete the same frozen CPU qualification first")
    if qual["frozen"]["source_hashes"]!=base.source_hashes() or any(
        reference["reference_hashes"].get(name)!=sha for name,sha in
        qual["frozen"]["reference"]["reference_hashes"].items()):
        raise RuntimeError("Qualification inputs differ")
    if not reference["passes"]:
        raise RuntimeError("Independent reference axes must pass")
    eligible={r["name"] for r in qual["summary"] if r["numerical_pass"]}
    specs={s["name"]:s for s in protocol["variants"]}
    for method,name in zip(("rfm","legendre","fd4"),SELECTED):
        first=next(s["name"] for s in protocol["variants"] if s["method"]==method and s["name"] in eligible)
        if first!=name:
            raise RuntimeError("Expected the registered first passing configuration")
    return qual,specs


def frozen_solution(qual,name,seed):
    row=next(r for r in qual["rows"] if r["name"]==name and r["seed"]==seed)
    if not row["numerical_pass"]:
        raise RuntimeError(f"Expected a qualifying CPU control for {name}/{seed}")
    path=OUT/row["artifact"]
    if base.digest(path)!=row["sha256"]:
        raise RuntimeError(f"Qualification checkpoint hash changed for {path.name}")
    model=base.FullFD if row["method"]=="fd4" else base.BufferedGalerkin
    return base.case.load_solution(path,model)


def trajectory_difference(actual, control):
    result={}
    if not np.array_equal(actual.times,control.times) or not np.array_equal(actual.probe_times,control.probe_times):
        raise RuntimeError("Compared runs did not return the same times")
    for name in ("states","transverse","vorticity"):
        a,b=getattr(actual,name),getattr(control,name)
        result[name]=dict(relative=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                          max_absolute=float(abs(a-b).max()), bitwise=bool(np.array_equal(a,b)))
    return result


def solver(protocol,spec,seed,backend):
    cls,settings=base.config(spec,protocol,seed)
    steps=protocol["horizon"]*spec["rate"]
    if backend=="cuda_graph":
        return solve_native(steps=steps,graph_steps=GRAPH_STEPS,**settings),settings
    if cls is base.BufferedGalerkin:
        cls=StridedGalerkin
    return base.case.solve(cls,steps=steps,**settings),settings


def validate(protocol, qual, specs, hashes):
    if VALIDATION.exists():
        raise FileExistsError(VALIDATION)
    rows=[]
    for name in SELECTED[:2]:
        seed=protocol["seeds"][0] if specs[name]["method"]=="rfm" else protocol["deterministic_seed"]
        control=frozen_solution(qual,name,seed)
        actual,_=solver(protocol,specs[name],seed,"cpu")
        delta=trajectory_difference(actual,control)
        passed=bool(max(r["relative"] for r in delta.values())<=1e-10)
        rows.append(dict(name=name,seed=seed,pass_cpu=passed,trajectory_difference=delta))
        print(name,seed,"equivalent",passed,"bitwise",all(r["bitwise"] for r in delta.values()),flush=True)
    if hashes!=hash_set():
        raise RuntimeError("Implementation changed during validation")
    base.save(VALIDATION,dict(source_hashes=hashes,rows=rows,passes=all(r["pass_cpu"] for r in rows)))


def nvidia():
    try:
        args=["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]
        return subprocess.run(args,check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return None


def complete_row(protocol, spec, name, seed, backend, block, qual, truth):
    row=dict(name=name,method=spec["method"],seed=seed,backend=backend,block=block)
    path=OUT/f"joint_{block}_{backend}_{name}_{seed}.npz"
    record_path=path.with_suffix(".json")
    if record_path.exists():
        saved=base.read(record_path)
        if any(saved[key]!=value for key,value in row.items()):
            raise RuntimeError(f"Checkpoint job changed for {record_path.name}")
        if "sha256" in saved and base.digest(path)!=saved["sha256"]:
            raise RuntimeError(f"Checkpoint artifact changed for {path.name}")
        return saved
    if path.exists():
        path.rename(path.with_suffix(f".orphan_{time.time_ns()}.npz"))
    control=frozen_solution(qual,name,seed)
    started=time.perf_counter()
    try:
        with np.errstate(over="ignore",invalid="ignore"):
            sol,settings=solver(protocol,spec,seed,backend)
        elapsed=time.perf_counter()-started
        scores=base.case.score(sol,truth,base.grid_set())
        errors=base.case.maxima(scores)
        diagnostics=base.wake(sol)
        comparison=trajectory_difference(sol,control)
        temp=path.with_suffix(".tmp.npz")
        base.case.store_solution(temp,sol,settings)
        temp.replace(path)
        row.update(seconds=elapsed,metadata=sol.row,errors=errors,scores=scores,wake=diagnostics,
            boundary_errors=base.boundary_errors(sol),artifact=path.name,sha256=base.digest(path),
            paired_qualification_difference=comparison,numerical_pass=bool(
                max(errors.values())<=protocol["gate"] and diagnostics["sustained_alternating_probe_signs"]))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError,RuntimeError,OSError) as exc:
        row.update(failure=str(exc),failure_seconds=time.perf_counter()-started,numerical_pass=False)
    base.save(record_path,row)
    return row


def summary(rows, count):
    out=[]
    for backend,name in BACKENDS:
        selected=[r for r in rows if r["name"]==name and r["backend"]==backend]
        measured=[r for r in selected if "seconds" in r]
        durations=[r["seconds"] for r in measured]
        keys=("velocity","vorticity","near_velocity","near_vorticity")
        worst={key:max(r["errors"][key] for r in measured) for key in keys} if measured else None
        out.append(dict(name=name,backend=backend,count=len(selected),
            passes=len(selected)==count and all(r["numerical_pass"] for r in selected),
            failures=sum("failure" in r for r in selected),worst=worst,samples_seconds=durations,
            median_seconds=statistics.median(durations) if durations else None,
            min_seconds=min(durations) if durations else None,
            max_seconds=max(durations) if durations else None))
    return out


def timing(protocol, reference, truth, qual, specs, hashes):
    validated=base.read(VALIDATION)
    if not validated["passes"] or validated["source_hashes"]!=hashes:
        raise RuntimeError("Full-trajectory CPU validation must match current sources")
    rng=np.random.default_rng(protocol["order_seed"])
    schedule=[]
    for block,seed in enumerate(protocol["seeds"],1):
        for i in rng.permutation(len(BACKENDS)):
            backend,name=BACKENDS[i]
            run_seed=seed if specs[name]["method"]=="rfm" else protocol["deterministic_seed"]
            schedule.append(dict(name=name,seed=run_seed,backend=backend,block=block))
    fixed=dict(protocol=protocol,reference=reference,source_hashes=hashes,schedule=schedule,
               cpu=base.pin(),platform=platform.platform(),cpu_processor=platform.processor(),
               gpu=nvidia(),graph_steps=GRAPH_STEPS,environment={
                   key:os.environ.get(key) for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
               policy="Five randomized blocks. Fresh model and CUDA runtime per solve. Timing covers "
                      "setup, full time stepping, synchronization and returned outputs. "
                      "Reference and score evaluation are outside the timed interval.")
    if TIMING.exists():
        result=base.read(TIMING)
        if result["frozen"]!=fixed:
            raise RuntimeError("Isolated benchmark inputs changed")
        for row in result["rows"]:
            if "artifact" in row and base.digest(OUT/row["artifact"])!=row["sha256"]:
                raise RuntimeError(f"Isolated artifact hash changed for {row['artifact']}")
    else:
        result=dict(status="running",frozen=fixed,rows=[])
        base.save(TIMING,result)
    known={(r["block"],r["backend"],r["name"]) for r in result["rows"]}
    np.linalg.svd(np.eye(10))
    for job in schedule:
        if (job["block"],job["backend"],job["name"]) in known:
            continue
        row=complete_row(protocol,specs[job["name"]],qual=qual,truth=truth,**job)
        result["rows"].append(row)
        base.save(TIMING,result)
        print(row["block"],row["backend"],row["name"],row.get("seconds"),
              row.get("errors",row.get("failure")),flush=True)
    if hash_set()!=hashes or any(base.digest(base.WORK/name)!=sha for name,sha in
                                reference["reference_hashes"].items()):
        raise RuntimeError("Benchmark implementation or reference changed")
    result.update(status="complete",summary=summary(result["rows"],len(protocol["seeds"])))
    base.save(TIMING,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("validate","timing"))
    parser.add_argument("--confirm-isolated",action="store_true")
    args=parser.parse_args()
    if args.phase=="timing" and not args.confirm_isolated:
        parser.error("Stop competing benchmark workloads, then confirm isolation")
    protocol=base.read(base.PROTOCOL)
    hashes=hash_set()
    reference,truth=base.reference_audit(protocol)
    qual,specs=frozen_qualification(protocol,reference)
    if args.phase=="validate":
        validate(protocol,qual,specs,hashes)
    else:
        timing(protocol,reference,truth,qual,specs,hashes)


if __name__=="__main__":
    main()
