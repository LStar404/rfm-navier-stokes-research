"""Run the frozen 0.1% independent FD reference ladder in four CPU processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import time

import cpu_formal as base

ROOT = base.HERE / "cpu_formal_point1_output"
PROTOCOL = base.HERE / "protocol_point1_ref.json"
JOURNAL = ROOT / "runs.json"


def fingerprint():
    extras = (PROTOCOL, Path(__file__))
    return base.source_hashes() | {p.relative_to(base.WORK).as_posix():base.digest(p) for p in extras}


def run(spec, protocol):
    path = base.WORK/spec["source"]
    record_path = path.with_suffix(".json")
    row = dict(spec=spec)
    if record_path.exists():
        saved = base.read(record_path)
        if saved["spec"]!=spec or ("sha256" in saved and base.digest(path)!=saved["sha256"]):
            raise RuntimeError(f"Reference checkpoint changed: {path.name}")
        return saved
    if path.exists():
        path.rename(path.with_suffix(f".orphan_{time.time_ns()}.npz"))
    started = time.perf_counter()
    print("START",spec["name"],"PID",os.getpid(),flush=True)
    try:
        cls,settings = base.config(spec,protocol)
        with base.case.np.errstate(over="ignore",invalid="ignore"):
            sol = base.case.solve(cls,steps=protocol["horizon"]*spec["rate"],**settings)
        temp=path.with_suffix(".tmp.npz")
        base.case.store_solution(temp,sol,settings)
        temp.replace(path)
        row.update(status="complete",metadata=sol.row,wake=base.wake(sol),
                   sha256=base.digest(path),concurrent_seconds=time.perf_counter()-started)
    except (ArithmeticError,ValueError,base.case.np.linalg.LinAlgError) as exc:
        row.update(status="failed",failure=str(exc),concurrent_seconds=time.perf_counter()-started)
    base.save(record_path,row)
    print("END",spec["name"],row["status"],row["concurrent_seconds"],flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4:
        parser.error("Choose one to four independent CPU workers")
    protocol=base.read(PROTOCOL)
    hashes=fingerprint()
    fixed=dict(protocol=protocol,source_hashes=hashes,legacy_sha256=base.digest(
        base.WORK/protocol["legacy_reference"]),threads={
            key:os.environ.get(key) for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    if JOURNAL.exists():
        result=base.read(JOURNAL)
        if result["frozen"]!=fixed:
            raise RuntimeError("Frozen 0.1% reference inputs changed")
        for row in result["rows"]:
            if "sha256" in row and base.digest(base.WORK/row["spec"]["source"])!=row["sha256"]:
                raise RuntimeError(f"Reference artifact changed: {row['spec']['source']}")
    else:
        result=dict(status="running",frozen=fixed,rows=[])
        base.save(JOURNAL,result)
    known={r["spec"]["name"] for r in result["rows"]}
    pending=[s for s in protocol["reference_path"] if s["name"] not in known]
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,s,protocol) for s in pending]
            for future in as_completed(futures):
                row=future.result()
                result["rows"].append(row)
                base.save(JOURNAL,result)
    if fingerprint()!=hashes:
        raise RuntimeError("Source changed during the ladder")
    if any(r["status"]!="complete" for r in result["rows"]):
        result.update(status="failed",passes=False)
    else:
        audit,_=base.reference_audit(protocol)
        base.save(ROOT/"reference_audit.json",audit)
        result.update(status="complete",passes=audit["passes"],
                      composed_difference=audit["composed_difference"])
        print("SCREEN",audit["passes"],audit["composed_difference"],flush=True)
    base.save(JOURNAL,result)


if __name__=="__main__":
    main()
