"""Complete matched CPU trajectories or preserve the first nonfinite stage."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
import traceback

import joint_hardware as joint
import fd_pow2_reference_chain as chain
from cpu_formal_step_screen import Screen

base,case,np = joint.base,joint.base.case,joint.np
HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/"protocol_cpu_step_failure_control.json"
OUT=HERE/"cpu_step_failure_control_output"
JOURNAL=OUT/"runs.json"


class FirstNonfinite(FloatingPointError):
    def __init__(self, row):
        self.row=row
        super().__init__(f"Nonfinite {row['stage']} at step={row['step']} t={row['time']:.12g}")


class Trace:
    def __init__(self, dt):
        self.step=0
        self.dt=dt
        self.recent=deque(maxlen=24)

    def check(self, stage, value):
        finite=np.isfinite(value)
        row=dict(step=self.step,time=self.step*self.dt,stage=stage)
        if not finite.all():
            bad=np.argwhere(~finite)
            good=value[finite]
            row.update(nonfinite_count=int(len(bad)),first_indices=bad[:8].tolist(),
                finite_max_component=self.max_component(good))
            raise FirstNonfinite(row)
        row["max_component"]=self.max_component(value)
        self.recent.append(row)

    @staticmethod
    def max_component(value):
        return float(max(np.max(np.abs(value.real),initial=0.),
                         np.max(np.abs(value.imag),initial=0.)))

    def wrap(self, model):
        advance,terms,probe=model.advance,model.terms,model.probes

        def checked_advance(rhs, amplitude, first):
            self.step+=1
            self.check("rhs_before_boundary_lift",rhs)
            state=advance(rhs,amplitude,first)
            self.check("state_after_implicit",state)
            return state

        def checked_terms(state, amplitude):
            mass,flux=terms(state,amplitude)
            self.check("mass",mass)
            self.check("nonlinear_flux",flux)
            return mass,flux

        def checked_probe(state, amplitude):
            vy,omega=probe(state,amplitude)
            self.check("transverse_probe",vy)
            self.check("vorticity_probe",omega)
            return vy,omega

        model.advance,model.terms,model.probes=checked_advance,checked_terms,checked_probe
        return model


def job_path(job):
    return OUT/f"{Screen.tag(job,'cpu')}.json"


def reference_truth(screen_freeze, grids):
    p=screen_freeze["protocol"]
    record=base.read(HERE/p["reference_manifest"])
    rp=record["frozen"]["protocol"]
    solutions=chain.references(rp,record["rows"])
    return {spec["name"]:case.reference_fields(sol,grids)
            for spec,sol in zip(rp["reference_path"],solutions)
            if spec["name"] in p["reference_names"]}


def run(job, protocol, screen_freeze):
    path=job_path(job)
    if path.exists():
        saved=base.read(path)
        if saved["job"]!=job or ("artifact" in saved and
                base.digest(OUT/saved["artifact"])!=saved["sha256"]):
            raise RuntimeError("Saved CPU diagnostic changed")
        return saved
    artifact=path.with_suffix(".npz")
    if artifact.exists():
        raise FileExistsError(artifact)
    spec=dict(screen_freeze["specs"][job["name"]],rate=job["rate"])
    _,settings=base.config(spec,base.read(base.PROTOCOL),job["seed"])
    cls=joint.BufferedFD if spec["method"]=="fd4" else joint.RecurrenceGalerkin
    trace=Trace(1/job["rate"])
    started=time.perf_counter()
    item=dict(job=job,config=settings,model_class=cls.__name__,pid=os.getpid())
    print("START",job,"PID",os.getpid(),flush=True)
    try:
        def factory(**kwargs):
            return trace.wrap(cls(**kwargs))
        with np.errstate(over="ignore",invalid="ignore"):
            sol=case.solve(factory,steps=protocol["horizon"]*job["rate"],horizon=protocol["horizon"],
                snapshots=protocol["checkpoints"],probe_spacing=protocol["probe_spacing"],
                time_order=protocol["time_order"],**settings)
        elapsed=time.perf_counter()-started
        grids=base.grid_set()
        truth=reference_truth(screen_freeze,grids)
        scores={n:case.score(sol,fields,grids) for n,fields in truth.items()}
        worst={n:case.maxima(rows) for n,rows in scores.items()}
        wake=base.wake(sol)
        tmp=artifact.with_suffix(".tmp.npz")
        case.store_solution(tmp,sol,settings)
        tmp.replace(artifact)
        item.update(status="complete",metadata=sol.row,scores=scores,worst=worst,wake=wake,
            direct_pass=bool(wake["sustained_alternating_probe_signs"] and
                all(max(r.values())<=screen_freeze["protocol"]["gate"] for r in worst.values())),
            artifact=artifact.name,sha256=base.digest(artifact),diagnostic_seconds=elapsed)
    except (ArithmeticError,RuntimeError,ValueError,OSError) as exc:
        details=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)
                 for f in traceback.extract_tb(exc.__traceback__)]
        item.update(status="failed",failure=str(exc),exception_type=type(exc).__name__,
            first_nonfinite=exc.row if isinstance(exc,FirstNonfinite) else None,
            trace=list(trace.recent),stack=details,diagnostic_seconds=time.perf_counter()-started)
    base.save(path,item)
    print("END",job,item["status"],item.get("first_nonfinite",item.get("worst")),flush=True)
    return item


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers",type=int,choices=(2,3),default=3)
    args=cli.parse_args()
    protocol=base.read(PROTOCOL)
    path=HERE/protocol["step_screen"]
    screen=base.read(path/"qualification.json")
    source=base.read(path/"freeze.json")
    if (screen["status"]!="complete" or screen["freeze_sha256"]!=base.digest(path/"freeze.json") or
            protocol["checkpoints"]!=source["protocol"]["checkpoints"] or
            protocol["horizon"]!=source["protocol"]["horizon"] or
            protocol["probe_spacing"]!=source["protocol"]["probe_spacing"]):
        raise RuntimeError("Step qualification input changed")
    for name,digest in (source["sources"]|source["references"]).items():
        if base.digest(base.WORK/name)!=digest:
            raise RuntimeError(f"Source or reference changed: {name}")
    gpu=[]
    for job in protocol["jobs"]:
        r=next(r for r in screen["gpu"] if r["job"]==job)
        if "failure" not in r or "nonfinite" not in r["failure"].lower():
            raise RuntimeError("Expected a matched failed CUDA trajectory")
        record=path/f"{Screen.tag(job,'cuda_graph')}.json"
        if base.read(record)!=r:
            raise RuntimeError("Saved CUDA failure row changed")
        gpu.append(dict(job=job,record=record.relative_to(HERE).as_posix(),
                        sha256=base.digest(record),failure=r["failure"]))
    fixed=dict(protocol=protocol,step_freeze_sha256=screen["freeze_sha256"],
               step_qualification_sha256=base.digest(path/"qualification.json"),
               source_sha256=base.digest(Path(__file__)),gpu_failures=gpu)
    OUT.mkdir(exist_ok=True)
    if JOURNAL.exists():
        journal=base.read(JOURNAL)
        if journal["frozen"]!=fixed:
            raise RuntimeError("Frozen CPU diagnostic changed")
    else:
        journal=dict(status="running",frozen=fixed,rows=[],parent_pid=os.getpid())
        base.save(JOURNAL,journal)
    if journal["status"]=="complete":
        return
    todo=[j for j in protocol["jobs"] if not any(r["job"]==j for r in journal["rows"])]
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for f in as_completed([workers.submit(run,j,protocol,source) for j in todo]):
            journal["rows"].append(f.result())
            base.save(JOURNAL,journal)
    journal["status"]="complete"
    base.save(JOURNAL,journal)


if __name__=="__main__":
    main()
