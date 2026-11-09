"""Audit fixed-state algebra and qualify the new CPU native contraction."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path

from cpu_precontracted_native import NativePrecontractedGalerkin, PrecontractedGalerkin, RecurrenceGalerkin
import cpu_formal as base
import cpu_formal_step_screen as step
import fd_pow2_reference_chain as chain
import joint_hardware_pow2 as joint


HERE=base.HERE
OUT=HERE/"cpu_native_qualification_output"
PROTOCOL=HERE/"protocol_cpu_native_qualification.json"
FIXED=OUT/"fixed_state_audit.json"
JOURNAL=OUT/"qualification.json"
case,np=base.case,base.case.np


def path_and_hash(row):
    path=HERE/row["artifact"]
    if base.digest(path)!=row["sha256"]:
        raise RuntimeError(f"Frozen trajectory changed: {path.name}")
    return path


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),np.finfo(float).tiny))


def manifest():
    p=base.read(PROTOCOL)
    prior=base.read(step.OUT/"qualification.json")
    screen=base.read(step.OUT/"freeze.json")
    if (prior["status"]!="complete" or base.digest(step.OUT/"freeze.json")!=prior["freeze_sha256"] or
            {r["name"]:r["selected_rate"] for r in prior["selected"]}!=dict.fromkeys(p["methods"],p["rate"])|{"fd176_k80":128}):
        raise RuntimeError("Need the completed selected-rate screen")
    hw=base.read(joint.FROZEN)
    joint.approved_inputs(hw)
    specs={s["name"]:s for s in hw["protocol"]["variants"] if s["name"] in p["methods"]}
    seeds=hw["protocol"]["seeds"]
    if len(seeds)!=5 or len(set(seeds))!=5:
        raise RuntimeError("Expected five distinct preregistered RFM seeds")
    jobs=[]
    for name in p["methods"]:
        for seed in (seeds if specs[name]["method"]=="rfm" else [hw["protocol"]["deterministic_seed"]]):
            records=[]
            for rate in (p["rate"],screen["protocol"]["anchor_rate"]):
                job=dict(name=name,seed=seed,rate=rate)
                control_path=step.OUT/f"{step.Screen.tag(job,'cpu')}.json"
                row=base.read(control_path)
                if (row["job"]!=job or row["backend"]!="cpu" or not row["direct_pass"] or
                        row["metadata"]["basis_evaluation"]!="differentiated-recurrence"):
                    raise RuntimeError("Saved CPU recurrence control differs from the selected discretization")
                path_and_hash(row)
                records.append(dict(source=control_path.relative_to(base.WORK).as_posix(),
                                    sha256=base.digest(control_path)))
            jobs.append(dict(name=name,seed=seed,controls=records))
    hashes=hw["source_hashes"]|screen["sources"]
    for name in (Path(__file__).name,PROTOCOL.name,"cpu_precontracted.py","cpu_precontracted_native.py",
                 "galerkin_cpu_contract.c","galerkin_cpu_contract.dll"):
        hashes[f"rfm_v7/{name}"]=base.digest(HERE/name)
    for name,sha in hashes.items():
        if base.digest(base.WORK/name)!=sha:
            raise RuntimeError(f"Source differs from the archived control: {name}")
    files={path.relative_to(base.WORK).as_posix():base.digest(path)
           for path in (step.OUT/"qualification.json",step.OUT/"freeze.json",joint.FROZEN)}
    return dict(protocol=p,solve_protocol=hw["protocol"],specs=specs,jobs=jobs,
                source_hashes=hashes,qualification_files=files,reference_hashes=screen["references"])


def control(job,index):
    entry=job["controls"][index]
    path=base.WORK/entry["source"]
    if base.digest(path)!=entry["sha256"]:
        raise RuntimeError("Frozen control record changed")
    row=base.read(path)
    return row,case.load_solution(path_and_hash(row),RecurrenceGalerkin)


def fixed_state(frozen):
    tolerance=frozen["protocol"]["fixed_state_tolerance"]
    rows=[]
    for name in frozen["protocol"]["methods"]:
        job=next(j for j in frozen["jobs"] if j["name"]==name)
        _,sol=control(job,0)
        spec=frozen["specs"][name]
        _,cfg=base.config(spec,frozen["solve_protocol"],job["seed"])
        dt=1/frozen["protocol"]["rate"]
        reference=PrecontractedGalerkin(dt=dt,**cfg)
        native=NativePrecontractedGalerkin(dt=dt,**cfg)
        state=sol.states[-1].copy()
        width,modes=state.shape
        amp=complex(.33,-.12)
        normal=reference.terms(state,amp)
        perturbed=state.copy()
        perturbed[:,0]+=1e3j
        perturbed[native.mean_width:,0]+=1e3
        padded=np.zeros((width,modes+3),dtype=np.complex128)[:,:modes]
        padded[:]=perturbed
        padded.setflags(write=False)
        snapshots=[np.array(x,copy=True) for x in (state,perturbed,padded)]
        candidates=[native.terms(x,amp) for x in (state,perturbed,padded)]
        labels=("normal", "finite_outside_mean_subspace", "positive_padded_readonly")
        for label,x,saved,outputs in zip(labels,(state,perturbed,padded),snapshots,candidates,strict=True):
            errors=[relative(y,z) for y,z in zip(outputs,normal,strict=True)]
            if (max(errors)>tolerance or not all(math.isfinite(e) for e in errors) or
                    not np.array_equal(x,saved) or np.shares_memory(*outputs) or
                    any(np.shares_memory(x,y) or np.any(y[:,0].imag) or np.any(y[native.mean_width:,0]) for y in outputs)):
                raise RuntimeError(f"Fixed-state native comparison failed: {name}/{label}: {errors}")
            rows.append(dict(name=name,test=label,errors=errors,state_stride_bytes=list(x.strides)))
        unaligned=np.ndarray(state.shape,dtype=np.complex128,buffer=bytearray(state.nbytes+1),offset=1)
        invalid=(('fortran',np.asfortranarray(state)),('negative_stride',state[::-1]),
                 ('wrong_shape',state[:,:-1]),('unaligned',unaligned))
        for reason,variant in invalid:
            try:
                native.terms(variant,amp)
            except ValueError:
                rows.append(dict(name=name,test=f"reject_{reason}",passed=True))
            else:
                raise RuntimeError(f"Expected the native ABI to reject {reason} state")
        rhs=1.5*candidates[0][0]+dt*candidates[0][1]
        rhs_reference=1.5*normal[0]+dt*normal[1]
        original_mass=candidates[0][0].copy()
        original_flux=candidates[0][1].copy()
        mapped=native.advance(rhs,amp,False)
        expected=reference.advance(rhs_reference,amp,False)
        lift_error=relative(mapped,expected)
        first_error=relative(native.advance(np.zeros_like(rhs),amp,True),
                             sol.model.advance(np.zeros_like(rhs),amp,True))
        if (mapped is not rhs or max(lift_error,first_error)>tolerance or
                not np.array_equal(original_mass,candidates[0][0]) or not np.array_equal(original_flux,candidates[0][1])):
            raise RuntimeError("Advance mutated histories or differs from the original startup/lift")
        rows.append(dict(name=name,test="owned_rhs_and_lifts",relative_lift_error=lift_error,relative_first_error=first_error))
    result=dict(status="passed",source_hashes=frozen["source_hashes"],rows=rows)
    if FIXED.exists() and base.read(FIXED)!=result:
        raise RuntimeError("Existing fixed-state audit differs")
    if not FIXED.exists():
        base.save(FIXED,result)
    return result


_run=None
_grids=None
_truth=None


def start_worker(frozen):
    global _run,_grids,_truth
    _run=frozen
    _grids=base.grid_set()
    journal=base.read(chain.JOURNAL)
    records={r["spec"]["name"]:r for r in journal["rows"]}
    _truth={}
    for name in frozen["protocol"]["reference_names"]:
        row=records[name]
        path=base.WORK/row["spec"]["source"]
        if row["status"]!="complete" or base.digest(path)!=frozen["reference_hashes"][row["spec"]["source"]]:
            raise RuntimeError("A frozen reference trajectory differs")
        sol=case.load_solution(path,chain.MODEL)
        if (not np.array_equal(sol.times,frozen["solve_protocol"]["checkpoints"]) or
                sol.row["row_scale"]!=sol.model.scale_metadata()):
            raise RuntimeError("Reference reconstruction differs")
        _truth[name]=case.reference_fields(sol,_grids)


def qualification(job):
    p,protocol=_run["protocol"],_run["solve_protocol"]
    name,seed=job["name"],job["seed"]
    path=OUT/f"{name}_{seed}_r{p['rate']}.npz"
    record=path.with_suffix(".json")
    if record.exists():
        row=base.read(record)
        if row["job"]!=job or base.digest(path)!=row["sha256"]:
            raise RuntimeError("Saved accuracy result or file differs")
        return row
    if path.exists():
        raise FileExistsError(f"Inspect orphaned candidate: {path.name}")
    print("START",name,seed,flush=True)
    _,cfg=base.config(_run["specs"][name],protocol,seed)
    sol=case.solve(NativePrecontractedGalerkin,steps=protocol["horizon"]*p["rate"],**cfg)
    _,cpu=control(job,0)
    anchor_row,anchor=control(job,1)
    if (not np.array_equal(sol.times,cpu.times) or not np.array_equal(sol.times,anchor.times) or
            not np.array_equal(sol.probe_times,cpu.probe_times)):
        raise RuntimeError("Candidate and controls differ in physical observation times")
    expected={(g,t) for g in _grids for t in protocol["checkpoints"]}
    direct={n:case.score(sol,truth,_grids) for n,truth in _truth.items()}
    worst={n:joint.score_rows(rows,expected) for n,rows in direct.items()}
    fields=joint.score_rows(case.score(sol,case.reference_fields(cpu,_grids),_grids),expected)
    probes={k:relative(getattr(sol,k),getattr(cpu,k)) for k in ("transverse","vorticity")}
    delta=case.score(sol,case.reference_fields(anchor,_grids),_grids)
    triangle={}
    for ref,scored in anchor_row["scores"].items():
        upper=[]
        for a,b in zip(delta,scored,strict=True):
            if (a["grid"],a["time"])!=(b["grid"],b["time"]):
                raise RuntimeError("Expected paired grid/time samples for anchor bound")
            upper.append(dict(grid=a["grid"],time=a["time"],**{
                k:a[k]*(1+b[k])+b[k] for k in joint.METRICS}))
        triangle[ref]=joint.score_rows(upper,expected)
    wake=base.wake(sol)
    implementation=all(math.isfinite(v) and v<=p["implementation_tolerance"] for v in (*fields.values(),*probes.values()))
    direct_pass=all(max(v.values())<=protocol["gate"] for v in worst.values()) and wake["sustained_alternating_probe_signs"]
    guard=all(max(v.values())<=protocol["gate"] for v in triangle.values())
    temp=path.with_suffix(".tmp.npz")
    case.store_solution(temp,sol,cfg)
    temp.replace(path)
    row=dict(job=job,artifact=path.name,sha256=base.digest(path),scores=direct,worst=worst,wake=wake,
             same_cpu_fields=fields,same_cpu_probes=probes,implementation_pass=implementation,
             direct_pass=direct_pass,rate128_triangle_worst=triangle,rate128_guard_pass=guard,
             qualified=bool(implementation and direct_pass and guard),metadata=sol.row)
    base.save(record,row)
    print("DONE",name,seed,row["qualified"],worst,flush=True)
    return row


def execute(frozen):
    fixed=fixed_state(frozen)
    existing=base.read(JOURNAL) if JOURNAL.exists() else None
    result=dict(status="running",frozen=frozen,fixed_state_sha256=base.digest(FIXED),rows=[])
    if existing is not None:
        if existing["frozen"]!=frozen or existing["fixed_state_sha256"]!=result["fixed_state_sha256"]:
            raise RuntimeError("Frozen qualification inputs differ")
        if existing["status"]=="complete":
            return
    base.save(JOURNAL,result)
    print("FIXED",fixed["status"],len(fixed["rows"]),flush=True)
    with ProcessPoolExecutor(max_workers=frozen["protocol"]["workers"],initializer=start_worker,
                             initargs=(frozen,)) as pool:
        for task in as_completed([pool.submit(qualification,job) for job in frozen["jobs"]]):
            result["rows"].append(task.result())
            base.save(JOURNAL,result)
    if manifest()!=frozen or base.digest(FIXED)!=result["fixed_state_sha256"]:
        raise RuntimeError("Frozen accuracy inputs changed")
    result.update(status="complete",passes=len(result["rows"])==6 and all(r["qualified"] for r in result["rows"]))
    base.save(JOURNAL,result)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--execute",action="store_true")
    cli.add_argument("--fixed-only",action="store_true")
    args=cli.parse_args()
    if not (args.execute or args.fixed_only):
        cli.error("Choose an accuracy action")
    frozen=manifest()
    if args.execute:
        execute(frozen)
    else:
        print(fixed_state(frozen)["status"],flush=True)


if __name__=="__main__":
    main()
