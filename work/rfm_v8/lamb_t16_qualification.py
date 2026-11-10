"""Preserve four independent accuracy-only Lamb trajectories and their work ledger."""

import argparse
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
sys.path.insert(0,str(V7))

from audit_lamb_linearized import apply_diffusion, diffusion, digest, norm_mass, pair, relative, save
from lamb_linearized import LambStepError, RecurrenceGalerkin, case, np
from lamb_symmetric import SymmetricLambGalerkin
import cpu_formal as base
import switch_qualification as old
from verify_switch_qualification import absolute_reference, fields_at, maxima, measured, require


TEMPLATE = HERE/"protocol_lamb_t16.json"
OUT = HERE/"lamb_t16_qualification_output"
AUDIT = HERE/"lamb_symmetric_fixed_audit_output"
PROBES = ("transverse","vorticity")


def freeze():
    require(not OUT.exists(),"Preserve the existing Lamb T16 attempt")
    p,a,s = (base.read(x) for x in (TEMPLATE,AUDIT/"protocol.json",AUDIT/"summary.json"))
    require(s["status"] == "passed" and s["practical_passed"] == 4 and s["protocol_sha256"] == digest(AUDIT/"protocol.json"),
            "The practical fixed-state cases must pass before registration")
    practical = [r for r in s["rows"] if r["job"]["size"] == "mid" and r["job"]["rate"] in p["rates"]]
    require(len(practical) == 4 and all(r["screen_pass"] and r["trials"]["screen"]["linear_work"]["requested_rtol"] == p["linear_rtol"]
                                       for r in practical),"Fixed and trajectory linear tolerance differ")
    for name,sha in s["raw_records"].items():
        require(digest(AUDIT/name) == sha,"Fixed case record changed")
    archive,summary = base.read(old.FROZEN),base.read(old.SUMMARY)
    require(summary["passes"] and summary["protocol_sha256"] == digest(old.FROZEN) and
            archive["physical_times"][:12] == p["physical_times"],"Registered anchor observations differ")
    jobs, inputs = [],[Path(__file__),TEMPLATE,AUDIT/"protocol.json",AUDIT/"summary.json",old.FROZEN,old.SUMMARY,
                      HERE/"lamb_symmetric.py",HERE/"lamb_linearized.py",HERE/"audit_lamb_linearized.py",HERE/"galerkin_minimal_model.py"]
    for name,seed in p["families"].items():
        anchor = next(r for r in summary["anchors"] if r["name"] == name and r["seed"] == seed)
        path = old.OUT/anchor["artifact"]
        require(digest(path) == anchor["sha256"],"Archived r128 anchor hash differs")
        inputs.append(path)
        config = base.config(archive["specs"][name],archive["solve_protocol"],seed)[1]
        jobs += [dict(name=name,seed=seed,rate=rate,config=config,control=dict(path=path.relative_to(HERE.parent).as_posix(),
                      sha256=anchor["sha256"])) for rate in p["rates"]]
    sources = {x.relative_to(HERE.parent).as_posix():digest(x) for x in inputs}
    OUT.mkdir()
    save(OUT/"protocol.json",dict(status="frozen",template=p,source_hashes=sources,jobs=jobs))
    print("FROZEN",digest(OUT/"protocol.json"),len(jobs),flush=True)


def work_counts(s,p,q):
    return dict(frozen_vorticity_evaluations=s,velocity_product_evaluations=p,fft_batches=s+2*p,
                inverse_field_transforms=s+2*p,forward_field_transforms=2*p,
                inverse_radial_line_transforms=q*(s+2*p),forward_radial_line_transforms=2*q*p,
                radial_line_transforms=q*(s+4*p))


def physical_equation(model, matrices, h, now, past, new, amplitudes):
    gprev,gnow,gnew = amplitudes
    lift = h*model.flux_frozen(np.zeros_like(now),gnew)
    lift[:,1] += (2*gnow-.5*gprev)*model.M_lift-gnew*model.bdf_lift
    rhs = 2*model.mass_only(now)-.5*model.mass_only(past)+lift
    viscous = apply_diffusion(model,matrices,new)
    residual = 1.5*model.mass_only(new)+h*viscous-h*model.flux_frozen(new)-rhs
    before = norm_mass(model,now)+norm_mass(model,2*now-past)
    after = norm_mass(model,new)+norm_mass(model,2*new-now)
    dissipated = norm_mass(model,new-2*now+past)+4*h*pair(model,new,viscous)
    lift_work = 4*pair(model,new,lift)
    residual_work = 4*pair(model,new,residual)
    balance = after-before+dissipated-lift_work
    scale = max(before,np.finfo(float).tiny)
    return dict(original_equation_relative=relative(model.pack(residual),model.pack(rhs)),
                raw_energy_defect_relative=abs(balance)/scale,work_corrected_energy_defect_relative=abs(balance-residual_work)/scale,
                lift_work_relative=lift_work/scale,residual_work_relative=residual_work/scale)


def march(job,p,directory):
    start = time.perf_counter()
    rate,h = job["rate"],1/job["rate"]
    steps = p["horizon"]*rate
    times = np.asarray(p["physical_times"])
    probe_times = np.arange(0,p["horizon"]+p["probe_spacing"],p["probe_spacing"])
    snapshots = dict(zip(case.step_indices(times,h),times,strict=True))
    probes = dict(zip(case.step_indices(probe_times,h),range(len(probe_times)),strict=True))
    model = SymmetricLambGalerkin(dt=h,**job["config"])
    matrices = diffusion(model)
    amplitude = case.speed(h*np.arange(steps+1))
    prior = np.zeros(model.rhs_shape,dtype=np.complex128)
    state = model.advance(prior.copy(),amplitude[1],True)
    values = {name:np.zeros((len(probe_times),len(case.PROBES))) for name in PROBES}
    saved,checks,products,diagnostic_products,iterations = [],[],0,0,0
    max_prec = 0.
    with (directory/"steps.jsonl").open("x",encoding="utf-8") as ledger:
        for i in range(1,steps+1):
            if i > 1:
                g = amplitude[i-2:i+1]
                try:
                    new,record = model.implicit_step(state,prior,g,h,rtol=p["linear_rtol"])
                except LambStepError as exc:
                    exc.diagnostics.update(step=i,time=i*h,previous_fft_work=work_counts(model.frozen_evaluations-1,products,len(model.r)),
                                           including_failed_step_work=work_counts(model.frozen_evaluations,model.velocity_products,len(model.r)))
                    ledger.write(json.dumps(dict(status="failed",**exc.diagnostics),allow_nan=False)+"\n")
                    raise
                products += record["velocity_product_calls"]
                iterations += record["iterations"]
                max_prec = max(max_prec,record["preconditioned_residual"])
                if i in snapshots:
                    before = model.velocity_products
                    check = physical_equation(model,matrices,h,state,prior,new,g)
                    diagnostic_products += model.velocity_products-before
                    check["time"] = float(snapshots[i])
                    checks.append(check)
                    record["physical_residual"] = check
                ledger.write(json.dumps(dict(step=i,time=i*h,**record),allow_nan=False)+"\n")
                prior,state = state,new
            if i in snapshots:
                require(np.isfinite(state).all(),"Nonfinite saved Lamb state")
                saved.append(state.copy())
            if i in probes:
                u,w = model.probes(state,amplitude[i])
                require(np.isfinite(u).all() and np.isfinite(w).all(),"Nonfinite Lamb probe")
                values["transverse"][probes[i]],values["vorticity"][probes[i]] = u,w
    require(len(checks) == len(times) and model.frozen_evaluations == steps-1 and
            model.velocity_products == products+diagnostic_products and diagnostic_products == 2*len(times),
            "Measured FFT work or residual observations differ from the full ledger")
    q = len(model.r)
    row = dict(model.metadata,dt=h,steps=steps,horizon=p["horizon"],implementation="symmetric_lamb_bdf2",
               solver_work=work_counts(model.frozen_evaluations,products,q),diagnostic_work=work_counts(0,diagnostic_products,q),
               full_work=work_counts(model.frozen_evaluations,model.velocity_products,q),gmres_iterations=iterations,
               max_preconditioned_residual=max_prec,diagnostic_seconds=time.perf_counter()-start)
    solution = case.Solution(model,np.asarray(saved),times,probe_times,values["transverse"],values["vorticity"],row)
    return solution,checks


def score(solution,control,p):
    times,grids = p["physical_times"],base.grid_set()
    require(np.array_equal(solution.times,control.times) and np.array_equal(solution.probe_times,control.probe_times) and
            old.switched.equal_spatial_models((solution.model,control.model)),"Same-space r128 control or absolute observations differ")
    truth,budget = absolute_reference(base.read(old.FROZEN),grids)
    s,c = fields_at(solution,times,grids),fields_at(control,times,grids)
    delta = measured(s,c,grids,times)
    scores = {}
    for name in p["reference_names"]:
        subset = {g:values[:len(times)] for g,values in truth[name].items()}
        direct,anchor = measured(s,subset,grids,times),measured(c,subset,grids,times)
        guard = {gt:{k:anchor[gt][k]+(1+anchor[gt][k])*delta[gt][k] for k in direct[gt]} for gt in direct}
        require(all(direct[gt][k] <= guard[gt][k]+2e-13 for gt in direct for k in direct[gt]),"Control triangle does not bound the direct error")
        scores[name] = dict(physical_max=maxima(direct),control_max=maxima(anchor),triangle_max=maxima(guard),
                            physical_rows=[dict(grid=g,time=t,**v) for (g,t),v in direct.items()])
    discrepancy = {k:relative(getattr(solution,k)-getattr(control,k),getattr(control,k)) for k in PROBES}
    physical = all(max(v["physical_max"].values()) <= p["physical_gate"] for v in scores.values())
    anchor_pass = all(max(v["control_max"].values()) <= p["physical_gate"] for v in scores.values())
    return dict(reference_budget=budget,references=scores,control_pass=anchor_pass,physical_pass=physical,
                full_probe_discrepancy=discrepancy,control_discrepancy_max=maxima(delta))


def run(index):
    fixed = base.read(OUT/"protocol.json")
    p,job = fixed["template"],fixed["jobs"][index]
    require(all(digest(HERE.parent/name)==sha for name,sha in fixed["source_hashes"].items()),"Frozen trajectory sources changed")
    directory = OUT/f"{job['name']}_r{job['rate']}"
    directory.mkdir()
    control_path = HERE.parent/job["control"]["path"]
    control = case.load_solution(control_path,RecurrenceGalerkin)
    try:
        solution,checks = march(job,p,directory)
        path = directory/"candidate.npz"
        with path.open("xb") as handle:
            case.store_solution(handle,solution,job["config"])
        evidence = score(solution,control,p)
        residual_pass = all(r["original_equation_relative"] <= p["original_equation_gate"] and
                            r["raw_energy_defect_relative"] <= p["raw_energy_defect_gate"] and
                            r["work_corrected_energy_defect_relative"] <= p["work_corrected_energy_gate"] for r in checks)
        q,n = len(solution.model.r),control.row["steps"]-1
        baseline = dict(nonlinear_evaluations=n,fft_batches=2*n,radial_line_transforms=5*q*n)
        work = solution.row["solver_work"]
        result = dict(job=job,status="complete",qualified=bool(evidence["physical_pass"] and evidence["control_pass"] and residual_pass),
                      residual_pass=residual_pass,checkpoint_residuals=checks,baseline_work=baseline,metadata=solution.row,
                      solver_fft_batch_ratio=work["fft_batches"]/baseline["fft_batches"],
                      solver_radial_transform_ratio=work["radial_line_transforms"]/baseline["radial_line_transforms"],
                      artifact_sha256=digest(path),**evidence)
    except (ArithmeticError,RuntimeError,ValueError) as exc:
        result = dict(job=job,status="failed",qualified=False,failure_type=type(exc).__name__,failure=str(exc),
                      failure_diagnostics=getattr(exc,"diagnostics",None))
    result["protocol_sha256"] = digest(OUT/"protocol.json")
    ledger = directory/"steps.jsonl"
    result["step_ledger_sha256"] = digest(ledger) if ledger.exists() else None
    save(directory/"qualification.json",result)
    print(directory.name,result["status"],result["qualified"],flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze",action="store_true")
    action.add_argument("--job",type=int,choices=range(4))
    args = cli.parse_args()
    freeze() if args.freeze else run(args.job)
