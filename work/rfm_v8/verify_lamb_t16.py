"""Reconcile raw Lamb FFT work and the T16 qualification from saved data."""

import json
from pathlib import Path

from lamb_t16_qualification import OUT, PROBES, RecurrenceGalerkin, case, digest, np, require, work_counts


HERE = Path(__file__).resolve().parent
SUMMARY = OUT/"summary.json"
METRICS = ("velocity","vorticity","near_velocity","near_vorticity")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def high_mode_fraction(state,model,first_mode):
    mass = model.M0+model.k2[:,None,None]*model.M1
    modes = state[:,1:].T
    e = 2*(modes.conj()*(mass@modes[...,None])[...,0]).sum(axis=1).real
    mean = state[:model.mean_width,0].real
    total = e.sum()+mean@model.mean_mass@mean
    require(total > 0 and np.isfinite(e).all(),"Invalid saved kinetic-energy projection")
    return float(e[first_mode-1:].sum()/total)


def verify(job,p,protocol_sha):
    directory = OUT/f"{job['name']}_r{job['rate']}"
    result = read(directory/"qualification.json")
    ledger = directory/"steps.jsonl"
    artifact = directory/"candidate.npz"
    require(result["job"] == job and result["protocol_sha256"] == protocol_sha and
            result["step_ledger_sha256"] == digest(ledger),"Lamb case inputs changed")
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if result["status"] != "complete":
        require(not result["qualified"],"A failed trajectory cannot qualify")
        return dict(job=job,status=result["status"],qualified=False,steps_observed=len(lines))
    require(digest(artifact) == result["artifact_sha256"],"Saved trajectory changed")
    n,q = p["horizon"]*job["rate"],job["config"]["q"]
    require([r["step"] for r in lines] == list(range(2,n+1)) and all(r["converged"] and r["gmres_info"] == 0 and
            r["requested_rtol"] == p["linear_rtol"] and r["velocity_product_calls"] == r["operator_calls"]+1 for r in lines),
            "Per-step FFT or iterative records are incomplete")
    products = sum(r["velocity_product_calls"] for r in lines)
    physical_checks = [r["physical_residual"] for r in lines if "physical_residual" in r]
    require(physical_checks == result["checkpoint_residuals"] and [r["time"] for r in physical_checks] == p["physical_times"],
            "Saved original-system residuals differ from step records")
    meta = result["metadata"]
    expected = work_counts(n-1,products,q)
    diagnostic_products = 2*len(physical_checks)
    require(meta["solver_work"] == expected and meta["diagnostic_work"] == work_counts(0,diagnostic_products,q) and
            meta["full_work"] == work_counts(n-1,products+diagnostic_products,q) and
            meta["gmres_iterations"] == sum(r["iterations"] for r in lines),"Archived FFT batch arithmetic differs")
    with np.load(artifact,allow_pickle=False) as raw:
        times = raw["times"]
        states = raw["states"]
        require(np.array_equal(times,p["physical_times"]) and np.array_equal(raw["probe_times"],np.arange(129)/8) and
                all(raw[k].shape == (129,3) and np.isfinite(raw[k]).all() for k in PROBES) and
                states.shape[0] == 12 and np.isfinite(states).all(),"Physical observations or full probes differ")
    worst,failures = {},set()
    for reference in p["reference_names"]:
        ref = result["references"][reference]
        rows = ref["physical_rows"]
        require(len({(r["grid"],r["time"]) for r in rows}) == len(rows) == 36 and
                {r["time"] for r in rows} == set(p["physical_times"]),"Physical scoring grid inventory differs")
        maxima = {k:max(r[k] for r in rows) for k in METRICS}
        require(maxima == ref["physical_max"],"Archived physical maxima differ")
        failures.update(r["time"] for r in rows if max(r[k] for k in METRICS) > p["physical_gate"])
        worst[reference] = maxima
    peak = {k:max(r[k] for r in physical_checks) for k in (
        "original_equation_relative","raw_energy_defect_relative","work_corrected_energy_defect_relative")}
    valid = (peak["original_equation_relative"] <= p["original_equation_gate"] and
             peak["raw_energy_defect_relative"] <= p["raw_energy_defect_gate"] and
             peak["work_corrected_energy_defect_relative"] <= p["work_corrected_energy_gate"])
    require(result["residual_pass"] == valid and result["physical_pass"] == (not failures) and
            result["qualified"] == bool(valid and not failures and result["control_pass"]),"Qualification differs from raw gates")
    anchor = HERE.parent/job["control"]["path"]
    require(digest(anchor) == job["control"]["sha256"],"Historical same-space control changed")
    control = case.load_solution(anchor,RecurrenceGalerkin)
    require(np.array_equal(control.times,times) and control.states.shape == states.shape,"Saved matched spatial states differ")
    first = min(failures,default=None)
    angular = None
    if first is not None:
        i = times.tolist().index(first)
        angular = dict(time=first,first_mode=48,scope="Post hoc homogeneous velocity mass diagnostic; no change to the physical gate",
                       candidate=high_mode_fraction(states[i],control.model,48),control=high_mode_fraction(control.states[i],control.model,48))
    base = result["baseline_work"]
    require(base == dict(nonlinear_evaluations=2047,fft_batches=4094,radial_line_transforms=5*q*2047),"Control FFT work differs")
    return dict(job=job,status="complete",qualified=result["qualified"],first_failed_saved_time=first,
                candidate_high_mode_fraction=angular,worst_by_reference=worst,checkpoint_residual_max=peak,
                solver_work=expected,diagnostic_fft_batches=meta["diagnostic_work"]["fft_batches"],
                fft_batch_ratio=expected["fft_batches"]/base["fft_batches"],
                radial_line_ratio=expected["radial_line_transforms"]/base["radial_line_transforms"],
                qualification_sha256=digest(directory/"qualification.json"))


def main():
    require(not SUMMARY.exists(),"Preserve the existing independent summary")
    path = OUT/"protocol.json"
    frozen = read(path)
    require(all(digest(HERE.parent/name) == sha for name,sha in frozen["source_hashes"].items()),"Frozen sources differ")
    sha = digest(path)
    rows = [verify(job,frozen["template"],sha) for job in frozen["jobs"]]
    result = dict(status="verified",protocol_sha256=sha,verifier_sha256=digest(Path(__file__)),
                  scope="Reconciled raw ledgers and archived scores. No new time solve.",rows=rows,
                  qualified=sum(r["qualified"] for r in rows),gpu_port_supported=False)
    with SUMMARY.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(digest(SUMMARY))


if __name__ == "__main__":
    main()
