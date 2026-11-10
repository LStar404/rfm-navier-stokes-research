"""Fixed algebra audit, including arbitrary frozen curl and homogeneous energy."""
import hashlib
import json
from pathlib import Path
import time
from lamb_linearized import LambGalerkin, np, case
from radial_layers import radial_quadrature

HERE = Path(__file__).resolve().parent
OUT = HERE / "lamb_linearized_audit.json"
RECORDS = HERE / "lamb_linearized_fixed_audit_output"
TEMPLATE = HERE / "protocol_lamb_fixed_audit.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)


def diffusion(model):
    _, w = radial_quadrature(model.metadata["q"], model.metadata["split"])
    f, d, d2 = model.samples
    a = (d2-f)*(np.sqrt(w)/model.r)[:, None]
    b = f*(np.sqrt(w)/model.r)[:, None]
    cross = a.T@b
    delta = (model.k2-1)[:, None, None]
    k = case.NU*((a.T@a)-delta*(cross+cross.T)+delta**2*(b.T@b))
    strain = (model.mean_samples[0]+model.mean_samples[1])*np.sqrt(w)[:,None]
    km = case.NU*(strain.T@strain)
    return k, km


def apply_diffusion(model, matrices, state):
    k, km = matrices
    out = np.zeros_like(state)
    values = np.ascontiguousarray(state[:, 1:].T).view(float).reshape(model.k,model.width,2)
    out[:, 1:] = (k@values).view(complex)[..., 0].T
    out[:model.mean_width, 0] = km@state[:model.mean_width, 0].real
    return out


def pair(model, a, b):
    return float(model.pack(a)@model.pack(b))


def norm_mass(model, state):
    return pair(model,state,model.mass_only(state))


def relative(delta, original):
    return float(np.linalg.norm(delta)/max(np.linalg.norm(original),np.finfo(float).tiny))


def condition(model, matrices, h):
    k,km = matrices
    mass = model.M0+model.k2[:,None,None]*model.M1
    blocks = 1.5*mass+h*k
    mean = 1.5*model.mean_mass+h*km
    eig = np.linalg.eigvalsh(blocks)
    eig_mean = np.linalg.eigvalsh(mean)
    lower,upper = min(eig.min(),eig_mean.min()),max(eig.max(),eig_mean.max())
    defect = np.linalg.norm(blocks@model.implicit-np.eye(model.width),axis=(1,2)).max()
    mean_defect = np.linalg.norm(mean@model.mean_inverse-np.eye(model.mean_width))
    return dict(min_eigenvalue=float(lower),max_eigenvalue=float(upper),euclidean_condition=float(upper/lower),
                max_block_inverse_defect_frobenius=float(max(defect,mean_defect)))


def audit_step(model, a, b, h, matrices, amplitudes, rtol, gates):
    begun = time.perf_counter()
    newer, work = model.implicit_step(a,b,amplitudes,h,rtol=rtol,enforce=False)
    elapsed = time.perf_counter()-begun
    gprev,gnow,gnew = amplitudes
    e_old = norm_mass(model,a)+norm_mass(model,2*a-b)
    e_new = norm_mass(model,newer)+norm_mass(model,2*newer-a)
    diss = norm_mass(model,newer-2*a+b)+4*h*pair(model,newer,apply_diffusion(model,matrices,newer))
    homogeneous = 2*model.mass_only(a)-.5*model.mass_only(b)
    lift = h*model.flux_frozen(np.zeros_like(a),gnew)
    lift[:,1] += (2*gnow-.5*gprev)*model.M_lift-gnew*model.bdf_lift
    rhs = homogeneous+lift
    lhs = 1.5*model.mass_only(newer)+h*apply_diffusion(model,matrices,newer)-h*model.flux_frozen(newer)
    residual = lhs-rhs
    scale = max(e_old,np.finfo(float).tiny)
    lift_work = 4*pair(model,newer,lift)
    residual_work = 4*pair(model,newer,residual)
    balance = e_new-e_old+diss-lift_work
    energy_defect = abs(balance)/scale
    corrected = abs(balance-residual_work)/scale
    equation = relative(model.pack(residual),model.pack(rhs))
    passed = (work["converged"] and equation <= gates["original_equation_relative"] and
              energy_defect <= gates["raw_energy_defect_relative"] and corrected <= gates["work_corrected_energy_defect_relative"] and
              (any(amplitudes) or e_new/e_old <= 1+gates["raw_energy_defect_relative"]))
    return dict(linear_work=work,diagnostic_step_seconds=elapsed,original_equation_relative=equation,
                energy_ratio=e_new/e_old,raw_energy_defect_relative=energy_defect,work_corrected_energy_defect_relative=corrected,
                lift_work_relative=lift_work/scale,residual_work_relative=residual_work/scale,passed=bool(passed))


def case_row(job, settings, rng):
    h = 1/job["rate"]
    model = LambGalerkin(dt=h,**job["config"])
    a = model.unpack(.01*rng.standard_normal(model.real_size))
    b = model.unpack(.01*rng.standard_normal(model.real_size))
    spatial = (model.samples,model.lift,model.r,model.mean_samples)
    omega = model.fields_on(a,.3+.2j,*spatial)[2].copy()
    model.freeze_vorticity(omega)
    fa,fb = model.flux_frozen(a),model.flux_frozen(b)
    cross = abs(pair(model,a,fb)+pair(model,b,fa))
    scale = np.linalg.norm(model.pack(a))*np.linalg.norm(model.pack(fb))+np.linalg.norm(model.pack(b))*np.linalg.norm(model.pack(fa))
    skew = float(cross/max(scale,np.finfo(float).tiny))
    frozen = model.frozen_omega.copy()
    same = model.flux_frozen(a,.3+.2j)
    expected = model.terms(a,.3+.2j)[1]
    matching = relative(model.pack(same-expected),model.pack(expected))
    past = model.fields_on(b,-.1+.05j,*spatial)[2].copy()
    predictor = model.fields_on(2*a-b,2*(.3+.2j)-(-.1+.05j),*spatial)[2]
    predictor_error = relative(predictor-(2*omega-past),2*omega-past)
    repeated = bool(np.array_equal(frozen,model.frozen_omega) and np.array_equal(model.flux_frozen(a),fa))
    matrices = diffusion(model)
    gates = settings["fixed_gates"]
    algebra = (repeated and skew <= gates["skew_pairing_relative"] and matching <= gates["original_flux_relative"] and
               predictor_error <= gates["extrapolated_curl_relative"])
    trials = {}
    for name,amplitudes,rtol in (("strict",(0.,0.,0.),settings["strict_rtol"]),
                                 ("screen",(0.,0.,0.),settings["screen_rtol"]),
                                 ("screen_lift",(-.1+.05j,.3+.2j,.4-.05j),settings["screen_rtol"])):
        try:
            trials[name] = audit_step(model,a,b,h,matrices,amplitudes,rtol,gates)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            trials[name] = dict(passed=False,failure_type=type(exc).__name__,failure=str(exc))
    return dict(job=job,status="complete",skew_pairing_relative=skew,original_flux_relative=matching,
                extrapolated_curl_relative=predictor_error,frozen_curl_and_matvec_repeat_bitwise=repeated,
                matrix=condition(model,matrices,h),algebra_pass=bool(algebra),trials=trials,
                screen_pass=bool(algebra and trials["screen"]["passed"] and trials["screen_lift"]["passed"]))


def main():
    if OUT.exists() or RECORDS.exists():
        raise FileExistsError("Preserve the existing fixed-state audit")
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    jobs = []
    for family in template["families"]:
        for size in template["sizes"]:
            small = size == "small"
            config = dict(m=13 if small else 37,k=8 if small else 72,q=36 if small else 66,scale=62.,
                          mean_order=10 if small else 24,layers=[] if small else [16,36,80,165,340,660],
                          split=True,backend="fft",seed=7331 if family=="rfm" else 17,family=family)
            jobs.extend(dict(family=family,size=size,rate=rate,config=config) for rate in template["rates"])
    RECORDS.mkdir()
    files = (Path(__file__),TEMPLATE,HERE/"lamb_linearized.py",HERE/"lamb_linearized_audit_strict_v1_failure.json")
    fingerprints = {p.name:digest(p) for p in files}
    save(RECORDS/"protocol.json",dict(status="frozen",settings=template,source_sha256=fingerprints,jobs=jobs))
    rows, evidence = [],{}
    rng = np.random.default_rng(template["state_seed"])
    for job in jobs:
        try:
            row = case_row(job,template,rng)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            row = dict(job=job,status="failed",screen_pass=False,failure_type=type(exc).__name__,failure=str(exc))
        name = "_".join(str(job[k]) for k in ("family","size","rate"))
        artifact = RECORDS/(name+".json")
        save(artifact,row)
        evidence[artifact.name] = digest(artifact)
        rows.append(row)
        print(name,row["screen_pass"],row.get("trials",{}).get("strict",{}).get("passed"),flush=True)
    if fingerprints != {p.name:digest(p) for p in files}:
        raise RuntimeError("Audit source changed during the experiment")
    strict_count = sum(r.get("trials",{}).get("strict",{}).get("passed",False) for r in rows)
    screen_count = sum(r["screen_pass"] for r in rows)
    save(OUT,dict(status="passed" if screen_count == len(rows) else "failed",scope=template["scope"],
                  protocol_sha256=digest(RECORDS/"protocol.json"),raw_records=evidence,strict_passed=strict_count,
                  screen_passed=screen_count,rows=rows))


if __name__ == "__main__":
    main()
