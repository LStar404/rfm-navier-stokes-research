"""Recheck the exact fixed-state inventory in mass-diffusion balanced coordinates."""

import json
from pathlib import Path

from audit_lamb_linearized import audit_step, diffusion, digest, pair, relative, save
from lamb_linearized import LambGalerkin
from lamb_symmetric import SymmetricLambGalerkin, np


HERE = Path(__file__).resolve().parent
OUT = HERE/"lamb_symmetric_fixed_audit_output"
TEMPLATE = HERE/"protocol_lamb_symmetric_fixed_audit.json"
LEFT = HERE/"lamb_linearized_fixed_audit_output"/"protocol.json"
LEFT_REPORT = HERE/"lamb_linearized_audit.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trial_set(model,a,b,h,settings,gates):
    matrices = diffusion(model)
    trials = {}
    for name,amplitudes,tolerance in (("strict",(0.,0.,0.),settings["strict_rtol"]),
                                      ("screen",(0.,0.,0.),settings["screen_rtol"]),
                                      ("screen_lift",(-.1+.05j,.3+.2j,.4-.05j),settings["screen_rtol"])):
        try:
            trials[name] = audit_step(model,a,b,h,matrices,amplitudes,tolerance,gates)
        except (ArithmeticError,RuntimeError,ValueError) as exc:
            trials[name] = dict(passed=False,failure_type=type(exc).__name__,failure=str(exc))
    return trials


def audit(job,a,b,model,settings,gates):
    h = 1/job["rate"]
    spatial = model.samples,model.lift,model.r,model.mean_samples
    model.freeze_vorticity(model.fields_on(2*a-b,.5+.3j,*spatial)[2])

    def skew_action(x):
        v = model.apply_lower(x,inverse=True,transpose=True)
        return model.apply_lower(model.flux_frozen(v),inverse=True,transpose=False)

    fa,fb = skew_action(a),skew_action(b)
    defect = abs(pair(model,a,fb)+pair(model,b,fa))
    denom = np.linalg.norm(model.pack(a))*np.linalg.norm(model.pack(fb))+np.linalg.norm(model.pack(b))*np.linalg.norm(model.pack(fa))
    skew = float(defect/max(denom,np.finfo(float).tiny))
    x = model.apply_lower(model.apply_lower(a,inverse=False,transpose=True),inverse=True,transpose=True)
    factor_roundtrip = relative(model.pack(x-a),model.pack(a))
    trials = trial_set(model,a,b,h,settings,gates)
    passed = skew <= settings["preconditioned_skew_gate"] and all(trials[k]["passed"] for k in ("screen","screen_lift"))
    return dict(job=job,status="complete",preconditioned_skew_pairing_relative=skew,
                factor_roundtrip_relative=factor_roundtrip,trials=trials,screen_pass=bool(passed))


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing balanced audit")
    settings,left,report = (read(path) for path in (TEMPLATE,LEFT,LEFT_REPORT))
    if report["protocol_sha256"] != digest(LEFT) or not all(r.get("algebra_pass",False) for r in report["rows"]):
        raise RuntimeError("The original frozen operator algebra must pass before comparing preconditioners")
    if any(settings[k] != left["settings"][k] for k in ("strict_rtol","screen_rtol")):
        raise ValueError("A preconditioner comparison cannot change tolerances")
    OUT.mkdir()
    inputs = (Path(__file__),TEMPLATE,LEFT,LEFT_REPORT,HERE/"lamb_symmetric.py",HERE/"lamb_linearized.py",HERE/"audit_lamb_linearized.py")
    fingerprints = {p.relative_to(HERE).as_posix():digest(p) for p in inputs}
    added = [dict(job,rate=64) for job in left["jobs"] if job["size"] == "mid" and job["rate"] == 128]
    jobs = left["jobs"]+added
    save(OUT/"protocol.json",dict(status="frozen",settings=settings,source_sha256=fingerprints,jobs=jobs,
                                  fixed_gates=left["settings"]["fixed_gates"]))
    rng = np.random.default_rng(left["settings"]["state_seed"])
    records,rows = {},[]
    for job in jobs:
        try:
            model = SymmetricLambGalerkin(dt=1/job["rate"],**job["config"])
            a = model.unpack(.01*rng.standard_normal(model.real_size))
            b = model.unpack(.01*rng.standard_normal(model.real_size))
            row = audit(job,a,b,model,settings,left["settings"]["fixed_gates"])
            if job in added:
                standard = LambGalerkin(dt=1/job["rate"],**job["config"])
                row["same_state_left_trials"] = trial_set(standard,a,b,1/job["rate"],settings,left["settings"]["fixed_gates"])
        except (ArithmeticError,RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            row = dict(job=job,status="failed",screen_pass=False,failure_type=type(exc).__name__,failure=str(exc))
        artifact = OUT/("_".join(str(job[k]) for k in ("family","size","rate"))+".json")
        save(artifact,row)
        records[artifact.name] = digest(artifact)
        rows.append(row)
        print(artifact.stem,row["screen_pass"],row.get("trials",{}).get("strict",{}).get("passed"),flush=True)
    if fingerprints != {p.relative_to(HERE).as_posix():digest(p) for p in inputs}:
        raise RuntimeError("Audit inputs changed")
    screen = sum(r["screen_pass"] for r in rows)
    strict = sum(r.get("trials",{}).get("strict",{}).get("passed",False) for r in rows)
    practical = [r for r in rows if r["job"]["size"] == "mid" and r["job"]["rate"] in settings["practical_rates"]]
    passed = len(practical) == settings["required_practical_cases"] and all(r["screen_pass"] for r in practical)
    save(OUT/"summary.json",dict(status="passed" if passed else "failed",all_fixed_cases_screen_passed=screen==len(rows),
         practical_passed=sum(r["screen_pass"] for r in practical),screen_passed=screen,strict_passed=strict,rows=rows,
         raw_records=records,protocol_sha256=digest(OUT/"protocol.json")))


if __name__ == "__main__":
    main()
