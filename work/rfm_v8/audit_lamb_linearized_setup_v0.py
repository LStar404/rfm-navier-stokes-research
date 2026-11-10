"""Fixed algebra audit, including arbitrary frozen curl and homogeneous energy."""
import hashlib
import json
from pathlib import Path
import time
from lamb_linearized import LambGalerkin, np, case
from full_galerkin import real_apply
from radial_layers import radial_quadrature

HERE = Path(__file__).resolve().parent
OUT = HERE / "lamb_linearized_audit.json"


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


def main():
    if OUT.exists():
        raise RuntimeError("Preserve existing audit")
    source = HERE / "lamb_linearized.py"
    rows = []
    rng = np.random.default_rng(374783)
    for family in ("rfm", "legendre"):
        for size in ("small", "mid"):
            config = dict(m=12 if size=="small" else 37,k=8 if size=="small" else 72,
                          q=36 if size=="small" else 66,scale=62.,mean_order=10 if size=="small" else 24,
                          layers=[] if size=="small" else [16,36,80,165,340,660],split=True,
                          backend="fft",seed=7331 if family=="rfm" else 17,family=family)
            for h in (1/128,1/32,1/8):
                model = LambGalerkin(dt=h,**config)
                a = model.unpack(.01*rng.standard_normal(model.real_size))
                b = model.unpack(.01*rng.standard_normal(model.real_size))
                omega = model.fields_on(a,.3+.2j,model.samples,model.lift,model.r,model.mean_samples)[2].copy()
                model.freeze_vorticity(omega)
                fa,fb = model.flux_frozen(a),model.flux_frozen(b)
                cross = abs(pair(model,a,fb)+pair(model,b,fa))
                cross_scale = np.linalg.norm(model.pack(a))*np.linalg.norm(model.pack(fb)) + np.linalg.norm(model.pack(b))*np.linalg.norm(model.pack(fa))
                skew = cross/max(cross_scale,np.finfo(float).tiny)
                same = model.flux_frozen(a,.3+.2j)
                expected = model.terms(a,.3+.2j)[1]
                matching = np.linalg.norm(model.pack(same-expected))/max(np.linalg.norm(model.pack(expected)),np.finfo(float).tiny)
                # Two calls must preserve the frozen physical curl after the
                # source reconstruction's reusable buffer is overwritten.
                frozen_copy = model.frozen_omega.copy()
                model.fields_on(b,0.,model.samples,model.lift,model.r,model.mean_samples)
                frozen_owned = np.array_equal(frozen_copy,model.frozen_omega)
                begun=time.perf_counter()
                newer, work = model.implicit_step(a,b,(0.,0.,0.),h,rtol=1e-12)
                elapsed=time.perf_counter()-begun
                matrices = diffusion(model)
                e_old = norm_mass(model,a)+norm_mass(model,2*a-b)
                e_new = norm_mass(model,newer)+norm_mass(model,2*newer-a)
                diss = norm_mass(model,newer-2*a+b)+4*h*pair(model,newer,apply_diffusion(model,matrices,newer))
                energy_defect = abs(e_new-e_old+diss)/max(e_old,np.finfo(float).tiny)
                rhs = 2*model.mass_only(a)-.5*model.mass_only(b)
                lhs = 1.5*model.mass_only(newer)+h*apply_diffusion(model,matrices,newer)-h*model.flux_frozen(newer)
                equation = np.linalg.norm(model.pack(lhs-rhs))/max(np.linalg.norm(model.pack(rhs)),np.finfo(float).tiny)
                row=dict(family=family,size=size,h=h,config=config,skew_pairing_relative=skew,
                         original_flux_relative=matching,frozen_curl_owned=frozen_owned,
                         homogeneous_energy_defect=energy_defect,energy_ratio=e_new/e_old,
                         original_equation_relative=equation,linear_work=work,diagnostic_step_seconds=elapsed)
                if not (skew<1e-12 and matching<1e-12 and frozen_owned and energy_defect<1e-7 and e_new/e_old<1+1e-7 and equation<1e-7):
                    raise ArithmeticError(str(row))
                rows.append(row)
    OUT.write_text(json.dumps(dict(status="passed",scope="Fixed algebra and homogeneous one-step tests only",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows),indent=2)+"\n",encoding="utf8")
    print("passed",len(rows),max(r["skew_pairing_relative"] for r in rows),max(r["homogeneous_energy_defect"] for r in rows),flush=True)


if __name__ == "__main__":
    main()
