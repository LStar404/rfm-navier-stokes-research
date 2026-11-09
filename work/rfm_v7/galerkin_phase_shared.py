"""One shared Galerkin spatial assembly for the three CUDA SBDF2 phases."""

from basis_recurrence_model import RecurrenceGalerkin
from cpu_sbdf2_switch import bridge_weights
from full_galerkin import inverse_spd
from galerkin_cuda_native import case, np, operators
from radial_layers import radial_quadrature


def spatial_blocks(model):
    """Recover the step-independent matrices in the frozen assembly order."""
    # The frozen constructor exposes the transformed basis, but keeps the
    # weights and diffusion Gram blocks local. Rebuild them once for both phases.
    s, w = radial_quadrature(model.metadata["q"], model.metadata["split"])
    if not np.array_equal(s, model.s):
        raise ArithmeticError("The shared quadrature differs from the startup model")
    sw = np.sqrt(w)
    f, _, d2 = model.samples
    lf, _, ld2 = model.lift
    mass = model.M0+model.k2[:, None, None]*model.M1
    l1 = (d2-f)*(sw/model.r)[:, None]
    fw = f*(sw/model.r)[:, None]
    cross = l1.T@fw
    delta = (model.k2-1)[:, None, None]
    diffusion = case.NU*((l1.T@l1)-delta*(cross+cross.T)+delta**2*(fw.T@fw))
    lift_diffusion = case.NU*l1.T@((ld2-lf)*sw/model.r)
    mv, md = model.mean_samples
    strain = (mv+md)*sw[:, None]
    return mass, diffusion, lift_diffusion, strain.T@strain


def pack_step(model, blocks, dt):
    """Rebuild just the inverse and the five step-dependent device operators."""
    mass, diffusion, lift_diffusion, mean_strain = blocks
    inverse = np.stack([inverse_spd(a) for a in 1.5*mass+dt*diffusion])
    mean_inverse = inverse_spd(1.5*model.mean_mass+dt*case.NU*mean_strain)
    lift = 1.5*model.M_lift+dt*lift_diffusion
    width, modes, q = model.width, model.k+1, len(model.r)
    mean_width = model.mean_width
    linear = np.zeros((width, width, modes))
    radial = np.zeros((width, q, modes))
    angular = np.zeros_like(radial)
    linear[..., 1:] = (inverse @ mass).transpose(1, 2, 0)
    linear[:mean_width, :mean_width, 0] = mean_inverse @ model.mean_mass
    scale = dt/model.kernel.n
    radial[..., 1:] = scale*(inverse @ model.test_s).transpose(1, 2, 0)
    radial[:mean_width, :, 0] = -scale*(mean_inverse @ model.mean_test)
    angular[..., 1:] = scale*model.modes*(inverse @ model.test).transpose(1, 2, 0)
    return linear, radial, angular, inverse[0] @ model.M_lift, inverse[0] @ lift


def phase_operators_shared(startup_rate, late_rate, config):
    """Return the frozen phase interface using one recurrence basis and one observation pack.

    A_eff = (3/2) M + s*h_new K = s A_bridge for s=(3/2)/a0.
    Multiply the bridge linear and history mass-lift packs by s. The flux
    already includes s*h_new and the future lift already uses A_eff.
    """
    old_dt, new_dt = 1/startup_rate, 1/late_rate
    a0, b0, b1, e0, e1 = bridge_weights(new_dt/old_dt)
    coefficients = b0, -b1, e0, -e1
    old = RecurrenceGalerkin(dt=old_dt, **config)
    initial = operators(old, old_dt)
    if old_dt == new_dt:
        return old, initial[5:], (initial[:5],)*3, coefficients
    blocks = spatial_blocks(old)
    scale = 1.5/a0
    linear, radial, angular, mass_lift, bdf_lift = pack_step(old, blocks, scale*new_dt)
    bridge = scale*linear, radial, angular, scale*mass_lift, bdf_lift
    return old, initial[5:], (initial[:5], bridge, pack_step(old, blocks, new_dt)), coefficients


def compare_arrays(previous, candidate):
    if previous.shape != candidate.shape or previous.dtype != candidate.dtype:
        return dict(bitwise=False, previous_shape=list(previous.shape), candidate_shape=list(candidate.shape),
                    previous_dtype=str(previous.dtype), candidate_dtype=str(candidate.dtype))
    difference = previous-candidate
    return dict(bitwise=previous.tobytes(order="C") == candidate.tobytes(order="C"),
                max_abs=float(np.max(np.abs(difference))),
                relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(previous), np.finfo(float).tiny)))


def fixed_audit():
    """Compare every returned array on the archived medium and high configurations."""
    from pathlib import Path
    import time

    import cpu_formal as base
    from galerkin_cuda_variable import phase_operators

    here = Path(__file__).resolve().parent
    medium = here/"switch_qualification_output"/"protocol.json"
    high = here/"multiscale_switch_qualification_output"/"protocol.json"
    sources = {str(path.relative_to(here).as_posix()):base.digest(path) for path in (medium, high)}
    model_fields = ("s", "r", "transform", "samples", "mean_transform", "mean_samples", "M0", "M1",
                    "mean_mass", "M_lift", "bdf_lift", "start", "implicit", "mean_inverse", "test", "test_s",
                    "mean_test", "time_jets", "mean_vort_samples", "lift_modes")
    phase_fields = ("linear", "radial", "angular", "mass_lift", "bdf_lift")
    observer_fields = ("mean", "probe", "probe_lift")
    groups = (("medium", base.read(medium), ("rfm37_k72", "leg37_k72"), 128, 112),
              ("high", base.read(high), ("rfm41_k80", "leg39_k80"), 152, 136))
    rows = []
    for scale, frozen, names, startup, late in groups:
        for name in names:
            protocol = frozen["solve_protocol"]
            seeds = protocol["seeds"] if name.startswith("rfm") else [protocol["deterministic_seed"]]
            for seed in seeds:
                _, config = base.config(frozen["specs"][name], protocol, seed)
                for after in (startup, late):
                    row = dict(scale=scale, name=name, seed=seed, startup_rate=startup, late_rate=after)
                    try:
                        start = time.perf_counter()
                        previous = phase_operators(startup, after, config)
                        previous_seconds = time.perf_counter()-start
                        start = time.perf_counter()
                        candidate = phase_operators_shared(startup, after, config)
                        candidate_seconds = time.perf_counter()-start
                        a, b = previous[0], candidate[0]
                        fields = {f"model.{key}":compare_arrays(getattr(a,key),getattr(b,key)) for key in model_fields}
                        fields.update({f"observe.{key}":compare_arrays(x,y) for key,x,y in
                                       zip(observer_fields,previous[1],candidate[1],strict=True)})
                        for phase, x, y in zip(("startup", "bridge", "late"), previous[2], candidate[2], strict=True):
                            fields.update({f"{phase}.{key}":compare_arrays(p,q) for key,p,q in zip(phase_fields,x,y,strict=True)})
                        fields["coefficients"] = compare_arrays(np.asarray(previous[3]),np.asarray(candidate[3]))
                        alias_ok = after != startup or candidate[2][0] is candidate[2][1] is candidate[2][2]
                        passed = a.metadata == b.metadata and alias_ok and all(item["bitwise"] for item in fields.values())
                        row.update(status="passed" if passed else "failed", fields=fields,
                                   metadata_exact=a.metadata == b.metadata, same_rate_alias_preserved=alias_ok,
                                   diagnostic_seconds=dict(original=previous_seconds, shared=candidate_seconds))
                    except (ArithmeticError, RuntimeError, ValueError, OSError, np.linalg.LinAlgError) as exc:
                        row.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
                    rows.append(row)
                    print(scale, name, seed, startup, after, row["status"], flush=True)
    return dict(status="passed" if all(r["status"] == "passed" for r in rows) else "completed_with_failures",
                source_sha256=base.digest(Path(__file__)), config_sha256=sources, rows=rows,
                scope="Fixed FP64 host operator identity. Individual setup clocks are non-isolated diagnostics. "
                      "No CUDA runtime, trajectory, or formal timing cohort was run.")


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = fixed_audit()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write("\n")
    raise SystemExit(0 if report["status"] == "passed" else 1)
