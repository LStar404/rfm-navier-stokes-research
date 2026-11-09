"""Reconstruct startup differences on a common basis from the twelve saved trajectories."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fd_cuda_pow2_replay import write_json
from startup_replay_audit import ARMS, OUT, base, case, check, np, RecurrenceGalerkin, NativePrecontractedGalerkin


HERE = Path(__file__).resolve().parent
HIGH_K = 60


def open_case(directory, row, model=None):
    path = directory/row["artifact"]
    check(path.parent == directory and base.digest(path) == row["sha256"], f"Startup trajectory changed: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        config = json.loads(str(archive["config"]))
        info = json.loads(str(archive["metadata"]))
        check(info == row["metadata"], f"Startup trajectory metadata changed: {path.name}")
        if model is None:
            model = RecurrenceGalerkin(dt=info["dt"], **config)
        sol = case.Solution(model, *(archive[key] for key in ("states", "times", "probe_times", "transverse", "vorticity")),
                            info)
    check(sol.states.shape[1:] == model.rhs_shape and np.isfinite(sol.states).all(), f"Startup shape differs: {path.name}")
    return sol, config


def fingerprint(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def basis_match(control, native):
    checks = {}
    for key in ("transform", "mean_transform"):
        a, b = getattr(control, key), getattr(native, key)
        check(a.shape == b.shape, f"Different startup basis shape: {key}")
        checks[key] = dict(array_equal=bool(np.array_equal(a, b)), max_abs=float(np.max(np.abs(a-b))),
                           recurrence_sha256=fingerprint(a), native_cpu_sha256=fingerprint(b))
    return checks


def modal_energy(reference, candidate, theta, weights):
    check(len(theta) > 2*(reference.shape[1]-1), "Angular grid cannot support the full Fourier energy")
    coefficients = candidate-reference
    multiplicity = np.full(reference.shape[1], 2.)
    multiplicity[0] = 1.
    squared = np.abs(coefficients)**2
    squared[:, 0] = coefficients[:, 0].real**2
    energy = (weights @ squared)*multiplicity*len(theta)
    total = float(energy.sum())
    real_energy = float(np.sum((weights @ (coefficients.real**2))*multiplicity)*len(theta))
    reference_squared = np.abs(reference)**2
    reference_squared[:, 0] = reference[:, 0].real**2
    reference_modal_energy = float(np.sum((weights @ reference_squared)*multiplicity)*len(theta))
    fields = case.reconstruct(candidate, theta), case.reconstruct(reference, theta)
    direct_norm = float(np.sqrt(weights@np.sum((fields[0]-fields[1])**2, axis=1)))
    reference_norm = float(np.sqrt(weights@np.sum(fields[1]**2, axis=1)))
    check(reference_norm > 0 and np.isfinite([total, reference_norm, direct_norm]).all(),
          "Invalid reconstructed startup norms")
    top = sorted(range(len(energy)), key=lambda k: energy[k], reverse=True)[:5]
    return dict(difference_weighted_l2=direct_norm, reference_weighted_l2=reference_norm,
                relative_l2=direct_norm/reference_norm, modal_difference_l2=float(np.sqrt(total)),
                modal_reference_l2=float(np.sqrt(reference_modal_energy)), modal_energy=energy.tolist(),
                high_modes_energy_fraction=float(energy[HIGH_K:].sum()/total) if total else None,
                real_energy_fraction=real_energy/total if total else None,
                top_modes=[dict(k=k, fraction=float(energy[k]/total)) for k in top] if total else [])


def probe_projection(control, candidate, i):
    reference = control.modal(i, case.PROBE_S)[2]
    actual = candidate.modal(i, case.PROBE_S)[2]
    delta = actual-reference
    physical = case.reconstruct(delta, np.zeros(1))[:, 0]
    real = np.abs(delta[:, 0].real)+2*np.sum(np.abs(delta[:, 1:].real), axis=1)
    envelope = np.abs(delta[:, 0])+2*np.sum(np.abs(delta[:, 1:]), axis=1)
    absolute = np.abs(physical)
    t = float(control.times[i])
    index, = np.flatnonzero(control.probe_times == t)
    archived = candidate.vorticity[index]-control.vorticity[index]
    return dict(time=t, reconstructed_delta=physical.tolist(), archived_delta=archived.tolist(),
                archived_reconstruction_max_abs_difference=float(np.max(np.abs(archived-physical))),
                all_coefficients_envelope=envelope.tolist(), real_projection_envelope=real.tolist(),
                coherent_over_real_envelope=[float(x/y) if y else None for x, y in zip(absolute, real)],
                coherent_over_all_coefficients=[float(x/y) if y else None for x, y in zip(absolute, envelope)])


def run(directory):
    journal_path = directory/"audit.json"
    protocol_path = directory/"protocol.json"
    audit = base.read(journal_path)
    check(audit["status"] == "complete" and audit["protocol_sha256"] == base.digest(protocol_path),
          "A complete frozen startup replay is required")
    rows = {(r["name"], r["rate"], r["arm"]): r for r in audit["trajectories"]}
    check(len(rows) == len(audit["trajectories"]) == 12, "Expected twelve unique startup trajectories")
    archived_comparisons = {(r["name"], r["rate"], r["arm"]): r for r in audit["comparisons"]}
    inputs = {name: base.digest(path) for name, path in (("startup_audit", journal_path), ("startup_protocol", protocol_path),
              ("startup_runner", HERE/"startup_replay_audit.py"), ("modal_audit_runner", Path(__file__).resolve()))}
    source_keys = ("rfm_v7/startup_replay_audit.py", "rfm_v7/basis_recurrence_model.py",
                   "rfm_v7/cpu_precontracted_native.py", "rfm_v7/galerkin_cuda_native.py", "rfm_v6/full_galerkin.py")
    protocol = base.read(protocol_path)
    source_hashes = {name: protocol["source_hashes"][name] for name in source_keys}
    for name, digest in source_hashes.items():
        check(base.digest(base.WORK/name) == digest, f"Startup reconstruction source changed: {name}")
    grids = base.grid_set()
    outcomes, basis = [], []
    for name, rate in dict.fromkeys((name, rate) for name, rate, _ in rows):
        baseline, settings = open_case(directory, rows[name, rate, ARMS[0]])
        native = NativePrecontractedGalerkin(dt=baseline.row["dt"], **settings)
        identity = basis_match(baseline.model, native)
        common = all(record["array_equal"] for record in identity.values())
        basis.append(dict(name=name, rate=rate, common_model_for_all_arms=common, transforms=identity,
                          cuda_basis="Frozen CUDA native_model uses RecurrenceGalerkin; saved coefficients use that basis."))
        for arm in ARMS[1:]:
            model = baseline.model if common or arm == "cuda_graph" else native
            candidate, config = open_case(directory, rows[name, rate, arm], model)
            check(settings == config and candidate.row["dt"] == baseline.row["dt"] and
                  np.array_equal(candidate.times, baseline.times) and np.array_equal(candidate.probe_times, baseline.probe_times),
                  "Startup reconstruction coordinates or absolute times differ")
            recorded = archived_comparisons[name, rate, arm]
            scored = {(row["grid"], row["time"]): row for row in recorded["fields"]}
            comparisons = []
            for grid, (s, theta, weights) in grids.items():
                near = weights*(case.radius(s) <= 2.5)
                for index, time in enumerate(baseline.times):
                    expected = baseline.modal(index, s)[2]
                    actual = candidate.modal(index, s)[2]
                    full = modal_energy(expected, actual, theta, weights)
                    close = modal_energy(expected, actual, theta, near)
                    raw = scored[grid, time]
                    check(abs(raw["vorticity"]-full["relative_l2"]) <= 1e-12 and
                          abs(raw["near_vorticity"]-close["relative_l2"]) <= 1e-12,
                          f"Common basis and archived vorticity score disagree: {name}/{rate}/{arm} {grid} t={time}")
                    comparisons.append(dict(grid=grid, time=float(time), full=full, near=close))
            probes = [probe_projection(baseline, candidate, i) for i in range(len(baseline.times))]
            outcomes.append(dict(name=name, rate=rate, arm=arm, common_model=model is baseline.model,
                                 comparisons=comparisons, theta_zero_probe_projection=probes))
    return dict(status="complete", inputs=inputs, sources=source_hashes,
                input_trajectories=[{k: r[k] for k in ("name", "rate", "arm", "artifact", "sha256")} for r in rows.values()],
                high_mode_start=HIGH_K, basis=basis, results=outcomes,
                norm="Physical area-weighted radial quadrature and unnormalized angular sample sum. "
                     "Modal energy uses exact discrete Fourier multiplicity on the uniform shifted score grid.",
                probe_scope="At theta=0 only the real parts of the positive modal coefficients contribute. "
                            "Different modes can cancel and the three probe radii do not sample the whole near-field grid.",
                limitation="The error distribution localizes the observed startup difference. "
                           "It does not prove an instability mechanism or a causal attribution.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, default=OUT)
    cli.add_argument("--output", type=Path, default=OUT/"modal_audit.json")
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.directory)
    write_json(args.output, result)
    print("COMPLETE", len(result["input_trajectories"]), len(result["results"]),
          all(row["common_model_for_all_arms"] for row in result["basis"]))
