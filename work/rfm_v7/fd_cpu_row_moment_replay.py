"""Freeze one CPU k=1 row-moment trajectory and compare to the original pow2 reference."""

import argparse
from decimal import localcontext
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

from fd_cpu_row_moment import CLASS, LIBRARY, VARIANT, Pow2RowMomentFD, compile_library, np
from fd_cuda_pow2_replay import PROTOCOL as BASE_PROTOCOL, reference
from fd_equilibration_precision import decimal_mv, decimal_rows, decimal_vector
import fd_pow2_long as original
from fd_row_compensated_audit import MomentRows, exact_curl, metrics
from scipy.fft import irfft, rfft


HERE = Path(__file__).resolve().parent
OUT = HERE/"cpu_fd_row_moment_output"
PROTOCOL = OUT/"protocol.json"
AUDIT = OUT/"fixed_audit.json"
TARGET = OUT/"fd448_k144_rate512_row_moment.npz"
REPORT = TARGET.with_suffix(".json")
base, case = original.base, original.case
SEED = 7692501


def exact_first_mode(row_moments, source, taylor):
    """Independent NumPy real-component evaluation in original CSR order."""
    components = np.column_stack((source.real, source.imag))
    if taylor:
        slopes = (components[2:]-components[:-2])/row_moments.spans[:, None]
    output = np.empty((len(row_moments.rows), 2))
    for i, (indices, weights, delta, mu0, mu1) in enumerate(row_moments.rows):
        center = components[i+1]
        residue = components[indices]-center
        correction = float(mu0)*center
        if taylor:
            residue -= delta[:, None]*slopes[i]
            correction += float(mu1)*slopes[i]
        acc = np.zeros(2)
        for a, d in zip(weights, residue):
            acc += a*d
        output[i] = acc+correction
    return output[:, 0]+1j*output[:, 1]


def verify_radial(model, previous, state, speed, rows1, rows2):
    old_mass, old_flux = previous.terms(state, speed)
    mass, flux = model.terms(state, speed)
    controls = np.arange(state.shape[0]) != 1
    if not (np.array_equal(model._psi, previous._psi) and
            np.array_equal(model._omega[[0, -1]], previous._omega[[0, -1]]) and
            np.array_equal(model._omega[:, controls], previous._omega[:, controls]) and
            np.array_equal(model._spectrum[:2], previous._spectrum[:2]) and
            np.array_equal(model._spectrum[2:, :, :state.shape[0]][:, :, controls],
                           previous._spectrum[2:, :, :state.shape[0]][:, :, controls]) and
            np.array_equal(model._spectrum[:, :, state.shape[0]:], previous._spectrum[:, :, state.shape[0]:]) and
            np.array_equal(mass[controls], old_mass[controls]) and
            np.array_equal(mass[0], state[0])):
        raise ArithmeticError("Unchanged boundary, velocity, modes, FFT padding or mass differs")
    omega = previous._omega[:, 1].copy()
    d2 = exact_first_mode(rows2, model._psi[:, 1], True)
    omega[1:-1] = -(d2-model._psi[1:-1, 1])*model.inv_r[1:-1]**2
    ds = exact_first_mode(rows1, omega, False)
    if not (np.array_equal(omega, model._omega[:, 1]) and
            np.array_equal(ds, model._spectrum[2, :, 1]) and
            np.array_equal(1j*omega[1:-1], model._spectrum[3, :, 1]) and
            np.array_equal(omega[1:-1], mass[1])):
        raise ArithmeticError("Native k=1 row arithmetic differs from the component reference")
    ur, curl = model._spectrum[0, :, 1:state.shape[0]], model._omega[1:-1, 1:]
    parseval = np.zeros(len(ur))
    for k in range(ur.shape[1]):
        parseval += ur[:, k].real*curl[:, k].real + ur[:, k].imag*curl[:, k].imag
    mean = -2.*parseval
    if not (np.array_equal(mean, model._mean) and np.array_equal(mean, flux[0])):
        raise ArithmeticError("Mean Parseval reduction or zero-mode flux differs")
    ur, ut, radial, azimuthal = irfft(model._spectrum, n=model.plan.n, norm="forward", workers=1)
    expected_flux = rfft(ur*radial+ut*azimuthal, axis=-1, norm="forward", workers=1)[:, :state.shape[0]]
    expected_flux *= -model.inv_r[1:-1, None]
    expected_flux[:, 0] = mean
    if not np.array_equal(expected_flux.T, flux):
        raise ArithmeticError("Four-plane FFT and physical products differ")
    return (mass, flux), dict(omega_max_abs=float(np.max(np.abs(model._omega-previous._omega))),
                             mean_max_abs=float(np.max(np.abs(mean-old_flux[0]))),
                             flux_max_abs=float(np.max(np.abs(flux-old_flux))))


def fixed_audit(truth, protocol):
    model = Pow2RowMomentFD(dt=truth.row["dt"], **protocol["config"])
    previous = truth.model
    if model.scale_metadata() != previous.scale_metadata():
        raise ArithmeticError("The original pow2 factors changed")
    report = []
    with localcontext() as context:
        context.prec = 160
        rows1, rows2 = (MomentRows(matrix, model.s) for matrix in (model.d1, model.d2))
    for key, expected in (
        ("d1_mu0", [float(row[3]) for row in rows1.rows]),
        ("d2_mu0", [float(row[3]) for row in rows2.rows]),
        ("d2_mu1", [float(row[4]) for row in rows2.rows]),
    ):
        if not np.array_equal(model._moments[key], expected):
            raise ArithmeticError(f"Exact binary rational {key} differs from the 160-digit check")
    with localcontext() as context:
        context.prec = 65
        d1, d2, wall = map(decimal_rows, (model.d1, model.d2, model.wall_d2[0]))
        earlier = np.asarray([float(row[4]) for row in MomentRows(model.d2, model.s).rows])
        exact = model._moments["d2_mu1"]
        changed = np.flatnonzero(earlier != exact)
        moment_rounding = [dict(row=int(i), decimal65=float(earlier[i]), exact_rational=float(exact[i]))
                           for i in changed]
        rng = np.random.default_rng(SEED)
        cases = [(f"t{t:g}", truth.states[np.flatnonzero(truth.times == t)[0]], case.speed(t), True)
                 for t in protocol["fixed_times"]]
        for i in range(4):
            state = rng.normal(size=model.rhs_shape)+1j*rng.normal(size=model.rhs_shape)
            if i % 2 == 0:
                state[0] = state[0].real
            cases.append((f"random{i}", state, .31+(.21*i-.43)*1j, False))
        retained = []
        for label, state, speed, dynamic in cases:
            (mass, flux), changes = verify_radial(model, previous, state, speed, rows1, rows2)
            retained.append((mass, flux, mass.copy(), flux.copy()))
            psi = decimal_vector(model._psi[:, 1])
            expected = decimal_mv(d1, exact_curl(model, d1, d2, wall, psi, 1))
            staged = metrics(previous._spectrum[2, :, 1], expected)
            candidate = metrics(model._spectrum[2, :, 1], expected)
            if dynamic and any(candidate[key] > min(1e-8, staged[key]/100)
                               for key in ("relative_l2", "outer8_relative_l2")):
                raise ArithmeticError((label, "fixed-state Taylor improvement changed", staged, candidate))
            report.append(dict(state=label, all_modes_checked=True, boundary_unchanged=True,
                               mean_reduction_exact=True, change=changes,
                               staged_gradient=staged, row_moment_gradient=candidate))
        if not all(np.array_equal(a, c) and np.array_equal(b, d) for a, b, c, d in retained):
            raise ArithmeticError("A subsequent terms call overwrote retained SBDF2 history")
    return dict(status="pass", rows=report, moment_tables=model.metadata["moment_tables"],
                unchanged_row_scale=model.scale_metadata(), history_independent=True,
                moments_verified_decimal_digits=160, dynamic_oracle_decimal_digits=65,
                decimal65_d2_mu1_rounding_differences=moment_rounding)


def fingerprint():
    names = ("fd_cpu_row_moment.py", "fd_cpu_row_moment.c", LIBRARY.name, Path(__file__).name,
             "fd_row_compensated_audit.py", "fd_row_compensated_audit.json",
             "fd_cuda_pow2_replay.py", BASE_PROTOCOL.name)
    return original.source_manifest() | {f"rfm_v7/{name}": base.digest(HERE/name) for name in names}


def frozen_config():
    prior = base.read(BASE_PROTOCOL)
    if tuple(prior["snapshots"]) != case.SNAPSHOTS or prior["probe_spacing"] != .125:
        raise RuntimeError("The reused marcher's fixed absolute times do not match the prior reference")
    return dict(
        status="frozen", variant=VARIANT, reconstruction_class=CLASS, seed=SEED,
        config=dict(prior["config"], linear="native"), rate=int(prior["steps"]/prior["horizon"]),
        fixed_times=prior["times_for_audit"], snapshots=prior["snapshots"],
        steps=prior["steps"], horizon=prior["horizon"], probe_spacing=prior["probe_spacing"],
        reference=prior["reference"], reference_sha256=prior["reference_sha256"],
        original_cpu_replay_relative_gate=prior["relative_gate"],
        original_cpu_replay_crossing_time_gate=prior["crossing_time_gate"],
        source_hashes=fingerprint(),
        selection="Only k=1. Before the trajectory, fixed-state audit found a large k=1 error "
                  "and small errors with some regressions in the other modes.",
        scope="Same stored operators, boundaries, physical probes, timestep and FFT. "
              "One concurrent CPU trajectory and no timing ranking or reference promotion.",
    )


def comparison(sol, ref, protocol):
    for key in ("times", "probe_times"):
        if not np.array_equal(getattr(sol, key), getattr(ref, key)):
            raise ArithmeticError(f"Absolute {key} schedule changed")
    if sol.row["row_scale"] != ref.row["row_scale"]:
        raise ArithmeticError("Trajectory used different row scales")
    fields = original.compare(sol, ref)
    probes = {}
    for name in ("transverse", "vorticity"):
        a, b = getattr(sol, name), getattr(ref, name)
        error, norm = np.linalg.norm(a-b, axis=0), np.linalg.norm(b, axis=0)
        if np.any(norm <= 1e-12):
            raise ArithmeticError("Insufficient probe scale for relative error")
        probes[name] = dict(relative_l2=(error/norm).tolist(), max_abs=np.max(np.abs(a-b), axis=0).tolist())
    states = []
    for t, a, b in zip(sol.times, sol.states, ref.states):
        states.append(dict(time=float(t), max_abs=float(np.max(np.abs(a-b))),
                           relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                           k1_relative_l2=float(np.linalg.norm(a[1]-b[1])/np.linalg.norm(b[1]))))
    new_wake, old_wake = base.wake(sol), base.wake(ref)
    crossings = {}
    for name in ("velocity", "vorticity"):
        crossings[name] = []
        for new, old in zip(new_wake[name], old_wake[name]):
            a, b = map(np.asarray, (new["positive_crossing_times"], old["positive_crossing_times"]))
            crossings[name].append(dict(updated=a.tolist(), original=b.tolist(),
                                        differences=(a-b).tolist() if len(a) == len(b) > 1 else None))
    same_trajectory_gate = dict(
        fields=max(fields["worst"].values()) <= protocol["original_cpu_replay_relative_gate"],
        probes=max(x for p in probes.values() for x in p["relative_l2"]) <= protocol["original_cpu_replay_relative_gate"],
        crossings=all(r["differences"] is not None and
                      max(map(abs, r["differences"])) <= protocol["original_cpu_replay_crossing_time_gate"]
                      for rows in crossings.values() for r in rows),
    )
    return dict(fields=fields, probes=probes, states=states, updated_wake=new_wake, original_wake=old_wake,
                paired_crossings=crossings, same_trajectory_gate=same_trajectory_gate,
                note="Absolute times and shared original postprocessing. Differences against the old trajectory "
                     "are a sensitivity measure, not an accuracy estimate against an exact solution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "audit", "run"))
    parser.add_argument("--cc", default="C:/mingw64/bin/gcc.exe" if sys.platform == "win32" else "cc")
    args = parser.parse_args()
    if args.action == "build":
        compile_library(args.cc)
        return
    protocol = frozen_config()
    ref = reference(base.read(BASE_PROTOCOL))
    if args.action == "audit":
        if AUDIT.exists():
            raise FileExistsError(AUDIT)
        audit = fixed_audit(ref, protocol)
        base.save(AUDIT, dict(audit, frozen=protocol))
        print("FIXED AUDIT PASS", len(audit["rows"]), flush=True)
        return
    audit = base.read(AUDIT)
    if audit["status"] != "pass" or audit["frozen"] != protocol:
        raise RuntimeError("Fixed audit or frozen source changed")
    if PROTOCOL.exists():
        if base.read(PROTOCOL) != protocol:
            raise RuntimeError("Run protocol changed")
    else:
        base.save(PROTOCOL, protocol)
    if REPORT.exists() or TARGET.exists():
        raise FileExistsError(TARGET)
    print("RUN", VARIANT, "PID", os.getpid(), "STEPS", protocol["steps"], flush=True)
    try:
        config = dict(protocol["config"])
        n = config.pop("n")
        with patch.multiple(original, OUT=OUT, RATE=protocol["rate"], HORIZON=protocol["horizon"],
                            STEPS=protocol["steps"], GRID=config, CLASS=CLASS,
                            Pow2EquilibratedBufferedFD=Pow2RowMomentFD):
            solution, config = original.march(n)
        if solution.row["moment_tables"] != audit["moment_tables"]:
            raise RuntimeError("Trajectory moment tables differ from the audited ones")
        if fingerprint() != protocol["source_hashes"]:
            raise RuntimeError("Source changed during the trajectory")
        temp = TARGET.with_suffix(".tmp.npz")
        case.store_solution(temp, solution, config)
        temp.replace(TARGET)
        results = comparison(solution, ref, protocol)
        report = dict(status="complete", artifact_sha256=base.digest(TARGET), metadata=solution.row,
                      protocol_sha256=base.digest(PROTOCOL), audit_sha256=base.digest(AUDIT), **results)
        base.save(REPORT, report)
        print("COMPLETE", solution.row["seconds"], results["fields"]["worst"], flush=True)
    except Exception:
        base.save(OUT/"last_failure.json", dict(pid=os.getpid(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
