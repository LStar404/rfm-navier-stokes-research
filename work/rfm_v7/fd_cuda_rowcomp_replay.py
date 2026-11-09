"""Fixed-state, short-march and archived full-trajectory row-moment checks."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from cpu_formal import digest, grid_set, wake
from fd_cpu_row_moment import Pow2RowMomentFD
from fd_cuda_pow2_replay import compare, reference, write_json
from fd_cuda_rowcomp import RowMomentRuntime, case, radial_stages, solve_fd_cuda_rowcomp
from fd_pow2_long import cross_phase


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"protocol_fd_cuda_rowcomp.json"
SOURCE_FILES = ("fd_cuda_rowcomp_replay.py", "fd_cuda_rowcomp.py", "fd_cuda_rowcomp.cu",
                "fd_cuda_pow2.py", "fd_cuda_pow2.cu", "fd_cuda_native.py", "fd_cuda_native.cu",
                "fd_cpu_row_moment.py", "fd_cpu_row_moment.c", "fd_cuda_pow2_replay.py")


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def old_reference(protocol):
    return reference(json.loads((ROOT/protocol["prior_cpu_protocol"]).read_text(encoding="utf-8")))


def error(actual, expected):
    diff = actual-expected
    norm = float(np.linalg.norm(expected))
    return dict(max_abs=float(np.max(np.abs(diff))), relative_l2=float(np.linalg.norm(diff))/norm if norm else None)


def fixed_state_checks(protocol, old):
    config = protocol["config"]
    cpu = Pow2RowMomentFD(dt=old.row["dt"], linear="native", **config)
    modes, w = cpu.rhs_shape
    half = cpu.plan.n//2+1
    states = np.zeros((2, w, modes), dtype=np.complex128)
    amps = np.zeros(1, dtype=np.complex128)
    mass = np.zeros_like(states)
    psi = np.zeros((w+2, modes), dtype=np.complex128)
    omega = np.zeros_like(psi)
    terms = np.zeros((w, modes))
    spectrum = np.zeros((4, w, half), dtype=np.complex128)
    rows = []
    with ExitStack() as resources:
        rt = resources.enter_context(RowMomentRuntime())
        arrays = (states, amps, np.zeros(1, dtype=np.int32), mass, terms, spectrum)
        state, amplitude, index, mass_gpu, terms_gpu, spectrum_gpu = [rt.upload(resources, a) for a in arrays]
        stages, (psi_gpu, omega_gpu) = radial_stages(rt, resources, cpu, state, amplitude, index,
                                                       mass_gpu, terms_gpu, spectrum_gpu)
        for time in protocol["fixed_times"]:
            selected, = np.flatnonzero(old.times == time)
            modal = old.states[selected]
            amps[0] = case.speed(time)
            states[0] = modal.T
            cpu._fill_radial(modal, amps[0])
            rt.driver.upload(state, states)
            rt.driver.upload(amplitude, amps)
            for stage in stages:
                stage()
            rt.sync()
            for address, target in ((psi_gpu, psi), (omega_gpu, omega), (mass_gpu, mass),
                                    (terms_gpu, terms), (spectrum_gpu, spectrum)):
                rt.driver.download(address, target)
            cpu_u = cpu._spectrum[0, :, 1]
            cpu_w = cpu._omega[1:-1, 1]
            expected_mean_k1 = -2.*(cpu_u.real*cpu_w.real + cpu_u.imag*cpu_w.imag)
            measurements = {"psi_k1": (psi[:, 1], cpu._psi[:, 1]), "omega_k1": (omega[:, 1], cpu._omega[:, 1]),
                            "mass_k1": (mass[0, :, 1], cpu_w), "mean_term_k1": (terms[:, 1], expected_mean_k1),
                            "d1_omega_k1": (spectrum[2, :, 1], cpu._spectrum[2, :, 1]),
                            "dtheta_omega_k1": (spectrum[3, :, 1], cpu._spectrum[3, :, 1])}
            row = {key: error(a, b) for key, (a, b) in measurements.items()}
            if not all(np.isfinite(v["max_abs"]) for v in row.values()):
                raise FloatingPointError("Nonfinite fixed-state selected mode")
            rows.append(dict(time=time, errors=row))
    worst = max(r["max_abs"] for row in rows for r in row["errors"].values())
    return dict(passed=worst <= protocol["fixed_selected_max_abs_gate"], worst_max_abs=worst, rows=rows)


def short_marches(protocol):
    rows = []
    for short in protocol["pilot"]:
        settings = dict(protocol["config"], **short)
        cpu = case.solve(Pow2RowMomentFD, linear="native", **settings)
        gpu = solve_fd_cuda_rowcomp(graph_steps=protocol["graph_steps"], **settings)
        grids = grid_set()
        fields = case.score(gpu, case.reference_fields(cpu, grids), grids)
        errors = {key: error(getattr(gpu, key), getattr(cpu, key))
                  for key in ("states", "transverse", "vorticity")}
        worst = case.maxima(fields)
        passed = all(np.isfinite(row[key]) and row[key] <= protocol["relative_gate"]
                     for row in fields for key in worst)
        passed = passed and all(errors[key]["max_abs"] <= protocol["pilot_probe_max_abs_gate"]
                                for key in ("transverse", "vorticity"))
        rows.append(dict(settings=settings, gpu=gpu.row, cpu=cpu.row, fields=fields,
                         worst_fields=worst, errors=errors, passed=passed))
    return rows


def solve(protocol, output):
    settings = {key: protocol[key] for key in ("steps", "horizon", "snapshots", "probe_spacing", "graph_steps")}
    print(f"START {settings['steps']} CUDA row-moment steps", flush=True)
    gpu = solve_fd_cuda_rowcomp(**settings, **protocol["config"])
    print(f"COMPLETE GPU {gpu.row['seconds']:.3f} s", flush=True)
    target = output.with_suffix(".npz")
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".tmp.npz")
    case.store_solution(tmp, gpu, dict(protocol["config"], linear="native"))
    tmp.replace(target)
    return dict(gpu=gpu.row, artifact=target.name, artifact_sha256=digest(target))


def diagnostic_against_old(actual, old):
    grids = grid_set()
    scores = case.score(actual, case.reference_fields(old, grids), grids)
    probes = {key: [error(a, b) for a, b in zip(getattr(actual, key).T, getattr(old, key).T)]
              for key in ("transverse", "vorticity")}
    return dict(worst_fields=case.maxima(scores), fields=scores, probes=probes, crossings=cross_phase(actual, old))


def verify(protocol, gpu_record):
    record = json.loads(gpu_record.read_text(encoding="utf-8"))
    artifact = gpu_record.parent/record["artifact"]
    if (record["protocol_sha256"] != digest(PROTOCOL) or record["source_hashes"] != source_hashes() or
            digest(artifact) != record["artifact_sha256"]):
        raise RuntimeError("CUDA artifact or protocol digest changed")
    cpu_record = json.loads((ROOT/protocol["paired_cpu_record"]).read_text(encoding="utf-8"))
    cpu_artifact = ROOT/protocol["paired_cpu_artifact"]
    if (cpu_record["status"] != "complete" or digest(cpu_artifact) != cpu_record["artifact_sha256"] or
            digest(ROOT/protocol["paired_cpu_protocol"]) != cpu_record["protocol_sha256"]):
        raise RuntimeError("CPU row-moment trajectory digest changed")
    gpu, cpu = [case.load_solution(path, Pow2RowMomentFD) for path in (artifact, cpu_artifact)]
    if gpu.row != record["gpu"] or cpu.row != cpu_record["metadata"]:
        raise ValueError("Trajectory metadata differs from its verified summary")
    for item in (gpu, cpu):
        expected = dict(protocol["config"], steps=protocol["steps"], horizon=protocol["horizon"],
                        radial_variant=protocol["radial_variant"], linear="native", time_order=2)
        if any(item.row[key] != value for key, value in expected.items()):
            raise ValueError("Row-moment trajectory metadata differs from the protocol")
        if item.row["moment_tables"] != item.model.metadata["moment_tables"]:
            raise ValueError("Row-moment trajectory used different stencil tables")
        if not np.array_equal(item.times, protocol["snapshots"]):
            raise ValueError("Row-moment trajectory uses different checkpoints")
        if not np.array_equal(item.probe_times, np.arange(0., protocol["horizon"]+.001, protocol["probe_spacing"])):
            raise ValueError("Row-moment trajectory uses different probe times")
        if not all(np.isfinite(getattr(item, key)).all() for key in ("states", "transverse", "vorticity")):
            raise FloatingPointError("Nonfinite row-moment trajectory artifact")
    paired = compare(gpu, cpu, protocol)
    paired["pass_shedding"] = all(paired[name]["passed"] for name in ("cpu_wake", "gpu_wake"))
    paired["passed"] = paired["passed"] and paired["pass_shedding"]
    old = old_reference(protocol)
    return dict(cpu=cpu.row, gpu=gpu.row, paired=paired, gpu_artifact_sha256=record["artifact_sha256"],
                cpu_artifact_sha256=cpu_record["artifact_sha256"], prior_wake=wake(old),
                gpu_vs_prior=diagnostic_against_old(gpu, old), cpu_vs_prior=diagnostic_against_old(cpu, old))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("pilot", "march", "verify"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--gpu-record", type=Path)
    cli.add_argument("--pilot-record", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "verify" and args.gpu_record is None:
        cli.error("verify requires --gpu-record")
    if args.mode == "march" and args.pilot_record is None:
        cli.error("march requires --pilot-record")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    hashes = source_hashes()
    if args.mode == "pilot":
        fixed = fixed_state_checks(protocol, old_reference(protocol))
        short = short_marches(protocol)
        result = dict(passed=fixed["passed"] and all(row["passed"] for row in short), fixed=fixed, short=short)
        print(json.dumps(dict(passed=result["passed"], fixed_max=fixed["worst_max_abs"],
                              short_worst=[row["worst_fields"] for row in short])), flush=True)
    elif args.mode == "march":
        pilot = json.loads(args.pilot_record.read_text(encoding="utf-8"))
        if not pilot["passed"] or pilot["source_hashes"] != hashes or pilot["protocol_sha256"] != digest(PROTOCOL):
            raise RuntimeError("Pilot record did not qualify the frozen sources and protocol")
        result = dict(solve(protocol, args.output), pilot_sha256=digest(args.pilot_record))
    else:
        result = verify(protocol, args.gpu_record)
        print(json.dumps({k: result["paired"][k] for k in ("passed", "worst_fields", "probes")}), flush=True)
    if source_hashes() != hashes:
        raise RuntimeError("Source files changed during the run")
    write_json(args.output, dict(protocol=protocol, protocol_sha256=digest(PROTOCOL), source_hashes=hashes, **result))
    if args.mode == "pilot" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
