"""Replay the published numerical protocols from rest without administrative inputs."""

import argparse
import json
from pathlib import Path

import cpu_formal as base
from cpu_precontracted_native import NativePrecontractedGalerkin, RecurrenceGalerkin
from cpu_sbdf2_switch import solve_cpu_switch
from fd_pow2_long import Pow2EquilibratedBufferedFD
from galerkin_cuda_variable import solve_cuda_switch


HERE = Path(__file__).resolve().parent
SWITCH = HERE/"switch_qualification_output"
REFERENCE = HERE/"early_reference_continuation_output"
case, np = base.case, base.case.np
CPU_MODELS = {"cpu_recurrence": RecurrenceGalerkin, "cpu_native_contract": NativePrecontractedGalerkin}
ARRAYS = ("states", "transverse", "vorticity")


class CountedReference(Pow2EquilibratedBufferedFD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations = 0

    def terms(self, state, amplitude):
        values = super().terms(state, amplitude)
        self.evaluations += 1
        return values


def selected_times(values, horizon):
    if horizon not in values:
        raise ValueError("Replay horizon must be an explicitly registered snapshot")
    return tuple(t for t in values if t <= horizon)


def switched(args):
    protocol = SWITCH/"protocol.json"
    frozen = base.read(protocol)
    p = frozen["template"]
    if frozen["status"] != "frozen" or args.method not in p["methods"] or args.arm not in p["arms"]:
        raise ValueError("Choose a registered switch method and implementation")
    seeds = p["rfm_seeds"] if frozen["specs"][args.method]["method"] == "rfm" else [p["deterministic_seed"]]
    seed = seeds[0] if args.seed is None else args.seed
    if seed not in seeds or not p["switch_time"] < args.horizon <= p["horizon"]:
        raise ValueError("Choose a registered seed and a horizon beyond the single bridge")
    snapshots = selected_times(frozen["checkpoints"], args.horizon)
    config = base.config(frozen["specs"][args.method], frozen["solve_protocol"], seed)[1]
    settings = dict(startup_rate=p["startup_rate"], late_rate=p["late_rate"], switch_time=p["switch_time"],
                    horizon=args.horizon, snapshots=snapshots, probe_spacing=p["probe_spacing"], **config)
    if args.arm == "cuda_graph":
        solution = solve_cuda_switch(graph_steps=p["graph_steps"], **settings)
        reconstruction = RecurrenceGalerkin
    else:
        reconstruction = CPU_MODELS[args.arm]
        solution = solve_cpu_switch(reconstruction, **settings)
    job = dict(role="candidate", name=args.method, seed=seed, arm=args.arm)
    rows = base.read(SWITCH/"runs.json")["rows"] if args.check_archive else []
    return solution, config, reconstruction, protocol, SWITCH, next((r for r in rows if r["job"] == job), None)


def fd_reference(args):
    protocol = REFERENCE/"protocol.json"
    frozen = base.read(protocol)
    p = frozen["protocol"]
    if frozen["status"] != "frozen" or not 0 < args.horizon <= p["horizon"]:
        raise ValueError("The FD replay must use the archived reference time range")
    snapshots = selected_times(p["checkpoints"], args.horizon)
    spec = next(r for r in p["reference_path"] if r["name"] == args.axis)
    config = dict(n=spec["n"], k=spec["k"], grid="outer", stretch=1., angular="fft", linear="native")
    steps = int(case.step_indices([args.horizon], 1/spec["rate"])[0])
    solution = case.solve(CountedReference, steps=steps, horizon=args.horizon, snapshots=snapshots,
                          probe_spacing=.125, **config)
    count = solution.model.evaluations
    if count != steps-1:
        raise RuntimeError("The FD reference used unexpected nonlinear work")
    solution.row.update(nonlinear_evaluations=count, marching_fft_batches=2*count, inverse_field_transforms=4*count,
                        forward_field_transforms=count, fft_radial_lines=spec["n"]-1, priming_fft_batches=0)
    rows = base.read(REFERENCE/"runs.json")["arms"] if args.check_archive else []
    record = next((r for r in rows if r["spec"]["name"] == args.axis), None)
    return solution, config, Pow2EquilibratedBufferedFD, protocol, REFERENCE, record


def archive_comparison(solution, cls, folder, row):
    if row is None or row["status"] != "complete" or Path(row["artifact"]).name != row["artifact"]:
        raise ValueError("The matching completed archive is missing")
    path = folder/row["artifact"]
    if base.digest(path) != row["sha256"]:
        raise ValueError("The matching trajectory digest changed")
    stored = case.load_solution(path, cls)
    selected = np.flatnonzero(np.isin(stored.times, solution.times))
    count = len(solution.probe_times)
    if not np.array_equal(stored.times[selected], solution.times) or not np.array_equal(stored.probe_times[:count], solution.probe_times):
        raise ValueError("Replay and archive use different absolute observation times")
    expected = (stored.states[selected], stored.transverse[:count], stored.vorticity[:count])
    output = {}
    for name, reference in zip(ARRAYS, expected, strict=True):
        values = getattr(solution, name)
        error = values-reference
        scale = float(np.linalg.norm(reference))
        maximum = float(np.max(abs(error)))
        relative = float(np.linalg.norm(error))/scale if scale > 1e-12 else None
        output[name] = dict(bitwise=bool(np.array_equal(values, reference)), max_abs=maximum, relative_l2=relative,
                            passed=relative <= 1e-5 if relative is not None else maximum <= 1e-12)
    return dict(passed=all(row["passed"] for row in output.values()), archive=path.name, archive_sha256=row["sha256"],
                arrays=output, scope="Implementation array comparison at matching times. No new physical qualification.")


def write_new(path, callback):
    with path.open("xb") as output:
        try:
            callback(output)
        except BaseException:
            output.close()
            path.unlink()
            raise


def execute(args):
    output, sidecar = args.output, args.json_output
    if output.suffix != ".npz" or (sidecar is not None and sidecar.suffix != ".json"):
        raise ValueError("Use a new .npz path and an optional new .json path")
    if output.exists() or (sidecar is not None and sidecar.exists()):
        raise FileExistsError("Replay outputs must have new paths")
    action = switched if args.mode == "switch" else fd_reference
    solution, config, cls, protocol, folder, record = action(args)
    if not all(np.isfinite(getattr(solution, name)).all() for name in ARRAYS):
        raise FloatingPointError("Replay produced a nonfinite field or probe")
    solution.row.setdefault("priming_fft_batches", 0)
    solution.row.update(replay_mode=args.mode, replay_numeric_protocol_sha256=base.digest(protocol),
                        timing_scope="Reproduction only. Solver times are diagnostic.")
    comparison = archive_comparison(solution, cls, folder, record) if args.check_archive else None
    output.parent.mkdir(parents=True, exist_ok=True)
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, lambda handle: case.store_solution(handle, solution, config))
    report = dict(status="complete" if comparison is None or comparison["passed"] else "archive_comparison_failed",
                  artifact=output.name, artifact_sha256=base.digest(output), runner_sha256=base.digest(Path(__file__)),
                  numeric_protocol=protocol.relative_to(base.WORK.parent).as_posix(), config=config,
                  observation_times=solution.times.tolist(), probe_samples=len(solution.probe_times),
                  metadata=solution.row, comparison=comparison)
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if sidecar is not None:
        write_new(sidecar, lambda handle: handle.write(payload.encode("utf-8")))
    print(payload, end="")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    switch = sub.add_parser("switch", help="Run the fixed 128 to 112 Galerkin bridge")
    switch.add_argument("--method", choices=("rfm37_k72", "leg37_k72"), default="rfm37_k72")
    switch.add_argument("--seed", type=int)
    switch.add_argument("--arm", choices=(*CPU_MODELS, "cuda_graph"), default="cpu_recurrence")
    switch.add_argument("--horizon", type=float, default=72.)
    fd = sub.add_parser("fd-reference", help="Run one member of the T16 FD reference chain from rest")
    fd.add_argument("--axis", choices=("radial_coarse", "radial_fine", "angular_fine", "time_fine"), default="time_fine")
    fd.add_argument("--horizon", type=float, default=16.)
    for parser in (switch, fd):
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--json-output", type=Path)
        parser.add_argument("--check-archive", action="store_true", help="Also compare saved arrays at matching absolute times")
    raise SystemExit(execute(cli.parse_args()))
