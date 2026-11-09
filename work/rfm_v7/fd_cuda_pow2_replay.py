"""Predeclared physical replay gate for the high-resolution CPU pow2 solution."""

import argparse
import json
from pathlib import Path

import numpy as np

from cpu_formal import digest, grid_set, wake
from fd_cuda_pow2 import case, Pow2EquilibratedBufferedFD, solve_fd_cuda_pow2

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"protocol_fd_cuda_pow2.json"


def write_json(path, record):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def reference(protocol):
    path = ROOT/protocol["reference"]
    if digest(path) != protocol["reference_sha256"]:
        raise RuntimeError("CPU reference digest changed")
    solution = case.load_solution(path, Pow2EquilibratedBufferedFD)
    expected = dict(protocol["config"], steps=protocol["steps"], horizon=protocol["horizon"],
                    time_order=2, linear="native", equilibration="row_power_of_two")
    for key, value in expected.items():
        if solution.row[key] != value:
            raise ValueError(f"CPU reference metadata mismatch for {key}")
    if not np.array_equal(solution.times, protocol["snapshots"]):
        raise ValueError("CPU reference snapshot schedule changed")
    if solution.model.scale_metadata() != solution.row["row_scale"]:
        raise ValueError("CPU reference factors could not be reconstructed")
    return solution


def compare(actual, cpu, protocol):
    for key in ("n", "k", "stretch", "grid", "angular", "linear", "ntheta", "dt", "horizon",
                "steps", "time_order", "row_scale", "reconstruction_class"):
        if actual.row[key] != cpu.row[key]:
            raise ValueError(f"CPU/GPU replay metadata mismatch for {key}")
    for key in ("times", "probe_times"):
        if not np.array_equal(getattr(actual, key), getattr(cpu, key)):
            raise ValueError(f"CPU/GPU replay schedule mismatch for {key}")
    grids = grid_set()
    fields = case.score(actual, case.reference_fields(cpu, grids), grids)
    if len(fields) != len(grids)*len(cpu.times):
        raise RuntimeError("Incomplete physical field score rows")
    worst = case.maxima(fields)
    probes = {}
    for name in ("transverse", "vorticity"):
        a, b = getattr(actual, name), getattr(cpu, name)
        norms = np.linalg.norm(b, axis=0)
        if np.any(norms <= 1e-12):
            raise ValueError("Reference wake signal is too small for a relative probe gate")
        probes[name] = dict(relative_l2=(np.linalg.norm(a-b, axis=0)/norms).tolist(),
                            max_abs=np.max(np.abs(a-b), axis=0).tolist())
    wake_cpu, wake_gpu = wake(cpu), wake(actual)
    crossings = {}
    for name in ("velocity", "vorticity"):
        crossings[name] = []
        for p, q in zip(wake_cpu[name], wake_gpu[name]):
            x, y = [np.asarray(r["positive_crossing_times"]) for r in (p, q)]
            differences = (y-x).tolist() if len(x) == len(y) and len(x) > 1 else None
            crossings[name].append(dict(cpu=x.tolist(), gpu=y.tolist(), differences=differences))
    field_values = [r[key] for r in fields for key in worst]
    probe_values = [value for row in probes.values() for value in row["relative_l2"]]
    pass_fields = bool(np.isfinite(field_values).all() and max(field_values) <= protocol["relative_gate"])
    pass_probes = bool(np.isfinite(probe_values).all() and max(probe_values) <= protocol["relative_gate"])
    pass_crossings = all(row["differences"] is not None and
                         max(map(abs, row["differences"])) <= protocol["crossing_time_gate"]
                         for rows in crossings.values() for row in rows)
    return dict(pass_fields=pass_fields, pass_probes=pass_probes, pass_crossings=pass_crossings,
                passed=pass_fields and pass_probes and pass_crossings, worst_fields=worst,
                fields=fields, probes=probes, crossings=crossings, cpu_wake=wake_cpu, gpu_wake=wake_gpu)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--linear", choices=("dense", "band"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    artifact = args.output.with_suffix(".npz")
    if args.output.exists() or artifact.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cpu = reference(protocol)
    config = protocol["config"]
    kwargs = {key: protocol[key] for key in ("steps", "horizon", "snapshots", "probe_spacing",
                                            "graph_steps", "radial")}
    print(f"START {args.linear} {kwargs['steps']} steps", flush=True)
    actual = solve_fd_cuda_pow2(linear=args.linear, **kwargs, **config)
    print(f"COMPLETE {args.linear} {actual.row['seconds']:.3f} s", flush=True)
    result = compare(actual, cpu, protocol)
    temp = artifact.with_suffix(".tmp.npz")
    case.store_solution(temp, actual, dict(config, linear="native"))
    temp.replace(artifact)
    write_json(args.output, dict(protocol=protocol, protocol_sha256=digest(PROTOCOL),
                                 cpu=cpu.row, gpu=actual.row, artifact_sha256=digest(artifact), **result))
    print(json.dumps({key: result[key] for key in ("passed", "pass_fields", "pass_probes",
                                                   "pass_crossings", "worst_fields", "probes")}), flush=True)


if __name__ == "__main__":
    main()
