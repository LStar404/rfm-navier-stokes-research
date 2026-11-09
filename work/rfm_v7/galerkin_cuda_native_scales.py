"""Restartable three-resolution native CUDA diagnostics on short physical windows."""

import argparse
from pathlib import Path
import statistics
import time

from cpu_formal import (PROTOCOL, WORK, BufferedGalerkin, case, config, digest, grid_set,
                        read, save, source_hashes)
from galerkin_cuda_native import solve_native


ROOT = Path(__file__).resolve().parent
ENGINES = ("cpu", "cuda-eager", "cuda-graph")
SOURCES = ("cuda_environment_native.py", "galerkin_cuda_native.py",
           "galerkin_cuda_native.cu", "galerkin_cuda_native_scales.py")


def verify_saved(path, row):
    if not path.is_file() or digest(path) != row.get("sha256"):
        raise RuntimeError(f"Checkpoint is missing or changed: {path.name}")


def run(job, directory, protocol, horizon, profile_steps):
    seq, engine, spec, seed = job
    stem = f"{seq:02d}_{spec['name']}_{engine}"
    data_path = directory / f"{stem}.npz"
    row_path = directory / f"{stem}.json"
    key = dict(sequence=seq, engine=engine, spec=spec, seed=seed)
    if row_path.exists():
        result = read(row_path)
        if any(result.get(k) != v for k, v in key.items()):
            raise RuntimeError(f"Checkpoint job changed: {stem}")
        if "artifact" in result:
            verify_saved(data_path, result)
        return result
    if data_path.exists():
        data_path.rename(data_path.with_suffix(f".orphan_{time.time_ns()}.npz"))
    _, settings = config(spec, protocol, seed)
    parameters = dict(steps=round(horizon*spec["rate"]), horizon=horizon,
                      snapshots=(horizon/2,horizon), **settings)
    start = time.perf_counter()
    try:
        if engine == "cpu":
            solution = case.solve(BufferedGalerkin, **parameters)
        else:
            solution = solve_native(**parameters, graph_steps=24 if engine == "cuda-graph" else 0,
                                    profile_steps=profile_steps if engine == "cuda-graph" else 0)
        elapsed = time.perf_counter() - start
        tmp = data_path.with_suffix(".tmp.npz")
        case.store_solution(tmp, solution, settings)
        tmp.replace(data_path)
        result = dict(key, artifact=data_path.name, sha256=digest(data_path), metadata=solution.row,
                      diagnostic_seconds=elapsed)
    except (ArithmeticError, RuntimeError, ValueError, OSError) as exc:
        result = dict(key, failure=str(exc))
    save(row_path, result)
    return result


def compare(rows, directory):
    grids = grid_set()
    for row in rows:
        if row["engine"] != "cpu" or "failure" in row:
            continue
        baseline = case.load_solution(directory / row["artifact"], BufferedGalerkin)
        truth = case.reference_fields(baseline, grids)
        for other in rows:
            if (other["engine"] != "cpu" and other["sequence"] == row["sequence"]
                    and other["spec"] == row["spec"] and "failure" not in other):
                sol = case.load_solution(directory / other["artifact"], BufferedGalerkin)
                other["fields_vs_cpu"] = case.maxima(case.score(sol, truth, grids))
                other["probe_max_abs"] = dict(
                    transverse=float(case.np.max(abs(sol.transverse - baseline.transverse))),
                    vorticity=float(case.np.max(abs(sol.vorticity - baseline.vorticity))))
    result = []
    for name in dict.fromkeys(row["spec"]["name"] for row in rows):
        group = [r for r in rows if r["spec"]["name"] == name]
        entries = {}
        for engine in ENGINES:
            items = [r for r in group if r["engine"] == engine]
            times = [r["diagnostic_seconds"] for r in items if "diagnostic_seconds" in r]
            entries[engine] = dict(count=len(items), times=times,
                                   median=statistics.median(times) if times else None,
                                   failures=sum("failure" in r for r in items))
        result.append(dict(name=name, engines=entries))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, default=ROOT / "galerkin_cuda_native_scales")
    cli.add_argument("--horizon", type=float, default=8.)
    cli.add_argument("--rounds", type=int, default=1)
    cli.add_argument("--profile-steps", type=int, default=0)
    args = cli.parse_args()
    protocol = read(PROTOCOL)
    if args.rounds < 1 or args.horizon <= 0:
        raise ValueError("Positive pilot rounds and horizon required")
    specs = [s for s in protocol["variants"] if s["method"] == "rfm"]
    ordered = [(engine, spec) for engine in ENGINES for spec in specs]
    rng = case.np.random.default_rng(protocol["order_seed"])
    jobs = [(j, ordered[i][0], ordered[i][1], protocol["seeds"][j % len(protocol["seeds"])])
            for j in range(args.rounds) for i in rng.permutation(len(ordered))]
    hashes = source_hashes()
    hashes.update({(ROOT / name).relative_to(WORK).as_posix(): digest(ROOT / name) for name in SOURCES})
    frozen = dict(jobs=[list(j) for j in jobs], hashes=hashes, horizon=args.horizon,
                  profile_steps=args.profile_steps, kind="short-window diagnostics")
    journal = args.directory / "journal.json"
    if journal.exists():
        data = read(journal)
        if data["frozen"] != frozen:
            raise RuntimeError("Frozen scale protocol or sources changed")
    else:
        data = dict(status="running", frozen=frozen, rows=[])
        save(journal, data)
    rows = []
    for job in jobs:
        row = run(job, args.directory, protocol, args.horizon, args.profile_steps)
        rows.append(row)
        data["rows"] = rows
        save(journal, data)
        print(row["spec"]["name"], row["engine"], row.get("diagnostic_seconds", row.get("failure")), flush=True)
    data.update(status="complete", summary=compare(rows, args.directory))
    save(journal, data)


if __name__ == "__main__":
    main()
