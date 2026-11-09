"""Derive checked, paired timing statistics from the completed six-way cohort."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


OUT = Path(__file__).resolve().parent / "hardware_pow2_output"
GPU_PHASES = ("host_setup_seconds", "host_linear_precompute_seconds", "runtime_init_seconds",
              "upload_seconds", "plan_seconds", "graph_setup_seconds", "march_seconds",
              "download_seconds")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def describe(samples):
    values = list(samples)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Expected nonempty finite nonnegative samples")
    return dict(median=median(values), minimum=min(values), maximum=max(values), samples=values)


def phase_values(row):
    metadata = row["metadata"]
    phases = {key:metadata[key] for key in GPU_PHASES if key in metadata}
    if row["backend"] == "cpu":
        phases.update(setup_seconds=metadata["setup_seconds"],
                      solve_minus_setup_seconds=metadata["seconds"]-metadata["setup_seconds"])
    else:
        required = set(GPU_PHASES)-{"host_linear_precompute_seconds"}
        if set(phases)!=(required | ({"host_linear_precompute_seconds"} if row["name"].startswith("fd") else set())):
            raise ValueError("GPU phase schema differs from the measured implementation")
    return dict(phases, wall_seconds=row["seconds"], solver_reported_seconds=metadata["seconds"],
                wall_minus_solver_reported_seconds=row["seconds"]-metadata["seconds"])


def summarize(source, verify_artifacts=False):
    data = json.loads(source.read_text(encoding="utf-8"))
    frozen, rows = data["frozen"], data["rows"]
    choices = {r["method"]:r["name"] for r in frozen["choices"]}
    backends = frozen["template"]["backends"]
    seeds = frozen["protocol"]["seeds"]
    deterministic = frozen["protocol"]["deterministic_seed"]
    blocks = range(1,6)
    if (data["status"]!="complete" or len(rows)!=30 or len(set(seeds))!=5 or len(seeds)!=5 or
            set(choices)!={"rfm","legendre","fd4"} or backends!=["cpu","cuda_graph"]):
        raise ValueError("Expected a completed five-block, six-method cohort")
    fields = ("block","backend","name","seed")
    actual = {tuple(r[k] for k in fields) for r in rows}
    scheduled = {tuple(r[k] for k in fields) for r in frozen["schedule"]}
    expected = {(b,backend,name,seeds[b-1] if family=="rfm" else deterministic)
                for b in blocks for backend in backends for family,name in choices.items()}
    if actual!=expected or scheduled!=expected or len(frozen["schedule"])!=len(expected):
        raise ValueError("Incomplete or duplicated block/backend/variant/seed inventory")
    indexed = {(r["block"],r["backend"],r["name"]):r for r in rows}
    failures = [dict({k:r[k] for k in fields}, detail=r.get("failure"),
                     implementation_pass=r["implementation_pass"], numerical_pass=r["numerical_pass"])
                for r in rows if ("failure" in r or not r["implementation_pass"] or not r["numerical_pass"])]
    groups=[]
    for family,name in choices.items():
        for backend in backends:
            group=[indexed[b,backend,name] for b in blocks]
            phases=[]
            for r in group:
                if "seconds" not in r:
                    continue
                if verify_artifacts and sha(source.parent/r["artifact"])!=r["sha256"]:
                    raise ValueError(f"Trajectory hash mismatch: {r['artifact']}")
                phases.append(phase_values(r))
            if phases and any(p.keys()!=phases[0].keys() for p in phases):
                raise ValueError("Phase sets differ within one method/backend")
            groups.append(dict(family=family,name=name,backend=backend,blocks=list(blocks),
                seeds=[r["seed"] for r in group],count=len(group),successful_timings=len(phases),
                failure_count=sum(f["backend"]==backend and f["name"]==name for f in failures),
                phases_seconds={key:describe(p[key] for p in phases) for key in phases[0]} if phases else {}))

    def paired(numerator, denominator):
        if failures:
            return dict(numerator=numerator,denominator=denominator,status="withheld_incomplete_acceptance")
        ratios = []
        for block in blocks:
            a,b = (indexed[block,*key] for key in (numerator,denominator))
            ratios.append(dict(block=block,numerator_seed=a["seed"],denominator_seed=b["seed"],
                               numerator_seconds=a["seconds"],denominator_seconds=b["seconds"],
                               ratio=a["seconds"]/b["seconds"]))
        stats = describe(r["ratio"] for r in ratios)
        stats.pop("samples")
        return dict(numerator=numerator,denominator=denominator,status="complete",blocks=ratios,
                    denominator_wins=sum(r["ratio"]>1 for r in ratios),ratio=stats)

    same_hardware=[paired((backend,choices[a]),(backend,choices[b])) for backend in backends for a,b in
                   (("fd4","rfm"),("fd4","legendre"),("legendre","rfm"))]
    same_method=[paired(("cpu",name),("cuda_graph",name)) for name in choices.values()]
    return dict(status="passed" if not failures else "completed_with_failures",
        derived_from=dict(file=source.name,sha256=sha(source),trajectory_hashes_verified=verify_artifacts),
        observations=len(rows),expected_observations=len(expected),registered_rfm_seeds=seeds,
        complete_registered_seed_inventory=True,failure_count=len(failures),failures=failures,
        groups=groups,paired_same_hardware=same_hardware,paired_same_method_cpu_over_gpu=same_method,
        scope=dict(wall_seconds="Fresh in-process end-to-end solve. Excludes score and artifact writes.",
                   phase_seconds="Solver instrumentation. CPU solve-minus-setup includes probes, outputs and other remaining solver work. GPU cleanup is in wall/solver total and is not instrumented as a separate phase.",
                   paired="Ratios computed within each randomized block. Range is min/max, not a confidence interval."))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify-artifacts",action="store_true")
    args=cli.parse_args()
    destination=OUT/"summary.json"
    result=summarize(OUT/"timing.json",args.verify_artifacts)
    with destination.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")


if __name__=="__main__":
    main()
