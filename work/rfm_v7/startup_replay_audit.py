"""Audit fixed T=8 startup sensitivity at the selected and control SBDF2 rates."""

import argparse
import math
from pathlib import Path

from basis_recurrence_model import native_model
from fd_cuda_pow2_replay import write_json
from galerkin_cuda_native import solve_native
from verify_v7_operator_round import (
    HERE, REPEAT, METRICS, NativePrecontractedGalerkin, RecurrenceGalerkin, base, case, check, np, worst,
)


OUT = HERE/"startup_replay_output"
METHODS = {"rfm37_k72": 7331, "leg37_k72": 17}
RATES = (112, 128)
HORIZON = 8.
TIMES = (.5, 1., 2., 4., 6., 8.)
ARMS = ("cpu_recurrence", "cpu_native_contract", "cuda_graph")


def relative(a, b):
    difference = a-b
    scale = float(np.linalg.norm(b))
    return dict(relative_l2=float(np.linalg.norm(difference))/scale if scale > 1e-12 else None,
                reference_l2=scale, max_abs=float(np.max(np.abs(difference))))


def validate(sol, settings, rate):
    probes = np.arange(0., HORIZON+.001, .125)
    check(sol.row["steps"] == int(HORIZON*rate) and sol.row["dt"] == 1/rate and
          np.array_equal(sol.times, TIMES) and np.array_equal(sol.probe_times, probes),
          "Startup observation schedule differs")
    check(all(np.isfinite(getattr(sol, key)).all() for key in ("states", "transverse", "vorticity")),
          "Nonfinite startup trajectory")
    check(all(sol.row[key] == settings[key] for key in ("m", "k", "q")), "Startup spatial setting differs")


def solve(arm, settings, rate, graph_steps):
    args = dict(steps=int(HORIZON*rate), horizon=HORIZON, snapshots=TIMES, **settings)
    if arm == "cpu_recurrence":
        return case.solve(RecurrenceGalerkin, **args)
    if arm == "cpu_native_contract":
        return case.solve(NativePrecontractedGalerkin, **args)
    with native_model():
        return solve_native(graph_steps=graph_steps, **args)


def compare(candidate, control, grids, reference, gate):
    scores = case.score(candidate, reference, grids)
    keys = {(grid, t) for grid in grids for t in TIMES}
    check(len(scores) == len(keys) and {(r["grid"], r["time"]) for r in scores} == keys,
          "Incomplete startup field score grid")
    fields = worst(scores)
    probe = {}
    for key in ("transverse", "vorticity"):
        actual, baseline = getattr(candidate, key), getattr(control, key)
        probe[key] = dict(total=relative(actual, baseline),
                          per_station=[relative(a, b) for a, b in zip(actual.T, baseline.T)],
                          pre_pulse=relative(actual[control.probe_times <= 4], baseline[control.probe_times <= 4]),
                          post_pulse=relative(actual[control.probe_times > 4], baseline[control.probe_times > 4]))
    state = [dict(time=t, **relative(a, b)) for t, a, b in zip(TIMES, candidate.states, control.states)]
    values = [r[k] for r in scores for k in METRICS]
    values.extend(row["total"]["relative_l2"] for row in probe.values())
    passed = all(v is not None and math.isfinite(v) and 0 <= v <= gate for v in values)
    return dict(passed=passed, fields=scores, worst_fields=fields, probes=probe, state=state)


def run(output):
    freeze_path = REPEAT/"protocol.json"
    frozen = base.read(freeze_path)
    template, protocol = frozen["template"], frozen["inputs"]["solve_protocol"]
    check((template["galerkin_rate"], template["galerkin_cuda_control_rate"]) == RATES and
          template["implementation_tolerance"] == 1e-5 and set(template["galerkin_names"]) == METHODS.keys() and
          protocol["seeds"][0] == METHODS["rfm37_k72"] and protocol["deterministic_seed"] == METHODS["leg37_k72"],
          "Startup audit differs from the paired full-trajectory settings")
    sources = dict(frozen["inputs"]["source_hashes"])
    for path in (Path(__file__).resolve(), HERE/"verify_v7_operator_round.py"):
        sources[path.relative_to(base.WORK).as_posix()] = base.digest(path)
    for name, digest in sources.items():
        check(base.digest(base.WORK/name) == digest, f"Frozen startup audit source changed: {name}")
    gate = template["implementation_tolerance"]
    audit = dict(status="frozen", original_protocol=freeze_path.relative_to(base.WORK).as_posix(),
                 original_protocol_sha256=base.digest(freeze_path), source_hashes=sources, methods=METHODS,
                 rates=list(RATES), horizon=HORIZON, snapshots=list(TIMES), arms=list(ARMS),
                 graph_steps=template["cuda_graph_steps"], implementation_tolerance=gate,
                 scope="Accuracy-only paired startup sensitivity. No time ranking or claim about the cause.")
    output.mkdir()
    write_json(output/"protocol.json", audit)
    journal = output/"audit.json"
    evidence = dict(status="running", protocol_sha256=base.digest(output/"protocol.json"), trajectories=[], comparisons=[])
    write_json(journal, evidence)
    grids = base.grid_set()
    specs = {s["name"]: s for s in protocol["variants"]}
    for name, seed in METHODS.items():
        _, settings = base.config(specs[name], protocol, seed)
        for rate in RATES:
            cohort = {}
            for arm in ARMS:
                sol = solve(arm, settings, rate, audit["graph_steps"])
                validate(sol, settings, rate)
                target = output/f"{name}_{seed}_r{rate}_{arm}.npz"
                temporary = target.with_suffix(".tmp.npz")
                case.store_solution(temporary, sol, settings)
                temporary.replace(target)
                cohort[arm] = sol
                evidence["trajectories"].append(dict(name=name, seed=seed, rate=rate, arm=arm,
                    artifact=target.name, sha256=base.digest(target), metadata=sol.row))
                write_json(journal, evidence)
            control = cohort["cpu_recurrence"]
            truth = case.reference_fields(control, grids)
            for arm in ARMS[1:]:
                result = compare(cohort[arm], control, grids, truth, gate)
                evidence["comparisons"].append(dict(name=name, seed=seed, rate=rate, arm=arm, **result))
                print(name, rate, arm, "PASS" if result["passed"] else "FAIL", result["worst_fields"], flush=True)
                write_json(journal, evidence)
    check(len(evidence["trajectories"]) == 12 and len(evidence["comparisons"]) == 8, "Incomplete startup audit")
    evidence["status"] = "complete"
    evidence["outcomes"] = {str(rate): {"passed": sum(row["passed"] for row in evidence["comparisons"] if row["rate"] == rate),
                                       "compared": sum(row["rate"] == rate for row in evidence["comparisons"])}
                            for rate in RATES}
    write_json(journal, evidence)
    return evidence["outcomes"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT, help="A new directory for immutable startup replay evidence")
    print(run(cli.parse_args().output))
