"""Recompute the startup and rate-128 follow-up from archived fields."""

import argparse
import json
import math
from pathlib import Path

import r128_operator_qualification as qualified
import startup_replay_audit as startup
from startup_replay_modal_audit import run as rebuild_modal_audit
import verify_v7_operator_round as prior


base, case, np = prior.base, prior.case, prior.np
ROOT, HERE = prior.ROOT, prior.HERE
check, numeric_match, compare_scores, worst = prior.check, prior.numeric_match, prior.compare_scores, prior.worst
EARLY = (.5, 1., 2., 4., 6., 8.)
LATE = (24., 36., 48., 54., 60., 66., 72.)
GATE = 1e-5
PHYSICAL_GATE = .005
R128 = 128
SEEDS = {"rfm37_k72": (7331, 7687, 8011, 8353, 9041), "leg37_k72": (17,)}
REFS = ("angular_fine", "time_fine")
PROBES = ("transverse", "vorticity")
OUTCOMES = {"112": {"passed": 0, "compared": 4}, "128": {"passed": 4, "compared": 4}}


def add_digest(records, path, expected=None):
    key = path.relative_to(ROOT).as_posix()
    actual = base.digest(path)
    check(expected in (None, actual) and records.get(key, actual) == actual, f"Archive digest differs: {key}")
    records[key] = actual
    return actual


def check_sources(records, entries):
    for name, digest in entries.items():
        add_digest(records, prior.archived(base.WORK, name), digest)


def unique(rows, key, expected, label):
    indexed = {key(row): row for row in rows}
    check(len(indexed) == len(rows) and indexed.keys() == expected, f"Unexpected or duplicate {label}")
    return indexed


def result_matches(raw, record, where):
    numeric_match(raw, {k: record[k] for k in raw}, where)


def match_times(sol, times, probes, steps, rate, label):
    check(np.array_equal(sol.times, times) and np.array_equal(sol.probe_times, probes) and
          sol.states.shape == (len(times), *sol.model.rhs_shape) and sol.states.dtype == np.complex128 and
          sol.row["steps"] == steps and sol.row["dt"] == 1/rate, f"State matrix or timing differs: {label}")
    for name in PROBES:
        check(getattr(sol, name).shape == (len(probes), len(case.PROBES)), f"Probe matrix differs: {label}/{name}")
    check(all(np.isfinite(getattr(sol, k)).all() for k in ("states", *PROBES)), f"Nonfinite trajectory: {label}")


def load_archive(path, row, config, times, probes, steps, rate, arm, raw_hashes):
    add_digest(raw_hashes, path, row["sha256"])
    with np.load(path, allow_pickle=False) as raw:
        check(json.loads(str(raw["config"])) == config and json.loads(str(raw["metadata"])) == row["metadata"],
              f"Archived settings differ: {path.name}")
    sol = case.load_solution(path, qualified.MODELS[arm])
    match_times(sol, times, probes, steps, rate, path.name)
    return sol


def relative(actual, reference):
    scale = float(np.linalg.norm(reference))
    difference = actual-reference
    return dict(reference_l2=scale, relative_l2=float(np.linalg.norm(difference))/scale if scale > 1e-12 else None,
                max_abs=float(np.max(np.abs(difference))))


def short_metrics(candidate, baseline, grids, truth):
    scores = case.score(candidate, truth, grids)
    probes = {}
    first = baseline.probe_times <= 4
    for name in PROBES:
        a, b = getattr(candidate, name), getattr(baseline, name)
        probes[name] = dict(total=relative(a, b), per_station=[relative(x, y) for x, y in zip(a.T, b.T)],
                            pre_pulse=relative(a[first], b[first]), post_pulse=relative(a[~first], b[~first]))
    state = [dict(time=t, **relative(a, b)) for t, a, b in zip(EARLY, candidate.states, baseline.states)]
    values = [row[k] for row in scores for k in prior.METRICS]
    values.extend(probes[k]["total"]["relative_l2"] for k in PROBES)
    return dict(passed=all(v is not None and math.isfinite(v) and 0 <= v <= GATE for v in values),
                fields=scores, worst_fields=worst(scores), probes=probes, state=state)


def verify_prior_archives(inputs):
    isolation = prior.verify_isolation()
    _, original = prior.verify_timing(prior.ORIGINAL, rescore=False)
    frozen, repeat = prior.verify_timing(prior.REPEAT, rescore=True)
    reference = prior.verify_reference_replays()
    check(not isolation["isolated_speed_claim_supported"] and original["publication_role"] == "diagnostic_only" and
          repeat["publication_role"] == "diagnostic_only", "Both older timing cohorts must stay diagnostic")
    for directory in (prior.ORIGINAL, prior.REPEAT):
        for name in ("protocol.json", "timing.json", "summary.json"):
            add_digest(inputs, directory/name)
    status_path = startup.OUT/"release_replay_status.json"
    status = base.read(status_path)
    add_digest(inputs, status_path)
    add_digest(inputs, startup.OUT/status["archive"]["result"], status["archive"]["sha256"])
    add_digest(inputs, HERE/"verify_v7_operator_round.py", status["verifier_source_sha256"])
    add_digest(inputs, startup.OUT/status["followup"]["artifact"], status["followup"]["sha256"])
    old_failure = status["original_optional_replay"]
    check(status["status"] == "archived_verification_passed_startup_replay_failed" and
          status["archive"]["exit_code"] == 0 and old_failure["exit_code"] == 1 and
          old_failure["completed_archive_checks"] and old_failure["rate"] == 112 and
          old_failure["implementation_tolerance"] == GATE and old_failure["message"] ==
          "Short implementation gate failed: rfm37_k72/cpu_native_contract" and
          status["followup"]["outcomes"] == OUTCOMES, "Original optional replay failure record changed")
    released = base.read(startup.OUT/status["archive"]["result"])
    check(released["status"] == "passed" and released["trajectory_solves_run"] == 0 and
          released["reference"] == reference and released["isolation"] == isolation,
          "Archived default release verification differs")
    return frozen, dict(status="passed", old_timing_rows=original["timing_rows"], repeat_timing_rows=repeat["timing_rows"],
                        repeated_field_values=repeat["field_errors_recomputed"], original_optional_replay="failed_at_rate112",
                        isolated_speed_claim_supported=False, reference=reference)


def verify_startup(frozen, grids, sources, inputs, raw_hashes):
    directory = startup.OUT
    audit_path, protocol_path = directory/"audit.json", directory/"protocol.json"
    audit, protocol = base.read(audit_path), base.read(protocol_path)
    add_digest(inputs, audit_path)
    add_digest(inputs, protocol_path, audit["protocol_sha256"])
    add_digest(inputs, prior.archived(base.WORK, protocol["original_protocol"]), protocol["original_protocol_sha256"])
    check(protocol["status"] == "frozen" and audit["status"] == "complete" and
          protocol["methods"] == {method: ss[0] for method, ss in SEEDS.items()} and protocol["rates"] == [112, 128] and
          tuple(protocol["arms"]) == startup.ARMS and tuple(protocol["snapshots"]) == EARLY and
          protocol["horizon"] == 8 and protocol["implementation_tolerance"] == GATE,
          "Startup cohort is not the fixed early-step comparison")
    check_sources(sources, protocol["source_hashes"])
    expected = {(name, seeds[0], rate, arm) for name, seeds in SEEDS.items() for rate in (112, 128) for arm in startup.ARMS}
    def key(row):
        return tuple(row[k] for k in ("name", "seed", "rate", "arm"))
    trajectories = unique(audit["trajectories"], key, expected, "startup trajectory identity")
    pairs = unique(audit["comparisons"], key, {i for i in expected if i[-1] != startup.ARMS[0]}, "startup pair")
    solve_protocol = frozen["inputs"]["solve_protocol"]
    specs = {s["name"]: s for s in solve_protocol["variants"]}
    probes = np.arange(0., 8.001, .125)
    keys = {(g, t) for g in grids for t in EARLY}
    outcomes = {str(rate): dict(passed=0, compared=0) for rate in (112, 128)}
    maxima = {str(rate): dict(field=0., probe=0.) for rate in (112, 128)}
    for name, seeds in SEEDS.items():
        seed = seeds[0]
        _, config = base.config(specs[name], solve_protocol, seed)
        for rate in (112, 128):
            models = {}
            for arm in startup.ARMS:
                row = trajectories[name, seed, rate, arm]
                expected_name = f"{name}_{seed}_r{rate}_{arm}.npz"
                check(row["artifact"] == expected_name, f"Unexpected startup archive: {expected_name}")
                models[arm] = load_archive(directory/expected_name, row, config, EARLY, probes, 8*rate, rate, arm, raw_hashes)
            baseline = models[startup.ARMS[0]]
            truth = case.reference_fields(baseline, grids)
            for arm in startup.ARMS[1:]:
                item = pairs[name, seed, rate, arm]
                metrics = short_metrics(models[arm], baseline, grids, truth)
                compare_scores(metrics["fields"], item["fields"], keys, f"startup {name} {rate} {arm}")
                result_matches({k: v for k, v in metrics.items() if k != "fields"}, item, f"startup {name} {rate} {arm}")
                outcome = outcomes[str(rate)]
                outcome["compared"] += 1
                outcome["passed"] += metrics["passed"]
                summary = maxima[str(rate)]
                summary["field"] = max(summary["field"], *metrics["worst_fields"].values())
                summary["probe"] = max(summary["probe"], *(metrics["probes"][k]["total"]["relative_l2"] for k in PROBES))
    check(audit["outcomes"] == outcomes == OUTCOMES, "Startup failed/passing rates changed")
    modal_path = directory/"modal_audit.json"
    modal = base.read(modal_path)
    add_digest(inputs, modal_path)
    numeric_match(rebuild_modal_audit(directory), modal, "same-basis modal startup audit")
    check(len(modal["basis"]) == 4 and all(r["common_model_for_all_arms"] for r in modal["basis"]) and
          len(modal["results"]) == len(pairs), "Incomplete shared startup basis checks")
    mode_values = 0
    for row in modal["results"]:
        check(len(row["comparisons"]) == len(keys), "Incomplete modal score matrix")
        for item in row["comparisons"]:
            for region in ("full", "near"):
                count = len(item[region]["modal_energy"])
                check(count == 73, "Unexpected startup modal energy matrix size")
                mode_values += count
    return dict(status="passed", trajectories=len(trajectories), outcome_by_rate=outcomes, worst_by_rate=maxima,
                field_values_recomputed=len(pairs)*len(keys)*len(prior.METRICS),
                common_basis_groups=len(modal["basis"]), modal_energy_values_recomputed=mode_values)


def verify_rate128(repeat, grids, sources, inputs, raw_hashes, references):
    p, journal, report = [base.read(path) for path in (qualified.FROZEN, qualified.JOURNAL, qualified.SUMMARY)]
    template = base.read(qualified.TEMPLATE)
    times = EARLY+LATE
    check(p["status"] == "frozen" and journal["status"] == report["status"] == "complete" and
          template == p["template"] and tuple(p["checkpoints"]) == times and
          tuple(template["early_checkpoints"]) == EARLY and tuple(template["late_checkpoints"]) == LATE and
          (template["rate"], template["horizon"], template["probe_spacing"]) == (R128, 72, .125) and
          template["implementation_tolerance"] == GATE and template["late_physical_gate"] == PHYSICAL_GATE and
          tuple(template["references"]) == REFS and tuple(template["arms"]) == startup.ARMS and
          "not an isolated comparison" in template["scope"] and report["scope"] == template["scope"],
          "Rate128 is not the fixed accuracy-only qualification")
    digest = add_digest(inputs, qualified.FROZEN, journal["protocol_sha256"])
    check(report["protocol_sha256"] == digest, "Rate128 report used another protocol")
    add_digest(inputs, qualified.JOURNAL, report["runs_sha256"])
    add_digest(inputs, qualified.SUMMARY)
    add_digest(inputs, qualified.old.FROZEN, p["original_protocol_sha256"])
    check_sources(sources, p["source_hashes"])
    check_sources(sources, p["reference"]["source_hashes"])
    check_sources(inputs, p["reference"]["files"])
    check_sources(references, p["reference"]["reference_hashes"])
    prior_refs = repeat["inputs"]["reference_hashes"]
    check(all(prior_refs[name] == digest for name, digest in p["reference"]["reference_hashes"].items()),
          "Rate128 and original archive cited different reference trajectories")
    check(p["protocol"] == repeat["inputs"]["solve_protocol"], "Rate128 and original solve protocols differ")
    truth = prior.dual_truth(repeat, grids)
    expected = {(name, seed, arm) for name, seeds in SEEDS.items() for seed in seeds for arm in startup.ARMS}
    def key(job):
        return tuple(job[k] for k in ("name", "seed", "arm"))
    unique(p["jobs"], key, expected, "rate128 frozen job")
    raw = unique(journal["rows"], lambda row: key(row["job"]), expected, "rate128 raw record")
    scored = unique(report["rows"], lambda row: key(row["job"]), expected, "rate128 scored record")
    check([row["job"] for row in report["rows"]] == p["jobs"], "Rate128 score order differs")
    early_keys = {(grid, t) for grid in grids for t in EARLY}
    late_keys = {(grid, t) for grid in grids for t in LATE}
    probes = np.arange(0., 72.001, .125)
    maxima = dict(implementation=0., early_implementation=0., late_implementation=0., probe=0.,
                  late_velocity=0., late_vorticity=0.)
    accepted = 0
    for name, seeds in SEEDS.items():
        for seed in seeds:
            _, config = base.config(p["specs"][name], p["protocol"], seed)
            cohort = {}
            for arm in startup.ARMS:
                row = raw[name, seed, arm]
                stem = f"{name}_{seed}_{arm}"
                record_path = qualified.OUT/f"{stem}.json"
                add_digest(inputs, record_path)
                check(row["status"] == "complete" and row["artifact"] == f"{stem}.npz" and row == base.read(record_path),
                      f"Rate128 per-run record differs: {stem}")
                cohort[arm] = load_archive(qualified.OUT/row["artifact"], row, config, times, probes, 9216, R128, arm, raw_hashes)
            baseline = cohort[startup.ARMS[0]]
            fields = case.reference_fields(baseline, grids)
            for arm, sol in cohort.items():
                raw_row, row = raw[name, seed, arm], scored[name, seed, arm]
                label = f"rate128 {name} {seed} {arm}"
                check((row["artifact"], row["sha256"]) == (raw_row["artifact"], raw_row["sha256"]),
                      f"Qualification scores cite a different trajectory: {label}")
                implementation = case.score(sol, fields, grids)
                compare_scores(implementation, row["implementation_scores"], early_keys | late_keys, label)
                full_max = worst(implementation)
                early_max = worst([x for x in implementation if x["time"] in EARLY])
                late_max = worst([x for x in implementation if x["time"] in LATE])
                probe = {name: relative(getattr(sol, name), getattr(baseline, name))["relative_l2"] for name in PROBES}
                result_matches(dict(implementation_worst=full_max, early_implementation_worst=early_max,
                                    late_implementation_worst=late_max, full_horizon_probe_error=probe), row, label)
                late = case.Solution(sol.model, sol.states[len(EARLY):], sol.times[len(EARLY):], sol.probe_times,
                                     sol.transverse, sol.vorticity, sol.row)
                phys = {ref: case.score(late, ref_fields, grids) for ref, ref_fields in truth.items()}
                check(row["late_physical_scores"].keys() == set(REFS), f"Reference matrix differs: {label}")
                for ref, scores in phys.items():
                    compare_scores(scores, row["late_physical_scores"][ref], late_keys, f"{label}/{ref}")
                physical_max = {ref: worst(scores) for ref, scores in phys.items()}
                wake = base.wake(sol)
                implementation_pass = all(v is not None and math.isfinite(v) and 0 <= v <= GATE
                                          for v in (*full_max.values(), *probe.values()))
                physical_pass = all(max(metrics.values()) <= PHYSICAL_GATE for metrics in physical_max.values()) and wake[
                    "sustained_alternating_probe_signs"]
                result_matches(dict(late_physical_worst=physical_max, wake=wake, implementation_pass=implementation_pass,
                                    late_physical_pass=physical_pass, qualified=bool(implementation_pass and physical_pass)), row, label)
                accepted += bool(implementation_pass and physical_pass)
                for k, values in (("implementation", full_max), ("early_implementation", early_max),
                                  ("late_implementation", late_max), ("probe", probe)):
                    maxima[k] = max(maxima[k], *values.values())
                for field in ("velocity", "vorticity"):
                    k = "late_"+field
                    maxima[k] = max(maxima[k], *(v[m] for v in physical_max.values() for m in (field, "near_"+field)))
    check(journal["passes"] and report["passes"] and accepted == len(expected) == 18,
          "Rate128 early/late qualification no longer passes")
    return dict(status="passed", rate=R128, trajectories=len(expected), qualified=accepted,
                checkpoints=list(times), late_reference_names=list(REFS), worst=maxima,
                implementation_field_values_recomputed=len(expected)*(len(early_keys)+len(late_keys))*len(prior.METRICS),
                late_physical_field_values_recomputed=len(expected)*len(REFS)*len(late_keys)*len(prior.METRICS),
                full_horizon_probe_errors_recomputed=len(expected)*len(PROBES), isolated_speed_claim_supported=False)


def optional_short_replay(frozen, grids):
    protocol = frozen["inputs"]["solve_protocol"]
    specs = {s["name"]: s for s in protocol["variants"]}
    rows = []
    for name, seeds in SEEDS.items():
        _, config = base.config(specs[name], protocol, seeds[0])
        models = {arm: startup.solve(arm, config, R128, frozen["template"]["cuda_graph_steps"]) for arm in startup.ARMS}
        baseline = models[startup.ARMS[0]]
        truth = case.reference_fields(baseline, grids)
        for arm in startup.ARMS[1:]:
            metrics = short_metrics(models[arm], baseline, grids, truth)
            check(metrics["passed"], f"Optional rate128 T=8 implementation check failed: {name}/{arm}")
            rows.append(dict(name=name, seed=seeds[0], arm=arm, worst_fields=metrics["worst_fields"],
                             probes={k: metrics["probes"][k]["total"] for k in PROBES}))
    return rows


def verify(replay=False):
    sources, inputs, raw, references = {}, {}, {}, {}
    add_digest(sources, Path(__file__).resolve())
    grids = base.grid_set()
    frozen, archive = verify_prior_archives(inputs)
    short = verify_startup(frozen, grids, sources, inputs, raw)
    rate128 = verify_rate128(frozen, grids, sources, inputs, raw, references)
    check(len(raw) == 30, "Expected 12 startup and 18 complete rate128 trajectory hashes")
    rows = optional_short_replay(frozen, grids) if replay else []
    return dict(status="passed", previous_archive=archive, startup=short, rate128_qualification=rate128,
                adoption=dict(default_rate=128, rate112_deferred=True, isolated_speed_claim_supported=False),
                optional_rate128_short_replay=dict(status="passed" if replay else "not_run", rows=rows),
                trajectory_solves_run=6 if replay else 0, source_hashes=sources, evidence_hashes=inputs,
                raw_trajectory_hashes=raw, reference_trajectory_hashes=references)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Also run six optional T=8 rate128 implementation checks")
    parser.add_argument("--output", type=Path, help="Write the verification to a new JSON file")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    text = json.dumps(verify(args.replay), indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(text+"\n")
    print(text)
