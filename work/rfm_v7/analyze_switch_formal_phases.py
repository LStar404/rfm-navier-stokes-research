"""Decompose paired stage times from the monitored diagnostic cohort."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
OUT = HERE/"switch_formal_timing_output"
TARGET = OUT/"phase_diagnostics.json"
GPU_PREPARATION = ("host_setup_seconds", "runtime_init_seconds", "upload_seconds", "plan_seconds", "graph_setup_seconds")
GPU_PHASES = (*GPU_PREPARATION, "march_seconds", "download_seconds", "model_release_seconds", "unattributed_solver_seconds")
CPU_PHASES = ("setup_seconds", "after_setup_excluding_model_release_seconds", "model_release_seconds")
CLOCKS = ("solver_seconds", "fresh_process_return_seconds", "fresh_process_overhead_seconds")


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stats(values):
    return dict(mean=mean(values), median=median(values), minimum=min(values), maximum=max(values), samples=values)


def times(row, gpu):
    metadata = row["metadata"]
    measured = {key:row[key] for key in CLOCKS}
    measured["model_release_seconds"] = row["model_release_seconds"]
    if gpu:
        measured.update({key:metadata[key] for key in (*GPU_PREPARATION, "march_seconds", "download_seconds")})
        measured["pre_march_preparation_seconds"] = sum(measured[key] for key in GPU_PREPARATION)
        measured["unattributed_solver_seconds"] = measured["solver_seconds"]-sum(measured[k] for k in GPU_PHASES[:-1])
    else:
        measured["setup_seconds"] = metadata["setup_seconds"]
        measured["after_setup_excluding_model_release_seconds"] = (
            measured["solver_seconds"]-measured["setup_seconds"]-measured["model_release_seconds"])
    if not all(math.isfinite(value) for value in measured.values()):
        raise ValueError("Nonfinite phase measurement")
    return measured


def contrast(rows, family, a, b):
    gpu = a.startswith("cuda_")
    advance = "march_seconds" if gpu else "after_setup_excluding_model_release_seconds"
    pairs = []
    for block in range(1, 6):
        baseline, candidate = (rows[family, arm, block] for arm in (a, b))
        previous, next_ = (times(row, gpu) for row in (baseline, candidate))
        work_a = baseline["work"]["nonlinear_evaluations"]
        work_b = candidate["work"]["nonlinear_evaluations"]
        pairs.append(dict(block=block, seed=baseline["seed"], baseline=previous, candidate=next_,
                          delta_seconds={key:next_[key]-previous[key] for key in previous},
                          work=dict(baseline=work_a, candidate=work_b),
                          advancing_seconds_per_evaluation=dict(baseline=previous[advance]/work_a,
                                                               candidate=next_[advance]/work_b)))
    values = {key:[row["delta_seconds"][key] for row in pairs] for key in pairs[0]["delta_seconds"]}
    means = {key:mean(entries) for key, entries in values.items()}
    summed = sum(means[k] for k in (GPU_PHASES if gpu else CPU_PHASES))
    if not math.isclose(summed, means["solver_seconds"], abs_tol=1e-13):
        raise ArithmeticError("Phase means do not sum to the paired solver delta")

    def ratio(key):
        values = [r["baseline"][key]/r["candidate"][key] for r in pairs]
        return dict(stats(values), candidate_wins=sum(value > 1 for value in values), blocks=len(values))

    return dict(baseline_arm=a, candidate_arm=b, mean_delta_seconds=means,
                paired_delta_seconds={key:stats(entries) for key, entries in values.items()},
                solver_speed_ratio=ratio("solver_seconds"), fresh_speed_ratio=ratio("fresh_process_return_seconds"),
                advancing_seconds_per_evaluation={side:stats([r["advancing_seconds_per_evaluation"][side] for r in pairs])
                                                 for side in ("baseline", "candidate")},
                blocks=pairs)


def main(target):
    if target.exists():
        raise FileExistsError(target)
    timing, summary, monitor, protocol = (read(name) for name in ("timing.json", "summary.json", "isolation.json", "protocol.json"))
    files = {name:digest(OUT/name) for name in ("timing.json", "summary.json", "isolation.json", "protocol.json")}
    if (summary["status"] != "diagnostic_only" or summary["isolated_speed_claim_supported"] or
            monitor["status"] != "completed_with_observed_external_activity" or timing["status"] != "complete" or
            any(summary[key] != files[name] for key, name in (("timing_sha256", "timing.json"),
                ("monitor_sha256", "isolation.json"), ("protocol_sha256", "protocol.json")))):
        raise ValueError("Expected the verified completed externally affected cohort")
    rows = {(r["name"],r["arm"],r["block"]):r for r in timing["rows"]}
    if len(rows) != len(timing["rows"]) or len(rows) != 60 or not all(r["qualified"] for r in rows.values()):
        raise ValueError("Expected all sixty numerically qualified diagnostic samples")
    contrasts = {}
    for name in protocol["template"]["families"]:
        contrasts[name] = dict(
            cpu_operator_128=contrast(rows,name,"cpu_recurrence128","cpu_native128"),
            cpu_schedule=contrast(rows,name,"cpu_native128","cpu_native_switch"),
            gpu_schedule=contrast(rows,name,"cuda_graph128","cuda_graph_switch"),
        )
    fd = protocol["template"]["fd_name"]
    fd_precompute = stats([rows[fd,"cuda_graph128",block]["metadata"]["host_linear_precompute_seconds"] for block in range(1,6)])
    limit = protocol["isolation"]["thresholds"]["unexpected_python_cpu_seconds"]
    children = {}
    for row in timing["rows"]:
        children.setdefault(row["child_pid"], []).append(row["index"])
    marked, unmatched, child_matches = [], [], []
    for sample in monitor["samples"]:
        target_seconds = sample["target_elapsed_seconds"]
        entries = [r for r in sample["external_python"] if r["label"] == "python" and r["cpu_seconds"] > limit]
        if entries:
            marked.append(target_seconds)
        if any(entry["pid"] not in children for entry in entries):
            unmatched.append(target_seconds)
        child_matches.extend(dict(sample_target_elapsed_seconds=target_seconds, child_pid=entry["pid"],
                                  recorded_run_indices=children[entry["pid"]], cpu_seconds=entry["cpu_seconds"])
                             for entry in entries if entry["pid"] in children)
    result = dict(status="diagnostic_only", isolated_speed_claim_supported=False, numerical_rows=60,
                  source_sha256=digest(Path(__file__)), input_sha256=files, contrasts=contrasts,
                  monitor_marked_external_python_sample_targets_seconds=marked,
                  monitor_marked_unmatched_cohort_child_sample_targets_seconds=unmatched,
                  monitor_marked_entries_matching_cohort_child=child_matches,
                  fd176_cuda_host_linear_precompute_seconds=fd_precompute,
                  conventions={
                      "delta":"Candidate minus baseline within the same block. Means add across solver phases. Paired medians and ranges describe spread.",
                      "gpu_host_setup":"Spatial model and phase operator assembly or precontraction are in host setup. Galerkin has no separate measured precompute phase. FD176 dense precompute is measured separately.",
                      "gpu_runtime_init":"CUDA context, module compile and load, library bindings and kernel registration are grouped. The switched runtime additionally loads the bridge module. No isolated JIT stopwatch was recorded.",
                      "gpu_graph_and_march":"Both GPU arms use graph chunks of 24 and two priming FFT batches. The switched implementation captures two segments. March includes launch scheduling and final synchronization.",
                      "cpu_after_setup":"Solver time after setup and before release of the returned model, including marching, observations and unseparated work. The fixed-rate operator contrast combines precontraction and mode-contiguous C/AVX2.",
                      "fresh_clock":"Popen to flushed completion marker includes interpreter, imports and brief bookkeeping. NPZ serialization follows the acknowledgement. System and driver caches are retained.",
                      "monitor_labels":"Count original above-threshold labels separately from entries that do not match a recorded cohort child PID. PID matching identifies a monitor label discrepancy without verifying all other process identities. Original monitoring records and isolation acceptance remain unchanged.",
                      "publication":"External Python load was observed. Keep all five observations and use these contrasts as descriptive diagnostics of this cohort."
                  })
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(target.name, digest(target), len(marked), len(unmatched), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=TARGET)
    main(cli.parse_args().output)
