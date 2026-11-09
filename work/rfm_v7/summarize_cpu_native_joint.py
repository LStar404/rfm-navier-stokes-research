"""Produce compact timing, accuracy, and work-count tables from a 50-run cohort."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


OUT=Path(__file__).resolve().parent/"cpu_native_joint_output"
PHASES=("setup_seconds","host_setup_seconds","host_linear_precompute_seconds","runtime_init_seconds",
        "upload_seconds","plan_seconds","graph_setup_seconds","march_seconds","download_seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def stats(values):
    values=list(values)
    if not values or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError("Invalid timing samples")
    return dict(median=median(values),minimum=min(values),maximum=max(values),samples=values)


def fixed_window_status(directory,parent_pid):
    protocol=directory/"isolation_protocol.json"
    path=directory/"isolation_repeat.json"
    execution=directory/"execution.json"
    if not all(p.exists() for p in (protocol,path,execution)):
        return dict(status="not_verifiably_isolated",reason="missing_fixed_window_evidence")
    fixed=read(protocol)
    record=read(path)
    worker=read(execution)
    supported={"completed_no_sampled_external_activity","completed_with_observed_external_activity",
               "completed_outside_fixed_window"}
    if (record["pid"]!=parent_pid or worker["pid"]!=parent_pid or record["exit_code"]!=0 or
            record["status"] not in supported or worker["status"]!="complete" or
            record["isolation_protocol_sha256"]!=digest(protocol) or
            worker["isolation_protocol_sha256"]!=digest(protocol) or record["execution_sha256"]!=digest(execution) or
            fixed["original_protocol_sha256"]!=digest(directory/"protocol.json") or
            record["window_seconds"]!=fixed["window_seconds"] or
            record["sampling_interval_seconds"]!=fixed["sample_seconds"] or
            len(record["preflight"])!=fixed["preflight_windows"] or
            [r["target_elapsed_seconds"] for r in record["samples"]]!=list(
                range(fixed["sample_seconds"],fixed["window_seconds"]+1,fixed["sample_seconds"]))):
        raise ValueError("Fixed-window observation evidence differs from its protocol")
    expected=("completed_outside_fixed_window" if not (record["sample_coverage_pass"] and
                                                       record["solver_finished_within_window"])
              else "completed_with_observed_external_activity" if record["external_activity_detected"]
              else "completed_no_sampled_external_activity")
    if record["status"]!=expected:
        raise ValueError("Fixed-window acceptance contradicts the recorded monitoring conditions")
    return dict(status=expected,start_utc=record["start_utc"],end_utc=record["end_utc"],
                planned_end_utc=record["planned_end_utc"],fixed_window_seconds=fixed["window_seconds"],
                record=path.name,record_sha256=digest(path),isolation_protocol_sha256=digest(protocol),
                execution_sha256=digest(execution),initial_gpu_percent=record["initial_gpu_percent"],
                final_gpu_percent=record["final_gpu_percent"],samples=len(record["samples"]),
                sampling_interval_seconds=fixed["sample_seconds"],
                scope="Ten-second process samples may miss external short-lived or non-Python activity and do not identify other GPU contexts.")


def isolation_status(parent_pid,directory=OUT):
    if (directory/"isolation_protocol.json").exists():
        return fixed_window_status(directory,parent_pid)
    records=[(p,read(p)) for p in sorted(directory.glob("isolation_*.json"))]
    matches=[(p,r) for p,r in records if r.get("pid")==parent_pid and r.get("exit_code")==0]
    if len(records)!=1 or len(matches)!=1:
        return dict(status="not_verifiably_isolated",recorded_sessions=len(records))
    path,r=matches[0]
    if r.get("status") not in ("completed_with_observed_external_activity","completed_no_sampled_external_activity"):
        return dict(status="not_verifiably_isolated",recorded_sessions=len(records))
    return dict(status=r["status"],start_utc=r["start_utc"],end_utc=r["end_utc"],
                record=path.name,record_sha256=digest(path),
                initial_gpu_percent=r["initial_gpu_percent"],final_gpu_percent=r["final_gpu_percent"],
                samples=len(r["samples"]),sampling_interval_seconds=r["sampling_interval_seconds"],
                scope="Sampled process counters cannot rule out activity that starts and ends between samples.")


def summarize(directory=OUT):
    source=directory/"timing.json"
    timing=read(source)
    protocol=directory/"protocol.json"
    frozen=read(protocol)
    if timing["status"]!="complete" or timing["protocol_sha256"]!=digest(protocol):
        raise ValueError("The complete isolated timing cohort is required")
    rows=timing["rows"]
    keys=("block","name","arm","rate","seed")
    if (len(rows)!=len(frozen["schedule"]) or len(rows)!=50 or
            [tuple(r[k] for k in keys) for r in rows]!=[tuple(r[k] for k in keys) for r in frozen["schedule"]]):
        raise ValueError("Timed observations differ from frozen schedule")
    result=[]
    for group in timing["groups"]:
        same=[r for r in rows if (r["name"],r["arm"])==(group["name"],group["arm"])]
        valid=[r for r in same if r["implementation_pass"] and r["numerical_pass"]]
        if len(same)!=5 or group["passes"]!=(len(valid)==5):
            raise ValueError("Inconsistent timing group")
        rate=same[0]["rate"]
        steps=rate*frozen["template"]["horizon"]
        phases={key:stats(r["metadata"][key] for r in valid) for key in PHASES
                if valid and all(key in r["metadata"] for r in valid)}
        if "setup_seconds" in phases:
            phases["solve_minus_setup_seconds"]=stats(r["metadata"]["seconds"]-r["metadata"]["setup_seconds"]
                                                        for r in valid)
        result.append(dict(name=group["name"],arm=group["arm"],rate=rate,count=len(same),passes=len(valid),
            seeds=[r["seed"] for r in same],end_to_end_seconds=stats(r["seconds"] for r in valid) if valid else None,
            phases_seconds=phases,worst=group["worst"],steps=steps,nonlinear_evaluations=steps-1,
            marching_fft_batches=2*(steps-1)))
    if len(result)!=10 or len({(r["name"],r["arm"]) for r in result})!=10:
        raise ValueError("Expected ten distinct arm groups")
    failures=[dict({k:r[k] for k in keys},failure=r.get("failure"),numerical_pass=r["numerical_pass"],
                   implementation_pass=r["implementation_pass"])
              for r in rows if not (r["implementation_pass"] and r["numerical_pass"])]
    isolation=isolation_status(timing.get("parent_pid"),directory)
    clean=isolation["status"]=="completed_no_sampled_external_activity"
    status="completed_with_failures" if failures else "passed" if clean else isolation["status"]
    return dict(status=status,isolation=isolation,isolated_speed_claim_supported=bool(clean and not failures),
                timing_sha256=digest(source),
                protocol_sha256=digest(protocol),runs=len(rows),failures=failures,groups=result,paired=timing["ratios"],
                scope="Five isolated randomized blocks. Ranges are min/max. All FFT counts exclude setup and priming. "
                      "The CUDA rate pair changes only the timestep and fixed-horizon step count. CPU operator pairs hold the rate fixed.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory",type=Path,default=OUT)
    args=cli.parse_args()
    result=summarize(args.directory)
    with (args.directory/"summary.json").open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(result["status"])
    print("Method | Arm | Rate | Passed | Median s | Min–max s")
    for group in result["groups"]:
        stat=group["end_to_end_seconds"]
        duration=(f'{stat["median"]:.4f} | {stat["minimum"]:.4f}–{stat["maximum"]:.4f}' if stat else "— | —")
        print(f'{group["name"]} | {group["arm"]} | {group["rate"]} | {group["passes"]}/5 | {duration}')
