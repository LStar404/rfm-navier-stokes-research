"""Postprocess the actual flat row schema without rewriting frozen timing evidence."""

import argparse
import json
from pathlib import Path

import prepared_formal_timing as frozen


base, np, require = frozen.base, frozen.np, frozen.require
JOB_FIELDS = ("block","name","arm","seed")


def verify_trajectory(path, raw, job, protocol):
    config = frozen.configuration(job,protocol)
    design = protocol["design"]
    times = design["physical_times"]
    probes = np.arange(0.,design["horizon"]+.01*design["probe_spacing"],design["probe_spacing"])
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files) == {"states","times","probe_times","transverse","vorticity","config","metadata"} and
                json.loads(str(saved["config"])) == json.loads(json.dumps(config)) and
                json.loads(str(saved["metadata"])) == raw["metadata"], "Raw NPZ configuration or metadata differs")
        require(np.array_equal(saved["times"],times) and np.array_equal(saved["probe_times"],probes) and
                all(saved[name].shape == (577,3) and np.isfinite(saved[name]).all()
                    for name in ("transverse","vorticity")), "Raw absolute observations or full probes differ")
        states = saved["states"]
        require(states.ndim == 3 and states.shape[0] == len(times) and states.dtype == np.complex128 and
                np.isfinite(states).all(), "Raw trajectory is missing finite FP64 complex observations")


def normalize_rows(output, protocol, timing):
    schedule = protocol["schedule"]
    require(timing["status"] == "complete" and len(timing["rows"]) == len(schedule),
            "This postprocessor requires the actual complete cohort and its frozen schedule")
    normalized, inputs = [], {}
    for i,(raw,job) in enumerate(zip(timing["rows"],schedule,strict=True)):
        require(set(job) == set(JOB_FIELDS) and "job" not in raw and type(raw["index"]) is int and raw["index"] == i and
                all(type(raw[key]) is type(job[key]) and raw[key] == job[key] for key in JOB_FIELDS),
                "Flat timing row does not match its frozen schedule slot")
        stem = frozen.previous.stem(job)
        sidecar = output/(stem+".json")
        require(base.read(sidecar) == raw, "Unmodified timing row differs from its raw sidecar")
        inputs[sidecar.name] = base.digest(sidecar)
        if "artifact" in raw:
            require(raw["artifact"] == stem+".npz", "Saved timing trajectory basename differs")
            artifact = output/raw["artifact"]
            inputs[artifact.name] = base.digest(artifact)
            require(inputs[artifact.name] == raw["sha256"], "Saved timing trajectory hash differs")
            verify_trajectory(artifact,raw,job,protocol)
        require(type(raw["qualified"]) is bool and (not raw["qualified"] or
                raw["status"] == "qualified" and {"implementation","artifact",*frozen.CLOCKS} <= raw.keys()),
                "Qualified slot has missing numerical or clock evidence")
        view = dict(raw,job={key:raw[key] for key in JOB_FIELDS})
        if "implementation" in raw:
            frozen.verify_numerical_row(view,protocol)
        normalized.append(view)
    require(timing["numerical_qualified"] == all(row["qualified"] for row in normalized),
            "Raw timing journal qualification flag differs")
    require(set(inputs) == {path.name for glob in ("b*.json","b*.npz") for path in output.glob(glob)},
            "Raw folder has missing or extra timed rows")
    return normalized,inputs


def window_findings(monitor):
    unresolved = [dict(target_elapsed_seconds=s["target_elapsed_seconds"],unresolved_python=s["snapshot"]["unresolved_python"])
                  for s in monitor["samples"] if s["snapshot"]["unresolved_python"]]
    return dict(sample_coverage_pass=monitor["sample_coverage_pass"],
                solver_finished_within_window=monitor["solver_finished_within_window"],
                external_activity_detected=monitor["external_activity_detected"],
                external_activity_targets_seconds=[s["target_elapsed_seconds"] for s in monitor["samples"]
                                                   if s["external_activity_detected"]],
                unresolved_process_windows=unresolved)


def summarize(output,scale,*,check_only=False):
    files = frozen.paths(output)
    protocol,protocol_sha = frozen.read_protocol(files,scale)
    require(check_only or not files["summary"].exists(), "Preserve any existing derived summary")
    keys = ("protocol","timing","execution","identity_ledger","coordinator_admission","isolation")
    checks = {key+"_sha256":base.digest(files[key]) for key in keys}
    t,e,l,a,m = (base.read(files[key]) for key in keys[1:])
    require(all(record["protocol_sha256"] == protocol_sha and record["run_token"] == m["run_token"]
                for record in (t,e,l,a,m)) and
            all(record["scale"] == scale for record in (t,e,l,m)) and
            all(m[key+"_sha256"] == checks[key+"_sha256"] for key in ("execution","identity_ledger","coordinator_admission","timing")) and
            e["identity_ledger_sha256"] == checks["identity_ledger_sha256"] and
            frozen.identity(a["coordinator"]) == frozen.identity(m["coordinator"]) == frozen.identity(e) and
            a["schema"] == frozen.SCHEMA and a["status"] == "admitted" and e["status"] == "complete",
            "Protocol, admission, journal or monitor bindings differ")
    frozen.previous.check_hashes(protocol["source_hashes"])
    rows,raw_inputs = normalize_rows(output,protocol,t)
    journal_view = dict(t,rows=rows)
    status = frozen.monitor_status(protocol,m,e,l,journal_view,a)
    numerical = all(row["qualified"] for row in rows)
    isolated = bool(numerical and status == "completed_no_sampled_external_activity")
    result = dict(status="passed" if isolated else "diagnostic_only",scale=scale,rows=len(rows),monitor_status=status,
                  numerical_qualified=numerical,isolated_speed_claim_supported=isolated,**checks,
                  summary_postprocessor_sha256=base.digest(Path(__file__)),
                  source_schema=dict(raw_job_fields=list(JOB_FIELDS),job_view="in-memory only",
                                     frozen_summary_defect="execute serialized flat job fields; its summarize expected row['job']"),
                  raw_evidence_sha256=raw_inputs,isolation_findings=window_findings(m),
                  **frozen.contrasts(protocol,rows,isolated))
    require(all(base.digest(files[k]) == checks[k+"_sha256"] for k in keys) and
            all(base.digest(output/name) == digest for name,digest in raw_inputs.items()),
            "Evidence changed while producing the derived summary")
    if not check_only:
        with files["summary"].open("x",encoding="utf-8") as output_stream:
            json.dump(result,output_stream,indent=2,allow_nan=False)
    print(scale,status,len(rows),"numerical",numerical,"isolated",isolated,flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale",required=True,choices=("mid","high"))
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--check-only",action="store_true")
    args = parser.parse_args()
    summarize(args.output_dir.resolve(),args.scale,check_only=args.check_only)
