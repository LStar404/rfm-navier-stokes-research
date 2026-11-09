"""Hash-bound inputs and complete preregistration for prepared CUDA timing."""

from pathlib import Path
import random

import fd184_band_followup as band
import prepared_qualification as qual
import switch_formal_timing as previous


base, fd, switch = previous.base, previous.fd_qual, previous.switch
HERE = base.HERE
TEMPLATE = HERE / "prepared_formal_timing_template.json"
require = previous.require


def bind(sources, path, digest=None):
    item = previous.path_record(path)
    require(digest is None or item["sha256"] == digest, f"Archive hash changed: {item['path']}")
    return previous.add_record(sources, item)


def group(sources, directory):
    protocol, runs, report = (directory / name for name in ("protocol.json", "runs.json", "qualification.json"))
    p, r, q = (base.read(path) for path in (protocol, runs, report))
    ph = bind(sources, protocol)["sha256"]
    rh = bind(sources, runs, q["runs_sha256"])["sha256"]
    bind(sources, report)
    require(p["status"] == "frozen" and r["status"] == q["status"] == "complete" and
            r["protocol_sha256"] == q["protocol_sha256"] == ph and q["runs_sha256"] == rh,
            f"Incomplete qualification archive: {directory.name}")
    return p, r, q


def artifact(sources, directory, row, raw_records=None, flat=False):
    item = previous.artifact_record(row, directory, sources)
    raw_path = (base.WORK / item["path"]).with_suffix(".json")
    bind(sources, raw_path, raw_records[raw_path.name] if raw_records is not None else None)
    raw = base.read(raw_path)
    require(raw["status"] in ("complete", "qualified") and
            all(raw[k] == row[k] for k in ("artifact", "sha256")), "Raw trajectory identity differs")
    if flat:
        require(all(raw[k] == row[k] for k in ("name", "seed", "arm", "block")), "Raw timing identity differs")
    else:
        require(raw["job"] == row["job"], "Raw qualification identity differs")
    return item


def uniform_evidence(scale, ids, sources, design):
    if scale == "high":
        out = HERE / "prepared_uniform_high_qualification_output"
        p, r, q = group(sources, out)
        rows = q["rows"]
        require(q["passes"] and len(rows) == len(ids) and all(row["qualified"] for row in rows),
                "High uniform GPU did not independently qualify every identity")
        for row in rows:
            require(row["job"]["resolution"] == "high" and row["job"]["arm"] == "uniform" and
                    row["job"] in p["jobs"] and len(row["implementation"]["scores"]) == 57 and
                    all(len(v) == 57 for v in row["physical_scores"].values()),
                    "High uniform GPU qualification observations differ")
        picked = {(row["job"]["name"], row["job"]["seed"]): artifact(sources, out, row, r["raw_records"])
                  for row in rows}
    else:
        out = previous.OUT
        p, r, q = (base.read(out / name) for name in ("protocol.json", "timing.json", "summary.json"))
        bind(sources, out / "protocol.json", q["protocol_sha256"])
        bind(sources, out / "timing.json", q["timing_sha256"])
        bind(sources, out / "summary.json")
        require(p["template"]["physical_times"] == design["physical_times"] and
                r["protocol_sha256"] == q["protocol_sha256"] and r["status"] == "complete" and q["numerical_qualified"],
                "Mid uniform numerical archive differs")
        picked = {}
        for row in r["rows"]:
            ident = row["name"], row["seed"]
            if row["arm"] == "cuda_graph128" and ident in ids and ident not in picked:
                require(row["qualified"] and all(len(v) == 57 for v in row["physical_scores"].values()),
                        "Mid uniform numeric trajectory omitted a qualification observation")
                picked[ident] = artifact(sources, out, row, flat=True)
    require(set(picked) == set(ids), "Uniform qualification does not cover every timed identity")
    return picked


def fd_evidence(sources, scale, design):
    p, r, q = group(sources, fd.OUT)
    for name, sha in p["source_hashes"].items():
        if name != qual.ADMINISTRATIVE:
            previous.add_record(sources, dict(path=name,sha256=sha))
    require(p["checkpoints"] == design["physical_times"], "FD qualification observations differ")
    result = {}
    for spec in scale["fd"]:
        name = spec["name"]
        rows = {row["job"]["arm"]:row for row in q["rows"] if row["job"]["name"] == name}
        cpu = rows["cpu_buffered"]
        require(cpu["qualified"] and cpu["job"]["rate"] == spec["rate"] and cpu["job"]["seed"] == spec["seed"] and
                cpu["job"] in p["jobs"] and any(raw["artifact"] == cpu["artifact"] and raw["sha256"] == cpu["sha256"]
                                                for raw in r["rows"]), "FD CPU control is unqualified")
        control = artifact(sources, fd.OUT, cpu)
        if spec["linear"] == "dense":
            gpu = rows["cuda_graph"]
            require(gpu["qualified"] and gpu["metadata"]["cuda_linear_solve"] == "dense" and
                    any(raw["artifact"] == gpu["artifact"] and raw["sha256"] == gpu["sha256"] for raw in r["rows"]),
                    "FD dense GPU is unqualified")
            evidence = artifact(sources, fd.OUT, gpu)
        else:
            bp, gpu = base.read(band.PROTOCOL), base.read(band.RESULT)
            bind(sources, band.PROTOCOL, gpu["protocol_sha256"])
            bind(sources, band.RESULT)
            bind(sources, band.RAW_RECORD, gpu["raw_record_sha256"])
            for dependency, sha in bp["source_hashes"].items():
                if dependency != qual.ADMINISTRATIVE:
                    previous.add_record(sources, dict(path=dependency,sha256=sha))
            require(bp["status"] == "frozen" and bp["control"] == control and gpu["qualified"] and
                    gpu["control_sha256"] == control["sha256"] and gpu["metadata"]["cuda_linear_solve"] == "band",
                    "FD band GPU is unqualified")
            evidence = artifact(sources, band.OUT, gpu)
        result[name] = dict(cpu=control, gpu=evidence, config=fd.config(cpu["job"], p["template"]), **spec)
    return result


def registrations(scale, design, components):
    family = scale["rfm"], scale["legendre"]
    arms = scale["timing_arms"]
    fd_names = [s["name"] for s in scale["fd"]]
    groups = [(name, arm) for name in family for arm in arms] + [(name, "fd") for name in fd_names]
    comparisons = []

    def pair(kind, baseline, candidate):
        require(baseline in groups and candidate in groups, "Unregistered comparison arm")
        comparisons.append(dict(kind=kind, baseline=list(baseline), candidate=list(candidate)))

    factorials = []
    for name in family:
        if scale["name"] == "mid":
            for kind, a, b in (("module_at_legacy_host", "switch_old", "switch_module"),
                               ("module_at_shared_host", "switch_host", "switch_both"),
                               ("host_at_legacy_module", "switch_old", "switch_host"),
                               ("host_at_bundled_module", "switch_module", "switch_both")):
                pair(kind, (name,a), (name,b))
            factorials.append(dict(name=name, response="seconds", components=components,
                                   coefficient={"switch_old":1,"switch_module":-1,"switch_host":-1,"switch_both":1}))
        pair("combined_preparation", (name,"switch_old"), (name,"switch_both"))
        pair("combined_schedule", (name,"uniform"), (name,"switch_both"))
        for arm in arms:
            for fd_name in fd_names:
                pair("same_cohort_fd_control", (fd_name,"fd"), (name,arm))
    for arm in arms:
        pair("legendre_rfm", (family[1],arm), (family[0],arm))
    if len(fd_names) == 2:
        pair("fd_radial_scale", (fd_names[0],"fd"), (fd_names[1],"fd"))
    order = random.Random(scale["order_seed"])
    schedule = []
    for block, rfm_seed in enumerate(design["rfm_seeds"], 1):
        jobs = [dict(block=block,name=name,arm=arm,seed=rfm_seed if name==family[0] else design["deterministic_seed"])
                for name, arm in groups]
        order.shuffle(jobs)
        schedule.extend(jobs)
    require(len(schedule) == scale["timing_rows"] and len({tuple(j.items()) for j in schedule}) == len(schedule),
            "Prepared cohort schedule is incomplete")
    return dict(groups=[list(v) for v in groups],comparisons=comparisons,factorials=factorials,schedule=schedule)


def make_protocol(scale_name, background, *, validate_hashes=True):
    template = base.read(TEMPLATE)
    design = base.read(base.WORK / template["design"])
    scale = next(s for s in design["scales"] if s["name"] == scale_name)
    sources = {}
    p, r, q = group(sources, qual.OUT)
    templates = (TEMPLATE, base.WORK / template["design"], Path(__file__).resolve(),
                 HERE/"prepared_formal_timing.py", HERE/"prepared_process_identity.py",
                 HERE/"prepared_process_snapshot.cs", HERE/"run_prepared_timing_isolated.ps1",
                 HERE/"prepared_process_identity_audit.json", HERE/"verify_prepared_process_identity.ps1",
                 HERE/"switch_formal_timing.py", HERE/"prepared_march_source_verification.json")
    for path in templates:
        bind(sources, path)
    audit = base.read(HERE/"prepared_process_identity_audit.json")
    audit_sources = ("prepared_process_identity.py","prepared_process_snapshot.cs",
                     "run_prepared_timing_isolated.ps1","verify_prepared_process_identity.ps1")
    require(audit["status"] == "passed" and set(audit["source_sha256"]) == set(audit_sources) and
            all(audit["source_sha256"][name] == sources[f"rfm_v7/{name}"] for name in audit_sources) and
            audit["owned_and_retained_terminal_identity_and_counters_exact"] and
            audit["mismatched_ready_and_return_rejected_without_partial_exemption"],
            "Exact Windows transport prerequisites differ from audited sources")
    march = base.read(HERE/"prepared_march_source_verification.json")
    require(march["status"] == "passed" and march["body_whitelist"]["restored_byte_identical"],
            "Prepared march source comparison failed")
    for name,sha in march["input_sha256"].items():
        bind(sources,HERE/name,sha)
    for name, sha in p["source_hashes"].items():
        if name != qual.ADMINISTRATIVE:
            previous.add_record(sources, dict(path=name,sha256=sha))
    require(q["low_failure_control_expected"] and len(q["rows"]) == 26 and scale_name in q["timing_eligible_scales"] and
            [row["job"] for row in q["rows"]] == p["jobs"] and
            design["physical_times"] == p["template"]["physical_times"] and len(design["physical_times"]) == 19 and
            all(design[k] == p["template"][k] for k in ("horizon","switch_time","probe_spacing","physical_gate",
                                                       "implementation_gate","reference_budget_fraction","references")) and
            design["cuda_graph_steps"] == p["template"]["graph_steps"] and
            design["rfm_seeds"] == p["template"]["rfm_seeds"] and len(design["rfm_seeds"]) == 5 and
            design["deterministic_seed"] == p["template"]["legendre_seed"],
            "Prepared qualification design or independent convergence gate differs")
    schedule = p["schedules"][scale_name]
    require([schedule["startup_rate"],schedule["late_rate"]] == scale["rates"] and
            len(schedule["checkpoints"]) == 21, "Qualified switched step schedule differs")
    identities = [(scale["rfm"], seed) for seed in design["rfm_seeds"]] + [(scale["legendre"], design["deterministic_seed"])]
    uniform = uniform_evidence(scale_name, identities, sources, design)
    controls = []
    for row in p["controls"]:
        if row["resolution"] == scale_name:
            ident = row["name"], row["seed"]
            controls.append(dict(row, uniform_gpu=uniform[ident]))
    require(len(controls) == len(identities) and {(x["name"],x["seed"]) for x in controls} == set(identities),
            "Prepared same-model controls are incomplete")
    selected = []
    for row in q["rows"]:
        job = row["job"]
        if job["resolution"] != scale_name:
            continue
        require(row["qualified"] and all(row[k] for k in ("implementation_pass","physical_pass","guard_pass")) and
                all(len(value["scores"]) == 63 for value in row["implementation"].values()),
                "Prepared GPU is unqualified")
        selected.append(dict(job=job,artifact=artifact(sources, qual.OUT, row, r["raw_records"])))
    require({(s["job"]["name"],s["job"]["seed"],s["job"]["preparation"]) for s in selected} ==
            {(name,seed,template["prepared_arms"][arm]) for name,seed in identities for arm in scale["new_accuracy_arms"]},
            "A timed prepared arm is missing from qualification")
    fd_controls = fd_evidence(sources, scale, design)
    ref = bind(sources, switch.FROZEN)
    isolation = dict(window_seconds=template["fixed_window_seconds"][scale_name],sample_seconds=design["sample_seconds"],
                     preflight_windows=design["preflight_windows"],thresholds=design["isolation_thresholds"],
                     known_background=[{k:background[k] for k in ("pid","birth_filetime_100ns")}])
    require(background["pid"] == template["background_server_pid"] and
            background["birth_filetime_100ns"] == template["background_birth_filetime_100ns"] and
            isolation["window_seconds"] == scale["proposed_window_seconds"], "Windows observation design differs")
    if validate_hashes:
        previous.check_hashes(sources)
    return dict(status="frozen",scale=scale_name,design=design,template=template,scale_design=scale,
                step_schedule=schedule,gal_controls=controls,fd_controls=fd_controls,prepared_evidence=selected,
                reference_protocol=ref,specs={name:p["specs"][name] for name,_ in identities},
                solve_protocol=p["solve_protocol"],source_hashes=sources,isolation=isolation,
                **registrations(scale,design,template["gpu_factorial_components"]))
