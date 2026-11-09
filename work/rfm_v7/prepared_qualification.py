"""Qualify prepared CUDA variants against frozen full-trajectory controls."""

import argparse
import json
import math
import os
from pathlib import Path

import multiscale_switch_qualification as outer
import switch_qualification as mid
from galerkin_cuda_prepared import PREPARATIONS, solve_cuda_prepared
from verify_switch_qualification import absolute_reference, checksums, require

base, case, np = mid.base, mid.case, mid.np
HERE = mid.HERE
OUT = HERE / "prepared_qualification_output"
TEMPLATE = HERE / "protocol_prepared_qualification.json"
FROZEN, RUNS, SUMMARY = (OUT / name for name in ("protocol.json", "runs.json", "qualification.json"))
MODEL = mid.switched.RecurrenceGalerkin
ADMINISTRATIVE = "rfm_v7/operator_round_checkpoint.json"
PROBES = ("transverse", "vorticity")


def key(job):
    return tuple(job[k] for k in ("resolution", "name", "seed"))


def record(path, digest=None):
    name, actual = mid.fingerprint(path)
    require(digest is None or digest == actual, f"Archive hash differs for {name}")
    return dict(path=name, sha256=actual)


def verify_inputs(source_hashes):
    checksums({name:sha for name,sha in source_hashes.items() if name != ADMINISTRATIVE})


def archived_controls(source, summary, identity):
    scale, name, seed = identity
    rows = [r for r in summary["rows"] if r["job"]["name"] == name and r["job"]["seed"] == seed and
            r["job"].get("resolution", "mid") == scale and r["job"]["arm"] in ("cpu_recurrence", "cuda_graph")]
    require(len(rows) == 2 and {r["job"]["arm"] for r in rows} == {"cpu_recurrence", "cuda_graph"} and
            all(r["qualified"] == (scale != "low") and r["implementation_pass"] for r in rows),
            f"Archive did not give the expected qualification for {identity}")
    controls = {r["job"]["arm"]:record(source.OUT / r["artifact"],r["sha256"]) for r in rows}
    anchors = [r for r in summary["anchors"] if (r.get("resolution", "mid"),r["name"],r["seed"]) == identity]
    require(len(anchors) == 1, f"Missing or duplicate uniform anchor for {identity}")
    anchor = anchors[0]
    if scale == "mid":
        first = record(source.OUT / anchor["artifact"],anchor["sha256"])
        last = anchor["late_source"]
    else:
        first = record(source.OUT / anchor["early_artifact"],anchor["early_sha256"])
        last = anchor["late"]
    last_path = base.WORK / last["path"] if "path" in last else source.OUT / last["artifact"]
    return dict(resolution=scale,name=name,seed=seed,early=first,late=record(last_path,last["sha256"]),**controls)


def prerequisites(pilot_evidence, kernel_evidence):
    pilot = base.read(pilot_evidence)
    plan_path,runs_path = (pilot_evidence.parent / name for name in ("protocol.json","runs.json"))
    plan,runs = base.read(plan_path),base.read(runs_path)
    rows = runs["rows"]
    require(pilot["status"] == "passed" and pilot["rows"] == pilot["passed"] == len(rows) == 24 and
            pilot["protocol_sha256"] == base.digest(plan_path) and pilot["runs_sha256"] == base.digest(runs_path) and
            plan["status"] == "frozen" and runs["status"] == "complete" and
            len({(r["name"],r["seed"],r["startup_rate"],r["late_rate"],r["preparation"]) for r in rows}) == 24 and
            all(r["passed"] and set(r["comparisons"]) == {"states","times","probe_times",*PROBES} and
                all(v["bitwise"] and v["max_abs"] == 0 for v in r["comparisons"].values()) for r in rows),
            "T9 preparation comparison is incomplete or changed")
    kernels = base.read(kernel_evidence)
    gpu = kernels["fixed_gpu"]
    require(kernels["status"] == "passed" and kernels["rollback"]["normal_teardown"] and gpu["passed"] and
            gpu["nvrtc_load_calls"] == {"legacy":2,"bundled":1} and len(gpu["kernel_outputs"]) == 6,
            "CUDA module prerequisite failed")
    sources = dict(mid.fingerprint(path) for path in (pilot_evidence,plan_path,runs_path,kernel_evidence))
    for fingerprints in (plan["source_hashes"],kernels["source_sha256"]):
        mid.merge_hashes(sources,{(HERE / name).relative_to(base.WORK).as_posix():sha
                                 for name,sha in fingerprints.items()})
    for row in rows:
        item = record(pilot_evidence.parent / row["artifact"],row["sha256"])
        mid.merge_hashes(sources,{item["path"]:item["sha256"]})
    return sources


def freeze(pilot_evidence, kernel_evidence):
    require(not FROZEN.exists(), "Preserve the existing prepared qualification freeze")
    p = base.read(TEMPLATE)
    sf, of = base.read(mid.FROZEN), base.read(outer.FROZEN)
    ms, os_ = base.read(mid.SUMMARY), base.read(outer.SUMMARY)
    require(ms["passes"] and os_["summary"] == {"low":{"qualified":0,"count":6},"high":{"qualified":18,"count":18}},
            "Prior middle or outer eligibility differs")
    require(p["physical_times"] == sf["physical_times"] and p["rfm_seeds"] == sf["template"]["rfm_seeds"] and
            p["legendre_seed"] == sf["template"]["deterministic_seed"] and p["gpu_workers"] == 1 and all(
                p[k] == sf["template"][k] for k in ("horizon","switch_time","graph_steps","probe_spacing","physical_gate",
                                                     "implementation_gate","reference_budget_fraction","references")),
            "Prepared numerical gate differs from the previous comparison")
    preparation_paths = tuple(HERE / name for name in ("galerkin_cuda_prepared.py", "galerkin_cuda_module_bundle.py",
        "galerkin_phase_shared.py", "galerkin_cuda_native.cu", "galerkin_cuda_variable.cu", "make_cuda_prepared_solver.py",
        "cuda_prepared_solver_generation.json", "galerkin_phase_shared_audit.json", "verify_switch_qualification.py"))
    host = base.read(HERE / "galerkin_phase_shared_audit.json")
    generated = base.read(HERE / "cuda_prepared_solver_generation.json")
    require(host["status"] == "passed" and host["source_sha256"] == base.digest(HERE / "galerkin_phase_shared.py") and
            len(host["rows"]) == 24 and all(r["status"] == "passed" for r in host["rows"]) and
            all(base.digest(HERE / name) == sha for name,sha in host["config_sha256"].items()) and
            generated["target_sha256"] == base.digest(HERE / "galerkin_cuda_prepared.py") and
            generated["source_sha256"] == base.digest(HERE / "galerkin_cuda_variable.py"), "Prepared source changed after auditing")
    sources = prerequisites(pilot_evidence,kernel_evidence)
    for entries in (sf["source_hashes"],of["source_hashes"]):
        mid.merge_hashes(sources,entries)
    mid.merge_hashes(sources,{r["path"]:r["sha256"] for r in sf["early_references"].values()})
    for source, summary in ((mid,ms),(outer,os_)):
        require(summary["status"] == "complete" and summary["protocol_sha256"] == base.digest(source.FROZEN) and
                summary["runs_sha256"] == base.digest(source.RUNS), "Archive journal, freeze or qualification changed")
    deps = (TEMPLATE, Path(__file__), mid.FROZEN, mid.RUNS, mid.SUMMARY, outer.FROZEN, outer.RUNS, outer.SUMMARY,
            *preparation_paths, pilot_evidence, kernel_evidence)
    mid.merge_hashes(sources,dict(mid.fingerprint(path) for path in deps))
    scales = p["scales"]
    require([s["name"] for s in scales] == ["low","mid","high"], "Prepared scale ordering differs")
    schedules, specs, controls, jobs = {}, {}, [], []
    for scale in scales:
        group = scale["name"]
        module, parent, summary = (mid,sf,ms) if group == "mid" else (outer,of,os_)
        h0,h1 = scale["rates"]
        times = sorted(p["physical_times"] + [p["switch_time"] + i/h1 for i in (1,2)])
        expected = sf["checkpoints"] if group == "mid" else of["schedules"][group]["implementation_times"]
        steps = p["switch_time"]*h0 + (p["horizon"]-p["switch_time"])*h1
        require(len(set(times)) == 21 and times == expected and set(scale["preparations"]) <= PREPARATIONS.keys(),
                "Prepared schedule or independent preparation flags differ")
        schedules[group] = dict(startup_rate=h0,late_rate=h1,checkpoints=times,steps=steps)
        for name,seeds in ((scale["rfm"],p["rfm_seeds"][:1] if group == "low" else p["rfm_seeds"]),
                           (scale["legendre"],[p["legendre_seed"]])):
            specs[name] = parent["specs"][name]
            require(specs[name]["rate"] == h0, "Uniform anchor step rate differs")
            for seed in seeds:
                control = archived_controls(module,summary,(group,name,seed))
                controls.append(control)
                mid.merge_hashes(sources,{control[k]["path"]:control[k]["sha256"] for k in ("early","late","cpu_recurrence","cuda_graph")})
                jobs += [dict(resolution=group,name=name,seed=seed,preparation=m) for m in scale["preparations"]]
    require(len(controls) == 14 and len(jobs) == len({(*key(j),j["preparation"]) for j in jobs}) == 26 and
            {s:sum(j["resolution"] == s for j in jobs) for s in schedules} == p["expected_rows"], "Prepared job coverage differs")
    verify_inputs(sources)
    frozen = dict(status="frozen",template=p,specs=specs,schedules=schedules,controls=controls,jobs=jobs,
                  solve_protocol=sf["solve_protocol"],source_hashes=sources,
                  evidence={"pilot":record(pilot_evidence),"kernels":record(kernel_evidence)})
    base.save(FROZEN,frozen)
    return frozen


def stem(job):
    return "_".join(map(str,(*key(job),job["preparation"])))


def validate(sol, job, frozen):
    p,s = frozen["template"],frozen["schedules"][job["resolution"]]
    h0,h1,steps = (s[k] for k in ("startup_rate","late_rate","steps"))
    mode = job["preparation"]
    split = h0*p["switch_time"]
    expected = dict(steps=steps,horizon=p["horizon"],dt=1/h0,startup_rate=h0,late_rate=h1,switch_step=split,
        graph_steps=p["graph_steps"],device_final_step=steps,nonlinear_evaluations=steps-1,marching_fft_batches=2*(steps-1),
        nonlinear_by_phase=dict(startup=split,late=steps-split-1),priming_fft_batches=2,bridge_extra_fft_batches=0,
        preparation=mode,phase_preparation_shared=mode!="module_only",compiled_cuda_modules=2 if mode=="shared_only" else 1)
    probes = np.arange(0,p["horizon"]+p["probe_spacing"],p["probe_spacing"])
    require(all(sol.row[k] == value for k,value in expected.items()) and np.array_equal(sol.times,s["checkpoints"]) and
            np.array_equal(sol.probe_times,probes) and sol.states.shape == (len(s["checkpoints"]),*sol.model.rhs_shape) and
            all(getattr(sol,k).shape == (len(probes),3) for k in PROBES) and
            all(np.isfinite(getattr(sol,k)).all() for k in ("states",*PROBES)), "Prepared work, observations or state differ")


def execute(frozen):
    verify_inputs(frozen["source_hashes"])
    require(all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")),
            "Qualification requires one host BLAS thread")
    for job in frozen["jobs"]:
        name = stem(job)
        path, sidecar = OUT / (name+".npz"),OUT / (name+".json")
        if sidecar.exists():
            prior = base.read(sidecar)
            require(prior["job"] == job,"Prepared resume job differs")
            if prior["status"] == "complete":
                record(path,prior["sha256"])
            continue
        require(not path.exists(),"A prepared raw trajectory is missing its sidecar")
        p,s = frozen["template"],frozen["schedules"][job["resolution"]]
        _,cfg = base.config(frozen["specs"][job["name"]],frozen["solve_protocol"],job["seed"])
        print("START",job,flush=True)
        result = dict(job=job)
        try:
            sol = solve_cuda_prepared(preparation=job["preparation"],startup_rate=s["startup_rate"],late_rate=s["late_rate"],
                switch_time=p["switch_time"],horizon=p["horizon"],snapshots=s["checkpoints"],probe_spacing=p["probe_spacing"],
                graph_steps=p["graph_steps"],**cfg)
            validate(sol,job,frozen)
            temporary = path.with_suffix(".tmp.npz")
            case.store_solution(temporary,sol,cfg)
            temporary.replace(path)
            result.update(status="complete",artifact=path.name,sha256=base.digest(path),metadata=sol.row)
        except (ArithmeticError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            result.update(status="failed",failure_type=type(exc).__name__,failure=str(exc))
        base.save(sidecar,result)
        print("DONE",job,result["status"],flush=True)


def load_saved(item):
    path = base.WORK / item["path"]
    record(path,item["sha256"])
    return case.load_solution(path,MODEL)


def scores(sol, truth, grids, keys):
    rows = case.score(sol,truth,grids)
    return rows,mid.old.score_rows(rows,keys)


def compare(sol, control, fields, grids, keys, gate):
    require(np.array_equal(sol.probe_times,control.probe_times),"Comparison probe times differ")
    probes = {}
    for k in PROBES:
        a,b = getattr(sol,k),getattr(control,k)
        norm = np.linalg.norm(b)
        require(norm > 1e-12,"Comparison probe denominator is near zero")
        probes[k] = float(np.linalg.norm(a-b)/norm)
    rows,worst = scores(sol,fields,grids,keys)
    passed = all(math.isfinite(x) and 0 <= x <= gate for x in (*worst.values(),*probes.values()))
    return dict(scores=rows,worst=worst,probes=probes,passed=passed)


def score(frozen):
    require(not RUNS.exists() and not SUMMARY.exists(), "Preserve the existing prepared scores")
    verify_inputs(frozen["source_hashes"])
    p = frozen["template"]
    times, grids = p["physical_times"], base.grid_set()
    truth, reference = absolute_reference(base.read(mid.FROZEN),grids)
    physical_keys = {(g,t) for g in grids for t in times}
    rows, raw_hashes = [], {}
    for control in frozen["controls"]:
        job_group = key(control)
        s = frozen["schedules"][job_group[0]]
        implementation_keys = {(g,t) for g in grids for t in s["checkpoints"]}
        first,last,cpu,gpu = (load_saved(control[k]) for k in ("early","late","cpu_recurrence","cuda_graph"))
        require(np.array_equal(first.times,times[:12]) and set(times[12:]) <= set(last.times) and
                all(np.array_equal(sol.times,s["checkpoints"]) for sol in (cpu,gpu)) and
                mid.switched.equal_spatial_models(tuple(sol.model for sol in (first,last,cpu,gpu))),
                "Prepared control coordinates, schedule or observations differ")
        selected = mid.select(last,times[12:])
        uniform = case.Solution(last.model,np.concatenate((first.states,selected.states)),np.asarray(times),
                                last.probe_times,last.transverse,last.vorticity,last.row)
        uniform_fields = case.reference_fields(uniform,grids)
        anchor = {name:{(r["grid"],r["time"]):r for r in scores(uniform,fields,grids,physical_keys)[0]}
                  for name,fields in truth.items()}
        cpu_fields,gpu_fields = (case.reference_fields(sol,grids) for sol in (cpu,gpu))
        for job in (j for j in frozen["jobs"] if key(j) == job_group):
            path = OUT / (stem(job)+".json")
            raw = base.read(path)
            raw_hashes[path.name] = base.digest(path)
            require(raw["job"] == job,"Prepared raw job differs from freeze")
            if raw["status"] != "complete":
                rows.append(dict(job=job,qualified=False,raw=raw))
                continue
            artifact = OUT / raw["artifact"]
            sol = load_saved(record(artifact,raw["sha256"]))
            require(sol.row == raw["metadata"] and mid.switched.equal_spatial_models((sol.model,cpu.model)),
                    "Prepared raw model or metadata differs")
            with np.load(artifact,allow_pickle=False) as saved:
                cfg = base.config(frozen["specs"][job["name"]],frozen["solve_protocol"],job["seed"])[1]
                require(json.loads(str(saved["config"])) == json.loads(json.dumps(cfg)), "Prepared configuration differs")
            validate(sol,job,frozen)
            impl = {name:compare(sol,original,fields,grids,implementation_keys,p["implementation_gate"])
                    for name,original,fields in (("cpu_recurrence",cpu,cpu_fields),("cuda_graph_old",gpu,gpu_fields))}
            physical = mid.select(sol,times)
            delta,_ = scores(physical,uniform_fields,grids,physical_keys)
            direct,guards,direct_worst,guard_worst = {},{},{},{}
            for name,fields in truth.items():
                direct[name],direct_worst[name] = scores(physical,fields,grids,physical_keys)
                guard = []
                for d in delta:
                    a = anchor[name][d["grid"],d["time"]]
                    guard.append(dict(grid=d["grid"],time=d["time"],**{k:d[k]*(1+a[k])+a[k] for k in mid.METRICS}))
                indexed = {(r["grid"],r["time"]):r for r in guard}
                require(all(r[k] <= indexed[r["grid"],r["time"]][k]+p["guard_roundoff_allowance"]
                            for r in direct[name] for k in mid.METRICS),"Direct physical error exceeds triangle guard")
                guards[name],guard_worst[name] = guard,mid.old.score_rows(guard,physical_keys)
            wake = base.wake(sol)
            ip = all(r["passed"] for r in impl.values())
            dp = all(max(r.values()) <= p["physical_gate"] for r in direct_worst.values())
            gp = all(max(r.values()) <= p["physical_gate"] for r in guard_worst.values())
            passed = ip and dp and gp and wake["sustained_alternating_probe_signs"]
            rows.append(dict(job=job,artifact=raw["artifact"],sha256=raw["sha256"],implementation=impl,physical_scores=direct,
                physical_worst=direct_worst,anchor_difference=delta,guard_scores=guards,guard_worst=guard_worst,wake=wake,
                implementation_pass=ip,physical_pass=dp,guard_pass=gp,qualified=passed))
            print("SCORE",job,passed,flush=True)
    require([r["job"] for r in rows] == frozen["jobs"], "Prepared scoring omitted or reordered a job")
    summary = {}
    for group in p["expected_rows"]:
        subset = [r for r in rows if r["job"]["resolution"] == group]
        summary[group] = dict(count=len(subset),qualified=sum(r["qualified"] for r in subset),
                              implementation_pass=sum(r.get("implementation_pass",False) for r in subset))
    low_expected = all(r.get("implementation_pass") and r["wake"]["sustained_alternating_probe_signs"] and
                       not (r["physical_pass"] and r["guard_pass"]) for r in rows if r["job"]["resolution"] == "low")
    sha = base.digest(FROZEN)
    base.save(RUNS,dict(status="complete",protocol_sha256=sha,raw_records=raw_hashes))
    base.save(SUMMARY,dict(status="complete",protocol_sha256=sha,runs_sha256=base.digest(RUNS),rows=rows,summary=summary,
        reference=reference,low_failure_control_expected=low_expected,
        timing_eligible_scales=[k for k in ("mid","high") if low_expected and summary[k]["qualified"] == p["expected_rows"][k]],
        isolated_speed_claim_supported=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    for flag in ("freeze-only","gpu","score"):
        action.add_argument("--"+flag,action="store_true")
    cli.add_argument("--pilot-evidence",type=Path)
    cli.add_argument("--kernel-evidence",type=Path)
    args = cli.parse_args()
    if args.freeze_only:
        require(args.pilot_evidence is not None and args.kernel_evidence is not None,"Freeze requires both independent prerequisite audits")
        freeze(args.pilot_evidence.resolve(),args.kernel_evidence.resolve())
        print("FROZEN",base.digest(FROZEN),flush=True)
    elif args.gpu:
        execute(base.read(FROZEN))
    else:
        score(base.read(FROZEN))
