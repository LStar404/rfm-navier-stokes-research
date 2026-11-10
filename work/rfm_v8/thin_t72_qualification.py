"""Preregister thin-worker T72 numerical evidence and compare the saved arrays."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
V7 = WORK/"rfm_v7"
sys.path.insert(0,str(V7))
ARRAYS = ("states","times","probe_times","transverse","vorticity")
JOB_FIELDS = ("block","name","arm","seed")
RFM = (7331,7687,8011,8353,9041)
THIN_ARMS = {"uniform":"uniform","switch_both":"combined","fd":"fd"}
RUNTIME_FILES = ("rfm_v7/galerkin_cuda_native.cu","rfm_v7/galerkin_cuda_variable.cu",
                 "rfm_v7/fd_cuda_native.cu","rfm_v6/fd_full_band.dll")


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def save_new(path,data):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")


def bind(path,expected=None):
    relative = path.resolve().relative_to(WORK).as_posix()
    sha = digest(path)
    require(expected is None or expected == sha,f"Archive identity differs for {relative}")
    return dict(path=relative,sha256=sha)


def check_record(record):
    path = (WORK/record["path"]).resolve()
    require(path.relative_to(WORK).as_posix() == record["path"],"Unsafe qualification path")
    bind(path,record["sha256"])
    return path


def check_manifest(entries):
    for name,sha in entries.items():
        check_record(dict(path=name,sha256=sha))


def failure_text(exc):
    return str(exc).replace(str(WORK.parent),"<workspace>").replace(str(Path.home()),"<user-root>")


def archive_context(scale):
    directory = V7/f"prepared_formal_timing_{scale}_output"
    p,t,q = (read(directory/name) for name in ("protocol.json","timing.json","summary.json"))
    records = {name:bind(directory/f"{name}.json",q.get(name+"_sha256"))
               for name in ("protocol","timing","summary")}
    require(p["status"] == "frozen" and t["status"] == "complete" and q["numerical_qualified"] and
            t["protocol_sha256"] == records["protocol"]["sha256"] and
            q["rows"] == len(p["schedule"]) == len(t["rows"]), "v7 archive is not fully numerically qualified")
    return directory,p,t,q,records


def unique_jobs(scale, directory, p, timing, summary):
    seen,jobs = set(),[]
    for index,(job,row) in enumerate(zip(p["schedule"],timing["rows"],strict=True)):
        require({key:row[key] for key in JOB_FIELDS} == job and row["index"] == index and row["qualified"],
                "v7 numerical slot differs from its freeze")
        name,arm,seed = (job[k] for k in ("name","arm","seed"))
        if arm not in THIN_ARMS or (arm == "fd" and name != ("fd176_k80" if scale=="mid" else "fd184_k80")):
            continue
        identity = name,arm,seed
        if identity in seen:
            continue
        seen.add(identity)
        artifact = directory/row["artifact"]
        sidecar = artifact.with_suffix(".json")
        require(read(sidecar) == row and summary["raw_evidence_sha256"][artifact.name] == row["sha256"],
                "v7 raw qualification sidecar differs")
        arm_id = THIN_ARMS[arm]
        design = p["design"]
        config = (p["fd_controls"][name]["config"] if arm=="fd" else
                  dict(m=p["specs"][name]["m"],k=p["specs"][name]["k"],q=p["specs"][name]["q"],
                       scale=62.,mean_order=24,layers=p["solve_protocol"]["layers"],split=True,
                       backend="fft",seed=seed,family=p["specs"][name]["method"]))
        args = dict(horizon=72,snapshots=design["physical_times"],probe_spacing=design["probe_spacing"],
                    graph_steps=design["cuda_graph_steps"],**config)
        schedule = p["step_schedule"]
        if arm=="fd":
            fd = p["fd_controls"][name]
            args.update(steps=fd["rate"]*72,cuda_linear_solve=fd["linear"])
        elif arm=="uniform":
            args.update(steps=schedule["startup_rate"]*72)
        else:
            args.update(startup_rate=schedule["startup_rate"],late_rate=schedule["late_rate"],
                        switch_time=design["switch_time"],preparation="combined")
        jobs.append(dict(scale=scale,archive_job=job,archive_artifact=bind(artifact,row["sha256"]),
                         archive_sidecar=bind(sidecar,summary["raw_evidence_sha256"][sidecar.name]),
                         expected_spatial_sha256=row["marker"]["spatial_sha256"],name=name,seed=seed,
                         arm=arm_id,family="fd" if arm=="fd" else config["family"],
                         applies_to_scales=["mid","high"] if name=="fd176_k80" else [scale],
                         config=config,solve_args=args))
    count = {"rfm":10,"legendre":2,"fd":1}
    require(len(jobs)==13 and all(sum(r["family"]==family for r in jobs)==n for family,n in count.items()) and
            {(r["arm"],r["seed"]) for r in jobs if r["family"]=="rfm"} ==
            {(arm,seed) for arm in ("uniform","combined") for seed in RFM}, "Incomplete T72 thin qualification inventory")
    return jobs


def discover_local_sources(arm):
    script = """import json,sys
from pathlib import Path
from solver_worker import select_solver
root=Path.cwd().parent.resolve()
select_solver(dict(imports='minimal',arm=sys.argv[1]))
files=set()
for m in tuple(sys.modules.values()):
    p=getattr(m,'__file__',None)
    if p:
        path=Path(p).resolve()
        if path.is_relative_to(root):
            files.add(path.relative_to(root).as_posix())
print(json.dumps(sorted(files)))
"""
    env = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    done = subprocess.run([sys.executable,"-B","-c",script,arm],cwd=HERE,env=env,
                          capture_output=True,text=True,check=True,timeout=45)
    found = json.loads(done.stdout)
    require("rfm_v8/solver_worker.py" in found and "rfm_v7/diagnostic_wake.py" not in found and
            "rfm_v7/cpu_formal.py" not in found, "Unexpected diagnostic import in the lean dependency closure")
    return found


def make_plan():
    contexts,jobs = {},[]
    for scale in ("mid","high"):
        directory,p,t,q,records = archive_context(scale)
        require(len(p["design"]["physical_times"]) == 19 and p["design"]["horizon"]==72 and
                p["design"]["physical_gate"]==.005 and p["design"]["implementation_gate"]==1e-5,
                "Thin qualification must preserve the v7 numerical gates")
        refs = p["reference_protocol"]
        check_record(refs)
        contexts[scale] = dict(archive=records,reference_protocol=refs,old_source_hashes=p["source_hashes"])
        jobs.extend(unique_jobs(scale,directory,p,t,q))
    _,mid,_,_,_ = archive_context("mid")
    _,high,_,_,_ = archive_context("high")
    require(mid["fd_controls"]["fd176_k80"]==high["fd_controls"]["fd176_k80"],
            "FD176 reuse across scales requires identical registered settings and archived controls")
    used = {name:discover_local_sources(name) for name in ("uniform","combined","fd")}
    source_names = set(RUNTIME_FILES)|{name for names in used.values() for name in names}
    source_names.update(("rfm_v7/prepared_process_identity.py","rfm_v8/solver_worker.py",
        "rfm_v8/fresh_worker_diagnostic.py","rfm_v8/thin_t72_qualification.py"))
    sources = {name:digest(WORK/name) for name in sorted(source_names)}
    static = dict(schema="rfm-v8-thin-t72-1",status="frozen",rows_expected=26,
        scope="Numerical qualification only. No timing or isolation claim. One FD176 trace covers both identical FD176 settings.",
        execution_environment=dict(python_version=platform.python_version(),platform=platform.platform(),affinity_mask=16,
                                   blas_threads=1),
        archive_contexts=contexts,observed_minimal_imports=used,source_hashes=sources)
    token = sha256(json.dumps(dict(protocol=static,jobs=jobs),sort_keys=True).encode()).hexdigest()
    static["run_token"] = token
    for index,job in enumerate(jobs):
        job["request"] = dict(index=index,schema="rfm-v8-worker-1",run_token=token,affinity_mask=16,
            name=job["name"],seed=job["seed"],family=job["family"],arm=job["arm"],imports="minimal",
            config=job["config"],solve_args=job["solve_args"],trajectory=f"trajectory_{index:03d}.npz")
    static["jobs"] = jobs
    return static


def verify_plan(p):
    require(p["status"]=="frozen" and p["schema"]=="rfm-v8-thin-t72-1" and
            p["rows_expected"]==len(p["jobs"])==26 and platform.python_version()==p["execution_environment"]["python_version"],
            "Thin T72 qualification protocol differs")
    check_manifest(p["source_hashes"])
    for context in p["archive_contexts"].values():
        for record in context["archive"].values():
            check_record(record)
        check_record(context["reference_protocol"])
    original_sources = {}
    for context in p["archive_contexts"].values():
        for name,sha in context["old_source_hashes"].items():
            require(original_sources.get(name,sha)==sha,"v7 qualifications use different scoring source bytes")
            original_sources[name] = sha
    check_manifest(original_sources)
    for index,job in enumerate(p["jobs"]):
        require(job["request"]["index"]==index and job["request"]["run_token"]==p["run_token"] and
                job["request"]["config"]==job["config"] and job["request"]["solve_args"]==job["solve_args"],
                "Request differs from its thin qualification slot")
        check_record(job["archive_artifact"])
        check_record(job["archive_sidecar"])


def freeze(output):
    require(not output.exists(),"Preserve existing thin-qualification evidence")
    p = make_plan()
    verify_plan(p)
    output.mkdir()
    save_new(output/"protocol.json",p)
    print("FROZEN",len(p["jobs"]),digest(output/"protocol.json"),flush=True)


def execute(output):
    from fresh_worker_diagnostic import THREADS,launch,pin
    path = output/"protocol.json"
    p = read(path)
    phash = digest(path)
    verify_plan(p)
    require(not (output/"runs.json").exists() and not list(output.glob("request_*.json")),
            "Each thin-qualification trajectory gets a single attempt")
    for key in THREADS:
        os.environ[key] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = str(output/"qualification_no_cache")
    pin()
    rows = []
    for job in p["jobs"]:
        request = job["request"]
        index = request["index"]
        try:
            row = launch(request,output)
        except Exception as exc:
            row = dict(index=index,status="failed",failure_type=type(exc).__name__,failure=failure_text(exc))
        save_new(output/f"run_{index:03d}.json",row)
        rows.append(row)
        print(index,job["scale"],job["name"],job["seed"],job["arm"],row["status"],flush=True)
    verify_plan(p)
    require(digest(path)==phash,"Thin qualification protocol changed during execution")
    save_new(output/"runs.json",dict(status="complete",protocol_sha256=phash,rows=rows))


def non_time_metadata(value):
    return {k:v for k,v in value.items() if k!="seconds" and not k.endswith("_seconds")}


def verify_transport(row,request):
    order = [row[k] for k in ("launch_monotonic_ns","ready_received_monotonic_ns","go_sent_monotonic_ns",
                              "return_received_monotonic_ns","save_sent_monotonic_ns","exit_monotonic_ns")]
    require(order==sorted(order) and 0 < row["solver_seconds"] <= (order[3]-order[2])/1e9 and
            row["solver_seconds"]==row["solver_core_seconds"]+row["model_release_seconds"] and
            row["fresh_process_return_seconds"]==(order[3]-order[0])/1e9, "Thin worker transport chronology differs")
    for key,event in (("ready","READY"),("marker","RETURN")):
        marker = row[key]
        require(marker["event"]==event and marker["index"]==request["index"] and marker["run_token"]==request["run_token"] and
                marker["request_sha256"]==row["request_sha256"] and marker["inherited_affinity_mask"]==request["affinity_mask"] and
                all(marker[k]==row["child_identity"][k]==row["terminal_identity"][k] for k in ("pid","birth_filetime_100ns")),
                "Thin worker marker or exact child identity differs")
    require(row["terminal_identity"]["exit_filetime_100ns"] > row["child_identity"]["birth_filetime_100ns"] and
            row["marker"]["status"]=="solved", "Thin worker termination or RETURN status differs")


def evaluate(output,job,row,old_protocol,scoring):
    import numpy as np
    import prepared_formal_timing as archived
    request = job["request"]
    index = request["index"]
    qpath = output/f"request_{index:03d}.json"
    target = output/request["trajectory"]
    old = read(check_record(job["archive_sidecar"]))
    saved_path = check_record(job["archive_artifact"])
    require(row["request_sha256"]==digest(qpath) and read(qpath)==request and
            row["trajectory_sha256"]==digest(target) and row["exit_code"]==0,
            "Thin trajectory and frozen request hashes differ")
    verify_transport(row,request)
    require(row["marker"]["spatial_sha256"]==job["expected_spatial_sha256"]==old["marker"]["spatial_sha256"],
            "Thin trajectory spatial reconstruction signature differs")
    old_job = job["archive_job"]
    require({k:old[k] for k in JOB_FIELDS}==old_job and archived.configuration(old_job,old_protocol)==request["config"],
            "Thin trajectory is not for the same frozen v7 configuration")
    archived.verify_numerical_row(dict(old,job=old_job),old_protocol)
    with np.load(target,allow_pickle=False) as candidate,np.load(saved_path,allow_pickle=False) as reference:
        cmeta,rmeta = (json.loads(str(x["metadata"])) for x in (candidate,reference))
        require(set(candidate.files)==set(reference.files) and
                all(json.loads(str(x["config"]))==request["config"] for x in (candidate,reference)) and
                rmeta==old["metadata"] and non_time_metadata(cmeta)==non_time_metadata(rmeta),
                "Saved NPZ configuration or non-time metadata differs")
        require(np.array_equal(candidate["times"],request["solve_args"]["snapshots"]) and
                np.array_equal(candidate["probe_times"],np.arange(0.,72.+.001,.125)) and
                len(candidate["times"])==19 and candidate["transverse"].shape==candidate["vorticity"].shape==(577,3) and
                all(np.isfinite(candidate[key]).all() for key in ARRAYS),
                "Incomplete absolute-time or finite T72 trajectory")
        arrays = {}
        for key in ARRAYS:
            new,prior = candidate[key],reference[key]
            arrays[key] = dict(shape=list(new.shape),dtype=new.dtype.str,sha256=sha256(new.tobytes()).hexdigest(),
                byte_exact=new.shape==prior.shape and new.dtype==prior.dtype and new.tobytes()==prior.tobytes())
    result = dict(archive_artifact=job["archive_artifact"],trajectory_sha256=row["trajectory_sha256"],
                  arrays=arrays,non_time_metadata_equal=True,spatial_signature_equal=True)
    if all(a["byte_exact"] for a in arrays.values()):
        return dict(result,qualified=True,gate="inherited_exact_v7_numerical_gate",physical_worst=old["physical_worst"],
                    guard_worst=old["guard_worst"],wake=old["wake"],implementation={k:v["worst"] for k,v in old["implementation"].items()})
    if job["scale"] not in scoring:
        scoring[job["scale"]] = archived.context(old_protocol)
    candidate_row = dict(marker=row["marker"],job=old_job)
    candidate_row.update(archived.score_solution(target,candidate_row,old_job,old_protocol,scoring[job["scale"]]))
    archived.verify_numerical_row(candidate_row,old_protocol)
    return dict(result,qualified=candidate_row["qualified"],gate="independent_v7_full_physical_and_implementation_rescore",
                scores=candidate_row)


def score(output):
    p = read(output/"protocol.json")
    verify_plan(p)
    require(not (output/"qualification.json").exists(),"Preserve previous thin qualification")
    protocol_sha = digest(output/"protocol.json")
    j = read(output/"runs.json")
    require(j["status"]=="complete" and j["protocol_sha256"]==protocol_sha and len(j["rows"])==len(p["jobs"]),
            "Thin qualification run journal differs")
    plans = {scale:read(check_record(record["archive"]["protocol"])) for scale,record in p["archive_contexts"].items()}
    scoring = {}
    results = []
    for index,(job,row) in enumerate(zip(p["jobs"],j["rows"],strict=True)):
        require(row["index"]==index and read(output/f"run_{index:03d}.json")==row,"Unbound thin raw slot")
        try:
            require(row["status"]=="complete","Thin worker failed")
            if index and "exit_monotonic_ns" in j["rows"][index-1]:
                require(j["rows"][index-1]["exit_monotonic_ns"] < row["launch_monotonic_ns"],
                        "Thin worker overlaps previous qualification slot")
            result = evaluate(output,job,row,plans[job["scale"]],scoring)
        except Exception as exc:
            result = dict(qualified=False,gate="failed_closed",failure_type=type(exc).__name__,failure=failure_text(exc))
        results.append(dict(job={k:job[k] for k in ("scale","name","seed","arm","applies_to_scales")},**result))
        print(index,job["name"],job["arm"],result["qualified"],result["gate"],flush=True)
    verify_plan(p)
    require(digest(output/"protocol.json")==protocol_sha,"Qualification protocol changed during scoring")
    good = all(r["qualified"] for r in results)
    save_new(output/"qualification.json",dict(status="passed" if good else "failed",protocol_sha256=protocol_sha,
        runs_sha256=digest(output/"runs.json"),independent_references={k:v["reference_protocol"] for k,v in p["archive_contexts"].items()},
        rows=results,scope="Same 19 physical times and 577 absolute probe samples. No timing or isolation claim."))
    return good


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    options = cli.add_mutually_exclusive_group(required=True)
    options.add_argument("--freeze-only",action="store_true")
    options.add_argument("--execute",action="store_true")
    options.add_argument("--score",action="store_true")
    options.add_argument("--check-plan",action="store_true")
    cli.add_argument("--output",type=Path,required=True)
    args = cli.parse_args()
    out = args.output.resolve()
    if args.check_plan:
        p = make_plan()
        verify_plan(p)
        print("PASS",len(p["jobs"]),len(p["source_hashes"]),flush=True)
    elif args.freeze_only:
        freeze(out)
    elif args.execute:
        execute(out)
    else:
        sys.exit(0 if score(out) else 1)
