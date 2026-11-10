"""Preregister, execute and audit six matched CUDA policies on full trajectories."""

import argparse
from collections import defaultdict
from contextlib import suppress
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time

import thin_t72_qualification as thin
from cuda129_case_runtime import VARIANTS, V7, bundled_source
from cuda129_runtime import TOOLKIT_DEFAULT, VENDOR
from formal_timing_coordinator import Inbox
from fresh_worker_diagnostic import pin
from prepared_process_identity import identity_from_handle
from solver_worker import MARKER, THREADS


HERE,WORK = thin.HERE,thin.WORK
THIN = HERE/"thin_t72_qualification_output"
PREFLIGHT = HERE/"cuda129_gate_summary.json"
WORKER = HERE/"cuda129_diagnostic_worker.py"
PHASES = ("host_setup_seconds","host_linear_precompute_seconds","runtime_init_seconds","upload_seconds",
          "plan_seconds","graph_setup_seconds","march_seconds","download_seconds")
CONTRASTS = {
    "compiler_same_compute52":("legacy","compiler129_compute52"),
    "virtual_target":("compiler129_compute52","compiler129_compute120"),
    "cubin_vs_ptx_same_gpu_generation":("compiler129_compute120","compiler129_sm120"),
    "fft_only":("legacy","fft129_only"),
    "fft_at_native_compiler":("compiler129_sm120","compiler129_sm120_fft129"),
    "native_compiler_at_modern_fft":("fft129_only","compiler129_sm120_fft129"),
    "combined_toolchain":("legacy","compiler129_sm120_fft129"),
}
read,digest,bind,save,require = thin.read,thin.digest,thin.bind,thin.save_new,thin.require


def verify_environment(plan):
    thin.check_manifest(plan["source_hashes"])
    vendor = read(thin.check_record(plan["vendor_manifest"]))
    for row in vendor["installed_members"]:
        require(digest(VENDOR/row["path"])==row["sha256"],"Pinned modern DLL or wheel metadata differs")
    for name,sha in plan["legacy_dll_sha256"].items():
        require(digest(TOOLKIT_DEFAULT/"bin"/name)==sha,"Legacy CUDA DLL differs")
    for name,record in plan["thin_archive"].items():
        thin.check_record(record)


def make_plan(blocks):
    tp,tq = (read(THIN/name) for name in ("protocol.json","qualification.json"))
    require(tq["status"]=="passed" and tq["protocol_sha256"]==digest(THIN/"protocol.json") and
            tq["runs_sha256"]==digest(THIN/"runs.json"),"Full thin numerical archive did not pass")
    thin.verify_plan(tp)
    selected = [j for j in tp["jobs"] if j["family"]=="fd" or j["arm"]=="combined" and j["seed"] in (7331,17)]
    require(len(selected)==6 and len({j["name"] for j in selected})==6,"Expected four Galerkin configurations and two FD controls")
    base = read(PREFLIGHT)
    require(base["status"]=="passed_fixed_and_t9" and base["preflight"]["passed"]==6 and base["t9"]["qualified"]==12 and
            base["evidence_sha256"]["vendor_manifest"]==digest(VENDOR/"manifest.json"),
            "Modern Galerkin fixed and short qualifications differ")
    sources = dict(tp["source_hashes"])
    for p in (Path(__file__).resolve(),WORKER,HERE/"cuda129_case_runtime.py",HERE/"fd_cuda_native_cuda129.py",
              HERE/"cuda129_fd_source_audit.json",HERE/"cuda129_runtime.py",HERE/"galerkin_cuda_prepared_cuda129.py",
              HERE/"cuda129_solver_source_audit.json",HERE/"formal_timing_coordinator.py",PREFLIGHT):
        item = bind(p)
        sources[item["path"]] = item["sha256"]
    for name,sha in read(HERE/"cuda129_fd_source_audit.json")["generated_sha256"].items():
        require(digest(HERE/name)==sha,"FD runtime factory whitelist no longer applies")
    env = dict(cache="CUDA_CACHE_DISABLE=1 for every arm, process-local",blas_threads=1,affinity_mask=16,
               legacy_toolkit="CUDA 11.8 default absolute toolkit path; CUDA_PATH removed from worker environment",
               accelerator=base["host"])
    plan = dict(status="frozen_diagnostic",scope="Complete T72 numerical and timing diagnostics; no isolation claim.",
        blocks=blocks,order_seed=601294,expected_rows=36*blocks,variants=VARIANTS,contrasts=CONTRASTS,
        case_names=[j["name"] for j in selected],execution_environment=env,source_hashes=sources,
        vendor_manifest=bind(VENDOR/"manifest.json"),legacy_dll_sha256=base["legacy_dll_sha256"],
        thin_archive={name:bind(THIN/name) for name in ("protocol.json","runs.json","qualification.json")},
        kernel_source_sha256={"fd":digest(V7/"fd_cuda_native.cu"),"galerkin":sha256(bundled_source()).hexdigest()},
        advancement_policy="Single-thread host, graph24, no warmed runtime or library cache. Save/scoring outside both clocks. Use multiple complete blocks plus a separately monitored cohort before adoption.")
    token = sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    plan["run_token"] = token
    jobs = []
    rng = random.Random(plan["order_seed"])
    for block in range(blocks):
        group = [(job,variant) for job in selected for variant in VARIANTS]
        rng.shuffle(group)
        for job,variant in group:
            index = len(jobs)
            current = {k:v for k,v in job.items() if k!="request"}
            request = dict(index=index,schema="rfm-v8-cuda129-diagnostic-1",run_token=token,affinity_mask=16,
                name=job["name"],family=job["family"],seed=job["seed"],arm=job["arm"],variant=variant,
                config=job["config"],solve_args=job["solve_args"],trajectory=f"trajectory_{index:03d}.npz")
            jobs.append(dict(current,block=block,variant=variant,request=request))
    plan["jobs"] = jobs
    require(len(jobs)==plan["expected_rows"],"Incorrect preregistered cohort size")
    verify_environment(plan)
    return plan


def launch(job,output,env):
    request = job["request"]
    i = request["index"]
    path = output/f"request_{i:03d}.json"
    save(path,request)
    request_sha = digest(path)
    command = [sys.executable,"-B",str(WORKER),str(path),request_sha]
    row = dict(index=i,status="failed",request_sha256=request_sha)
    process,inbox = None,None
    try:
        with (output/f"worker_{i:03d}.stderr.log").open("xb") as error:
            launch_ns = time.perf_counter_ns()
            process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,
                                       creationflags=subprocess.CREATE_NO_WINDOW,env=env)
        identity = identity_from_handle(process._handle)
        inbox = Inbox(process.stdout,output/f"worker_{i:03d}.stdout.log",MARKER)
        row.update(launch_monotonic_ns=launch_ns,child_identity=identity)

        def receive(event):
            stamp,value = inbox.receive(120)
            require(value["event"]==event and value["index"]==i and value["run_token"]==request["run_token"] and
                    value["request_sha256"]==request_sha and value["inherited_affinity_mask"]==request["affinity_mask"] and
                    value["pid"]==identity["pid"] and value["birth_filetime_100ns"]==identity["birth_filetime_100ns"],
                    "Exact worker marker differs from its request or Popen handle")
            return stamp,value

        row["ready_received_monotonic_ns"],row["ready"] = receive("READY")
        row["go_sent_monotonic_ns"] = time.perf_counter_ns()
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        returned,marker = receive("RETURN")
        row.update(return_received_monotonic_ns=returned,marker=marker)
        require(marker["status"]=="solved" and not (output/request["trajectory"]).exists(),"Solver failed or saved before RETURN")
        solver = marker["solver_seconds"]
        fresh = (returned-launch_ns)/1e9
        require(0 < solver <= (returned-row["go_sent_monotonic_ns"])/1e9 <= fresh,"Dual-clock scope differs")
        row.update(solver_seconds=solver,solver_core_seconds=marker["solver_core_seconds"],
                   model_release_seconds=marker["model_release_seconds"],fresh_process_return_seconds=fresh,
                   fresh_process_overhead_seconds=fresh-solver)
        row["save_sent_monotonic_ns"] = time.perf_counter_ns()
        process.stdin.write(b"SAVE\n")
        process.stdin.flush()
        process.wait(timeout=60)
        require(process.returncode==0,"Toolchain worker failed after SAVE")
        row.update(status="complete",trajectory_sha256=digest(output/request["trajectory"]))
    except Exception as exc:
        row.update(failure_type=type(exc).__name__,failure=thin.failure_text(exc))
    finally:
        if process is not None:
            with suppress(OSError):
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            row.update(exit_code=process.returncode,exit_monotonic_ns=time.perf_counter_ns(),
                       terminal_identity=identity_from_handle(process._handle))
            try:
                if inbox is not None:
                    inbox.close()
            finally:
                process.stdout.close()
    return row


def execute(output):
    protocol = output/"protocol.json"
    plan = read(protocol)
    before = digest(protocol)
    require(plan["status"]=="frozen_diagnostic" and len(plan["jobs"])==plan["expected_rows"] and
            not (output/"runs.json").exists() and not list(output.glob("request_*.json")),
            "Preserve existing registered diagnostic attempts")
    verify_environment(plan)
    env = dict(os.environ,CUDA_CACHE_DISABLE="1",**{k:"1" for k in THREADS})
    env.pop("CUDA_PATH",None)
    pin()
    rows = []
    for job in plan["jobs"]:
        i = job["request"]["index"]
        try:
            row = launch(job,output,env)
        except Exception as exc:
            row = dict(index=i,status="failed",failure_type=type(exc).__name__,failure=thin.failure_text(exc))
        rows.append(row)
        save(output/f"run_{i:03d}.json",row)
        print(i,job["name"],job["variant"],row["status"],flush=True)
    verify_environment(plan)
    require(digest(protocol)==before,"Toolchain protocol changed")
    save(output/"runs.json",dict(status="complete",protocol_sha256=before,rows=rows))


def phases(output,job,row,plan):
    import numpy as np
    req = job["request"]
    variant = job["variant"]
    compiler,target,artifact,fft = VARIANTS[variant]
    evidence = row["marker"]["toolchain"]
    expected_source = plan["kernel_source_sha256"]["fd" if job["family"]=="fd" else "galerkin"]
    require(evidence["variant"]==variant and evidence["nvrtc_version"]==([11,8] if compiler=="legacy" else [12,9]) and
            evidence["compiler_options"]==["--gpu-architecture="+target] and evidence["artifact"]==artifact and
            evidence["cufft_version"]==(10900 if fft=="legacy" else 11401) and evidence["kernel_source_sha256"]==expected_source and
            evidence["driver_version"]==plan["execution_environment"]["accelerator"]["driver_api_version"],
            "Loaded toolchain or original CUDA source differs from registered policy")
    with np.load(output/req["trajectory"],allow_pickle=False) as raw:
        metadata = json.loads(str(raw["metadata"]))
    stage = {k:metadata[k] for k in PHASES if k in metadata}
    require(all(np.isfinite(x) and x>=0 for x in stage.values()),"Invalid phase data")
    return dict(stage,solver_seconds=row["solver_seconds"],fresh_process_return_seconds=row["fresh_process_return_seconds"],
        model_release_seconds=row["model_release_seconds"],fresh_overhead_seconds=row["fresh_process_overhead_seconds"],
        solver_core_unattributed_seconds=row["solver_core_seconds"]-sum(stage.values()))


def compare(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row.get("qualified"):
            groups[row["name"],row["variant"]][row["block"]] = row["clocks"]
    result = []
    for (name,variant),values in groups.items():
        result.append(dict(name=name,variant=variant,blocks=sorted(values),medians={
            key:median(r[key] for r in values.values()) for key in next(iter(values.values()))}))
    pairs = []
    for name in sorted({row["name"] for row in rows}):
        for label,(baseline,candidate) in CONTRASTS.items():
            old,new = groups[name,baseline],groups[name,candidate]
            common = sorted(old.keys() & new.keys())
            if common:
                keys = old[common[0]].keys() & new[common[0]].keys()
                deltas = {key:[new[i][key]-old[i][key] for i in common] for key in sorted(keys)}
                ratios = {key:[old[i][key]/new[i][key] for i in common] for key in ("solver_seconds","fresh_process_return_seconds")}
                pairs.append(dict(name=name,contrast=label,baseline=baseline,candidate=candidate,blocks=common,
                    paired_delta_candidate_minus_baseline_seconds=deltas,paired_baseline_over_candidate=ratios,
                    median_ratio={key:median(values) for key,values in ratios.items()}))
    return dict(groups=result,pairs=pairs)


def score(output):
    path = output/"protocol.json"
    p = read(path)
    saved = read(output/"runs.json")
    require(saved["protocol_sha256"]==digest(path) and len(saved["rows"])==p["expected_rows"],
            "Toolchain diagnostic protocol/run identity differs")
    verify_environment(p)
    archive = read(THIN/"protocol.json")
    contexts = {name:read(thin.check_record(c["archive"]["protocol"])) for name,c in archive["archive_contexts"].items()}
    scoring = {}
    results = []
    for job,row in zip(p["jobs"],saved["rows"],strict=True):
        req = job["request"]
        index = req["index"]
        require(row==read(output/f"run_{index:03d}.json") and row["index"]==index,"Raw diagnostic slot differs")
        result = dict(name=job["name"],scale=job["scale"],seed=job["seed"],variant=job["variant"],block=job["block"])
        try:
            require(row["status"]=="complete","Worker slot failed")
            if index and "exit_monotonic_ns" in saved["rows"][index-1]:
                require(saved["rows"][index-1]["exit_monotonic_ns"]<row["launch_monotonic_ns"],"Diagnostic workers overlap")
            result.update(thin.evaluate(output,job,row,contexts[job["scale"]],scoring))
            result["clocks"] = phases(output,job,row,p)
        except Exception as exc:
            result.update(qualified=False,failure_type=type(exc).__name__,failure=thin.failure_text(exc))
        results.append(result)
        print(index,result["name"],result["variant"],result["qualified"],result.get("gate",""),flush=True)
    verify_environment(p)
    success = all(r["qualified"] for r in results)
    save(output/"qualification.json",dict(status="passed" if success else "failed",protocol_sha256=digest(path),
        runs_sha256=digest(output/"runs.json"),rows=results,scope="Numerical and performance diagnostics, no isolated ranking.",
        isolated_speed_claim_supported=False,**compare(results)))
    return success


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze-only","execute","score"):
        actions.add_argument("--"+name,action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--blocks",type=int,choices=(1,2,3),default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.freeze_only:
        require(not output.exists(),"Preserve existing cohort")
        plan = make_plan(args.blocks)
        output.mkdir()
        save(output/"protocol.json",plan)
        print("FROZEN",plan["expected_rows"],digest(output/"protocol.json"),flush=True)
    elif args.execute:
        execute(output)
    else:
        sys.exit(0 if score(output) else 1)
