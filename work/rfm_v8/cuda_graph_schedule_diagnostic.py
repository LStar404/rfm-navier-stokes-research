"""Change only the graph chunk, preserving the full T72 CUDA trajectory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
V7 = HERE.parent / "rfm_v7"
OUT = HERE / "cuda_graph_schedule_diagnostic_output"
TEMPLATE = HERE / "protocol_cuda_graph_schedule.json"
ARRAYS = ("states","times","probe_times","transverse","vorticity")
MARKER = b"V8_GRAPH_RETURN "
THREADS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")
PHASES = ("host_setup_seconds","runtime_init_seconds","upload_seconds","plan_seconds","graph_setup_seconds",
          "march_seconds","download_seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path,data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def fingerprint(data):
    return {name:dict(shape=list(data[name].shape),dtype=data[name].dtype.str,
                      sha256=hashlib.sha256(data[name].tobytes()).hexdigest()) for name in ARRAYS}


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def stem(job):
    return f"block{job['block']}_{job['name']}_g{job['graph_steps']}"


def reference_bytes(path,times,np):
    with np.load(path,allow_pickle=False) as raw:
        indices = [raw["times"].tolist().index(t) for t in times]
        arrays = {k:raw[k][indices] if k in ("states","times") else raw[k] for k in ARRAYS}
        require(np.array_equal(arrays["times"],times) and np.array_equal(arrays["probe_times"],np.arange(577)/8),
                "The qualified source omits common observations")
        return fingerprint(arrays)


def graph_inventory(rates,horizon,width):
    if len(rates) == 1:
        phases = [horizon*rates[0]-1]
        bridge = 0
    else:
        phases = [8*rates[0]-1,(horizon-8)*rates[1]-1]
        bridge = 1
    counts = [divmod(n,min(n,width)) for n in phases]
    return dict(graph_executables=len(phases),graph_launches=sum(q for q,_ in counts),
                eager_nonlinear_evaluations=bridge+sum(r for _,r in counts),
                nonlinear_evaluations=sum(phases)+bridge,marching_fft_batches=2*(sum(phases)+bridge))


def freeze():
    require(not OUT.exists(),"Preserve the existing graph diagnostic attempt")
    sys.path.insert(0,str(V7))
    import cpu_formal as base
    root = V7/"prepared_formal_timing_mid_output"
    frozen_path = root/"protocol.json"
    old = read(frozen_path)
    template = read(TEMPLATE)
    times = old["design"]["physical_times"]
    configs, controls, inputs = {},{},[Path(__file__),TEMPLATE,frozen_path]
    for arm in template["arms"]:
        name,seed = arm["name"],arm["seed"]
        if len(arm["rates"]) == 1:
            cfg = old["fd_controls"][name]
            configs[name],ref = cfg["config"],cfg["gpu"]
        else:
            configs[name] = base.config(old["specs"][name],old["solve_protocol"],seed)[1]
            ref = next(x["artifact"] for x in old["prepared_evidence"] if x["job"]["name"] == name and
                       x["job"]["seed"] == seed and x["job"]["preparation"] == "combined")
        raw = HERE.parent/ref["path"]
        require(digest(raw) == ref["sha256"],"Qualified graph24 reference hash differs")
        controls[name] = dict(ref,array_fingerprints=reference_bytes(raw,times,base.case.np))
        inputs.append(raw)
    # Bind the already frozen solver dependencies instead of inferring a smaller runtime closure.
    sources = {name:sha for name,sha in old["source_hashes"].items() if name.endswith((".py",".cu",".c",".dll"))}
    sources.update({p.relative_to(HERE.parent).as_posix():digest(p) for p in inputs})
    require(all(digest(HERE.parent/name) == sha for name,sha in sources.items()),"Graph dependency differs")
    rng = random.Random(template["order_seed"])
    jobs = []
    for block in range(1,template["blocks"]+1):
        cohort = [dict(arm,block=block,graph_steps=g,config=configs[arm["name"]])
                  for arm in template["arms"] for g in template["graph_steps"]]
        rng.shuffle(cohort)
        jobs.extend(cohort)
    OUT.mkdir()
    save(OUT/"protocol.json",dict(status="frozen",template=template,physical_times=times,controls=controls,
                                  source_hashes=sources,jobs=jobs))
    print("FROZEN",len(jobs),digest(OUT/"protocol.json"),flush=True)


def worker(index):
    sys.path.insert(0,str(V7))
    # Every child traverses the same legacy dependency closure.
    import prepared_formal_timing as dependencies
    from fd_cuda_native import solve_fd_native
    from galerkin_cuda_prepared import solve_cuda_prepared

    p = read(OUT/"protocol.json")
    job = p["jobs"][index]
    args = dict(job["config"],horizon=72.,snapshots=p["physical_times"],probe_spacing=.125,graph_steps=job["graph_steps"])
    if len(job["rates"]) == 1:
        solve = solve_fd_native
        args.update(steps=9216,cuda_linear_solve="dense")
    else:
        solve = solve_cuda_prepared
        args.update(preparation="combined",startup_rate=128,late_rate=112,switch_time=8.)
    started = time.perf_counter()
    solution = solve(**args)
    core = time.perf_counter()-started
    releasing = time.perf_counter()
    solution.model = None
    release = time.perf_counter()-releasing
    phases = {k:solution.row[k] for k in PHASES}
    work = graph_inventory(job["rates"],72,job["graph_steps"])
    require(solution.row["steps"]-1 == work["nonlinear_evaluations"] and
            solution.row.get("marching_fft_batches",work["marching_fft_batches"]) == work["marching_fft_batches"],
            "Graph diagnostic changed the FFT work")
    payload = dict(index=index,job=job,solver_seconds=core+release,model_release_seconds=release,phase_seconds=phases,
                   setup_seconds=sum(phases[k] for k in PHASES[:5]),work=work,
                   array_fingerprints=fingerprint({k:getattr(solution,k) for k in ARRAYS}))
    print(MARKER.decode()+json.dumps(payload,separators=(",",":"),allow_nan=False),flush=True)
    if job["block"] == 1:
        path = OUT/(stem(job)+".npz")
        with path.open("xb") as handle:
            dependencies.case.store_solution(handle,solution,job["config"])


def launch(index,job):
    env = dict(os.environ,**{k:"1" for k in THREADS},PYTHONUNBUFFERED="1")
    log = OUT/(stem(job)+".stderr.log")
    with log.open("xb") as err:
        start = time.perf_counter_ns()
        with subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(index)],
                              stdout=subprocess.PIPE,stderr=err,env=env) as child:
            for line in child.stdout:
                returned = time.perf_counter_ns()
                if line.startswith(MARKER):
                    row = json.loads(line[len(MARKER):])
                    break
            else:
                raise RuntimeError(f"Graph worker exited before its result ({child.wait(timeout=20)})")
            require(child.wait(timeout=30) == 0 and row["job"] == job and row["index"] == index,"Graph worker changed identity")
    row["fresh_process_seconds"] = (returned-start)/1e9
    require(row["fresh_process_seconds"] >= row["solver_seconds"] > 0,"Unexpected fresh-clock accounting")
    if job["block"] == 1:
        path = OUT/(stem(job)+".npz")
        row.update(artifact=path.name,artifact_sha256=digest(path))
    return row


def summarize(p,rows):
    if not all(r["status"] == "complete" for r in rows):
        return dict(status="failed",rows=len(rows))
    groups = {}
    for arm in p["template"]["arms"]:
        name = arm["name"]
        data = {(r["job"]["block"],r["job"]["graph_steps"]):r for r in rows if r["job"]["name"] == name}
        groups[name] = {}
        for g in p["template"]["graph_steps"]:
            samples = []
            for block in range(1,4):
                original,candidate = (data[block,k] for k in (24,g))
                a,b = (dict(r,**r["phase_seconds"]) for r in (original,candidate))
                keys = (*PHASES,"solver_seconds","setup_seconds","fresh_process_seconds")
                samples.append(dict(block=block,delta={k:b[k]-a[k] for k in keys},
                                    ratio={k:a[k]/b[k] for k in ("march_seconds","solver_seconds","fresh_process_seconds")}))
            deltas = {k:median(v["delta"][k] for v in samples) for k in samples[0]["delta"]}
            ratios = {k:median(v["ratio"][k] for v in samples) for k in samples[0]["ratio"]}
            screen = g != 24 and all(v["ratio"]["march_seconds"] > 1 for v in samples) and all(
                ratios[k] >= 1.02 for k in ("march_seconds","solver_seconds")) and ratios["fresh_process_seconds"] >= 1
            groups[name][str(g)] = dict(inventory=data[1,g]["work"],samples=samples,median_deltas=deltas,
                                        median_ratios=ratios,meets_diagnostic_promotion_screen=screen)
    passed = all(r["bitwise_qualified_baseline"] for r in rows)
    return dict(status="passed" if passed else "failed",rows=len(rows),all_archived_arrays_exact=passed,groups=groups)


def run():
    require(not any((OUT/k).exists() for k in ("started.json","runs.json","summary.json")),"Preserve the existing graph attempt")
    p = read(OUT/"protocol.json")
    hashes = p["source_hashes"]
    require(all(digest(HERE.parent/name) == sha for name,sha in hashes.items()),"Graph source changed after freeze")
    with (OUT/"started.json").open("x",encoding="utf-8") as marker:
        json.dump(dict(status="started",protocol_sha256=digest(OUT/"protocol.json")),marker)
    rows = []
    for i,job in enumerate(p["jobs"]):
        try:
            row = launch(i,job)
            row.update(status="complete",bitwise_qualified_baseline=(row["array_fingerprints"] ==
                p["controls"][job["name"]]["array_fingerprints"]))
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            row = dict(index=i,job=job,status="failed",failure_type=type(exc).__name__)
        rows.append(row)
        save(OUT/"runs.json",dict(status="running",rows=rows))
        print(i+1,job["name"],job["graph_steps"],row["status"],row.get("bitwise_qualified_baseline"),flush=True)
    save(OUT/"runs.json",dict(status="complete",rows=rows))
    require(all(digest(HERE.parent/name) == sha for name,sha in hashes.items()),"Graph source changed during diagnostic")
    result = summarize(p,rows)
    save(OUT/"summary.json",dict(result,protocol_sha256=digest(OUT/"protocol.json"),runs_sha256=digest(OUT/"runs.json"),
                                isolated_speed_claim_supported=False))
    require(result["status"] == "passed","Graph diagnostic kept a failed row")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze",action="store_true")
    action.add_argument("--run",action="store_true")
    action.add_argument("--worker",type=int)
    args = cli.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        worker(args.worker)
