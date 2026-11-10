"""Register three prospective cache-policy windows from complete exact T72 qualification."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from types import FunctionType

import freeze_rank_fair_timing as rank
import rank_fair_extended_qualification as qual
from rank_fair_owned_transport import SECOND,read,require


HERE,WORK,T,timing = qual.HERE,qual.WORK,qual.T,qual.original.timing
SOURCE = Path(__file__).resolve()
WORKER = HERE/"rank_fair_cache_formal_worker.py"
VERIFIER = HERE/"verify_rank_fair_cache_formal.py"
FIXTURE = HERE/"rank_fair_cache_formal_fixture.json"
WINDOWS = ("mid_b1_b5","high_b1_b3","high_b4_b5")
CUDA_ENV = ("CUDA_CACHE_DISABLE","CUDA_CACHE_PATH","CUDA_CACHE_MAXSIZE","CUDA_FORCE_PTX_JIT","CUDA_FORCE_JIT",
            "CUDA_MODULE_LOADING","CUDA_LAUNCH_BLOCKING","CUDA_VISIBLE_DEVICES","CUDA_DISABLE_PTX_JIT","CUDA_DEVICE_ORDER")


def environment():
    value = rank.frozen.environment_identity()
    settings = {name:os.environ.get(name) for name in CUDA_ENV if name != "CUDA_CACHE_PATH"}
    path = os.environ.get("CUDA_CACHE_PATH")
    settings["CUDA_CACHE_PATH_sha256"] = sha256(path.encode("utf8")).hexdigest() if path is not None else None
    value["cuda_environment"] = settings
    return value


def load_qualification(directory):
    protocol,proof = directory/"protocol.json",directory/"qualification_summary.json"
    p,s = map(read,(protocol,proof))
    qual.verify_registered_plan(p)
    psha = T.digest(protocol)
    require(s["status"] == "passed" and s["protocol_sha256"] == psha and len(s["rows"]) == 78 and
            s["flow_solves_run_by_audit"] == 0 and not s["speed_claim_supported"] and not s["missing"] and
            len(s["policy_pairs"]) == 39 and all(r["identity_index"] == i and r["passed"] for i,r in enumerate(s["policy_pairs"])),
            "All 78 exact qualifications and all 39 policy pairs are required")
    sources = dict(p["sources_sha256"])
    cap_path = qual.shared.bound(s["capture_summary"])
    cap = read(cap_path)
    require(cap["status"] == "complete" and cap["protocol_sha256"] == psha and not cap["unattempted"] and len(cap["rows"]) == 78,
            "GPU qualification was interrupted")
    bind = qual.original.bind
    bind(sources,qual.shared.record(protocol),qual.shared.record(proof),s["capture_summary"],*s["lanes"])
    rows = []
    for i,(j,srow,cap_record) in enumerate(zip(p["slots"],s["rows"],cap["rows"],strict=True)):
        score = read(qual.shared.bound(srow["score"]))
        raw = read(qual.shared.bound(cap_record))
        require(srow["index"] == score["index"] == raw["index"] == i and srow["qualified"] and score["qualified"] and
                score["job"] == raw["job"] == j and srow["source_capture"] == cap_record == score["capture"] and
                raw["status"] == "complete" and score["protocol_sha256"] == raw["protocol_sha256"] == psha,
                "A qualifying raw artifact or its CPU score changed")
        result = read(qual.shared.bound(raw["result"]))
        bind(sources,srow["score"],cap_record,raw["result"],raw["artifact"],raw["phases"],raw["watchdog"])
        if raw["fresh_artifact"] is not None:
            bind(sources,raw["fresh_artifact"])
        rows.append(dict(index=i,job=j,artifact=raw["artifact"],score=srow["score"],result=raw["result"],spatial_sha256=result["spatial_sha256"]))
    return p,rows,sources,dict(protocol=qual.shared.record(protocol),summary=qual.shared.record(proof))


def requests_for(qualification,window):
    jobs = []
    for entry in window["schedule"]:
        control = qualification["slots"][entry["qualification_index"]]
        require(all(control[k] == entry[k] for k in ("name","arm","seed","policy")),"A formal slot was assigned a different exact qualification")
        args = dict(control["args"],snapshots=qualification["physical_times"])
        common = {k:control[k] for k in ("family","engine","config","ptx_artifact","ptx_cache_key","ptx_sha256","runtime_id")}
        jobs.append(dict(entry,**common,imports="minimal",solve_args=args,compiler_identity=qualification["compiler_identity"]))
    return jobs


def evidence(plan):
    return {name:plan[name] for name in ("qualification","environment","source_hashes")}


def make_window(name,directory,env=None):
    qp,controls,sources,qualified = load_qualification(directory)
    spec = next(w for w in qp["windows"] if w["name"] == name)
    paths = (SOURCE,WORKER,VERIFIER,FIXTURE,HERE/"verify_rank_fair_cache_formal_fixture.py",HERE/"rank_fair_cache_worker.py",
             HERE/"verify_rank_fair_timing.py",HERE/"rank_fair_owned_transport.py",HERE/"rank_fair_solver_worker.py",
             HERE/"rank_fair_coordinator.py",rank.SMOKE,rank.CHECKER_FIXTURE)
    names = T.REQUIRED_SOURCES|{p.relative_to(WORK).as_posix() for p in paths}
    qual.original.merge(sources,{name:T.digest(WORK/name) for name in names})
    fixture = read(FIXTURE)
    require(fixture["status"] == "passed" and fixture["gpu_contexts_created"] == 0 and fixture["formal_source_sha256"] == T.digest(SOURCE) and
            fixture["worker_sha256"] == T.digest(WORKER) and fixture["verifier_sha256"] == T.digest(VERIFIER) and
            fixture["qualification_sha256"] == qualified["summary"]["sha256"],"Formal transport lacks its matching CPU fixture")
    env = environment() if env is None else env
    require(env["cuda_runtime_sha256"]["nvcuda.dll"] == qp["nvcuda_sha256"] and all(
        env["cuda_runtime_sha256"][k] == v for k,v in qp["toolkit_sha256"].items()),"The live runtime differs from its exact qualification")
    design = read(qual.original.THIN_DESIGN)
    deadlines = read(qual.original.DESIGN)["deadlines_seconds"]
    plan = dict(schema=timing.SCHEMA,status="frozen",mode="formal",study="rank_fair_two_cache_policies",window=name,scale=spec["scale"],
        design=design,window_spec=spec,coordinator_cpu=4,monitor_cpu=6,poll_interval_ns=SECOND,max_poll_gap_ns=5*SECOND//2,
        budget_interval_ns=10*SECOND,max_budget_lateness_ns=2*SECOND,maximum_active_ns=720*SECOND,
        known_background=design["monitor"]["known_background"],ready_timeout_seconds=deadlines["ready"],
        return_timeout_seconds=deadlines["solve_to_return"],exit_timeout_seconds=deadlines["save_to_exit"],
        slot_hard_timeout_seconds=deadlines["whole_slot_from_popen"],source_hashes=sources,environment=env,qualification=qualified,
        controls=[controls[j["qualification_index"]] for j in spec["schedule"]])
    jobs = requests_for(qp,spec)
    token = rank.frozen.canonical_sha(dict(plan,jobs=jobs))
    plan.update(run_token=token,requests=[dict(job,index=i,schema=timing.WORKER_SCHEMA,run_token=token,affinity_mask=16,
                                             trajectory=f"trajectory_{i:03d}.npz") for i,job in enumerate(jobs)])
    return plan


def verify_plan(plan,*,live=False):
    require(plan["study"] == "rank_fair_two_cache_policies" and plan["window"] in WINDOWS,"Unexpected cache timing cohort")
    T.check_sources(plan["source_hashes"])
    directory = qual.shared.bound(plan["qualification"]["protocol"]).parent
    actual = environment() if live else plan["environment"]
    require(plan == make_window(plan["window"],directory,actual),"The clock, environment, schedule or exact qualification changed")


def coordinator():
    # Preserve the reviewed implementation; substitute only the registered solver entry.
    f = timing.run
    bound = FunctionType(f.__code__,dict(f.__globals__,SOLVER_WORKER=WORKER),f.__name__,f.__defaults__,f.__closure__)
    bound.__kwdefaults__ = f.__kwdefaults__
    return bound


def execute(protocol,output):
    raw = protocol.read_bytes()
    plan = json.loads(raw)
    verify_plan(plan,live=True)
    before = time.perf_counter_ns()
    summary = coordinator()(protocol,output,plan,raw,worker=WORKER)
    after = time.perf_counter_ns()
    guard = dict(status="failed",protocol_sha256=sha256(raw).hexdigest(),opening_environment=plan["environment"],
                 opening_guard_qpc_ns=before,closing_guard_qpc_ns=after,coordinator_summary_sha256=T.digest(output/"summary.json"))
    try:
        T.check_sources(plan["source_hashes"])
        require(environment() == plan["environment"] and protocol.read_bytes() == raw == (output/"protocol.json").read_bytes(),
                "Frozen runtime or protocol changed over the registered window")
        guard.update(status="passed",closing_environment=plan["environment"])
    except Exception as error:
        guard.update(failure_type=type(error).__name__)
    T.save(output/"execution_environment_guard.json",guard)
    return dict(status=summary["status"],environment_guard=guard["status"],window=plan["window"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-all",type=Path)
    actions.add_argument("--execute",type=Path)
    cli.add_argument("--qualification",type=Path)
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    if args.execute:
        if args.output is None:
            cli.error("Execution needs a new window directory")
        print(json.dumps(execute(args.execute.resolve(),args.output.resolve())))
    else:
        if args.qualification is None:
            cli.error("Complete exact T72 qualification required")
        env = environment()
        plans = [make_window(name,args.qualification.resolve(),env) for name in WINDOWS]
        require(all(evidence(p) == evidence(plans[0]) for p in plans),"Window evidence must share one exact scope")
        root = args.freeze_all.resolve()
        root.mkdir()
        for p in plans:
            T.save(root/(p["window"]+".json"),p)
