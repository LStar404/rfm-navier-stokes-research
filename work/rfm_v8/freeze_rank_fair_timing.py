"""Register stock-only mid/high cohorts after exact-runtime numerical admission."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import freeze_thin_formal as frozen
import rank_fair_coordinator as timing
import rank_fair_stock_qualification as qualify
from rank_fair_owned_transport import SECOND, read, require


HERE,WORK,T = timing.HERE,timing.WORK,timing.prior
VERIFIER = HERE/"verify_rank_fair_timing.py"
SMOKE = HERE/"rank_fair_transport_smoke_v3_output"/"certificate.json"
CHECKER_FIXTURE = HERE/"rank_fair_verifier_fixture_v2.json"
NORMS = {"velocity","vorticity","near_velocity","near_vorticity"}
REFERENCES = {"angular_fine","time_fine"}


def validate_norms(row, limit):
    require(set(row) == NORMS and all(0 <= value <= limit for value in row.values()),"Frozen physical or implementation norm failed")


def validate_table(rows,times,worst,limit):
    targets = {(g,t) for g in ("gauss123","mid129","layer144") for t in times}
    require(len(rows) == len(targets) and {(r["grid"],r["time"]) for r in rows} == targets,"Norm table does not cover exact common times")
    for row in rows:
        validate_norms({k:v for k,v in row.items() if k not in ("grid","time")},limit)
    require({metric:max(row[metric] for row in rows) for metric in NORMS} == worst,"Norm table disagrees with the saved maxima")


def load_controls(directory):
    protocol,summary = directory/"protocol.json",directory/"summary.json"
    q,s = read(protocol),read(summary)
    require(q == qualify.build(toolkit_hashes=q["toolkit_sha256"]),"Qualification settings or sources differ from their reviewed design")
    require(s["status"] == "passed" and s["protocol_sha256"] == T.digest(protocol) and not s["unattempted"] and
            not s["isolated_speed_claim_supported"] and len(s["rows"]) == len(q["slots"]) == 28,
            "All 28 exact stock identities must pass before a formal freeze")
    sources = dict(q["sources_sha256"])
    qualify.bind(sources,qualify.shared.record(protocol),qualify.shared.record(summary))
    controls = []
    for i,(job,row) in enumerate(zip(q["slots"],s["rows"],strict=True)):
        slot = directory/f"slot_{i:03d}"
        sidecar = slot/"qualification.json"
        require(i == row["index"] and job == row["job"] and row == read(sidecar) and row["qualified"] and
                row["protocol_sha256"] == s["protocol_sha256"] and row["watchdog"] == read(slot/"watchdog.json") and
                row["watchdog"]["status"] == "complete","Exact qualification sidecar failed")
        require(set(row["physical_worst"]) == REFERENCES and set(row["guard_worst"]) == (REFERENCES if job["arm"] == "combined" else set()) and
                row["wake"]["sustained_alternating_probe_signs"],"Independent physical references, bridge guard or wake failed")
        for norms in (*row["physical_worst"].values(),*row["guard_worst"].values()):
            validate_norms(norms,.005)
        validate_norms(row["implementation_worst"],1e-5)
        require(set(row["tables"]) == REFERENCES and row["reference_budget"]["budget"] == .00125 and all(
            0 <= row["reference_budget"][name] <= .00125 for name in ("early_bound","late_bound")),"Reference budget did not qualify")
        validate_table(row["implementation_rows"],job["args"]["snapshots"],row["implementation_worst"],1e-5)
        for name,table in row["tables"].items():
            validate_table(table["direct"],q["physical_times"],row["physical_worst"][name],.005)
            if job["arm"] == "combined":
                validate_table(table["guard"],q["physical_times"],row["guard_worst"][name],.005)
            else:
                require(not table["guard"],"Uniform or FD control unexpectedly has a bridge guard")
        strict = row["strict_observer"]
        require(set(row["probes"]) == set(strict["probes"]) == set(qualify.shared.HISTORY) and
                all(0 <= value <= 1e-5 for value in row["probes"].values()) and strict["saved_states_bitwise"] and
                strict["common_times"] >= 19 and all(0 <= value <= 1e-12 for pair in strict["probes"].values() for value in pair.values()),
                "Full probe or strict observer gate failed")
        qualify.bind(sources,qualify.shared.record(sidecar),qualify.shared.record(slot/"watchdog.json"),row["result"],row["artifact"])
        require(qualify.shared.bound(row["result"]) == slot/"result.json" and qualify.shared.bound(row["artifact"]) == slot/"trajectory.npz",
                "Raw exact-runtime control was moved or remapped")
        result = read(slot/"result.json")
        require(result["index"] == i and result["job"] == job and result["trajectory_sha256"] == row["artifact"]["sha256"] and
                result["protocol_sha256"] == s["protocol_sha256"],"Raw exact qualification result differs")
        controls.append(dict(index=i,job=job,spatial_sha256=result["spatial_sha256"],artifact=row["artifact"],
                             sidecar=qualify.shared.record(sidecar),result=row["result"]))
    return q,controls,sources,dict(protocol=qualify.shared.record(protocol),summary=qualify.shared.record(summary))


def design_jobs(scale, controls):
    design,thind,bounded = map(read,(qualify.PARENT_DESIGN,qualify.THIN_DESIGN,qualify.DESIGN))
    frozen.validate_design(thind)
    require(bounded["engine"] == "warp_cufft" and bounded["imports"] == "minimal" and bounded["observer"] == "warp" and
            len(design["rfm_seeds"]) == 5,"Stock timing design is not rank-fair")
    spec = next(s for s in design["scales"] if s["name"] == scale)
    indexed = {qualify.order(c["job"]):c for c in controls if scale in c["job"]["applies_to_scales"]}
    require(len(indexed) == bounded["scales"][scale]["unique_controls"],"Exact qualification scale coverage differs")
    groups = [(name,"fd" if arm in ("dense","band") else arm,"minimal") for name,arm in spec["groups"]]
    rng = random.Random(spec["order_seed"])
    jobs,slots,pairs = [],[],[]
    for block,rfm_seed in enumerate(design["rfm_seeds"],1):
        order = groups.copy()
        rng.shuffle(order)
        for name,arm,imports in order:
            seed = rfm_seed if name.startswith("rfm") else design["deterministic_seed"]
            control = indexed[name,arm,seed]
            job = control["job"]
            args = dict(job["args"],snapshots=thind["physical_times"])
            jobs.append(dict(block=block,name=name,arm=arm,imports=imports,seed=seed,family=job["family"],
                             engine=job["engine"],config=job["config"],solve_args=args))
            slots.append(control["index"])
    require(len(jobs) == spec["expected_rows_if_all_qualified"] == bounded["scales"][scale]["slots"],"Rank-fair slot count differs")

    def add(kind,baseline,candidate):
        require(baseline in groups and candidate in groups,"Contrast uses an unregistered arm")
        pairs.append(dict(kind=kind,baseline=list(baseline),candidate=list(candidate)))

    names = [name for name,arm,imports in groups if arm == "uniform"]
    rfm,matched = names[:2]
    fds = [g for g in groups if g[1] == "fd"]
    for name in names:
        add("schedule_and_preparation",(name,"uniform","minimal"),(name,"combined","minimal"))
    for arm in ("uniform","combined"):
        add("same_retained_rank_descriptive",(matched,arm,"minimal"),(rfm,arm,"minimal"))
        if scale == "high":
            cheaper = names[2]
            add("rfm_vs_lower_rank_legendre",(cheaper,arm,"minimal"),(rfm,arm,"minimal"))
            add("legendre_rank_cost",(matched,arm,"minimal"),(cheaper,arm,"minimal"))
        for fd in fds:
            for name in names:
                add("fd_vs_galerkin",fd,(name,arm,"minimal"))
    if scale == "high":
        add("fd_scale_and_linear_layout_descriptive",fds[1],fds[0])
    return jobs,slots,[list(g) for g in groups],pairs


def discover_imports():
    command = """import json,sys
from pathlib import Path
from rank_fair_solver_worker import select_solver
root=Path.cwd().parent.resolve()
select_solver(dict(imports='minimal',engine='warp_cufft',arm=sys.argv[1]))
files=set()
for module in tuple(sys.modules.values()):
    name=getattr(module,'__file__',None)
    if name:
        path=Path(name).resolve()
        if path.is_relative_to(root):
            files.add(path.relative_to(root).as_posix())
print(json.dumps(sorted(files)))
"""
    env = dict(os.environ,**{key:"1" for key in T.THREADS})
    closures = {}
    for arm in ("uniform","combined","fd"):
        child = subprocess.run([sys.executable,"-B","-c",command,arm],cwd=HERE,env=env,capture_output=True,text=True,
                               check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        files = json.loads(child.stdout)
        require({"rfm_v8/fused_marches.py","rfm_v8/galerkin_minimal_model.py","rfm_v8/rank_fair_solver_worker.py"} <= set(files) and
                not any("qualification" in p or "formal_timing" in p or "report" in p for p in files),
                "Timed worker import closure unexpectedly contains scoring or a nonminimal Galerkin model")
        closures[arm] = files
    return closures


def build(scale, directory):
    q,controls,sources,qualification = load_controls(directory)
    closures = discover_imports()
    names = T.REQUIRED_SOURCES|{p.relative_to(WORK).as_posix() for p in (
        Path(__file__),VERIFIER,SMOKE,CHECKER_FIXTURE,HERE/"verify_rank_fair_timing_fixture.py",
        HERE/"rank_fair_owned_transport.py",HERE/"rank_fair_coordinator.py",
        HERE/"rank_fair_transport_smoke.py",timing.SOLVER_WORKER)}|{file for values in closures.values() for file in values}
    qualify.merge(sources,{name:T.digest(WORK/name) for name in sorted(names)})
    smoke = read(SMOKE)
    require(smoke["status"] == "passed" and smoke["time_solves_run"] == 0 and
            [r["mode"] for r in smoke["test_cases"]] == ["success","before_ready","before_return","after_save"],
            "Bounded transport has no complete passing CPU fixture")
    for record in smoke["test_cases"]:
        directory = SMOKE.parent/record["mode"]
        source_plan,source_summary = (directory/name for name in ("protocol.json","summary.json"))
        require(T.digest(source_summary) == record["summary_sha256"] and read(source_summary)["protocol_sha256"] == T.digest(source_plan),
                "Bounded smoke raw evidence changed")
        T.check_sources(read(source_plan)["source_hashes"])
        qualify.merge(sources,{p.relative_to(WORK).as_posix():T.digest(p) for p in (source_plan,source_summary)})
    checker = read(CHECKER_FIXTURE)
    require(checker["status"] == "passed" and checker["time_solves_run"] == 0 and checker["gpu_contexts_created"] == 0 and
            all(checker["expected_rejections"].values()) and checker["verifier_source_sha256"] == T.digest(VERIFIER) and
            checker["transport_certificate_sha256"] == T.digest(SMOKE),"Independent verifier has no current CPU fixture")
    jobs,slots,groups,pairs = design_jobs(scale,controls)
    thin_design,draft = read(qualify.THIN_DESIGN),read(qualify.DESIGN)
    m,d = thin_design["monitor"],draft["deadlines_seconds"]
    environment = frozen.environment_identity()
    require(all(environment["cuda_runtime_sha256"][name] == digest for name,digest in q["toolkit_sha256"].items()),
            "Timed runtime differs from its qualification toolkit")
    plan = dict(schema=timing.SCHEMA,status="frozen",mode="formal",scale=scale,design=thin_design,
        source_hashes=sources,qualification=qualification,controls=controls,slot_controls=slots,groups=groups,paired=pairs,
        environment=environment,observed_worker_imports=closures,coordinator_cpu=m["coordinator_cpu"],monitor_cpu=m["monitor_cpu"],
        poll_interval_ns=int(m["poll_interval_seconds"]*SECOND),max_poll_gap_ns=int(m["max_poll_gap_seconds"]*SECOND),
        budget_interval_ns=int(m["budget_interval_seconds"]*SECOND),max_budget_lateness_ns=int(m["max_budget_lateness_seconds"]*SECOND),
        maximum_active_ns=m["maximum_active_seconds"]*SECOND,known_background=m["known_background"],
        ready_timeout_seconds=d["ready"],return_timeout_seconds=d["solve_to_return"],exit_timeout_seconds=d["save_to_exit"],
        slot_hard_timeout_seconds=d["whole_slot_from_popen"])
    plan["run_token"] = frozen.canonical_sha(dict(plan,jobs=jobs))
    plan["requests"] = [dict(job,index=i,schema=timing.WORKER_SCHEMA,run_token=plan["run_token"],
                             affinity_mask=1<<m["coordinator_cpu"],trajectory=f"trajectory_{i:03d}.npz") for i,job in enumerate(jobs)]
    return plan


def verify_frozen(plan, *, live_environment=False):
    T.check_sources(plan["source_hashes"])
    require(plan["schema"] == timing.SCHEMA and plan["mode"] == "formal" and plan["status"] == "frozen" and
            plan["design"] == read(qualify.THIN_DESIGN),"The registered timing protocol differs")
    mon = plan["design"]["monitor"]
    require((plan["coordinator_cpu"],plan["monitor_cpu"],plan["poll_interval_ns"],plan["max_poll_gap_ns"],plan["budget_interval_ns"],
             plan["max_budget_lateness_ns"],plan["maximum_active_ns"],plan["known_background"]) ==
            (4,6,SECOND,5*SECOND//2,10*SECOND,2*SECOND,720*SECOND,mon["known_background"]),"Isolation policy changed")
    d = read(qualify.DESIGN)["deadlines_seconds"]
    require((plan["ready_timeout_seconds"],plan["return_timeout_seconds"],plan["exit_timeout_seconds"],plan["slot_hard_timeout_seconds"]) ==
            (d["ready"],d["solve_to_return"],d["save_to_exit"],d["whole_slot_from_popen"]),"Registered hard deadline changed")
    directory = qualify.shared.bound(plan["qualification"]["protocol"]).parent
    q,controls,_,qualification = load_controls(directory)
    require(controls == plan["controls"] and qualification == plan["qualification"] and all(
        plan["environment"]["cuda_runtime_sha256"][name] == digest for name,digest in q["toolkit_sha256"].items()),
        "Admission controls or toolkit differ from their frozen source")
    jobs,slots,groups,pairs = design_jobs(plan["scale"],controls)
    base = {name:value for name,value in plan.items() if name not in ("requests","run_token")}
    token = frozen.canonical_sha(dict(base,jobs=jobs))
    expected = [dict(job,index=i,schema=timing.WORKER_SCHEMA,run_token=token,affinity_mask=16,
                     trajectory=f"trajectory_{i:03d}.npz") for i,job in enumerate(jobs)]
    require((slots,groups,pairs,token,expected) == (plan["slot_controls"],plan["groups"],plan["paired"],plan["run_token"],plan["requests"]),
            "Ordered block slots, pair definitions or cohort identity differ")
    if live_environment:
        require(frozen.environment_identity() == plan["environment"],"Live environment differs from frozen hardware/runtime")
    return q,directory


def execute(protocol, output):
    raw = protocol.read_bytes()
    plan = json.loads(raw)
    verify_frozen(plan,live_environment=True)
    before = time.perf_counter_ns()
    summary = timing.run(protocol,output,plan,raw)
    after = time.perf_counter_ns()
    guard = dict(status="failed",protocol_sha256=sha256(raw).hexdigest(),opening_environment=plan["environment"],
                 opening_guard_qpc_ns=before,closing_guard_qpc_ns=after,coordinator_summary_sha256=T.digest(output/"summary.json"))
    try:
        T.check_sources(plan["source_hashes"])
        require(protocol.read_bytes() == raw == (output/"protocol.json").read_bytes() and frozen.environment_identity() == plan["environment"],
                "Protocol or runtime changed across the timing window")
        guard.update(status="passed",closing_environment=plan["environment"])
    except Exception as error:
        guard.update(failure_type=type(error).__name__)
    T.save(output/"execution_environment_guard.json",guard)
    return dict(status=summary["status"],environment_guard=guard["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze",choices=("mid","high"))
    action.add_argument("--execute",type=Path)
    parser.add_argument("--qualification",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(execute(args.execute.resolve(),args.output.resolve())))
    else:
        if args.qualification is None:
            parser.error("A completed exact-runtime T72 qualification is required to freeze a cohort")
        plan = build(args.freeze,args.qualification.resolve())
        verify_frozen(plan)
        T.save(args.output,plan)
        print(json.dumps(dict(scale=plan["scale"],slots=len(plan["requests"]),sources=len(plan["source_hashes"]))))
