"""Prepare a thin-worker timing protocol without running any flow solve."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import formal_timing_coordinator as transport
import thin_t72_qualification as thin


HERE, WORK = transport.HERE, transport.WORK
DESIGN = HERE/"protocol_thin_formal_design.json"
QUAL = HERE/"thin_t72_qualification_output"
FD_NAMES = {"fd176_dense":"fd176_k80","fd184_band":"fd184_k80"}
LIBRARIES = ("nvrtc64_112_0.dll","nvrtc-builtins64_118.dll","cufft64_10.dll")
SOURCES = transport.REQUIRED_SOURCES | set(thin.RUNTIME_FILES) | {
    "rfm_v8/freeze_thin_formal.py","rfm_v8/verify_thin_formal.py","rfm_v8/protocol_thin_formal_design.json",
    "rfm_v8/formal_timing_transport_protocol.md","rfm_v8/win32_isolation_protocol.md",
    "rfm_v8/formal_transport_smoke_final_output/certificate.json","rfm_v8/win32_isolation_monitor_audit.json",
    "rfm_v8/verify_thin_formal_fixture.py","rfm_v8/thin_formal_checker_fixture_output/certificate.json",
    "rfm_v8/thin_t72_qualification.py","rfm_v7/prepared_formal_timing.py","rfm_v7/verify_prepared_formal_timing.py",
}
require = transport.require


def read(path):
    return json.loads(path.read_bytes())


def array_identity(array):
    return dict(shape=list(array.shape),dtype=array.dtype.str,sha256=sha256(array.tobytes()).hexdigest())


def canonical_sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")).hexdigest()


def merge(target,mapping):
    for name,sha in mapping.items():
        require(target.get(name,sha)==sha,f"Frozen dependency identities conflict for {name}")
        target[name]=sha


def bind_records(target,records):
    for record in records:
        merge(target,{record["path"]:record["sha256"]})


def validate_design(d):
    require(d["status"]=="design_only_not_a_run_protocol" and d["blocks"]==5 and tuple(d["rfm_seeds"])==thin.RFM and
            d["deterministic_seed"]==17 and d["physical_gate"]==.005 and d["implementation_gate"]==1e-5 and
            len(d["physical_times"])==19 and d["probe_spacing"]==.125 and d["graph_steps"]==24,
            "Formal physical controls differ")
    require([(s["name"],s["rows"],s["fd_controls"],s["rates"]) for s in d["scales"]]==
            [("mid",45,["fd176_dense"],[128,112]),("high",50,["fd176_dense","fd184_band"],[152,136])],
            "Formal scale coverage differs")
    arms=d["galerkin_arms_each_family"]
    require(len(arms)==4 and {(a["arm"],a["imports"]) for a in arms}==
            {(a,m) for a in ("uniform","combined") for m in ("selected","minimal")},"Formal arms differ")
    m=d["monitor"]
    keys=("coordinator_cpu","monitor_cpu","poll_interval_seconds","max_poll_gap_seconds","budget_interval_seconds",
          "max_budget_lateness_seconds","preflight_windows","preflight_gpu_percent_max","external_python_cpu_seconds_max",
          "known_background_individual_cpu_seconds_max","known_background_total_cpu_seconds_max","maximum_active_seconds")
    require([m[k] for k in keys]==[4,6,1,2.5,10,2,2,5,.2,.5,1.,720] and
            m["known_background"]==[{"pid":68656,"birth_filetime_100ns":134387082580552623}],"Monitor policy differs")
    r=d["reporting"]
    require(r["attempts_per_scale"]==1 and r["practical_improvement_ratio"]==1.05 and
            r["consistent_wins_required_per_clock"]==5 and r["both_clocks_required_for_overall_speed_claim"],
            "Registered practical rule differs")


def load_controls(design):
    """Recompute all 130 array identities from the archived thin qualification."""
    import numpy as np

    paths=[QUAL/name for name in ("protocol.json","runs.json","qualification.json")]
    p,raw,q=map(read,paths)
    thin.verify_plan(p)
    protocol,runs,report=map(thin.bind,paths)
    references={name:item["reference_protocol"] for name,item in p["archive_contexts"].items()}
    require(raw["status"]=="complete" and q["status"]=="passed" and raw["protocol_sha256"]==q["protocol_sha256"]==protocol["sha256"] and
            q["runs_sha256"]==runs["sha256"] and len(p["jobs"])==len(raw["rows"])==len(q["rows"])==26 and
            q["independent_references"]==references,"Thin qualification evidence differs")
    sources=dict(p["source_hashes"])
    bind_records(sources,(protocol,runs,report))
    for context in p["archive_contexts"].values():
        merge(sources,context["old_source_hashes"])
        bind_records(sources,(*context["archive"].values(),context["reference_protocol"]))
    controls=[]
    for i,(job,run,score) in enumerate(zip(p["jobs"],raw["rows"],q["rows"],strict=True)):
        request=job["request"]
        identity={key:job[key] for key in ("scale","name","seed","arm","applies_to_scales")}
        require(request["index"]==run["index"]==i and score["job"]==identity and run["status"]=="complete" and
                score["qualified"] and score["gate"]=="inherited_exact_v7_numerical_gate" and score["non_time_metadata_equal"] and
                score["spatial_signature_equal"] and run["marker"]["spatial_sha256"]==job["expected_spatial_sha256"] and
                set(score["arrays"])==set(thin.ARRAYS),"Thin numerical control failed its frozen match")
        req,sidecar=QUAL/f"request_{i:03d}.json",QUAL/f"run_{i:03d}.json"
        require(read(req)==request and read(sidecar)==run and run["trajectory_sha256"]==score["trajectory_sha256"],
                "Thin control raw journal or request differs")
        artifact=thin.bind(QUAL/request["trajectory"],run["trajectory_sha256"])
        archived=read(thin.check_record(job["archive_sidecar"]))
        require(archived["qualified"] and archived["wake"]["sustained_alternating_probe_signs"] and
                all(max(row.values())<=design["physical_gate"] for key in ("physical_worst","guard_worst")
                    for row in archived[key].values()) and
                all(result["passed"] and all(0<=value<=design["implementation_gate"]
                    for value in (*result["worst"].values(),*result["probes"].values())) for result in archived["implementation"].values()),
                "Inherited control does not satisfy the original field and probe gates")
        with np.load(WORK/artifact["path"],allow_pickle=False) as data:
            metadata=thin.non_time_metadata(json.loads(str(data["metadata"])))
            require(json.loads(str(data["config"]))==request["config"] and metadata==thin.non_time_metadata(archived["metadata"]) and
                    np.array_equal(data["times"],design["physical_times"]) and np.array_equal(data["probe_times"],np.arange(577)/8),
                    "Qualified observations, configuration or metadata differ")
            arrays={key:array_identity(data[key]) for key in thin.ARRAYS}
            require(all(score["arrays"][key]["byte_exact"] and arrays[key]==
                        {field:score["arrays"][key][field] for field in ("shape","dtype","sha256")} and
                        np.isfinite(data[key]).all() for key in arrays),"Qualified control array digest differs")
        sidecar_record=thin.bind(sidecar)
        bind_records(sources,(artifact,sidecar_record,thin.bind(req,run["request_sha256"]),job["archive_artifact"],job["archive_sidecar"]))
        controls.append(dict(index=i,job=identity,config=job["config"],solve_args=job["solve_args"],family=job["family"],
            artifact=artifact,qualified_sidecar=sidecar_record,archive_artifact=job["archive_artifact"],archive_sidecar=job["archive_sidecar"],
            archive_protocol=p["archive_contexts"][job["scale"]]["archive"]["protocol"],archive_job=job["archive_job"],
            arrays=arrays,spatial_sha256=job["expected_spatial_sha256"],non_time_metadata_sha256=canonical_sha(metadata)))
    return controls,sources,dict(protocol=protocol,runs=runs,qualification=report,independent_references=references,
                                 qualified_rows=26,array_digests_verified=130)


def discover_imports():
    script="""import json,sys
from pathlib import Path
from solver_worker_formal import select_solver
root=Path.cwd().parent.resolve()
select_solver(dict(imports=sys.argv[1],arm=sys.argv[2]))
import full_case
files=set()
for module in tuple(sys.modules.values()):
    name=getattr(module,'__file__',None)
    if name:
        path=Path(name).resolve()
        if path.is_relative_to(root):
            files.add(path.relative_to(root).as_posix())
print(json.dumps(sorted(files)))
"""
    env=dict(os.environ,**{key:"1" for key in transport.THREADS})
    closures={}
    for mode in ("selected","minimal"):
        for arm in ("uniform","combined","fd"):
            done=subprocess.run([sys.executable,"-B","-c",script,mode,arm],cwd=HERE,env=env,capture_output=True,text=True,
                                check=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
            files=json.loads(done.stdout)
            require("rfm_v8/solver_worker_formal.py" in files and "rfm_v6/full_case.py" in files and
                    "rfm_v7/prepared_formal_timing.py" not in files,"Timed import closure unexpectedly includes scoring")
            closures[f"{mode}/{arm}"]=files
    return closures


def environment_identity():
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as handle:
        cpu=winreg.QueryValueEx(handle,"ProcessorNameString")[0].strip()
    args=["nvidia-smi","--query-gpu=name,driver_version,compute_cap,driver_model.current","--format=csv,noheader,nounits"]
    gpu=subprocess.run(args,capture_output=True,text=True,check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=[line.split(", ") for line in gpu.stdout.strip().splitlines()]
    require(len(rows)==1 and len(rows[0])==4,"Expected one identified NVIDIA GPU")
    toolkit=Path(os.environ.get("CUDA_PATH",r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"))
    require(toolkit.name.casefold()=="v11.8","This cohort requires the original CUDA 11.8 runtime")
    runtime={name:transport.digest(toolkit/"bin"/name) for name in LIBRARIES}
    runtime["nvcuda.dll"]=transport.digest(Path(os.environ["SystemRoot"])/"System32"/"nvcuda.dll")
    return dict(platform=platform.platform(),windows_build=sys.getwindowsversion().build,cpu_model=cpu,logical_cpu_count=os.cpu_count(),
        gpu=dict(zip(("name","driver_version","compute_capability","driver_model"),rows[0],strict=True)),
        python=dict(version=platform.python_version(),architecture=platform.machine(),executable_sha256=transport.digest(Path(sys.executable))),
        packages={name:importlib.metadata.version(name) for name in ("numpy","scipy")},cuda_toolkit="11.8",
        cuda_runtime_sha256=runtime,blas_threads=1)


def register_jobs(design,scale,controls):
    spec=next(item for item in design["scales"] if item["name"]==scale)
    indexed={(c["job"]["name"],c["job"]["seed"],c["job"]["arm"]):c for c in controls if scale in c["job"]["applies_to_scales"]}
    groups=[(spec[family],a["arm"],a["imports"]) for family in ("rfm","legendre") for a in design["galerkin_arms_each_family"]]
    groups += [(FD_NAMES[name],"fd","minimal") for name in spec["fd_controls"]]
    rng=random.Random(spec["order_seed"])
    jobs,slots,pairs=[],[],[]
    for block,rfm_seed in enumerate(design["rfm_seeds"],1):
        order=groups.copy()
        rng.shuffle(order)
        for name,arm,imports in order:
            seed=rfm_seed if name==spec["rfm"] else design["deterministic_seed"]
            control=indexed[name,seed,arm]
            args=control["solve_args"]
            require(args["horizon"]==72 and args["snapshots"]==design["physical_times"] and
                    args["probe_spacing"]==design["probe_spacing"] and args["graph_steps"]==design["graph_steps"] and
                    (args["steps"]==72*spec["rates"][0] if arm=="uniform" else
                     [args["startup_rate"],args["late_rate"]]==spec["rates"] if arm=="combined" else
                     args["steps"]==72*128 and args["cuda_linear_solve"]==("dense" if name=="fd176_k80" else "band")),
                    "A timed control differs from the original schedule or FD solve")
            jobs.append(dict(block=block,name=name,arm=arm,imports=imports,seed=seed,family=control["family"],
                             config=control["config"],solve_args=control["solve_args"]))
            slots.append(control["index"])
    require(len(jobs)==spec["rows"] and len(indexed)==12+len(spec["fd_controls"]),"Formal job or control count differs")

    def add(kind,baseline,candidate):
        require(baseline in groups and candidate in groups,"Pair uses an unregistered group")
        pairs.append(dict(kind=kind,baseline=list(baseline),candidate=list(candidate)))

    for family in ("rfm","legendre"):
        name=spec[family]
        for arm in ("uniform","combined"):
            add("import_only",(name,arm,"selected"),(name,arm,"minimal"))
            for fd in spec["fd_controls"]:
                add("fd_vs_galerkin",(FD_NAMES[fd],"fd","minimal"),(name,arm,"minimal"))
        add("schedule_and_preparation",(name,"uniform","minimal"),(name,"combined","minimal"))
    for arm in ("uniform","combined"):
        add("rfm_legendre_descriptive",(spec["legendre"],arm,"minimal"),(spec["rfm"],arm,"minimal"))
    if len(spec["fd_controls"])==2:
        add("fd_scale_descriptive",(FD_NAMES["fd184_band"],"fd","minimal"),(FD_NAMES["fd176_dense"],"fd","minimal"))
    return jobs,slots,[list(group) for group in groups],pairs


def build(scale):
    design=read(DESIGN)
    validate_design(design)
    controls,sources,qualification=load_controls(design)
    closures=discover_imports()
    names=SOURCES|{name for files in closures.values() for name in files}
    merge(sources,{name:transport.digest(WORK/name) for name in names})
    m=design["monitor"]
    jobs,slots,groups,pairs=register_jobs(design,scale,controls)
    frozen=dict(schema=transport.SCHEMA,status="frozen",mode="formal",scale=scale,design=design,
        coordinator_cpu=m["coordinator_cpu"],monitor_cpu=m["monitor_cpu"],poll_interval_ns=int(m["poll_interval_seconds"]*transport.SECOND),
        max_poll_gap_ns=int(m["max_poll_gap_seconds"]*transport.SECOND),budget_interval_ns=transport.BUDGET_INTERVAL_NS,
        max_budget_lateness_ns=transport.MAX_ANCHOR_LATENESS_NS,maximum_active_ns=m["maximum_active_seconds"]*transport.SECOND,
        known_background=m["known_background"],ready_timeout_seconds=60,return_timeout_seconds=120,exit_timeout_seconds=60,
        source_hashes=sources,environment=environment_identity(),observed_worker_imports=closures,qualification=qualification,
        controls=controls,slot_controls=slots,groups=groups,paired=pairs)
    token=canonical_sha(dict(frozen,jobs=jobs))
    frozen["run_token"]=token
    frozen["requests"]=[dict(job,index=i,schema="rfm-v8-worker-1",run_token=token,affinity_mask=1<<frozen["coordinator_cpu"],
                             trajectory=f"trajectory_{i:03d}.npz") for i,job in enumerate(jobs)]
    return frozen


def verify_frozen(plan,*,live_environment=False):
    design=read(DESIGN)
    validate_design(design)
    require(plan["design"]==design,"Cohort differs from its exact design")
    jobs,slots,groups,pairs=register_jobs(design,plan["scale"],plan["controls"])
    require((slots,groups,pairs)==(plan["slot_controls"],plan["groups"],plan["paired"]) and len(plan["controls"])==26 and
            [{key:request[key] for key in job} for request,job in zip(plan["requests"],jobs,strict=True)]==jobs,
            "Frozen order, paired contrasts or controls differ")
    token_base={key:value for key,value in plan.items() if key not in ("requests","run_token")}
    require(plan["run_token"]==canonical_sha(dict(token_base,jobs=jobs)),"Registered cohort token differs")
    if live_environment:
        require(environment_identity()==plan["environment"],"Registered hardware or runtime versions/bytes changed")


def execute(path,output):
    plan,raw=transport.protocol(path)
    verify_frozen(plan,live_environment=True)
    before=time.perf_counter_ns()
    summary=transport.run(path,output)
    after=time.perf_counter_ns()
    guard=dict(status="failed",protocol_sha256=sha256(raw).hexdigest(),opening_environment=plan["environment"],
               opening_guard_qpc_ns=before,closing_guard_qpc_ns=after,coordinator_summary_sha256=transport.digest(output/"summary.json"))
    try:
        transport.check_sources(plan["source_hashes"])
        verify_frozen(plan,live_environment=True)
        require(path.read_bytes()==raw==(output/"protocol.json").read_bytes(),"Frozen protocol bytes changed")
        guard.update(status="passed",closing_environment=plan["environment"])
    except Exception as error:
        guard["failure_type"]=type(error).__name__
    transport.save(output/"execution_environment_guard.json",guard)
    return dict(status=summary["status"],environment_guard=guard["status"])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--preview",choices=("mid","high"))
    action.add_argument("--freeze",choices=("mid","high"))
    action.add_argument("--execute",type=Path)
    cli.add_argument("--output",type=Path)
    args=cli.parse_args()
    if args.execute is not None:
        if args.output is None:
            cli.error("Execution needs a new cohort output directory")
        print(json.dumps(execute(args.execute,args.output)))
    else:
        if args.freeze and (args.output is None or args.output.exists()):
            cli.error("Freeze needs a new protocol JSON path")
        plan=build(args.freeze or args.preview)
        verify_frozen(plan)
        if args.freeze:
            transport.save(args.output,plan)
        print(json.dumps(dict(scale=plan["scale"],slots=len(plan["requests"]),paired=len(plan["paired"]),controls=len(plan["controls"]),
                              source_files=len(plan["source_hashes"]),written=bool(args.freeze))))
