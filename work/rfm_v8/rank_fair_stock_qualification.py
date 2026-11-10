"""Qualify every rank-fair stock/warp runtime identity before timed cohorts."""

import argparse
import json
import os
from pathlib import Path
import sys

import fused_march_qualification as shared
import rank_fair_coordinator as timing
from rank_fair_owned_transport import key, read, require


HERE,WORK = timing.HERE,timing.WORK
PARENT_DESIGN = HERE/"protocol_rank_fair_final_design.json"
DESIGN = HERE/"protocol_rank_fair_bounded_draft.json"
THIN_DESIGN = HERE/"protocol_thin_formal_design.json"
EXPECTED_RANKS = {"rfm37_k72":39,"leg35_k72_q66":39,"rfm41_k80":40,"leg36_k80_q76":40,"leg35_k80_q76":39,
                  "fd176_k80":176,"fd184_k80":184}
RUNTIME_FILES = (
    "fused_marches.py","make_fused_marches.py","fused_marches_generation.json","fused_march_runtime.py",
    "fused_nonlinear_cuda_runtime.py","fused_nonlinear_fft.py","fused_nonlinear_fft.cu","warp_observe_cuda.py",
    "gpu_owned_watchdog.py","galerkin_cuda_prepared_minimal.py","rank_fair_solver_worker.py",
)


def merge(target, inventory):
    for name,digest in inventory.items():
        require(name not in target or target[name] == digest, f"Conflicting frozen digest for {name}")
        target[name] = digest


def bind(target,*records):
    for record in records:
        path = shared.bound(record)
        require(path.resolve().is_relative_to(WORK) and path.relative_to(WORK).as_posix() == record["path"],"Invalid evidence path")
        merge(target,{record["path"]:record["sha256"]})


def order(job):
    return job["name"],job["arm"],job["seed"]


def build(*,toolkit_hashes=None):
    parent,settings,base = map(read,(PARENT_DESIGN,DESIGN,THIN_DESIGN))
    ref = read(shared.REFERENCE)
    physical = ref["physical_times"]
    require(physical == base["physical_times"] and len(physical) == 19 and
            parent["rfm_seeds"] == base["rfm_seeds"] == list(shared.RFM_SEEDS) and
            parent["physical_gate"] == base["physical_gate"] == .005 and parent["implementation_gate"] == base["implementation_gate"] == 1e-5,
            "Registered physical comparison differs")
    thin,thin_plan = shared.thin_jobs()
    strong,strong_plan = shared.strong_jobs(physical)
    originals = {**thin,**{order(job):job for job in strong}}
    sources = {}
    for inventory in (thin_plan["source_hashes"],strong_plan["source_hashes"],ref["source_hashes"],
                      *(ctx["old_source_hashes"] for ctx in thin_plan["archive_contexts"].values())):
        merge(sources,inventory)
    files = [Path(__file__),Path(shared.__file__),PARENT_DESIGN,DESIGN,THIN_DESIGN,shared.REFERENCE,
             *(HERE/name for name in RUNTIME_FILES),*(shared.THIN/name for name in ("protocol.json","runs.json","qualification.json")),
             *(shared.LEG/name for name in ("protocol.json","summary.json"))]
    bind(sources,*(shared.record(path) for path in files))
    thin_raw = read(shared.THIN/"runs.json")
    old = {}
    for item,row in zip(thin_plan["jobs"],thin_raw["rows"],strict=True):
        req = item["request"]
        require(row["index"] == req["index"] and row["status"] == "complete","The serial observer control is not complete")
        old[order(item)] = shared.record(shared.THIN/req["trajectory"],row["trajectory_sha256"])
    for row in read(shared.LEG/"summary.json")["rows"]:
        job = row["job"]
        if job["observer"] == "serial":
            require(row["qualified"],"The strong Legendre serial control did not qualify")
            old[job["name"],job["schedule"],job["seed"]] = shared.record(shared.LEG/row["artifact"],row["artifact_sha256"])
    targets = {}
    for scale in parent["scales"]:
        count = 0
        for name,arm in scale["groups"]:
            seeds = parent["rfm_seeds"] if name.startswith("rfm") else [parent["deterministic_seed"]]
            for seed in seeds:
                label = name,"fd" if arm in ("dense","band") else arm,seed
                count += 1
                if label in targets:
                    targets[label]["applies_to_scales"].append(scale["name"])
                    continue
                original = originals[label]
                args = dict(original["args"])
                if arm == "combined":
                    require([args["startup_rate"],args["late_rate"]] == scale["rates"] and args["switch_time"] == 8,
                            "Inherited combined schedule differs")
                    args["snapshots"] = sorted(physical+[8+i/args["late_rate"] for i in (1,2)])
                else:
                    require(args["steps"] == 72*(scale["rates"][0] if arm == "uniform" else 128),"Inherited uniform schedule differs")
                    args["snapshots"] = physical
                if arm in ("dense","band"):
                    require(args["cuda_linear_solve"] == arm,"Inherited FD update differs")
                require(args["horizon"] == 72 and args["graph_steps"] == 24 and args["probe_spacing"] == .125 and
                        (arm in ("dense","band") or original["config"]["seed"] == seed),"Inherited common controls differ")
                bind(sources,*original["control"].values(),*original["anchor"].values(),old[label])
                targets[label] = dict(original,args=args,engine=settings["engine"],serial_control=old[label],
                                      retained_rank=EXPECTED_RANKS[name],applies_to_scales=[scale["name"]])
        require(count == settings["scales"][scale["name"]]["unique_controls"],"Qualification scale identity count differs")
    slots = list(targets.values())
    require(len(slots) == 28 and settings["engine"] == "warp_cufft","Expected 28 unique all-warp stock trajectories")
    names = ("nvrtc64_112_0.dll","nvrtc-builtins64_118.dll","cufft64_10.dll")
    if toolkit_hashes is None:
        toolkit_hashes = {name:shared.sha(shared.TOOLKIT/"bin"/name) for name in names}
    require(set(toolkit_hashes) == set(names) and all(len(digest) == 64 for digest in toolkit_hashes.values()),"Toolkit evidence is incomplete")
    return dict(status="frozen",stage="rank_fair_stock_t72",horizon=72,slots=slots,physical_times=physical,sources_sha256=sources,
                toolkit_sha256=toolkit_hashes,hard_deadline_seconds=30,
                implementation_gate=1e-5,physical_gate=.005,serial_probe_relative_and_scaled_absolute_gate=1e-12,
                scope="Accuracy qualification only. No isolated timing claims.")


def strict_observer(path,job,plan):
    import numpy as np

    with np.load(path,allow_pickle=False) as a, np.load(shared.bound(job["serial_control"]),allow_pickle=False) as b:
        expected = (len(job["args"]["snapshots"]),job["retained_rank"],job["config"]["k"]+1)
        if job["arm"] == "fd":
            require(a["states"].shape == b["states"].shape,"FD state shape differs from its exact control")
        else:
            require(a["states"].shape == expected,"Retained Galerkin rank differs")
        lookup = [int(np.flatnonzero(a["times"] == t).item()) for t in b["times"]]
        require(set(plan["physical_times"]) <= set(b["times"]) and np.array_equal(a["probe_times"],b["probe_times"]) and
                a["states"][lookup].tobytes() == b["states"].tobytes(),"Warp observer changed matched stored states or times")
        norms = {}
        for name in shared.HISTORY:
            left,right = a[name],b[name]
            scale = np.linalg.norm(right)
            require(left.shape == right.shape == (577,3) and scale > 1e-12,"Incomplete serial probe control")
            delta = left-right
            norms[name] = dict(relative=float(np.linalg.norm(delta)/scale),
                               scaled_absolute=float(np.max(np.abs(delta)/np.maximum(1.,np.abs(right)))))
        require(all(0 <= value <= plan["serial_probe_relative_and_scaled_absolute_gate"] for norm in norms.values() for value in norm.values()),
                "Warp probe change exceeded the strict serial comparison")
        return dict(saved_states_bitwise=True,common_times=len(lookup),probes=norms)


def execute(protocol, output):
    plan = read(protocol)
    require(plan == build(),"Frozen exact-runtime qualification differs from its predeclared design")
    shared.validate(plan)
    output.mkdir()
    raw = protocol.read_bytes()
    (output/"protocol.json").write_bytes(raw)
    digest = shared.sha(output/"protocol.json")
    shared.pin(6)
    census = shared.WindowsCensus()
    parent = census.times(census.current_process())
    for name in shared.THREADS:
        os.environ[name] = "1"
    evaluate,_ = shared.assessment(output,plan)
    rows = []
    for index,job in enumerate(plan["slots"]):
        slot = output/f"slot_{index:03d}"
        env_name = "RFM_FUSED_OWNER"
        os.environ[env_name] = f"{digest}:{parent['pid']}:{parent['birth_filetime_100ns']}:{index}"
        try:
            command = [sys.executable,"-B",str(Path(shared.__file__).resolve()),"--worker",str(index),"--output",str(output)]
            watchdog = shared.run_owned(command,slot,deadline_seconds=plan["hard_deadline_seconds"])
        finally:
            os.environ.pop(env_name,None)
        shared.save(slot/"watchdog.json",watchdog)
        row = dict(index=index,job=job,protocol_sha256=digest,watchdog=watchdog,qualified=False)
        if watchdog["status"] == "complete":
            try:
                result = read(slot/"result.json")
                require(result["index"] == index and result["protocol_sha256"] == digest and result["job"] == job and
                        key(result["identity"]) == key(watchdog["identity"]) == key(watchdog["terminal_identity"]) and
                        result["affinity_mask"] == 16,"Qualification worker ownership differs")
                row.update(evaluate(index,result))
                row.update(strict_observer=strict_observer(slot/"trajectory.npz",job,plan),
                           result=shared.record(slot/"result.json"),artifact=shared.record(slot/"trajectory.npz",result["trajectory_sha256"]))
            except Exception as error:
                row.update(qualified=False,failure_type=type(error).__name__)
        shared.save(slot/"qualification.json",row)
        rows.append(row)
        print(index,job["name"],job["arm"],job["seed"],row["qualified"],flush=True)
        if watchdog["status"] != "complete":
            break
    shared.validate(plan)
    require(protocol.read_bytes() == raw == (output/"protocol.json").read_bytes(),"Frozen qualification changed")
    passed = len(rows) == len(plan["slots"]) and all(row["qualified"] for row in rows)
    shared.save(output/"summary.json",dict(status="passed" if passed else "failed",protocol_sha256=digest,rows=rows,
        unattempted=list(range(len(rows),len(plan["slots"]))),isolated_speed_claim_supported=False))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preview",action="store_true")
    action.add_argument("--freeze",type=Path)
    action.add_argument("--execute",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.execute:
        if args.output is None:
            parser.error("Qualification requires an unused output directory")
        raise SystemExit(0 if execute(args.execute.resolve(),args.output.resolve()) else 2)
    plan = build()
    shared.validate(plan)
    if args.freeze:
        shared.save(args.freeze,plan)
    print(json.dumps(dict(status="registered" if args.freeze else "preview_only",unique_slots=len(plan["slots"]),
                          sources=len(plan["sources_sha256"]),time_solves_run=0)))
