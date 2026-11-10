"""Bind 160 prospective slots to three stock PTX artifacts and 39 CPU controls."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import rank_fair_extended_inventory as base
from rank_fair_owned_transport import read, require


HERE,WORK = base.HERE,base.WORK
SOURCE = Path(__file__).resolve()
DESIGN = HERE/"protocol_rank_fair_cache_design.json"
ARTIFACTS = HERE/"cuda_ptx_artifact_output"
POLICIES = ("empty_first","strict_hit")
STOCK = "warp_cufft"
RUNTIME = "private_cuCtxCreate_v2_scoped_cufft"


def draft():
    from build_cuda_ptx_artifacts import recipe
    from cuda_ptx_cache import key_for,unpack

    old = base.draft_inventory()
    design = read(DESIGN)
    require(design["status"] == "fixed_design_only_no_execution_freeze" and design["gpu_engine"] == STOCK and
            design["gpu_context"] == RUNTIME and design["policies"] == list(POLICIES),"Fixed cache design changed")
    manifest_path = ARTIFACTS/"manifest.json"
    manifest = read(manifest_path)
    require(manifest["status"] == "passed" and manifest["compiler"]["api_version"] == [11,8] and all(
        old["toolkit_sha256"][name] == digest for name,digest in manifest["compiler"]["library_sha256"].items()),
        "Compiler identity differs from the existing original 11.8 cohort")
    modules = {x["family"]:x for x in manifest["modules"] if x["engine"] == STOCK}
    require(set(modules) == {"uniform","combined","fd"},"Expected exactly three stock PTX artifacts")
    inputs = dict(old["sources_sha256"])
    for path in (SOURCE,DESIGN,manifest_path,HERE/"build_cuda_ptx_artifacts.py",HERE/"cuda_ptx_cache.py"):
        base.original.bind(inputs,base.original.shared.record(path))
    slots,index = [],{}
    for i,control in enumerate(old["slots"]):
        arm = control["arm"]
        n = 225 if control["config"]["k"] == 72 else 243
        _,spec = recipe(arm,STOCK,n,manifest["compiler"])
        key = key_for(spec)
        m = modules[arm]
        require(spec == m["manifest"]["contract"] and key == m["manifest"]["key"],"Matched warp stock recipe has a different PTX key")
        file = ARTIFACTS/"artifacts"/(key+".rfmptx")
        art = base.original.shared.record(file,m["cpu_cold_build"]["artifact_sha256"])
        ptx,decoded = unpack(file.read_bytes(),spec)
        require(decoded == m["manifest"] and sha256(ptx).hexdigest() == decoded["ptx_sha256"],"Archived PTX payload differs")
        base.original.bind(inputs,art)
        pair = POLICIES if i%2 == 0 else POLICIES[::-1]
        for policy in pair:
            identity = (*base.original.order(control),policy)
            index[identity] = len(slots)
            slots.append(dict(control,policy=policy,ptx_cache_key=key,ptx_artifact=art,ptx_sha256=decoded["ptx_sha256"],
                              angular_fft_length=n,identity_index=i,runtime_id=RUNTIME))
    require(len(slots) == len(index) == 78 and len({s["ptx_cache_key"] for s in slots}) == 3 and
            sum(s["serial_control"] is None for s in slots) == 22,"Policy-expanded exact qualification differs")
    windows = []
    block_sets = {"mid":[],"high":[]}
    for window in design["formal_windows"]:
        scale = window["scale"]
        definitions = old["schedules"][scale]
        block_sets[scale].extend(window["blocks"])
        rng = random.Random(window["order_seed"])
        schedule = []
        for block in window["blocks"]:
            group = [dict(j,policy=policy,qualification_index=index[j["name"],j["arm"],j["seed"],policy])
                     for j in definitions["slots"] if j["block"] == block for policy in POLICIES]
            rng.shuffle(group)
            require(len(group) == (10 if scale == "mid" else 22) and len({(x["name"],x["arm"],x["policy"]) for x in group}) == len(group),
                    "A policy or deterministic comparator was dropped from a paired block")
            schedule.extend(group)
        require(len(schedule) == window["slots"],"Formal window size differs")
        windows.append(dict(window,schedule=schedule,method_contrasts=definitions["paired"],
                            cache_contrasts=[dict(name=j[0],arm=j[1],baseline="empty_first",candidate="strict_hit") for j in
                                next(s["groups"] for s in old["design"]["scales"] if s["name"] == scale)]))
    require([w["slots"] for w in windows] == [50,66,44] and all(b == list(range(1,6)) for b in block_sets.values()),"Full five-block window assignment differs")
    return dict(status="draft_no_authorization_no_gpu_qualification",design=design,base_design=old["design"],
        sources_sha256=inputs,slots=slots,windows=windows,compiler_identity=manifest["compiler"],
        ptx_manifest=base.original.shared.record(manifest_path),physical_times=old["physical_times"],horizon=72,
        cpu_auditor_affinities=[10,12,14,16],hard_deadline_seconds=30,toolkit_sha256=old["toolkit_sha256"],
        implementation_gate=old["implementation_gate"],physical_gate=old["physical_gate"],
        serial_probe_relative_and_scaled_absolute_gate=old["serial_probe_relative_and_scaled_absolute_gate"],gpu_trajectories_run=0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    plan = draft()
    if args.output:
        base.original.shared.save(args.output,plan)
    print(json.dumps(dict(status=plan["status"],windows=[w["slots"] for w in plan["windows"]],unique_qualification=len(plan["slots"]),
                          stock_ptx_modules=len({s["ptx_cache_key"] for s in plan["slots"]}),sources=len(plan["sources_sha256"]),
                          gpu_trajectories_run=0)))
