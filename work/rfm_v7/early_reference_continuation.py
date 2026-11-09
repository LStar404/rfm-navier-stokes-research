"""Continue the startup reference from immutable full-history T6 checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import early_reference_chain as early


base,case,np=early.base,early.case,early.np
HERE=early.HERE
OUT=HERE/"early_reference_continuation_output"
TEMPLATE=HERE/"protocol_early_reference_continuation.json"
FROZEN=OUT/"protocol.json"
RUNS=OUT/"runs.json"
REFERENCE=OUT/"reference_audit.json"
HISTORIES=("mass","flux","old_mass","old_flux")


def digest_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def bitwise(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and digest_array(a)==digest_array(b)


def files(path):
    return dict(source=path.relative_to(base.WORK).as_posix(),sha256=base.digest(path))


def freeze():
    p=base.read(TEMPLATE)
    old=base.read(early.FROZEN)
    run=base.read(early.RUNS)
    if (run["status"]!="complete" or run["protocol_sha256"]!=base.digest(early.FROZEN) or
            len(run["arms"])!=4 or p["horizon"]!=16 or p["checkpoint_time"]!=6 or p["cpu_workers"]!=3 or
            p["checkpoints"][:6]!=old["protocol"]["checkpoints"] or
            p["checkpoints"][6:]!=[8.125,8.5,9,10,12,16] or any(p[k]!=old["protocol"][k] for k in
                ("gate","reference_budget_fraction","near_zero_relative_floor"))):
        raise RuntimeError("T16 continuation differs from the authorized reference protocol")
    early.hardware.sha_matches(early.REFERENCE,run["reference_audit_sha256"])
    sources=old["source_hashes"] | {path.relative_to(base.WORK).as_posix():base.digest(path)
                                   for path in (TEMPLATE,Path(__file__))}
    for name,sha in sources.items():
        early.hardware.sha_matches(base.WORK/name,sha)
    specs=old["protocol"]["reference_path"]
    indexed={r["spec"]["name"]:r for r in run["arms"]}
    inputs=[]
    for s in specs:
        previous=indexed[s["name"]]
        arm=early.OUT/previous["artifact"]
        checkpoint=early.OUT/s["name"]/f"resume_fd{s['n']}_k128_rate384.npz"
        early.hardware.sha_matches(arm,previous["sha256"])
        inputs.append(dict(spec=s,old_arm=files(arm),old_history=files(checkpoint)))
    result=dict(status="frozen",protocol=dict(p,reference_path=specs),original_protocol=files(early.FROZEN),
                original_runs=files(early.RUNS),original_audit=files(early.REFERENCE),source_hashes=sources,arms=inputs)
    if FROZEN.exists():
        if base.read(FROZEN)!=result:
            raise RuntimeError("Frozen continuation inputs differ")
    else:
        base.save(FROZEN,result)
    return result


def padded(array, target_length):
    zeros=np.zeros((target_length-len(array),*array.shape[1:]),dtype=array.dtype)
    return np.concatenate((array,zeros))


def restart_input(job,p,old):
    spec=job["spec"]
    source=base.WORK/job["old_history"]["source"]
    early.hardware.sha_matches(source,job["old_history"]["sha256"])
    out=OUT/f"restart_input_{spec['name']}.npz"
    info=out.with_suffix(".json")
    if info.exists():
        evidence=base.read(info)
        if evidence["old_history"]!=job["old_history"]:
            raise RuntimeError("Derived restart input differs from its frozen history")
        early.hardware.sha_matches(out,evidence["sha256"])
        return out,evidence
    if out.exists():
        raise FileExistsError(out)
    driver=early.late.radial_driver
    expected=dict(config=dict(early.late.grid(spec),n=spec["n"]),dt=1/spec["rate"],
                  steps=8*spec["rate"],class_name=driver.CLASS,scales=old.model.scale_metadata())
    with np.load(source,allow_pickle=False) as data:
        saved={name:data[name] for name in data.files}
    source_identity=json.loads(str(saved["identity"]))
    if (source_identity!=expected or int(saved["step"])!=p["checkpoint_time"]*spec["rate"] or
            not bitwise(saved["states"],old.states[:5]) or not bitwise(saved["last_state"],old.states[4])):
        raise RuntimeError("T6 restart identity or checkpointed state differs")
    probe_count=1+int(p["checkpoint_time"]/.125)
    for k in ("transverse","vorticity"):
        if (saved[k].shape!=getattr(old,k).shape or not bitwise(saved[k][:probe_count],getattr(old,k)[:probe_count]) or
                np.any(saved[k][probe_count:])):
            raise RuntimeError("T6 restart probe prefix or unused slots differ")
    if any(saved[k].shape!=old.model.rhs_shape or not np.isfinite(saved[k]).all() for k in HISTORIES):
        raise RuntimeError("T6 SBDF2 histories missing or nonfinite")
    original_amplitudes=case.speed(old.row["dt"]*np.arange(old.row["steps"]+1))
    mass,flux=old.model.terms(saved["last_state"],original_amplitudes[int(saved["step"])])
    if not bitwise(mass,saved["mass"]) or not bitwise(flux,saved["flux"]):
        raise RuntimeError("T6 current history does not match the stored state")
    new_identity=dict(source_identity,steps=p["horizon"]*spec["rate"])
    unchanged={k:digest_array(saved[k]) for k in ("states","last_state",*HISTORIES)}
    saved["identity"]=json.dumps(new_identity,sort_keys=True)
    for k in ("transverse","vorticity"):
        saved[k]=padded(saved[k],1+int(p["horizon"]/.125))
    driver.write_npz(out,**saved)
    evidence=dict(status="derived",old_history=job["old_history"],old_identity=source_identity,new_identity=new_identity,
                  unchanged_array_sha256=unchanged,artifact=out.name,sha256=base.digest(out),
                  changes="New step total and probe capacity. Initial time, states, current and older SBDF2 histories are preserved.")
    base.save(info,evidence)
    return out,evidence


def run_arm(job,p):
    spec=job["spec"]
    target=OUT/f"{spec['name']}_n{spec['n']}_k{spec['k']}_r{spec['rate']}.npz"
    record=target.with_suffix(".json")
    if record.exists():
        old=base.read(record)
        if old["spec"]!=spec:
            raise RuntimeError("Existing continuation differs from the frozen arm")
        early.hardware.sha_matches(target,old["sha256"])
        return old
    if target.exists():
        raise FileExistsError(target)
    old_path=base.WORK/job["old_arm"]["source"]
    early.hardware.sha_matches(old_path,job["old_arm"]["sha256"])
    original=case.load_solution(old_path,early.late.MODEL)
    resume=OUT/spec["name"]
    resume.mkdir(exist_ok=True)
    seed,evidence=restart_input(job,p,original)
    checkpoint=resume/f"resume_fd{spec['n']}_k128_rate384.npz"
    if not checkpoint.exists():
        with checkpoint.open("xb") as out:
            out.write(seed.read_bytes())
    driver=early.late.radial_driver
    with patch.multiple(driver,OUT=resume,RATE=spec["rate"],HORIZON=p["horizon"],
                        STEPS=p["horizon"]*spec["rate"],GRID=early.late.grid(spec)), patch.object(case,"SNAPSHOTS",p["checkpoints"]):
        sol,cfg=driver.march(spec["n"])
    replay={k:bitwise(getattr(sol,k)[:len(getattr(original,k))],getattr(original,k))
            for k in ("states","times","probe_times","transverse","vorticity")}
    temp=target.with_suffix(".tmp.npz")
    case.store_solution(temp,sol,cfg)
    temp.replace(target)
    row=dict(status="complete",spec=spec,artifact=target.name,sha256=base.digest(target),configuration=cfg,
             metadata=sol.row,restart=evidence,replays_original_t8=replay,bitwise_t8_pass=all(replay.values()))
    base.save(record,row)
    print("COMPLETE",spec["name"],row["bitwise_t8_pass"],flush=True)
    return row


def execute(frozen):
    p=frozen["protocol"]
    result=dict(status="running",protocol_sha256=base.digest(FROZEN),arms=[])
    if RUNS.exists() and base.read(RUNS)["protocol_sha256"]!=result["protocol_sha256"]:
        raise RuntimeError("Continuation journal differs from freeze")
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    base.save(RUNS,result)
    with ProcessPoolExecutor(max_workers=p["cpu_workers"]) as pool:
        futures=[pool.submit(run_arm,job,p) for job in frozen["arms"]]
        for task in as_completed(futures):
            result["arms"].append(task.result())
            base.save(RUNS,result)
    if freeze()!=frozen:
        raise RuntimeError("Continuation inputs changed")
    passed=len(result["arms"])==4 and all(r["bitwise_t8_pass"] for r in result["arms"])
    if passed:
        with patch.multiple(early,OUT=OUT,FROZEN=FROZEN,REFERENCE=REFERENCE):
            _,_,_,budget_pass=early.reference_audit(frozen,result["arms"])
        result.update(reference_pass=budget_pass,reference_audit_sha256=base.digest(REFERENCE))
    else:
        result.update(reference_pass=False,reason="At least one continuation failed exact T8 replay")
    result.update(status="complete",bitwise_t8_pass=passed)
    base.save(RUNS,result)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-only",action="store_true")
    action.add_argument("--execute",action="store_true")
    args=cli.parse_args()
    fixed=freeze()
    print("FROZEN T16",base.digest(FROZEN),flush=True)
    if args.execute:
        execute(fixed)
