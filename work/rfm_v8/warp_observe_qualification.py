"""Requalify the observation reduction against immutable full-trajectory controls."""

import json
from pathlib import Path

from warp_observe_micro import base, case, np, reference, require, V7
from warp_observe_full_runtime import solve_with_observation
from galerkin_cuda_prepared_observe import solve_cuda_prepared_observe
from basis_recurrence_model import native_model


HERE = Path(__file__).resolve().parent
OUT = HERE/"warp_observe_qualification_retry_output"
MICRO = HERE/"warp_observe_micro_retry_output"
PROBES = ("transverse","vorticity")


def saved_baseline(ref,times):
    path = HERE.parent/ref["path"]
    require(base.digest(path) == ref["sha256"],"Qualified baseline artifact changed")
    with np.load(path,allow_pickle=False) as archive:
        indices = [archive["times"].tolist().index(t) for t in times]
        data = {k:archive[k] for k in ("states","times","probe_times",*PROBES)}
        data["states"] = data["states"][indices]
        data["times"] = data["times"][indices]
        config = json.loads(str(archive["config"]))
        metadata = json.loads(str(archive["metadata"]))
    return data,config,metadata


def same_bytes(a,b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def run_arm(arm,ref,times,gates,protocol_sha):
    baseline,config,old = saved_baseline(ref,times)
    args = dict(config,horizon=72.,snapshots=times,probe_spacing=.125,graph_steps=24)
    if arm["family"] == "combined":
        solution = solve_cuda_prepared_observe(warp=True,startup_rate=128,late_rate=112,switch_time=8.,**args)
    elif arm["family"] == "uniform":
        with native_model():
            solution = solve_with_observation("uniform",True,steps=9216,**args)
    else:
        solution = solve_with_observation("fd",True,steps=9216,cuda_linear_solve=old["cuda_linear_solve"],**args)
    path = OUT/(arm["name"]+".npz")
    with path.open("xb") as handle:
        case.store_solution(handle,solution,config)
    exact = {k:same_bytes(getattr(solution,k),baseline[k]) for k in ("states","times","probe_times")}
    comparison = {}
    for k in PROBES:
        observed,ref_probe = getattr(solution,k),baseline[k]
        delta = observed-ref_probe
        norm = np.linalg.norm(ref_probe)
        require(norm > 1e-12 and np.isfinite(observed).all(),"Qualified baseline probe is undefined")
        comparison[k] = dict(relative_l2=float(np.linalg.norm(delta)/norm),max_absolute=float(np.max(abs(delta))),
                             scaled_absolute=float(np.max(abs(delta)/np.maximum(1.,abs(ref_probe)))))
    wake = base.wake(solution)
    passing = all(exact.values()) and all(r["relative_l2"] <= gates["original_implementation_relative_gate"] and
        r["relative_l2"] <= gates["tight_relative_l2_gate"] and r["scaled_absolute"] <= gates["tight_scaled_absolute_gate"]
        for r in comparison.values()) and wake["sustained_alternating_probe_signs"]
    require(solution.row["steps"] == old["steps"] and solution.row["graph_steps"] == old["graph_steps"],"Solver schedule differs")
    return dict(arm=arm,status="complete",qualified=bool(passing),physical_field_evidence="All archived qualified saved coefficients and times are bitwise identical",
                same_fields_and_times=exact,probe_reduction_order_changed=True,probe_differences=comparison,wake=wake,
                metadata=solution.row,artifact=path.name,artifact_sha256=base.digest(path),protocol_sha256=protocol_sha)


def main():
    require(not OUT.exists(),"Preserve the existing warp observation qualification")
    micro = base.read(MICRO/"summary.json")
    mp = base.read(MICRO/"protocol.json")
    require(micro["status"] == "passed" and micro["protocol_sha256"] == base.digest(MICRO/"protocol.json"),
            "The registered microdiagnostic must pass before T72")
    mid_path,high_path = (V7/f"prepared_formal_timing_{s}_output"/"protocol.json" for s in ("mid","high"))
    mid,high = base.read(mid_path),base.read(high_path)
    times = mid["design"]["physical_times"]
    arms,gates = (mp["template"][k] for k in ("arms","accuracy"))
    refs = {arm["name"]:reference(arm,mid,high) for arm in arms}
    require(refs == mp["references"] and len(times) == 19,"Observation baseline or common times differ")
    gen = base.read(HERE/"cuda_observe_solver_generation.json")
    require(gen["source_sha256"] == base.digest(HERE.parent/gen["source"]) and
            gen["target_sha256"] == base.digest(HERE/gen["target"]),"Generated combined observation adapter changed")
    files = [Path(__file__),MICRO/"protocol.json",MICRO/"summary.json",mid_path,high_path,HERE/"warp_observe_cuda.py",HERE/"warp_observe_full_runtime.py",
             HERE/"cuda_observe_solver_generation.json",HERE/gen["target"],*(HERE.parent/r["path"] for r in refs.values())]
    hashes = {x.relative_to(HERE.parent).as_posix():base.digest(x) for x in files}
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="frozen",arms=arms,references=refs,physical_times=times,accuracy=gates,
                source_hashes=hashes,scope="Five complete independent T72 GPU correctness-only trajectories, no speed claim"))
    sha = base.digest(OUT/"protocol.json")
    rows = []
    for arm in arms:
        try:
            row = run_arm(arm,refs[arm["name"]],times,gates,sha)
        except (ArithmeticError,RuntimeError,ValueError,TypeError,OSError) as exc:
            row = dict(arm=arm,status="failed",qualified=False,failure_type=type(exc).__name__,failure=str(exc))
        base.save(OUT/(arm["name"]+".json"),row)
        rows.append(row)
        print(arm["name"],row["status"],row["qualified"],flush=True)
    require(hashes == {x.relative_to(HERE.parent).as_posix():base.digest(x) for x in files},"Observation qualification source changed")
    count = sum(r["qualified"] for r in rows)
    base.save(OUT/"summary.json",dict(status="passed" if count == len(arms) else "failed",qualified=count,rows=rows,
                                     protocol_sha256=sha,isolated_speed_claim_supported=False))
    require(count == len(arms),"Some observation solver arms did not qualify")


if __name__ == "__main__":
    main()
