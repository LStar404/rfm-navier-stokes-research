"""Check the released v7 hardware evidence using saved fields and no time marching."""
import json
import math

from basis_recurrence_model import RecurrenceGalerkin
from joint_hardware_pow2 import (FROZEN, METRICS, OUT, PREFLIGHT, TIMING, approved_inputs,
                                 audit_inputs, base, score_rows, select)
from summarize_joint_hardware_pow2 import summarize


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def verify():
    fine, candidate_rows, audited = audit_inputs()
    frozen, measured = base.read(FROZEN), base.read(TIMING)
    preflight = base.read(PREFLIGHT)
    check(measured["frozen"]==frozen and preflight["protocol_sha256"]==base.digest(FROZEN)
          and preflight["status"]=="complete" and preflight["passes"] and len(preflight["rows"])==6,
          "Stored protocol or preflight differs")
    approved_inputs(frozen)
    choices, decisions, _ = select(frozen["protocol"],frozen["template"],candidate_rows)
    check(choices==frozen["choices"] and decisions==frozen["all_candidate_decisions"],
          "First-passing method selections differ")
    check(audited["composed_difference"]==frozen["reference"]["composed_difference"] and
          audited["composed_difference"]<=frozen["reference_limit"], "Reference screen differs")
    summary=base.read(OUT/"summary.json")
    recomputed=summarize(TIMING,summary["derived_from"]["trajectory_hashes_verified"])
    check(summary==json.loads(json.dumps(recomputed,allow_nan=False)),
          "Derived paired statistics differ")
    grids=base.grid_set()
    truth=base.case.reference_fields(fine,grids)
    times=frozen["protocol"]["checkpoints"]
    indices={(g,t) for g in grids for t in times}
    representatives=[r for r in measured["rows"] if r["block"]==1]
    check(len(representatives)==6,"Expected one first-block trajectory per method/backend")
    for r in representatives:
        path=OUT/r["artifact"]
        check(base.digest(path)==r["sha256"],f"Trajectory hash mismatch: {path.name}")
        model=base.FullFD if r["reconstruction_class"]=="FullFD" else RecurrenceGalerkin
        sol=base.case.load_solution(path,model)
        check(base.case.np.array_equal(sol.times,times),f"Physical checkpoint mismatch: {path.name}")
        scores=base.case.score(sol,truth,grids)
        worst=score_rows(scores,indices)
        check(r["worst"]==score_rows(r["scores"],indices),f"Recorded score maxima changed: {path.name}")
        expected={(row["grid"],row["time"]):row for row in r["scores"]}
        check(all(math.isclose(row[k],expected[row["grid"],row["time"]][k],rel_tol=1e-8,abs_tol=1e-12)
                  for row in scores for k in METRICS),f"Reconstructed scores differ: {path.name}")
        check(max(worst.values())<=frozen["template"]["gate"] and
              base.wake(sol)["sustained_alternating_probe_signs"],f"Acceptance differs: {path.name}")
    return dict(status="passed",reference_edges=3,rescored_candidates_audited=len(candidate_rows),
                representative_trajectories_rescored=len(representatives),
                field_errors_recomputed=len(representatives)*len(indices)*len(METRICS),
                archived_timing_rows=len(measured["rows"]),trajectory_solves_run=0)


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
