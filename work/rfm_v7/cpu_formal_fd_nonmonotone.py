"""Preserve high-resolution FD comparisons and unshifted wake timing diagnostics."""
from itertools import combinations

import cpu_formal as base
from cpu_formal_error_diagnostics import crossing_drift
from cpu_formal_reference_point1 import ROOT, PROTOCOL


def main():
    target=ROOT/"fd_resolution_nonmonotone.json"
    if target.exists():
        raise FileExistsError(target)
    protocol=base.read(PROTOCOL)
    old=dict(name="prior_fd320",n=320,k=112,rate=320,source=protocol["legacy_reference"])
    specs=[old,*protocol["reference_path"]]
    solutions=[base.open_reference(s,protocol) for s in specs]
    grids=base.grid_set()
    fields=[base.case.reference_fields(s,grids) for s in solutions]
    wakes=[base.wake(s) for s in solutions]
    rows=[]
    for i,j in combinations(range(len(specs)),2):
        scores=base.case.score(solutions[i],fields[j],grids)
        by_time=[dict(time=float(t),maxima=base.case.maxima([r for r in scores if r["time"]==t]))
                 for t in solutions[j].times]
        changed=[key for key in ("n","k","rate") if specs[i][key]!=specs[j][key]]
        rows.append(dict(candidate=specs[i]["name"],reference=specs[j]["name"],changed=changed,
            worst=base.case.maxima(scores),by_time=by_time,scores=scores,
            absolute_velocity_probe_crossing_lags=crossing_drift(wakes[i],wakes[j])))
    state=[dict(spec=spec,sha256=base.digest(base.WORK/spec["source"]),
                min_log_spacing=float(base.case.np.diff(s.model.s).min()),wake=w)
           for spec,s,w in zip(specs,solutions,wakes)]
    base.save(target,dict(states=state,comparisons=rows,
              note="Scores remain at frozen physical times. Pairs changing multiple axes "
                   "give descriptive nonmonotonicity, not independent refinement increments."))
    for row in rows:
        print(row["candidate"],row["reference"],row["changed"],
              max(row["worst"].values()),flush=True)


if __name__=="__main__":
    main()
