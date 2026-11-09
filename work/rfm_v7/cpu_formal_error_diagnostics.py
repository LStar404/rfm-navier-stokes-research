"""Reference Fourier truncation floors and absolute probe crossing drift."""
import argparse
from pathlib import Path

from cpu_formal import OUT, PROTOCOL, case, grid_set, open_reference, read, save, wake


def floors(modals, weights, cutoff):
    # Positive and negative coefficients are conjugate pairs. Modal amplitudes
    # follow the frozen real-FFT convention, so positive mode power is doubled.
    ur, ut, omega = modals
    velocity = abs(ur)**2 + abs(ut)**2
    vorticity = abs(omega)**2
    for power in (velocity, vorticity):
        power[:,1:] *= 2
    return [float(case.np.sqrt((weights @ power[:,cutoff+1:].sum(axis=1))/
                              (weights @ power.sum(axis=1))))
            for power in (velocity, vorticity)]


def crossing_drift(target, reference):
    values = []
    for variant, base in zip(target["velocity"], reference["velocity"]):
        candidates = case.np.asarray(variant["positive_crossing_times"])
        observations = case.np.asarray(base["positive_crossing_times"])
        if candidates.size and observations.size:
            nearest = case.np.abs(candidates[:,None] - observations).argmin(axis=1)
            lag = candidates - observations[nearest]
            values.append(dict(first_lag=float(lag[0]),last_lag=float(lag[-1]),
                               max_abs_lag=float(abs(lag).max()),pairs=int(len(lag))))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT/"error_diagnostics.json")
    parser.add_argument("--cutoffs",type=int,nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = read(PROTOCOL)
    finest = open_reference(protocol["reference_path"][-1],protocol)
    cutoffs = sorted(set(args.cutoffs or (s["k"] for s in protocol["variants"])))
    if min(cutoffs)<0 or max(cutoffs)>protocol["reference_path"][-1]["k"]:
        parser.error("Cutoffs must lie within the final reference Fourier resolution")
    labels = ("velocity","vorticity","near_velocity","near_vorticity")
    diagnostics = []
    for grid,(s,theta,weight) in grid_set().items():
        if len(theta)<=2*protocol["reference_path"][-1]["k"]:
            raise RuntimeError("Angular grid is too small for the discrete Parseval bound")
        for i,t in enumerate(finest.times):
            fields=finest.modal(i,s)
            near=weight*(case.radius(s)<=2.5)
            for cutoff in cutoffs:
                power=floors(fields,weight,cutoff)+floors(fields,near,cutoff)
                diagnostics.append(dict(grid=grid,time=float(t),cutoff=cutoff,**dict(zip(labels,power))))
    worst = {str(k):{label:max(r[label] for r in diagnostics if r["cutoff"]==k)
                     for label in labels} for k in cutoffs}
    qualifying = read(OUT/"qualification.json")
    baseline = wake(finest)
    probe_data = [dict(name=r["name"],seed=r["seed"],crossing_lag=crossing_drift(r["wake"],baseline))
                  for r in qualifying["rows"] if "wake" in r]
    save(args.output,dict(worst_tail_floor=worst,tail_floor_rows=diagnostics,probe_data=probe_data,
         note="Fourier tail is a lower bound for fields represented with the given K on these discrete score grids. "
              "Crossing lags use the original times and have no effect on qualification."))
    print(worst,flush=True)


if __name__=="__main__":
    main()
