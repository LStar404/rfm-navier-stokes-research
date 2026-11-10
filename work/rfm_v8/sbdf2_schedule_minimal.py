"""Numerical SBDF2 grid and bridge weights without CPU solver imports."""

from dataclasses import dataclass
from pathlib import Path
import sys

V6 = Path(__file__).resolve().parents[1] / "rfm_v6"
if str(V6) not in sys.path:
    sys.path.insert(0, str(V6))

import full_case as case

np = case.np


EARLY = (.5, 1., 2., 4., 6., 8.)


BRIDGE = (8.+1/112, 8.+2/112, 8.125, 8.5, 9., 10., 12., 16.)


LATE = case.SNAPSHOTS


SNAPSHOTS = EARLY+BRIDGE+LATE


@dataclass(frozen=True)
class StepSchedule:
    times: np.ndarray
    switch_step: int
    snapshot_times: np.ndarray
    snapshot_steps: np.ndarray
    probe_times: np.ndarray
    probe_steps: np.ndarray


def step_schedule(*, startup_rate=128, late_rate=112, switch_time=8., horizon=72.,
                  snapshots=SNAPSHOTS, probe_spacing=.125):
    """Integer phase indices preserve absolute observation times at the switch."""
    if min(startup_rate, late_rate, horizon, switch_time, probe_spacing) <= 0:
        raise ValueError("Require positive rates, horizon, switch time and probe spacing")
    h0, h1 = 1/startup_rate, 1/late_rate
    split = min(switch_time, horizon)
    n0 = int(case.step_indices([split], h0)[0])
    n1 = int(case.step_indices([horizon-split], h1)[0])
    times = h0*np.arange(n0+1)
    if n1:
        times = (h0*np.arange(n0+n1+1) if startup_rate == late_rate else
                 np.concatenate((times, split+h1*np.arange(1, n1+1))))
    snap = np.asarray(snapshots, dtype=float)
    if (snap.ndim != 1 or not len(snap) or not np.isfinite(snap).all() or snap[0] <= 0 or
            snap[-1] > horizon or np.any(np.diff(snap) <= 0)):
        raise ValueError("Require positive ordered snapshots within the horizon")
    probes = np.arange(0., horizon+.01*probe_spacing, probe_spacing)

    def locate(values):
        early = values <= split
        indices = np.empty(len(values), dtype=int)
        indices[early] = case.step_indices(values[early], h0)
        indices[~early] = n0+case.step_indices(values[~early]-split, h1)
        if not np.allclose(times[indices], values, atol=1e-10, rtol=0):
            raise ValueError("Observation is not on the mixed time grid")
        return indices

    return StepSchedule(times, n0, snap, locate(snap), probes, locate(probes))


def bridge_weights(ratio):
    a0 = (1+2*ratio)/(1+ratio)
    return a0, 1+ratio, ratio**2/(1+ratio), 1+ratio, ratio
