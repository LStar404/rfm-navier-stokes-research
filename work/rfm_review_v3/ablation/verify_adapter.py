"""Verify the ablation controls without modifying any solver module."""
import hashlib
import json
from pathlib import Path
from ablation_model import AblationGalerkin, np
from basis_recurrence_model import RecurrenceGalerkin

HERE = Path(__file__).resolve().parent
protocol = json.loads((HERE / "protocol.json").read_text())
rows = []
for resolution in ("mid", "high"):
    spec = next(j for j in protocol["jobs"] if j["resolution"]==resolution and j["role"]=="cpu-control")
    config = {k:v for k,v in spec["config"].items() if k not in ("midpoint", "nonzero_layers")}
    dt = 72 / spec["steps"]
    original = RecurrenceGalerkin(dt=dt, **config)
    controls = {(mid, layers):AblationGalerkin(dt=dt, midpoint=mid, nonzero_layers=layers, **config)
                for mid in (False, True) for layers in (False, True)}
    control = controls[False, True]
    arrays = ("samples", "transform", "implicit", "start", "mean_samples", "mean_inverse", "time_jets")
    assert all(np.array_equal(getattr(control, k), getattr(original, k)) for k in arrays)
    assert all(np.array_equal(m.mean_samples, control.mean_samples) and
               np.array_equal(m.mean_inverse, control.mean_inverse) for m in controls.values())
    rng = np.random.default_rng(314159)
    state = np.ascontiguousarray((rng.normal(size=control.rhs_shape) + 1j*rng.normal(size=control.rhs_shape))*.001)
    state[:, 0] = state[:, 0].real
    actual, expected = control.terms(state, -.3j), original.terms(state, -.3j)
    assert all(np.array_equal(a, b) for a,b in zip(actual, expected))
    rhs = np.ascontiguousarray(actual[0] + dt*actual[1])
    assert np.array_equal(control.advance(rhs.copy(), -.3j, False), original.advance(rhs.copy(), -.3j, False))
    alternate = dict(config, seed=9041)
    for layers in (True, False):
        replica = AblationGalerkin(dt=dt, midpoint=True, nonzero_layers=layers, **alternate)
        assert all(np.array_equal(getattr(replica, k), getattr(controls[True, layers], k)) for k in arrays)
    rows.append(dict(resolution=resolution, baseline_arrays_bitwise_equal=True, baseline_terms_bitwise_equal=True,
        baseline_advance_bitwise_equal=True, all_mean_blocks_bitwise_equal=True,
        midpoint_independent_of_unused_seed=True, ranks={f"midpoint={m},layers={l}":v.width for (m,l),v in controls.items()},
        mean_rank=control.mean_width))
result = dict(status="pass", protocol_sha256=hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(), checks=rows)
(HERE / "adapter_verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
