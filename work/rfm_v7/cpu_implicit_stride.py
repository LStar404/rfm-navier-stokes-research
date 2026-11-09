"""Test an exact, copy-reducing layout for the Galerkin implicit application."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

from cpu_profile import (
    HERE, V6, buffered, case, equal_trajectory, np, prepare, selected_specs,
)


class StridedGalerkin(buffered.BufferedGalerkin):
    """Keep both input and output in radial-first complex storage."""

    def advance(self, rhs, amplitude, first):
        if first or rhs.dtype != np.complex128 or not rhs.flags.c_contiguous:
            return super().advance(rhs, amplitude, first)
        rhs[:, 1] -= amplitude * self.bdf_lift
        out = np.empty_like(rhs)
        # Adjacent real and imaginary columns remain one real matrix RHS.
        shape = (self.width, self.k + 1, 2)
        values = rhs.view(float).reshape(shape)[:, 1:].transpose(1, 0, 2)
        result = out.view(float).reshape(shape)[:, 1:].transpose(1, 0, 2)
        np.matmul(self.implicit, values, out=result)
        out[:, 0] = 0.
        out[:self.mean_width, 0] = self.mean_inverse @ rhs[:self.mean_width, 0].real
        return out


def difference(a, b):
    denominator = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / max(denominator, np.finfo(float).tiny))


def fixed_profile(spec, iterations, repeats):
    base, config = prepare(spec)
    dt = 1 / spec["rate"]
    models = [cls(dt=dt, **config) for cls in (base, StridedGalerkin)]
    with np.load(V6 / f"full_state_{spec['name']}.npz", allow_pickle=False) as saved:
        states = saved["states"]
    rhs = np.ascontiguousarray(states[-1])
    amp = case.speed(72.)
    reference, candidate = [model.advance(rhs.copy(), amp, False) for model in models]
    error = difference(reference, candidate)
    if error > 1e-13:
        raise ArithmeticError(f"Changed implicit application for {spec['name']}: {error}")
    outputs = [[], []]
    order = np.random.default_rng(83771)
    for _ in range(repeats):
        for which in order.permutation(2):
            inputs = [rhs.copy() for _ in range(iterations)]
            start = time.perf_counter()
            for item in inputs:
                models[which].advance(item, amp, False)
            outputs[which].append((time.perf_counter() - start) / iterations)
    return dict(seconds_per_call=outputs, relative_difference=error,
                bitwise=np.array_equal(reference, candidate),
                mean_bitwise=np.array_equal(reference[:, 0], candidate[:, 0]),
                explicit_temporaries_removed_bytes=2 * rhs[:, 1:].nbytes)


def trajectory_profile(spec, horizon, repeats):
    base, config = prepare(spec)
    models = (base, StridedGalerkin)
    outputs = [[], []]
    comparisons = []
    for i in range(repeats):
        solutions = [None, None]
        for which in ((0, 1) if i % 2 == 0 else (1, 0)):
            sol = case.solve(models[which], horizon=horizon, steps=round(horizon*spec["rate"]),
                             snapshots=(horizon,), **config)
            solutions[which] = sol
            outputs[which].append(sol.row["seconds"] - sol.row["setup_seconds"])
        baseline, variant = solutions
        errors = {name: difference(getattr(baseline, name), getattr(variant, name))
                  for name in ("states", "transverse", "vorticity")}
        if max(errors.values()) > 1e-10:
            raise ArithmeticError(f"Changed trajectory for {spec['name']}: {errors}")
        comparisons.append(dict(relative_difference=errors, bitwise=equal_trajectory(baseline, variant)))
    return dict(integration_seconds=outputs, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "cpu_implicit_stride_pilot.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for spec in selected_specs(("rfm37_k72", "leg37_k72", "rfm41_k80")):
        fixed = fixed_profile(spec, iterations=80, repeats=9)
        trajectory = trajectory_profile(spec, horizon=1, repeats=3)
        name = spec["name"]
        print(name, "implicit us", [1e6 * median(v) for v in fixed["seconds_per_call"]],
              "integration seconds", [median(v) for v in trajectory["integration_seconds"]],
              "bitwise", fixed["bitwise"], all(v["bitwise"] for v in trajectory["comparisons"]), flush=True)
        rows.append(dict(name=name, fixed=fixed, trajectory=trajectory))
    result = dict(purpose="non-isolated short pilot; timings are diagnostic",
                  columns=["existing", "strided_out"], rows=rows)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
