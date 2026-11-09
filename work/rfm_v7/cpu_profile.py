"""Short, additive hot-path profile of the existing v6 CPU trajectory solver."""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
from pathlib import Path
import platform
from statistics import median
import sys
import time

HERE = Path(__file__).resolve().parent
V6 = HERE.parent / "rfm_v6"
sys.path.insert(0, str(V6))

import full_case as case
import fd_full as fd
import full_galerkin as gal
import full_galerkin_buffered as buffered
import full_nonlinear as nonlinear
import numpy as np
import scipy


class ExclusiveClocks:
    def __init__(self):
        self.active = False
        self.stack = []
        self.ns = Counter()
        self.calls = Counter()
        self.allocation_bytes = Counter()

    def wrap(self, label, func, *, allocation=False):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if not self.active:
                return func(*args, **kwargs)
            stamp = time.perf_counter_ns()
            frame = [0]
            self.stack.append(frame)
            try:
                value = func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - stamp
                self.stack.pop()
                if self.stack:
                    self.stack[-1][0] += elapsed
                self.ns[label] += elapsed - frame[0]
                self.calls[label] += 1
            if allocation and isinstance(value, np.ndarray) and (not args or value is not args[0]):
                self.allocation_bytes[label] += value.nbytes
            return value
        return wrapped


class TimedNumpy:
    """Time explicit constructors in the solver modules, leaving NumPy untouched."""
    def __init__(self, clocks):
        names = ("array", "ascontiguousarray", "concatenate", "empty", "empty_like",
                 "stack", "zeros", "zeros_like")
        self.timed = {name: clocks.wrap("explicit_numpy_buffers", getattr(np, name), allocation=True)
                      for name in names}

    def __getattr__(self, name):
        return self.timed[name] if name in self.timed else getattr(np, name)


@contextmanager
def replace(obj, name, value):
    previous = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, previous)


@contextmanager
def instrument(clocks):
    with ExitStack() as cleanup:
        proxy = TimedNumpy(clocks)
        for module in (case, fd, gal, buffered, nonlinear):
            cleanup.enter_context(replace(module, "np", proxy))
        tasks = [
            (buffered.BufferedGalerkin, "fields_on", "radial_fields"),
            (fd.FullFD, "native", "radial_fields"),
            (fd.sparse.csr_matrix, "__matmul__", "radial_sparse_apply"),
            (gal, "real_apply", "radial_dense_apply"),
            (gal.FullGalerkin, "terms", "mass_projection_and_bookkeeping"),
            (fd.FullFD, "terms", "mass_projection_and_bookkeeping"),
            (gal.FullGalerkin, "advance", "implicit_apply"),
            (fd.FullFD, "advance", "implicit_apply"),
            (nonlinear.FullNonlinear, "_values", "angular_pad_and_pack"),
            (nonlinear.FullNonlinear, "_coefficients", "angular_output_view"),
            (nonlinear.FullNonlinear, "__call__", "nonlinear_arithmetic"),
            (nonlinear.FullNonlinear, "advect", "nonlinear_arithmetic"),
            (nonlinear, "irfft", "fft_inverse"),
            (nonlinear, "rfft", "fft_forward"),
            (nonlinear, "fft", "fft_forward"),
        ]
        for obj, attr, label in tasks:
            cleanup.enter_context(replace(obj, attr, clocks.wrap(label, getattr(obj, attr))))
        yield


def selected_specs(names):
    with (V6 / "full_specs.json").open(encoding="utf-8") as handle:
        specs = {row["name"]: row for row in json.load(handle)}
    specs["fd176_k80"] = dict(specs["fd152_k80"], name="fd176_k80",
                             config=dict(specs["fd152_k80"]["config"], n=176))
    return [specs[name] for name in names]


def prepare(spec):
    kind = spec["kind"]
    config = spec["config"].copy()
    if kind == "rfm":
        config["seed"] = 7331
    model = fd.FullFD if kind == "fd" else buffered.BufferedGalerkin
    if spec["engine"] == "gal":
        model = gal.FullGalerkin
    return model, config


def solve_short(spec, horizon, clocks=None):
    model, config = prepare(spec)
    steps = round(horizon * spec["rate"])
    if steps != horizon * spec["rate"]:
        raise ValueError("The horizon must use an integral number of steps.")

    def factory(**kwargs):
        instance = model(**kwargs)
        if isinstance(instance, fd.FullFD):
            for band in (instance.first, instance.bdf):
                if band.native is not None:
                    band.native = clocks.wrap("fd_native_band_kernel", band.native)
        clocks.active = True
        return instance

    params = dict(config, steps=steps, horizon=horizon, snapshots=(horizon,), probe_spacing=.125)
    if clocks is None:
        return case.solve(model, **params)
    with instrument(clocks):
        try:
            return case.solve(factory, **params)
        finally:
            clocks.active = False


def equal_trajectory(a, b):
    return all(np.array_equal(getattr(a, name), getattr(b, name))
               for name in ("states", "transverse", "vorticity"))


def profile(spec, horizon, repeats):
    plain_seconds, plain_setup, timed_seconds, timed_setup = [], [], [], []
    accumulated, counts, bytes_seen = Counter(), Counter(), Counter()
    for i in range(repeats):
        clocks = ExclusiveClocks()
        if i % 2:
            timed = solve_short(spec, horizon, clocks)
            plain = solve_short(spec, horizon)
        else:
            plain = solve_short(spec, horizon)
            timed = solve_short(spec, horizon, clocks)
        if not equal_trajectory(timed, plain):
            raise ArithmeticError(f"Profile changed trajectory for {spec['name']}")
        plain_seconds.append(plain.row["seconds"])
        plain_setup.append(plain.row["setup_seconds"])
        timed_seconds.append(timed.row["seconds"])
        timed_setup.append(timed.row["setup_seconds"])
        accumulated.update(clocks.ns)
        counts.update(clocks.calls)
        bytes_seen.update(clocks.allocation_bytes)
    steps = round(horizon * spec["rate"])
    integration = sum(a - b for a, b in zip(timed_seconds, timed_setup))
    accumulated["driver_and_profiler_residual"] = integration * 1e9 - sum(accumulated.values())
    parts = {name: dict(seconds=value * 1e-9 / repeats,
                        microseconds_per_step=value * 1e-3 / (repeats * steps),
                        share=value * 1e-9 / integration,
                        calls=counts[name], explicit_buffer_bytes=bytes_seen[name])
             for name, value in sorted(accumulated.items(), key=lambda pair: -pair[1])}
    return dict(spec=spec, horizon=horizon, steps=steps, repeats=repeats,
                plain_cold_seconds=plain_seconds, plain_setup_seconds=plain_setup,
                instrumented_cold_seconds=timed_seconds, instrumented_setup_seconds=timed_setup,
                plain_cold_median_seconds=median(plain_seconds),
                instrumented_cold_median_seconds=median(timed_seconds),
                instrumented_exclusive_categories=parts,
                host_device_transfer_seconds=0.0, device_synchronization_seconds=0.0,
                bitwise_checks=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", default=["rfm37_k72", "leg37_k72", "fd176_k80"])
    parser.add_argument("--horizon", type=float, default=1.)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "cpu_profile_pilot.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for spec in selected_specs(args.names):
        row = profile(spec, args.horizon, args.repeats)
        rows.append(row)
        print(spec["name"], round(row["plain_cold_median_seconds"], 4),
              {name: round(part["microseconds_per_step"], 1) for name, part in
               row["instrumented_exclusive_categories"].items()}, flush=True)
    result = dict(purpose="diagnostic, not isolated; not an end-to-end benchmark",
                  platform=platform.platform(), processor=platform.processor(),
                  numpy=np.__version__, scipy=scipy.__version__, rows=rows,
                  limits=[
                      "Short rest-start runs preserve actual v6 layout and SBDF2 path.",
                      "Categories are exclusive, with instrumentation overhead in parent/residual clocks.",
                      "Explicit NumPy constructor clocks include copy or zero-fill; bytes are summed, not peak.",
                      "FFT clocks include SciPy transform output allocation; ufunc temporaries remain bundled.",
                      "FD native band kernel is separate; Galerkin implicit apply retains unavoidable operator glue.",
                  ])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
