"""Time stage boundaries inside one captured CUDA graph tail."""

import argparse
from contextlib import ExitStack
import ctypes as C
import json
from pathlib import Path
import time

from cuda_environment_native import I, P, S, U, bind, call
from fd_cuda_native import FDRuntime, solve_fd_native
from galerkin_cuda_native import Runtime, np, solve_native


EVENT_EXTERNAL = 1
EVENT_RECORD_NODE = 7


class GraphStageMixin:
    stage_labels = ()
    name_map = {}

    def __init__(self, steps, samples=8, toolkit=None, scheme="sparse"):
        if steps < 2 or samples < 1 or scheme not in ("sparse", "dense"):
            raise ValueError("Require positive stage samples, a timing scheme and at least two steps")
        self.steps = steps
        self.samples = samples
        self.scheme = scheme
        self.active = None
        self.marker = 0
        self.profiled = False
        self.remaining = None
        self.events = ExitStack()
        super().__init__(toolkit)
        dll = self.driver_lib
        self.create_event = bind(dll, "cuEventCreate", [C.POINTER(P), U])
        self.destroy_event = bind(dll, "cuEventDestroy_v2", [P])
        self.event_record = bind(dll, "cuEventRecordWithFlags", [P, P, U])
        self.event_elapsed = bind(dll, "cuEventElapsedTime", [C.POINTER(C.c_float), P, P])
        self.get_nodes = bind(dll, "cuGraphGetNodes", [P, C.POINTER(P), C.POINTER(S)])
        self.get_node_type = bind(dll, "cuGraphNodeGetType", [P, C.POINTER(I)])
        self.graph_launch = self.driver.cuGraphLaunch
        self.d2z = self.fft_marker(self.d2z, "forward_fft")
        self.z2d = self.fft_marker(self.z2d, "inverse_fft")

    def __exit__(self, *exc):
        try:
            self.sync()
            self.events.close()
        finally:
            super().__exit__(*exc)

    def fft_marker(self, operation, stage):
        def wrapped(*args):
            status = operation(*args)
            if not status:
                self.after(stage)
            return status

        wrapped.__name__ = operation.__name__
        return wrapped

    def launch(self, name, count, *args):
        operation = super().launch(name, count, *args)
        stage = self.name_map[name]

        def wrapped():
            operation()
            self.after(stage)

        return wrapped

    def after(self, stage):
        if self.active is None:
            return
        if self.marker >= len(self.stage_labels) or self.stage_labels[self.marker] != stage:
            raise RuntimeError(f"Unexpected stage in graph capture: {stage}")
        chosen, markers = self.active
        old = self.marker
        self.marker += 1
        if chosen is None:
            self.record(markers[self.marker])
        elif self.marker == chosen:
            self.record(markers[0])
        elif old == chosen:
            self.record(markers[1])

    def record(self, event):
        call(self.event_record, event, self.driver.stream, EVENT_EXTERNAL)

    def new_markers(self, count):
        markers = [P() for _ in range(count)]
        for event in markers:
            call(self.create_event, C.byref(event), 0)
            self.events.callback(call, self.destroy_event, event)
        return markers

    def with_markers(self, operation, chosen, markers):
        self.active = chosen, markers
        self.marker = 0
        try:
            if chosen is None or chosen == 0:
                self.record(markers[0])
            operation()
            if self.marker != len(self.stage_labels):
                raise RuntimeError("Incomplete device stage trace")
        finally:
            self.active = None

    def captured_with_event_count(self, resources, operation, expected_events):
        driver = self.driver
        graph, executable = P(), P()
        call(driver.cuStreamBeginCapture, driver.stream, 0)
        try:
            operation()
        except BaseException:
            driver.cuStreamEndCapture(driver.stream, C.byref(graph))
            if graph:
                driver.cuGraphDestroy(graph)
            raise
        call(driver.cuStreamEndCapture, driver.stream, C.byref(graph))
        try:
            count = S()
            call(self.get_nodes, graph, None, C.byref(count))
            nodes = (P * count.value)()
            call(self.get_nodes, graph, nodes, C.byref(count))
            kinds = []
            for node in nodes:
                kind = I()
                call(self.get_node_type, node, C.byref(kind))
                kinds.append(kind.value)
            if kinds.count(EVENT_RECORD_NODE) != expected_events:
                raise RuntimeError(f"Expected {expected_events} external timing nodes, saw {kinds}")
            call(driver.cuGraphInstantiateWithFlags, C.byref(executable), graph, 0)
        finally:
            call(driver.cuGraphDestroy, graph)
        resources.callback(call, driver.cuGraphExecDestroy, executable)
        return executable, len(kinds)

    def capture(self, resources, operation, steps):
        if self.remaining is not None:
            raise RuntimeError("The stage profiler supports a single solve per runtime")
        count = (self.steps - 1) // steps
        if count < 1:
            raise ValueError("The profile needs at least one full replayed graph chunk")
        self.remaining = count
        self.chunk_steps = steps
        self.first_step = 1 + (count-1)*steps
        if self.scheme == "dense":
            trace_count = min(steps, self.samples)
            self.trace = [(None, self.new_markers(len(self.stage_labels) + 1))
                          for _ in range(trace_count)]
        else:
            per_stage = min(self.samples, steps // len(self.stage_labels))
            if per_stage == 0:
                raise ValueError("Sparse stage timing requires at least one step per stage")
            self.trace = [(i, self.new_markers(2)) for _ in range(per_stage)
                          for i in range(len(self.stage_labels))]
        self.control = self.new_markers(min(16, len(self.trace)) + 1)
        base = super().capture(resources, operation, steps)
        start = time.perf_counter()

        def instrumented():
            for _ in range(steps - len(self.trace)):
                operation()
            for chosen, markers in self.trace:
                self.with_markers(operation, chosen, markers)

        expected_events = sum(len(m) for _, m in self.trace)
        tail, self.node_count = self.captured_with_event_count(resources, instrumented, expected_events)
        self.event_count = expected_events

        def marker_only():
            for event in self.control:
                self.record(event)

        control, self.control_node_count = self.captured_with_event_count(
            resources, marker_only, len(self.control))
        self.profile_graph_setup_seconds = time.perf_counter() - start

        def cuGraphLaunch(executable, stream):
            if executable.value == base.value:
                self.remaining -= 1
                if self.remaining == 0:
                    status = self.graph_launch(tail, stream)
                    if status:
                        return status
                    self.profiled = True
                    return self.graph_launch(control, stream)
            return self.graph_launch(executable, stream)

        self.driver.cuGraphLaunch = cuGraphLaunch
        return base

    def intervals(self, markers):
        row = []
        for a, b in zip(markers, markers[1:]):
            duration = C.c_float()
            call(self.event_elapsed, C.byref(duration), a, b)
            row.append(duration.value)
        return row

    def profile(self):
        if not self.profiled or self.remaining != 0:
            raise RuntimeError("The instrumented graph was not replayed")
        control = self.intervals(self.control)
        stage_data = {name:[] for name in self.stage_labels}
        step_total = []
        for i, (chosen, markers) in enumerate(self.trace):
            n = self.first_step + self.chunk_steps - len(self.trace) + i
            values = self.intervals(markers)
            if chosen is None:
                for name, value in zip(self.stage_labels, values):
                    stage_data[name].append(dict(step=n, ms=value))
                step_total.append(sum(values))
            else:
                stage_data[self.stage_labels[chosen]].append(dict(step=n, ms=values[0]))
        medians = {name:float(np.median([r["ms"] for r in rows])) for name, rows in stage_data.items()}
        return dict(kind="captured CUDA graph tail with external timing nodes",
                    scheme=self.scheme, sampled_steps=len(self.trace), chunk_steps=self.chunk_steps,
                    graph_nodes=self.node_count,
                    event_record_nodes=self.event_count, marker_control_nodes=self.control_node_count,
                    marker_control_ms=control, stage_labels=self.stage_labels, stage_samples_ms=stage_data,
                    median_stage_ms=medians, median_step_ms=float(np.median(step_total)) if step_total else None,
                    profiling_graph_setup_seconds=self.profile_graph_setup_seconds)


class GraphGalerkinProfile(GraphStageMixin, Runtime):
    stage_labels = ("radial_reconstruction", "inverse_fft", "products_and_history", "forward_fft",
                    "projection_and_advance", "probes_checkpoints_and_step")
    name_map = dict(spectrum="radial_reconstruction", products="products_and_history",
                    advance="projection_and_advance", observe="probes_checkpoints_and_step")


class GraphFDProfile(GraphStageMixin, FDRuntime):
    stage_labels = ("radial_stencils", "inverse_fft", "products_history_mean", "forward_fft",
                    "rhs", "linear_solve", "probe_snapshot_step")
    name_map = dict(fd_fields="radial_stencils", fd_products="products_history_mean",
                    fd_rhs="rhs", fd_dense="linear_solve", fd_observe="probe_snapshot_step")


def profile_trajectory(method, settings, *, steps, horizon, snapshots, graph_steps=24,
                       samples=8, compare=True, scheme="sparse"):
    is_fd = method == "fd4"
    solve = solve_fd_native if is_fd else solve_native
    runtime_class = GraphFDProfile if is_fd else GraphGalerkinProfile
    if graph_steps < 1:
        raise ValueError("The stage profiler needs a graph chunk")
    arguments = dict(settings, steps=steps, horizon=horizon, snapshots=snapshots,
                     graph_steps=graph_steps)
    if is_fd:
        arguments["cuda_linear_solve"] = "dense"
    with runtime_class(steps, samples, scheme=scheme) as runtime:
        recorded = solve(**arguments, runtime=runtime)
        profile = runtime.profile()
    result = dict(method=method, diagnostic_only=True, profile=profile)
    if compare:
        baseline = solve(**arguments)
        fields = ("states", "transverse", "vorticity")
        result["bitwise_trajectory_match"] = {f:bool(np.array_equal(
            getattr(recorded, f), getattr(baseline, f))) for f in fields}
        result["trajectory_max_abs_difference"] = {f:float(np.max(np.abs(
            getattr(recorded, f) - getattr(baseline, f)))) for f in fields}
        if not all(result["bitwise_trajectory_match"].values()):
            raise FloatingPointError("Adding timing nodes changed the trajectory")
    return result


def main():
    from cpu_formal import PROTOCOL, config, read, save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=2.)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--scheme", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--graph-steps", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    specs = {s["name"]:s for s in protocol["variants"]}
    result = dict(status="running", diagnostic_only=True, rows=[])
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output, result)
    for name in args.names:
        spec = specs[name]
        seed = protocol["seeds"][0] if spec["method"] == "rfm" else protocol["deterministic_seed"]
        _, settings = config(spec, protocol, seed)
        row = profile_trajectory(spec["method"], settings, steps=round(spec["rate"]*args.horizon),
                                 horizon=args.horizon, snapshots=(args.horizon,), samples=args.samples,
                                 graph_steps=args.graph_steps, scheme=args.scheme)
        row.update(name=name, horizon=args.horizon)
        result["rows"].append(row)
        save(args.output, result)
        print(name, json.dumps(row["profile"]["median_stage_ms"]), row["bitwise_trajectory_match"], flush=True)
    result["status"] = "complete"
    save(args.output, result)


if __name__ == "__main__":
    main()
