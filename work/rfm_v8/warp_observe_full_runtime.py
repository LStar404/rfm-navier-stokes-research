"""Scope full-run observation runtime costs without shadowing the Galerkin family."""

import time

from warp_observe_cuda import ObservationRuntime, solve_fd_native, solve_native


def solve_with_observation(solver_family,warp,/,*,toolkit=None,**settings):
    if solver_family not in ("uniform","fd"):
        raise ValueError("Use the same-march adapter for combined Galerkin")
    start = time.perf_counter()
    with ObservationRuntime(solver_family,warp,toolkit) as runtime:
        runtime_setup = time.perf_counter()-start
        solver = solve_native if solver_family == "uniform" else solve_fd_native
        result = solver(runtime=runtime,toolkit=toolkit,**settings)
    result.row.update(runtime_init_seconds=runtime_setup,seconds=time.perf_counter()-start,
                      observation_reduction="warp" if warp else "serial")
    return result
