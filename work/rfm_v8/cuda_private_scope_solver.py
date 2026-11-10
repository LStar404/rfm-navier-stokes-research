"""Thin stock cuFFT march using the qualified private context and explicit PTX."""

from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from types import FunctionType

from build_cuda_ptx_artifacts import TOOLKIT
from cuda_private_context_scope import FreshScopedDriver
from cuda_ptx_march_runtime import ExplicitPtxMarchRuntime
from fused_march_runtime import FusedMarchRuntime
from fused_marches import solve_combined_fused, solve_fd_fused, solve_uniform_fused


DRIVER_API=13020
CONTEXT_MODE="private_cuCtxCreate_v2_scoped_cufft"
ENGINE="warp_cufft"
POLICIES=("disabled","empty_first","strict_hit")
MARCHES={"uniform":solve_uniform_fused,"combined":solve_combined_fused,"fd":solve_fd_fused}


def scoped_plan_stack(get_driver):
    class ScopedPlanStack(ExitStack):
        def callback(self,callback,*args,**kwargs):
            if callback.__name__=="call" and args and args[0].__name__=="cufftDestroy":
                args=(get_driver().guard_plan_destroy(args[0]),*args[1:])
            return super().callback(callback,*args,**kwargs)

    return ScopedPlanStack


def bind_globals(function,**bindings):
    bound=FunctionType(function.__code__,dict(function.__globals__,**bindings),
                       function.__name__,function.__defaults__,function.__closure__)
    bound.__kwdefaults__=function.__kwdefaults__
    return bound


class RegisteredPrivateDriver(FreshScopedDriver):
    def __init__(self,*args):
        super().__init__(*args)
        if self.version!=DRIVER_API:
            raise RuntimeError("CUDA Driver API version differs from the qualified device")


class StockScopedRuntime(FusedMarchRuntime):
    __init__=bind_globals(FusedMarchRuntime.__init__,_OwnedDriver=RegisteredPrivateDriver)


class CachedScopedRuntime(ExplicitPtxMarchRuntime):
    __init__=bind_globals(ExplicitPtxMarchRuntime.__init__,_OwnedDriver=RegisteredPrivateDriver)


def bind_march(arm,policy,settings):
    original=MARCHES[arm]
    created=[]
    runtime_class=StockScopedRuntime if policy=="disabled" else CachedScopedRuntime

    def runtime(*args):
        value=runtime_class(*args,**settings)
        created.append(value)
        return value

    stack=scoped_plan_stack(lambda: created[0].driver)
    return bind_globals(original,FusedMarchRuntime=runtime,ExitStack=stack),created


def solve_scoped(arm,*,policy,engine=ENGINE,artifact_directory=None,expected_artifact_sha256=None,
                 compiler_identity=None,toolkit=None,**solve_args):
    """Run one qualified stock march and charge all private-context preparation."""
    started=perf_counter()
    if arm not in MARCHES or policy not in POLICIES or engine!=ENGINE:
        raise ValueError("Use a registered stock solver arm and explicit PTX policy")
    if solve_args.get("profile_steps",0)!=0:
        raise ValueError("Private-context qualification does not include event profiling")
    settings={}
    if policy!="disabled":
        if artifact_directory is None or not expected_artifact_sha256 or compiler_identity is None:
            raise ValueError("Cached solvers require the registered directory, artifact SHA and compiler contract")
        directory=Path(artifact_directory)
        if policy=="empty_first" and directory.exists():
            raise ValueError("Each first build requires an unused cache namespace")
        settings=dict(policy=policy,artifact_directory=directory,
                      expected_artifact_sha256=expected_artifact_sha256,compiler_identity=compiler_identity)
    bound,created=bind_march(arm,policy,settings)
    solution=bound(engine=ENGINE,toolkit=toolkit or TOOLKIT,**solve_args)
    if len(created)!=1 or created[0].driver.current() is not None:
        raise RuntimeError("The stock march must own and release exactly one private context")
    solution.row.update(ptx_cache_policy=policy,cuda_context_mode=CONTEXT_MODE,cuda_driver_api=DRIVER_API,
                        private_context_finally_unbound=True)
    if settings:
        cache=created[0].cache_row
        if (cache["ptx_nvrtc_seconds"]>0)!=(policy=="empty_first"):
            raise RuntimeError("Observed NVRTC preparation differs from the registered policy")
        solution.row.update(cache,ptx_cache_observed_mode="miss" if policy=="empty_first" else "hit")
    solution.row["seconds"]=perf_counter()-started
    return solution
