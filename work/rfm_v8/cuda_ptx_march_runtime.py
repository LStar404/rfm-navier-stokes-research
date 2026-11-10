"""Bind explicit PTX preparation to the unchanged one-module warp march."""

from contextlib import ExitStack
import ctypes as C
from functools import partial
import os
from pathlib import Path
import time
from types import FunctionType

from build_cuda_ptx_artifacts import recipe, TOOLKIT
from cuda_environment_native import DP, I, P, bind, call
from cuda_ptx_cache import Nvrtc, artifact, key_for, load_device
from fused_march_runtime import FusedMarchRuntime, march_source
from warp_observe_cuda import _OwnedDriver


class ExplicitPtxMarchRuntime(FusedMarchRuntime):
    def __init__(self,family,engine,n,toolkit,*,policy,artifact_directory,expected_artifact_sha256,compiler_identity):
        if policy not in ("strict_hit","empty_first"):
            raise ValueError("Unknown explicit artifact preparation policy")
        root=Path(toolkit or TOOLKIT)
        self.fused=engine=="warp_fused"
        self.warp=True
        begin=time.perf_counter()
        source,spec=recipe(family,engine,n,compiler_identity)
        original,names,self.observer=march_source(family,n,self.fused)
        if original!=source or sorted(names)!=spec["entry_points"]:
            raise ValueError("PTX recipe differs from the current original runtime")
        recipe_seconds=time.perf_counter()-begin
        with ExitStack() as resources:
            resources.enter_context(os.add_dll_directory(str(root/"bin")))
            context_start=time.perf_counter()
            self.driver_lib=C.WinDLL("nvcuda.dll")
            self.driver=_OwnedDriver(self.driver_lib,resources)
            context_seconds=time.perf_counter()-context_start
            begin=time.perf_counter()
            compiler=Nvrtc(root) if policy=="empty_first" else None
            compiler_identity_seconds=time.perf_counter()-begin if compiler is not None else 0.
            ptx,manifest,read=artifact(artifact_directory,source,spec,compiler=compiler,strict_hit=compiler is None)
            mode="miss" if compiler is not None else "hit"
            if read["mode"]!=mode or read["artifact_sha256"]!=expected_artifact_sha256:
                raise ValueError("PTX preparation did not use the registered artifact and policy")
            self.module,load_seconds=load_device(self.driver,ptx)
            self.compile_load_seconds=time.perf_counter()-begin
            resources.callback(call,self.driver.cuModuleUnload,self.module)
            self.kernels={}
            for name in names:
                handle=P()
                call(self.driver.cuModuleGetFunction,C.byref(handle),self.module,name.encode("ascii"))
                self.kernels[name]=handle
            if not self.fused:
                self.fft=C.WinDLL(str(root/"bin"/"cufft64_10.dll"))
                self.plan_many=bind(self.fft,"cufftPlanMany",[C.POINTER(I),I,C.POINTER(I),P,I,I,P,I,I,I,I])
                self.set_stream=bind(self.fft,"cufftSetStream",[I,P])
                self.d2z=bind(self.fft,"cufftExecD2Z",[I,DP,DP])
                self.z2d=bind(self.fft,"cufftExecZ2D",[I,DP,DP])
                self.destroy_plan=bind(self.fft,"cufftDestroy",[I])
            self.cache_row=dict(ptx_cache_policy=policy,ptx_cache_key=key_for(spec),ptx_sha256=manifest["ptx_sha256"],
                ptx_artifact_sha256=read["artifact_sha256"],ptx_recipe_seconds=recipe_seconds,
                ptx_context_seconds=context_seconds,ptx_compiler_identity_seconds=compiler_identity_seconds,
                ptx_cache_total_seconds=read["cache_seconds"],ptx_nvrtc_seconds=read["nvrtc_seconds"],
                ptx_driver_load_seconds=load_seconds)
            self.resources=resources.pop_all()


def solve_with_ptx_policy(arm,*,engine,policy,artifact_directory=None,expected_artifact_sha256=None,
                         compiler_identity=None,toolkit=TOOLKIT,**solve_args):
    from fused_marches import solve_combined_fused, solve_fd_fused, solve_uniform_fused

    original={"uniform":solve_uniform_fused,"combined":solve_combined_fused,"fd":solve_fd_fused}[arm]
    if policy=="disabled":
        result=original(engine=engine,toolkit=toolkit,**solve_args)
        result.row["ptx_cache_policy"]=policy
        return result
    runtime_instances=[]
    build=partial(ExplicitPtxMarchRuntime,policy=policy,artifact_directory=artifact_directory,
                  expected_artifact_sha256=expected_artifact_sha256,compiler_identity=compiler_identity)

    def runtime(*args):
        instance=build(*args)
        runtime_instances.append(instance)
        return instance

    solver=FunctionType(original.__code__,dict(original.__globals__,FusedMarchRuntime=runtime),
                        original.__name__,original.__defaults__,original.__closure__)
    solver.__kwdefaults__=original.__kwdefaults__
    result=solver(engine=engine,toolkit=toolkit,**solve_args)
    if len(runtime_instances)!=1:
        raise ValueError("Each march must own one module runtime")
    result.row.update(runtime_instances[0].cache_row)
    return result
