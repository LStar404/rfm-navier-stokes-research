"""One-module CUDA 11.8 runtime for matched warp-only and fused marches."""

from contextlib import ExitStack
import ctypes as C
import os
from pathlib import Path
import re
import time

from warp_observe_cuda import ObservationRuntime, _OwnedDriver, source_and_names
from cuda_environment_native import DP, I, P, bind, call, nvrtc_load
from fused_nonlinear_cuda_runtime import FusedNonlinearCUDA, SOURCE, SUPPORTED_LENGTHS
from galerkin_cuda_native import TOOLKIT_DEFAULT


def active_fused_source(family,n):
    if family not in ("gal","fd") or n not in SUPPORTED_LENGTHS:
        raise ValueError("Unsupported fused nonlinear family or angular length")
    source=SOURCE.read_bytes()
    calls=re.findall(rb"^RFM_FUSED_ENTRY\(fused_(?:gal|fd)_\d+,\d+,(?:true|false)\)\r?$",source,re.MULTILINE)
    name=f"fused_{family}_{n}"
    if len(calls)!=6 or sum(name.encode("ascii")+b"," in entry for entry in calls)!=1:
        raise ValueError("Frozen fused entry list differs")
    for entry in calls:
        if name.encode("ascii")+b"," not in entry:
            source=source.replace(entry+b"\n",b"",1)
    if len(re.findall(rb"^RFM_FUSED_ENTRY\(",source,re.MULTILINE))!=1:
        raise ValueError("The march must compile exactly its active fused entry")
    return name,source


def march_source(family,n,fused):
    source,names,observer=source_and_names(family,True)
    if not fused:
        return source,names,observer
    name,active=active_fused_source("fd" if family=="fd" else "gal",n)
    return source+b'\n#line 1 "fused_nonlinear_fft.cu"\n'+active,(*names,name),observer


class FusedMarchRuntime(ObservationRuntime):
    def __init__(self,family,engine,n,toolkit=None):
        if engine not in ("warp_cufft","warp_fused"):
            raise ValueError("Use either the warp-only cuFFT or warp fused engine")
        root=Path(toolkit or os.environ.get("CUDA_PATH",TOOLKIT_DEFAULT))
        self.fused=engine=="warp_fused"
        self.warp=True
        source,names,self.observer=march_source(family,n,self.fused)
        with ExitStack() as resources:
            resources.enter_context(os.add_dll_directory(str(root/"bin")))
            self.driver_lib=C.WinDLL("nvcuda.dll")
            self.driver=_OwnedDriver(self.driver_lib,resources)
            begin=time.perf_counter()
            self.module=nvrtc_load(root,self.driver,source,b"rfm_v8_warp_march.cu")
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
            self.resources=resources.pop_all()


def nonlinear_nodes(rt,resources,family,n,k,rows,spectrum,physical,products,flux,history,
                    *,mean_terms=None,step=None,mean=None):
    fd=family=="fd"
    if rt.fused:
        packed=FusedNonlinearCUDA(rt,resources,n=n,k=k,rows=rows,family=family,module=rt.module)
        operation=packed.launch(spectrum,flux,history,mean_terms=mean_terms,step=step,mean=mean)
        return (operation,),(operation,),("fused_inverse_product_forward",)
    half=n//2+1
    inverse=rt.plan(resources,n,(4 if fd else 3)*rows,True)
    forward=rt.plan(resources,n,(1 if fd else 2)*rows,False)
    product=(rt.launch("fd_products",rows*n,physical,flux,mean_terms,step,products,history,mean,rows,k+1,n,half) if fd else
             rt.launch("products",max(rows*n,2*rows*half),physical,flux,products,history,rows*n,2*rows*half))
    backwards=lambda:call(rt.z2d,inverse,spectrum,physical)
    forwards=lambda:call(rt.d2z,forward,products,flux)
    return (backwards,product,forwards),(backwards,forwards),("inverse_fft","products_and_history","forward_fft")


def nonlinear_metadata(rt,steps,rows,family):
    fd=family=="fd"
    inverse,forward=(4,1) if fd else (3,2)
    return dict(nonlinear_engine="warp_fused" if rt.fused else "warp_cufft",observation_reduction="warp",
                compiled_cuda_modules=1,active_fused_entries=int(rt.fused),nonlinear_evaluations=steps,
                logical_fft_groups=2*steps,inverse_field_transforms=inverse*steps,forward_field_transforms=forward*steps,
                fft_radial_lines=rows,logical_field_transforms_all_radial_rows=5*steps*rows,
                cufft_marching_calls=0 if rt.fused else 2*steps,nonlinear_graph_nodes_per_evaluation=1 if rt.fused else 3,
                priming_logical_field_transforms=5*rows,priming_cuda_nodes=1 if rt.fused else 2,
                nvrtc_compile_load_seconds=rt.compile_load_seconds)
