"""Fixed-state GPU comparison of fused and original cuFFT nonlinear nodes."""

import argparse
from contextlib import ExitStack
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from fused_nonlinear_cuda_runtime import FusedNonlinearCUDA, call
from verify_fused_nonlinear_fft import compare, make_spectra, reference, require
from galerkin_cuda_module_bundle import BundledSwitchRuntime
from fd_cuda_native import FDRuntime


HERE = Path(__file__).resolve().parent


def download(rt, address, prototype):
    result=np.empty_like(prototype)
    rt.driver.download(address,result)
    return result


def stock_node(rt, stack, family, n, rows, k, flux, history, mean):
    fd=family=="fd"
    inputs,outputs=(4,1) if fd else (3,2)
    half=n//2+1
    physical=rt.upload(stack,np.zeros((inputs,rows,n)))
    products=rt.upload(stack,np.zeros((outputs,rows,n)))
    inverse=rt.plan(stack,n,inputs*rows,True)
    forward=rt.plan(stack,n,outputs*rows,False)
    prime=rt.upload(stack,np.zeros((inputs,rows,half),dtype=np.complex128))
    call(rt.z2d,inverse,prime,physical)
    call(rt.d2z,forward,products,prime)
    rt.sync()

    def bind_state(spectrum,terms,step):
        multiply=(rt.launch("fd_products",rows*n,physical,flux,terms,step,products,history,mean,rows,k+1,n,half) if fd else
                  rt.launch("products",max(rows*n,2*rows*half),physical,flux,products,history,rows*n,2*rows*half))

        def evaluate():
            call(rt.z2d,inverse,spectrum,physical)
            multiply()
            call(rt.d2z,forward,products,flux)
        return evaluate
    return bind_state


def one_case(rt, family, n, k, rows, toolkit, rng):
    fd=family=="fd"
    inputs,outputs=(4,1) if fd else (3,2)
    spectra=[make_spectra(rng,inputs,rows,k,n,mode) for mode in ("full","smooth","edge","imbalanced")]
    old=rng.normal(size=(outputs,rows,n//2+1))+1j*rng.normal(size=(outputs,rows,n//2+1))
    terms=[rng.normal(size=(rows,k+1)) for _ in spectra] if fd else [None]*len(spectra)
    mean_start=np.full((2,rows),12345.,dtype=np.float64)
    expected=[reference(spectrum,old,n,term)[0] for spectrum,term in zip(spectra,terms,strict=True)]
    result={}
    for variant in ("stock","fused"):
        with ExitStack() as stack:
            device_spectra=[rt.upload(stack,spectrum) for spectrum in spectra]
            device_terms=[rt.upload(stack,term) if term is not None else None for term in terms]
            steps=[rt.upload(stack,np.array([index],dtype=np.int32)) for index in (1,2,3,4)]
            flux=rt.upload(stack,old)
            history=rt.upload(stack,np.zeros_like(old))
            mean=rt.upload(stack,mean_start) if fd else None
            if variant=="stock":
                bind_state=stock_node(rt,stack,family,n,rows,k,flux,history,mean)
            else:
                fused=FusedNonlinearCUDA(rt,stack,n=n,k=k,rows=rows,family=family,toolkit=toolkit)
                def bind_state(spectrum,terms,step):
                    return fused.launch(spectrum,flux,history,mean_terms=terms,step=step,mean=mean)
            operation=[bind_state(*data) for data in zip(device_spectra,device_terms,steps,strict=True)]
            operation[0]()
            first=download(rt,flux,old)
            require(download(rt,history,old).tobytes()==old.tobytes(),"Eager node changed original raw flux history")

            def two_states():
                operation[1]()
                operation[2]()

            graph=rt.capture(stack,two_states,1)
            call(rt.driver.cuGraphLaunch,graph,rt.driver.stream)
            after=download(rt,flux,old)
            graph_prior=download(rt,history,old)
            operation[3]()
            imbalanced=download(rt,flux,old)
            require(download(rt,history,old).tobytes()==after.tobytes(),"Next launch did not copy captured raw history")
            fd_mean_exact=None
            if fd:
                mean_expected=np.stack([reference(spectra[i],old,n,terms[i])[2] for i in (3,2)])
                fd_mean_exact=download(rt,mean,mean_start).tobytes()==mean_expected.tobytes()
                require(fd_mean_exact,"FD mean or time-ring slot differs")
            result[variant]=dict(raw=(first,graph_prior,after,imbalanced),mean_byte_exact=fd_mean_exact)
    arrays=[]
    for i,profile in enumerate(("eager_full","graph_prior_smooth","graph_final_edge","eager_imbalanced")):
        old_flux=result["stock"]["raw"][i]
        new_flux=result["fused"]["raw"][i]
        comparison=compare(new_flux,old_flux)
        oracle=compare(new_flux,expected[i])
        per_flux=[compare(a,b) for a,b in zip(new_flux,expected[i],strict=True)]
        require(max(comparison["relative_l2"],oracle["relative_l2"])<1e-12,"Fused overall GPU flux differs")
        if i<3:
            require(max(v["relative_l2"] for v in per_flux)<1e-12,"Ordinary fused GPU flux differs by channel")
        arrays.append(dict(profile=profile,vs_cufft=comparison,vs_numpy=oracle,per_flux_vs_numpy=per_flux))
    return dict(family=family,ntheta=n,k=k,radial_rows=rows,arrays=arrays,history_copy_byte_exact=True,
                fd_mean_byte_exact=result["fused"]["mean_byte_exact"])


def run(toolkit):
    cases=[]
    rng=np.random.default_rng(287835)
    for family,runtime,layouts in (
        ("gal",BundledSwitchRuntime,((200,64,60),(225,72,66),(243,80,76))),
        ("fd",FDRuntime,((200,64,60),(225,72,66),(243,80,175),(243,80,183)))):
        with runtime(toolkit) as rt:
            for layout in layouts:
                cases.append(one_case(rt,family,*layout,toolkit,rng))
    return dict(status="passed",scope="Fixed-state eager/graph checks only; no isolated timing or flow trajectory",cases=cases)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--toolkit",type=Path)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    require(not args.output.exists(),"Preserve prior fixed-state CUDA evidence")
    result=run(args.toolkit)
    files=[Path(__file__),*(HERE/name for name in ("fused_nonlinear_fft.py","fused_nonlinear_fft.cu",
            "fused_nonlinear_cuda_runtime.py","verify_fused_nonlinear_fft.py")),
           *(HERE.parent/"rfm_v7"/name for name in ("galerkin_cuda_native.cu","fd_cuda_native.cu",
            "galerkin_cuda_module_bundle.py","fd_cuda_native.py"))]
    result["source_sha256"]={}
    for path in files:
        with path.open("rb") as handle:
            result["source_sha256"][path.relative_to(HERE.parent).as_posix()]=file_digest(handle,"sha256").hexdigest()
    with args.output.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print("passed",len(result["cases"]))
