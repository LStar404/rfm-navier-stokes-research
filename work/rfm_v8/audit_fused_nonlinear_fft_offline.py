"""Audit CUDA stage ownership and FP64 conditioning without a GPU context."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np

import fused_nonlinear_fft as fused
from verify_fused_nonlinear_fft import compare, reference, require, retained_convolution


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
SOURCE = HERE/"fused_nonlinear_fft.cu"


def fingerprint(path):
    with path.open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def stage_contract(source, n):
    algorithm=source.split("void transform(double2 *scratch",1)[1].split("template<int N>\n",1)[0]
    cases=re.split(r"\}\s*else(?:\s+if\s+\(N == 225\))?\s*\{",algorithm)
    require(len(cases)==3,"Unexpected CUDA compile-time transform cases")
    actual=[(int(stride),int(radix)) for stride,radix in re.findall(r"stage<N,(\d+),(\d+),Inverse,Planes>",cases[(200,225,243).index(n)])]
    expected=fused.plan(n).radices
    require([radix for _,radix in actual]==list(expected),"CPU and CUDA radix plans disagree")
    for planes in (1,2):
        stride_expected=1
        for stride,radix in actual:
            require(stride==stride_expected and n%(stride*radix)==0,"CUDA stage stride does not partition N")
            nbut=n//radix
            ownership={}
            for lane in range(128):
                for at in range(lane,planes*nbut,128):
                    plane,column=divmod(at,nbut)
                    group,j=divmod(column,stride)
                    for t in range(radix):
                        index=plane*n+group*radix*stride+j+t*stride
                        require(index not in ownership,"Two threads write the same butterfly slot")
                        require(0<=t*j*(n//(radix*stride))<n,"CUDA twiddle index escapes its table")
                        ownership[index]=lane
            require(sorted(ownership)==list(range(planes*n)),"Stage ownership does not cover shared memory")
            stride_expected*=radix
        require(stride_expected==n,"CUDA transform did not reach the full length")
    permutation=fused.plan(n).permutation.tolist()
    require(all(0<=min(k,n-k)<=n//2 for k in permutation) and sorted(permutation)==list(range(n)),
            "Hermitian lookup or product permutation escapes its bounds")
    product_owners={i:lane for lane in range(128) for i in (lane,lane+128) if i<n}
    require(sorted(product_owners)==list(range(n)) and all(i<n for i in range(128)) and n//2+1<=128,
            "Fixed 128-thread block geometry differs from supported layouts")
    return dict(ntheta=n,stages=actual,one_and_two_plane_writes_disjoint=True,
                packed_product_permutation_complete=True,hermitian_and_twiddle_bounds_valid=True)


def global_contract(source):
    gal=(V7/"galerkin_cuda_native.cu").read_text(encoding="utf-8")
    fd=(V7/"fd_cuda_native.cu").read_text(encoding="utf-8")
    wrapper=(HERE/"fused_nonlinear_cuda_runtime.py").read_text(encoding="utf-8")
    body=source.split("void nonlinear(",1)[1].split("} // namespace",1)[0]
    needles=("history[row+lane] = flux[row+lane];","transform<N,true,2>(scratch,roots);",
             "const double2 first = physical_product<N,FD>","__syncthreads();\n    scratch[lane] = first;",
             "__syncthreads();\n    transform<N,false,1>(scratch,roots);","flux[row+lane] = make_double2(")
    offsets=[body.index(needle) for needle in needles]
    require(offsets==sorted(offsets) and "history[plane+row+lane] = flux[plane+row+lane]" in body,
            "Shared read/write barrier or old raw flux history order differs")
    require("__syncthreads();" in source.split("void stage(",1)[1].split("void transform(",1)[0] and
            "__syncthreads();\n    transform<N,true,2>" in body and "driver.cuLaunchKernel,self.kernel,self.rows,1,1,128" in wrapper,
            "Stage barrier or one-block-per-row launch geometry differs")
    require("const double2 *spectrum" in body and "const double *mean_terms, const int *step" in body and
            "state" not in body and "mean[(*step%2)*rows+radial]" in body and
            "for (int mode = 1; mode < modes; ++mode) value += mean_terms[radial*modes+mode];" in body and
            "mean[(*step % 2)*w+row]" in fd and "previous[i] = spectrum[i]" in gal and "prior[i] = spectrum[i]" in fd,
            "Read-only inputs, state exclusion, or FD mean history differs")
    layouts=[]
    for n,rows,fields,out in ((225,66,3,2),(243,76,3,2),(243,175,4,1),(243,183,4,1)):
        half=n//2+1
        addresses=[row*half+k+plane*rows*half for row in range(rows) for k in range(half) for plane in range(out)]
        require(len(set(addresses))==len(addresses)==rows*half*out and min(addresses)==0 and
                max(addresses)==rows*half*out-1,"Radial blocks overlap output or history addresses")
        layouts.append(dict(ntheta=n,rows=rows,input_fields=fields,output_fields=out,history_spectrum_slots=len(addresses)))
    return dict(phase_order_checked=True,source_stage_barriers=True,state_not_passed=True,
                input_spectra_mean_terms_and_step_const=True,ascending_fd_mean_and_ring_same=True,layouts=layouts,
                host_precondition="Spectrum, flux, history and FD mean buffers are independent allocations. Equal-pointer validation is not an arbitrary subview-overlap detector.")


def cancellation_oracles():
    n,k=243,80
    fft=fused.plan(n)
    spectrum=np.zeros((4,1,n//2+1),dtype=np.complex128)
    spectrum[0,0,1]=.5+.25j
    spectrum[2,0,2]=.125-.25j
    spectrum[3]=-spectrum[2]
    terms=np.zeros((1,k+1))
    terms[0,1:7]=(1e16,1,-1e16,2,-2,3)
    old=np.zeros((1,1,n//2+1),dtype=np.complex128)
    old.view(np.float64)[0,0,1]=-0.
    clean=retained_convolution(spectrum[0],spectrum[2],n,k)
    normalizer=float(np.linalg.norm(clean))
    fd=[]
    for exponent in (10,20,30,40,48,None):
        delta=2.**(-exponent) if exponent is not None else 0.
        spectrum[1]=(1+delta)*spectrum[0]
        new,prior,mean=fused.nonlinear(spectrum,old,fft,k,terms)
        standard,_,expected_mean=reference(spectrum,old,n,terms)
        exact=-delta*clean
        require(prior.tobytes()==old.tobytes() and mean.tobytes()==expected_mean.tobytes() and mean[0]==3.,
                "FD ordered mean or raw history failed the signed-cancellation fixture")
        metrics=[]
        for result in (new[0,:,:k+1],standard[0,:,:k+1]):
            error=float(np.linalg.norm(result-exact))
            metrics.append(dict(error_over_uncancelled_term=error/normalizer,
                                relative_error_if_nonzero=error/(delta*normalizer) if delta else None))
        require(max(item["error_over_uncancelled_term"] for item in metrics)<2e-15,
                "FD advective cancellation loses absolute FP64 consistency")
        fd.append(dict(delta=delta,fused=metrics[0],numpy_separate=metrics[1]))

    gal=[]
    for exponent in (0,4,8,10,12,14):
        ratio=10.**(-exponent)
        data=spectrum[:3].copy()
        data[0]*=ratio
        data[1]=spectrum[0]
        history=np.zeros((2,1,n//2+1),dtype=np.complex128)
        new,_,_=fused.nonlinear(data,history,fft,k)
        expected,_,_=reference(data,history,n,None)
        weak=compare(new[0],expected[0])
        all_flux=compare(new,expected)
        require(all_flux["relative_l2"]<2e-15,"Packed imbalance loses absolute FP64 consistency")
        gal.append(dict(first_channel_ratio=ratio,weak_flux=weak,all_flux=all_flux))
    return dict(fd_near_cancellation=fd,gal_weak_channel=gal,fd_mean_signed_fixture_byte_exact=True,
                history_signed_zero_byte_exact=True,warning="Tiny per-channel or cancelled outputs are ill-conditioned. Neither aggregate flux norm nor this fixed CPU check substitutes for the existing T72 per-field and probe gates.")


def audit():
    source=SOURCE.read_text(encoding="utf-8")
    paths=(Path(__file__),SOURCE,HERE/"fused_nonlinear_fft.py",HERE/"fused_nonlinear_cuda_runtime.py",
           HERE/"verify_fused_nonlinear_fft.py",V7/"galerkin_cuda_native.cu",V7/"fd_cuda_native.cu")
    return dict(status="passed",scope="CPU source/index and conditioning checks only. No kernel, device query, or trajectory.",
                source_sha256={path.relative_to(HERE.parent).as_posix():fingerprint(path) for path in paths},
                stages=[stage_contract(source,n) for n in (200,225,243)],global_contract=global_contract(source),
                stress=cancellation_oracles())


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    require(not args.output.exists(),"Preserve prior offline audit")
    result=audit()
    with args.output.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print("passed")
