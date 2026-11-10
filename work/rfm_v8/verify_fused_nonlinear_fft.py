"""Independent NumPy FFT and retained-mode convolution oracle for the fused path."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

import fused_nonlinear_fft as fused


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected):
    error=actual-expected
    return dict(relative_l2=float(np.linalg.norm(error.ravel())/max(np.linalg.norm(expected.ravel()),1e-300)),
                max_abs=float(np.max(abs(error))))


def reference(spectrum, old, n, terms):
    values=np.fft.irfft(spectrum,n=n,norm="forward")
    if terms is None:
        products=values[:2]*values[2]
        return np.fft.rfft(products,n=n),old.copy(),None
    products=values[0]*values[2]+values[1]*values[3]
    mean=np.empty(len(terms))
    for i,row in enumerate(terms):
        value=0.
        for coefficient in row[1:]:
            value+=float(coefficient)
        mean[i]=value
    return np.fft.rfft(products,n=n)[None],old.copy(),mean


def retained_convolution(first,second,n,k):
    """Linear Fourier convolution, independent of FFT and of circular aliasing."""
    output=np.zeros((first.shape[0],k+1),np.complex128)
    # Retained +k receives no circular alias when n > 3k.
    def coefficient(spectrum,mode):
        return spectrum[:,mode] if mode>=0 else spectrum[:,-mode].conj()
    for mode in range(k+1):
        for source in range(-k,k+1):
            target=mode-source
            if -k<=target<=k:
                output[:,mode]+=coefficient(first,source)*coefficient(second,target)
    return n*output


def make_spectra(rng,fields,rows,k,n,profile):
    result=np.zeros((fields,rows,n//2+1),dtype=np.complex128)
    coef=rng.standard_normal((fields,rows,k+1))+1j*rng.standard_normal((fields,rows,k+1))
    coef[...,0]=coef[...,0].real
    if profile=="smooth":
        coef*=1/(1+np.arange(k+1))**2
    elif profile=="edge":
        coef[...,1:k]=0
    elif profile=="imbalanced":
        coef[0]*=1e-10
    result[...,:k+1]=coef
    return result


def run():
    rng=np.random.default_rng(753288)
    lengths=(6,15,200,225,243)
    transforms=[]
    for n in lengths:
        fft=fused.plan(n)
        values=rng.standard_normal((4,n))+1j*rng.standard_normal((4,n))
        forward=compare(fused.transform(values,fft),np.fft.fft(values))
        inverse=compare(fused.transform(values,fft,inverse=True),n*np.fft.ifft(values))
        first,second=rng.standard_normal((2,4,n))
        forward_paired=fused.forward_pair(first,second,fft)
        inverse_paired=fused.inverse_pair(np.fft.rfft(first),np.fft.rfft(second),fft)
        paired_forward=[compare(got,np.fft.rfft(expected)) for got,expected in zip(forward_paired,(first,second),strict=True)]
        paired_inverse=[compare(got,n*expected) for got,expected in zip(inverse_paired,(first,second),strict=True)]
        require(max(row["relative_l2"] for row in (forward,inverse,*paired_forward,*paired_inverse))<3e-14,"FFT transform differs")
        require(len(set(fft.permutation.tolist()))==n,"Mixed digit reversal is not a permutation")
        transforms.append(dict(ntheta=n,forward=forward,inverse=inverse,paired_forward=paired_forward,paired_inverse=paired_inverse))
    cases=[]
    for n,k,rows in ((200,64,60),(225,72,66),(243,80,76),(243,80,175),(243,80,183)):
        fft=fused.plan(n)
        for family in ("galerkin","fd4"):
            fields,outputs=(3,2) if family=="galerkin" else (4,1)
            for profile in ("full","smooth","edge","imbalanced"):
                spectrum=make_spectra(rng,fields,rows,k,n,profile)
                old=rng.standard_normal((outputs,rows,n//2+1))+1j*rng.standard_normal((outputs,rows,n//2+1))
                before=old.copy()
                terms=rng.standard_normal((rows,k+1)) if family=="fd4" else None
                got,history,mean=fused.nonlinear(spectrum,old,fft,k,terms)
                expected,old_expected,mean_expected=reference(spectrum,old,n,terms)
                errors=[compare(new,ref) for new,ref in zip(got,expected,strict=True)]
                full_error=compare(got,expected)
                require(np.array_equal(history,old_expected) and np.array_equal(old,before) and
                        not np.shares_memory(history,old) and
                        (mean is mean_expected is None or np.array_equal(mean,mean_expected)),"History or FD mean differs")
                if profile!="imbalanced":
                    require(max(item["relative_l2"] for item in errors)<4e-14,"Ordinary FP64 nonlinear output differs")
                # The independent retained-mode check uses only two radial rows.
                full=spectrum[:,:2]
                if family=="galerkin":
                    direct=np.stack([retained_convolution(full[field],full[2],n,k) for field in (0,1)])
                else:
                    direct=(retained_convolution(full[0],full[2],n,k)+retained_convolution(full[1],full[3],n,k))[None]
                dealias=compare(got[:,:2,:k+1],direct)
                require(dealias["relative_l2"]<4e-14 and full_error["relative_l2"]<4e-14,"Unnormalized product or retained-mode de-aliasing differs")
                cases.append(dict(ntheta=n,k=k,radial_rows=rows,family=family,profile=profile,all_flux=full_error,
                    per_flux=errors,retained_convolution=dealias,history_byte_exact=True,fd_mean_byte_exact=family=="fd4"))
    rejected={}
    for name,operation in (
        ("unsupported_prime",lambda:fused.plan(7)),
        ("undersized_dealias",lambda:fused.nonlinear(np.zeros((3,1,4),complex),np.zeros((2,1,4),complex),fused.plan(6),2))):
        try:
            operation()
        except ValueError:
            rejected[name]=True
        else:
            raise AssertionError(f"Invalid case passed: {name}")
    return dict(status="passed",scope="CPU algebra and fixed-state arrays; no trajectory, CUDA execution, or performance claim",
                transforms=transforms,cases=cases,invalid_inputs_rejected=rejected,
                cost_model=[fused.cost(n,rows) for n,rows in ((200,60),(225,66),(243,76),(243,175),(243,183))],
                packing_conditioning_worst_relative_l2=max(row["per_flux"][0]["relative_l2"] for row in cases
                    if row["family"]=="galerkin" and row["profile"]=="imbalanced"))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    require(not args.output.exists(),"Preserve the original FFT oracle record")
    result=run()
    result["source_sha256"]={}
    for path in (Path(__file__),Path(fused.__file__)):
        with path.open("rb") as handle:
            result["source_sha256"][path.name]=file_digest(handle,"sha256").hexdigest()
    with args.output.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print("passed",len(result["cases"]),result["packing_conditioning_worst_relative_l2"])
