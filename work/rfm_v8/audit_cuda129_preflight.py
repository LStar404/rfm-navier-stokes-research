"""Validate frozen FP64 kernels and the actual-size cuFFT graph paths."""

import argparse
from contextlib import ExitStack
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path

from cuda129_runtime import HERE, V7, VENDOR, PinnedBundledRuntime, VARIANTS, bundled_source, call
from verify_galerkin_cuda_module_bundle import fixed_kernels
import numpy as np


OUTPUT = HERE/"cuda129_preflight.json"
FFT_GATE = 2e-13
KERNEL_GATE = 2e-13


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))


def compare(a,b):
    return dict(shape=list(a.shape),dtype=a.dtype.str,byte_exact=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),
                maximum_absolute=float(np.max(abs(a-b))),relative_l2=relative(a,b),
                candidate_sha256=sha256(a.tobytes()).hexdigest(),control_sha256=sha256(b.tobytes()).hexdigest())


def fft_case(rt,n,batch):
    values = np.random.default_rng(76321+n+batch).normal(size=(batch,n))
    spectrum = np.empty((batch,n//2+1),dtype=np.complex128)
    returned = np.empty_like(values)
    graph_spectrum,graph_returned = np.empty_like(spectrum),np.empty_like(returned)
    with ExitStack() as resources:
        dx,dy,dz = (rt.upload(resources,arr) for arr in (values,np.zeros_like(spectrum),np.zeros_like(returned)))
        pf,pi = rt.plan(resources,n,batch,False),rt.plan(resources,n,batch,True)

        def both():
            call(rt.d2z,pf,dx,dy)
            call(rt.z2d,pi,dy,dz)

        call(rt.d2z,pf,dx,dy)
        rt.driver.download(dy,spectrum)
        call(rt.z2d,pi,dy,dz)
        rt.driver.download(dz,returned)
        forward_exec = rt.capture(resources,lambda:call(rt.d2z,pf,dx,dy),1)
        call(rt.driver.cuGraphLaunch,forward_exec,rt.driver.stream)
        rt.driver.download(dy,graph_spectrum)
        executable = rt.capture(resources,both,1)
        call(rt.driver.cuGraphLaunch,executable,rt.driver.stream)
        rt.driver.download(dz,graph_returned)
    forward = compare(spectrum,np.fft.rfft(values,axis=-1))
    inverse = compare(returned/n,values)
    capture_forward = compare(graph_spectrum,spectrum)
    capture_inverse = compare(graph_returned,returned)
    return dict(n=n,batch=batch,forward=forward,roundtrip=inverse,graph_forward=capture_forward,graph_inverse=capture_inverse,
                passed=max(forward["relative_l2"],inverse["relative_l2"])<=FFT_GATE and
                       capture_forward["byte_exact"] and capture_inverse["byte_exact"])


def archived_shapes():
    frozen = V7/"prepared_qualification_output"/"protocol.json"
    inputs,shapes = {},set()
    for scale,name in (("mid","rfm37_k72"),("high","rfm41_k80")):
        p = V7/"prepared_qualification_output"/f"{scale}_{name}_7331_combined.json"
        row = json.loads(p.read_text(encoding="utf-8"))["metadata"]
        n,q = row["ntheta"],row["fft_radial_lines"]
        shapes.update(((n,2*q),(n,3*q)))
        inputs[p.name] = digest(p)
    inputs[frozen.name] = digest(frozen)
    return sorted(shapes),inputs


def main(output):
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("CUDA_CACHE_DISABLE")!="1":
        raise ValueError("Run this correctness probe with the process-local CUDA cache disabled")
    manifest = VENDOR/"manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["status"]!="installed":
        raise ValueError("The pinned CUDA runtime is missing")
    for item in data["installed_members"]:
        if digest(VENDOR/item["path"])!=item["sha256"]:
            raise ValueError(f"Vendor runtime checksum differs: {item['path']}")
    shapes,inputs = archived_shapes()
    baseline = None
    rows = []
    for variant in VARIANTS:
        try:
            with PinnedBundledRuntime(variant=variant) as rt:
                evidence = rt.evidence
                actual = fixed_kernels(rt)
                ffts = [fft_case(rt,n,batch) for n,batch in shapes]
            if baseline is None:
                if variant!="legacy":
                    raise RuntimeError("Legacy control unavailable")
                baseline = actual
            kernels = {name:[compare(a,b) for a,b in zip(actual[name],expected,strict=True)]
                       for name,expected in baseline.items()}
            passed = all(item["relative_l2"]<=KERNEL_GATE for outputs in kernels.values() for item in outputs)
            passed = passed and all(item["passed"] for item in ffts)
            rows.append(dict(variant=variant,passed=passed,runtime=evidence,fixed_kernels=kernels,fft_graph=ffts))
            print(variant,passed,"FFT worst",max(x["forward"]["relative_l2"] for x in ffts),flush=True)
        except Exception as exc:
            rows.append(dict(variant=variant,passed=False,failure_type=type(exc).__name__,
                             failure=str(exc).replace(str(HERE.parent.parent),"<workspace>").replace(str(Path.home()),"<user>")))
            print(variant,"failed",type(exc).__name__,flush=True)
    good = all(r["passed"] for r in rows)
    summary = dict(status="passed" if good else "failed",source_sha256=digest(Path(__file__)),
                   runtime_source_sha256=digest(HERE/"cuda129_runtime.py"),vendor_manifest_sha256=digest(manifest),
                   unchanged_bundled_source_sha256=sha256(bundled_source()).hexdigest(),archive_shape_sources=inputs,
                   kernel_relative_gate=KERNEL_GATE,fft_relative_gate=FFT_GATE,rows=rows,process_cuda_cache_disabled=True,
                   scope="Fixed kernels and actual-size FFT graph correctness only. No isolated timing.")
    with output.open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
        stream.write("\n")
    return good


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    raise SystemExit(0 if main(args.output) else 1)
