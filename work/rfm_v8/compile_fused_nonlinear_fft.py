"""Compile the fused CUDA source to PTX without loading a GPU driver context."""

import argparse
import ctypes as C
from hashlib import sha256
import json
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
EXPECTED = {f"fused_{family}_{length}" for family in ("gal","fd") for length in (200,225,243)}


def compile_ptx(toolkit, files):
    with os.add_dll_directory(str(toolkit/"bin")):
        library = C.WinDLL(str(toolkit/"bin"/"nvrtc64_112_0.dll"))
        pointer, char, size = C.c_void_p,C.c_char_p,C.c_size_t

        def bind(name,*args):
            fn = getattr(library,name)
            fn.restype = C.c_int
            fn.argtypes = list(args)
            return fn

        def checked(fn,*args):
            status=fn(*args)
            if status:
                raise RuntimeError(f"{fn.__name__} returned {status}")

        create=bind("nvrtcCreateProgram",C.POINTER(pointer),char,char,C.c_int,pointer,pointer)
        compile_program=bind("nvrtcCompileProgram",pointer,C.c_int,C.POINTER(char))
        get_log_size=bind("nvrtcGetProgramLogSize",pointer,C.POINTER(size))
        get_log=bind("nvrtcGetProgramLog",pointer,pointer)
        get_ptx_size=bind("nvrtcGetPTXSize",pointer,C.POINTER(size))
        get_ptx=bind("nvrtcGetPTX",pointer,pointer)
        destroy=bind("nvrtcDestroyProgram",C.POINTER(pointer))
        source=b"\n".join(f'#line 1 "{path.name}"\n'.encode("ascii")+path.read_bytes() for path in files)
        program=pointer()
        checked(create,C.byref(program),source,b"rfm_fused_compile_probe.cu",0,None,None)
        try:
            status=compile_program(program,1,(char*1)(b"--gpu-architecture=compute_52"))
            if status:
                length=size()
                checked(get_log_size,program,C.byref(length))
                log=C.create_string_buffer(length.value)
                checked(get_log,program,log)
                raise RuntimeError(log.value.decode("utf-8",errors="replace"))
            length=size()
            checked(get_ptx_size,program,C.byref(length))
            ptx=C.create_string_buffer(length.value)
            checked(get_ptx,program,ptx)
        finally:
            checked(destroy,C.byref(program))
    entries={value.decode("ascii") for value in re.findall(rb"\.entry\s+([A-Za-z_]\w*)\(",ptx.value)}
    if entries.intersection(EXPECTED)!=EXPECTED:
        raise RuntimeError("NVRTC output lacks a supported fused kernel entry")
    return dict(entries=sorted(entries),ptx_bytes=len(ptx.value),ptx_sha256=sha256(ptx.value).hexdigest(),
                sources_sha256={path.name:sha256(path.read_bytes()).hexdigest() for path in files})


def audit(toolkit):
    source=HERE/"fused_nonlinear_fft.cu"
    return dict(status="passed",scope="CPU NVRTC compilation only; no CUDA context, kernel launch or timing",
        option="--gpu-architecture=compute_52",toolchain="CUDA 11.8",
        standalone=compile_ptx(toolkit,[source]),
        galerkin_bundle=compile_ptx(toolkit,[V7/"galerkin_cuda_native.cu",V7/"galerkin_cuda_variable.cu",source]),
        fd_bundle=compile_ptx(toolkit,[V7/"fd_cuda_native.cu",source]))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--toolkit",type=Path,default=Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"))
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():
        cli.error("Preserve earlier compile evidence")
    result=audit(args.toolkit)
    result["audit_source_sha256"]=sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(result,handle,indent=2)
        handle.write("\n")
    print("passed")
