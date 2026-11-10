"""Generate an FD runtime factory and a shared compiler/FFT loader."""

import ast
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
AUDIT = HERE/"cuda129_fd_source_audit.json"


def substitute(source,changes):
    result = source
    for old,new in changes:
        if result.count(old)!=1:
            raise ValueError(f"Expected one source span: {old}")
        result = result.replace(old,new)
    restored = result
    for old,new in reversed(changes):
        if restored.count(new)!=1:
            raise ValueError(f"Expected one inverse source span: {new}")
        restored = restored.replace(new,old)
    if restored!=source:
        raise ValueError("The source whitelist did not restore the original text")
    return result


def generate():
    original_fd = V7/"fd_cuda_native.py"
    fd = original_fd.read_text(encoding="utf-8")
    fd = substitute(fd,(
        ("runtime=None, toolkit=None, **config):","runtime=None, toolkit=None, runtime_factory=None, **config):"),
        ("        rt = resources.enter_context(FDRuntime(toolkit)) if runtime is None else runtime",
         "        if runtime is not None and runtime_factory is not None:\n"
         "            raise ValueError(\"Choose a runtime object or a factory, not both\")\n"
         "        runtime_type = FDRuntime if runtime_factory is None else runtime_factory\n"
         "        rt = resources.enter_context(runtime_type(toolkit)) if runtime is None else runtime"),
    ))
    original_runtime = HERE/"cuda129_runtime.py"
    runtime = original_runtime.read_text(encoding="utf-8")
    class_node = next(n for n in ast.parse(runtime).body if isinstance(n,ast.ClassDef) and n.name=="PinnedBundledRuntime")
    original_class = "".join(runtime.splitlines(keepends=True)[class_node.lineno-1:class_node.end_lineno])
    class_text = substitute(original_class,(
        ("class PinnedBundledRuntime(BundledSwitchRuntime):","class PinnedCaseRuntime(BundledSwitchRuntime):"),
        ("def __init__(self,toolkit=None,*,variant):","def __init__(self,toolkit=None,*,variant,source,filename,kernel_names):"),
        ("        source = bundled_source()\n","        source = bytes(source)\n"),
        ('            filename = b"galerkin_cuda_module_bundle.cu"\n',"            filename = bytes(filename)\n"),
        ("for name in KERNEL_NAMES:","for name in kernel_names:"),
        ("bundled_source_sha256=sha256(source).hexdigest()","kernel_source_sha256=sha256(source).hexdigest()"),
    ))
    header = '''"""Use exactly the same pinned toolchain loader for Galerkin and FD."""

from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import os
from pathlib import Path
import time

from cuda129_runtime import (
    V7, NVRTC, BUILTINS, CUFFT, NVJIT, VARIANTS, BundledSwitchRuntime, _OwnedDriver, TOOLKIT_DEFAULT,
    I, P, DP, bind, call, nvrtc_load, modern_program, bundled_source, KERNEL_NAMES,
)


'''
    adapters = '''

class GalerkinCaseRuntime(PinnedCaseRuntime):
    def __init__(self, toolkit=None, *, variant):
        super().__init__(toolkit,variant=variant,source=bundled_source(),
                         filename=b"galerkin_cuda_module_bundle.cu",kernel_names=KERNEL_NAMES)


class FdCaseRuntime(PinnedCaseRuntime):
    def __init__(self, toolkit=None, *, variant):
        names = ("fd_fields","fd_products","fd_rhs","fd_band","fd_dense","fd_observe")
        super().__init__(toolkit,variant=variant,source=(V7/"fd_cuda_native.cu").read_bytes(),
                         filename=b"fd_cuda_native.cu",kernel_names=names)
'''
    generated = {"fd_cuda_native_cuda129.py":fd,"cuda129_case_runtime.py":header+class_text+adapters}
    for candidate in generated.values():
        ast.parse(candidate)
    sources = (original_fd,original_runtime,V7/"fd_cuda_native.cu",V7/"galerkin_cuda_native.cu",
               V7/"galerkin_cuda_variable.cu",Path(__file__))
    return generated,dict(status="passed",source_sha256={p.relative_to(HERE.parent).as_posix():
        sha256(p.read_bytes()).hexdigest() for p in sources},
        generated_sha256={name:sha256(text.encode("utf-8")).hexdigest() for name,text in generated.items()},
        scope="FD solve differs only in runtime factory and exclusive runtime-object check. Shared loader has only source, filename, kernel symbols and evidence key parameterized.")


def main():
    generated,audit = generate()
    for name,source in generated.items():
        with (HERE/name).open("xb") as stream:
            stream.write(source.encode("utf-8"))
    with AUDIT.open("x",encoding="utf-8") as stream:
        json.dump(audit,stream,indent=2)
        stream.write("\n")
    print("PASS",len(generated),flush=True)


if __name__=="__main__":
    main()
