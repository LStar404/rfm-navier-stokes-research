"""Exercise fresh-scope context stack transitions with a fake CUDA ABI."""

import argparse
from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/"rfm_v7"))
from cuda_private_context import SIGNATURES
from cuda_private_context_scope import FreshScopedDriver


OWNED, OTHER, DEEPER=0x1000,0x2000,0x3000


def check(ok,message):
    if not ok:
        raise AssertionError(message)


def simulate(mode,after,expected,terminal,*,initial=(),raw_status=0):
    stack=list(initial)
    events=[]
    labels=[]
    destroys=[]
    def write(pointer,value,ctype):
        C.cast(pointer,C.POINTER(ctype))[0]=value
    def fake(name):
        def run(*args):
            events.append(name)
            if name=="cuCtxGetCurrent":
                write(args[0],stack[-1] if stack else None,C.c_void_p)
            elif name=="cuCtxCreate_v2":
                stack.append(OWNED)
                write(args[0],OWNED,C.c_void_p)
            elif name=="cuCtxDestroy_v2":
                value=args[0].value
                destroys.append(value)
                if stack and stack[-1]==value:
                    stack.pop()
            elif name=="cuCtxPopCurrent_v2":
                write(args[0],stack.pop(),C.c_void_p)
            elif name=="cuCtxPushCurrent_v2":
                stack.append(args[0].value)
            elif name=="cuStreamCreate":
                write(args[0],0x4321,C.c_void_p)
            return 0
        run.__name__=name
        return run
    library=type("FakeCudaLibrary",(),{})()
    for name in (*SIGNATURES,"cuCtxPopCurrent_v2","cuCtxPushCurrent_v2"):
        setattr(library,name,fake(name))
    def marker(name,**details):
        if name=="scoped_cufft_context_restored":
            labels.append(details["action"])
    def raw_destroy(plan):
        check(stack==[OWNED] and plan==17,"Plan must enter under the owned context")
        stack[:]=after
        return raw_status

    def require_guard_failure(guarded):
        try:
            guarded(17)
        except RuntimeError as error:
            check("unregistered context stack" in str(error),"Wrong guard failure")
        else:
            raise AssertionError("Guard should reject an unexpected deeper stack")

    error_type=None
    try:
        with ExitStack() as resources:
            scope=FreshScopedDriver(library,resources,mark=marker)
            guarded=scope.guard_plan_destroy(raw_destroy)
            if mode=="unexpected_deeper":
                require_guard_failure(guarded)
            else:
                check(guarded(17)==raw_status,"Unexpected cuFFT return code")
    except RuntimeError as error:
        error_type=str(error)
    check(labels==expected and stack==terminal and all(value==OWNED for value in destroys),"Scope transition differs")
    if initial:
        check(error_type is not None and "fresh unbound host thread" in error_type and
              "cuCtxCreate_v2" not in events and not destroys,"Nested entry must fail before acquiring a context")
    else:
        check(destroys==[OWNED],"Owned context must have exactly one destroy")
        check(events.count("cuStreamDestroy_v2")==int(mode!="unexpected_deeper"),
              "Stream destruction must run only under its owned context")
        check((error_type is not None)==(mode in ("unexpected_deeper","duplicate_owned")),
              "Scope did not reject an unsupported final stack")
    return dict(mode=mode,events=events,restore_actions=labels,error=error_type)


def run(output):
    cases=[
        simulate("unchanged",[OWNED],["owned_unchanged"],[]),
        simulate("pushed_extra",[OWNED,OTHER],["detached_extra_top"],[]),
        simulate("replaced_owned",[OTHER],["detached_replacement_and_pushed_owned"],[]),
        simulate("popped_owned",[],["push_owned_from_empty"],[]),
        simulate("unexpected_deeper",[DEEPER,OTHER],[],[DEEPER,OTHER]),
        simulate("duplicate_owned",[OWNED,OWNED],["owned_unchanged"],[OWNED]),
        simulate("nested_entry",[],[],[DEEPER,OTHER],initial=(DEEPER,OTHER)),
        simulate("cufft_error",[OWNED],[],[],raw_status=7),
    ]
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        inputs={name:sha256((HERE/name).read_bytes()).hexdigest() for name in
                ("verify_cuda_private_scope.py","cuda_private_context_scope.py","cuda_private_context.py")}
        json.dump(dict(status="passed",scope="CPU fake context ABI only",cases=cases,source_sha256=inputs),
                  stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",required=True,type=Path)
    run(cli.parse_args().output)
