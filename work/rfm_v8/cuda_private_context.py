"""Own one driver-created CUDA context and one stream on the calling thread."""

import ctypes as C
from threading import get_native_id

from cuda_environment_native import CHAR, DP, I, P, S, U, Driver, bind, call


SIGNATURES={
    "cuInit":[U],
    "cuDriverGetVersion":[C.POINTER(I)],
    "cuDeviceGet":[C.POINTER(I),I],
    "cuCtxGetCurrent":[C.POINTER(P)],
    "cuCtxCreate_v2":[C.POINTER(P),U,I],
    "cuCtxDestroy_v2":[P],
    "cuStreamCreate":[C.POINTER(P),U],
    "cuStreamDestroy_v2":[P],
    "cuStreamSynchronize":[P],
    "cuMemAlloc_v2":[C.POINTER(DP),S],
    "cuMemFree_v2":[DP],
    "cuMemcpyHtoD_v2":[DP,P,S],
    "cuMemcpyDtoH_v2":[P,DP,S],
    "cuModuleLoadData":[C.POINTER(P),P],
    "cuModuleUnload":[P],
    "cuModuleGetFunction":[C.POINTER(P),P,CHAR],
    "cuLaunchKernel":[P,U,U,U,U,U,U,U,P,P,P],
    "cuStreamBeginCapture":[P,I],
    "cuStreamEndCapture":[P,C.POINTER(P)],
    "cuGraphInstantiateWithFlags":[C.POINTER(P),P,C.c_uint64],
    "cuGraphLaunch":[P,P],
    "cuGraphExecDestroy":[P],
    "cuGraphDestroy":[P],
}


def quiet_mark(*args,**kwargs):
    pass


class PrivateDriver(Driver):
    def __init__(self,library,resources,*,mark=quiet_mark):
        self.mark=mark
        self.owner=get_native_id()
        for name,types in SIGNATURES.items():
            setattr(self,name,bind(library,name,types))
        mark("cuInit_begin")
        call(self.cuInit,0)
        mark("cuInit_end")
        version,device=I(),I()
        call(self.cuDriverGetVersion,C.byref(version))
        self.version=version.value
        call(self.cuDeviceGet,C.byref(device),0)
        self.previous=self.current()
        context=P()
        mark("private_context_create_begin",flags=0,prior_context_present=self.previous is not None)
        call(self.cuCtxCreate_v2,C.byref(context),0,device)
        self.context=context
        resources.callback(self.close_context)
        mark("private_context_create_end")
        self.assert_current()
        self.stream=P()
        mark("stream_create_begin")
        call(self.cuStreamCreate,C.byref(self.stream),1)
        resources.callback(call,self.cuStreamDestroy_v2,self.stream)
        mark("stream_create_end")

    def current(self):
        if get_native_id()!=self.owner:
            raise RuntimeError("CUDA context ownership is tied to its native host thread")
        value=P()
        call(self.cuCtxGetCurrent,C.byref(value))
        return value.value

    def assert_current(self):
        if not self.context.value or self.current()!=self.context.value:
            raise RuntimeError("The owned private CUDA context is not current")

    def close_context(self):
        current=self.current()
        self.mark("private_context_destroy_begin",owned_context_current=current==self.context.value)
        call(self.cuCtxDestroy_v2,self.context)
        restored=self.current()==self.previous
        self.mark("private_context_destroy_end",prior_context_restored=restored)
        if current!=self.context.value or not restored:
            raise RuntimeError("Private CUDA context teardown did not restore the observed predecessor")
