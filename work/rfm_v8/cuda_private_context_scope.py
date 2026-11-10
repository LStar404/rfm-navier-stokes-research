"""A bounded context guard for cuFFT cleanup in an initially unbound worker."""

import ctypes as C
from threading import get_native_id

from cuda_environment_native import I, P, bind, call
from cuda_private_context import PrivateDriver, SIGNATURES, quiet_mark


OWNED_CLEANUP=("cuStreamDestroy_v2","cuMemFree_v2","cuModuleUnload","cuGraphExecDestroy","cuGraphDestroy")


class FreshScopedDriver(PrivateDriver):
    def __init__(self,library,resources,*,mark=quiet_mark):
        self.mark=mark
        self.owner=get_native_id()
        signatures=dict(SIGNATURES,cuCtxPopCurrent_v2=[C.POINTER(P)],cuCtxPushCurrent_v2=[P])
        for name,types in signatures.items():
            setattr(self,name,bind(library,name,types))
        mark("cuInit_begin")
        call(self.cuInit,0)
        mark("cuInit_end")
        version,device=I(),I()
        call(self.cuDriverGetVersion,C.byref(version))
        self.version=version.value
        call(self.cuDeviceGet,C.byref(device),0)
        self.previous=self.current()
        if self.previous is not None:
            raise RuntimeError("Scoped private context requires a fresh unbound host thread")
        self.context=P()
        mark("private_context_create_begin",flags=0,prior_context_present=False)
        call(self.cuCtxCreate_v2,C.byref(self.context),0,device)
        resources.callback(self.close_context)
        mark("private_context_create_end")
        self.assert_current()
        for name in OWNED_CLEANUP:
            setattr(self,name,self.guard_driver_cleanup(getattr(self,name)))
        self.stream=P()
        mark("stream_create_begin")
        call(self.cuStreamCreate,C.byref(self.stream),1)
        resources.callback(call,self.cuStreamDestroy_v2,self.stream)
        mark("stream_create_end")

    def guard_driver_cleanup(self,raw):
        def cleanup(*args):
            if self.current()!=self.context.value:
                self.mark("owned_cleanup_skipped",callback=raw.__name__)
                raise RuntimeError(f"{raw.__name__} requires the owned CUDA context")
            status=raw(*args)
            if status==0:
                self.assert_current()
            return status

        cleanup.__name__=raw.__name__
        return cleanup

    def guard_plan_destroy(self,raw):
        def destroy(plan):
            self.assert_current()
            self.mark("scoped_cufft_destroy_begin")
            status=raw(plan)
            self.mark("scoped_cufft_destroy_result",return_code=status)
            if status==0:
                self.restore_owned_after_plan()
            return status

        destroy.__name__=raw.__name__
        return destroy

    def restore_owned_after_plan(self):
        current=self.current()
        owned=self.context.value
        if current==owned:
            action="owned_unchanged"
        elif current is None:
            call(self.cuCtxPushCurrent_v2,self.context)
            action="push_owned_from_empty"
        else:
            popped=P()
            call(self.cuCtxPopCurrent_v2,C.byref(popped))
            if popped.value!=current:
                raise RuntimeError("CUDA context pop did not detach the observed top")
            below=self.current()
            if below==owned:
                action="detached_extra_top"
            elif below is None:
                call(self.cuCtxPushCurrent_v2,self.context)
                action="detached_replacement_and_pushed_owned"
            else:
                # Restore only the observed post-cuFFT top. This is not a
                # recovery of the unknown stack below it.
                call(self.cuCtxPushCurrent_v2,popped)
                if self.current()!=current:
                    raise RuntimeError("CUDA context diagnostic could not restore the observed top")
                raise RuntimeError("cuFFT cleanup exposed an unregistered context stack")
        self.assert_current()
        self.mark("scoped_cufft_context_restored",action=action)
