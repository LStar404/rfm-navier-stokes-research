"""Controlled CPU-only replacement matrix; results are observations, not production guarantees."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import formal_timing_coordinator as original

kernel=C.WinDLL("kernel32",use_last_error=True)
kernel.CreateFileW.argtypes=(C.c_wchar_p,C.c_uint,C.c_uint,C.c_void_p,C.c_uint,C.c_uint,C.c_void_p)
kernel.CreateFileW.restype=C.c_void_p
kernel.CloseHandle.argtypes=(C.c_void_p,)
kernel.MoveFileExW.argtypes=(C.c_wchar_p,C.c_wchar_p,C.c_uint)
kernel.MoveFileExW.restype=C.c_int
kernel.ReplaceFileW.argtypes=(C.c_wchar_p,C.c_wchar_p,C.c_wchar_p,C.c_uint,C.c_void_p,C.c_void_p)
kernel.ReplaceFileW.restype=C.c_int


def replace(source,target,api):
    if api=="python_path_replace":
        source.replace(target)
    elif api=="MoveFileExW":
        if not kernel.MoveFileExW(str(source),str(target),1):
            raise C.WinError(C.get_last_error())
    else:
        if not kernel.ReplaceFileW(str(target),str(source),None,0,None,None):
            raise C.WinError(C.get_last_error())


def state(path):
    try:
        return dict(exists=True,content=json.loads(path.read_bytes()))
    except FileNotFoundError:
        return dict(exists=False)
    except Exception as error:
        return dict(exists=path.exists(),read_error=type(error).__name__,winerror=getattr(error,"winerror",None))


def attempt(source,target,api):
    try:
        replace(source,target,api)
        result=dict(success=True)
    except Exception as error:
        result=dict(success=False,type=type(error).__name__,errno=getattr(error,"errno",None),winerror=getattr(error,"winerror",None))
    return dict(result,source_after=state(source),target_after=state(target))


def main():
    output=HERE/"api_matrix_output"
    output.mkdir()
    records=[]
    helper=HERE/"reader_fixture.exe"
    for mode in ("no_reader","native_share_read","native_share_read_write_delete","dotnet_share_read_write_delete"):
        for api in ("python_path_replace","MoveFileExW","ReplaceFileW"):
            folder=output/(mode+"__"+api)
            folder.mkdir()
            target,source=folder/"state.json",folder/"state.json.tmp"
            original.save(target,dict(sequence=-1));original.save(source,dict(sequence=0))
            handle,process=None,None
            if mode.startswith("native"):
                share=1 if mode=="native_share_read" else 7
                handle=kernel.CreateFileW(str(target),0x80000000,share,None,3,0,None)
                assert handle not in (None,C.c_void_p(-1).value)
            elif mode.startswith("dotnet"):
                ready,stop,report=(folder/name for name in ("ready.json","stop.flag","reader_report.json"))
                process=subprocess.Popen([str(helper),"hold_share_delete",str(target),str(ready),str(stop),str(report)],
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                until=time.perf_counter()+3
                while not ready.exists() and time.perf_counter()<until and process.poll() is None:
                    time.sleep(.005)
                assert ready.exists()
            try:
                result=dict(mode=mode,api=api,**attempt(source,target,api))
                original.save(folder/"result.json",result)
                records.append(result)
            finally:
                if handle is not None:kernel.CloseHandle(handle)
                if process is not None:
                    stop.write_text("stop",encoding="ascii");process.wait(timeout=3)
    # Exercise the exact frozen private Read implementation, not an imitation.
    exe=HERE.parent/"rank_fair_cache_recovery_watchdog_fixture_v6_output/rank_fair_outer_watchdog.exe"
    for api in ("python_path_replace","ReplaceFileW"):
        folder=output/("exact_v6_poll__"+api)
        folder.mkdir();target=folder/"state.json";original.save(target,dict(sequence=-1))
        ready,stop,report=(folder/name for name in ("ready.json","stop.flag","reader_report.json"))
        process=subprocess.Popen([str(helper),"poll_exact_v6",str(target),str(ready),str(stop),str(report),str(exe)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        until=time.perf_counter()+3
        while not ready.exists() and time.perf_counter()<until and process.poll() is None:time.sleep(.005)
        assert ready.exists()
        successes=0;failure=None
        try:
            for i in range(512):
                source=target.with_name("state.json.tmp")
                original.save(source,dict(sequence=i))
                result=attempt(source,target,api)
                if not result["success"]:
                    failure=dict(index=i,**result);break
                successes+=1
        finally:
            stop.write_text("stop",encoding="ascii");process.wait(timeout=3)
        result=dict(mode="exact_v6_poll",api=api,successful_replacements=successes,failure=failure,
                    reader=json.loads(report.read_text(encoding="utf-8-sig")))
        original.save(folder/"result.json",result);records.append(result)
    report=dict(status="completed_observation_not_production_approval",gpu_contexts_created=0,flow_solves_run=0,
                records=records,no_retries=True,no_acl_changes=True,
                replacefile_caution="ReplaceFileW can have partially modified failure states and does not offer a generic transaction or supported WRITE_THROUGH flag; these tests do not establish a production replacement.")
    original.save(output/"summary.json",report)
    print(json.dumps(records))


if __name__=="__main__":main()
