"""CPU-only evidence audit and exact frozen-v6 Read/atomic-replace regression."""
import ctypes as C
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
V8=HERE.parent
sys.path.insert(0,str(V8))
import formal_timing_coordinator as original
from atomic_publish_observed import save_observed


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def wait_file(path, process):
    until=time.perf_counter()+4
    while not path.exists() and time.perf_counter()<until and process.poll() is None:
        time.sleep(.005)
    assert path.exists(), (path, process.poll())


def error_record(error):
    return dict(type=type(error).__name__,errno=getattr(error,"errno",None),winerror=getattr(error,"winerror",None),
                filename=getattr(error,"filename",None),filename2=getattr(error,"filename2",None))


def run():
    capture=V8/"rank_fair_cache_recovery_formal_v3_output/mid_b1_b5/capture"
    outer=capture.parent/"outer"
    request=read(capture.parent/"watchdog_request.json")
    source=V8/"rank_fair_outer_watchdog.cs"
    frozen=V8/"rank_fair_cache_recovery_watchdog_fixture_v6_output/source_snapshot/rank_fair_outer_watchdog.cs"
    exe=V8/"rank_fair_cache_recovery_watchdog_fixture_v6_output/rank_fair_outer_watchdog.exe"
    row=read(capture/"attempt_011.json")
    terminal=read(capture/"outer_slot_terminal_011.json")
    state=read(capture/"outer_slot_state.json")
    pending=read(capture/"outer_slot_state.json.tmp")
    ledger=read(capture/"identity_ledger.json")
    assert digest(source)==digest(frozen)=="285b5df008d19a2229369abf186f4355770a77f2395fdfb1922e852967b3de39"
    assert str((capture/"outer_slot_state.json").resolve()) not in request["inputs_sha256"]
    assert terminal==pending and digest(capture/"outer_slot_terminal_011.json")==digest(capture/"outer_slot_state.json.tmp")
    assert row["cleanup_failure_type"]=="PermissionError" and row["exit_code"]==0 and row["watchdog"]["terminal_unconfirmed"] is False
    assert state["phase"]=="active" and state["index"]==11 and ledger["children"][11]["status"]=="monitor_registered"
    evidence_files=[source,frozen,exe,V8/"rank_fair_isolation_v2/coordinator_v2.py",V8/"formal_timing_coordinator.py",
                    capture/"attempt_011.json",capture/"outer_slot_terminal_011.json",capture/"outer_slot_state.json.tmp",
                    capture/"outer_slot_state.json",capture/"identity_ledger.json",outer/"watchdog_terminal.json",
                    capture.parent/"watchdog_request.json"]
    before={str(p):digest(p) for p in evidence_files}
    tests=HERE/"fixtures"
    tests.mkdir()
    helper=HERE/"reader_fixture.exe"
    records=[]
    for mode in ("hold_default","hold_share_delete","poll_exact_v6"):
        directory=tests/mode
        directory.mkdir()
        target=directory/"state.json"
        original.save(target,dict(sequence=-1))
        ready,stop,report=(directory/name for name in ("ready.json","stop.flag","reader_report.json"))
        command=[str(helper),mode,str(target),str(ready),str(stop),str(report),str(exe)]
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        failures=[];successes=0
        try:
            wait_file(ready,process)
            loops=1 if mode=="hold_default" else 512
            for index in range(loops):
                try:
                    original.save(target,dict(sequence=index),replace=True)
                    successes+=1
                except Exception as error:
                    failures.append(error_record(error));break
            if mode=="hold_default":
                assert successes==0 and len(failures)==1 and failures[0]["type"]=="PermissionError"
                assert read(target)==dict(sequence=-1) and read(target.with_name("state.json.tmp"))==dict(sequence=0)
                # A separate fresh namespace exercises the future diagnostic-only helper.
                other=directory/"observed.json"
                original.save(other,dict(sequence=-1))
                kernel=C.WinDLL("kernel32",use_last_error=True)
                kernel.CreateFileW.argtypes=(C.c_wchar_p,C.c_uint,C.c_uint,C.c_void_p,C.c_uint,C.c_uint,C.c_void_p)
                kernel.CreateFileW.restype=C.c_void_p
                kernel.CloseHandle.argtypes=(C.c_void_p,)
                handle=kernel.CreateFileW(str(other),0x80000000,1,None,3,0,None)
                assert handle not in (None,C.c_void_p(-1).value)
                try:
                    try:
                        save_observed(other,dict(sequence=0),replace=True,failure_record=directory/"future_failure_record.json")
                        raise AssertionError("Expected denial while target reader omits shared delete")
                    except PermissionError:
                        diagnosis=read(directory/"future_failure_record.json")
                        assert diagnosis["operation"]=="replace_target" and diagnosis["retries"]==0 and diagnosis["winerror"] in (5,32)
                finally:
                    kernel.CloseHandle(handle)
            else:
                assert successes==512 and not failures
        finally:
            stop.write_text("stop",encoding="ascii")
            process.wait(timeout=3)
        result=read(report)
        if mode=="poll_exact_v6":
            assert result["reads"]>0 and result["read_errors"]==0
        records.append(dict(mode=mode,successful_replacements=successes,replace_failures=failures,reader=result))
    assert {str(p):digest(p) for p in evidence_files}==before
    report=dict(status="passed",audit_scope="CPU-only post-failure source/call-path and controlled concurrency checks",
        gpu_contexts_created=0,flow_solves_run=0,original_evidence_unchanged=True,evidence_sha256=before,tests=records,
        failed_operation_scope="Terminal live-state publication after terminal sidecar, before ledger and EXIT; replace is strongly indicated by complete matching .tmp but the original errno/stack were not recorded.",
        v6_slot_reader_share_delete=True,v6_hash_uses_default_share_read=True,
        hash_is_called_on_live_slot_state=False,live_slot_path_present_in_actual_input_guard=False,
        lock_holder_identified=False,root_cause="unknown",
        conclusions=["The v6 SlotGuard has no subsequent hash or default-share open of the live slot-state file.",
            "File.OpenRead can block replacement when held on that target, proved by negative control; that call is not on the v6 live-state path.",
            "The exact frozen v6 Read method and a continuously held share-delete reader permitted all tested atomic replacements.",
            "The recorded failure cannot identify an OS lock owner or attribute the conflict to the observer.",
            "Future proposed instrumentation preserves fail-closed behavior and records operation/errno/winerror/paths without retrying or promoting failure."],
        future_patch_not_integrated=True,no_new_gpu_attempt_authorized=True,
        source_sha256={p.name:digest(p) for p in (Path(__file__),HERE/"reader_fixture.cs",HERE/"atomic_publish_observed.py")})
    original.save(HERE/"audit_summary.json",report)
    print(json.dumps({k:report[k] for k in ("status","gpu_contexts_created","flow_solves_run","hash_is_called_on_live_slot_state","lock_holder_identified","root_cause")}))


if __name__=="__main__":
    run()
