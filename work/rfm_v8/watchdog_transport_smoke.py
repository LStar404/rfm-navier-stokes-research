"""Exercise a deadline-bound Windows owner with ordinary child processes only."""

import argparse
import ctypes as C
from ctypes import wintypes as W
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter, sleep


HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"rfm_v7"))
from prepared_process_identity import identity_from_handle


MODES = ("sleep","hang","pipe","before_ready")
OWNER_CPU,WORKER_CPU = 6,4
TIMEOUT,GRACE = 1.2,.7
TIMEOUT_EXIT = 0xE1A9
WAIT_SIGNALED = 0
WAIT_TIMEOUT = 258
OUTPUT = HERE/"watchdog_transport_smoke.json"
KERNEL = C.WinDLL("kernel32",use_last_error=True)


class JobBasic(C.Structure):
    _fields_ = [("process_limit",C.c_longlong),("job_limit",C.c_longlong),("flags",W.DWORD),
                ("min_working_set",C.c_size_t),("max_working_set",C.c_size_t),("processes",W.DWORD),
                ("affinity",C.c_size_t),("priority_class",W.DWORD),("scheduling_class",W.DWORD)]


class JobExtended(C.Structure):
    _fields_ = [("basic",JobBasic),("io",C.c_uint64*6),("process_memory",C.c_size_t),("job_memory",C.c_size_t),
                ("peak_process_memory",C.c_size_t),("peak_job_memory",C.c_size_t)]


def api(name,result,*parameters):
    call = getattr(KERNEL,name)
    call.restype,call.argtypes = result,parameters
    return call


def checked(result):
    if not result:
        raise C.WinError(C.get_last_error())
    return result


def pin(cpu):
    current = api("GetCurrentProcess",W.HANDLE)()
    checked(api("SetProcessAffinityMask",W.BOOL,W.HANDLE,C.c_size_t)(current,1<<cpu))
    mask,system = C.c_size_t(),C.c_size_t()
    checked(api("GetProcessAffinityMask",W.BOOL,W.HANDLE,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t))(
        current,C.byref(mask),C.byref(system)))
    if mask.value != 1<<cpu:
        raise RuntimeError("Affinity mask differs")
    return mask.value


def contain_owner():
    """Only this dedicated owner and its descendants join the private job."""
    job = checked(api("CreateJobObjectW",W.HANDLE,W.LPVOID,W.LPCWSTR)(None,None))
    close = api("CloseHandle",W.BOOL,W.HANDLE)
    try:
        limits = JobExtended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        checked(api("SetInformationJobObject",W.BOOL,W.HANDLE,C.c_int,W.LPVOID,W.DWORD)(
            job,9,C.byref(limits),C.sizeof(limits)))
        checked(api("AssignProcessToJobObject",W.BOOL,W.HANDLE,W.HANDLE)(
            job,api("GetCurrentProcess",W.HANDLE)()))
    except Exception:
        close(job)
        raise
    return job


def emit(event,**fields):
    print(json.dumps(dict(event=event,**fields)),flush=True)


def child(mode):
    cpu = pin(WORKER_CPU)
    if mode=="before_ready":
        sleep(20)
        return
    emit("READY",pid=os.getpid(),affinity_mask=cpu)
    if mode=="pipe":
        sleep(20)  # Keep the read handle open without reading any bytes.
        return
    if sys.stdin.buffer.readline()!=b"GO\n":
        raise RuntimeError("Synthetic worker missing GO")
    if mode=="hang":
        while True:
            pass
    sleep(.075)
    emit("RETURN",pid=os.getpid())
    if sys.stdin.buffer.readline()!=b"SAVE\n":
        raise RuntimeError("Synthetic worker missing SAVE")


def supervise(mode):
    owner_mask = pin(OWNER_CPU)
    job = contain_owner()
    timeline = []
    launch_events,markers = Queue(),Queue()
    write_finished = Event()
    wait = api("WaitForSingleObject",W.DWORD,W.HANDLE,W.DWORD)
    terminate = api("TerminateProcess",W.BOOL,W.HANDLE,W.UINT)
    get_code = api("GetExitCodeProcess",W.BOOL,W.HANDLE,C.POINTER(W.DWORD))

    def mark(event,**extra):
        timeline.append(dict(event=event,elapsed_seconds=perf_counter()-started,**extra))

    def get(inbox,deadline):
        try:
            return inbox.get(timeout=max(0.,deadline-perf_counter()))
        except Empty as exc:
            raise TimeoutError("Absolute lifetime deadline") from exc

    def launch():
        try:
            command = [sys.executable,"-B",str(Path(__file__).resolve()),"--child",mode]
            mark("create_entered")
            process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            ident = identity_from_handle(process._handle)
            if ident["pid"]!=process.pid or ident["exit_filetime_100ns"]:
                raise RuntimeError("Invalid owned process birth")
            mark("create_returned",identity=ident)
            launch_events.put((process,ident))
            for line in iter(process.stdout.readline,b""):
                markers.put(json.loads(line))
        except Exception as exc:
            launch_events.put(exc)
            mark("reader_or_create_error",error=type(exc).__name__)

    def write(payload):
        write_finished.clear()
        try:
            mark("pipe_write_entered",bytes=len(payload))
            process.stdin.write(payload)
            process.stdin.flush()
            mark("pipe_write_completed")
        except OSError as exc:
            mark("pipe_write_failed",error=type(exc).__name__)
        finally:
            write_finished.set()

    def signal_by(deadline):
        while perf_counter()<deadline:
            code = wait(process._handle,0)
            if code==WAIT_SIGNALED:
                return True
            if code!=WAIT_TIMEOUT:
                raise RuntimeError("Unexpected process wait result")
            sleep(.005)
        return wait(process._handle,0)==WAIT_SIGNALED

    def kill_exact():
        observed = identity_from_handle(process._handle)
        if (observed["pid"],observed["birth_filetime_100ns"]) != (ident["pid"],ident["birth_filetime_100ns"]):
            mark("terminate_identity_rejected")
            return
        mark("terminate_entered",identity=observed)
        ok = terminate(process._handle,TIMEOUT_EXIT)
        mark("terminate_returned",success=bool(ok),win32_error=None if ok else C.get_last_error())

    def close_pipes():
        mark("pipe_close_entered")
        errors = []
        for handle in (process.stdin,process.stdout):
            try:
                handle.close()
            except OSError as exc:
                errors.append(type(exc).__name__)
        mark("pipe_close_returned",errors=errors,closed=process.stdin.closed and process.stdout.closed)

    process,ident,terminal = None,None,None
    finished = False
    failure = None
    write_finished.set()
    started = perf_counter()
    deadline,hard = started+TIMEOUT,started+TIMEOUT+GRACE
    Thread(target=launch,daemon=True).start()
    try:
        launched = get(launch_events,deadline)
        if isinstance(launched,Exception):
            raise launched
        process,ident = launched
        ready = get(markers,deadline)
        if (ready.get("event"),ready.get("pid"),ready.get("affinity_mask"))!=("READY",ident["pid"],1<<WORKER_CPU):
            raise RuntimeError("Synthetic worker identity or affinity differs")
        mark("ready",pid=ready["pid"],affinity_mask=ready["affinity_mask"])
        payload = b"x"*(1<<20) if mode=="pipe" else b"GO\n"
        writer = Thread(target=write,args=(payload,),daemon=True)
        writer.start()
        if mode!="pipe":
            writer.join(timeout=max(0.,deadline-perf_counter()))
        returned = get(markers,deadline)
        if (returned.get("event"),returned.get("pid"))!=("RETURN",ident["pid"]):
            raise RuntimeError("Synthetic return identity differs")
        mark("return",pid=returned["pid"])
        Thread(target=write,args=(b"SAVE\n",),daemon=True).start()
        finished = signal_by(deadline)
        if not finished:
            raise TimeoutError("Child did not exit by the absolute deadline")
    except Exception as exc:
        failure = type(exc).__name__
        mark("deadline_or_failure",error=failure,pending_pipe_write=not write_finished.is_set())
        if process is not None and wait(process._handle,0)!=WAIT_SIGNALED:
            Thread(target=kill_exact,daemon=True).start()
            finished = signal_by(hard)
    if finished:
        code = W.DWORD()
        checked(get_code(process._handle,C.byref(code)))
        terminal = dict(identity_from_handle(process._handle),exit_code=code.value)
        mark("process_signaled",identity=terminal)
        if write_finished.wait(max(0.,hard-perf_counter())):
            closer = Thread(target=close_pipes,daemon=True)
            closer.start()
            closer.join(timeout=max(0.,hard-perf_counter()))
    else:
        mark("exit_not_proven_by_deadline")
    result = dict(mode=mode,owner_mask=owner_mask,expected_worker_mask=1<<WORKER_CPU,timeout_seconds=TIMEOUT,
                  grace_seconds=GRACE,elapsed_seconds=perf_counter()-started,failure_type=failure,
                  terminal=terminal,events=sorted(timeline,key=lambda r:r["elapsed_seconds"]),
                  dedicated_kill_on_close_job=bool(job))
    print(json.dumps(result),flush=True)
    # Avoid Python finalization of potentially blocked I/O threads. The job
    # handle is closed by process exit, even when no child handle was acquired.
    os._exit(0)


def verify(result):
    mode = result["mode"]
    events = result["events"]
    by_name = {event["event"]:event for event in events}
    terminal = result["terminal"]
    if result["owner_mask"] != 1<<OWNER_CPU or terminal is None or result["elapsed_seconds"] > TIMEOUT+GRACE+.1:
        return False
    required = {"process_signaled","pipe_close_entered","pipe_close_returned"}
    if (not required <= by_name.keys() or not by_name["pipe_close_returned"]["closed"] or
            not by_name["process_signaled"]["elapsed_seconds"] < by_name["pipe_close_entered"]["elapsed_seconds"]):
        return False
    if mode=="sleep":
        return result["failure_type"] is None and terminal["exit_code"]==0 and "terminate_entered" not in by_name
    names = ("deadline_or_failure","terminate_entered","terminate_returned","process_signaled","pipe_close_entered")
    if not set(names) <= by_name.keys():
        return False
    ordered = [by_name[n]["elapsed_seconds"] for n in names]
    return (ordered==sorted(ordered) and result["failure_type"]=="TimeoutError" and
            terminal["exit_code"]==TIMEOUT_EXIT and by_name["terminate_returned"]["success"] and
            (mode!="pipe" or by_name["deadline_or_failure"]["pending_pipe_write"]) and
            (mode!="before_ready" or "ready" not in by_name))


def test():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    results = []
    for mode in MODES:
        run = subprocess.run([sys.executable,"-B",str(Path(__file__)),"--supervise",mode],capture_output=True,
                             text=True,timeout=5,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        item = json.loads(run.stdout)
        item["passed"] = verify(item)
        results.append(item)
        print(mode,item["passed"],round(item["elapsed_seconds"],3),flush=True)
    passed = all(item["passed"] for item in results)
    evidence = dict(status="passed" if passed else "failed",source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    rows=results,scope="Ordinary synthetic Windows children, no GPU or CUDA context. "
                    "The owner records bounded exit uncertainty; this does not guarantee a driver-held process will exit.")
    with OUTPUT.open("x",encoding="utf-8") as stream:
        json.dump(evidence,stream,indent=2,allow_nan=False)
        stream.write("\n")
    return passed


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--child",choices=MODES)
    group.add_argument("--supervise",choices=MODES)
    args = parser.parse_args()
    if args.child:
        child(args.child)
    elif args.supervise:
        supervise(args.supervise)
    else:
        raise SystemExit(0 if test() else 1)
