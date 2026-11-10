"""Trace each original private-context cleanup callback once, without repairing it."""

import os
import sys
from contextlib import ExitStack
import ctypes as C
from pathlib import Path
from types import FunctionType

import cuda_private_context_diagnostic as diagnostic
from cuda_ptx_phase_t9 import mark


HERE=Path(__file__).resolve().parent
FAILED=HERE/"cuda_private_context_health_output"
DEADLINE=20


def worker(root):
    mark("teardown_probe_python_entry")
    owned={}
    labels={}

    def identity():
        driver=owned["driver"]
        value=C.c_void_p()
        status=driver.cuCtxGetCurrent(C.byref(value))
        if status:
            return dict(current_context=None,get_current_status=status)
        pointer=value.value
        if pointer is None:
            label="none"
        elif pointer==driver.context.value:
            label="owned"
        elif pointer==driver.previous:
            label="entry_predecessor"
        else:
            label=labels.setdefault(pointer,f"other_{len(labels)+1}")
        return dict(current_context=label,get_current_status=0)

    def imports():
        base,driver_type,signatures=diagnostic.imports()

        class ProbeDriver(driver_type):
            def __init__(self,*args,**kwargs):
                owned["driver"]=self
                super().__init__(*args,**kwargs)
                original=self.cuCtxDestroy_v2

                def destroy(context):
                    mark("native_context_destroy_before",**identity())
                    status=original(context)
                    mark("native_context_destroy_after",return_code=status,**identity())
                    return status

                destroy.__name__="cuCtxDestroy_v2"
                self.cuCtxDestroy_v2=destroy

        return base,ProbeDriver,signatures

    class TracedStack(ExitStack):
        def __init__(self):
            super().__init__()
            self.counts={}

        def callback(self,callback,*args,**kwargs):
            called_native=callback.__name__=="call" and bool(args)
            raw=args[0] if called_native else None
            name=raw.__name__ if called_native else callback.__name__
            self.counts[name]=self.counts.get(name,0)+1
            callback_id=f"{name}_{self.counts[name]}"

            def execute():
                mark("cleanup_callback_before",callback=callback_id,**identity())
                if called_native:
                    status=raw(*args[1:],**kwargs)
                    mark("cleanup_callback_after",callback=callback_id,return_code=status,**identity())
                    if status:
                        raise RuntimeError(f"{name}: status {status}")
                    return
                try:
                    callback(*args,**kwargs)
                except BaseException as error:
                    mark("cleanup_callback_after",callback=callback_id,return_code=None,
                         python_exception_type=type(error).__name__,**identity())
                    raise
                mark("cleanup_callback_after",callback=callback_id,return_code=None,**identity())

            return super().callback(execute)

    old=diagnostic.health_worker
    probe=FunctionType(old.__code__,dict(old.__globals__,imports=imports,ExitStack=TracedStack),
                       old.__name__,old.__defaults__,old.__closure__)
    return probe(root,0)


def run(root):
    base,_,_=diagnostic.setup_parent()
    previous=base.read(FAILED/"summary.json")
    earlier_protocol=base.read(FAILED/"protocol.json")
    diagnostic.require(previous["status"]=="stopped" and
                       previous["protocol_sha256"]==base.sha(FAILED/"protocol.json") and
                       previous["rows"][0]["watchdog"]["last_phase"]=="private_context_destroy_end",
                       "Frozen private-context cleanup failure differs")
    sources=dict(earlier_protocol["sources_sha256"])
    files=(Path(__file__).resolve(),FAILED/"protocol.json",FAILED/"summary.json",FAILED/"health_00"/"phase_log.json")
    sources.update({path.relative_to(diagnostic.WORK).as_posix():base.sha(path) for path in files})
    diagnostic.require(all(base.sha(diagnostic.WORK/name)==value for name,value in sources.items()),
                       "An original lifecycle source or evidence artifact changed")
    root.mkdir()
    protocol=root/"protocol.json"
    plan=dict(status="frozen",context_mode=diagnostic.MODE,worker_environment=diagnostic.environment(),
              deadline_seconds=DEADLINE,driver_dll_sha256=base.sha(base.DRIVER),sources_sha256=sources,
              scope="One fresh private-context health probe. Preserve original cleanup order and never repair.")
    base.save(protocol,plan)
    psha=base.sha(protocol)
    census=base.WindowsCensus()
    parent=census.times(census.current_process())
    os.environ["RFM_PRIVATE_HEALTH_OWNER"]=base.identity_tag(psha,parent,0)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker","--output",str(root)]
    output=root/"health_00"
    raw_watchdog=base.run_owned(command,output,deadline_seconds=DEADLINE)
    watchdog=diagnostic.read_phases(base,output,raw_watchdog,tag="private_context_teardown_callback_probe")
    base.save(output/"watchdog.json",watchdog)
    phase_log=base.read(output/"phase_log.json")
    before={}
    pairs=[]
    for row in phase_log["markers"]:
        if row["phase"]=="cleanup_callback_before":
            before[row["callback"]]=row
        elif row["phase"]=="cleanup_callback_after":
            prior=before.pop(row["callback"])
            pairs.append(dict(callback=row["callback"],before=prior["current_context"],after=row["current_context"],
                before_get_current_status=prior["get_current_status"],after_get_current_status=row["get_current_status"],
                return_code=row["return_code"],python_exception_type=row.get("python_exception_type")))
    changes=[row for row in pairs if not row["before_get_current_status"] and not row["after_get_current_status"] and
             row["before"]!=row["after"]]
    diagnostic.require(base.sha(protocol)==psha and base.sha(base.DRIVER)==plan["driver_dll_sha256"] and
        all(base.sha(diagnostic.WORK/name)==value for name,value in sources.items()),"Callback probe inputs changed")
    result=dict(status="completed_observation" if pairs else "incomplete_observation",protocol_sha256=psha,watchdog=watchdog,
                cleanup_callbacks=pairs,context_changes=changes,pending_callbacks=list(before),
                evidence_is_lifecycle_qualification=False)
    base.save(root/"summary.json",result)
    for row in pairs:
        print(row["callback"],row["before"],"->",row["after"],"return",row["return_code"],flush=True)
    print("callback probe terminal",watchdog["status"],watchdog["last_phase"],flush=True)
    return bool(pairs) and not before


if __name__=="__main__":
    import argparse

    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    root=args.output.resolve()
    if args.worker:
        worker(root)
    else:
        raise SystemExit(0 if run(root) else 2)
