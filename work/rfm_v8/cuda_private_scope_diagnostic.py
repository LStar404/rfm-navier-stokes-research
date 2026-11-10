"""Qualify the guarded private-context cuFFT lifecycle as a separate cohort."""

import os
import sys

if "--health-worker" in sys.argv or "--private-worker" in sys.argv:
    os.write(1,b"RFM_SCOPE_BOOTSTRAP\n")

import argparse
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType

import cuda_private_context_diagnostic as original
import cuda_ptx_phase_t9 as phased
from cuda_ptx_phase_diagnostic import observed_cuda


HERE=Path(__file__).resolve().parent
PROBE=HERE/"cuda_private_context_teardown_probe_output"
FAILED=HERE/"cuda_private_context_health_output"
FIXTURE=HERE/"cuda_private_scope_fixture_v2.json"
MODE="private_cuCtxCreate_v2_scoped_cufft"
RESTORE_ACTIONS={
    "owned_unchanged","push_owned_from_empty","detached_extra_top","detached_replacement_and_pushed_owned",
}


def adapt(function,**updates):
    copy=FunctionType(function.__code__,dict(function.__globals__,**updates),
                      function.__name__,function.__defaults__,function.__closure__)
    copy.__kwdefaults__=function.__kwdefaults__
    return copy


def scoped_plan_stack(get_driver):
    class ScopedPlanStack(ExitStack):
        def callback(self,callback,*args,**kwargs):
            if callback.__name__=="call" and args and args[0].__name__=="cufftDestroy":
                args=(get_driver().guard_plan_destroy(args[0]),*args[1:])
            return super().callback(callback,*args,**kwargs)

    return ScopedPlanStack


def health_worker(root,index):
    phased.mark("private_scoped_python_entry")
    owned={}
    def imports():
        base,_,signatures=original.imports()
        from cuda_private_context_scope import FreshScopedDriver

        class Driver(FreshScopedDriver):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                owned["driver"]=self
        return base,Driver,signatures

    run=adapt(original.health_worker,imports=imports,MODE=MODE,ExitStack=scoped_plan_stack(lambda: owned["driver"]))
    return run(root,index)


def private_worker(root,index):
    phased.mark("private_scoped_python_entry")
    def observed():
        original.imports()
        from cuda_private_context_scope import FreshScopedDriver

        native,_,loader,solver=observed_cuda()
        class Driver(FreshScopedDriver):
            def __init__(self,*args):
                super().__init__(*args,mark=phased.mark)
        return native,Driver,loader,solver

    base=adapt(phased.observed_solver,observed_cuda=observed)
    def solver(arm,policy,settings):
        unguarded,created=base(arm,policy,settings)
        return adapt(unguarded,ExitStack=scoped_plan_stack(lambda: created[0].driver)),created

    return adapt(phased.worker,observed_solver=solver)(root,index)


def binding_files(base):
    paths=(Path(__file__).resolve(),HERE/"cuda_private_context_scope.py",HERE/"verify_cuda_private_scope.py",FIXTURE,
           PROBE/"protocol.json",PROBE/"summary.json",PROBE/"health_00"/"phase_log.json",
           FAILED/"protocol.json",FAILED/"summary.json")
    values=original.source_hashes(base)
    values.update({path.relative_to(original.WORK).as_posix():base.sha(path) for path in paths})
    return values


def static_evidence(base,driver_type,signatures):
    cpu=base.read(FIXTURE)
    prior=base.read(PROBE/"summary.json")
    observed=prior["context_changes"]
    original.require(cpu["status"]=="passed" and len(cpu["cases"])==8 and len(observed)==1 and
                     all(base.sha(HERE/name)==value for name,value in cpu["source_sha256"].items()) and
                     observed[0]["callback"]=="cufftDestroy_2" and observed[0]["before"]=="owned" and
                     observed[0]["after"].startswith("other_") and observed[0]["return_code"]==0 and
                     prior["protocol_sha256"]==base.sha(PROBE/"protocol.json") and
                     prior["watchdog"]["phase_log_sha256"]==base.sha(PROBE/"health_00"/"phase_log.json"),
                     "Scoped cleanup lacks the frozen callback diagnosis or CPU fixture")
    result=original.static_audit(base,driver_type,signatures)
    result.update(scope_fixture_sha256=base.sha(FIXTURE),callback_probe_sha256=base.sha(PROBE/"summary.json"),
                  scope_contract="NULL entry only, one exact detach, known-owned restore, verified NULL exit")
    return result


def read_phases(base,output,watchdog,*,tag):
    watchdog=original.read_phases(base,output,watchdog,tag=tag)
    phase_path=output/"phase_log.json"
    evidence=base.read(phase_path)
    markers=evidence["markers"]
    actions=[row["action"] for row in markers if row["phase"]=="scoped_cufft_context_restored"]
    engine=[row["engine"] for row in markers if row["phase"]=="phase_binding_begin"]
    expected=0 if engine==["warp_fused"] else 2
    cleaned=[row["prior_context_restored"] for row in markers if row["phase"]=="private_context_destroy_end"]
    if watchdog["status"]=="complete" and (len(actions)!=expected or
            any(action not in RESTORE_ACTIONS for action in actions) or cleaned!=[True]):
        watchdog=dict(watchdog,status="phase_incomplete")
    return dict(watchdog,scoped_cufft_recovery_actions=actions,expected_cufft_teardown_callbacks=expected)


def health(root):
    run=adapt(original.health,__file__=__file__,MODE=MODE,source_hashes=binding_files,
              static_audit=static_evidence,read_phases=read_phases)
    return run(root)


def qualify(root,health_root):
    base,_,_=original.setup_parent()
    summary=base.read(health_root/"summary.json")
    rows=summary["rows"]
    original.require(summary["status"]=="passed" and len(rows)==4,"All four health workers must complete")
    actions=[]
    for index,row in enumerate(rows):
        log=health_root/f"health_{index:02d}"/"phase_log.json"
        original.require(base.sha(log)==row["watchdog"]["phase_log_sha256"],"Health phases no longer match")
        actions.extend(row["watchdog"]["scoped_cufft_recovery_actions"])
    original.require(any(action!="owned_unchanged" for action in actions) and len(actions)==8,
                     "The health check did not exercise the observed cuFFT context transition")
    run=adapt(original.qualify,__file__=__file__,MODE=MODE,source_hashes=binding_files,read_phases=read_phases)
    return run(root,health_root)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--health",action="store_true")
    action.add_argument("--qualify",action="store_true")
    action.add_argument("--health-worker",type=int,help=argparse.SUPPRESS)
    action.add_argument("--private-worker",type=int,help=argparse.SUPPRESS)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--health-root",type=Path)
    args=cli.parse_args()
    root=args.output.resolve()
    if args.health:
        raise SystemExit(0 if health(root) else 2)
    if args.qualify:
        original.require(args.health_root is not None,"Provide the guarded four-worker health directory")
        raise SystemExit(0 if qualify(root,args.health_root.resolve()) else 2)
    if args.health_worker is not None:
        health_worker(root,args.health_worker)
    else:
        private_worker(root,args.private_worker)
