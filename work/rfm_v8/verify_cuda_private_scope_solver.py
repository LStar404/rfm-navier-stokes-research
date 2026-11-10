"""Audit the thin stock solver bindings and import closure without CUDA calls."""

import argparse
import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
T9=HERE/"cuda_private_scope_t9_output"
CACHE=HERE/"cuda_ptx_artifact_output"
FORBIDDEN=("cpu_formal","switch_qualification","verify_switch_qualification","legendre_rank_uniform",
           "cuda_ptx_march_diagnostic","cuda_private_scope_diagnostic","fused_march_qualification",
           "thin_t72_qualification","fresh_worker_diagnostic")


def require(ok,message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def normalized_function(path,name):
    tree=ast.parse(path.read_bytes())
    entry=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
    return ast.dump(entry,include_attributes=False)


def changes(original,bound):
    a,b=original.__globals__,bound.__globals__
    require(set(a)==set(b),"Binding added or removed global names")
    return {name for name in a if a[name] is not b[name]}


def audit():
    before=set(sys.modules)
    import cuda_private_scope_solver as solver
    loaded=set(sys.modules)-before
    from build_cuda_ptx_artifacts import recipe
    from cuda_private_context_scope import FreshScopedDriver
    from cuda_ptx_cache import key_for, unpack
    from make_fused_marches import build

    restored,audit=build()
    require(restored.encode("utf-8")== (HERE/"fused_marches.py").read_bytes() and
            audit==read(HERE/"fused_marches_generation.json"),"Reverse restoration of the stock math differs")
    path=Path(inspect.getsourcefile(solver.scoped_plan_stack))
    fresh_guard=normalized_function(path,"scoped_plan_stack")
    require(fresh_guard==normalized_function(HERE/"cuda_private_scope_diagnostic.py","scoped_plan_stack"),
            "Thin plan cleanup adapter differs from the qualified T9 function AST")
    bound_results=[]
    for arm,original in solver.MARCHES.items():
        for policy in solver.POLICIES:
            bound,created=solver.bind_march(arm,policy,{})
            require(bound.__code__ is original.__code__ and bound.__kwdefaults__==original.__kwdefaults__ and
                    bound.__closure__==original.__closure__ and not created and
                    changes(original,bound)=={"FusedMarchRuntime","ExitStack"},
                    f"Thin march changed math or unregistered global bindings for {arm}/{policy}")
            bound_results.append(dict(arm=arm,policy=policy,same_code_object=True,
                                      replaced_globals=["ExitStack","FusedMarchRuntime"]))
    for type_ in (solver.StockScopedRuntime,solver.CachedScopedRuntime):
        old=type_.__bases__[0].__init__
        new=type_.__init__
        require(new.__code__ is old.__code__ and changes(old,new)=={"_OwnedDriver"} and
                new.__globals__["_OwnedDriver"] is solver.RegisteredPrivateDriver and
                solver.RegisteredPrivateDriver.__bases__==(FreshScopedDriver,),"Runtime preparation changed")

    invalid_calls=(
        dict(arm="unknown",policy="disabled"),dict(arm="uniform",policy="unknown"),
        dict(arm="uniform",policy="disabled",engine="warp_fused"),dict(arm="uniform",policy="strict_hit"),
        dict(arm="uniform",policy="disabled",profile_steps=1),
        dict(arm="uniform",policy="empty_first",artifact_directory=HERE,expected_artifact_sha256="x",compiler_identity={}),
    )
    for invalid in invalid_calls:
        try:
            solver.solve_scoped(**invalid)
        except ValueError:
            continue
        raise AssertionError("Unregistered thin solver call was accepted")

    manifest=read(CACHE/"manifest.json")
    controls=[]
    for family,n in (("uniform",225),("combined",243),("fd",243)):
        source,spec=recipe(family,solver.ENGINE,n,manifest["compiler"])
        record=next(row for row in manifest["modules"] if row["manifest"]["key"]==key_for(spec))
        path=CACHE/"artifacts"/record["cpu_cold_build"]["artifact"]
        raw=path.read_bytes()
        ptx,contract=unpack(raw,spec)
        require(record["engine"]==solver.ENGINE and sha256(source).hexdigest()==spec["source_sha256"] and
                contract==record["manifest"] and sha256(raw).hexdigest()==record["cpu_cold_build"]["artifact_sha256"],
                "Registered stock artifact differs")
        controls.append(dict(family=family,key=contract["key"],artifact=path.relative_to(WORK).as_posix(),
                             artifact_sha256=sha256(raw).hexdigest(),ptx_sha256=sha256(ptx).hexdigest()))

    plan=read(T9/"protocol.json")
    previous=read(T9/"summary.json")
    require(previous["status"]=="passed" and len(previous["rows"])==24 and
            all(row["qualified"] for row in previous["rows"]) and
            previous["protocol_sha256"]==digest(T9/"protocol.json") and
            plan["context_mode"]==solver.CONTEXT_MODE and plan["phase_expected_driver_api"]==solver.DRIVER_API and
            plan["sources_sha256"]["rfm_v8/cuda_private_context_scope.py"]==digest(HERE/"cuda_private_context_scope.py"),
            "Scoped CUDA driver does not match the successful short cohort")
    forbidden=[name for name in loaded if name.split(".",1)[0] in FORBIDDEN or name.startswith("rank_fair_")]
    require(not forbidden,"The thin CUDA path imported a forbidden scoring or diagnostic module")
    local={}
    for name in sorted(loaded):
        path=getattr(sys.modules[name],"__file__",None)
        if path is not None and Path(path).resolve().is_relative_to(WORK):
            relative=Path(path).resolve().relative_to(WORK).as_posix()
            local[relative]=digest(WORK/relative)
    local.update({name:digest(WORK/name) for name in (
        "rfm_v7/galerkin_cuda_native.cu","rfm_v7/galerkin_cuda_variable.cu","rfm_v7/fd_cuda_native.cu",
        "rfm_v6/fd_full_band.dll")})
    require("rfm_v8/cuda_private_scope_solver.py" in local and "rfm_v8/fused_marches.py" in local,
            "Run the import-closure verification in a fresh Python process")
    return dict(status="passed",scope="CPU code-binding, ABI closure and artifact checks only",
                thin_source_sha256=digest(HERE/"cuda_private_scope_solver.py"),
                verifier_source_sha256=digest(Path(__file__).resolve()),
                original_march_sha256=audit["target_sha256"],scoped_plan_stack_ast_sha256=sha256(fresh_guard.encode()).hexdigest(),
                context_mode=solver.CONTEXT_MODE,driver_api=solver.DRIVER_API,
                runtime_bindings={"stock":"_OwnedDriver","cache":"_OwnedDriver"},arm_policy_bindings=bound_results,
                invalid_calls_rejected=len(invalid_calls),forbidden_loaded=forbidden,local_dependency_sha256=local,
                numpy_version=sys.modules["numpy"].__version__,scipy_version=sys.modules["scipy"].__version__,
                stock_artifacts=controls,artifact_manifest_sha256=digest(CACHE/"manifest.json"),
                t9_protocol_sha256=digest(T9/"protocol.json"),t9_summary_sha256=digest(T9/"summary.json"),
                gpu_contexts_created=0)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    result=audit()
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print("passed",len(result["arm_policy_bindings"]),len(result["local_dependency_sha256"]),flush=True)
