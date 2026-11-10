"""Build the three stock PTX modules or measure independent full installations."""

import os
import sys
from time import perf_counter, perf_counter_ns

ENTRY_NS=perf_counter_ns()

import argparse
from hashlib import sha256
import json
from pathlib import Path

from cuda_ptx_phase_diagnostic import affinity


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
V7=WORK/"rfm_v7"
REFERENCE=HERE/"cuda_ptx_artifact_output"/"manifest.json"
STOCK=(("uniform",225),("combined",225),("fd",243))
THREADS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")
PARENT_CPU,CHILD_CPU=20,18
REPEATS=3


def require(ok,message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def install(output,reference,toolkit=None):
    started=perf_counter()
    from build_cuda_ptx_artifacts import TOOLKIT, recipe
    from cuda_ptx_cache import Nvrtc, artifact, file_sha, key_for

    imports_seconds=perf_counter()-started
    require(not output.exists(),"Each stock installation requires a new output directory")
    tick=perf_counter()
    archive=read(reference)
    ref_sha=digest(reference)
    stock=[row for row in archive["modules"] if row["engine"]=="warp_cufft"]
    require(archive["status"]=="passed" and len(stock)==3 and
            [(row["family"],row["ntheta_recipe"]) for row in stock]==list(STOCK),"Reference stock matrix differs")
    reference_read_seconds=perf_counter()-tick
    root=Path(toolkit or TOOLKIT)
    tick=perf_counter()
    compiler=Nvrtc(root)
    require(compiler.identity==archive["compiler"],"Installed compiler does not match the registered stock artifacts")
    compiler_identity_seconds=perf_counter()-tick
    output.mkdir()
    modules=[]
    for row,(family,n) in zip(stock,STOCK,strict=True):
        tick=perf_counter()
        source,spec=recipe(family,"warp_cufft",n,compiler.identity)
        expected=row["manifest"]
        require(spec==expected["contract"] and key_for(spec)==expected["key"],"Current stock translation unit differs")
        recipe_seconds=perf_counter()-tick
        directory=output/"artifacts"
        _,manifest,built=artifact(directory,source,spec,compiler=compiler)
        require(built["mode"]=="miss" and built["nvrtc_seconds"]>0 and manifest==expected,
                "First installation did not compile its registered stock key")
        tick=perf_counter()
        _,checked,hit=artifact(directory,source,spec,strict_hit=True)
        installed=directory/built["artifact"]
        original=reference.parent/"artifacts"/row["cpu_cold_build"]["artifact"]
        new_bytes,old_bytes=installed.read_bytes(),original.read_bytes()
        exact_sha=sha256(new_bytes).hexdigest()
        require(hit["mode"]=="hit" and hit["nvrtc_seconds"]==0 and checked==expected and new_bytes==old_bytes and
                exact_sha==built["artifact_sha256"]==hit["artifact_sha256"]==row["cpu_cold_build"]["artifact_sha256"],
                "Installed stock artifact is not byte-identical to its registration")
        modules.append(dict(family=family,ntheta_recipe=n,manifest=manifest,artifact=installed.name,
            artifact_sha256=exact_sha,bytes=len(new_bytes),recipe_seconds=recipe_seconds,
            cpu_empty_cache=built,cpu_verified_hit=hit,
            byte_comparison_and_recheck_seconds=perf_counter()-tick,original_bytes_equal=True))
        print("stock",family,built["artifact"],f"{built['nvrtc_seconds']:.6f}",flush=True)
    tick=perf_counter()
    for row,(family,n) in zip(modules,STOCK,strict=True):
        _,spec=recipe(family,"warp_cufft",n,compiler.identity)
        require(spec==row["manifest"]["contract"],"Stock compiler inputs changed during installation")
    require(digest(reference)==ref_sha and all(file_sha(root/"bin"/name)==value for name,value in
            compiler.identity["library_sha256"].items()),"Registered manifest or compiler changed during installation")
    closing_input_validation_seconds=perf_counter()-tick
    manifest=dict(status="passed",scope="Three stock NVRTC PTX modules, CPU only",engine="warp_cufft",
                  reference_manifest_sha256=ref_sha,compiler=compiler.identity,modules=modules,
                  source_sha256={name:digest(HERE/name) for name in
                    ("install_stock_ptx.py","build_cuda_ptx_artifacts.py","cuda_ptx_cache.py")},
                  compiler_compiles=3,gpu_contexts_created=0)
    tick=perf_counter()
    path=output/"manifest.json"
    save(path,manifest)
    manifest_sha=digest(path)
    require(read(path)==manifest,"Installed manifest did not round trip")
    return dict(status="passed",manifest_sha256=manifest_sha,phase_seconds=dict(
        install_imports=imports_seconds,reference_read_and_hash=reference_read_seconds,
        compiler_identity=compiler_identity_seconds,closing_input_validation=closing_input_validation_seconds,
        write_hash_verify_manifest=perf_counter()-tick,install_function=perf_counter()-started),
        nvrtc_compile_seconds={row["family"]:row["cpu_empty_cache"]["nvrtc_seconds"] for row in modules},
        artifact_sha256={row["family"]:row["artifact_sha256"] for row in modules},
        total_artifact_bytes=sum(row["bytes"] for row in modules),gpu_contexts_created=0)


def owner_tag(psha,identity,index):
    return f"{psha}:{identity['pid']}:{identity['birth_filetime_100ns']}:{index}"


def bench_worker(root,index,reference,toolkit):
    inherited=affinity()
    affinity(CHILD_CPU)
    sys.path.append(str(V7))
    from prepared_process_identity import identity_from_pid
    from win32_isolation_monitor import WindowsCensus

    psha=digest(root/"protocol.json")
    plan=read(root/"protocol.json")
    require(os.environ.get("RFM_STOCK_INSTALL_OWNER")==owner_tag(psha,identity_from_pid(os.getppid()),index) and
            all(os.environ.get(name)=="1" for name in THREADS) and digest(reference)==plan["reference_manifest_sha256"],
            "Installer worker identity or registered reference differs")
    pre_install_seconds=(perf_counter_ns()-ENTRY_NS)/1e9
    output=root/f"attempt_{index:02d}"
    result=install(output/"installation",reference,toolkit)
    census=WindowsCensus()
    save(output/"result.json",dict(result,index=index,protocol_sha256=psha,affinity=affinity(),
        inherited_affinity=inherited,python_entry_to_install_seconds=pre_install_seconds,
        identity=census.times(census.current_process())))


def measure(root,reference,toolkit):
    affinity(PARENT_CPU)
    for name in THREADS:
        os.environ[name]="1"
    sys.path.append(str(V7))
    from gpu_owned_watchdog import run_owned
    from win32_isolation_monitor import WindowsCensus, process_key

    source_names=("install_stock_ptx.py","cuda_ptx_phase_diagnostic.py","build_cuda_ptx_artifacts.py",
                  "cuda_ptx_cache.py","gpu_owned_watchdog.py","win32_isolation_monitor.py")
    sources={name:digest(HERE/name) for name in source_names}
    require(not root.exists(),"Preserve every previous installation cohort")
    root.mkdir()
    protocol=root/"protocol.json"
    plan=dict(status="frozen",scope="Three CPU-only independent stock installs; no performance isolation claim",
              repeats=REPEATS,deadline_seconds=30,parent_cpu=PARENT_CPU,child_cpu=CHILD_CPU,
              threads={name:"1" for name in THREADS},reference_manifest_sha256=digest(reference),
              stock_families=[family for family,_ in STOCK],source_sha256=sources)
    save(protocol,plan)
    psha=digest(protocol)
    census=WindowsCensus()
    parent=census.times(census.current_process())
    rows=[]
    for index in range(REPEATS):
        output=root/f"attempt_{index:02d}"
        os.environ["RFM_STOCK_INSTALL_OWNER"]=owner_tag(psha,parent,index)
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--bench-worker",str(index),
                 "--output",str(root),"--reference",str(reference)]
        if toolkit:
            command.extend(("--toolkit",str(toolkit)))
        dispatch_ns=perf_counter_ns()
        guard=run_owned(command,output,deadline_seconds=plan["deadline_seconds"])
        returned_ns=perf_counter_ns()
        save(output/"watchdog.json",guard)
        result=read(output/"result.json") if guard["status"]=="complete" else None
        if result is not None:
            require(result["protocol_sha256"]==psha and result["index"]==index and result["affinity"]==1<<CHILD_CPU and
                    process_key(result["identity"])==process_key(guard["terminal_identity"]) and
                    guard["terminal_identity"]["exit_filetime_100ns"]>guard["identity"]["birth_filetime_100ns"] and
                    digest(output/"installation"/"manifest.json")==result["manifest_sha256"],
                    "Installer result or exact terminal identity differs")
        row=dict(index=index,watchdog=guard,result=result,popen_through_terminal_seconds=guard["wall_seconds"],
                 dispatch_begin_ns=dispatch_ns,dispatch_return_ns=returned_ns,
                 result_sha256=digest(output/"result.json") if result is not None else None)
        rows.append(row)
        print("installer attempt",index,guard["status"],f"{guard['wall_seconds']:.6f}s",flush=True)
        if result is None:
            break
    require(digest(protocol)==psha and digest(reference)==plan["reference_manifest_sha256"] and
            all(digest(HERE/name)==value for name,value in sources.items()),"Frozen stock installation inputs changed")
    passed=len(rows)==REPEATS and all(row["result"] is not None for row in rows)
    if passed:
        known=rows[0]["result"]["artifact_sha256"]
        require(all(row["result"]["artifact_sha256"]==known and len(known)==3 for row in rows),
                "Independent installer artifacts differ")
    save(root/"summary.json",dict(status="passed" if passed else "stopped",protocol_sha256=psha,rows=rows,
        unattempted=list(range(len(rows),REPEATS)),gpu_contexts_created=0,isolated_speed_claim_supported=False))
    return passed


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--install",action="store_true")
    action.add_argument("--measure",action="store_true")
    action.add_argument("--bench-worker",type=int,help=argparse.SUPPRESS)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--reference",type=Path,default=REFERENCE)
    cli.add_argument("--toolkit",type=Path)
    args=cli.parse_args()
    root,reference=args.output.resolve(),args.reference.resolve()
    toolkit=args.toolkit.resolve() if args.toolkit else None
    if args.install:
        print(json.dumps(install(root,reference,toolkit),allow_nan=False))
    elif args.measure:
        raise SystemExit(0 if measure(root,reference,toolkit) else 2)
    else:
        bench_worker(root,args.bench_worker,reference,toolkit)
