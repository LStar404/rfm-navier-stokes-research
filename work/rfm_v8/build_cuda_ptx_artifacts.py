"""Build or strictly verify the matched eight-module PTX prototype on the CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from cuda_ptx_cache import Nvrtc, artifact, canonical, file_sha, key_for
from fused_march_runtime import march_source


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
TOOLKIT=Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
FILENAME="rfm_v8_warp_march.cu"
RECIPES=(
    ("uniform","warp_cufft",225),("combined","warp_cufft",225),("fd","warp_cufft",243),
    ("uniform","warp_fused",225),("uniform","warp_fused",243),
    ("combined","warp_fused",225),("combined","warp_fused",243),("fd","warp_fused",243),
)
METHODS=(
    ("mid","rfm37_k72","uniform",225),("mid","rfm37_k72","combined",225),
    ("mid","leg35_k72_q66","uniform",225),("mid","leg35_k72_q66","combined",225),
    ("high","rfm41_k80","uniform",243),("high","rfm41_k80","combined",243),
    ("high","leg35_k80_q76","uniform",243),("high","leg35_k80_q76","combined",243),
    ("high","leg36_k80_q76","uniform",243),("high","leg36_k80_q76","combined",243),
    ("both","fd176_dense","fd",243),("high","fd184_band","fd",243),
)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def save(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def recipe(family,engine,n,compiler):
    from cuda_ptx_cache import contract

    fused=engine=="warp_fused"
    source,entries,_=march_source(family,n,fused)
    dependencies={"rfm_v8/fused_march_runtime.py","rfm_v8/warp_observe_cuda.py"}
    if family=="fd":
        dependencies.add("rfm_v7/fd_cuda_native.cu")
    else:
        dependencies.add("rfm_v7/galerkin_cuda_native.cu")
        if family=="combined":
            dependencies.update(("rfm_v7/galerkin_cuda_variable.cu","rfm_v7/galerkin_cuda_module_bundle.py"))
    if fused:
        dependencies.add("rfm_v8/fused_nonlinear_fft.cu")
    spec=contract(source,FILENAME,entries,compiler,{name:file_sha(WORK/name) for name in dependencies})
    return source,spec


def make_matrix(compiler):
    plans={key:recipe(*key,compiler) for key in RECIPES}
    mapping=[]
    for scale,name,family,n in METHODS:
        for engine in ("warp_cufft","warp_fused"):
            registered=n if engine=="warp_fused" or family=="fd" else 225
            emitted=recipe(family,engine,n,compiler)
            require(emitted==plans[family,engine,registered],"Method dimensions unexpectedly specialize the module")
            mapping.append(dict(scale=scale,name=name,family=family,engine=engine,ntheta=n,key=key_for(emitted[1])))
    require(len({key_for(spec) for _,spec in plans.values()})==8 and len(mapping)==24,
            "Expected eight reusable modules for 24 method/schedule/engine settings")
    return plans,mapping


def build(output,toolkit):
    require(not output.exists(),"Preserve the prior compiler artifact build")
    start=time.perf_counter()
    compiler=Nvrtc(toolkit)
    compiler_identity_seconds=time.perf_counter()-start
    plans,mapping=make_matrix(compiler.identity)
    output.mkdir()
    root=output/"artifacts"
    rows=[]
    for (family,engine,n),(source,spec) in plans.items():
        _,manifest,cold=artifact(root,source,spec,compiler=compiler)
        _,same,warm=artifact(root,source,spec,strict_hit=True)
        require(cold["mode"]=="miss" and warm["mode"]=="hit" and manifest==same and
                cold["artifact_sha256"]==warm["artifact_sha256"],"Build and immediate verified reuse differ")
        rows.append(dict(family=family,engine=engine,ntheta_recipe=n,manifest=manifest,cpu_cold_build=cold,
                         cpu_verified_hit_same_process=warm))
        print(family,engine,n,round(cold["nvrtc_seconds"],4),flush=True)
    final_plans,final_mapping=make_matrix(Nvrtc(toolkit).identity)
    require(final_plans==plans and final_mapping==mapping,"Compiler inputs changed during artifact generation")
    result=dict(status="passed",scope="CPU NVRTC and verified artifact IO only. No CUDA context, JIT load, solve or speed ranking.",
        compiler=compiler.identity,compiler_identity_seconds=compiler_identity_seconds,modules=rows,method_mapping=mapping,
        source_sha256={path.name:file_sha(path) for path in (Path(__file__),HERE/"cuda_ptx_cache.py")},
        manifest_key=sha256(canonical([row["manifest"] for row in rows])).hexdigest())
    save(output/"manifest.json",result)
    return result


def hits(output):
    before=time.perf_counter()
    archive=json.loads((output/"manifest.json").read_bytes())
    require(archive["status"]=="passed" and all(file_sha(HERE/name)==sha for name,sha in archive["source_sha256"].items()),
            "Artifact build manifest or source changed")
    plans,mapping=make_matrix(archive["compiler"])
    require(mapping==archive["method_mapping"],"Compiler method mapping changed")
    rows=[]
    for row,(key,(source,spec)) in zip(archive["modules"],plans.items(),strict=True):
        require(key==(row["family"],row["engine"],row["ntheta_recipe"]),"Artifact recipe order changed")
        _,manifest,observed=artifact(output/"artifacts",source,spec,strict_hit=True)
        require(manifest==row["manifest"] and observed["artifact_sha256"]==row["cpu_cold_build"]["artifact_sha256"],
                "Strict cache hit differs from the compiler artifact")
        rows.append(dict(family=key[0],engine=key[1],ntheta_recipe=key[2],**observed))
    return dict(status="passed",scope="Fresh CPU process verifies prebuilt PTX artifacts without importing NVRTC",
        manifest_sha256=file_sha(output/"manifest.json"),module_hits=rows,verification_seconds=time.perf_counter()-before,
        nvrtc_compiles=0)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    mode=cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build",type=Path)
    mode.add_argument("--verify-hits",type=Path)
    cli.add_argument("--toolkit",type=Path,default=TOOLKIT)
    cli.add_argument("--record",type=Path)
    args=cli.parse_args()
    if args.build:
        build(args.build,args.toolkit)
    else:
        result=hits(args.verify_hits)
        if args.record:
            save(args.record,result)
        print(json.dumps(result,allow_nan=False))
