"""CPU-only failure and concurrency tests for explicit PTX artifact reuse."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import cuda_ptx_cache as cache


SOURCE=b'extern "C" __global__ void sample() {}'
PTX=b".version 7.8\n.target sm_52\n.address_size 64\n.visible .entry sample() { ret; }\n"
IDENTITY={"api_version":[11,8],"library_sha256":{"fixture":"a"*64}}
DEPS={"fixture.cu":"b"*64}
SPEC=cache.contract(SOURCE,"fixture.cu",["sample"],IDENTITY,DEPS)


class FakeCompiler:
    def __init__(self,journal):
        self.journal=journal

    def compile(self,source,spec):
        cache.validate_source(source,spec)
        with self.journal.open("ab") as handle:
            handle.write(b"one build\n")
        time.sleep(.12)
        return PTX


def require(ok,message):
    if not ok:
        raise ValueError(message)


def child(root):
    _,_,stats=cache.artifact(root,SOURCE,SPEC,compiler=FakeCompiler(root.parent/"process_builds.txt"))
    print(stats["mode"])


def audit(root):
    root.mkdir()
    compiler=FakeCompiler(root/"thread_builds.txt")
    shared=root/"thread_entries"
    def fetch(index):
        return cache.artifact(shared,SOURCE,SPEC,compiler=compiler)[2]["mode"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        modes=list(pool.map(fetch,range(4)))
    require(modes.count("miss")==1 and modes.count("hit")==3 and
            (root/"thread_builds.txt").read_bytes()==b"one build\n","Thread build was duplicated")

    process_entries=root/"process_entries"
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child",str(process_entries)]
    processes=[subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW) for _ in range(4)]
    process_modes=[]
    for process in processes:
        try:
            stdout,_=process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
            raise
        require(process.returncode==0,"A concurrent artifact reader failed")
        process_modes.append(stdout.decode("ascii").strip())
    require(process_modes.count("miss")==1 and process_modes.count("hit")==3 and
            (root/"process_builds.txt").read_bytes()==b"one build\n","Process build was duplicated")

    key=cache.key_for(SPEC)
    target=process_entries/(key+".rfmptx")
    original=target.read_bytes()
    corrupt=original[:-2]+b"??"
    target.write_bytes(corrupt)
    try:
        cache.artifact(process_entries,SOURCE,SPEC,strict_hit=True)
    except cache.InvalidArtifact:
        pass
    else:
        raise ValueError("Strict cache hit accepted corrupted payload")
    require(target.read_bytes()==corrupt,"Strict cache hit modified the artifact")
    repair=FakeCompiler(root/"repair_builds.txt")
    with patch.object(cache.os,"replace",side_effect=OSError("injected replacement failure")):
        try:
            cache.artifact(process_entries,SOURCE,SPEC,compiler=repair)
        except OSError:
            pass
        else:
            raise ValueError("Atomic replacement failure was swallowed")
    require(target.read_bytes()==corrupt and not tuple(process_entries.glob("*.tmp")),"Failed atomic write published partial bytes")
    _,_,repaired=cache.artifact(process_entries,SOURCE,SPEC,compiler=repair)
    require(repaired["mode"]=="repaired_invalid" and target.read_bytes()==original,"Exact repair did not restore the artifact")

    code="""import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from cuda_ptx_cache import exclusive
with exclusive(Path(sys.argv[2]),2):
    os._exit(17)
"""
    abandoned=subprocess.run([sys.executable,"-B","-c",code,str(Path(__file__).resolve().parent),
                              str(process_entries/(key+".lock"))],timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
    require(abandoned.returncode==17 and cache.artifact(process_entries,SOURCE,SPEC,strict_hit=True,lock_timeout=2)[2]["mode"]=="hit",
            "OS lock was not released after abrupt owner exit")
    changed=[
        cache.contract(SOURCE+b"\n","fixture.cu",["sample"],IDENTITY,DEPS),
        cache.contract(SOURCE,"fixture.cu",["sample"],{"api_version":[12,0]},DEPS),
        cache.contract(SOURCE,"fixture.cu",["sample"],IDENTITY,{"fixture.cu":"c"*64}),
        cache.contract(SOURCE,"fixture.cu",["sample"],IDENTITY,DEPS,("--gpu-architecture=compute_70",)),
        cache.contract(SOURCE,"fixture.cu",["sample"],IDENTITY,DEPS,(*cache.FLAGS,"--fmad=false")),
    ]
    require(len({key,*map(cache.key_for,changed)})==6,"A relevant compile input failed to invalidate the key")
    return dict(status="passed",scope="Synthetic CPU artifact transactions, no NVRTC or GPU",
                one_compile_for_four_threads=True,one_compile_for_four_processes=True,
                strict_corruption_rejected=True,atomic_replace_failure_preserves_old_bytes=True,
                exact_repair_after_corruption=True,dead_owner_releases_os_lock=True,compile_input_invalidation=True,
                source_sha256={path.name:cache.file_sha(path) for path in (Path(__file__),Path(cache.__file__))})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    choice=cli.add_mutually_exclusive_group(required=True)
    choice.add_argument("--output",type=Path)
    choice.add_argument("--child",type=Path,help=argparse.SUPPRESS)
    args=cli.parse_args()
    if args.child:
        child(args.child)
    else:
        result=audit(args.output)
        with (args.output/"certificate.json").open("x",encoding="utf-8",newline="\n") as handle:
            json.dump(result,handle,indent=2)
            handle.write("\n")
        print("passed")
