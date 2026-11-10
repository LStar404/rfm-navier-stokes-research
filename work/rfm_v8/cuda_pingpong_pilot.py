"""Same-plan, same-evaluation CUDA history buffer pilot without isolation claims."""

import hashlib
import json
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
V7 = HERE.parent / "rfm_v7"
sys.path.insert(0,str(V7))

import cpu_formal as base  # noqa: E402
from galerkin_cuda_prepared import solve_cuda_prepared  # noqa: E402
from galerkin_cuda_pingpong import solve_cuda_pingpong  # noqa: E402


OUT = HERE / "cuda_pingpong_pilot_output"
CHECKS = ("states","times","probe_times","transverse","vorticity")
WORK = ("steps","device_final_step","nonlinear_evaluations","marching_fft_batches","priming_fft_batches",
        "bridge_extra_fft_batches","inverse_field_transforms","forward_field_transforms")


def fingerprint(sol):
    return {k:hashlib.sha256(getattr(sol,k).tobytes()).hexdigest() for k in CHECKS}


def check_parity():
    examples = []
    for split,steps,graph in ((1024,1136,24),(1025,1152,24),(1024,1152,0),(13,19,24),(14,20,7),(1216,9920,24)):
        def phase(begin,count):
            width = min(count,graph)//2*2
            blocks, tail = divmod(count,width) if width else (0,count)
            pattern = [(begin+i)%2 for i in range(width)]
            return pattern*blocks+[(begin+width*blocks+i)%2 for i in range(tail)]

        assignments = phase(1,split-1)+[split%2]+phase(split+1,steps-split-1)
        history = [0,None]
        for n,p in enumerate(assignments,1):
            assert history[1-p] == n-1
            history[p] = n
        assert len(assignments) == steps-1
        examples.append(dict(steps=steps,split=split,graph=graph,assignments_checked=len(assignments)))
    return examples


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing ping-pong diagnostic")
    registered = V7 / "switch_qualification_output" / "protocol.json"
    p = base.read(registered)
    config = base.config(p["specs"]["rfm37_k72"],p["solve_protocol"],7331)[1]
    generation = base.read(HERE/"cuda_pingpong_generation.json")
    assert generation["source_sha256"] == base.digest(V7/"galerkin_cuda_prepared.py")
    assert generation["target_sha256"] == base.digest(HERE/"galerkin_cuda_pingpong.py")
    jobs = [dict(case=name,repeat=i,startup_rate=128,late_rate=rate,switch_time=split,graph_steps=g)
            for name,rate,split,g,repeats in (("equal",128,8.,24,3),("bridge",112,8.,24,3),
                ("odd_equal",128,8.+1/128,24,1),("eager_bridge",112,8.,0,1)) for i in range(repeats)]
    parity = check_parity()
    inputs = [Path(__file__),HERE/"galerkin_cuda_pingpong.py",HERE/"cuda_pingpong_generation.json",registered,
              *(V7/x for x in ("galerkin_cuda_prepared.py","galerkin_cuda_native.py","galerkin_cuda_native.cu",
                 "galerkin_cuda_variable.py","galerkin_cuda_variable.cu","galerkin_cuda_module_bundle.py","galerkin_phase_shared.py"))]
    hashes = {x.relative_to(HERE.parent).as_posix():base.digest(x) for x in inputs}
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="registered_diagnostic",jobs=jobs,config=config,source_hashes=hashes,
        scope="T9 pairs, same in-process imports and new GPU runtime each solve, no external-load isolation",parity_checks=parity))
    rows = []
    for job in jobs:
        split,late = job["switch_time"],job["late_rate"]
        snapshots = sorted({.5,2.,4.,6.,8.,split,split+1/late,split+2/late,8.125,8.5,9.})
        args = dict(config,preparation="combined",startup_rate=128,late_rate=late,switch_time=split,
                    graph_steps=job["graph_steps"],horizon=9.,snapshots=snapshots)
        arms = (("original",solve_cuda_prepared),("parity",solve_cuda_pingpong))
        if job["repeat"] % 2:
            arms = arms[::-1]
        data = {}
        for label,solve in arms:
            sol = solve(**args)
            data[label] = dict(array_fingerprints=fingerprint(sol),metadata=sol.row)
        a,b = data["original"],data["parity"]
        same_arrays = a["array_fingerprints"] == b["array_fingerprints"]
        same_work = all(a["metadata"][k] == b["metadata"][k] for k in WORK)
        q,n = b["metadata"]["q"],b["metadata"]["ntheta"]
        row = dict(job=job,order=[label for label,_ in arms],arms=data,bitwise=same_arrays,same_work=same_work,
                   estimated_copy_bytes_removed_per_evaluation=64*q*(n//2+1))
        rows.append(row)
        base.save(OUT/"runs.json",dict(status="running",rows=rows))
        print(job["case"],job["repeat"],same_arrays,same_work,
              b["metadata"]["march_seconds"]-a["metadata"]["march_seconds"],flush=True)
    base.save(OUT/"runs.json",dict(status="complete",rows=rows))
    assert hashes == {x.relative_to(HERE.parent).as_posix():base.digest(x) for x in inputs}
    groups = {}
    for group in ("equal","bridge"):
        sample = [r for r in rows if r["job"]["case"] == group]
        stages = ("seconds","host_setup_seconds","runtime_init_seconds","upload_seconds","plan_seconds","graph_setup_seconds",
                  "march_seconds","download_seconds")
        groups[group] = {k:dict(samples=(diff:=[r["arms"]["parity"]["metadata"][k]-r["arms"]["original"]["metadata"][k]
                    for r in sample]),median=median(diff)) for k in stages}
    passed = all(r["bitwise"] and r["same_work"] for r in rows)
    base.save(OUT/"summary.json",dict(status="passed" if passed else "failed",rows=len(rows),protocol_sha256=base.digest(OUT/"protocol.json"),
        runs_sha256=base.digest(OUT/"runs.json"),paired_delta_parity_minus_original_seconds=groups,
        scope="Short diagnostic only; no isolation, timing rankings, or full-horizon claims"))
    if not passed:
        raise ArithmeticError("Ping-pong differs in saved observations or FFT work")


if __name__ == "__main__":
    main()
