"""Rerun the exact frozen hardware cohort in a separate output directory."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import cpu_native_joint as original


base=original.base
HERE=original.HERE
OUT=HERE/"cpu_native_joint_repeat_output"
ORIGINAL=original.FROZEN
PROTOCOL=OUT/"protocol.json"
ISOLATION=OUT/"isolation_protocol.json"
EXECUTION=OUT/"execution.json"


def freeze():
    paths=(Path(__file__),HERE/"run_cpu_native_joint_repeat.ps1",HERE/"cpu_native_joint.py")
    manifest=dict(status="frozen",original_protocol_sha256=base.digest(ORIGINAL),
        sources={p.relative_to(base.WORK).as_posix():base.digest(p) for p in paths},
        window_seconds=600,sample_seconds=10,preflight_windows=2,
        thresholds=dict(preflight_gpu_percent=5,background_individual_cpu_seconds=0.5,
                        background_total_cpu_seconds=1.0,unexpected_python_cpu_seconds=0.2,
                        maximum_sample_lateness_seconds=2.0),
        scope="One full-cohort repeat, with fixed-duration external monitoring. All 50 observations "
              "use the original order and fresh in-process solver. No per-arm selection or retries. "
              "Sampling may miss activity between samples and does not identify other GPU contexts.")
    if ISOLATION.exists():
        if base.read(ISOLATION)!=manifest or base.digest(PROTOCOL)!=manifest["original_protocol_sha256"]:
            raise RuntimeError("Repeat isolation protocol or copied original changed")
    else:
        OUT.mkdir(exist_ok=True)
        with PROTOCOL.open("xb") as handle:
            handle.write(ORIGINAL.read_bytes())
        base.save(ISOLATION,manifest)
    return manifest


def execute():
    freeze()
    if EXECUTION.exists() or (OUT/"timing.json").exists():
        raise FileExistsError("Only one complete-cohort repeat is authorized")
    stamp=lambda:datetime.now(timezone.utc).isoformat()
    record=dict(status="running",pid=os.getpid(),start_utc=stamp(),isolation_protocol_sha256=base.digest(ISOLATION))
    base.save(EXECUTION,record)
    original.OUT=OUT
    original.FROZEN=PROTOCOL
    original.TIMING=OUT/"timing.json"
    try:
        args=original.read_frozen()
        original.time_cohort(*args)
        record["status"]="complete"
    except BaseException:
        record["status"]="failed"
        raise
    finally:
        record["end_utc"]=stamp()
        base.save(EXECUTION,record)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-only",action="store_true")
    action.add_argument("--execute",action="store_true")
    args=cli.parse_args()
    if args.freeze_only:
        freeze()
        print("FROZEN REPEAT",base.digest(ISOLATION),flush=True)
    else:
        if any(os.environ.get(k)!="1" for k in original.THREAD_KEYS):
            cli.error("Repeat timing requires one BLAS thread")
        execute()
