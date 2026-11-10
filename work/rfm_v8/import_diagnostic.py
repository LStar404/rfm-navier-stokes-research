"""Read-only import profiling; diagnostic clocks include process launch and return."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "import_diagnostic_output"
MODULES = {"empty": None, "formal": "prepared_formal_timing", "uniform": "galerkin_cuda_native",
           "prepared": "galerkin_cuda_prepared", "fd": "fd_cuda_native"}


def worker(label):
    sys.path.insert(0, str(ROOT.parent / "rfm_v7"))
    start = time.perf_counter()
    if MODULES[label]:
        __import__(MODULES[label])
    elapsed = time.perf_counter() - start
    print(json.dumps({"label": label, "import_seconds": elapsed,
                      "loaded_modules": sorted(sys.modules)}), flush=True)


def main():
    if OUT.exists():
        raise RuntimeError("Preserve every existing attempt")
    OUT.mkdir()
    rng = random.Random(813719)
    jobs = []
    for block in range(3):
        labels = list(MODULES)
        rng.shuffle(labels)
        jobs.extend((block, label) for label in labels)
    protocol = {"scope": "Import-only diagnostic, not isolated performance evidence", "order_seed": 813719,
                "jobs": jobs, "module_map": MODULES,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "clock": "Before Popen through process termination, includes stdout and importtime stderr transfer",
                "cuda_context": "No solver invoked; imported modules only", "blas_threads": 1}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf8")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    for block, label in jobs:
        before = time.perf_counter()
        p = subprocess.run([sys.executable, "-B", "-X", "importtime", str(Path(__file__).resolve()),
                            "--worker", label], capture_output=True, env=env, timeout=60)
        elapsed = time.perf_counter() - before
        (OUT / f"{block}_{label}.importtime.txt").write_bytes(p.stderr)
        if p.returncode:
            raise RuntimeError(f"Worker failed: {block} {label}")
        row = dict(json.loads(p.stdout), block=block, fresh_seconds=elapsed)
        rows.append(row)
        print(block, label, round(row["import_seconds"], 4), round(elapsed, 4), flush=True)
        (OUT / "runs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
