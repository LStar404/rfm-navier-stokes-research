"""Matched fresh-worker import ablation. Diagnostic only; no isolation claim."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import queue
import random
from statistics import median
import subprocess
import sys
from threading import Thread
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.append(str(WORK / "rfm_v7"))
from prepared_process_identity import current_process_identity, identity_from_handle
from solver_worker import MARKER, THREADS, affinity_mask, require


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(cpu=4):
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = C.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [C.c_void_p, C.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
        raise C.WinError(C.get_last_error())
    require(affinity_mask() == 1 << cpu, "Coordinator affinity mismatch")


def read_messages(stream, mailbox, path):
    with path.open("xb") as output:
        for line in iter(stream.readline, b""):
            received = time.perf_counter_ns()
            output.write(line)
            if line.startswith(MARKER):
                mailbox.put((received, json.loads(line[len(MARKER):])))
    mailbox.put(None)


def launch(request, output, before_go=None, after_exit=None):
    index = request["index"]
    path = output / f"request_{index:03d}.json"
    require(not path.exists(), "Existing request cannot be overwritten")
    save(path, request)
    sha = digest(path)
    mailbox = queue.Queue()
    row = dict(index=index, request_sha256=sha, status="running")
    with (output / f"worker_{index:03d}.stderr.txt").open("xb") as stderr:
        begin = time.perf_counter_ns()
        child = subprocess.Popen([sys.executable, "-B", str(HERE / "solver_worker.py"), str(path), sha],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        identity = identity_from_handle(child._handle)
        row.update(child_identity=identity, launch_monotonic_ns=begin)
        reader = Thread(target=read_messages, args=(child.stdout, mailbox, output / f"worker_{index:03d}.stdout.txt"), daemon=True)
        reader.start()

        def receive(event):
            result = mailbox.get(timeout=120)
            require(result is not None, "Child exited before marker")
            stamp, marker = result
            require(marker["event"] == event and marker["index"] == index and
                    marker["run_token"] == request["run_token"] and marker["request_sha256"] == sha and
                    marker["pid"] == identity["pid"] and marker["birth_filetime_100ns"] == identity["birth_filetime_100ns"] and
                    marker["inherited_affinity_mask"] == request["affinity_mask"], "Child marker differs from registered identity")
            return stamp, marker

        try:
            row["ready_received_monotonic_ns"], row["ready"] = receive("READY")
            if before_go:
                before_go(identity, row)
            row["go_sent_monotonic_ns"] = time.perf_counter_ns()
            child.stdin.write(b"GO\n")
            child.stdin.flush()
            returned, marker = receive("RETURN")
            row.update(return_received_monotonic_ns=returned, marker=marker)
            require(marker["status"] == "solved", "Solver reported a preserved failure")
            require(not (output / request["trajectory"]).exists(), "Serialization occurred before RETURN")
            fresh = (returned - begin) / 1e9
            require(0 < marker["solver_seconds"] <= fresh and
                    marker["solver_seconds"] == marker["solver_core_seconds"] + marker["model_release_seconds"], "Invalid dual clocks")
            row.update(fresh_process_return_seconds=fresh,
                       fresh_process_overhead_seconds=fresh-marker["solver_seconds"],
                       **{key:marker[key] for key in ("solver_seconds", "solver_core_seconds", "model_release_seconds")})
            row["save_sent_monotonic_ns"] = time.perf_counter_ns()
            child.stdin.write(b"SAVE\n")
            child.stdin.flush()
        except Exception as exc:
            row.update(status="failed", failure_type=type(exc).__name__)
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
            row.update(exit_monotonic_ns=time.perf_counter_ns(), exit_code=child.returncode,
                       terminal_identity=identity_from_handle(child._handle))
            reader.join(timeout=5)
            require(not reader.is_alive(), "Reader did not close")
            if after_exit:
                after_exit(row["terminal_identity"], row)
    if row["status"] != "failed" and row["exit_code"] == 0:
        row.update(status="complete", trajectory_sha256=digest(output / request["trajectory"]))
    return row


def make_requests(scale, horizon, repeats, modes=("formal", "selected", "minimal")):
    path = WORK / "rfm_v7" / f"prepared_formal_timing_{scale}_output" / "protocol.json"
    old = json.loads(path.read_text(encoding="utf8"))
    times = [t for t in old["design"]["physical_times"] if t <= horizon]
    rate, late = (old["step_schedule"][k] for k in ("startup_rate", "late_rate"))
    jobs = []
    rng = random.Random(915761)
    for block in range(repeats):
        group = []
        for name, spec in old["specs"].items():
            family = spec["method"]
            config = dict(m=spec["m"], k=spec["k"], q=spec["q"], scale=62., mean_order=24,
                          layers=old["solve_protocol"]["layers"], split=True, backend="fft",
                          seed=7331 if family == "rfm" else 17, family=family)
            for arm in ("uniform", "combined"):
                args = dict(horizon=horizon, snapshots=times, probe_spacing=.125, graph_steps=24, **config)
                args.update(dict(steps=round(horizon*rate)) if arm == "uniform" else
                            dict(startup_rate=rate, late_rate=late, switch_time=8., preparation="combined"))
                group.extend(dict(block=block, name=name, family=family, arm=arm, imports=mode,
                                  config=config, solve_args=args) for mode in modes)
        fd = old["fd_controls"]["fd176_k80"]
        args = dict(horizon=horizon, snapshots=times, probe_spacing=.125, graph_steps=24,
                    steps=round(horizon*fd["rate"]), cuda_linear_solve=fd["linear"], **fd["config"])
        group.extend(dict(block=block, name="fd176_k80", family="fd", arm="fd", imports=mode,
                          config=fd["config"], solve_args=args) for mode in modes)
        rng.shuffle(group)
        jobs.extend(group)
    token = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
    return [dict(job, index=i, schema="rfm-v8-worker-1", run_token=token, affinity_mask=16,
                 trajectory=f"trajectory_{i:03d}.npz") for i, job in enumerate(jobs)]


def summarize(output, protocol, rows):
    import numpy as np
    arrays = ("states", "times", "probe_times", "transverse", "vorticity")
    groups = {}
    for request, row in zip(protocol["requests"], rows):
        require(row["status"] == "complete", "A diagnostic worker failed; retain all evidence")
        key = request["name"] + "/" + request["arm"]
        with np.load(output / request["trajectory"], allow_pickle=False) as raw:
            fingerprint = {name:dict(shape=raw[name].shape, dtype=raw[name].dtype.str,
                                    sha256=hashlib.sha256(raw[name].tobytes()).hexdigest()) for name in arrays}
            metadata = json.loads(str(raw["metadata"]))
        group = groups.setdefault(key, dict(reference=fingerprint, signature=row["marker"]["spatial_sha256"], modes={}))
        require(group["reference"] == fingerprint and group["signature"] == row["marker"]["spatial_sha256"],
                "Import policy changed the trajectory or spatial signature")
        sample = {key:row[key] for key in ("solver_seconds", "fresh_process_return_seconds", "fresh_process_overhead_seconds")}
        sample.update(block=request["block"], phases={k:v for k,v in metadata.items() if k.endswith("_seconds")})
        group["modes"].setdefault(request["imports"], []).append(sample)
    for group in groups.values():
        group["medians"] = {mode:{key:median(r[key] for r in sample) for key in
                            ("solver_seconds", "fresh_process_return_seconds", "fresh_process_overhead_seconds")}
                            for mode, sample in group["modes"].items()}
    save(output / "summary.json", dict(status="complete", scope="Diagnostic only: no isolation monitoring",
        protocol_sha256=digest(output / "protocol.json"), rows_sha256=digest(output / "runs.json"),
        trajectories_bitwise_equal_within_solver=True, groups=groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", choices=("mid", "high"), default="mid")
    parser.add_argument("--horizon", type=int, choices=(9,72), default=9)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous attempts")
    args.output.mkdir()
    for name in THREADS:
        os.environ[name] = "1"
    pin()
    requests = make_requests(args.scale, args.horizon, args.repeats)
    sources = {p.relative_to(WORK).as_posix():digest(p) for folder in ("rfm_v3","rfm_v5","rfm_v6","rfm_v7","rfm_v8")
               for p in (WORK / folder).glob("*") if p.suffix in (".py", ".cu", ".c")}
    protocol = dict(status="registered_diagnostic", scope="Import ablation, no isolation claim", order_seed=915761,
                    scale=args.scale, horizon=args.horizon, requests=requests, source_hashes=sources,
                    coordinator=current_process_identity(), clocks="Solver setup-to-release and Popen-to-RETURN")
    save(args.output / "protocol.json", protocol)
    rows = []
    for request in requests:
        row = launch(request, args.output)
        rows.append(row)
        save(args.output / "runs.json", rows)
        print(request["index"],request["name"],request["arm"],request["imports"],row["status"],
              row.get("fresh_process_return_seconds"),flush=True)
        require(row["status"] == "complete", "Preserved diagnostic failure")
    summarize(args.output, protocol, rows)


if __name__ == "__main__":
    main()
