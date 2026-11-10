"""Registered v8 transport with an independent, active-interval Windows sampler."""

import argparse
from contextlib import suppress
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path, PurePosixPath
import queue
import subprocess
import sys
from threading import Event, Thread
import time

from fresh_worker_diagnostic import pin
from solver_worker import MARKER, THREADS
from win32_isolation_monitor import (BUDGET_INTERVAL_NS, MAX_ANCHOR_LATENESS_NS, LifecycleSampler, WindowsCensus,
                                     evaluate_active_interval, evaluate_quiet_preflight, process_key)


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SCHEMA = "rfm-v8-active-timing-1"
SECOND = 1_000_000_000
REQUIRED_SOURCES = {
    "rfm_v8/formal_timing_coordinator.py", "rfm_v8/solver_worker.py", "rfm_v8/solver_worker_formal.py",
    "rfm_v8/win32_isolation_monitor.py",
    "rfm_v8/fresh_worker_diagnostic.py", "rfm_v7/prepared_process_identity.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def raw_json(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))+"\n").encode("ascii")


def save(path, value, *, replace=False):
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("xb") as handle:
        handle.write(raw_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        temporary.replace(path)
    else:
        temporary.rename(path)


def check_sources(sources):
    require(REQUIRED_SOURCES <= sources.keys(), "The timing protocol omits required sources")
    for name, expected in sources.items():
        relative = PurePosixPath(name)
        path = (WORK/name).resolve()
        require(not relative.is_absolute() and relative.as_posix() == name and ":" not in name and "\\" not in name and
                path.is_relative_to(WORK.resolve()) and digest(path) == expected, "A frozen timing source differs")


def protocol(path, *, transport=False):
    raw = path.read_bytes()
    value = json.loads(raw)
    require(value["schema"] == SCHEMA and value["status"] == "frozen" and
            value["mode"] == ("transport_only" if transport else "formal"), "Wrong registered timing protocol type")
    require(type(value["coordinator_cpu"]) is int and type(value["monitor_cpu"]) is int and
            0 <= min(value["coordinator_cpu"], value["monitor_cpu"]) < max(value["coordinator_cpu"], value["monitor_cpu"]) < 64,
            "Coordinator and monitor require distinct registered logical CPUs")
    require(value["budget_interval_ns"] == BUDGET_INTERVAL_NS and value["max_budget_lateness_ns"] == MAX_ANCHOR_LATENESS_NS and
            value["poll_interval_ns"] in (SECOND//10, SECOND) and
            value["max_poll_gap_ns"] == 5*value["poll_interval_ns"]//2 and
            type(value["maximum_active_ns"]) is int and 0 < value["maximum_active_ns"] <= 3600*SECOND,
            "Timing cadence, deadline or unchanged v7 budget differs")
    require(0 < value["ready_timeout_seconds"] <= 120 and 0 < value["return_timeout_seconds"] <= 120 and
            0 < value["exit_timeout_seconds"] <= 60, "Registered per-slot timeouts exceed their bounds")
    known = [process_key(row) for row in value["known_background"]]
    require(len(known) == len(set(known)) and all(min(key) > 0 for key in known) and
            (transport or len(known) <= 1), "Invalid or widened background allowlist")
    requests = value["requests"]
    require(bool(requests) and value["run_token"], "Register the entire ordered schedule and a run token")
    for index, request in enumerate(requests):
        require(request["index"] == index and request["schema"] == "rfm-v8-worker-1" and
                request["run_token"] == value["run_token"] and request["affinity_mask"] == 1 << value["coordinator_cpu"] and
                request["trajectory"] == f"trajectory_{index:03d}.npz" and
                all(key in request for key in ("name", "seed", "arm", "block", "family", "imports", "config", "solve_args")),
                "A worker request differs from its registered slot")
    if any(request["family"] == "fd" or request["arm"] == "fd" for request in requests):
        require("rfm_v6/fd_full_band.dll" in value["source_hashes"], "The actual FD band DLL must be bound to the protocol")
    check_sources(value["source_hashes"])
    return value, raw


class Inbox:
    def __init__(self, stream, transcript, prefix=b""):
        self.messages = queue.Queue()
        self.unexpected = False
        self.error = None
        self.count = 0
        self.thread = Thread(target=self.read, args=(stream, transcript, prefix), daemon=True)
        self.thread.start()

    def read(self, stream, transcript, prefix):
        try:
            with transcript.open("xb") as handle:
                for line in iter(stream.readline, b""):
                    stamp = time.perf_counter_ns()
                    handle.write(line)
                    if not line.startswith(prefix):
                        self.unexpected = True
                    else:
                        self.count += 1
                        self.messages.put((stamp, json.loads(line[len(prefix):])))
        except Exception as error:
            self.error = type(error).__name__
        finally:
            self.messages.put(None)

    def receive(self, seconds):
        if seconds <= 0:
            raise TimeoutError("Registered deadline expired")
        try:
            item = self.messages.get(timeout=seconds)
        except queue.Empty as error:
            raise TimeoutError("Registered transport event timed out") from error
        require(item is not None, "Owned subprocess closed the marker stream")
        return item

    def close(self):
        self.thread.join(timeout=3)
        require(not self.thread.is_alive() and not self.unexpected and self.error is None, "Worker marker transcript is malformed")


def kill_owned(process):
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


def remaining(deadline, maximum):
    return max(0., min(maximum, (deadline-time.perf_counter_ns())/SECOND))


def send(process, value):
    process.stdin.write(raw_json(value))
    process.stdin.flush()


class MonitorPeer:
    def __init__(self, output, plan, census):
        self.process = None
        self.serial = 0
        self.failed = False
        with (output/"monitor.stderr.log").open("xb") as stderr:
            self.process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--monitor", str(output)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            self.identity = census.times(self.process._handle)
            self.inbox = Inbox(self.process.stdout, output/"monitor.stdout.log")
            self.token = plan["run_token"]
        except BaseException:
            kill_owned(self.process)
            raise

    def command(self, kind, seconds=5, **fields):
        try:
            require(not self.failed, "The monitor transport already failed")
            self.serial += 1
            send(self.process, dict(event=kind, serial=self.serial, run_token=self.token, **fields))
            _, reply = self.inbox.receive(seconds)
            require(reply["event"] == kind+"_ACK" and reply["serial"] == self.serial and reply["run_token"] == self.token,
                    "Monitor did not acknowledge the registered event")
            return reply
        except Exception:
            self.failed = True
            raise

    def close(self):
        with suppress(OSError):
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_owned(self.process)
        try:
            self.inbox.close()
            require(self.process.returncode == 0, "Owned monitor exited unsuccessfully")
        finally:
            self.process.stdout.close()


def launch(request, output, worker, plan, census, peer, ledger, deadline, row):
    index = request["index"]
    request_path = output/f"request_{index:03d}.json"
    save(request_path, request)
    request_hash = digest(request_path)
    row.update(request_sha256=request_hash)
    process, inbox, entry = None, None, None
    try:
        require(remaining(deadline, 1) > 0, "Cohort deadline expired before Popen")
        command = [sys.executable, "-B", str(worker), str(request_path), request_hash]
        with (output/f"worker_{index:03d}.stderr.log").open("xb") as stderr:
            begin = time.perf_counter_ns()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        identity = census.times(process._handle)
        row.update(launch_monotonic_ns=begin, child_identity=identity)
        require(identity["pid"] == process.pid and not identity["exit_filetime_100ns"], "Popen identity was already invalid")
        entry = dict(index=index, identity=identity, launch_monotonic_ns=begin, request_sha256=request_hash, status="registered")
        ledger["children"].append(entry)
        save(output/"identity_ledger.json", ledger, replace=True)
        inbox = Inbox(process.stdout, output/f"worker_{index:03d}.stdout.log", MARKER)
        registered = peer.command("REGISTER", remaining(deadline, 5), index=index, identity=identity, launch_ns=begin)
        row["monitor_registration_qpc_ns"] = registered["ack_qpc_ns"]
        entry.update(status="monitor_registered", monitor_registration_qpc_ns=registered["ack_qpc_ns"])
        save(output/"identity_ledger.json", ledger, replace=True)

        def marker(event, timeout):
            stamp, value = inbox.receive(remaining(deadline, timeout))
            require(value["event"] == event and value["index"] == index and value["run_token"] == request["run_token"] and
                    value["request_sha256"] == request_hash and process_key(value) == process_key(identity) and
                    value["inherited_affinity_mask"] == request["affinity_mask"], "Worker marker differs from its registered identity")
            return stamp, value

        row["ready_received_monotonic_ns"], row["ready"] = marker("READY", plan["ready_timeout_seconds"])
        row["go_sent_monotonic_ns"] = time.perf_counter_ns()
        require(begin <= row["ready_received_monotonic_ns"] <= row["go_sent_monotonic_ns"] and
                begin <= row["monitor_registration_qpc_ns"] <= row["go_sent_monotonic_ns"], "READY or registration chronology differs")
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        returned, returned_marker = marker("RETURN", plan["return_timeout_seconds"])
        row.update(return_received_monotonic_ns=returned, marker=returned_marker)
        require(returned_marker["status"] == "solved" and not (output/request["trajectory"]).exists(),
                "Solver failed or serialized before RETURN")
        fresh = (returned-begin)/SECOND
        solver = returned_marker["solver_seconds"]
        go_to_return = (returned-row["go_sent_monotonic_ns"])/SECOND
        require(0 < solver <= go_to_return <= fresh and solver == returned_marker["solver_core_seconds"]+returned_marker["model_release_seconds"],
                "Dual-clock boundaries differ")
        row.update(fresh_process_return_seconds=fresh, solver_seconds=solver, go_to_return_seconds=go_to_return,
                   fresh_process_overhead_seconds=fresh-solver)
        row["save_sent_monotonic_ns"] = time.perf_counter_ns()
        process.stdin.write(b"SAVE\n")
        process.stdin.flush()
        process.wait(timeout=remaining(deadline, plan["exit_timeout_seconds"]))
        require(process.returncode == 0, "Worker exited unsuccessfully after SAVE")
        row.update(status="complete", artifact=request["trajectory"], trajectory_sha256=digest(output/request["trajectory"]))
    except Exception as error:
        row.update(status="failed", failure_type=type(error).__name__)
    finally:
        if process is not None:
            with suppress(OSError):
                process.stdin.close()
            try:
                try:
                    process.wait(timeout=remaining(deadline, 2))
                except subprocess.TimeoutExpired:
                    kill_owned(process)
                    row.update(status="failed", failure_type="OwnedWorkerTimeout")
                terminal = census.times(process._handle)
                row.update(exit_monotonic_ns=time.perf_counter_ns(), exit_code=process.returncode, terminal_identity=terminal)
                require(terminal["pid"] == process.pid and terminal["exit_filetime_100ns"] > terminal["birth_filetime_100ns"] and
                        (entry is None or process_key(entry["identity"]) == process_key(terminal)) and
                        row.get("save_sent_monotonic_ns", begin) <= row["exit_monotonic_ns"], "Owned terminal identity or chronology differs")
                if entry is not None:
                    entry.update(status="exited", exit_monotonic_ns=row["exit_monotonic_ns"], terminal=terminal)
                    save(output/"identity_ledger.json", ledger, replace=True)
                    try:
                        peer.command("EXIT", 5, index=index, terminal=terminal, exit_ns=row["exit_monotonic_ns"])
                    except Exception as error:
                        row.update(status="failed", monitor_exit_failure_type=type(error).__name__)
            finally:
                if inbox is not None:
                    try:
                        inbox.close()
                        require(inbox.count <= 2 and (row["status"] != "complete" or inbox.count == 2),
                                "Worker transcript has the wrong machine marker count")
                    except Exception as error:
                        row.update(status="failed", transcript_failure_type=type(error).__name__)
                process.stdout.close()


class Poller:
    def __init__(self, output, plan, census):
        self.output, self.plan, self.census = output, plan, census
        self.samples = []
        self.failure = None
        self.stop = Event()
        self.thread = Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        try:
            with (LifecycleSampler(self.census) as sampler, (self.output/"process_samples.jsonl").open("xb") as raw):
                target = time.perf_counter_ns()
                while not self.stop.is_set():
                    row = sampler.sample()
                    row["target_qpc_ns"] = target
                    raw.write(raw_json(row))
                    raw.flush()
                    self.samples.append(row)
                    target += self.plan["poll_interval_ns"]
                    self.stop.wait(max(0, (target-time.perf_counter_ns())/SECOND))
                os.fsync(raw.fileno())
        except Exception as error:
            self.failure = type(error).__name__

    def observe_after(self, qpc_ns, timeout=5):
        deadline = time.perf_counter()+timeout
        while not self.failure and time.perf_counter() < deadline:
            if self.samples and self.samples[-1]["qpc_start_ns"] >= qpc_ns:
                return len(self.samples)-1
            self.stop.wait(.01)
        raise RuntimeError("Required monitor boundary observation is missing")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        require(not self.thread.is_alive() and self.failure is None, "Native polling did not complete")


def gpu_percent():
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, check=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
    return int(result.stdout.splitlines()[0])


def monitor_main(output):
    plan = json.loads((output/"protocol.json").read_bytes())
    pin(plan["monitor_cpu"])
    census = WindowsCensus()
    own = census.times(census.current_process())
    calibration = census.calibrate()
    poller = Poller(output, plan, census)
    known = {process_key(row) for row in plan["known_background"]}
    children, terminals, launch_times = {}, {}, {}
    latest_exit = 0
    serial = 0
    opened, closed, coordinator, preflight, opening_cpu = None, None, None, None, None
    try:
        with (output/"monitor_events.jsonl").open("xb") as raw_events:
            for line in sys.stdin.buffer:
                command = json.loads(line)
                serial += 1
                require(command["serial"] == serial and command["run_token"] == plan["run_token"], "Monitor command stream differs")
                kind = command["event"]
                payload = {}
                if kind == "PREFLIGHT":
                    require(coordinator is None, "The monitor already admitted its coordinator")
                    coordinator = process_key(command["coordinator"])
                    require(coordinator not in known and coordinator != process_key(own), "Coordinator identity is ambiguous")
                    first = poller.observe_after(time.perf_counter_ns())
                    if plan["mode"] == "formal":
                        guards = [first]
                        percentages = []
                        for _ in range(2):
                            guards.append(poller.observe_after(poller.samples[guards[-1]]["qpc_start_ns"]+BUDGET_INTERVAL_NS, 13))
                            percentages.append(gpu_percent())
                        preflight = evaluate_quiet_preflight([poller.samples[a:b+1] for a,b in zip(guards,guards[1:])],
                            percentages, plan["max_poll_gap_ns"], known_background=known | {coordinator}, monitor_key=process_key(own))
                        preflight.update(sample_anchor_indices=guards)
                    else:
                        preflight = dict(passed=True, transport_only=True, sample_anchor_indices=[first])
                    save(output/"preflight.json", preflight)
                    payload.update(passed=preflight["passed"], identity=own)
                elif kind == "OPEN":
                    require(opened is None and preflight is not None and preflight["passed"], "The cohort is not armed")
                    index = poller.observe_after(time.perf_counter_ns())
                    opening_cpu = census.times(census.current_process())
                    opened = time.perf_counter_ns()
                    payload.update(open_qpc_ns=opened, boundary_sample_index=index)
                elif kind == "REGISTER":
                    key = process_key(command["identity"])
                    require(opened is not None and closed is None and command["index"] == len(children) and
                            len(children) == len(terminals) and latest_exit <= command["launch_ns"] and
                            key not in children.values() and key not in known | {coordinator,process_key(own)} and
                            opened <= command["launch_ns"] <= time.perf_counter_ns(), "Worker registration differs from its prospective slot")
                    children[command["index"]] = key
                    launch_times[command["index"]] = command["launch_ns"]
                elif kind == "EXIT":
                    index, terminal = command["index"], command["terminal"]
                    key = process_key(terminal)
                    require(index in children and children[index] == key and key not in terminals and
                            terminal["exit_filetime_100ns"] > key[1] and launch_times[index] <= command["exit_ns"] <= time.perf_counter_ns(),
                            "Terminal identity differs from the registered worker")
                    terminals[key] = terminal
                    latest_exit = command["exit_ns"]
                elif kind == "CLOSE":
                    require(opened is not None and closed is None, "The monitor has no open cohort")
                    closed = command["close_qpc_ns"]
                    require(opened < closed <= time.perf_counter_ns() and latest_exit <= closed, "Closing timestamp is invalid")
                    last = poller.observe_after(closed)
                    poller.close()
                    closing_cpu = census.times(census.current_process())
                    closing_cpu_qpc_ns = time.perf_counter_ns()
                    roles = {key:"cohort_child" for key in children.values()}
                    roles.update({coordinator:"cohort_coordinator",process_key(own):"monitor"})
                    try:
                        decision = evaluate_active_interval(poller.samples[:last+1], opened, closed,
                            plan["maximum_active_ns"], plan["max_poll_gap_ns"], known_background=known,
                            owned_roles=roles, owned_terminal=terminals)
                    except ValueError:
                        decision = dict(status="coverage_incomplete", reason="registered_active_duration_or_ledger",
                                        full_lifecycle_verified=False)
                    if len(children) != len(terminals):
                        decision.update(status="coverage_incomplete", reason="missing_owned_terminal")
                    if any(row.get("parent_pid") != coordinator[0] for sample in poller.samples[:last+1] for row in sample["live"]
                           if process_key(row) in children.values()):
                        decision.update(status="coverage_incomplete", reason="observed_child_parent_differs")
                    save(output/"isolation.json", dict(preflight=preflight, active=decision, calibration=calibration,
                        open_qpc_ns=opened, close_qpc_ns=closed, end_sample_index=last, identity=own,
                        monitor_cpu=dict(opening=opening_cpu, after_post_close_sample=closing_cpu,
                                         post_close_cpu_sample_qpc_ns=closing_cpu_qpc_ns),
                        process_samples_sha256=digest(output/"process_samples.jsonl"), close_raw_hashes=command["raw_hashes"]))
                    payload["isolation_sha256"] = digest(output/"isolation.json")
                else:
                    raise ValueError("Unknown monitor event")
                reply = dict(event=kind+"_ACK", serial=serial, run_token=plan["run_token"], ack_qpc_ns=time.perf_counter_ns(), **payload)
                raw_events.write(raw_json(dict(command=command, reply=reply)))
                raw_events.flush()
                sys.stdout.buffer.write(raw_json(reply))
                sys.stdout.buffer.flush()
                if kind == "CLOSE":
                    return
    finally:
        if poller.thread.is_alive():
            poller.close()


def run(protocol_path, output, *, fake_worker=None):
    plan, protocol_bytes = protocol(protocol_path, transport=fake_worker is not None)
    output.mkdir()
    with (output/"protocol.json").open("xb") as handle:
        handle.write(protocol_bytes)
    worker = HERE/"solver_worker_formal.py" if fake_worker is None else fake_worker.resolve()
    if fake_worker is not None:
        require(plan["source_hashes"].get(worker.relative_to(WORK).as_posix()) == digest(worker), "Unfrozen test worker")
    peer = None
    rows, raw_hashes = [], {}
    opened = None
    summary = dict(status="diagnostic_incomplete", isolated_speed_claim_supported=False,
                   protocol_sha256=sha256(protocol_bytes).hexdigest(), intended_slots=len(plan["requests"]))
    try:
        for variable in THREADS:
            os.environ[variable] = "1"
        pin(plan["coordinator_cpu"])
        census = WindowsCensus()
        coordinator = census.times(census.current_process())
        peer = MonitorPeer(output, plan, census)
        ledger = dict(run_token=plan["run_token"], coordinator=coordinator, monitor=peer.identity, children=[])
        save(output/"identity_ledger.json", ledger)
        pre = peer.command("PREFLIGHT", 40, coordinator=coordinator)
        require(process_key(pre["identity"]) == process_key(peer.identity), "Monitor identity differs from Popen handle")
        if not pre["passed"]:
            summary["status"] = "preflight_rejected"
            return summary
        opening = peer.command("OPEN")
        opened = opening["open_qpc_ns"]
        deadline = opened+plan["maximum_active_ns"]
        for request in plan["requests"]:
            if time.perf_counter_ns() >= deadline or peer.failed:
                break
            index = request["index"]
            row = dict(index=index, status="attempted")
            sidecar = output/f"attempt_{index:03d}.json"
            save(sidecar,row)
            try:
                launch(request, output, worker, plan, census, peer, ledger, deadline, row)
            except Exception as error:
                row.update(status="failed", coordinator_failure_type=type(error).__name__)
            save(sidecar, row, replace=True)
            rows.append(row)
            raw_hashes[sidecar.name] = digest(sidecar)
            for name in (f"request_{index:03d}.json",f"worker_{index:03d}.stdout.log",f"worker_{index:03d}.stderr.log", request["trajectory"]):
                if (output/name).is_file():
                    raw_hashes[name] = digest(output/name)
            save(output/"runs.json", rows, replace=(output/"runs.json").exists())
        for name in ("protocol.json", "identity_ledger.json", "runs.json"):
            if (output/name).is_file():
                raw_hashes[name] = digest(output/name)
        closed = time.perf_counter_ns()
        reply = peer.command("CLOSE", 10, close_qpc_ns=closed, raw_hashes=raw_hashes)
        isolation = json.loads((output/"isolation.json").read_bytes())
        require(reply["isolation_sha256"] == digest(output/"isolation.json"), "Closed isolation record changed")
        summary.update(open_qpc_ns=opened, close_qpc_ns=closed, isolation_sha256=reply["isolation_sha256"],
                       raw_hashes=raw_hashes, active_verdict=isolation["active"]["status"])
        complete = len(rows) == len(plan["requests"]) and all(row["status"] == "complete" for row in rows)
        if plan["mode"] == "transport_only":
            summary["status"] = "transport_complete" if complete else "transport_failed"
        elif complete and isolation["active"]["status"] == "no_sampled_external_activity":
            summary["status"] = "awaiting_independent_numerical_qualification"
        else:
            summary["status"] = "diagnostic_only"
        check_sources(plan["source_hashes"])
        require(digest(protocol_path) == digest(output/"protocol.json") == summary["protocol_sha256"],
                "The frozen protocol changed during execution")
    except Exception as error:
        summary.update(status="diagnostic_incomplete", coordinator_failure_type=type(error).__name__)
    finally:
        if peer is not None:
            try:
                peer.close()
            except Exception as error:
                summary.update(status="diagnostic_incomplete", monitor_close_failure_type=type(error).__name__)
        if "raw_hashes" not in summary:
            names = ("protocol.json", "identity_ledger.json", "runs.json")
            summary["raw_hashes"] = {**raw_hashes, **{name:digest(output/name) for name in names if (output/name).is_file()}}
        monitor_names = ("preflight.json", "process_samples.jsonl", "monitor_events.jsonl", "isolation.json")
        summary["monitor_evidence_sha256"] = {name:digest(output/name) for name in monitor_names if (output/name).is_file()}
        summary.update(attempted_slots=len(rows), successful_slots=sum(row["status"] == "complete" for row in rows),
                       unattempted_slots=list(range(len(rows), len(plan["requests"]))))
        save(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--monitor", type=Path, help=argparse.SUPPRESS)
    cli.add_argument("--protocol", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.monitor:
        monitor_main(args.monitor)
    else:
        if args.protocol is None or args.output is None:
            cli.error("Supply a registered protocol and new output directory")
        print(json.dumps(run(args.protocol, args.output), ensure_ascii=True))
