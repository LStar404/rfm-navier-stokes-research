"""Run one registered rank-fair cohort with exact-owned, bounded fresh children."""

from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
import time

import formal_timing_coordinator as prior
from rank_fair_owned_transport import ExactOwned, SECOND, key, latch, publish, read, require, require_identity
from rank_fair_solver_worker import SCHEMA as WORKER_SCHEMA


HERE, WORK = prior.HERE, prior.WORK
SCHEMA = "rfm-v8-rank-fair-active-1"
SOLVER_WORKER = HERE/"rank_fair_solver_worker.py"
LATCHES = ("READY", "GO", "RETURN", "SAVE")
RAW_MONITOR = ("preflight.json","process_samples.jsonl","monitor_events.jsonl","isolation.json","monitor.stdout.log","monitor.stderr.log")


def evidence_files(output, request):
    i = request["index"]
    return [output/f"{stem}_{i:03d}.{suffix}" for stem,suffix in (("attempt","json"),("request","json"),
            ("worker","stdout.log"),("worker","stderr.log"))]+[output/request["trajectory"],
            (output/request["trajectory"]).with_suffix(".tmp.npz")]+[path for name in LATCHES for path in
            (latch(output,i,name),latch(output,i,name).with_suffix(".pending"))]


class BoundedMonitor(prior.MonitorPeer):
    def __init__(self, output, plan, census):
        self.census = census
        self.writer = None
        self.closed = False
        super().__init__(output,plan,census)

    def command(self, kind, seconds=5, **fields):
        deadline = time.perf_counter_ns()+round(seconds*SECOND)
        require(not self.failed, "The independent monitor already failed")
        self.serial += 1
        message = dict(event=kind,serial=self.serial,run_token=self.token,**fields)
        errors = []

        def write():
            try:
                prior.send(self.process,message)
            except Exception as error:
                errors.append(error)

        self.writer = Thread(target=write,daemon=True)
        self.writer.start()
        try:
            self.writer.join(prior.remaining(deadline,seconds))
            if self.writer.is_alive():
                raise TimeoutError("Independent monitor did not receive its command")
            if errors:
                raise errors[0]
            _,reply = self.inbox.receive(prior.remaining(deadline,seconds))
            require(reply["event"] == kind+"_ACK" and reply["serial"] == self.serial and reply["run_token"] == self.token,
                    "Independent monitor acknowledged a different event")
            self.closed = kind == "CLOSE"
            return reply
        except Exception:
            self.failed = True
            raise

    def close(self):
        owned = ExactOwned(self.process,self.census)
        watchdog = {}
        if self.closed:
            try:
                owned.wait_until(time.perf_counter_ns()+5*SECOND)
            except TimeoutError:
                self.failed = True
        terminal,code = owned.finish(self.identity,watchdog)
        if self.writer is not None:
            self.writer.join(timeout=3)
            require(not self.writer.is_alive(), "Monitor writer did not close after exact process exit")
        self.inbox.close()
        self.process.stdin.close()
        self.process.stdout.close()
        require(not self.closed or code == 0, "Independent monitor exited unsuccessfully")
        return dict(terminal=terminal,exit_code=code,**watchdog)


def launch(request, output, worker, plan, census, peer, ledger, deadline, row):
    index = request["index"]
    request_path = output/f"request_{index:03d}.json"
    prior.save(request_path,request)
    request_hash = prior.digest(request_path)
    row.update(request_sha256=request_hash,watchdog={})
    target = output/request["trajectory"]
    command = [sys.executable,"-B",str(worker),str(request_path),request_hash]
    with ((output/f"worker_{index:03d}.stdout.log").open("xb") as stdout,
          (output/f"worker_{index:03d}.stderr.log").open("xb") as stderr):
        process,owned,identity,entry = None,None,None,None
        try:
            begin = time.perf_counter_ns()
            require(begin < deadline,"Cohort deadline expired before child launch")
            hard = min(deadline,begin+round(plan["slot_hard_timeout_seconds"]*SECOND))
            row.update(launch_monotonic_ns=begin,hard_deadline_ns=hard)
            process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = ExactOwned(process,census)
            identity = census.times(process._handle)
            require(identity["pid"] == process.pid and not identity["exit_filetime_100ns"],"Popen handle birth was already invalid")
            row["child_identity"] = identity
            entry = dict(index=index,identity=identity,launch_monotonic_ns=begin,request_sha256=request_hash,status="registered")
            ledger["children"].append(entry)
            prior.save(output/"identity_ledger.json",ledger,replace=True)
            reply = peer.command("REGISTER",prior.remaining(hard,5),index=index,identity=identity,launch_ns=begin)
            row["monitor_registration_qpc_ns"] = reply["ack_qpc_ns"]
            entry.update(status="monitor_registered",monitor_registration_qpc_ns=reply["ack_qpc_ns"])
            prior.save(output/"identity_ledger.json",ledger,replace=True)

            def receive(event, start, seconds):
                stamp,value = owned.receive(latch(output,index,event),min(hard,start+round(seconds*SECOND)))
                require_identity(value,event,request,request_hash,identity)
                require(start <= value["published_monotonic_ns"] <= stamp, "Latch time differs from worker chronology")
                return stamp,value

            def send(event, seconds):
                stamp = time.perf_counter_ns()
                require(stamp < hard,"Cohort deadline expired before dispatch")
                value = {field:row["ready"][field] for field in ("pid","birth_filetime_100ns","run_token","index","request_sha256",
                                                               "inherited_affinity_mask")}
                publish(latch(output,index,event),dict(value,event=event,coordinator=ledger["coordinator"],sent_monotonic_ns=stamp))
                row[event.lower()+"_sent_monotonic_ns"] = stamp
                return min(hard,stamp+round(seconds*SECOND))

            row["ready_received_monotonic_ns"],row["ready"] = receive("READY",begin,plan["ready_timeout_seconds"])
            send("GO",plan["return_timeout_seconds"])
            go = row["go_sent_monotonic_ns"]
            returned,marker = receive("RETURN",go,plan["return_timeout_seconds"])
            row.update(return_received_monotonic_ns=returned,marker=marker)
            require(marker["status"] == "solved" and not target.exists() and not target.with_suffix(".tmp.npz").exists(),
                    "Solver failed or began serializing before RETURN")
            solver,fresh,go_wall = marker["solver_seconds"],(returned-begin)/SECOND,(returned-go)/SECOND
            require(0 < solver <= go_wall <= fresh and solver == marker["solver_core_seconds"]+marker["model_release_seconds"] and
                    go <= marker["solve_started_ns"] <= marker["solve_completed_ns"] <= marker["model_release_started_ns"] <=
                    marker["model_released_ns"] <= marker["published_monotonic_ns"],"Solver clock boundaries differ")
            row.update(solver_seconds=solver,fresh_process_return_seconds=fresh,go_to_return_seconds=go_wall,
                       fresh_process_overhead_seconds=fresh-solver)
            row["exit_wait_completed_monotonic_ns"] = owned.wait_until(send("SAVE",plan["exit_timeout_seconds"]))
            require(process.poll() == 0,"Worker failed after SAVE")
            row.update(status="complete",artifact=target.name,trajectory_sha256=prior.digest(target))
        except Exception as error:
            row.update(status="failed",failure_type=type(error).__name__)
        finally:
            if process is not None:
                try:
                    if owned is None:
                        owned = ExactOwned(process,census)
                    terminal,code = owned.finish(identity,row["watchdog"])
                    exited = time.perf_counter_ns()
                    row.update(exit_monotonic_ns=exited,exit_code=code,terminal_identity=terminal)
                    if entry is not None:
                        entry.update(status="exited",exit_monotonic_ns=exited,terminal=terminal)
                        prior.save(output/"identity_ledger.json",ledger,replace=True)
                        peer.command("EXIT",5,index=index,terminal=terminal,exit_ns=exited)
                except Exception as error:
                    row.update(status="failed",cleanup_failure_type=type(error).__name__)
                    row["watchdog"]["terminal_unconfirmed"] = "terminal_identity" not in row


def run(path, output, plan, protocol_bytes, *, worker=SOLVER_WORKER):
    require(path.read_bytes() == protocol_bytes and read(path) == plan,"Frozen protocol changed before run")
    require(plan["schema"] == SCHEMA and plan["status"] == "frozen", "Invalid rank-fair protocol")
    worker_name = worker.relative_to(WORK).as_posix()
    require(plan["source_hashes"].get(worker_name) == prior.digest(worker), "Worker differs from its registered source")
    require(plan["mode"] in ("formal","transport_only") and (plan["mode"] == "formal") == (worker == SOLVER_WORKER),
            "A transport-only worker cannot enter a formal queue")
    prior.check_sources(plan["source_hashes"])
    output.mkdir()
    with (output/"protocol.json").open("xb") as handle:
        handle.write(protocol_bytes)
    summary = dict(status="diagnostic_incomplete",isolated_speed_claim_supported=False,
                   protocol_sha256=sha256(protocol_bytes).hexdigest(),intended_slots=len(plan["requests"]))
    rows,raw_hashes,peer = [],{},None
    try:
        for variable in prior.THREADS:
            os.environ[variable] = "1"
        prior.pin(plan["coordinator_cpu"])
        census = prior.WindowsCensus()
        coordinator = census.times(census.current_process())
        peer = BoundedMonitor(output,plan,census)
        ledger = dict(run_token=plan["run_token"],coordinator=coordinator,monitor=peer.identity,children=[])
        prior.save(output/"identity_ledger.json",ledger)
        pre = peer.command("PREFLIGHT",40,coordinator=coordinator)
        require(key(pre["identity"]) == key(peer.identity),"Independent monitor identity differs")
        if not pre["passed"]:
            summary["status"] = "preflight_rejected"
            return summary
        opened = peer.command("OPEN")["open_qpc_ns"]
        deadline = opened+plan["maximum_active_ns"]
        for request in plan["requests"]:
            if time.perf_counter_ns() >= deadline or peer.failed:
                break
            row = dict(index=request["index"],status="attempted")
            sidecar = output/f"attempt_{request['index']:03d}.json"
            prior.save(sidecar,row)
            try:
                launch(request,output,worker,plan,census,peer,ledger,deadline,row)
            except Exception as error:
                row.update(status="failed",coordinator_failure_type=type(error).__name__)
            prior.save(sidecar,row,replace=True)
            rows.append(row)
            raw_hashes.update({file.name:prior.digest(file) for file in evidence_files(output,request) if file.is_file()})
            prior.save(output/"runs.json",rows,replace=(output/"runs.json").exists())
            if row["status"] != "complete":
                break
        for name in ("protocol.json","identity_ledger.json","runs.json"):
            file = output/name
            if file.is_file():
                raw_hashes[name] = prior.digest(file)
        closed = time.perf_counter_ns()
        reply = peer.command("CLOSE",10,close_qpc_ns=closed,raw_hashes=raw_hashes)
        isolation = read(output/"isolation.json")
        require(reply["isolation_sha256"] == prior.digest(output/"isolation.json"),"Monitor changed after CLOSE")
        summary.update(open_qpc_ns=opened,close_qpc_ns=closed,isolation_sha256=reply["isolation_sha256"],
                       raw_hashes=raw_hashes,active_verdict=isolation["active"]["status"])
        complete = len(rows) == len(plan["requests"]) and all(row["status"] == "complete" for row in rows)
        if plan["mode"] == "transport_only":
            summary["status"] = "transport_complete" if complete else "transport_failed"
        elif complete and summary["active_verdict"] == "no_sampled_external_activity":
            summary["status"] = "awaiting_independent_numerical_qualification"
        else:
            summary["status"] = "diagnostic_only"
        prior.check_sources(plan["source_hashes"])
        require(path.read_bytes() == protocol_bytes == (output/"protocol.json").read_bytes(),"Frozen protocol changed during run")
    except Exception as error:
        summary.update(status="diagnostic_incomplete",coordinator_failure_type=type(error).__name__)
    finally:
        if peer is not None:
            try:
                summary["monitor_terminal"] = peer.close()
            except Exception as error:
                summary.update(status="diagnostic_incomplete",monitor_close_failure_type=type(error).__name__)
        if "raw_hashes" not in summary:
            names = ("protocol.json","identity_ledger.json","runs.json")
            summary["raw_hashes"] = {**raw_hashes,**{name:prior.digest(output/name) for name in names if (output/name).is_file()}}
        summary["monitor_evidence_sha256"] = {name:prior.digest(output/name) for name in RAW_MONITOR if (output/name).is_file()}
        summary.update(attempted_slots=len(rows),successful_slots=sum(row["status"] == "complete" for row in rows),
                       unattempted_slots=list(range(len(rows),len(plan["requests"]))))
        prior.save(output/"summary.json",summary)
    return summary
