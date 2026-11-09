"""Single-attempt prepared CUDA timing with exact Windows child identity."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
from threading import Thread
import time

import prepared_formal_eligibility as eligibility
from prepared_process_identity import current_process_identity, identity_from_handle, identity_from_pid
from galerkin_cuda_prepared import solve_cuda_prepared
import switch_formal_timing as previous


base, case, np, switch = previous.base, previous.case, previous.np, previous.switch
require, utc = previous.require, previous.utc
TOKEN = "RFM_PREPARED_RUN_TOKEN"
HASH = "RFM_PREPARED_PROTOCOL_SHA256"
ADMISSION = "RFM_PREPARED_ADMISSION_PATH"
SCHEMA = "win32-process-identity/v1"
CLOCKS = ("solver_seconds", "fresh_process_return_seconds")
MODEL = switch.switched.RecurrenceGalerkin
MARKER = b"PREPARED_TIMING "
IDENTITY = ("pid", "birth_filetime_100ns")


def identity(row):
    return tuple(row[key] for key in IDENTITY)


def role_identity(row):
    return row["parent_pid"], row["parent_birth_filetime_100ns"]


def public_failure(exc):
    message = str(exc)
    roots = (base.WORK.parent,Path.home(),Path(sys.executable).parent)
    for root in sorted(roots,key=lambda p:len(str(p)),reverse=True):
        for value in {str(root),root.as_posix(),str(root).replace("\\","\\\\")}:
            message = message.replace(value,"<local-root>")
    return message


def paths(output):
    return {name:output / (name + ".json") for name in
            ("protocol", "execution", "timing", "isolation", "summary", "identity_ledger", "coordinator_admission")}


def read_protocol(files, scale):
    p = base.read(files["protocol"])
    require(p["status"] == "frozen" and p["scale"] == scale and
            len(p["schedule"]) == {"mid":55,"high":40}[scale], "Formal protocol/scale differs")
    return p, base.digest(files["protocol"])


def configuration(job, p):
    if job["arm"] == "fd":
        return p["fd_controls"][job["name"]]["config"]
    return base.config(p["specs"][job["name"]], p["solve_protocol"], job["seed"])[1]


def solve(job, p, config):
    d = p["design"]
    args = dict(horizon=d["horizon"], snapshots=d["physical_times"], probe_spacing=d["probe_spacing"],
                graph_steps=d["cuda_graph_steps"], **config)
    name, arm = job["name"], job["arm"]
    schedule = p["step_schedule"]
    if arm == "fd":
        fd = p["fd_controls"][name]
        return previous.solve_fd_native(steps=d["horizon"]*fd["rate"], cuda_linear_solve=fd["linear"], **args)
    if arm == "uniform":
        with previous.native_model():
            return previous.solve_native(steps=d["horizon"]*schedule["startup_rate"], **args)
    switched = dict(startup_rate=schedule["startup_rate"], late_rate=schedule["late_rate"], switch_time=d["switch_time"])
    if arm == "switch_old":
        return switch.solve_cuda_switch(**switched, **args)
    return solve_cuda_prepared(preparation=p["template"]["prepared_arms"][arm], **switched, **args)


def emit(event, payload):
    message = dict(payload, event=event)
    sys.stdout.buffer.write(MARKER + json.dumps(message, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()


def admission(files, protocol_sha, *, coordinator):
    token = os.environ.get(TOKEN)
    item = os.environ.get(ADMISSION)
    require(bool(token) and os.environ.get(HASH) == protocol_sha and item is not None and
            Path(item).resolve() == files["coordinator_admission"].resolve() and
            all(os.environ.get(k) == "1" for k in previous.THREADS),
            "Use the matching fixed-window wrapper with one BLAS thread")
    until = time.monotonic() + 30
    while not files["coordinator_admission"].exists() and time.monotonic() < until:
        time.sleep(.025)
    a = base.read(files["coordinator_admission"])
    require((a["schema"],a["status"],a["protocol_sha256"],a["run_token"]) ==
            (SCHEMA,"admitted",protocol_sha,token), "Coordinator admission differs from this protocol attempt")
    admitted = identity(a["coordinator"])
    if coordinator:
        require(admitted == identity(current_process_identity()), "Wrapper admitted a different coordinator")
    else:
        require(admitted == identity(identity_from_pid(os.getppid())), "Worker parent does not match the admitted coordinator")
    return a


def child(index, output, scale, solver=solve):
    files = paths(output)
    p, sha = read_protocol(files, scale)
    require(0 <= index < len(p["schedule"]), "Child index is outside the formal cohort")
    a = admission(files, sha, coordinator=False)
    job = p["schedule"][index]
    target = output / (previous.stem(job)+".npz")
    require(not target.exists(), "A timed child cannot replace a trajectory")
    cpu = previous.affinity_mask()
    ident = current_process_identity()
    marker = dict(index=index,run_token=a["run_token"],inherited_affinity_mask=cpu,**{k:ident[k] for k in IDENTITY})
    emit("READY", marker)
    require(sys.stdin.buffer.readline() == b"GO\n", "Coordinator did not authorize this child to solve")
    ledger = base.read(files["identity_ledger"])
    slot = ledger["children"][index]
    require(ledger["schema"] == SCHEMA and ledger["status"] == "running" and ledger["protocol_sha256"] == sha and
            ledger["run_token"] == a["run_token"] and identity(ledger["coordinator"]) == identity(a["coordinator"]) and
            slot["index"] == index and slot["status"] == "registered" and identity(slot) == identity(ident) and
            role_identity(slot) == identity(a["coordinator"]), "Exact child identity was not registered before GO")
    config = configuration(job, p)
    started = time.perf_counter()
    try:
        solution = solver(job, p, config)
        core = time.perf_counter()-started
        signature = previous.model_signature(solution.model)
        release = time.perf_counter()
        solution.model = None
        released = time.perf_counter()-release
    except Exception as exc:
        emit("RETURN", dict(marker,status="failed",failure_type=type(exc).__name__,failure=public_failure(exc)))
        return 1
    emit("RETURN", dict(marker,status="solved",solver_seconds=core+released,solver_core_seconds=core,
                        model_release_seconds=released,spatial_sha256=signature))
    require(sys.stdin.buffer.readline() == b"SAVE\n", "Coordinator did not acknowledge the return marker")
    temporary = target.with_suffix(".tmp.npz")
    case.store_solution(temporary, solution, config)
    temporary.replace(target)
    return 0


def messages(stdout, out, transcript):
    try:
        with transcript.open("xb") as log:
            for line in iter(stdout.readline, b""):
                received = time.perf_counter_ns()
                if line.startswith(MARKER):
                    out.put((received,line[len(MARKER):]))
                log.write(line)
    finally:
        out.put(None)


def receive(out, event, slot, a, cpu, timeout):
    try:
        result = out.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"Fresh child timed out before {event}") from exc
    require(result is not None, f"Fresh child exited before {event}")
    received, payload = result
    marker = json.loads(payload)
    require(marker["event"] == event and marker["index"] == slot["index"] and identity(marker) == identity(slot) and
            marker["run_token"] == a["run_token"] and marker["inherited_affinity_mask"] == 1 << cpu,
            f"{event} marker does not match registered child, token or inherited CPU affinity")
    return received, marker


def child_attempt(index, row, p, output, files, ledger, a, cpu, worker_script=Path(__file__).resolve()):
    job = p["schedule"][index]
    stem = previous.stem(job)
    target = output / (stem+".npz")
    command = [sys.executable,"-B",str(worker_script),"--child",str(index),
               "--output-dir",str(output),"--scale",p["scale"]]
    require(not target.exists(), "Timed trajectory exists before launch")
    parent = identity(a["coordinator"])
    messages_out = queue.Queue()
    with (output / (stem+".stderr.log")).open("xb") as stderr:
        launched = time.perf_counter_ns()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        slot, reader = None, None
        try:
            slot = dict(identity_from_handle(process._handle),index=index,parent_pid=parent[0],parent_birth_filetime_100ns=parent[1],
                        launch_monotonic_ns=launched,status="registered")
            require(process.pid == slot["pid"] and not any(identity(entry) == identity(slot) for entry in ledger["children"]),
                    "Popen and open-handle process identity disagree or repeat")
            ledger["children"].append(slot)
            row.update(child_pid=slot["pid"],child_birth_filetime_100ns=slot["birth_filetime_100ns"],
                       launch_monotonic_ns=launched)
            base.save(files["identity_ledger"], ledger)
            reader = Thread(target=messages,args=(process.stdout,messages_out,output/(stem+".stdout.log")),daemon=True)
            reader.start()
            row["ready_received_monotonic_ns"], row["ready"] = receive(messages_out,"READY",slot,a,cpu,60)
            row["go_sent_monotonic_ns"] = time.perf_counter_ns()
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            returned, marker = receive(messages_out,"RETURN",slot,a,cpu,120)
            row.update(return_received_monotonic_ns=returned,marker=marker)
            require(marker["status"] in ("solved","failed") and not target.exists(),
                    "Fresh child status or serialization boundary differs")
            if marker["status"] == "solved":
                fresh = (returned-launched)/1e9
                solver = marker["solver_seconds"]
                require(0 < solver <= fresh and math.isfinite(solver) and
                        solver == marker["solver_core_seconds"] + marker["model_release_seconds"],
                        "Clock boundaries or solver phase accounting differ")
                row.update({key:marker[key] for key in ("solver_seconds","solver_core_seconds","model_release_seconds")})
                row.update(fresh_process_return_seconds=fresh,fresh_process_overhead_seconds=fresh-solver)
                row["save_sent_monotonic_ns"] = time.perf_counter_ns()
                process.stdin.write(b"SAVE\n")
                process.stdin.flush()
            else:
                row.update(failure_type=marker["failure_type"],failure=marker["failure"])
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                returncode = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=15)
            exited = time.perf_counter_ns()
            row.update(child_exit_code=returncode,exit_monotonic_ns=exited)
            if slot is not None and slot in ledger["children"]:
                end = identity_from_handle(process._handle)
                require(identity(end) == identity(slot) and end["exit_filetime_100ns"] > end["birth_filetime_100ns"],
                        "Exited process handle lacks a valid terminal identity")
                slot.update(status="exited",exit_monotonic_ns=exited,exit_code=returncode,
                            exit_filetime_100ns=end["exit_filetime_100ns"],
                            exit_kernel_cpu_100ns=end["kernel_cpu_100ns"],exit_user_cpu_100ns=end["user_cpu_100ns"])
                base.save(files["identity_ledger"], ledger)
            if reader is not None:
                reader.join(timeout=5)
            process.stdout.close()
    require(row["child_exit_code"] == 0 and target.is_file(), "Registered child failed or did not save its trajectory")
    row.update(artifact=target.name,sha256=base.digest(target))
    return target


def context(p):
    times = p["design"]["physical_times"]
    grids = base.grid_set()
    truth, ref = eligibility.qual.absolute_reference(base.read(base.WORK/p["reference_protocol"]["path"]),grids)
    keys = {(grid,t) for grid in grids for t in times}
    controls = {}

    def load(item, cls=MODEL):
        return case.load_solution(base.WORK/item["path"],cls)

    def fields(sol):
        require(np.array_equal(sol.times,times), "Scoring control omits a common physical observation")
        return sol, case.reference_fields(sol,grids)

    for c in p["gal_controls"]:
        first,last = load(c["early"]),switch.select(load(c["late"]),times[12:])
        cpu,gpu = (switch.select(load(c[k]),times) for k in ("cpu_recurrence","cuda_graph"))
        require(np.array_equal(first.times,times[:12]) and switch.switched.equal_spatial_models(
                (first.model,last.model,cpu.model,gpu.model)), "Archived Galerkin control coordinates differ")
        uniform = case.Solution(last.model,np.concatenate((first.states,last.states)),np.asarray(times),
                                last.probe_times,last.transverse,last.vorticity,last.row)
        u, f = fields(uniform)
        anchors = {name:{(r["grid"],r["time"]):r for r in eligibility.qual.scores(u,target,grids,keys)[0]}
                   for name,target in truth.items()}
        controls[c["name"],c["seed"]] = dict(signature=previous.model_signature(u.model),uniform={"cpu_uniform":(u,f)},
                switched={"cpu_switch":fields(cpu),"cuda_switch_old":fields(gpu)},anchors=anchors)
    for name,c in p["fd_controls"].items():
        cpu = load(c["cpu"],base.FullFD)
        controls[name,c["seed"]] = dict(signature=previous.model_signature(cpu.model),uniform={"cpu_fd":fields(cpu)})
    return grids, truth, keys, controls, ref


def work_inventory(metadata, job, p):
    arm = job["arm"]
    design, schedule = p["design"], p["step_schedule"]
    mixed = arm.startswith("switch_")
    rate = p["fd_controls"][job["name"]]["rate"] if arm == "fd" else schedule["startup_rate"]
    steps = schedule["steps"] if mixed else design["horizon"]*rate
    n = steps-1
    require(metadata["steps"] == steps and metadata["horizon"] == design["horizon"] and metadata["dt"] == 1/rate and
            metadata["graph_steps"] == design["cuda_graph_steps"], "Timed schedule or graph chunk differs")
    if arm == "fd":
        require(metadata["cuda_linear_solve"] == p["fd_controls"][job["name"]]["linear"], "FD linear variant differs")
    if mixed:
        split = rate*design["switch_time"]
        require(metadata["switch_step"] == split and metadata["late_rate"] == schedule["late_rate"] and
                metadata["nonlinear_evaluations"] == n and metadata["marching_fft_batches"] == 2*n and
                metadata["nonlinear_by_phase"] == dict(startup=split,late=n-split) and
                metadata["bridge_extra_fft_batches"] == 0 and metadata["priming_fft_batches"] == 2 and
                metadata["device_final_step"] == steps, "Switched work inventory differs from qualification")
        if arm != "switch_old":
            require(metadata["preparation"] == p["template"]["prepared_arms"][arm] and
                    metadata["compiled_cuda_modules"] == (2 if arm == "switch_host" else 1) and
                    metadata["phase_preparation_shared"] == (arm != "switch_module"), "Preparation variant differs")
    return dict(steps=steps,nonlinear_evaluations=n,marching_fft_batches=2*n,
                inverse_field_transforms=(4 if arm == "fd" else 3)*n,forward_field_transforms=(1 if arm == "fd" else 2)*n,
                compiled_cuda_modules=metadata.get("compiled_cuda_modules",2 if mixed else 1),
                phase_preparation_shared=metadata.get("phase_preparation_shared",False),
                source="verified_device_switch_inventory" if mixed else "uniform_registered_kernel")


class CachedScoringTrajectory:
    """Share reconstructed candidate fields across independent score targets."""

    def __init__(self, solution, grids):
        self.solution = solution
        self.by_grid = case.reference_fields(solution,grids)
        self.grid_keys = {(id(s),id(theta)):name for name,(s,theta,_) in grids.items()}

    def fields(self, index, s, theta):
        return self.by_grid[self.grid_keys[id(s),id(theta)]][index]

    def __getattr__(self, name):
        return getattr(self.solution,name)


def score_solution(target, row, job, p, scoring):
    grids,truth,keys,controls,_ = scoring
    d, c = p["design"], controls[job["name"],job["seed"]]
    require(row["marker"]["spatial_sha256"] == c["signature"], "Fresh child spatial coordinates differ")
    uniform, uniform_fields = next(iter(c["uniform"].values()))
    sol = previous.read_trajectory(target,uniform.model,configuration(job,p))
    require(np.array_equal(sol.times,d["physical_times"]) and np.array_equal(sol.probe_times,uniform.probe_times) and
            sol.probe_times.shape == (577,) and sol.states.shape == uniform.states.shape and
            all(getattr(sol,k).shape == (577,3) for k in eligibility.qual.PROBES) and
            all(np.isfinite(getattr(sol,k)).all() for k in ("states",*eligibility.qual.PROBES)),
            "Common absolute observations, complete probes or finite trajectory differ")
    row["metadata"] = sol.row
    work = work_inventory(sol.row,job,p)
    switched = job["arm"].startswith("switch_")
    matched = c["switched"] if switched else c["uniform"]
    scored = CachedScoringTrajectory(sol,grids)
    implementation = {name:eligibility.qual.compare(scored,anchor,fields,grids,keys,d["implementation_gate"])
                      for name,(anchor,fields) in matched.items()}
    direct, maxima, guards, guard_worst = {}, {}, {}, {}
    delta = eligibility.qual.scores(scored,uniform_fields,grids,keys)[0] if switched else []
    for name, target_fields in truth.items():
        direct[name], maxima[name] = eligibility.qual.scores(scored,target_fields,grids,keys)
        if switched:
            a = c["anchors"][name]
            guard = [dict(grid=r["grid"],time=r["time"],**{k:r[k]*(1+a[r["grid"],r["time"]][k])+a[r["grid"],r["time"]][k]
                                                        for k in switch.METRICS}) for r in delta]
            indexed = {(r["grid"],r["time"]):r for r in guard}
            require(all(r[k] <= indexed[r["grid"],r["time"]][k] + 2e-13 for r in direct[name] for k in switch.METRICS),
                    "Direct physical error exceeds independent triangle guard")
            guards[name], guard_worst[name] = guard, switch.old.score_rows(guard,keys)
    ip = all(v["passed"] for v in implementation.values())
    pp = all(max(v.values()) <= d["physical_gate"] for v in maxima.values())
    gp = all(max(v.values()) <= d["physical_gate"] for v in guard_worst.values())
    wake = base.wake(sol)
    return dict(implementation=implementation,physical_scores=direct,physical_worst=maxima,guard_scores=guards,
                guard_worst=guard_worst,wake=wake,implementation_pass=ip,physical_pass=pp,guard_pass=gp,
                qualified=bool(ip and pp and gp and wake["sustained_alternating_probe_signs"]),work=work)


def execute(output, scale):
    files = paths(output)
    p, sha = read_protocol(files, scale)
    a = admission(files,sha,coordinator=True)
    require(not any(files[k].exists() for k in ("execution","timing","identity_ledger","summary")),
            "Only one prepared timing attempt is registered")
    coordinator = current_process_identity()
    bindings = dict(protocol_sha256=sha,run_token=a["run_token"],scale=scale)
    execution = dict(bindings,status="running",start_utc=utc(),**{k:coordinator[k] for k in IDENTITY})
    ledger = dict(bindings,schema=SCHEMA,status="running",coordinator=coordinator,children=[])
    timing = dict(bindings,status="running",coordinator=coordinator,rows=[])
    base.save(files["execution"],execution)
    base.save(files["identity_ledger"],ledger)
    base.save(files["timing"],timing)
    try:
        previous.check_hashes(p["source_hashes"])
        cpu = base.pin()
        scoring = context(p)
        timing.update(logical_cpu=cpu,reference=scoring[-1],gpu=switch.old.old_joint.nvidia(),
                      platform=platform.platform(),python_version=platform.python_version(),
                      environment={key:os.environ[key] for key in previous.THREADS})
        for index,job in enumerate(p["schedule"]):
            row = dict(job,index=index,status="failed",qualified=False)
            try:
                target = child_attempt(index,row,p,output,files,ledger,a,cpu)
                row.update(score_solution(target,row,job,p,scoring))
                row["status"] = "qualified" if row["qualified"] else "numerical_gate_failed"
            except Exception as exc:
                row.setdefault("failure_type",type(exc).__name__)
                row.setdefault("failure",public_failure(exc))
            base.save(output/(previous.stem(job)+".json"),row)
            timing["rows"].append(row)
            base.save(files["timing"],timing)
            print(index+1,job,row["status"],row.get("solver_seconds"),row.get("fresh_process_return_seconds"),flush=True)
        previous.check_hashes(p["source_hashes"])
        require(base.digest(files["protocol"]) == sha, "Prepared formal protocol changed during the cohort")
        timing.update(status="complete",numerical_qualified=all(r["qualified"] for r in timing["rows"]))
        base.save(files["timing"],timing)
        execution["status"] = "complete"
    except BaseException as exc:
        execution.update(status="failed",failure_type=type(exc).__name__,failure=public_failure(exc))
        raise
    finally:
        complete = [r["index"] for r in ledger["children"] if r["status"] == "exited"] == list(range(len(p["schedule"])))
        ledger["status"] = "complete" if complete else "incomplete"
        base.save(files["identity_ledger"],ledger)
        execution.update(end_utc=utc(),identity_ledger_sha256=base.digest(files["identity_ledger"]))
        base.save(files["execution"],execution)


def clock_order(row, slot, token, cpu):
    left, right = slot["launch_monotonic_ns"], slot["exit_monotonic_ns"]
    observed = [row[k] for k in ("ready_received_monotonic_ns","go_sent_monotonic_ns",
                                 "return_received_monotonic_ns","save_sent_monotonic_ns") if k in row]
    require(row["launch_monotonic_ns"] == left and row["exit_monotonic_ns"] == right and
            row["child_exit_code"] == slot["exit_code"] and
            left <= min(observed,default=right) <= max(observed,default=left) <= right and observed == sorted(observed),
            "Coordinator subprocess events do not match terminal ledger")
    for kind,event in (("ready","READY"),("marker","RETURN")):
        if kind in row:
            marker = row[kind]
            require(identity(marker) == identity(slot) and marker["index"] == slot["index"] and
                    marker["event"] == event and marker["run_token"] == token and marker["inherited_affinity_mask"] == 1<<cpu,
                    "Raw transport marker differs from exact identity ledger")
    if "solver_seconds" in row:
        solver, fresh = (row[k] for k in CLOCKS)
        require(len(observed) == 4 and
                all(row[k] == row["marker"][k] for k in ("solver_seconds","solver_core_seconds","model_release_seconds")) and
                0 < solver <= (row["return_received_monotonic_ns"]-row["go_sent_monotonic_ns"])/1e9 <= fresh and
                solver == row["solver_core_seconds"] + row["model_release_seconds"] and
                fresh == (row["return_received_monotonic_ns"]-left)/1e9 and
                row["fresh_process_overhead_seconds"] == fresh-solver, "Saved dual clocks differ from raw events")


def ledger_roles(p, journal, ledger, a):
    known = identity(p["isolation"]["known_background"][0])
    coordinator = identity(a["coordinator"])
    require(known != coordinator, "Coordinator identity equals the allowed background")
    roles = {known:("known_background",None),coordinator:("cohort_coordinator",None)}
    children, rows = ledger["children"], journal["rows"]
    valid = (journal["status"] == ledger["status"] == "complete" and ledger["schema"] == SCHEMA and
             len(children) == len(rows) == len(p["schedule"]) and
             identity(ledger["coordinator"]) == identity(journal["coordinator"]) == coordinator)
    candidate = {}
    prior = 0
    for i,(slot,row) in enumerate(zip(children,rows)):
        key = identity(slot)
        joined = (key not in roles and key not in candidate and role_identity(slot) == coordinator and
                  slot["status"] == "exited" and slot["index"] == row["index"] == i and
                  (row.get("child_pid"),row.get("child_birth_filetime_100ns")) == key and
                  slot["exit_filetime_100ns"] > slot["birth_filetime_100ns"] and
                  min(slot["exit_kernel_cpu_100ns"],slot["exit_user_cpu_100ns"]) >= 0 and
                  prior < slot["launch_monotonic_ns"] < slot["exit_monotonic_ns"] and
                  all(key_name in row and identity(row[key_name]) == key and row[key_name]["index"] == i and
                      row[key_name]["run_token"] == a["run_token"] for key_name in ("ready","marker")))
        valid = valid and joined
        if joined:
            clock_order(row,slot,a["run_token"],journal["logical_cpu"])
        candidate[key] = ("cohort_child",i)
        prior = slot["exit_monotonic_ns"]
    return (roles | candidate if valid else roles), valid


def filetime(value):
    dt = datetime.fromisoformat(value).astimezone(timezone.utc)
    delta = dt - datetime(1601,1,1,tzinfo=timezone.utc)
    fraction = value.split("T",1)[1].split(".",1)
    digits = fraction[1][:7] if len(fraction) == 2 else ""
    last = int(digits[-1]) if len(digits) == 7 and digits.isdecimal() else 0
    return ((delta.days*86400+delta.seconds)*1_000_000+delta.microseconds)*10+last


def classify(earlier, later, roles):
    old = {identity(r):r for r in earlier["python"]}
    require(len(old) == len(earlier["python"]) and
            len({identity(r) for r in later["python"]}) == len(later["python"]), "Duplicate snapshot process identity")
    rows = []
    for current in later["python"]:
        key = identity(current)
        before = old.get(key,{})
        kernel,user = (current[name]-before.get(name,0) for name in ("kernel_cpu_100ns","user_cpu_100ns"))
        label,index = roles.get(key,("external_python",None))
        rows.append(dict(pid=key[0],birth_filetime_100ns=key[1],label=label,child_index=index,
                         kernel_delta_100ns=kernel,user_delta_100ns=user,cpu_delta_100ns=kernel+user,
                         previous_identity_observed=bool(before),exited=current["exit_filetime_100ns"] != 0))
    return rows


def monitor_status(p, m, e, ledger, journal, a):
    isolation = p["isolation"]
    threshold = isolation["thresholds"]
    roles, valid = ledger_roles(p,journal,ledger,a)
    require(m["identity_ledger_valid"] == valid and m["identity_schema"] == SCHEMA and
            m["scale"] == p["scale"] and m["known_background"] == isolation["known_background"],
            "Wrapper identity or background registry differs")
    if m["status"] == "monitor_failed":
        return "monitor_failed"
    known, coordinator = identity(isolation["known_background"][0]), identity(a["coordinator"])

    def active(records):
        bg = [r["cpu_delta_100ns"] for r in records if r["label"] == "known_background"]
        return (any(v > threshold["background_individual_cpu_seconds"]*1e7 for v in bg) or
                sum(bg) > threshold["background_total_cpu_seconds"]*1e7 or
                any(r["cpu_delta_100ns"] > threshold["unexpected_python_cpu_seconds"]*1e7
                    for r in records if r["label"] == "external_python"))

    def valid_sample(s):
        return not (s["snapshot_failure_type"] or s["unresolved_python"])

    def quiet(s):
        return valid_sample(s) and len(s["python"]) == 1 and identity(s["python"][0]) == known and not s["python"][0]["exit_filetime_100ns"]

    def nonnegative(rows):
        return all(min(r["kernel_delta_100ns"],r["user_delta_100ns"]) >= 0 for r in rows)

    before = m["initial_snapshot"]
    require(quiet(before) and [r["window"] for r in m["preflight"]] == list(range(1,isolation["preflight_windows"]+1)),
            "Initial exact-identity preflight differs")
    for record in m["preflight"]:
        later = record["snapshot"]
        derived = classify(before,later,{known:("known_background",None)})
        require(quiet(later) and derived == record["classification"] and nonnegative(derived) and not active(derived) and
                0 <= record["gpu_percent"] <= threshold["preflight_gpu_percent"], "Quiet preflight failed independent checks")
        before = later
    duration, interval = isolation["window_seconds"], isolation["sample_seconds"]
    targets = list(range(interval,duration+1,interval))
    require(m["window_seconds"] == duration and m["sampling_interval_seconds"] == interval and
            [s["target_elapsed_seconds"] for s in m["samples"]] == targets, "Missing fixed-window sample targets")
    start,deadline,end = (filetime(m[key]) for key in ("start_utc","planned_end_utc","end_utc"))
    start_work,end_work,admitted = (filetime(v) for v in (e["start_utc"],e["end_utc"],a["admitted_utc"]))
    require(deadline == m["planned_end_filetime_100ns"] and deadline-start == duration*10_000_000 and
            end >= deadline and m["end_elapsed_seconds"] >= duration, "Fixed monitor interval differs")
    final = m["coordinator_exit"]
    within = (e["status"] == "complete" and identity(e) == coordinator and start <= admitted <= start_work <= end_work <= deadline and
              final is not None and identity(final) == coordinator and end_work <= final["exit_filetime_100ns"] <= deadline)
    coverage = m["final_gpu_percent"] is not None and 0 <= m["final_gpu_percent"] <= 100
    external, overlapping, parent_consistent, terminal_consistent = False, False, True, True
    terminals = {identity(r):r for r in ledger["children"]} if valid else {}
    for record,target in zip(m["samples"],targets,strict=True):
        later = record["snapshot"]
        derived = classify(before,later,roles)
        triggered = active(derived)
        require(derived == record["classification"] and triggered == record["external_activity_detected"],
                "Raw external CPU counters disagree with monitor")
        external |= triggered
        overlapping |= sum(r["label"] == "cohort_child" and not r["exited"] for r in derived) > 1
        parent_consistent &= all(r["enumerated_parent_pid"] == coordinator[0] for r in later["python"]
                                 if roles.get(identity(r),(None,None))[0] == "cohort_child")
        terminal_consistent &= all(r[k] == terminals[identity(r)][target_key]
                                   for r in later["python"] if identity(r) in terminals and r["exit_filetime_100ns"]
                                   for k,target_key in (("exit_filetime_100ns","exit_filetime_100ns"),
                                                        ("kernel_cpu_100ns","exit_kernel_cpu_100ns"),
                                                        ("user_cpu_100ns","exit_user_cpu_100ns")))
        elapsed, sampled = record["elapsed_seconds"], record["sample_started_elapsed_seconds"]
        coverage &= (valid_sample(later) and nonnegative(derived) and sampled <= elapsed and
                     0 <= elapsed-target <= threshold["maximum_sample_lateness_seconds"])
        before = later
    coverage &= parent_consistent and terminal_consistent
    require((coverage,within,external,overlapping,parent_consistent,terminal_consistent) ==
            tuple(m[k] for k in ("sample_coverage_pass","solver_finished_within_window","external_activity_detected",
                                 "child_overlap_detected","observed_child_parent_consistent","observed_child_exit_consistent")),
            "Monitoring flags differ from raw observations")
    expected = ("stopped_after_post_window_grace" if m["coordinator_force_stopped"] else "failed" if m["exit_code"] != 0
                else "completed_with_invalid_identity_ledger" if not valid
                else "completed_outside_fixed_window" if not (within and coverage)
                else "completed_with_overlapping_children" if overlapping
                else "completed_with_observed_external_activity" if external else "completed_no_sampled_external_activity")
    require(m["status"] == expected, "Monitor disposition differs from reconstructed window")
    return expected


def timing_components(row):
    own = {key:value for key,value in row.items() if key.endswith("_seconds")}
    native = {key:value for key,value in row["metadata"].items() if key.endswith("_seconds") and key != "seconds"}
    require(not set(own)&set(native) and all(type(x) in (float,int) and math.isfinite(x) and x >= 0
                                            for x in (*own.values(),*native.values())), "Invalid timing components")
    return own | native


def verify_numerical_row(row,p):
    d = p["design"]
    keyset = {(grid,t) for grid in base.grid_set() for t in d["physical_times"]}
    switched = row["job"]["arm"].startswith("switch_")
    expected_implementations = {"cpu_switch","cuda_switch_old"} if switched else (
        {"cpu_fd"} if row["job"]["arm"] == "fd" else {"cpu_uniform"})
    require(set(row["implementation"]) == expected_implementations and
            set(row["physical_scores"]) == set(d["references"]) and
            set(row["guard_scores"]) == (set(d["references"]) if switched else set()),
            "Saved numerical row omits a required implementation or physical gate")
    for prefix in ("physical","guard"):
        maxima = {ref:switch.old.score_rows(scores,keyset) for ref,scores in row[prefix+"_scores"].items()}
        passed = all(max(v.values()) <= d["physical_gate"] for v in maxima.values())
        require(maxima == row[prefix+"_worst"] and passed == row[prefix+"_pass"],
                "Saved physical or guard result differs from its raw norms")
    outcomes = []
    for result in row["implementation"].values():
        worst = switch.old.score_rows(result["scores"],keyset)
        require(set(result["probes"]) == set(eligibility.qual.PROBES), "Saved probe coverage differs")
        passed = all(math.isfinite(v) and 0 <= v <= d["implementation_gate"]
                     for v in (*worst.values(),*result["probes"].values()))
        require(worst == result["worst"] and passed == result["passed"], "Saved implementation norms or probes differ")
        outcomes.append(passed)
    expected = (all(outcomes) and row["physical_pass"] and row["guard_pass"] and
                row["wake"]["sustained_alternating_probe_signs"])
    require(all(outcomes) == row["implementation_pass"] and bool(expected) == row["qualified"] and
            work_inventory(row["metadata"],row["job"],p) == row["work"], "Saved numerical gate or work ledger differs")


def contrasts(p, rows, isolated):
    groups = {tuple(key):[] for key in p["groups"]}
    for row in rows:
        job = row["job"]
        groups[job["name"],job["arm"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row:row["job"]["block"])
        require([row["job"]["block"] for row in group] == list(range(1,6)), "Paired blocks are incomplete")
    summaries = []
    for (name,arm), group in groups.items():
        qualified = sum(row["qualified"] for row in group)
        records = [timing_components(row) for row in group] if qualified == 5 else []
        stats = {key:previous.statistics([r[key] for r in records])
                 for key in sorted(set.intersection(*(set(r) for r in records)))} if records else {}
        summaries.append(dict(name=name,arm=arm,qualified=qualified,phase_seconds=stats))
    paired = []
    for item in p["comparisons"]:
        a,b = (groups[tuple(item[key])] for key in ("baseline","candidate"))
        complete = all(r["qualified"] for r in a+b)
        clocks = {}
        for key in CLOCKS:
            if complete:
                sample = [x[key]/y[key] for x,y in zip(a,b,strict=True)]
                clocks[key] = dict(previous.statistics(sample),candidate_wins=sum(v>1 for v in sample))
            else:
                clocks[key] = None
        rules = p["design"]["reporting"]
        practical = complete and all(clocks[k]["candidate_wins"] >= rules["consistent_improvement_required_wins_per_clock"] and
                                     clocks[k]["median"] >= rules["median_minimum_practical_ratio"] for k in CLOCKS)
        phases = {}
        if complete:
            left,right = ([timing_components(r) for r in subset] for subset in (a,b))
            phase_names = sorted(set.intersection(*(set(r) for r in left+right)))
            phases = {key:previous.statistics([y[key]-x[key] for x,y in zip(left,right,strict=True)]) for key in phase_names}
        paired.append(dict(item,clocks=clocks,phase_delta_candidate_minus_baseline_seconds=phases,
                           meets_practical_threshold=bool(practical),
                           isolated_practical_improvement_supported=bool(practical and isolated)))
    factorials = []
    for item in p["factorials"]:
        source = {arm:groups[item["name"],arm] for arm in item["coefficient"]}
        complete = all(row["qualified"] for group in source.values() for row in group)
        by_block = [{arm:timing_components(group[i]) for arm,group in source.items()} for i in range(5)] if complete else []
        stats = {}
        for key in item["components"]:
            sample = [sum(item["coefficient"][arm]*row[key] for arm,row in data.items())
                      for data in by_block] if by_block and all(key in row for data in by_block for row in data.values()) else []
            stats[key] = previous.statistics(sample)
        factorials.append(dict(item,seconds=stats))
    return dict(groups=summaries,paired=paired,factorials=factorials)


def summarize(output,scale):
    files = paths(output)
    p,sha = read_protocol(files,scale)
    require(not files["summary"].exists(), "Preserve the existing prepared timing summary")
    m = base.read(files["isolation"])
    checks = {k+"_sha256":base.digest(v) for k,v in files.items() if k != "summary" and v.exists()}
    require(m["protocol_sha256"] == sha and m["scale"] == scale, "Monitor belongs to another cohort")
    require(m["status"] not in ("preflight","running","launching"), "The monitor has not reached a terminal state")
    execution = base.read(files["execution"]) if files["execution"].exists() else {}
    incomplete_failure = m["status"] in ("failed","stopped_after_post_window_grace") and not (
        {"identity_ledger_sha256","end_utc"} <= execution.keys() and
        all(files[k].exists() for k in ("timing","identity_ledger","coordinator_admission")))
    if (m["status"] == "monitor_failed" or incomplete_failure or
            (m["status"] == "preflight_rejected" and not execution)):
        rows = len(base.read(files["timing"])["rows"]) if files["timing"].exists() else 0
        base.save(files["summary"],dict(status="diagnostic_incomplete" if rows else "diagnostic_no_cohort",
                                        scale=scale,rows=rows,monitor_status=m["status"],
                                        numerical_qualified=False,isolated_speed_claim_supported=False,**checks))
        return
    t,e,l,a = (base.read(files[k]) for k in ("timing","execution","identity_ledger","coordinator_admission"))
    require(all(record["protocol_sha256"] == sha and record["run_token"] == m["run_token"] for record in (t,e,l,a)) and
            all(m.get(key+"_sha256") == checks[key+"_sha256"] for key in
                ("execution","identity_ledger","coordinator_admission","timing")) and
            e["identity_ledger_sha256"] == checks["identity_ledger_sha256"] and
            identity(a["coordinator"]) == identity(m["coordinator"]) == identity(e) and
            a["schema"] == SCHEMA and a["status"] == "admitted", "Protocol, admission, journal or monitor bindings differ")
    rows = t["rows"]
    require(len(rows) <= len(p["schedule"]), "Timing journal exceeds registered schedule")
    previous.check_hashes(p["source_hashes"])
    for i,row in enumerate(rows):
        job = p["schedule"][i]
        require(row["index"] == i and row["job"] == job and base.read(output/(previous.stem(job)+".json")) == row,
                "Timing row differs from the registered slot")
        if "artifact" in row:
            require(row["artifact"] == previous.stem(job)+".npz" and base.digest(output/row["artifact"]) == row["sha256"],
                    "Saved timing trajectory differs")
        require(not row["qualified"] or row["status"] == "qualified" and set(CLOCKS) <= row.keys(),
                "Qualified timing row has incomplete numerical or clock evidence")
        if "implementation" in row:
            verify_numerical_row(row,p)
    status = monitor_status(p,m,e,l,t,a)
    complete = len(rows) == len(p["schedule"]) and t["status"] == e["status"] == "complete"
    numerical = complete and all(row["qualified"] for row in rows)
    isolated = bool(numerical and status == "completed_no_sampled_external_activity")
    data = contrasts(p,rows,isolated) if complete else {}
    base.save(files["summary"],dict(status="passed" if isolated else "diagnostic_only" if complete else "diagnostic_incomplete",
                                    scale=scale,rows=len(rows),monitor_status=status,numerical_qualified=numerical,
                                    isolated_speed_claim_supported=isolated,**checks,**data))
    print(scale,status,len(rows),isolated,flush=True)


def freeze(output, scale):
    require(not output.exists() or not any(output.iterdir()), "Preserve existing prepared timing evidence")
    require(output.parent == base.HERE.resolve(), "Prepared cohort output must be directly below the study directory")
    template = base.read(eligibility.TEMPLATE)
    background = identity_from_pid(template["background_server_pid"])
    plan = eligibility.make_protocol(scale,background)
    sources = plan["source_hashes"]
    original,cache_path,budget_path,transport_path = (base.HERE/name for name in (
        "prepared_formal_scoring_audit.json","prepared_formal_scoring_cache_audit.json",
        "prepared_formal_window_budget.json","prepared_formal_transport_smoke.json"))
    cache,budget,transport = (base.read(path) for path in (cache_path,budget_path,transport_path))
    require(cache["status"] == transport["status"] == "passed" and len(cache["rows"]) == 19 and
            all(r["passed"] and r["cached_score_sha256"] == r["uncached_score_sha256"] for r in cache["rows"]) and
            cache["uncached_original_sha256"] == base.digest(original) and
            budget["status"] == "prospective_only" and budget["thresholds_unchanged"] and
            cache["source_sha256"] == budget["source_sha256"],
            "Cached score equality, transport or prospective budget evidence differs")
    estimate = budget["scales"][scale]
    require(estimate["rows"] == len(plan["schedule"]) and estimate["window_seconds"] == plan["isolation"]["window_seconds"] and
            estimate["current_window_supported_by_estimate"] and
            estimate["conservative_planning_seconds"] <= estimate["window_seconds"],
            "Frozen observation window lacks an adequate prospective budget")
    for manifest in (cache["source_sha256"],transport["sha256"]):
        for name,digest in manifest.items():
            eligibility.bind(sources,base.HERE/name,digest)
    eligibility.bind(sources,previous.OUT/"timing.json",budget["prior_diagnostic_timing_sha256"])
    plan["readiness_evidence"] = [eligibility.bind(sources,path) for path in (original,cache_path,budget_path,transport_path)]
    base.save(paths(output)["protocol"],plan)
    print("FROZEN",scale,len(plan["schedule"]),base.digest(paths(output)["protocol"]),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-only",action="store_true")
    actions.add_argument("--execute",action="store_true")
    actions.add_argument("--summarize",action="store_true")
    actions.add_argument("--child",type=int)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--scale",required=True,choices=("mid","high"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.freeze_only:
        freeze(output,args.scale)
    elif args.execute:
        execute(output,args.scale)
    elif args.summarize:
        summarize(output,args.scale)
    else:
        sys.exit(child(args.child,output,args.scale))
