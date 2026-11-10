"""Freeze a prospective CPU-separated cohort with unchanged 160 numerical jobs."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
import freeze_rank_fair_cache_formal as old
import coordinator_v2 as timing
from worker_bootstrap_v2 import SCHEMA as WORKER_SCHEMA
from transport_v2 import read, require
from cpu_fixture_v2 import assert_outer_owner

HERE = Path(__file__).resolve().parent
WORK, T, qual = old.WORK, old.T, old.qual
WINDOWS = old.WINDOWS
SCHEMA = timing.SCHEMA
READINESS_SCHEMA = "rfm-v8-affinity-recovery-readiness-2"
FRESH_BOUNDARY = "Immediately before atomic outer-slot active record through receipt of RETURN; includes that supervision commit, Popen, imports, identity admission and spatial fingerprint"


def source_hashes():
    return {p.relative_to(WORK).as_posix(): T.digest(p) for p in sorted(HERE.glob("*.py"))}


def bound(record):
    return qual.shared.bound(record)


def checked_readiness(record):
    value = read(bound(record))
    require(value["schema"] == READINESS_SCHEMA and value["status"] == "passed" and value["sources_sha256"] == source_hashes(),
            "Recovery readiness does not bind the complete current v2 source")
    cpu = read(bound(value["cpu_fixture"]))
    gpu = read(bound(value["t9_equivalence"]))
    require(cpu["status"] == "passed" and cpu["cpu_only"] is True and cpu["sources_sha256"] == source_hashes(),
            "CPU8/6/4 independent deadline fixture is missing or stale")
    require(gpu["status"] == "passed" and gpu["all_cases_equivalent"] is True and gpu["formal_timing_eligible"] is False and
            gpu["sources_sha256"] == source_hashes(), "Exact T9 v2-entry equivalence smoke is missing or stale")
    bound(value["outer_watchdog_exe"])
    bound(value["outer_watchdog_source"])
    return value


def make_window(name, qualification, readiness_record, env=None, background_record=None):
    readiness = checked_readiness(readiness_record)
    base = old.make_window(name, qualification, env)
    p = deepcopy(base)
    p.pop("run_token")
    jobs = p.pop("requests")
    background = read(bound(background_record))["known_background"] if background_record is not None else []
    require(len({(r["pid"], r["birth_filetime_100ns"]) for r in background}) == len(background), "Duplicate identified background")
    p.update(schema=SCHEMA, coordinator_cpu=8, monitor_cpu=6, known_background=background, identified_background_evidence=background_record,
             recovery_revision="coordinator8_monitor6_worker4_external_job_slot_deadline_v2", recovery_readiness=readiness_record,
             outer_supervision=dict(observer_cpu=10, owner_cpu=12, atomic_job_before_resume=True, slot_state_schema="rfm-v8-outer-slot-1",
                                    slot_hard_timeout_seconds=45, whole_window_watchdog_seconds=840, confirmation_seconds=5,
                                    executable=readiness["outer_watchdog_exe"], source=readiness["outer_watchdog_source"]))
    p["design"]["monitor"].update(coordinator_cpu=8, monitor_cpu=6, known_background=background)
    p["design"]["clocks"]["fresh_process_return_seconds"] = FRESH_BOUNDARY
    p["source_hashes"].update(source_hashes())
    for record in (readiness_record, readiness["cpu_fixture"], readiness["t9_equivalence"], readiness["outer_watchdog_exe"], readiness["outer_watchdog_source"]):
        p["source_hashes"][record["path"]] = record["sha256"]
    if background_record is not None:
        p["source_hashes"][background_record["path"]] = background_record["sha256"]
    for job in jobs:
        job.pop("run_token")
        job.update(schema=WORKER_SCHEMA, inherited_affinity_mask=256, effective_affinity_mask=16)
    token = old.rank.frozen.canonical_sha(dict(p, jobs=jobs))
    p.update(run_token=token, requests=[dict(job, run_token=token) for job in jobs])
    # Every numerical request, policy, order, control and gate remains unchanged.
    removed = {"schema", "run_token", "inherited_affinity_mask", "effective_affinity_mask"}
    require([{k: v for k, v in j.items() if k not in removed} for j in p["requests"]] ==
            [{k: v for k, v in j.items() if k not in removed} for j in base["requests"]] and
            p["controls"] == base["controls"] and p["window_spec"] == base["window_spec"], "Numerical scope or ordering changed")
    return p


def verify_plan(p, *, live=False):
    require(p["schema"] == SCHEMA and p["window"] in WINDOWS and p["study"] == "rank_fair_two_cache_policies", "Wrong v2 cohort")
    T.check_sources(p["source_hashes"])
    actual = old.environment() if live else p["environment"]
    require(p == make_window(p["window"], bound(p["qualification"]["protocol"]).parent, p["recovery_readiness"], actual, p["identified_background_evidence"]),
            "V2 frozen deadline, environment, exact numerical scope or source changed")


def evidence(p):
    return {name: p[name] for name in ("qualification", "environment", "source_hashes", "recovery_readiness", "outer_supervision", "identified_background_evidence")}


def execute(protocol, output, outer):
    assert_outer_owner(outer)
    raw = protocol.read_bytes()
    p = json.loads(raw)
    verify_plan(p, live=True)
    before = time.perf_counter_ns()
    summary = timing.run(protocol, output, p, raw, worker=timing.SOLVER_WORKER)
    after = time.perf_counter_ns()
    guard = dict(status="failed", protocol_sha256=sha256(raw).hexdigest(), opening_environment=p["environment"],
                 opening_guard_qpc_ns=before, closing_guard_qpc_ns=after, coordinator_summary_sha256=T.digest(output / "summary.json"),
                 outer_evidence_directory=outer.relative_to(WORK).as_posix(), outer_owner_ready_sha256=T.digest(outer / "owner_ready.json"))
    try:
        T.check_sources(p["source_hashes"])
        require(old.environment() == p["environment"] and protocol.read_bytes() == raw == (output / "protocol.json").read_bytes(), "Runtime or protocol changed")
        guard.update(status="passed", closing_environment=p["environment"])
    except Exception as error:
        guard.update(failure_type=type(error).__name__)
    T.save(output / "execution_environment_guard.json", guard)
    return dict(status=summary["status"], environment_guard=guard["status"], window=p["window"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-all", type=Path)
    actions.add_argument("--execute", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--outer-evidence", type=Path)
    parser.add_argument("--background-record", type=Path)
    args = parser.parse_args()
    if args.execute:
        require(args.output is not None and args.outer_evidence is not None, "Externally owned output is required")
        print(json.dumps(execute(args.execute.resolve(), args.output.resolve(), args.outer_evidence.resolve())))
    else:
        require(args.qualification is not None and args.readiness is not None, "Exact prior qualification and new readiness are required")
        record = qual.shared.record(args.readiness.resolve())
        env = old.environment()
        background = qual.shared.record(args.background_record.resolve()) if args.background_record else None
        plans = [make_window(name, args.qualification.resolve(), record, env, background) for name in WINDOWS]
        require(all(evidence(p) == evidence(plans[0]) for p in plans), "Window evidence differs")
        root = args.freeze_all.resolve()
        root.mkdir()
        for p in plans:
            T.save(root / (p["window"] + ".json"), p)
