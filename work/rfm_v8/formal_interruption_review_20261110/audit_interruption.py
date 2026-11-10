"""Read-only standard-library audit of the interrupted 2026-11-10 formal window.

Only this new audit directory receives writes. No solver/library imports or GPU calls.
"""
import ast
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

OUT = Path(__file__).resolve().parent
V8 = OUT.parent
WORK = V8.parent
RAW = V8 / "rank_fair_cache_formal_mid_output"
PROTO = V8 / "rank_fair_cache_formal_protocols"
QUAL = V8 / "rank_fair_cache_qualification_output"
UTC = dt.timezone.utc
SHANGHAI = dt.timezone(dt.timedelta(hours=8))


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def relative(p):
    return p.relative_to(WORK).as_posix()


def file_record(p):
    s = p.stat()
    return dict(path=relative(p), sha256=digest(p), size=s.st_size,
                last_write_utc=dt.datetime.fromtimestamp(s.st_mtime, UTC).isoformat(),
                last_write_shanghai=dt.datetime.fromtimestamp(s.st_mtime, SHANGHAI).isoformat())


def check_record(r):
    p = WORK / r["path"]
    return dict(path=r["path"], expected_sha256=r["sha256"], actual_sha256=digest(p),
                matched=digest(p) == r["sha256"])


def stamp(filetime):
    return (dt.datetime(1601, 1, 1, tzinfo=UTC) +
            dt.timedelta(microseconds=filetime // 10)).astimezone(SHANGHAI).isoformat()


def npy(data):
    assert data[:6] == b"\x93NUMPY"
    off = 10 if data[6] == 1 else 12
    size = int.from_bytes(data[8:off], "little")
    h = ast.literal_eval(data[off:off + size].decode("latin1"))
    assert not h["fortran_order"]
    return h, data[off + size:]


def archive(p):
    with zipfile.ZipFile(p) as z:
        return {n[:-4]: npy(z.read(n)) for n in z.namelist()}


def unicode_json(pair):
    h, b = pair
    assert h["descr"].startswith("<U") and h["shape"] == ()
    return json.loads(b.decode("utf-32-le"))


def doubles(pair):
    h, b = pair
    assert h["descr"] == "<f8"
    return [v[0] for v in struct.iter_unpack("<d", b)]


def non_time(meta):
    return {k: v for k, v in meta.items() if k != "seconds" and not k.endswith("_seconds")}


def finite(pair):
    assert pair[0]["descr"] in ("<c16", "<f8")
    return all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", pair[1]))


def identity(r):
    return (r["pid"], r["birth_filetime_100ns"])


def main():
    before = {relative(p): file_record(p) for p in sorted(RAW.rglob("*")) if p.is_file()}
    p = read(RAW / "protocol.json")
    rows = read(RAW / "runs.json")
    ledger = read(RAW / "identity_ledger.json")
    events = [json.loads(l) for l in (RAW / "monitor_events.jsonl").read_text().splitlines()]
    samples = [json.loads(l) for l in (RAW / "process_samples.jsonl").read_text().splitlines()]
    opened = next(e["reply"]["open_qpc_ns"] for e in events if e["command"]["event"] == "OPEN")
    qsummary = read(QUAL / "qualification_summary.json")
    qprotocol = read(QUAL / "protocol.json")
    qhash = digest(QUAL / "qualification_summary.json")
    assert qhash == "1da5cd2cf5771a701ba172a3e979b2c8003f84328810ade76a692756f33231df"
    qrefs = [check_record(r) for r in p["qualification"].values()]
    assert all(r["matched"] for r in qrefs)
    assert len(qsummary["rows"]) == 78 and all(r["qualified"] for r in qsummary["rows"])
    completed = []
    for row in rows:
        i = row["index"]
        request = read(RAW / f"request_{i:03d}.json")
        control = p["controls"][i]
        qi = request["qualification_index"]
        qrow = qsummary["rows"][qi]
        capture = read(WORK / qrow["source_capture"]["path"])
        score = read(WORK / qrow["score"]["path"])
        result = read(WORK / capture["result"]["path"])
        refs = [check_record(r) for r in [qrow["score"], qrow["source_capture"], capture["result"], capture["artifact"],
                                         capture["watchdog"], capture["phases"], request["ptx_artifact"]]]
        checks = dict(raw_sha256=digest(RAW / row["artifact"]) == row["trajectory_sha256"],
                      request_exactly_frozen=request == p["requests"][i],
                      request_sha256=digest(RAW / f"request_{i:03d}.json") == row["request_sha256"],
                      attempt_equals_runs=read(RAW / f"attempt_{i:03d}.json") == row,
                      qualification_refs=all(r["matched"] for r in refs),
                      qualification_job=score["job"] == capture["job"] == result["job"] == qprotocol["slots"][qi] == control["job"],
                      qualification_capture=score["capture"] == qrow["source_capture"],
                      qualification_status=qrow["qualified"] and score["qualified"] and capture["status"] == result["status"] == "complete",
                      formal_control=control["index"] == qi and control["artifact"] == capture["artifact"] and control["score"] == qrow["score"],
                      job_match=all(request[k] == capture["job"][k] for k in ("name", "arm", "seed", "policy", "config", "family", "engine")),
                      spatial_sha256=row["marker"]["spatial_sha256"] == result["spatial_sha256"] == control["spatial_sha256"])
        a = archive(RAW / row["artifact"])
        b = archive(WORK / capture["artifact"]["path"])
        times, bt = doubles(a["times"]), doubles(b["times"])
        ids = [bt.index(t) for t in times]
        stride = math.prod(b["states"][0]["shape"][1:]) * 16
        expect = b"".join(b["states"][1][j * stride:(j + 1) * stride] for j in ids)
        checks.update(array_keys=set(a) == set(b) == {"states", "times", "probe_times", "transverse", "vorticity", "config", "metadata"},
                      snapshot_schedule=times == request["solve_args"]["snapshots"] and len(times) == 19 and bt == capture["job"]["args"]["snapshots"],
                      states_bytewise_exact=a["states"][1] == expect and a["states"][0]["shape"] == (19, *b["states"][0]["shape"][1:]),
                      probe_times_complete=doubles(a["probe_times"]) == [n / 8 for n in range(577)],
                      probes_bytewise_exact=all(a[k] == b[k] for k in ("probe_times", "transverse", "vorticity")),
                      arrays_finite=all(finite(a[k]) for k in ("states", "times", "probe_times", "transverse", "vorticity")),
                      configuration=unicode_json(a["config"]) == unicode_json(b["config"]) == request["config"])
        meta, qmeta = unicode_json(a["metadata"]), unicode_json(b["metadata"])
        checks["non_time_metadata"] = non_time(meta) == non_time(qmeta)
        cache = row["marker"]["cache"]
        checks["cache"] = all(meta[k] == v for k, v in cache.items()) and cache["ptx_cache_policy"] == request["policy"] == "strict_hit" and cache["ptx_cache_observed_mode"] == "hit" and cache["ptx_nvrtc_seconds"] == 0 and cache["private_context_finally_unbound"] is True and cache["cuda_context_mode"] == request["runtime_id"] and cache["ptx_cache_key"] == request["ptx_cache_key"] and cache["ptx_artifact_sha256"] == request["ptx_artifact"]["sha256"] and not (RAW / f"explicit_cache_{i:03d}").exists()
        entry = ledger["children"][i]
        markers = {kind: read(RAW / f"{kind.lower()}_{i:03d}.json") for kind in ("READY", "GO", "RETURN", "SAVE")}
        checks["latch_identity"] = all(m["index"] == i and m["event"] == kind and m["run_token"] == p["run_token"] and m["request_sha256"] == row["request_sha256"] and identity(m) == identity(row["child_identity"]) and m["inherited_affinity_mask"] == request["affinity_mask"] for kind, m in markers.items())
        checks["latch_records"] = markers["READY"] == row["ready"] and markers["RETURN"] == row["marker"] and markers["GO"]["sent_monotonic_ns"] == row["go_sent_monotonic_ns"] and markers["SAVE"]["sent_monotonic_ns"] == row["save_sent_monotonic_ns"]
        hard = min(opened + p["maximum_active_ns"], row["launch_monotonic_ns"] + round(p["slot_hard_timeout_seconds"] * 1e9))
        chronology = [row[k] for k in ("launch_monotonic_ns", "ready_received_monotonic_ns", "go_sent_monotonic_ns", "return_received_monotonic_ns", "save_sent_monotonic_ns", "exit_wait_completed_monotonic_ns", "exit_monotonic_ns")]
        checks["transport_deadlines"] = chronology == sorted(chronology) and row["hard_deadline_ns"] == hard and chronology[1] <= chronology[0] + p["ready_timeout_seconds"] * 1e9 and chronology[3] <= chronology[2] + p["return_timeout_seconds"] * 1e9 and chronology[5] <= min(hard, chronology[4] + p["exit_timeout_seconds"] * 1e9)
        checks["terminal"] = row["status"] == "complete" and row["exit_code"] == 0 and entry["status"] == "exited" and entry["terminal"] == row["terminal_identity"] and identity(entry["terminal"]) == identity(entry["identity"]) and any(all(x.get(k) == v for k, v in entry["terminal"].items()) for s in samples for x in s["terminals"])
        assert all(checks.values()), (i, [k for k, v in checks.items() if not v])
        completed.append(dict(index=i, qualification_index=qi, job={k: request[k] for k in ("name", "arm", "seed", "policy", "block")},
                              raw=file_record(RAW / row["artifact"]), checks=checks, qualification_references=refs,
                              qualification_snapshot_indices=ids, numerical_qualification_link_reverified=True,
                              timing_eligible=False, clocks_diagnostic_only={k: row[k] for k in ("solver_seconds", "fresh_process_return_seconds")},
                              note="Exact numerical values are recovered; cohort isolation and full five-block comparison remain unavailable."))
    unknown = ledger["children"][2]
    go = read(RAW / "go_002.json")
    ready = read(RAW / "ready_002.json")
    latest = samples[-1]
    worker_samples = [(j, s, next((x for x in s["live"] if identity(x) == identity(unknown["identity"])), None)) for j, s in enumerate(samples)]
    worker_samples = [(j, s, x) for j, s, x in worker_samples if x is not None]
    last_worker = worker_samples[-1][2]
    return_deadline = go["sent_monotonic_ns"] + round(p["return_timeout_seconds"] * 1e9)
    slot_deadline = min(opened + p["maximum_active_ns"], unknown["launch_monotonic_ns"] + round(p["slot_hard_timeout_seconds"] * 1e9))
    gaps = [(r["qpc_start_ns"] - l["qpc_start_ns"]) / 1e9 for l, r in zip(samples, samples[1:])]
    cpus = [(i, s, x) for i, s in enumerate(samples) for x in s["live"] if identity(x) == identity(ledger["coordinator"])]
    changes = [(i, s, x) for j, (i, s, x) in enumerate(cpus) if j == 0 or (x["kernel_cpu_100ns"], x["user_cpu_100ns"]) != (cpus[j-1][2]["kernel_cpu_100ns"], cpus[j-1][2]["user_cpu_100ns"])]
    monitored_identity_counts = collections.Counter(identity(x) for s in samples for x in s["live"])
    sources = []
    for n in ("rank_fair_coordinator.py", "rank_fair_owned_transport.py", "rank_fair_cache_formal_worker.py", "freeze_rank_fair_cache_formal.py", "verify_rank_fair_cache_formal.py"):
        sources.append(check_record(dict(path="rfm_v8/" + n, sha256=p["source_hashes"]["rfm_v8/" + n])))
    assert all(r["matched"] for r in sources)
    windows = []
    for f in sorted(PROTO.glob("*.json")):
        wp = read(f)
        windows.append(dict(window=wp["window"], slots=len(wp["requests"]), protocol=file_record(f),
                            observed_raw_output=(relative(RAW) if wp["window"] == p["window"] else None)))
    after = {relative(f): file_record(f) for f in sorted(RAW.rglob("*")) if f.is_file()}
    assert before == after, "Raw evidence changed during audit"
    report = dict(schema="rfm-formal-interruption-review-1", created_utc=dt.datetime.now(UTC).isoformat(),
                  audit_mode="new_outputs_only_standard_library_cpu_no_gpu_no_flow_solve", flow_solves_run=0,
                  status="interrupted_incomplete_diagnostic_only", speed_claim_supported=False,
                  raw_directory=relative(RAW), frozen_protocol_copy_matches=(RAW / "protocol.json").read_bytes() == (PROTO / "mid_b1_b5.json").read_bytes(),
                  input_evidence_unchanged_during_audit=True, evidence_inventory=list(before.values()),
                  qualification=dict(summary_sha256=qhash, references=qrefs, qualified_rows=78, status=qsummary["status"],
                                     scope="Summary integrity and its two used numerical qualification links reverified; no new 78-row physical rescoring."),
                  registered_windows=windows, counts=dict(mid_intended=50, completed_artifacts=2, attempted_unknown=1, not_attempted=47),
                  completed_rows=completed,
                  unknown_row=dict(index=2, job={k: p["requests"][2][k] for k in ("name", "arm", "seed", "policy", "block", "qualification_index")},
                                   identity=unknown["identity"], last_durable_worker_stage="READY; coordinator GO published; no RETURN, SAVE, trajectory, or terminal",
                                   request_sha256=digest(RAW / "request_002.json"), ready=ready, go=go,
                                   attempted_sidecar=read(RAW / "attempt_002.json"), ledger_status=unknown["status"],
                                   latest_live_sample_shanghai=stamp(latest["end_filetime_100ns"]),
                                   seconds_from_go_to_last_live_sample=(latest["qpc_end_ns"] - go["sent_monotonic_ns"]) / 1e9,
                                   last_kernel_cpu_seconds=last_worker["kernel_cpu_100ns"] / 1e7,
                                   last_user_cpu_seconds=last_worker["user_cpu_100ns"] / 1e7,
                                   run_outcome="unknown; cannot classify as numerical failure or recorded timeout"),
                  not_attempted=[dict(index=i, **{k: p["requests"][i][k] for k in ("name", "arm", "seed", "policy", "block", "qualification_index")}) for i in range(3, 50)],
                  monitor=dict(sample_count=len(samples), first_sample_shanghai=stamp(samples[0]["start_filetime_100ns"]),
                               last_sample_shanghai=stamp(latest["end_filetime_100ns"]), maximum_sample_start_gap_seconds=max(gaps),
                               acquisition_issue_count=sum(len(s["acquisition_issues"]) for s in samples),
                               recorded_terminal_count=sum(len(s["terminals"]) for s in samples),
                               events=[dict(event=e["command"]["event"], serial=e["command"]["serial"], ack_qpc_ns=e["reply"]["ack_qpc_ns"]) for e in events],
                               observed_live_identities=[dict(pid=k[0], birth_filetime_100ns=k[1], samples=v) for k, v in sorted(monitored_identity_counts.items())],
                               last_sample=latest,
                               coordinator_last_recorded_cpu_change=dict(sample_index=changes[-1][0], shanghai=stamp(changes[-1][1]["end_filetime_100ns"]),
                                                                         kernel_cpu_100ns=changes[-1][2]["kernel_cpu_100ns"], user_cpu_100ns=changes[-1][2]["user_cpu_100ns"]),
                               caveat="Sampled process presence and CPU counters cannot locate an executing or blocked call, prove GPU activity, or identify the interruption cause."),
                  deadlines=dict(return_seconds=p["return_timeout_seconds"], slot_seconds=p["slot_hard_timeout_seconds"], active_seconds=p["maximum_active_ns"] / 1e9,
                                 return_deadline_qpc_ns=return_deadline, slot_deadline_qpc_ns=slot_deadline, cohort_deadline_qpc_ns=opened+p["maximum_active_ns"],
                                 live_samples_after_return_deadline=sum(s["qpc_start_ns"] > return_deadline for _, s, _ in worker_samples),
                                 live_samples_after_slot_deadline=sum(s["qpc_start_ns"] > slot_deadline for _, s, _ in worker_samples),
                                 live_samples_after_cohort_deadline=sum(s["qpc_start_ns"] > opened+p["maximum_active_ns"] for _, s, _ in worker_samples),
                                 timeout_or_termination_persisted=False, bounded_termination_proven=False),
                  source_hash_checks=sources,
                  implementation_risks=[dict(file="rfm_v8/rank_fair_owned_transport.py", line=106,
                                             finding="Synchronous TerminateProcess has no enclosing independent call deadline; the 4000 ms wait starts only after it returns.",
                                             epistemic_limit="Static vulnerability, not proof that this call blocked in this run."),
                                        dict(file="rfm_v8/rank_fair_coordinator.py", line=207,
                                             finding="Attempt sidecar is updated only after launch and its finally cleanup return; in-memory failure/termination timestamps can remain unpersisted during interruption."),
                                        dict(file="rfm_v8/rank_fair_coordinator.py", line=196,
                                             finding="The 720 second cohort deadline is enforced by the coordinator, not an independent supervisory kill deadline; it does not independently bound blocking cleanup calls.")],
                  absent_closing_evidence=[n for n in ("summary.json", "isolation.json", "execution_environment_guard.json", "return_002.json", "save_002.json", "trajectory_002.npz") if not (RAW / n).exists()],
                  current_process_observation=dict(source="Separate read-only Win32_Process query during audit", queried_pids=[61060,75120,12208], returned_count=0,
                                                   limit="Absent now is not an exact terminal record or explanation of how/when they ended."),
                  conclusions=["The mid formal window is incomplete and cannot support isolated timing or cache-policy speed claims.",
                               "Rows 0 and 1 preserve exact raw hashes and match their existing qualified numerical states and full probes.",
                               "Row 2 is attempted with unknown outcome; rows 3 through 49 have no persisted launch evidence.",
                               "Last available monitor evidence still shows exact worker, coordinator and monitor alive well past the registered deadlines.",
                               "No persisted timeout, worker terminal, monitor close, isolation result, or closing environment guard exists.",
                               "App restart, driver fault, GPU failure, cancellation and exact blocking call are not established by these records.",
                               "Retain the interrupted window as-is. Never reconstruct its missing closing guard or promote the first two diagnostic times to a complete-window comparison."])
    assert report["frozen_protocol_copy_matches"]
    for name in ("formal_interruption_review.json", "formal_interruption_review.txt"):
        assert not (OUT / name).exists(), "Review output is create-only"
    (OUT / "formal_interruption_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = ["Formal interruption review — 2026-11-10", "", "Verdict: interrupted / incomplete / diagnostic only. No speed claim is supported.",
            "Mid window: 50 registered slots; 2 completed artifacts; 1 attempted with unknown outcome; 47 not attempted.",
            "Existing qualification summary: 78 qualified rows, pinned SHA-256 " + qhash,
            "", "Recovered numerical evidence:"]
    for r in completed:
        text.append(f"  Formal {r['index']:03d} -> qualification {r['qualification_index']:03d}: exact states at all 19 formal times, full 577-time probe arrays, configuration, non-time metadata, cache identity, raw SHA and terminal/latches verified. Raw SHA-256 {r['raw']['sha256']}.")
    text += ["", "Last provable progress:",
             "  Slot 002: leg35_k72_q66 / uniform / seed 17 / empty_first. READY and GO exist; no RETURN, SAVE, trajectory or terminal.",
             f"  Last monitor sample: {report['monitor']['last_sample_shanghai']}; GO-to-last-live {report['unknown_row']['seconds_from_go_to_last_live_sample']:.6f} seconds.",
             f"  Worker accumulated CPU: kernel {report['unknown_row']['last_kernel_cpu_seconds']:.6f}s, user {report['unknown_row']['last_user_cpu_seconds']:.6f}s. These do not reveal GPU progress or a blocked stack.",
             f"  Samples: {len(samples)}; max start-to-start gap {max(gaps):.7f}s; 0 acquisition issues; only slots 000/001 have terminal records.",
             "  No summary.json, isolation.json or execution_environment_guard.json. Current read-only process query finds none of the three old PIDs; no historical termination cause follows from that.",
             "", "Deadline risk:",
             "  Registered deadlines were 30s RETURN, 45s slot and 720s cohort. Exact worker identity persists in samples beyond all three; no timeout or termination was durably recorded.",
             "  Synchronous TerminateProcess has no independent call bound; its subsequent 4s wait does not bound that call. Attempt updates are written only after all launch/cleanup returns. This is an implementation risk, not proof of the call or root cause of this interruption.",
             "", "Required interpretation:",
             "  Preserve all original evidence. The two exact numerical recoveries are diagnostic times only. Slot 002 remains unknown. Do not fabricate closing guards, silently retry the failed prospective window, trim to completed rows, or report a cache-policy/method speed gain.",
             "  No app restart, driver fault, GPU failure, cancellation or exact blocked call is established.",
             "", "Audit performed no flow solve, no GPU calls, no modification of original files. All original mid-output file hashes and last-write values were stable across this audit."]
    (OUT / "formal_interruption_review.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], completed_numerical_links=2, missing_terminal_slot=2,
                          go_to_last_sample_seconds=report["unknown_row"]["seconds_from_go_to_last_live_sample"],
                          output=relative(OUT), json_sha256=digest(OUT / "formal_interruption_review.json"))))


if __name__ == "__main__":
    main()
