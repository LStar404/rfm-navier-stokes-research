"""Close the independent CPU-only publication-failure review without modifying frozen evidence."""
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
V8=HERE.parent
WORK=V8.parent


def digest(path):return sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())


def main():
    window=V8/"rank_fair_cache_recovery_formal_v3_output/mid_b1_b5"
    capture,outer=window/"capture",window/"outer"
    plan,request,summary=(read(p) for p in (window/"protocol.json",window/"watchdog_request.json",capture/"summary.json"))
    row=read(capture/"attempt_011.json")
    terminal,pending,state=(read(capture/name) for name in ("outer_slot_terminal_011.json","outer_slot_state.json.tmp","outer_slot_state.json"))
    ledger=read(capture/"identity_ledger.json")
    assert row["status"]=="failed" and row["cleanup_failure_type"]=="PermissionError" and row["exit_code"]==0
    assert row["watchdog"]["terminal_unconfirmed"] is False and terminal==pending and state["phase"]=="active"
    assert ledger["children"][11]["status"]=="monitor_registered"
    actual_source_files=[V8/"rank_fair_outer_watchdog.cs",V8/"rank_fair_isolation_v2/coordinator_v2.py",V8/"formal_timing_coordinator.py"]
    for path in actual_source_files:assert plan["source_hashes"][path.relative_to(WORK).as_posix()]==digest(path)
    raw_verified={name:digest(capture/name)==value for name,value in summary["raw_hashes"].items()}
    assert all(raw_verified.values())
    live=str((capture/"outer_slot_state.json").resolve())
    assert live not in request["inputs_sha256"]
    matrix=read(HERE/"api_matrix_output/summary.json")
    immutable=read(HERE/"immutable_publication_cpu_output/certificate.json")
    candidate=read(HERE/"future_candidate_cpu_output/certificate.json")
    assert immutable["immutable_records"]==512 and immutable["reader"]["errors"]==0
    tested={(r["mode"],r["api"]):r for r in matrix["records"]}
    assert not tested[("native_share_read_write_delete","python_path_replace")]["success"]
    assert not tested[("dotnet_share_read_write_delete","MoveFileExW")]["success"]
    assert tested[("exact_v6_poll","python_path_replace")]["failure"]["winerror"]==5
    replace_poll=tested[("exact_v6_poll","ReplaceFileW")]
    assert replace_poll["successful_replacements"]==512 and replace_poll["reader"]["read_errors"]==776
    files=actual_source_files+[window/"protocol.json",window/"watchdog_request.json",capture/"summary.json",capture/"attempt_011.json",
        capture/"outer_slot_terminal_011.json",capture/"outer_slot_state.json.tmp",capture/"outer_slot_state.json",capture/"identity_ledger.json",
        outer/"watchdog_terminal.json",HERE/"api_matrix_output/summary.json",HERE/"future_candidate_cpu_output/certificate.json",
        HERE/"immutable_publication_cpu_output/certificate.json"]
    report=dict(schema="rfm-v8-live-slot-publication-postfailure-audit-1",status="completed_cpu_only_limited_diagnosis",
        gpu_contexts_created=0,flow_solves_run=0,no_frozen_source_or_raw_edits=True,
        formal_retry_allowed=False,formal_window_accepted=False,speed_claim_supported=False,
        preserved_raw_hashes_verified=len(raw_verified),evidence_sha256={p.relative_to(WORK).as_posix():digest(p) for p in files},
        observed_failure=dict(slot_index=11,solver_return="solved",exact_worker_exit_code=0,complete_npz=True,
            terminal_sidecar_and_temporary_byte_identical=digest(capture/"outer_slot_terminal_011.json")==digest(capture/"outer_slot_state.json.tmp"),
            live_state_phase="active",ledger_phase="monitor_registered",recorded_exception_type="PermissionError",
            recorded_exception_winerror=None,recorded_exception_stack=None,
            localized_operation="Terminal live-state save/replace after terminal sidecar and before ledger/EXIT update; the retained complete temporary strongly identifies replacement, but exact original Win32 error was not captured."),
        static_call_path=dict(slot_read="SlotGuard.Poll -> TryRead -> Read -> one FileStream(Read, ShareReadWrite|Delete)",
            hash_read="Hash uses File.OpenRead for request and immutable input hashes; no Hash call on live slot-state path",
            actual_input_guard_contains_live_slot_state=False,closing_hash_runs_after_owner_wait=True,
            synchronous_read_and_publish_os_calls_remain=True),
        experimentally_confirmed_risk=dict(description="On this host, even a target reader granting Read/Write/Delete sharing conflicts with Path.replace/MoveFileExW replacement.",
            held_native_share7_winerror=5,held_dotnet_share7_winerror=5,exact_frozen_v6_reader_reproduction_winerror=5,
            historical_slot011_lock_owner_identified=False,historical_slot011_root_cause="not established"),
        replacement_api_matrix=matrix["records"],
        replacefile_future_candidate=dict(not_integrated=True,cpu_certificate_status=candidate["status"],
            matrix_write_successes=512,matrix_read_ioexceptions=776,direct_production_deployment_supported=False,
            limitations="ReplaceFileW can expose transient read errors and documented partially modified failure states; unsupported WRITE_THROUGH and ACL-ignore flags were not used. Any failure remains unconfirmed, with no retries or next slot."),
        minimal_future_direction=dict(description="Retain immutable per-slot active and terminal records as the authoritative publication; publish complete temporaries to previously unused names, and avoid replacing a shared live-state target.",
            targeted_test_records=512,slot_pairs=256,reader_exact_frozen_v6=True,reader_errors=0,
            existing_name_overwrite_rejected=True,production_integration=False,T72_qualified=False,
            future_requirements="An integrated watcher would still need exact slot identity/token/deadline and terminal audits, fail-closed behavior and fresh qualification. No such integration or GPU run occurred here."),
        discarded_test_assumption="The first targeted fixture assumed ShareDelete would make Path.replace succeed; it failed and is retained under fixtures/ and first_fixture_source_snapshot/. Its unproduced audit_summary is not a passing certificate.",
        official_reference_urls=["https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw",
            "https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew"],
        local_review_source_sha256={p.name:digest(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py",".cs",".exe")})
    with (HERE/"independent_diagnosis.json").open("x",encoding="utf-8") as handle:json.dump(report,handle,ensure_ascii=False,indent=2);handle.write("\n")
    text="""CPU-only 独立诊断：活跃槽状态发布兼容性\n\n结论：发现可复现的文件发布组合风险，但没有证明正式槽011当时的具体锁持有者或底层根因。不能用本次工程诊断恢复失败计时资格，也没有再次运行GPU。\n\n正式证据：槽011有solved RETURN、完整NPZ和精确exit0；独立terminal记录与遗留.tmp逐字节一致，live state仍为active，ledger仍monitor_registered。失败位于终态live-state发布之后续未执行的边界；原日志只存PermissionError，未存Win32错误码、调用栈或锁持有者。\n\n静态链：v6 SlotGuard只通过一个ShareReadWrite|Delete FileStream读取live state。Hash确实使用File.OpenRead，但目标是request与不可变inputs；实际inputs不含本次live-state文件，closing hash也在owner结束后。因此没有‘SlotGuard随后又用无shareDelete的Hash读state’这条调用链。\n\nCPU交叉实验：没有读者时Path.replace、MoveFileExW、ReplaceFileW均成功。持有native share7或.NET ShareReadWrite|Delete目标句柄时，前两者均WinError5，旧目标与临时文件保留；ReplaceFileW成功。直接反射调用冻结v6 Read与原Python替换并发，也重现WinError5。这证明shareDelete本身不是该API组合在本机可靠替换的充分条件；不等同于追溯证明正式运行由同一读者触发。\n\nReplaceFileW仅是未集成对照：并发512次写均成功，但冻结读者产生776次短暂IOException。官方文档还列出可能已部分改名/改元数据的失败状态，WRITE_THROUGH标志不受支持。不能将其直接当成原子事务或稳定生产替代；未修改ACL、未加入重试、未启动后续GPU。\n\n更小的未来方向：直接使用每槽不可变active/terminal文件，完整临时文件发布到此前不存在的名称，避免覆盖共享live-state目标。针对性测试发布512个完整JSON（256对active/terminal），冻结v6 Read并发逐序校验，512全部通过、0读取异常；故意覆写既有名称被拒绝，原512份哈希不变。这仅验证发布原语，未集成新watchdog，也不是T72资格。\n\n所有冻结production源和原始计时证据保留。本目录含实际Win32错误记录、失败的早期测试、14格API矩阵、未集成候选及不可变发布CPU证书。未来首先保留失败operation/errno/winerror/filename信息，仍须完整身份、截止、终态和独立数值隔离检查。\n\n官方API参考：\nhttps://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw\nhttps://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew\n"""
    with (HERE/"independent_diagnosis.txt").open("x",encoding="utf-8") as handle:handle.write(text)
    print(json.dumps(dict(status=report["status"],preserved_raw_hashes_verified=len(raw_verified),
                         diagnosis_sha256=digest(HERE/"independent_diagnosis.json"),gpu_contexts_created=0)))


if __name__=="__main__":main()
