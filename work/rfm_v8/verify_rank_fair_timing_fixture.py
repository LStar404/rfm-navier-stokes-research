"""CPU-only replay of the bounded transport verifier and the planned arm inventory."""

import argparse
from copy import deepcopy
from pathlib import Path

import verify_rank_fair_timing as check
from rank_fair_owned_transport import SECOND, read, require


HERE,T,frozen = check.timing.HERE,check.T,check.frozen
SMOKE = HERE/"rank_fair_transport_smoke_v3_output"


def must_reject(function):
    try:
        function()
    except (ValueError, FileNotFoundError):
        return True
    raise AssertionError("Independent checker admitted deliberately incomplete evidence")


def replay():
    output = SMOKE/"success"
    plan,ledger,rows = (read(output/name) for name in ("protocol.json","identity_ledger.json","runs.json"))
    opened = read(output/"summary.json")["open_qpc_ns"]
    latencies = []
    for i,(row,request,entry) in enumerate(zip(rows,plan["requests"],ledger["children"],strict=True)):
        check.verify_transport(output,row,request,entry,ledger,plan,opened)
        latencies.append((row["return_received_monotonic_ns"]-row["marker"]["published_monotonic_ns"])/SECOND)
    row,request,entry = rows[0],plan["requests"][0],ledger["children"][0]
    bad_clock = dict(row,solver_seconds=row["solver_seconds"]+.5)
    bad_deadline = dict(row,exit_wait_completed_monotonic_ns=row["hard_deadline_ns"]+1)
    bad_owner = deepcopy(row)
    bad_owner["terminal_identity"]["birth_filetime_100ns"] += 1
    missing = [dict(grid="gauss123",time=0.5,velocity=0.001,vorticity=0.001,near_velocity=0.001,near_vorticity=0.001)]
    rejects = dict(clock=must_reject(lambda:check.verify_transport(output,bad_clock,request,entry,ledger,plan,opened)),
                   late_terminal=must_reject(lambda:check.verify_transport(output,bad_deadline,request,entry,ledger,plan,opened)),
                   reused_pid=must_reject(lambda:check.verify_transport(output,bad_owner,request,entry,ledger,plan,opened)),
                   missing_norm_cells=must_reject(lambda:frozen.validate_table(missing,[.5],{},.005)),
                   no_exact_qualification=must_reject(lambda:frozen.load_controls(HERE/"rank_fair_not_yet_qualified")))
    q = frozen.qualify.build()
    frozen.qualify.shared.validate(q)
    controls = [dict(index=i,job=job) for i,job in enumerate(q["slots"])]
    scopes = {}
    for scale,count,groups_expected in (("mid",25,5),("high",40,8)):
        jobs,slots,groups,pairs = frozen.design_jobs(scale,controls)
        require(len(jobs) == len(slots) == count and len(groups) == groups_expected and
                all(job["engine"] == "warp_cufft" and job["imports"] == "minimal" and len(job["solve_args"]["snapshots"]) == 19 for job in jobs),
                "Registered stock schedule differs")
        for block,seed in enumerate(frozen.qualify.shared.RFM_SEEDS,1):
            members = [job for job in jobs if job["block"] == block]
            require(len(members) == groups_expected and len({(job["name"],job["arm"]) for job in members}) == groups_expected and
                    {job["seed"] for job in members if job["family"] == "rfm"} == {seed} and
                    {job["seed"] for job in members if job["family"] != "rfm"} == {17} and
                    any(job["name"] == "fd176_k80" and job["solve_args"]["cuda_linear_solve"] == "dense" for job in members),
                    "Block lost its paired seed or cheapest qualified FD control")
        scopes[scale] = dict(slots=count,groups=groups,paired=len(pairs),unique_qualified_slots_required=len(set(slots)))
    closures = frozen.discover_imports()
    return dict(status="passed",time_solves_run=0,gpu_contexts_created=0,replayed_successful_slots=len(rows),
                return_latch_receive_latency_seconds=latencies,expected_rejections=rejects,scopes=scopes,
                qualification_unique_controls=len(controls),qualification_sources=len(q["sources_sha256"]),
                import_closure_counts={arm:len(files) for arm,files in closures.items()},
                transport_certificate_sha256=T.digest(SMOKE/"certificate.json"),verifier_source_sha256=T.digest(Path(check.__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(),"Preserve prior fixture evidence")
    T.save(args.output,replay())
