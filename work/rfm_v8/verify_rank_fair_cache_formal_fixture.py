"""Replay the exact request and aggregate admissions with ordinary CPU data."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import freeze_rank_fair_cache_formal as formal
import verify_rank_fair_cache_formal as verify
from rank_fair_cache_formal_worker import CACHE_FIELDS
from rank_fair_owned_transport import require


def check(qualification):
    formal.qual.shared.pin(10)
    qp,controls,_,proof = formal.load_qualification(qualification)
    bound = formal.coordinator()
    old = formal.timing.run
    require(bound.__code__ is old.__code__ and bound.__kwdefaults__ == old.__kwdefaults__ and
            {key for key in old.__globals__ if bound.__globals__[key] is not old.__globals__[key]} == {"SOLVER_WORKER"} and
            bound.__globals__["SOLVER_WORKER"] == formal.WORKER,"Bounded monitor transport changed beyond the entry point")
    fixtures = formal.rank.read(formal.rank.SMOKE)
    require(fixtures["status"] == "passed" and fixtures["time_solves_run"] == 0 and len(fixtures["test_cases"]) == 4,
            "Original exact-owned handshake fixture differs")
    reports = []
    for i,w in enumerate(qp["windows"]):
        jobs = formal.requests_for(qp,w)
        require(len(jobs) == w["slots"] and all(len(j["solve_args"]["snapshots"]) == 19 and j["policy"] == controls[j["qualification_index"]]["job"]["policy"]
                     for j in jobs),"Window differs from exact numerical qualification")
        plan = dict(window=w["name"],window_spec=w,scale=w["scale"],qualification=proof,environment={},source_hashes={})
        rows = []
        for j in jobs:
            clock = 2. if j["policy"] == "empty_first" else 1.
            rows.append(dict(qualified=True,job=j,clocks=dict(solver_seconds=clock,fresh_process_return_seconds=clock+.25),phases={"seconds":clock}))
        reports.append((plan,dict(window=w["name"],eligible=True,open_qpc_ns=100+100*i,close_qpc_ns=180+100*i,rows=rows)))
    clean = verify.summarize(reports)
    require(clean["isolated_speed_claim_supported"] and clean["nonoverlapping_registered_order"] and
            all(pair["supports_both_clock_practical_gain"] for scope in clean["scales"].values() for pair in scope["pairs"]
                if pair["kind"] == "explicit_cache_policy"),"Clean full blocks did not yield the two independent ratios")
    dirty = deepcopy(reports)
    dirty[2][1]["eligible"] = False
    overlap = deepcopy(reports)
    overlap[1][1]["open_qpc_ns"] = 150
    no_pooling = all(not result["isolated_speed_claim_supported"] and result["scales"] == {} for result in
                     (verify.summarize(dirty),verify.summarize(overlap)))
    require(no_pooling,"Failed or overlapping windows leaked into combined rankings")
    cold,hit = qp["slots"][:2]
    file = formal.qual.shared.bound(cold["ptx_artifact"])
    with TemporaryDirectory(dir=formal.HERE,prefix="rfm_formal_fixture_") as temporary:
        output = Path(temporary).resolve()
        require(output.is_relative_to(formal.HERE),"Formal fixture output escaped its workspace")
        cp = dict(cold,index=0)
        hp = dict(hit,index=1)
        directory = output/"explicit_cache_000"
        directory.mkdir()
        (directory/file.name).write_bytes(file.read_bytes())
        meta = dict(ptx_cache_policy="empty_first",ptx_cache_observed_mode="miss",ptx_cache_key=cold["ptx_cache_key"],
            ptx_artifact_sha256=cold["ptx_artifact"]["sha256"],ptx_sha256=cold["ptx_sha256"],cuda_context_mode=cold["runtime_id"],
            cuda_driver_api=13020,private_context_finally_unbound=True,ptx_nvrtc_seconds=.37,seconds=1.)
        warm = dict(meta,ptx_cache_policy="strict_hit",ptx_cache_observed_mode="hit",ptx_nvrtc_seconds=0.)
        a = verify.verify_cache({"marker":{"cache":{k:meta[k] for k in CACHE_FIELDS}}},cp,meta,output)
        b = verify.verify_cache({"marker":{"cache":{k:warm[k] for k in CACHE_FIELDS}}},hp,warm,output)
        require(a["sha256"] == cp["ptx_artifact"]["sha256"] and b is None,"Local compiled artifact admission differs")
    return dict(status="passed",gpu_contexts_created=0,solves_run=0,qualification_sha256=proof["summary"]["sha256"],
                windows=[w["slots"] for w in qp["windows"]],coordinator_code_reused=True,failed_or_overlapping_window_refused=no_pooling,
                formal_source_sha256=formal.T.digest(formal.SOURCE),worker_sha256=formal.T.digest(formal.WORKER),
                verifier_sha256=formal.T.digest(formal.VERIFIER))


if __name__ == "__main__":
    import sys

    require(len(sys.argv) == 2,"Expected the completed exact qualification directory")
    result = check(Path(sys.argv[1]).resolve())
    formal.T.save(formal.FIXTURE,result)
    print(json.dumps(result))
