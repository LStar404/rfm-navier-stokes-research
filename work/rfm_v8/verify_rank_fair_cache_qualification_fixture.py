"""CPU-only fixture for private-context qualification admission and evidence."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import rank_fair_extended_qualification as q


def rejected(action):
    try:
        action()
    except (ValueError,FileNotFoundError):
        return True
    raise AssertionError("Invalid qualification input passed")


def check():
    q.shared.pin(10)
    plan = q.inventory.draft()
    cold,warm = plan["slots"][:2]
    q.require((cold["policy"],warm["policy"]) == q.inventory.POLICIES and
              cold["identity_index"] == warm["identity_index"],"Fixture policies are not matched")
    artifact = q.shared.bound(cold["ptx_artifact"])
    with TemporaryDirectory(dir=q.HERE,prefix="rank_fair_cache_fixture_") as temporary:
        root = Path(temporary).resolve()
        q.require(root.is_relative_to(q.HERE),"Fixture escaped its local workspace")
        empty,strict = root/"empty",root/"strict"
        empty.mkdir()
        strict.mkdir()
        local = empty/"explicit_cache"
        c = q.gpu_worker.select(cold,plan["compiler_identity"],local)
        w = q.gpu_worker.select(warm,plan["compiler_identity"],strict/"explicit_cache")
        q.require(c.keywords["artifact_directory"] == local and w.keywords["artifact_directory"] == artifact.parent and
                  not local.exists() and not (strict/"explicit_cache").exists(),"Job path selection touched the namespace")
        local.mkdir()
        (local/artifact.name).write_bytes(artifact.read_bytes())
        meta = dict(ptx_cache_policy="empty_first",ptx_cache_observed_mode="miss",ptx_cache_key=cold["ptx_cache_key"],
                    ptx_artifact_sha256=cold["ptx_artifact"]["sha256"],ptx_sha256=cold["ptx_sha256"],cuda_context_mode=q.inventory.RUNTIME,
                    cuda_driver_api=13020,private_context_finally_unbound=True,nonlinear_engine="warp_cufft",observation_reduction="warp",
                    ptx_nvrtc_seconds=.37,seconds=9.)
        other = dict(meta,ptx_cache_policy="strict_hit",ptx_cache_observed_mode="hit",ptx_nvrtc_seconds=0.)
        q.require(q.cache_evidence(empty,cold,{"metadata":meta})["sha256"] == cold["ptx_artifact"]["sha256"] and
                  q.cache_evidence(strict,warm,{"metadata":other}) is None,"Valid fixture cache evidence failed")
        bad_path = deepcopy(warm)
        bad_path["ptx_artifact"]["path"] = "../"+artifact.name
        owner = {"pid":241,"birth_filetime_100ns":315}
        phases = root/"phases.jsonl"
        entries = [dict(event=event,index=0,qpc_ns=i,identity=owner) for i,event in enumerate(
                   ("python_ready","solver_enter","solver_return_after_context_cleanup"),1)]
        phases.write_text("\n".join(json.dumps(row) for row in entries)+"\n",encoding="ascii")
        q.verify_phases(phases,0,{"identity":owner})
        rejects = dict(draft_execution=rejected(lambda:q.verify_registered_plan(plan)),
            compile_on_strict=rejected(lambda:q.cache_evidence(strict,warm,{"metadata":dict(other,ptx_nvrtc_seconds=.001)})),
            wrong_ptx=rejected(lambda:q.cache_evidence(empty,cold,{"metadata":dict(meta,ptx_sha256="forged")})),
            path_escape=rejected(lambda:q.gpu_worker.select(bad_path,plan["compiler_identity"],local)),
            wrong_worker=rejected(lambda:q.verify_phases(phases,0,{"identity":dict(owner,birth_filetime_100ns=316)})))
    return dict(status="passed",gpu_contexts_created=0,flow_solves_run=0,expected_rejections=rejects,
                qualification_source_sha256=q.shared.sha(q.SOURCE),worker_source_sha256=q.shared.sha(Path(q.gpu_worker.__file__)),
                window_sizes=[w["slots"] for w in plan["windows"]],unique_policy_slots=len(plan["slots"]),
                stock_keys=sorted({r["ptx_cache_key"] for r in plan["slots"]}))


if __name__ == "__main__":
    result = check()
    q.T.save(q.HERE/"rank_fair_cache_qualification_fixture.json",result)
    print(json.dumps(result))
