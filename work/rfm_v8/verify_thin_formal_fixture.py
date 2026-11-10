"""No-fluid fixture for thin formal verification and claim gates."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

import verify_thin_formal as check


T=check.T
HERE=Path(__file__).resolve().parent


def trajectory_fixture(output):
    times=check.read(check.frozen.DESIGN)["physical_times"]
    config={"m":3,"k":2}
    metadata={"steps":8192,"seconds":1.,"march_seconds":.5}
    fields=dict(states=np.ones((19,2,3)),times=np.array(times),probe_times=np.arange(577)/8,
                transverse=np.zeros((577,3)),vorticity=np.zeros((577,3)))
    request=dict(config=config,solve_args=dict(snapshots=times))
    control=dict(index=0,config=config,solve_args=request["solve_args"],spatial_sha256="a"*64,
                 non_time_metadata_sha256=check.frozen.canonical_sha({"steps":8192}),
                 arrays={key:check.frozen.array_identity(value) for key,value in fields.items()})

    def write(name,values,meta=metadata):
        path=output/name
        with path.open("xb") as handle:
            np.savez(handle,**values,config=T.raw_json(config).decode().strip(),metadata=T.raw_json(meta).decode().strip())
        return path

    exact=write("exact.npz",fields,dict(metadata,seconds=2.))
    def unexpected(*args):
        raise AssertionError("Exact trajectory unexpectedly triggered full scoring")
    result,_=check.compare_trajectory(exact,request,control,"a"*64,unexpected)
    T.require(result["qualified"] and result["gate"]=="inherited_bitwise_thin_control","Exact inheritance failed")
    changed={key:value.copy() for key,value in fields.items()}
    changed["states"][0,0,0]=np.nextafter(1.,2.)
    path=write("one_ulp.npz",changed)
    calls=[]
    def fallback(candidate,reference,signature):
        calls.append((candidate,reference,signature))
        return {"qualified":False,"scope":"Synthetic fallback test"}
    result,_=check.compare_trajectory(path,request,control,"a"*64,fallback)
    T.require(calls==[(path,control,"a"*64)] and not result["qualified"] and not result["arrays"]["states"]["byte_exact"] and
              result["gate"]=="independent_v7_full_norm_rescore","One ULP did not take the original-norm fallback")
    wrong=write("wrong_work.npz",fields,dict(metadata,steps=8191))
    for candidate,signature in ((wrong,"a"*64),(exact,"b"*64)):
        try:
            check.compare_trajectory(candidate,request,control,signature,unexpected)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong work inventory or spatial identity was accepted")
    return dict(bitwise_inheritance=True,ulp_enters_original_norm_callback=True,wrong_work_and_model_rejected=True)


def paired_fixture():
    design=check.read(check.frozen.DESIGN)
    a=["rfm","combined","selected"]
    b=["rfm","combined","minimal"]
    plan=dict(groups=[a,b],paired=[dict(kind="import_only",baseline=a,candidate=b)],design=design)
    rows=[dict(job=dict(name=key[0],arm=key[1],imports=key[2],block=block),
               clocks=dict(solver_seconds=1.,fresh_process_return_seconds=fresh),qualified=True)
          for block in range(1,6) for key,fresh in ((a,1.2),(b,1.1))]
    _,rank=check.paired_statistics(plan,rows,True)
    T.require(rank[0]["startup_only_practical_improvement"] and not rank[0]["overall_two_clock_practical_improvement"],
              "Fresh-only improvement was confused with two-clock improvement")
    _,diagnostic=check.paired_statistics(plan,rows,False)
    T.require(diagnostic[0]["complete_five_pairs"] and not diagnostic[0]["startup_only_practical_improvement"] and
              not any(c["isolated_practical_improvement_supported"] for c in diagnostic[0]["clocks"].values()),
              "Unisolated timings were promoted")
    rows[2]["qualified"]=False
    _,failed=check.paired_statistics(plan,rows,True)
    T.require(not failed[0]["complete_five_pairs"] and all(value is None for value in failed[0]["clocks"].values()),
              "A failed slot was filtered from a paired statistic")
    for incomplete in ([],rows[:1],rows[::2]):
        _,pairs=check.paired_statistics(plan,incomplete,False)
        T.require(not pairs[0]["complete_five_pairs"] and all(value is None for value in pairs[0]["clocks"].values()),
                  "A rejected preflight or partial deadline produced a partial paired ratio")
    return dict(fresh_only_rule=True,diagnostic_claim_suppressed=True,no_slot_filtering=True,
                empty_preflight_and_partial_deadline_safe=True)


def transport_fixture():
    output=HERE/"formal_transport_smoke_final_output"/"cadence_1000ms"
    row=check.read(output/"runs.json")[0]
    request=check.read(output/"request_000.json")
    slot=check.read(output/"identity_ledger.json")["children"][0]
    artifact=check.verify_slot(output,row,request,slot)
    T.require(artifact.name==request["trajectory"],"Raw fixture trajectory was not found")
    broken=deepcopy(row)
    broken["go_sent_monotonic_ns"]=broken["return_received_monotonic_ns"]
    try:
        check.verify_slot(output,broken,request,slot)
    except ValueError:
        pass
    else:
        raise AssertionError("Impossible solver clock passed transport audit")
    return dict(actual_synthetic_worker_log_verified=True,impossible_go_to_return_rejected=True)


def verify(output):
    output.mkdir()
    result=dict(status="passed",scope="Independent no-fluid fixtures. The divergence fixture exercises dispatch, not a flow solve.",
                trajectory=trajectory_fixture(output),paired=paired_fixture(),transport=transport_fixture(),
                source_sha256={name:T.digest(HERE/name) for name in
                               ("freeze_thin_formal.py","verify_thin_formal.py","verify_thin_formal_fixture.py")})
    T.save(output/"certificate.json",result)
    print("passed")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    verify(cli.parse_args().output)
