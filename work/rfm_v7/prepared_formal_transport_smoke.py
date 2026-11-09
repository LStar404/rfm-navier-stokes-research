"""Exercise actual prepared worker transport with two tiny synthetic arrays."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import prepared_formal_timing as timing


base, case, np = timing.base, timing.case, timing.np
require = timing.require


def synthetic(job, _protocol, _config):
    if job["seed"] == 18:
        raise ArithmeticError("Synthetic failed solve")
    fields = ("transform","mean_transform","samples","mean_samples","M0","M1","M_lift",
              "mean_mass","test","test_s","mean_test","r","lift","k2")
    model = SimpleNamespace(**{k:np.ones(1) for k in fields})
    vector = np.array([1.,2.])
    return case.Solution(model,vector,vector,vector,vector,vector,{"synthetic":True})


def verify():
    cpu = base.pin()
    for key in timing.previous.THREADS:
        os.environ[key] = "1"
    with TemporaryDirectory(prefix="prepared_transport_smoke_") as tmp:
        output = Path(tmp)
        files = timing.paths(output)
        jobs = [dict(block=1,name="synthetic",arm="fd",seed=k) for k in range(17,72)]
        p = dict(status="frozen",scale="mid",schedule=jobs,fd_controls={"synthetic":{"config":{"synthetic":True}}})
        base.save(files["protocol"],p)
        sha = base.digest(files["protocol"])
        token = "prepared-synthetic-transport"
        os.environ.update({timing.HASH:sha,timing.TOKEN:token,timing.ADMISSION:str(files["coordinator_admission"])})
        here = timing.current_process_identity()
        a = dict(schema=timing.SCHEMA,status="admitted",protocol_sha256=sha,run_token=token,coordinator=here)
        base.save(files["coordinator_admission"],a)
        timing.admission(files,sha,coordinator=True)
        ledger = dict(a,status="running",children=[])
        base.save(files["identity_ledger"],ledger)
        attempts = []
        for i in range(3):
            row = dict(index=i)
            try:
                artifact = timing.child_attempt(i,row,p,output,files,ledger,a,cpu,Path(__file__).resolve())
            except RuntimeError:
                require(i == 1 and row["failure_type"] == "ArithmeticError" and "artifact" not in row,
                        "Failed synthetic slot did not preserve its failure")
            else:
                require(i != 1 and artifact.is_file() and row["child_exit_code"] == 0, "Synthetic artifact was not saved")
                with np.load(artifact,allow_pickle=False) as raw:
                    require(np.array_equal(raw["states"],[1.,2.]), "Synthetic transport changed the artifact")
            slot = ledger["children"][i]
            timing.clock_order(row,slot,token,cpu)
            require(slot["exit_filetime_100ns"] > slot["birth_filetime_100ns"] and
                    (i == 0 or ledger["children"][i-1]["exit_monotonic_ns"] < slot["launch_monotonic_ns"]),
                    "Synthetic slot overlap or missing terminal exact identity")
            attempts.append(dict(index=i,expected_success=i!=1,ready_return_identity_exact=True,
                                 exited_before_next_slot=True,exit_code=row["child_exit_code"],
                                 saved_after_return="artifact" in row,clock_order_passed=True))
    report = dict(status="passed",scope="Three Windows transport workers with synthetic arrays. No fluid solver or CUDA.",
                  slots=attempts,sha256={name:base.digest(timing.base.HERE/name) for name in
                                       ("prepared_formal_timing.py","prepared_formal_transport_smoke.py",
                                        "prepared_process_identity.py")})
    base.save(timing.base.HERE/"prepared_formal_transport_smoke.json",report)
    print("PASS",len(attempts),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",type=int)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--scale",choices=("mid",))
    args = parser.parse_args()
    if args.child is None:
        verify()
    else:
        raise SystemExit(timing.child(args.child,args.output_dir,args.scale,solver=synthetic))
