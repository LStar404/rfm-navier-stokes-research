"""Fixed numerical and fresh import checks for the v8 solver-only modules."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
FIXED = HERE/"galerkin_minimal_fixed_audit.json"
IMPORT_OUT = HERE/"galerkin_minimal_import_audit_output"
IMPORTS = {
    "old_prepared":("galerkin_cuda_prepared",),
    "lean_prepared":("galerkin_cuda_prepared_minimal",),
    "old_uniform_recurrence":("galerkin_cuda_native","basis_recurrence_model"),
    "lean_uniform_recurrence":("galerkin_cuda_native","galerkin_minimal_model"),
    "fd":("fd_cuda_native",),
}
EXCLUDED = (
    "basis_recurrence","cpu_formal","cpu_implicit_stride","cpu_precontracted",
    "cpu_precontracted_native","cpu_profile","cpu_sbdf2_switch","diagnostic_wake",
    "fd_full","prepared_formal_eligibility","prepared_qualification",
    "scipy.signal","scipy.stats","matplotlib","reportlab",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x",encoding="utf-8") as out:
        json.dump(value,out,indent=2,allow_nan=False)
        out.write("\n")


def fixed():
    from galerkin_minimal_model import native_model, np
    import galerkin_cuda_native as native
    import galerkin_cuda_prepared as old_prepared
    import galerkin_cuda_prepared_minimal as lean_prepared
    import galerkin_cuda_variable as old_switch
    import galerkin_cuda_variable_minimal as lean_switch
    import galerkin_phase_shared as old_shared
    import galerkin_phase_shared_minimal as lean_shared
    from cpu_formal import config
    import cpu_sbdf2_switch as old_schedule
    import sbdf2_schedule_minimal as lean_schedule

    archive = V7/"prepared_qualification_output"/"protocol.json"
    plan = json.loads(archive.read_text(encoding="utf-8"))
    source = HERE/"galerkin_minimal_source_audit.json"
    generation = json.loads(source.read_text(encoding="utf-8"))
    for name,sha in generation["original_sha256"].items():
        if digest(V7/name) != sha:
            raise ValueError(f"Frozen v7 input differs: {name}")
    for name,sha in generation["generated_sha256"].items():
        if digest(HERE/name) != sha:
            raise ValueError(f"Generated v8 input differs: {name}")
    if (lean_switch.V7 != V7 or lean_prepared.PREPARATIONS["combined"][0] is not
            old_prepared.PREPARATIONS["combined"][0]):
        raise ValueError("CUDA source or bundle runtime identity differs")
    original_native = native.BufferedGalerkin
    with native_model():
        if native.BufferedGalerkin is not lean_switch.RecurrenceGalerkin:
            raise ValueError("Uniform solver selected an unexpected numerical model")
    if native.BufferedGalerkin is not original_native:
        raise ValueError("Uniform solver context did not restore its original model")

    def equal(a,b):
        return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

    row_checks = []
    names = ("s","r","transform","samples","mean_transform","mean_samples","M0","M1","mean_mass",
             "M_lift","bdf_lift","start","implicit","mean_inverse","test","test_s","mean_test",
             "time_jets","mean_vort_samples","lift_modes")
    for scale in ("mid","high"):
        schedule = plan["schedules"][scale]
        controls = [r for r in plan["controls"] if r["resolution"] == scale]
        for mode in ("constant","switched"):
            h0 = schedule["startup_rate"]
            h1 = h0 if mode == "constant" else schedule["late_rate"]
            times = plan["template"]["physical_times"] if mode == "constant" else schedule["checkpoints"]
            old_grid,new_grid = (module.step_schedule(startup_rate=h0,late_rate=h1,snapshots=times)
                                 for module in (old_schedule,lean_schedule))
            scheduling = (old_grid.switch_step == new_grid.switch_step and
                all(equal(getattr(old_grid,k),getattr(new_grid,k)) for k in (
                    "times","snapshot_times","snapshot_steps","probe_times","probe_steps")))
            if not scheduling:
                raise ArithmeticError("Lean absolute time indices differ")
            for row in controls:
                name,seed = row["name"],row["seed"]
                _,settings = config(plan["specs"][name],plan["solve_protocol"],seed)
                builds = {}
                for arm,before,after in (("shared",old_shared.phase_operators_shared,lean_shared.phase_operators_shared),
                                          ("independent",old_switch.phase_operators,lean_switch.phase_operators)):
                    old,new = (fn(h0,h1,settings) for fn in (before,after))
                    om,oo,op,oc = old
                    nm,no,npack,nc = new
                    if om.metadata != nm.metadata or oc != nc:
                        raise ArithmeticError("Lean model metadata or bridge weights differ")
                    pairs = [(getattr(om,k),getattr(nm,k)) for k in names]
                    pairs.extend(zip(oo,no,strict=True))
                    pairs.extend(pair for a,b in zip(op,npack,strict=True) for pair in zip(a,b,strict=True))
                    rng = np.random.default_rng(36720)
                    state = rng.normal(size=om.rhs_shape)+1j*rng.normal(size=om.rhs_shape)
                    state[:,0] = state[:,0].real
                    state[om.mean_width:,0] = 0
                    state *= .0002
                    amp = -.05-.41j
                    pairs.extend(zip(om.terms(state,amp),nm.terms(state,amp),strict=True))
                    pairs.extend((om.advance(state.copy(),amp,first),nm.advance(state.copy(),amp,first))
                                 for first in (True,False))
                    builds[arm] = dict(comparisons=len(pairs),byte_exact=sum(equal(a,b) for a,b in pairs))
                row_checks.append(dict(scale=scale,name=name,seed=seed,rate_mode=mode,schedule_exact=scheduling,
                    startup_rate=h0,late_rate=h1,builders=builds))
    passed = len(row_checks) == 24 and all(a["byte_exact"]==a["comparisons"]
                                         for r in row_checks for a in r["builders"].values())
    save(FIXED,dict(status="passed" if passed else "failed",archive_protocol_sha256=digest(archive),
                    source_audit_sha256=digest(source),runner_sha256=digest(Path(__file__)),rows=row_checks,
                    scope="Fixed NumPy FP64 arrays and CPU probe inputs only. No full trajectory or CUDA context."))
    print("fixed",passed,len(row_checks),sum(a["comparisons"] for r in row_checks for a in r["builders"].values()),flush=True)
    return passed


def worker(label):
    sys.path.insert(0,str(V7))
    before = perf_counter()
    for module in IMPORTS[label]:
        __import__(module)
    import_seconds = perf_counter()-before
    print(json.dumps(dict(label=label,import_seconds=import_seconds,modules=sorted(sys.modules))),flush=True)


def import_audit():
    if IMPORT_OUT.exists():
        raise FileExistsError(IMPORT_OUT)
    IMPORT_OUT.mkdir()
    order = random.Random(882393)
    env = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    rows = []
    for block in range(3):
        labels = list(IMPORTS)
        order.shuffle(labels)
        for label in labels:
            start = perf_counter()
            proc = subprocess.run([sys.executable,"-B","-X","importtime",str(Path(__file__)),"--worker",label],
                                  env=env,capture_output=True,timeout=60,check=True)
            elapsed = perf_counter()-start
            path = IMPORT_OUT/f"{block}_{label}.importtime.txt"
            path.write_bytes(proc.stderr)
            raw = json.loads(proc.stdout)
            extras = sorted(set(raw["modules"])&set(EXCLUDED))
            rows.append(dict(block=block,label=label,excluded_modules_loaded=extras,
                             total_modules=len(raw["modules"]),import_seconds=raw["import_seconds"],
                             fresh_seconds=elapsed,importtime=path.name,importtime_sha256=digest(path)))
            print(block,label,round(raw["import_seconds"],4),round(elapsed,4),extras,flush=True)
    medians = {label:{k:median(r[k] for r in rows if r["label"]==label)
                      for k in ("import_seconds","fresh_seconds")} for label in IMPORTS}
    passed = all(not r["excluded_modules_loaded"] for r in rows if r["label"].startswith("lean_"))
    save(IMPORT_OUT/"audit.json",dict(status="passed" if passed else "failed",runner_sha256=digest(Path(__file__)),
         source_audit_sha256=digest(HERE/"galerkin_minimal_source_audit.json"),order_seed=882393,
         modules=IMPORTS,excluded_modules=EXCLUDED,rows=rows,medians=medians,
         scope="Randomized 3-block import-only diagnostics with -X importtime. No solver or CUDA context."))
    return passed


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixed",action="store_true")
    group.add_argument("--imports",action="store_true")
    group.add_argument("--worker",choices=tuple(IMPORTS))
    args = cli.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        sys.exit(0 if (fixed() if args.fixed else import_audit()) else 1)
