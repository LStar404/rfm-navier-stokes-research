"""Predeclared CPU-only rank reduction screen for high-scale RFM seed 7331."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SOURCE = Path(__file__).resolve()
THIN = HERE/"thin_t72_qualification_output"/"protocol.json"
REFERENCE = WORK/"rfm_v7"/"switch_qualification_output"/"protocol.json"
OUT = HERE/"rfm_rank_high_screen_paired_output"
FAILED = HERE/"rfm_rank_high_screen_output"/"t16"
THREADS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS")
MS = (35,37,39)
SEED = 7331
CPU = (10,12)
GATE = .005


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def record(path):
    return dict(path=path.relative_to(WORK).as_posix(),sha256=digest(path))


def save(path,value):
    with path.open("x",encoding="utf8",newline="\n") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def local_inputs():
    return dict(os.environ,**{name:"1" for name in THREADS})


def set_affinity(cpu):
    require(cpu in CPU,"Screen is restricted to logical CPUs 10 and 12")
    for name in THREADS:
        os.environ[name] = "1"
    import ctypes as C
    from ctypes import wintypes as W
    kernel = C.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.SetProcessAffinityMask.argtypes = (W.HANDLE,C.c_size_t)
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),1<<cpu):
        raise C.WinError(C.get_last_error())


def trees(value):
    if isinstance(value,dict):
        yield value
        for v in value.values():
            yield from trees(v)
    elif isinstance(value,list):
        for v in value:
            yield from trees(v)


def archive_census(configs):
    import numpy as np
    from zipfile import BadZipFile

    targets = {d["m"]:d for d in configs}
    exact,near,count,unreadable = [],[],{"npz":0,"json":0},{"npz":0,"json":0}

    def classify(d,path,metadata=None):
        if not isinstance(d,dict) or d.get("m") not in MS or d.get("k") != 80 or d.get("q") != 76:
            return
        if d.get("family","rfm") != "rfm":
            return
        expected = targets[d["m"]]
        if not all(d.get(k,expected[k]) == expected[k] for k in ("mean_order","layers","scale","split","backend")):
            return
        row = dict(path=path.relative_to(WORK).as_posix(),m=d["m"],seed=d.get("seed"))
        if metadata is not None:
            row.update({k:metadata.get(k) for k in ("horizon","dt","startup_rate","late_rate")})
        if d.get("seed") == SEED:
            exact.append(row)
        elif metadata is not None:
            near.append(row)

    for path in WORK.rglob("*.npz"):
        count["npz"] += 1
        try:
            with np.load(path,allow_pickle=False) as data:
                if "config" in data:
                    d = json.loads(str(data["config"]))
                    meta = json.loads(str(data["metadata"])) if "metadata" in data else {}
                    classify(d,path,meta)
        except (ValueError,OSError,KeyError,BadZipFile):
            unreadable["npz"] += 1
    for path in WORK.rglob("*.json"):
        count["json"] += 1
        try:
            raw = path.read_bytes()
            if b"7331" in raw and all(k in raw for k in (b'"m"',b'"k"',b'"q"')):
                for d in trees(json.loads(raw)):
                    classify(d,path)
        except (ValueError,OSError,RecursionError):
            unreadable["json"] += 1
    return dict(scope="Saved NPZ configs and JSON files with explicit m/k/q/seed in work",counts=count,unreadable=unreadable,
                exact=exact,other_seed_near_neighbors=near)


def freeze_short():
    import galerkin_minimal_model
    from basis_recurrence_model import RecurrenceGalerkin

    p,q = read(THIN),read(REFERENCE)
    old = next(j for j in p["jobs"] if (j["name"],j["arm"],j["seed"]) == ("rfm41_k80","uniform",SEED))
    cfg = old["config"]
    require(old["solve_args"]["steps"] == 72*152 and (cfg["m"],cfg["k"],cfg["q"],cfg["mean_order"]) == (41,80,76,24),
            "Registered high-scale spatial or temporal baseline differs")
    configs = [dict(cfg,m=m) for m in (*MS,41)]
    models = [RecurrenceGalerkin(dt=1/152,**d) for d in configs]
    ranks = {d["m"]:dict(rank=model.width,mean_rank=model.mean_width,angular_fft=model.kernel.n,
                         real_state_dofs=2*d["k"]*model.width+model.mean_width) for d,model in zip(configs,models,strict=True)}
    require(ranks[41]["rank"] == 40 and all(ranks[d["m"]]["rank"] < 40 and ranks[d["m"]]["mean_rank"] == 30 and
            ranks[d["m"]]["angular_fft"] == 243 for d in configs[:-1]),"The candidate does not reduce actual space at fixed FFT geometry")
    archives = archive_census(configs[:-1])
    prior_protocol,prior_summary = (FAILED/name for name in ("protocol.json","summary.json"))
    prior = read(prior_summary)
    require(prior["protocol_sha256"] == digest(prior_protocol) and len(prior["rows"]) == 3 and
            all(not r["qualified"] and r["arms"]["uniform"]["status"] == "complete" and
                r["arms"]["combined"]["failure_type"] == "ValueError" for r in prior["rows"]),
            "The retained harness failure differs")
    from cpu_sbdf2_switch import RecurrenceGalerkin as RequiredClass
    require(not issubclass(galerkin_minimal_model.RecurrenceGalerkin,RequiredClass) and
            issubclass(RecurrenceGalerkin,RequiredClass),"The rejected and corrected model identities differ")
    prior_inputs = [prior_protocol,prior_summary,HERE/"rfm_rank_high_screen.py"]
    prior_dir = FAILED.relative_to(WORK).as_posix()+"/"
    require(all(r["path"].startswith(prior_dir) for r in archives["exact"]),
            "A matching archive outside the failed harness was found")
    archives["matching_archives_are_original_failed_harness"] = bool(archives["exact"])
    import rank_fair_stock_qualification as frozen
    sources = dict(p["source_hashes"])
    frozen.merge(sources,q["source_hashes"])
    frozen.bind(sources,*(record(path) for path in (SOURCE,THIN,REFERENCE,HERE/"galerkin_minimal_model.py",
                   HERE/"rank_fair_stock_qualification.py",HERE/"legendre_rank_uniform.py",
                   WORK/"rfm_v7"/"verify_switch_qualification.py",*prior_inputs)))
    frozen.timing.prior.check_sources(sources|{name:digest(WORK/name) for name in frozen.timing.prior.REQUIRED_SOURCES})
    return dict(status="frozen",stage="t16",horizon=16,seed=SEED,physical_times=q["physical_times"][:12],
                grid_names=["gauss123","mid129","layer144"],physical_gate=GATE,reference_budget=.00125,probe_spacing=.125,
                switch_time=8,startup_rate=152,late_rate=136,cpu=list(CPU),ranks=ranks,archive_census=archives,
                model_class="basis_recurrence_model.RecurrenceGalerkin",first_attempt=record(prior_summary),
                jobs=[dict(config=d,expected=ranks[d["m"]]) for d in configs[:-1]],source_hashes=sources,
                promotion="Require both physical references and the own-uniform triangle guard, exact prefix and 129 probes at T16. Only passes advance to T72. Wake gate is tested at T72.",
                scope="Seed 7331 only, accuracy screening without speed claims or any changes to the 65-slot formal design.")


def load_frozen(path):
    p = read(path)
    require(p["status"] == "frozen" and all(digest(WORK/name) == sha for name,sha in p["source_hashes"].items()),
            "Frozen RFM screen source or reference differs")
    return p


def execute_pair(directory,index,cpu):
    set_affinity(cpu)
    import numpy as np
    from galerkin_minimal_model import case
    from basis_recurrence_model import RecurrenceGalerkin
    from cpu_sbdf2_switch import solve_cpu_switch

    p = load_frozen(directory/"protocol.json")
    require(cpu == CPU[index%len(CPU)],"Worker CPU differs from its registered slot")
    job = p["jobs"][index]
    cfg = job["config"]
    h,physical = p["horizon"],p["physical_times"]
    records = {}
    for arm in ("uniform","combined"):
        row = dict(status="failed",arm=arm)
        name = f"m{cfg['m']}_{arm}"
        common = dict(horizon=h,probe_spacing=p["probe_spacing"],**cfg)
        try:
            if arm == "uniform":
                times = physical
                result = case.solve(RecurrenceGalerkin,steps=h*p["startup_rate"],snapshots=times,**common)
            else:
                times = sorted(physical+[p["switch_time"]+i/p["late_rate"] for i in (1,2)])
                result = solve_cpu_switch(RecurrenceGalerkin,startup_rate=p["startup_rate"],late_rate=p["late_rate"],
                                          switch_time=p["switch_time"],snapshots=times,**common)
            require(np.array_equal(result.times,times) and np.array_equal(result.probe_times,np.arange(8*h+1)/8) and
                    result.model.width == job["expected"]["rank"] and result.model.mean_width == job["expected"]["mean_rank"] and
                    all(np.isfinite(getattr(result,k)).all() for k in ("states","transverse","vorticity")),"Bad rank, snapshots or probes")
            target = directory/(name+".npz")
            with target.open("xb") as handle:
                case.store_solution(handle,result,cfg)
            row.update(status="complete",artifact=record(target),steps=result.row["steps"],physical_times=len(times),probes=len(result.probe_times))
        except Exception as error:
            row.update(failure_type=type(error).__name__,failure=str(error).replace(str(WORK.parent),"<workspace>"))
        save(directory/(name+".json"),row)
        records[arm] = row
    return records


def run_workers(directory):
    p = read(directory/"protocol.json")

    def lane(which):
        for i in range(which,len(p["jobs"]),len(CPU)):
            command = [sys.executable,"-B",str(SOURCE),"--worker",str(directory),str(i),str(CPU[which])]
            m = p["jobs"][i]["config"]["m"]
            with (directory/f"m{m}_worker.log").open("xb") as log:
                try:
                    result = subprocess.run(command,env=local_inputs(),stdout=log,stderr=subprocess.STDOUT,timeout=180,check=False,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
                except subprocess.TimeoutExpired:
                    save(directory/f"m{m}_worker_failure.json",dict(status="timeout",deadline_seconds=180))
                    continue
            if result.returncode:
                save(directory/f"m{m}_worker_failure.json",dict(status="failed",exit_code=result.returncode))

    with ThreadPoolExecutor(max_workers=len(CPU)) as pool:
        list(pool.map(lane,range(len(CPU))))


def score(directory):
    import cpu_formal as base
    from galerkin_minimal_model import case,np
    from basis_recurrence_model import RecurrenceGalerkin
    from cpu_sbdf2_switch import equal_spatial_models
    from legendre_rank_uniform import scores
    from verify_switch_qualification import absolute_reference,fields_at

    p = load_frozen(directory/"protocol.json")
    ref = read(REFERENCE)
    grids = base.grid_set()
    physical = p["physical_times"]
    truth,budget = absolute_reference(ref,grids)
    require(budget["budget"] == p["reference_budget"] and set(grids) == set(p["grid_names"]),"Independent FD budget or grids differ")
    truth = {name:{grid:[values[ref["physical_times"].index(t)] for t in physical] for grid,values in fields.items()}
             for name,fields in truth.items()}
    rows = []
    for job in p["jobs"]:
        m = job["config"]["m"]
        row = dict(job=job,qualified=False,status="failed",arms={})
        try:
            raw = {arm:read(directory/f"m{m}_{arm}.json") for arm in ("uniform","combined")}
            row["arms"] = raw
            require(all(r["status"] == "complete" and digest(WORK/r["artifact"]["path"]) == r["artifact"]["sha256"] for r in raw.values()),
                    "A registered paired trajectory failed")
            uniform,mixed = (case.load_solution(WORK/raw[arm]["artifact"]["path"],RecurrenceGalerkin) for arm in raw)
            require(equal_spatial_models((uniform.model,mixed.model)),"The compared pair uses a different radial space")
            end = p["switch_time"]
            prefix = uniform.probe_times <= end
            bitwise = (uniform.states[uniform.times<=end].tobytes() == mixed.states[mixed.times<=end].tobytes() and
                       all(getattr(uniform,k)[prefix].tobytes() == getattr(mixed,k)[prefix].tobytes() for k in ("transverse","vorticity")))
            direct,guard,anchor,tables,delta = scores(fields_at(mixed,physical,grids),fields_at(uniform,physical,grids),truth,grids,physical)
            gates = {key:all(max(e.values())<=GATE for e in d.values()) for key,d in (("mixed",direct),("triangle",guard),("uniform",anchor))}
            wake = {arm:base.wake(sol) for arm,sol in (("uniform",uniform),("combined",mixed))} if p["horizon"]==72 else {}
            short_replay = True
            if p["horizon"] == 72:
                for arm,long in (("uniform",uniform),("combined",mixed)):
                    old = p["short_records"][str(m)][arm]
                    require(digest(WORK/old["path"]) == old["sha256"],"Short registered trajectory changed")
                    short = case.load_solution(WORK/old["path"],RecurrenceGalerkin)
                    short_replay &= all(getattr(long,k)[:len(getattr(short,k))].tobytes() == getattr(short,k).tobytes()
                                        for k in ("times","states","probe_times","transverse","vorticity"))
            passed = bitwise and short_replay and all(gates.values()) and all(r["sustained_alternating_probe_signs"] for r in wake.values())
            row.update(status="complete",qualified=passed,prefix_bitwise=bitwise,short_replay_bitwise=short_replay,gates=gates,
                       physical_worst=direct,guard_worst=guard,uniform_worst=anchor,same_space_difference=delta,tables=tables,wake=wake)
        except Exception as error:
            row.update(failure_type=type(error).__name__,failure=str(error).replace(str(WORK.parent),"<workspace>"))
        rows.append(row)
    report = dict(status="complete",protocol_sha256=digest(directory/"protocol.json"),reference_budget=budget,rows=rows,
                  time_ranking_supported=False)
    save(directory/"summary.json",report)
    return report


def main():
    set_affinity(CPU[0])
    p = freeze_short()
    OUT.mkdir()
    short = OUT/"t16"
    short.mkdir()
    save(short/"protocol.json",p)
    run_workers(short)
    outcome = score(short)
    passed = [row for row in outcome["rows"] if row["qualified"]]
    print("T16",[(row["job"]["config"]["m"],row["qualified"]) for row in outcome["rows"]],flush=True)
    if not passed:
        return
    refs = {str(row["job"]["config"]["m"]):{arm:r["artifact"] for arm,r in row["arms"].items()} for row in passed}
    sources = dict(p["source_hashes"])
    for item in (record(short/"protocol.json"),record(short/"summary.json"),*(r for pair in refs.values() for r in pair.values())):
        sources[item["path"]] = item["sha256"]
    q = dict(p,stage="t72",horizon=72,physical_times=read(REFERENCE)["physical_times"],jobs=[row["job"] for row in passed],
             source_hashes=sources,short_records=refs)
    long = OUT/"t72"
    long.mkdir()
    save(long/"protocol.json",q)
    run_workers(long)
    outcome = score(long)
    print("T72",[(row["job"]["config"]["m"],row["qualified"]) for row in outcome["rows"]],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=3,metavar=("DIRECTORY","INDEX","CPU"))
    args = parser.parse_args()
    if args.worker:
        directory,index,cpu = args.worker
        execute_pair(Path(directory),int(index),int(cpu))
    else:
        main()
