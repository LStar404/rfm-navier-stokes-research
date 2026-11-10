"""Execute the registered high five-seed and deterministic rank-parity CPU checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import rfm_rank_high_screen_paired as prior


HERE,WORK,CPU = prior.HERE,prior.WORK,prior.CPU
SOURCE = Path(__file__).resolve()
DESIGN = HERE/"protocol_rank_parity_expansion.json"
OUT = HERE/"rank_parity_expansion_output"
require,read,record,digest,save = prior.require,prior.read,prior.record,prior.digest,prior.save


def check_record(item):
    path = (WORK/item["path"]).resolve()
    require(path.is_relative_to(WORK) and record(path) == item,"A registered artifact or its path changed")
    return path


def freeze():
    import galerkin_minimal_model
    from basis_recurrence_model import RecurrenceGalerkin
    from rank_fair_stock_qualification import merge

    p,ref,thin = (read(path) for path in (DESIGN,prior.REFERENCE,prior.THIN))
    require(p["physical_times_t72"] == ref["physical_times"] and p["physical_times_t16"] == ref["physical_times"][:12] and
            p["cpu_affinities"] == list(CPU) and p["threads"] == 1,"Frozen physical or CPU scope differs")
    require([j["seed"] for j in p["jobs"][:4]] == [7687,8011,8353,9041],"The existing four high seeds differ")
    old_rfm_path = WORK/p["inherited"]["high_rfm35_7331"]
    old_leg_path = WORK/p["inherited"]["high_leg35"]
    old_rfm,old_leg = map(read,(old_rfm_path,old_leg_path))
    old_rfm_protocol = old_rfm_path.parent/"protocol.json"
    old_leg_protocol = old_leg_path.parent/"protocol.json"
    require(digest(old_rfm_protocol) == old_rfm["protocol_sha256"] and digest(old_leg_protocol) == old_leg["protocol_sha256"],
            "Inherited numerical control changed")
    existing_rfm = next(r for r in old_rfm["rows"] if r["job"]["config"]["m"] == 35)
    existing_leg = next(r for r in old_leg["rows"] if r["job"]["name"] == "leg35_k80_q76")
    existing_leg36 = next(r for r in old_leg["rows"] if r["job"]["name"] == "leg36_k80_q76")
    require(existing_rfm["qualified"] and existing_leg["qualified"] and existing_leg36["qualified"] and
            existing_leg36["actual_rank"] == 40 and existing_rfm["job"]["config"]["seed"] == 7331,
            "The prior high RFM and Legendre arms are not qualified")
    baseline = {(j["config"]["k"],j["seed"]):j["config"] for j in thin["jobs"] if j["family"] == "rfm" and j["arm"] == "uniform"}
    jobs = []
    for j in p["jobs"]:
        cfg = (baseline[j["k"],j["seed"]] if j["family"] == "rfm" else existing_leg["job"]["config"])
        cfg = dict(cfg,m=j["m"])
        require(all(cfg[k] == j[k] for k in ("m","k","q","seed","family")),"Only the registered feature count may differ")
        model = RecurrenceGalerkin(dt=1/j["rates"][0],**cfg)
        rank = int(model.width)
        jobs.append(dict(j,config=cfg,horizon=j["first_horizon"],rank=rank,mean_rank=int(model.mean_width),fft_length=int(model.kernel.n),
                         real_state_dofs=int(2*j["k"]*rank+model.mean_width)))
    sources = dict(thin["source_hashes"])
    for inventory in (ref["source_hashes"],read(old_rfm_protocol)["source_hashes"],read(old_leg_protocol)["source_hashes"]):
        merge(sources,inventory)
    prior_leg_files = [old_leg_path.parent/r["artifact"] for r in (existing_leg,existing_leg36)]
    require(all(digest(path) == row["artifact_sha256"] for path,row in zip(prior_leg_files,(existing_leg,existing_leg36),strict=True)),
            "Prior deterministic Legendre trajectory changed")
    files = [SOURCE,DESIGN,prior.THIN,prior.REFERENCE,old_rfm_path,old_rfm_protocol,old_leg_path,old_leg_protocol,
             check_record(existing_rfm["arms"]["uniform"]["artifact"]),check_record(existing_rfm["arms"]["combined"]["artifact"]),
             *prior_leg_files,Path(galerkin_minimal_model.__file__)]
    merge(sources,{path.relative_to(WORK).as_posix():digest(path) for path in files})
    return dict(status="frozen",stage="initial",design=p,jobs=jobs,source_hashes=sources,
                inherited_rfm35_seed7331=dict(summary=record(old_rfm_path),qualified=existing_rfm["qualified"],
                    rank=existing_rfm["job"]["expected"]["rank"],physical_worst=existing_rfm["physical_worst"],
                    guard_worst=existing_rfm["guard_worst"],uniform_worst=existing_rfm["uniform_worst"]),
                inherited_leg35=dict(summary=record(old_leg_path),qualified=existing_leg["qualified"],rank=existing_leg["actual_rank"]),
                inherited_leg36=dict(summary=record(old_leg_path),qualified=existing_leg36["qualified"],rank=existing_leg36["actual_rank"]))


def worker(stage,index,cpu):
    prior.set_affinity(cpu)
    from galerkin_minimal_model import case,np
    from basis_recurrence_model import RecurrenceGalerkin
    from cpu_sbdf2_switch import solve_cpu_switch

    p = prior.load_frozen(stage/"protocol.json")
    require(cpu == CPU[index%len(CPU)],"Worker differs from registered CPU affinity")
    d,j = p["design"],p["jobs"][index]
    here = stage/j["name"]
    h = j["horizon"]
    first,last = j["rates"]
    common = dict(horizon=h,probe_spacing=d["probe_spacing"],**j["config"])
    physical = d[f"physical_times_t{h}"]
    for arm in ("uniform","combined"):
        row = dict(status="failed",arm=arm,job=j["name"],cpu=cpu)
        try:
            if arm == "uniform":
                times = physical
                solution = case.solve(RecurrenceGalerkin,steps=h*first,snapshots=times,**common)
                steps = h*first
            else:
                t0 = d["switch_time"]
                times = sorted(physical+[t0+i/last for i in (1,2)])
                solution = solve_cpu_switch(RecurrenceGalerkin,startup_rate=first,late_rate=last,switch_time=t0,snapshots=times,**common)
                steps = t0*first+(h-t0)*last
            require(solution.model.width == j["rank"] and solution.model.mean_width == j["mean_rank"] and
                    solution.model.kernel.n == j["fft_length"] and solution.row["steps"] == steps and
                    np.array_equal(solution.times,times) and np.array_equal(solution.probe_times,np.arange(8*h+1)/8) and
                    solution.transverse.shape == solution.vorticity.shape == (8*h+1,3) and
                    all(np.isfinite(getattr(solution,k)).all() for k in ("states","transverse","vorticity")),
                    "Raw rank, schedule, snapshots or complete probes differ")
            path = here/f"{arm}.npz"
            with path.open("xb") as handle:
                case.store_solution(handle,solution,j["config"])
            row.update(status="complete",artifact=record(path),steps=steps,snapshots=len(times),probes=len(solution.probe_times))
        except Exception as error:
            row.update(failure_type=type(error).__name__,failure=str(error).replace(str(WORK.parent),"<workspace>"))
        save(here/f"{arm}.json",row)


def run_workers(stage,p):
    def lane(which):
        for i in range(which,len(p["jobs"]),len(CPU)):
            here = stage/p["jobs"][i]["name"]
            here.mkdir()
            command = [sys.executable,"-B",str(SOURCE),"--worker",str(stage),str(i),str(CPU[which])]
            with (here/"worker.log").open("xb") as log:
                try:
                    run = subprocess.run(command,env=prior.local_inputs(),stdout=log,stderr=subprocess.STDOUT,
                                         timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
                    status = dict(status="complete" if not run.returncode else "failed",exit_code=run.returncode)
                except subprocess.TimeoutExpired:
                    status = dict(status="timeout",deadline_seconds=180)
            save(here/"worker.json",status)

    with ThreadPoolExecutor(max_workers=len(CPU)) as pool:
        list(pool.map(lane,range(len(CPU))))


def reference_fields():
    from galerkin_minimal_model import case
    import cpu_formal as base
    from verify_switch_qualification import absolute_reference

    grids = base.grid_set()
    truth,budget = absolute_reference(read(prior.REFERENCE),grids)
    return base,case,grids,truth,budget


def score(stage,reference):
    from basis_recurrence_model import RecurrenceGalerkin
    from cpu_sbdf2_switch import equal_spatial_models
    from legendre_rank_uniform import scores
    from verify_switch_qualification import fields_at

    protocol = stage/"protocol.json"
    p = prior.load_frozen(protocol)
    d = p["design"]
    base,case,grids,truth,budget = reference
    require(set(truth) == set(d["reference_names"]) and set(grids) == set(d["grid_names"]) and budget["budget"] == d["reference_budget"],
            "Absolute reference identities or budgets differ")
    rows = []
    for j in p["jobs"]:
        here = stage/j["name"]
        row = dict(job=j,status="failed",qualified=False,worker=read(here/"worker.json"))
        try:
            h = j["horizon"]
            times = d[f"physical_times_t{h}"]
            refs = {name:{grid:[values[d["physical_times_t72"].index(t)] for t in times] for grid,values in f.items()}
                    for name,f in truth.items()}
            raw = {arm:read(here/f"{arm}.json") for arm in ("uniform","combined")}
            row["arms"] = raw
            require(all(v["status"] == "complete" for v in raw.values()),"A member of the registered pair failed")
            paths = {arm:check_record(v["artifact"]) for arm,v in raw.items()}
            sol = {arm:case.load_solution(path,RecurrenceGalerkin) for arm,path in paths.items()}
            u,m = sol["uniform"],sol["combined"]
            require(equal_spatial_models((u.model,m.model)),"Uniform and mixed spaces differ")
            t0 = d["switch_time"]
            prefix = u.probe_times <= t0
            bitwise = (u.states[u.times<=t0].tobytes() == m.states[m.times<=t0].tobytes() and all(
                getattr(u,k)[prefix].tobytes() == getattr(m,k)[prefix].tobytes() for k in ("transverse","vorticity")))
            direct,guard,anchor,tables,delta = scores(fields_at(m,times,grids),fields_at(u,times,grids),refs,grids,times)
            gates = {label:all(set(v) == set(d["norms"]) and max(v.values()) <= d["physical_gate"] for v in table.values())
                     for label,table in (("mixed",direct),("triangle",guard),("uniform",anchor))}
            wake = {arm:base.wake(v) for arm,v in sol.items()} if h == 72 else {}
            replay = None
            if "short_controls" in j:
                replay = {}
                for arm,v in sol.items():
                    old = case.load_solution(check_record(j["short_controls"][arm]),RecurrenceGalerkin)
                    replay[arm] = all(getattr(v,k)[:len(getattr(old,k))].tobytes() == getattr(old,k).tobytes()
                                      for k in ("times","states","probe_times","transverse","vorticity"))
            passed = bitwise and (replay is None or all(replay.values())) and all(gates.values()) and all(
                v["sustained_alternating_probe_signs"] for v in wake.values())
            row.update(status="complete",qualified=passed,prefix_bitwise=bitwise,t16_replay_bitwise=replay,gates=gates,
                       physical_worst=direct,guard_worst=guard,uniform_worst=anchor,same_space_difference=delta,tables=tables,wake=wake)
        except Exception as error:
            row.update(failure_type=type(error).__name__,failure=str(error).replace(str(WORK.parent),"<workspace>"))
        rows.append(row)
    outcome = dict(status="complete",protocol_sha256=digest(protocol),reference_budget=budget,rows=rows,speed_claim_supported=False)
    save(stage/"summary.json",outcome)
    return outcome


def run_stage(p,ref=None):
    stage = OUT/p["stage"]
    stage.mkdir()
    save(stage/"protocol.json",p)
    run_workers(stage,p)
    reference = ref if ref is not None else reference_fields()
    outcome = score(stage,reference)
    print(p["stage"],[(r["job"]["name"],r["job"]["rank"],r["qualified"]) for r in outcome["rows"]],flush=True)
    return stage,outcome,reference


def main():
    prior.set_affinity(CPU[0])
    require(not OUT.exists(),"Preserve the first frozen parity attempt")
    p = freeze()
    OUT.mkdir()
    first,initial,ref = run_stage(p)
    promoted = [r for r in initial["rows"] if r["qualified"] and r["job"]["horizon"] == 16]
    initial_full = [r for r in initial["rows"] if r["job"]["horizon"] == 72]
    later = []
    records = [record(first/"protocol.json"),record(first/"summary.json")]
    if promoted:
        inputs = [v["artifact"] for row in promoted for v in row["arms"].values()]
        sources = dict(p["source_hashes"])|{item["path"]:item["sha256"] for item in (*records,*inputs)}
        q = dict(p,stage="promoted_t72",source_hashes=sources,jobs=[dict(r["job"],horizon=72,short_controls={
            arm:v["artifact"] for arm,v in r["arms"].items()}) for r in promoted])
        stage,result,_ = run_stage(q,ref)
        records.extend((record(stage/"protocol.json"),record(stage/"summary.json")))
        later = result["rows"]
    rfm = [{"seed":7331,"rank":p["inherited_rfm35_seed7331"]["rank"],"qualified":True}]+[
        dict(seed=r["job"]["seed"],rank=r["job"]["rank"],qualified=r["qualified"]) for r in initial_full]
    high = len(rfm) == 5 and all(row["qualified"] and row["rank"] < 40 for row in rfm)
    leg = [dict(name=f"high_leg{m}_s17",rank=p[f"inherited_leg{m}"]["rank"],qualified=True) for m in (35,36)]+[
        dict(name=r["job"]["name"],rank=r["job"]["rank"],qualified=r["qualified"]) for r in later if r["job"]["family"] == "legendre"]
    winner = min((r for r in leg if r["qualified"]),key=lambda r:r["rank"])
    ranks = set(row["rank"] for row in rfm) if high else {40}
    same = [r for r in leg if r["qualified"] and r["rank"] in ranks]
    save(OUT/"summary.json",dict(status="complete",sources=records,high_rfm35_by_seed=rfm,high_rfm35_eligible_for_gpu=high,
        high_rfm_lowest_five_seed_candidate="rfm35" if high else "rfm41",high_legendre_qualified_options=leg,
        high_lowest_rank_legendre=winner,high_same_rank_legendre=same,initial_rows=initial["rows"],promoted_rows=later,
        speed_claim_supported=False,formal_draft_changed=False))
    print("high five-seed eligible",high,"lowest deterministic",winner,"same rank",same,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=3)
    args = parser.parse_args()
    if args.worker:
        stage,index,cpu = args.worker
        worker(Path(stage),int(index),int(cpu))
    else:
        main()
