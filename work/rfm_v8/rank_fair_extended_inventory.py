"""Resolve the draft 80-slot inventory to fixed CPU controls without starting CUDA."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import rank_fair_stock_qualification as original
from rank_fair_owned_transport import read, require


HERE,WORK = original.HERE,original.WORK
SOURCE = Path(__file__).resolve()
DESIGN = HERE/"protocol_rank_fair_extended_design.json"
PARITY = HERE/"rank_parity_expansion_output"
SCREEN = HERE/"rfm_rank_high_screen_paired_output"/"t72"
SHA = original.shared.sha
SEEDS = (7331,7687,8011,8353,9041)
RANKS = (38,38,38,39,38)


def audit_input(path):
    summary,protocol = path/"summary.json",path/"protocol.json"
    s,p = map(read,(summary,protocol))
    require(s["status"] == "complete" and s["protocol_sha256"] == SHA(protocol),"Registered CPU qualification changed")
    return s,p,[original.shared.record(item) for item in (summary,protocol)]


def new_job(name,row,arm,physical):
    job = row["job"]
    cfg = job["config"]
    uniform = row["arms"]["uniform"]["artifact"]
    control = row["arms"][arm]["artifact"]
    start,late = job.get("rates",[152,136])
    args = dict(cfg,horizon=72,probe_spacing=.125,graph_steps=24)
    if arm == "combined":
        args.update(startup_rate=start,late_rate=late,switch_time=8,preparation="combined",snapshots=sorted(physical+[8+i/late for i in (1,2)]))
    else:
        args.update(steps=72*start,snapshots=physical)
    return dict(name=name,arm=arm,seed=cfg["seed"],scale="high",family=cfg["family"],config=cfg,args=args,
                control={"single":control},anchor={"single":uniform},engine="warp_cufft",serial_control=None,
                retained_rank=job["rank"] if "rank" in job else job["expected"]["rank"],applies_to_scales=["high"])


def draft_inventory():
    base = original.build()
    design = read(DESIGN)
    require(design["rfm_seeds_in_block_order"] == list(SEEDS) and design["deterministic_seed"] == 17 and
            design["runtime_policy"]["admitted_for_timing"] == [],"Draft candidates are not independently timing-qualified")
    parity = read(PARITY/"summary.json")
    require(parity["status"] == "complete" and parity["high_rfm35_eligible_for_gpu"] and not parity["formal_draft_changed"] and
            not parity["speed_claim_supported"] and [(r["seed"],r["rank"],r["qualified"]) for r in parity["high_rfm35_by_seed"]] ==
            [(seed,rank,True) for seed,rank in zip(SEEDS,RANKS,strict=True)],"Five-seed numerical admission or recorded ranks differ")
    for r in parity["sources"]:
        original.shared.bound(r)
    high,p_high,high_records = audit_input(PARITY/"initial")
    leg,p_leg,leg_records = audit_input(PARITY/"promoted_t72")
    early,p_early,early_records = audit_input(SCREEN)
    by_seed = {r["job"]["seed"]:r for r in high["rows"] if r["job"]["name"].startswith("high_rfm35_")}
    by_seed[7331] = next(r for r in early["rows"] if r["job"]["config"]["m"] == 35)
    lower = next(r for r in leg["rows"] if r["job"]["name"] == "high_leg34_s17")
    require(not lower["qualified"] and lower["gates"] == {"mixed":True,"triangle":False,"uniform":True} and
            lower["wake"]["uniform"]["sustained_alternating_probe_signs"] and lower["t16_replay_bitwise"]["uniform"],
            "Legendre34 uniform-only qualification differs")
    require(set(by_seed) == set(SEEDS) and all(r["qualified"] and all(wake["sustained_alternating_probe_signs"]
            for wake in r["wake"].values()) for r in by_seed.values()),"A fixed RFM35 seed does not qualify")
    physical = base["physical_times"]
    added = [new_job("rfm35_k80",by_seed[seed],arm,physical) for seed in SEEDS for arm in ("uniform","combined")]
    added.append(new_job("leg34_k80_q76",lower,"uniform",physical))
    sources = dict(base["sources_sha256"])
    for p in (p_high,p_leg,p_early):
        original.merge(sources,p["source_hashes"])
    original.bind(sources,original.shared.record(SOURCE),original.shared.record(DESIGN),original.shared.record(PARITY/"summary.json"),
                  *high_records,*leg_records,*early_records,*[v for j in added for v in (*j["control"].values(),*j["anchor"].values())])
    controls = {original.order(j):j for j in (*base["slots"],*added)}
    require(len(controls) == 39 and all((SHA(WORK/name) == digest) for name,digest in sources.items()),"The 39-slot CPU source inventory changed")
    schedule = {}
    ids = {}
    for scale in design["scales"]:
        slots,pairs = block_schedule(scale,design)
        tags = {tuple((j["name"],j["arm"],j["seed"])) for j in slots}
        ids[scale["name"]] = tags
        require(len(tags) == scale["unique_qualifications"] and len(slots) == scale["slots"] and tags <= set(controls) and all(
            scale["name"] in controls[tag]["applies_to_scales"] for tag in tags),"Scale schedules and exact identities differ")
        schedule[scale["name"]] = dict(slots=slots,paired=pairs,order_seed=scale["order_seed"])
    require(ids["mid"] & ids["high"] == {("fd176_k80","fd",17)} and ids["mid"] | ids["high"] == set(controls),
            "Only exact FD176 may be reused across scales")
    for name,values in design["retained_ranks"].items():
        seeds = SEEDS if name.startswith("rfm") else (17,)
        for seed,rank in zip(seeds,values,strict=True):
            require(all(j["retained_rank"] == rank for tag,j in controls.items() if tag[0] == name and tag[2] == seed),
                    "The frozen retained-rank contrast differs")
    return dict(base,status="draft_no_authorization_no_gpu_qualification",stage="extended_stock_qualification_draft",slots=list(controls.values()),
                design=design,schedules=schedule,sources_sha256=sources,cpu_auditor_affinities=[10,12,14,16],
                gpu_trajectories_run=0,exact_runtime_adapter_selected=False)


def block_schedule(scale,design):
    groups = [tuple(g) for g in scale["groups"]]
    rng = random.Random(scale["order_seed"])
    slots,pairs = [],[]
    for block,seed in enumerate(SEEDS,1):
        order = groups.copy()
        rng.shuffle(order)
        slots += [dict(block=block,name=name,arm=arm,seed=seed if name.startswith("rfm") else 17) for name,arm in order]

    def add(kind,left,right):
        require(left in groups and right in groups,"Pair is absent from registered groups")
        pairs.append(dict(kind=kind,baseline=list(left),candidate=list(right)))

    high = scale["name"] == "high"
    rfm,leg = ("rfm41_k80","leg36_k80_q76") if high else ("rfm37_k72","leg35_k72_q66")
    for arm in ("uniform","combined"):
        add("same_retained_rank",(leg,arm),(rfm,arm))
        if high:
            smaller = ("rfm35_k80",arm)
            add("rfm_reduced_rank_same_schedule",(rfm,arm),smaller)
            for fixed in (("leg34_k80_q76","uniform"),("leg35_k80_q76","uniform"),("leg35_k80_q76","combined")):
                kind = "accuracy_matched_fixed_legendre_"+("same_schedule" if fixed[1] == arm else "cross_schedule")
                add(kind,fixed,smaller)
    for name,arm in groups:
        if arm == "uniform" and (name,"combined") in groups:
            add("steps_and_shared_preparation",(name,"uniform"),(name,"combined"))
        if arm != "fd":
            for fd in (g for g in groups if g[1] == "fd"):
                add("qualified_fd",fd,(name,arm))
    if high:
        add("qualified_fd_scale",("fd184_k80","fd"),("fd176_k80","fd"))
    return slots,pairs


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    plan = draft_inventory()
    if args.output:
        original.shared.save(args.output,plan)
    print(json.dumps(dict(status=plan["status"],unique=len(plan["slots"]),sources=len(plan["sources_sha256"]),
         schedule={scale:dict(slots=len(v["slots"]),contrasts=len(v["paired"])) for scale,v in plan["schedules"].items()},
         draft_sha256=sha256(original.timing.prior.raw_json(plan)).hexdigest(),gpu_trajectories_run=0)))
