"""Rank-guided RFM bandwidth check over five fixed seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import cpu_formal as base
from cpu_formal_layer_tangent import rank_record
from cpu_formal_projection import projection

case = base.case
np = case.np
ROOT = base.HERE/"cpu_formal_bandwidth_output"
PROTOCOL = base.HERE/"protocol_rfm_bandwidth.json"


def settings(protocol, scale, seed):
    cfg={key:protocol[key] for key in ("m","k","q","mean_order","layers")}
    return dict(cfg,family="rfm",scale=scale,seed=seed,split=True,backend="fft")


def projection_tables(protocol, ref_path):
    result=dict(status="running",protocol=protocol,reference_sha256=base.digest(ref_path),
        source_sha256=base.digest(Path(__file__)),cpu_sources=base.source_hashes(),rows=[])
    ref=case.load_solution(ref_path,base.FullFD)
    grids=base.grid_set()
    target={n:tuple(np.stack(a,axis=-1) for a in zip(
        *(ref.modal(i,s) for i in range(len(ref.times))))) for n,(s,_,_) in grids.items()}
    path=ROOT/"oracle.json"
    base.save(path,result)
    for seed in protocol["seeds"]:
        for scale in (protocol["base_scale"],protocol["proposed_scale"]):
            cfg=settings(protocol,scale,seed)
            model=base.BufferedGalerkin(dt=1/protocol["rate"],**cfg)
            fits=[]
            for n,(s,_,w) in grids.items():
                for region,weight in (("full",w),("near",w*(case.radius(s)<=2.5))):
                    fits.append(dict(grid=n,region=region,**projection(model,s,weight,target[n],ref.times)))
            worst={f"{region}_{label}":max(max(r[label]) for r in fits if r["region"]==region)
                   for region in ("full","near") for label in ("velocity","vorticity")}
            rank=rank_record(model,cfg)
            result["rows"].append(dict(seed=seed,scale=scale,rank=rank,worst=worst,fits=fits))
            base.save(path,result)
            print(seed,scale,rank["streamfunction"]["actual_retained_rank"],worst,flush=True)
    table={(r["seed"],r["scale"]):r["worst"]["full_vorticity"] for r in result["rows"]}
    left=[table[s,protocol["base_scale"]] for s in protocol["seeds"]]
    right=[table[s,protocol["proposed_scale"]] for s in protocol["seeds"]]
    count=sum(b<a for a,b in zip(left,right))
    ratio=max(right)/max(left)
    result.update(status="complete",improved_seeds=count,worst_ratio=ratio,
                  promotion_passes=bool(count>=4 and ratio<=.8))
    base.save(path,result)
    print("PROMOTION",result["promotion_passes"],"worst_ratio",ratio,flush=True)


def march_one(protocol, ref_path, seed):
    scale=protocol["proposed_scale"]
    path=ROOT/f"t72_scale{scale}_seed{seed}.npz"
    details=path.with_suffix(".json")
    cfg=settings(protocol,scale,seed)
    if details.exists():
        old=base.read(details)
        if old["config"]!=cfg or ("sha256" in old and base.digest(path)!=old["sha256"]):
            raise RuntimeError(f"Changed saved bandwidth experiment {seed}")
        return old
    if path.exists():
        path.rename(path.with_suffix(f".orphan_{time.time_ns()}.npz"))
    row=dict(seed=seed,config=cfg)
    try:
        grids=base.grid_set()
        ref=case.load_solution(ref_path,base.FullFD)
        truth=case.reference_fields(ref,grids)
        with np.errstate(over="ignore",invalid="ignore"):
            sol=case.solve(base.BufferedGalerkin,steps=protocol["horizon"]*protocol["rate"],**cfg)
        scores=case.score(sol,truth,grids)
        temp=path.with_suffix(".tmp.npz")
        case.store_solution(temp,sol,cfg)
        temp.replace(path)
        row.update(status="complete",metadata=sol.row,worst=case.maxima(scores),scores=scores,
                   wake=base.wake(sol),rank=rank_record(sol.model,cfg),sha256=base.digest(path))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError) as exc:
        row.update(status="failed",failure=str(exc))
    base.save(details,row)
    return row


def march(protocol, ref_path, workers):
    oracle=base.read(ROOT/"oracle.json")
    if oracle["status"]!="complete" or not oracle["promotion_passes"]:
        raise RuntimeError("The registered sampled-fit promotion gate must pass")
    path=ROOT/"march.json"
    fixed=dict(protocol=protocol,reference_sha256=base.digest(ref_path),
        oracle_sha256=base.digest(ROOT/"oracle.json"),cpu_sources=base.source_hashes(),
        source_sha256=base.digest(Path(__file__)))
    if path.exists():
        result=base.read(path)
        if result["frozen"]!=fixed:
            raise RuntimeError("Changed frozen bandwidth run")
        for row in result["rows"]:
            artifact=ROOT/f"t72_scale{protocol['proposed_scale']}_seed{row['seed']}.npz"
            if "sha256" in row and base.digest(artifact)!=row["sha256"]:
                raise RuntimeError("Saved bandwidth trajectory changed")
    else:
        result=dict(status="running",frozen=fixed,rows=[])
        base.save(path,result)
    known={r["seed"] for r in result["rows"]}
    pending=[seed for seed in protocol["seeds"] if seed not in known]
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(march_one,protocol,ref_path,seed) for seed in pending]):
                row=future.result()
                result["rows"].append(row)
                base.save(path,result)
                print(row["seed"],row.get("worst",row.get("failure")),flush=True)
    result["status"]="complete"
    base.save(path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("oracle","march"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=2:
        parser.error("Use one or two CPU workers for this diagnostic")
    protocol=base.read(PROTOCOL)
    ROOT.mkdir(exist_ok=True)
    reference=base.WORK/protocol["reference"]
    if args.phase=="oracle":
        if (ROOT/"oracle.json").exists():
            raise FileExistsError(ROOT/"oracle.json")
        projection_tables(protocol,reference)
    else:
        march(protocol,reference,args.workers)


if __name__=="__main__":
    main()
