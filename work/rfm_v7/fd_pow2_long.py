"""Frozen, restartable two-arm FD reference experiment with row-power-of-two LU."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
from unittest.mock import patch

from fd_cpu_buffered import BufferedFD
from fd_equilibration_pow2 import PowerOfTwoBand, base, fd_full, np
from cpu_formal_reference_point1 import PROTOCOL

HERE = Path(__file__).resolve().parent
OUT = HERE/"cpu_formal_point1_pow2_output"
JOURNAL = OUT/"runs.json"
case = base.case
GRID = dict(k=128, stretch=1., grid="outer", angular="fft", linear="native")
RATE = 384
HORIZON = 72
STEPS = HORIZON*RATE
SPECS = (384,448)
CLASS = "fd_pow2_long.Pow2EquilibratedBufferedFD"


class Pow2EquilibratedBufferedFD(BufferedFD):
    def __init__(self,*args,**kwargs):
        with patch.object(fd_full,"BandSolve",PowerOfTwoBand):
            super().__init__(*args,**kwargs)
        self.metadata.update(equilibration="row_power_of_two", reconstruction_class=CLASS,
            equilibration_setup_seconds=self.bdf.equilibration_seconds+self.first.equilibration_seconds)

    def scale_metadata(self):
        scales = {}
        for name in ("first","bdf"):
            d = getattr(self,name).row_scale
            exponents = np.frexp(d)[1]-1
            scales[name] = dict(sha256=hashlib.sha256(d.tobytes()).hexdigest(),
                shape=list(d.shape), exponent_min=int(exponents.min()),exponent_max=int(exponents.max()))
        return scales


def write_npz(path, **arrays):
    temp=path.with_suffix(".tmp.npz")
    np.savez(temp,**arrays)
    temp.replace(path)


def march(n):
    import json
    dt = 1/RATE
    start = time.perf_counter()
    cfg = dict(GRID,n=n)
    model = Pow2EquilibratedBufferedFD(dt=dt,**cfg)
    scales = model.scale_metadata()
    identity = json.dumps(dict(config=cfg,dt=dt,steps=STEPS,class_name=CLASS,scales=scales),sort_keys=True)
    snaps = np.asarray(case.SNAPSHOTS,dtype=float)
    snapshots = set(case.step_indices(snaps,dt))
    probe_times = np.arange(0.,HORIZON+.001,.125)
    stride = int(case.step_indices(probe_times,dt)[1])
    speeds = case.speed(dt*np.arange(STEPS+1))
    state = np.zeros(model.rhs_shape,dtype=np.complex128)
    mass = flux = old_mass = old_flux = state
    transverse = np.zeros((len(probe_times),len(case.PROBES)))
    vorticity = np.zeros_like(transverse)
    states = []
    offset = 0
    elapsed = 0.
    checkpoint = OUT/f"resume_fd{n}_k128_rate384.npz"
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            if str(saved["identity"])!=identity:
                raise RuntimeError("Resume state differs from frozen scales or settings")
            offset = int(saved["step"])
            if not 0<offset<STEPS or offset%(RATE*6):
                raise RuntimeError("Invalid checkpoint step")
            elapsed = float(saved["active_seconds"])
            states = list(saved["states"])
            transverse,vorticity = saved["transverse"],saved["vorticity"]
            mass,flux,old_mass,old_flux = (saved[key] for key in
                ("mass","flux","old_mass","old_flux"))
    setup = time.perf_counter()-start
    print("START",n,"PID",os.getpid(),"resume_step",offset,flush=True)
    for i in range(offset+1,STEPS+1):
        rhs = mass+dt*flux if i==1 else 2*mass-.5*old_mass+dt*(2*flux-old_flux)
        state = model.advance(rhs,speeds[i],i==1)
        if i in snapshots:
            if not np.isfinite(state).all():
                raise FloatingPointError(f"Nonfinite FD{n} state at t={i*dt}")
            states.append(state.copy())
        if i%stride==0:
            j = i//stride
            transverse[j],vorticity[j] = model.probes(state,speeds[i])
            if not np.isfinite(transverse[j]).all() or not np.isfinite(vorticity[j]).all():
                raise FloatingPointError(f"Nonfinite FD{n} probe at t={i*dt}")
        if i!=STEPS:
            old_mass,old_flux = mass,flux
            mass,flux = model.terms(state,speeds[i])
        if i%(RATE*6)==0 and i<STEPS:
            write_npz(checkpoint,identity=identity,step=i,active_seconds=elapsed+time.perf_counter()-start,
                      states=np.asarray(states),last_state=state,transverse=transverse,
                      vorticity=vorticity,mass=mass,flux=flux,old_mass=old_mass,old_flux=old_flux)
            print("CHECKPOINT",n,"t",i//RATE,"seconds",round(elapsed+time.perf_counter()-start,2),flush=True)
    row = dict(steps=STEPS,horizon=HORIZON,dt=dt,setup_seconds=setup,
               seconds=elapsed+time.perf_counter()-start,time_order=2,row_scale=scales,
               resumed_from_step=offset,**model.metadata)
    return case.Solution(model,np.asarray(states),snaps,probe_times,transverse,vorticity,row),cfg


def saved_solution(n):
    return OUT/f"fd{n}_k128_rate384_pow2.npz"


def run(n):
    artifact = saved_solution(n)
    row_path = artifact.with_suffix(".json")
    if row_path.exists():
        row=base.read(row_path)
        if base.digest(artifact)!=row["sha256"]:
            raise RuntimeError("Frozen completed trajectory changed")
        return row
    if artifact.exists():
        raise RuntimeError(f"Inspect uncommitted trajectory {artifact.name}")
    sol,cfg = march(n)
    tmp=artifact.with_suffix(".tmp.npz")
    case.store_solution(tmp,sol,cfg)
    tmp.replace(artifact)
    row = dict(n=n,configuration=cfg,metadata=sol.row,class_name=CLASS,wake=base.wake(sol),
               artifact=artifact.name,sha256=base.digest(artifact))
    base.save(row_path,row)
    print("COMPLETE",n,sol.row["seconds"],row["wake"]["strouhal"],flush=True)
    return row


def cross_phase(left,right):
    a,b = base.wake(left),base.wake(right)
    shifts = {}
    for name in ("velocity","vorticity"):
        shifts[name]=[]
        for i,(p,q) in enumerate(zip(a[name],b[name])):
            x,y=map(np.asarray,(p["positive_crossing_times"],q["positive_crossing_times"]))
            # Compare crossings by proximity without modifying any field time.
            diffs=[float(t-y[np.argmin(abs(t-y))]) for t in x if len(y)]
            shifts[name].append(dict(probe_index=i,raw_crossing_time_differences=diffs,
                                     max_abs=max(map(abs,diffs),default=None)))
    return dict(strouhal_left=a["strouhal"],strouhal_right=b["strouhal"],crossings=shifts)


def compare(left,right):
    grids=base.grid_set()
    truth=case.reference_fields(right,grids)
    scored=case.score(left,truth,grids)
    s,theta,w = grids["layer144"]
    outer = []
    for ds in (.018,.12):
        mask=s>=case.LENGTH-ds
        for i,t in enumerate(left.times):
            u,v,o=left.fields(i,s[mask],theta)
            U,V,O=[a[mask] for a in truth["layer144"][i]]
            def relative(e,r):
                return float(np.sqrt((w[mask]@e.sum(axis=1))/(w[mask]@r.sum(axis=1))))
            outer.append(dict(log_depth=ds,time=float(t),
                velocity=relative((u-U)**2+(v-V)**2,U**2+V**2),
                vorticity=relative((o-O)**2,O**2)))
    return dict(worst=case.maxima(scored),scores=scored,outer_layer=outer,phase=cross_phase(left,right))


def source_manifest():
    names=("fd_pow2_long.py","fd_cpu_buffered.py","fd_cpu_buffered.c","fd_cpu_buffered.dll",
           "fd_equilibration_pow2.py","fd_equilibration_precision.py","cpu_formal_fd_equilibrate.py")
    return base.source_hashes() | {f"rfm_v7/{name}":base.digest(HERE/name) for name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2,choices=(2,))
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol=base.read(PROTOCOL)
    originals=protocol["reference_path"]
    fixed=dict(n=list(SPECS),rate=RATE,horizon=HORIZON,grid=GRID,checkpoints=list(case.SNAPSHOTS),
        class_name=CLASS,source_hashes=source_manifest(),scaling="FP64 powers of two, rows only",
        selection="Forward and unscaled backward errors versus Decimal oracle on matching FD384/448 matrices",
        no_timing_claim=True,unscaled_inputs={spec["source"]:base.digest(base.WORK/spec["source"])
            for spec in (*originals[:2],originals[-1])})
    if JOURNAL.exists():
        journal=base.read(JOURNAL)
        if journal["frozen"]!=fixed:
            raise RuntimeError("Frozen experiment changed")
    else:
        journal=dict(status="running",parent_pid=os.getpid(),frozen=fixed,rows=[])
        base.save(JOURNAL,journal)
    if journal["status"]=="complete":
        return
    completed={r["n"] for r in journal["rows"]}
    remaining=[n for n in SPECS if n not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,n) for n in remaining]
        for job in as_completed(futures):
            journal["rows"].append(job.result())
            base.save(JOURNAL,journal)
    if source_manifest()!=fixed["source_hashes"]:
        raise RuntimeError("Source changed while equilibrated trajectories ran")
    old384,old448,newest=(base.open_reference(spec,protocol) for spec in (*originals[:2],originals[-1]))
    eq384,eq448=(case.load_solution(saved_solution(n),Pow2EquilibratedBufferedFD) for n in SPECS)
    checks={"old_radial":compare(old384,old448),"pow2_radial":compare(eq384,eq448),
            "pow2_n384_vs_original":compare(eq384,old384),"pow2_n448_vs_original":compare(eq448,old448),
            "pow2_n384_vs_latest_unscaled":compare(eq384,newest),
            "pow2_n448_vs_latest_unscaled":compare(eq448,newest)}
    base.save(OUT/"comparisons.json",checks)
    journal.update(status="complete",comparisons="comparisons.json")
    base.save(JOURNAL,journal)
    print("RESULT",{name:row["worst"] for name,row in checks.items()},flush=True)


if __name__=="__main__":
    main()
