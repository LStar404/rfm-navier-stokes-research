"""Fixed-step Galerkin accuracy pilot for a genuinely sparser nonlinear cadence."""

import argparse
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent / "rfm_v7"))

from cpu_sbdf2_switch import RecurrenceGalerkin, case, np  # noqa: E402
import cpu_formal as base  # noqa: E402
from full_galerkin import real_apply  # noqa: E402
import switch_qualification as previous  # noqa: E402
from verify_switch_qualification import absolute_reference, fields_at, measured, maxima, require  # noqa: E402


OUT = HERE / "nonlinear_stride2_pilot_output"
T16 = (.5,1.,2.,4.,6.,8.,8.125,8.5,9.,10.,12.,16.)
METRICS = ("velocity","vorticity","near_velocity","near_vorticity")
PROBES = ("transverse","vorticity")
SWITCH_TIME = 8


class CountedGalerkin(RecurrenceGalerkin):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original = self.kernel
        self.nonlinear_calls = 0

        def count(*fields):
            self.nonlinear_calls += 1
            return original(*fields)

        self.kernel = count

    def mass_only(self,state,amplitude):
        """The full frozen mass application, including both lifting and mean mode."""
        mass = np.empty_like(state)
        x = np.ascontiguousarray(state[:,1:])
        mass[:,1:] = real_apply(self.M0,x)+real_apply(self.M1,x)*self.k2
        mass[:,1:] += amplitude*self.M_lift
        mass[:,0] = 0.
        mass[:self.mean_width,0] = self.mean_mass@state[:self.mean_width,0].real
        return mass


def sparse_march(*, rate, horizon, snapshots, **config):
    steps, split = int(rate*horizon),int(rate*SWITCH_TIME)
    h = 1/rate
    times = np.asarray(snapshots)
    snap_steps = set(case.step_indices(times,h))
    probe_times = np.arange(0.,horizon+.001,.125)
    probe_steps = dict(zip(case.step_indices(probe_times,h),range(len(probe_times)),strict=True))
    g = case.speed(h*np.arange(steps+1))
    started = time.perf_counter()
    model = CountedGalerkin(dt=h,**config)
    test = np.random.default_rng(811).standard_normal(model.rhs_shape).astype(np.complex128)
    reference_mass,_ = model.terms(test,g[split])
    require(np.array_equal(reference_mass,model.mass_only(test,g[split])),"Skipped-step mass differs from frozen operator")
    model.nonlinear_calls = 0
    setup = time.perf_counter()-started
    zero = np.zeros(model.rhs_shape,dtype=np.complex128)
    mass = prior_mass = zero
    flux = {0:zero}
    probes = {name:np.zeros((len(probe_times),len(case.PROBES))) for name in PROBES}
    states = []
    evaluated = []
    for n in range(steps):
        try:
            with np.errstate(over="raise",invalid="raise"):
                if n == 0:
                    rhs = mass+h*flux[0]
                else:
                    if n < split+2:
                        predicted = 2*flux[n]-flux[n-1]
                    elif (n-split)%2 == 0:
                        predicted = 1.5*flux[n]-.5*flux[n-2]
                    else:
                        predicted = 2*flux[n-1]-flux[n-3]
                    rhs = 2*mass-.5*prior_mass+h*predicted
                i = n+1
                state = model.advance(rhs,g[i],n == 0)
                if i in snap_steps:
                    require(np.isfinite(state).all(),"Nonfinite saved state")
                    states.append(state.copy())
                if i in probe_steps:
                    u,w = model.probes(state,g[i])
                    require(np.isfinite(u).all() and np.isfinite(w).all(),"Nonfinite probe")
                    j = probe_steps[i]
                    probes["transverse"][j],probes["vorticity"][j] = u,w
                if i == steps:
                    break
                prior_mass = mass
                if i <= split+2 or (i-split)%2 == 0:
                    mass,flux[i] = model.terms(state,g[i])
                    evaluated.append(i)
                else:
                    mass = model.mass_only(state,g[i])
                flux.pop(i-4,None)
        except (FloatingPointError, RuntimeError) as exc:
            raise ArithmeticError(f"Sparse advance stopped at n={n}, t={n*h:.9f}: {exc}") from exc
    expected = sum(i <= split+2 or (i-split)%2 == 0 for i in range(1,steps))
    require(expected == len(evaluated) == model.nonlinear_calls,"Real FFT call count differs from the schedule")
    row = dict(model.metadata,steps=steps,dt=h,horizon=horizon,nonlinear_evaluations=expected,
               marching_fft_batches=2*expected,setup_seconds=setup,seconds=time.perf_counter()-started,
               dense_evaluation_through=split+2,first_skipped_evaluation=split+3,time_order=2,
               nonlinear_stride=2,work_source="counted native FFT-kernel calls")
    return case.Solution(model,np.asarray(states),times,probe_times,probes["transverse"],probes["vorticity"],row)


def score_pair(control, sparse, refs, grids):
    times = control.times.tolist()
    c = fields_at(control,times,grids)
    s = fields_at(sparse,times,grids)
    delta = measured(s,c,grids,times)
    results = {}
    for name, truth in refs.items():
        anchor = measured(c,truth,grids,times)
        direct = measured(s,truth,grids,times)
        guard = {gt:{k:anchor[gt][k]+(1+anchor[gt][k])*delta[gt][k] for k in METRICS} for gt in delta}
        require(all(direct[gt][k] <= guard[gt][k]+2e-13 for gt in direct for k in METRICS),"Triangle bound violated")
        results[name] = dict(anchor_max=maxima(anchor),physical_max=maxima(direct),guard_max=maxima(guard),
            per_time={str(t):{k:max(direct[g,t][k] for g in grids) for k in METRICS} for t in times},
            direct_rows=[dict(grid=g,time=t,**v) for (g,t),v in direct.items()],
            guard_rows=[dict(grid=g,time=t,**v) for (g,t),v in guard.items()])
    early = control.times <= SWITCH_TIME
    probes = control.probe_times <= SWITCH_TIME
    exact = (control.states[early].tobytes() == sparse.states[early].tobytes() and
             all(getattr(control,k)[probes].tobytes() == getattr(sparse,k)[probes].tobytes() for k in PROBES))
    require(exact,"Subsampling unexpectedly changed an early state or probe")
    probe_difference = {k:float(np.linalg.norm(getattr(sparse,k)-getattr(control,k))/np.linalg.norm(getattr(control,k))) for k in PROBES}
    direct = all(max(v["physical_max"].values()) <= .005 for v in results.values())
    guard = all(max(v["guard_max"].values()) <= .005 for v in results.values())
    base_ok = all(max(v["anchor_max"].values()) <= .005 for v in results.values())
    return dict(early_bitwise_equal=exact,scheme_difference_max=maxima(delta),full_probe_difference=probe_difference,
                references=results,control_pass=base_ok,physical_pass=direct,guard_pass=guard,
                eligible_for_t72_followup=bool(base_ok and direct and guard))


def protocol():
    mid = previous.FROZEN
    high = previous.HERE / "multiscale_switch_qualification_output" / "protocol.json"
    jobs = []
    for name,rate,path in (("rfm37_k72",128,mid),("rfm41_k80",152,high)):
        p = base.read(path)
        jobs.append(dict(name=name,rate=rate,seed=7331,config=base.config(p["specs"][name],p["solve_protocol"],7331)[1]))
    inputs = [Path(__file__),mid,high,HERE/"nonlinear_stride2_scalar.json",
              previous.HERE/"cpu_sbdf2_switch.py",previous.HERE/"basis_recurrence_model.py",
              previous.base.V6/"full_case.py",previous.base.V6/"full_galerkin.py",
              previous.base.V6/"full_nonlinear.py",previous.HERE/"verify_switch_qualification.py"]
    return dict(status="frozen",horizon=16.,observations=T16,grid_count=3,physical_gate=.005,
                candidate_rule="Keep both warm-up updates after t=8, then evaluate only even indices; use two true flux samples",
                control_rule="Frozen recurrence Galerkin fixed-step case.solve",
                sources={p.relative_to(HERE.parent).as_posix():base.digest(p) for p in inputs},jobs=jobs)


def run(index):
    frozen = base.read(OUT/"protocol.json")
    job = frozen["jobs"][index]
    d = OUT/job["name"]
    d.mkdir()
    config = job["config"]
    common = dict(horizon=frozen["horizon"],snapshots=frozen["observations"],**config)
    control = case.solve(CountedGalerkin,steps=16*job["rate"],**common)
    require(control.model.nonlinear_calls == control.row["steps"]-1,"Frozen control nonlinear count differs")
    with (d/"control.npz").open("xb") as handle:
        case.store_solution(handle,control,config)
    try:
        sparse = sparse_march(rate=job["rate"],**common)
    except ArithmeticError as exc:
        result = dict(job=job,status="failed",failure=str(exc),control_nonlinear_calls=control.model.nonlinear_calls,
                      eligible_for_t72_followup=False)
    else:
        with (d/"candidate.npz").open("xb") as handle:
            case.store_solution(handle,sparse,config)
        grids = base.grid_set()
        references,budget = absolute_reference(base.read(previous.FROZEN),grids)
        refs = {name:{g:field[:len(T16)] for g,field in values.items()} for name,values in references.items()}
        scores = score_pair(control,sparse,refs,grids)
        result = dict(job=job,status="complete",reference_budget=budget,control_nonlinear_calls=control.model.nonlinear_calls,
                      candidate_nonlinear_calls=sparse.model.nonlinear_calls,**scores)
    result["artifact_sha256"] = {p.name:base.digest(p) for p in d.glob("*.npz")}
    base.save(d/"qualification.json",result)
    print(job["name"],result["status"],result["eligible_for_t72_followup"],flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze",action="store_true")
    action.add_argument("--job",type=int,choices=(0,1))
    args = cli.parse_args()
    if args.freeze:
        require(not OUT.exists(),"Preserve prior pilot")
        base.save(OUT/"protocol.json",protocol())
    else:
        run(args.job)
