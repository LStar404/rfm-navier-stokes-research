"""CUDA event probe of observation-only work at the actual production frame sizes."""

import ctypes as C
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
from statistics import median

from warp_observe_cuda import ObservationRuntime, P, bind, call
from basis_recurrence_model import RecurrenceGalerkin
from fd_cuda_native import FullFD, radial_operators
from galerkin_cuda_native import case, np, operators
import cpu_formal as base


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
OUT = HERE/"warp_observe_micro_output"
TEMPLATE = HERE/"protocol_warp_observe_micro.json"


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def reference(arm,mid,high):
    name,seed = arm["case"],arm["seed"]
    if arm["family"] == "fd":
        return (mid if name in mid["fd_controls"] else high)["fd_controls"][name]["gpu"]
    if arm["family"] == "uniform":
        return next(r["uniform_gpu"] for r in mid["gal_controls"] if r["name"] == name and r["seed"] == seed)
    return next(r["artifact"] for r in mid["prepared_evidence"] if r["job"]["name"] == name and
                r["job"]["seed"] == seed and r["job"]["preparation"] == "combined")


def host_fixture(arm,ref,t):
    path = HERE.parent/ref["path"]
    require(base.digest(path) == ref["sha256"],"Archived observation state changed")
    with np.load(path,allow_pickle=False) as raw:
        state = raw["states"][raw["times"].tolist().index(t)]
        cfg = json.loads(str(raw["config"]))
    if arm["family"] == "fd":
        model = FullFD(dt=1/128,**cfg)
        *_,weights,lift = radial_operators(model)
        state = np.ascontiguousarray(state.T)
        copies = 2
    else:
        model = RecurrenceGalerkin(dt=1/128,**cfg)
        *_,weights,lift = operators(model,1/128)
        copies = 3
    require(weights.size == 6*state.size and lift.size == 6 and np.isfinite(state).all(),"Observation weights do not match state")
    return dict(state=state,states=np.stack([state]*copies),weights=weights,lift=lift,
                frame=state.size,w=state.shape[0],modes=state.shape[1])


class Measure:
    def __init__(self,rt,stack,launch,graph_steps,repetitions):
        self.rt,self.launch,self.graph_steps,self.repetitions = rt,launch,graph_steps,repetitions
        self.graph = rt.capture(stack,launch,graph_steps)
        self.a,self.b = P(),P()
        dll = rt.driver_lib
        create = bind(dll,"cuEventCreate",[C.POINTER(P),C.c_uint])
        destroy = bind(dll,"cuEventDestroy_v2",[P])
        self.record = bind(dll,"cuEventRecord",[P,P])
        self.elapsed = bind(dll,"cuEventElapsedTime",[C.POINTER(C.c_float),P,P])
        for event in (self.a,self.b):
            call(create,C.byref(event),0)
            stack.callback(call,destroy,event)
        call(rt.driver.cuGraphLaunch,self.graph,rt.driver.stream)
        rt.sync()

    def measure(self):
        stream = self.rt.driver.stream
        call(self.record,self.a,stream)
        for _ in range(self.repetitions):
            call(self.rt.driver.cuGraphLaunch,self.graph,stream)
        call(self.record,self.b,stream)
        self.rt.sync()
        elapsed = C.c_float()
        call(self.elapsed,C.byref(elapsed),self.a,self.b)
        return 1000*elapsed.value/(self.graph_steps*self.repetitions)


def device_fixture(rt,stack,arm,host,scenario):
    probe = scenario != "no_probe_no_snapshot"
    save = scenario == "probe_and_snapshot"
    n = 1024 if probe else 1025
    stride = 16
    args = dict(states=host["states"],amplitudes=case.speed(np.arange(n+1)/128),index=np.array([n],dtype=np.int32),
                slots=np.full(n+1,-1,dtype=np.int32),probe_slots=np.full(n+1,-1,dtype=np.int32),
                weights=host["weights"],lift=host["lift"],checkpoints=np.zeros((1,host["frame"]),dtype=np.complex128),
                probes=np.zeros((n//stride+1,6),dtype=np.float64))
    if save:
        args["slots"][n] = 0
    if probe:
        args["probe_slots"][n] = n//stride
    d = {k:rt.upload(stack,value) for k,value in args.items()}
    first = d["states"],d["amplitudes"],d["index"],d["slots"]
    last = d["weights"],d["lift"],d["checkpoints"],d["probes"]
    if arm["family"] == "combined":
        params = (*first,d["probe_slots"],*last,host["frame"],0)
    elif arm["family"] == "fd":
        params = (*first,*last,host["w"],host["modes"],stride,0)
    else:
        params = (*first,*last,host["frame"],stride,0)
    launch = rt.launch(rt.observer,1,*params)

    def download():
        values = {k:np.empty_like(args[k]) for k in ("states","index","checkpoints","probes")}
        for k,value in values.items():
            rt.driver.download(d[k],value)
        reference_checkpoint = (host["state"].T if arm["family"] == "fd" else host["state"]).ravel()
        expected = reference_checkpoint if save else np.zeros_like(reference_checkpoint)
        require(values["states"].tobytes() == host["states"].tobytes() and values["index"][0] == n and
                values["checkpoints"][0].tobytes() == expected.tobytes(),"Observation modified state, index or checkpoint")
        return values["probes"]

    return launch,download,probe,n,args["amplitudes"][n].real


def check_values(serial,warp,host,probe,n,amplitude,gates):
    sample = n//16
    untouched = np.delete(warp,sample,axis=0)
    require(not np.count_nonzero(untouched) and np.isfinite(warp).all() and np.isfinite(serial).all(),"Wrong probe slot was updated")
    if not probe:
        require(not np.count_nonzero(warp) and not np.count_nonzero(serial),"Inactive sample wrote probes")
        return dict(relative_l2=0.,scaled_absolute=0.,max_absolute=0.,warp_vs_host_reference=0.)
    delta = warp[sample]-serial[sample]
    denominator = np.linalg.norm(serial[sample])
    require(denominator > 1e-12,"The selected archive does not exercise a nonzero probe")
    relative = float(np.linalg.norm(delta)/denominator)
    absolute = float(max(abs(delta)))
    scaled = float(max(abs(delta)/np.maximum(1.,abs(serial[sample]))))
    state = host["state"].real.ravel()
    expected = np.array([math.fsum(a*b for a,b in zip(w,state,strict=True))+amplitude*l
                         for w,l in zip(host["weights"].reshape(6,-1),host["lift"].ravel(),strict=True)])
    require(relative <= gates["tight_relative_l2_gate"] and relative <= gates["original_implementation_relative_gate"] and
            scaled <= gates["tight_scaled_absolute_gate"],"FP64 warp reduction exceeded a registered difference gate")
    host_norm = np.linalg.norm(expected)
    require(host_norm > 1e-12,"Independent host dot product is undefined")
    host_error = float(np.linalg.norm(warp[sample]-expected)/host_norm)
    require(host_error <= gates["tight_relative_l2_gate"],"Warp reduction differs from the independent host dot product")
    return dict(relative_l2=relative,scaled_absolute=scaled,max_absolute=absolute,warp_vs_host_reference=host_error)


def main():
    require(not OUT.exists(),"Preserve the existing CUDA observation diagnostic")
    p = base.read(TEMPLATE)
    mid_path,high_path = (V7/f"prepared_formal_timing_{s}_output"/"protocol.json" for s in ("mid","high"))
    mid,high = base.read(mid_path),base.read(high_path)
    refs = {a["name"]:reference(a,mid,high) for a in p["arms"]}
    files = [Path(__file__),TEMPLATE,HERE/"warp_observe_cuda.py",mid_path,high_path,
             *(V7/name for name in ("galerkin_cuda_native.cu","galerkin_cuda_variable.cu","fd_cuda_native.cu",
                 "galerkin_cuda_module_bundle.py","cuda_environment_native.py","fd_cuda_native.py","galerkin_cuda_native.py"))]
    sources = {f.relative_to(HERE.parent).as_posix():base.digest(f) for f in files}
    OUT.mkdir()
    base.save(OUT/"protocol.json",dict(status="registered_diagnostic",template=p,references=refs,source_hashes=sources))
    rows = []
    rng = random.Random(p["order_seed"])
    for arm in p["arms"]:
        try:
            host = host_fixture(arm,refs[arm["name"]],p["frozen_source_time"])
            with ExitStack() as resources:
                runtimes = {label:resources.enter_context(ObservationRuntime(arm["family"],warp))
                            for label,warp in (("serial",False),("warp",True))}
                for scenario in p["cases"]:
                    with ExitStack() as buffer_scope:
                        clocks,reads,observed = {},{},{}
                        for label,rt in runtimes.items():
                            launch,reads[label],probe,n,amplitude = device_fixture(rt,buffer_scope,arm,host,scenario)
                            clocks[label] = Measure(rt,buffer_scope,launch,p["graph_observations"],p["graph_repetitions_per_event"])
                        for i in range(p["event_pairs"]):
                            order = list(clocks)
                            rng.shuffle(order)
                            observed[i] = {label:clocks[label].measure() for label in order}
                        measured = check_values(reads["serial"](),reads["warp"](),host,probe,n,amplitude,p["accuracy"])
                    samples = list(observed.values())
                    row = dict(arm=arm,scenario=scenario,status="complete",frame=host["frame"],error=measured,
                               event_microseconds_per_observation=samples,
                               serial_median_us=median(r["serial"] for r in samples),warp_median_us=median(r["warp"] for r in samples),
                               paired_ratio=median(r["serial"]/r["warp"] for r in samples))
                    rows.append(row)
                    base.save(OUT/(arm["name"]+"_"+scenario+".json"),row)
                    print(arm["name"],scenario,row["paired_ratio"],flush=True)
        except (ArithmeticError,OSError,RuntimeError,ValueError) as exc:
            row = dict(arm=arm,status="failed",failure_type=type(exc).__name__,failure=str(exc))
            rows.append(row)
            base.save(OUT/(arm["name"]+"_failure.json"),row)
    require(all(base.digest(HERE.parent/name)==sha for name,sha in sources.items()),"Observation source changed")
    expected = len(p["arms"])*len(p["cases"])
    passed = len(rows) == expected and all(r["status"] == "complete" for r in rows)
    estimates = {}
    if passed:
        for arm in p["arms"]:
            steps = 8192 if arm["family"] == "combined" else 9216
            counts = dict(no_probe_no_snapshot=steps-576,probe_only=576-19,probe_and_snapshot=19)
            own = [r for r in rows if r["arm"] == arm]
            estimates[arm["name"]] = dict(observation_counts=counts,weighted_warp_minus_serial_microseconds=sum(
                counts[r["scenario"]]*(r["warp_median_us"]-r["serial_median_us"]) for r in own))
    base.save(OUT/"summary.json",dict(status="passed" if passed else "failed",rows=rows,protocol_sha256=base.digest(OUT/"protocol.json"),
                                  hot_kernel_extrapolation_only=estimates,isolated_speed_claim_supported=False))
    require(passed,"An observation microdiagnostic failed; original records preserved")


if __name__ == "__main__":
    main()
