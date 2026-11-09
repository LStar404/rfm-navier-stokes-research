"""Draft preregistration and resumable runner for fixed-space SBDF2 rate selection."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/"protocol_step_screen_draft.json"
OUT = HERE/"cpu_formal_step_screen_output"
# These paths feed solver dispatch, construction, stepping or the native
# runtime. Reference-audit and historical benchmark drivers are provenance
# of the anchor experiment rather than required dependencies of its march.
ANCHOR_RUNTIME_FILES = (
    "rfm_v7/protocol_cpu.json", "rfm_v7/cpu_formal.py", "rfm_v7/joint_hardware.py",
    "rfm_v7/cpu_implicit_stride.py", "rfm_v7/cpu_profile.py",
    "rfm_v7/basis_recurrence.py", "rfm_v7/basis_recurrence_model.py",
    "rfm_v7/galerkin_cuda_native.py", "rfm_v7/galerkin_cuda_native.cu",
    "rfm_v7/cuda_environment_native.py", "rfm_v7/fd_cuda_native.py", "rfm_v7/fd_cuda_native.cu",
    "rfm_v7/fd_cpu_buffered.py", "rfm_v7/fd_cpu_buffered.c",
    "rfm_v7/fd_cpu_buffered.dll" if sys.platform=="win32" else "rfm_v7/fd_cpu_buffered.so",
    "rfm_v6/full_case.py", "rfm_v6/fd_full.py", "rfm_v6/fd_full_band.c",
    "rfm_v6/fd_full_band.dll" if sys.platform=="win32" else "rfm_v6/fd_full_band.so",
    "rfm_v6/full_galerkin.py", "rfm_v6/full_galerkin_buffered.py", "rfm_v6/full_nonlinear.py",
    "rfm_v6/radial_layers.py", "rfm_v3/cylinder_case.py", "rfm_v3/cylinder_rfm.py",
    "rfm_v3/cylinder_fd.py",
)


def work_counts(rate, horizon, layout):
    steps = rate*horizon
    evaluations = steps-1
    inverse_fields,forward_fields = layout["inverse_fields"],layout["forward_fields"]
    radial_lines = layout["radial_lines"]
    return dict(steps=steps,nonlinear_evaluations=evaluations,
                inverse_fft_batches=evaluations,forward_fft_batches=evaluations,
                total_marching_fft_batches=2*evaluations,
                inverse_field_transforms=inverse_fields*evaluations,
                forward_field_transforms=forward_fields*evaluations,
                inverse_radial_line_transforms=inverse_fields*evaluations*radial_lines,
                forward_radial_line_transforms=forward_fields*evaluations*radial_lines)


def planned(protocol):
    jobs = []
    for name in protocol["methods"]:
        seeds = protocol["rfm_seeds"] if name.startswith("rfm") else (protocol["deterministic_seed"],)
        for seed in seeds:
            for rate in protocol["rates"]:
                jobs.append(dict(name=name,seed=seed,rate=rate))
    return jobs


class Screen:
    def __init__(self, protocol, rerun_anchor):
        import fd_pow2_reference_chain as chain
        import joint_hardware as joint

        self.p,self.chain,self.joint = protocol,chain,joint
        self.base,self.case,self.np = joint.base,joint.base.case,joint.np
        self.jobs = planned(protocol)
        self.grids = self.base.grid_set()
        self.specs = {r["name"]:r for r in self.base.read(self.base.PROTOCOL)["variants"]
                      if r["name"] in protocol["methods"]}
        if (set(self.specs)!=set(protocol["methods"]) or len(self.jobs)!=28
                or len({(j["name"],j["seed"],j["rate"]) for j in self.jobs})!=len(self.jobs)
                or protocol["rates"]!=sorted(set(protocol["rates"]))
                or protocol["anchor_rate"]!=protocol["rates"][-1]
                or len(set(protocol["rfm_seeds"]))!=5
                or any(rate*protocol["probe_spacing"]!=int(rate*protocol["probe_spacing"])
                       for rate in protocol["rates"])
                or list(self.case.SNAPSHOTS)!=protocol["checkpoints"]
                or any(self.specs[n]["rate"]!=protocol["anchor_rate"] for n in self.specs)
                or joint.GRAPH_STEPS!=protocol["graph_steps"]
                or protocol["backend"]!="cuda_graph" or protocol["cuda_fd_linear"]!="dense"):
            raise RuntimeError("Draft differs from the validated implementation")
        self.solve_protocol = self.base.read(self.base.PROTOCOL)
        self.solve_protocol["horizon"] = self.p["horizon"]
        self.reference_hashes,self.reference_gap,self.truth = self.reference_data()
        try:
            self.anchor,self.anchor_reason,self.anchor_provenance = self.anchor_data()
        except (RuntimeError,ValueError,KeyError,OSError) as exc:
            if not rerun_anchor:
                raise RuntimeError(f"Exact rate-128 reuse unavailable ({exc}). "
                    "Inspect the cause before requesting explicit anchor remeasurement.") from exc
            self.anchor,self.anchor_reason,self.anchor_provenance = {},str(exc),None

    def reference_data(self):
        p = self.p
        journal_path = HERE/p["reference_manifest"]
        audit_path = HERE/p["reference_audit"]
        record,audit = map(self.base.read,(journal_path,audit_path))
        rp = record["frozen"]["protocol"]
        if (record["status"]!="complete" or not audit["independent_axes"]
                or [x["axis"] for x in audit["edges"]]!=["radial","angle","time"]
                or rp["checkpoints"]!=p["checkpoints"]):
            raise RuntimeError("Complete, independent scaled reference ladder required")
        for a,b,axis in zip(rp["reference_path"],rp["reference_path"][1:],("n","k","rate")):
            if [k for k in ("n","k","rate") if a[k]!=b[k]]!=[axis] or a[axis]>=b[axis]:
                raise RuntimeError("Scaled reference edge changes more than its registered axis")
        edge_gap = 1.
        times = {(g,t) for g in self.grids for t in p["checkpoints"]}
        for edge in audit["edges"]:
            scores = edge["scores"]
            if (len(scores)!=len(times) or {(r["grid"],r["time"]) for r in scores}!=times
                    or self.case.maxima(scores)!=edge["worst"]):
                raise RuntimeError("Scaled reference edge score rows or maxima changed")
            edge_gap *= 1+max(edge["worst"].values())
        gap = edge_gap-1
        if (abs(gap-audit["composed_difference"])>1e-13 or
                gap>p["gate"]*p["reference_budget_fraction"]):
            raise RuntimeError("Reference ladder does not meet this gate's empirical budget")
        solutions = self.chain.references(rp,record["rows"])
        slots = {spec["name"]:i for i,spec in enumerate(rp["reference_path"])}
        truths = {n:self.case.reference_fields(solutions[slots[n]],self.grids) for n in p["reference_names"]}
        identities = {path.relative_to(self.base.WORK).as_posix():self.base.digest(path)
                      for path in (journal_path,audit_path)}
        identities.update({spec["source"]:self.base.digest(self.base.WORK/spec["source"])
                           for spec in rp["reference_path"]})
        return identities,gap,truths

    def class_and_config(self, job):
        spec = dict(self.specs[job["name"]],rate=job["rate"])
        _,config = self.base.config(spec,self.solve_protocol,job["seed"])
        model = self.base.FullFD if spec["method"]=="fd4" else self.joint.RecurrenceGalerkin
        return model,config

    def expected_times(self, solution, job):
        p = self.p
        probes = self.np.arange(0.,p["horizon"]+.01*p["probe_spacing"],p["probe_spacing"])
        if (not self.np.array_equal(solution.times,p["checkpoints"])
                or not self.np.array_equal(solution.probe_times,probes)
                or solution.row["steps"]!=p["horizon"]*job["rate"]
                or solution.row["time_order"]!=2):
            raise RuntimeError("Time integrator, output grid or step count changed")
        self.layout(job,solution)

    def layout(self, job, solution=None):
        frozen = self.p["march_fft_layouts"][job["name"]]
        if solution is None:
            return frozen
        model = solution.model
        is_fd = self.specs[job["name"]]["method"]=="fd4"
        # Mirrors native plan_many definitions. Galerkin counts actual
        # quadrature coordinates; FD counts the band's interior columns.
        inverse,forward = (4,1) if is_fd else (3,2)
        radial = model.rhs_shape[1] if is_fd else len(model.r)
        observed = dict(radial_lines=radial,inverse_fields=inverse,forward_fields=forward)
        if observed!=frozen:
            raise RuntimeError(f"Reconstructed FFT shape differs for {job['name']}: {observed}")
        return observed

    def verify_artifact(self, path, digest, job):
        if self.base.digest(path)!=digest:
            raise RuntimeError(f"Saved trajectory hash changed: {path.name}")
        cls,expected = self.class_and_config(job)
        with self.np.load(path,allow_pickle=False) as saved:
            if json.loads(str(saved["config"]))!=expected:
                raise RuntimeError(f"Saved trajectory configuration changed: {path.name}")
        solution = self.case.load_solution(path,cls)
        self.expected_times(solution,job)
        return solution

    def anchor_data(self):
        p = self.p
        path = HERE/p["anchor_manifest"]
        old = self.base.read(path)
        freeze = old["frozen"]
        template,prior_protocol = freeze["template"],freeze["protocol"]
        runtime = {n:self.base.digest(self.base.WORK/n) for n in ANCHOR_RUNTIME_FILES}
        recorded = freeze["source_hashes"]
        if (old["status"]!="complete" or any(recorded.get(n)!=sha for n,sha in runtime.items())
                or template["graph_steps"]!=p["graph_steps"] or template["cuda_fd_linear"]!=p["cuda_fd_linear"]
                or template["gate"]!=p["gate"] or prior_protocol["horizon"]!=p["horizon"]
                or prior_protocol["checkpoints"]!=p["checkpoints"]
                or prior_protocol["seeds"]!=p["rfm_seeds"]
                or prior_protocol["deterministic_seed"]!=p["deterministic_seed"]
                or any(self.reference_hashes.get(n)!=sha
                       for n,sha in freeze["reference"]["reference_hashes"].items())
                or {c["name"] for c in freeze["choices"]}!=set(p["methods"])):
            raise RuntimeError("Original joint run is incomplete or implementation differs")
        matches = {}
        for job in self.jobs:
            if job["rate"]!=p["anchor_rate"]:
                continue
            for backend in (p["backend"],"cpu"):
                rows = sorted((r for r in old["rows"] if r["name"]==job["name"] and
                               r["seed"]==job["seed"] and r["backend"]==backend),key=lambda r:r["block"])
                if not rows or not rows[0]["implementation_pass"] or not rows[0]["numerical_pass"]:
                    raise RuntimeError(f"No validated first anchor row for {job}, {backend}")
                r = rows[0]
                artifact = path.parent/r["artifact"]
                sol = self.verify_artifact(artifact,r["sha256"],job)
                meta = sol.row
                is_fd = self.specs[job["name"]]["method"]=="fd4"
                if backend==p["backend"] and (meta["graph_steps"]!=p["graph_steps"]
                        or (is_fd and meta["cuda_linear_solve"]!=p["cuda_fd_linear"])):
                    raise RuntimeError("Saved CUDA plan differs")
                if (is_fd and backend=="cpu" and meta.get("implementation")!="cpu-fused-fd4-csr-avx2"
                        or not is_fd and meta.get("basis_evaluation")!="differentiated-recurrence"):
                    raise RuntimeError("Saved radial implementation differs")
                matches[self.tag(job,backend)] = dict(artifact=artifact.relative_to(HERE).as_posix(),
                    sha256=r["sha256"],block=r["block"],metadata=meta)
        provenance = dict(manifest=path.relative_to(self.base.WORK).as_posix(),
            sha256=self.base.digest(path),runtime_source_hashes=runtime,
            additional_anchor_source_hashes={n:sha for n,sha in recorded.items() if n not in runtime})
        return matches,"Validated equilibrated-reference joint anchors, earliest matching block.",provenance

    @staticmethod
    def tag(job, backend):
        return f"{backend}_{job['name']}_{job['seed']}_r{job['rate']}"

    def work(self, job, solution=None):
        return work_counts(job["rate"],self.p["horizon"],self.layout(job,solution))

    def score(self, solution):
        scores = {n:self.case.score(solution,truth,self.grids) for n,truth in self.truth.items()}
        worst = {n:self.case.maxima(rows) for n,rows in scores.items()}
        wake = self.base.wake(solution)
        passes = (all(max(row.values())<=self.p["gate"] for row in worst.values())
                  and wake["sustained_alternating_probe_signs"])
        return dict(scores=scores,worst=worst,wake=wake,direct_pass=bool(passes))

    def trial(self, job, backend):
        tag = self.tag(job,backend)
        record = OUT/f"{tag}.json"
        if record.exists():
            old = self.base.read(record)
            if old["job"]!=job or old["backend"]!=backend:
                raise RuntimeError("Saved candidate journal differs from frozen schedule")
            if "artifact" in old and self.base.digest(HERE/old["artifact"])!=old["sha256"]:
                raise RuntimeError("Saved candidate hash changed")
            return old
        item = dict(job=job,backend=backend,planned_work=self.work(job))
        if tag in self.anchor:
            reuse = self.anchor[tag]
            solution = self.verify_artifact(HERE/reuse["artifact"],reuse["sha256"],job)
            item.update(**reuse,reused=True,actual_work=self.work(job,solution),**self.score(solution))
        else:
            path = OUT/f"{tag}.npz"
            if path.exists():
                raise FileExistsError(f"Inspect uncommitted trajectory before restart: {path.name}")
            start = time.perf_counter()
            try:
                spec = dict(self.specs[job["name"]],rate=job["rate"])
                solution,config = self.joint.solve(spec,self.solve_protocol,job["seed"],backend)
                seconds = time.perf_counter()-start
                self.expected_times(solution,job)
                scored = self.score(solution)
                temp = path.with_suffix(".tmp.npz")
                self.case.store_solution(temp,solution,config)
                temp.replace(path)
                item.update(artifact=path.relative_to(HERE).as_posix(),sha256=self.base.digest(path),
                    metadata=solution.row,reused=False,screen_seconds=seconds,actual_work=self.work(job,solution),
                    **scored)
            except (ArithmeticError,ValueError,RuntimeError,OSError,self.np.linalg.LinAlgError) as exc:
                item.update(failure=str(exc),failure_seconds=time.perf_counter()-start,direct_pass=False)
        self.base.save(record,item)
        print("RATE",tag,item["direct_pass"],item.get("worst",item.get("failure")),flush=True)
        return item

    def solution(self, item):
        return self.verify_artifact(HERE/item["artifact"],item["sha256"],item["job"])

    def trends(self, gpu):
        p = self.p
        indexed = {(r["job"]["name"],r["job"]["seed"],r["job"]["rate"]):r for r in gpu}
        keys = sorted({(job["name"],job["seed"]) for job in self.jobs})
        result = {}
        for name,seed in keys:
            seq = {r:indexed[name,seed,r] for r in p["rates"]}
            anchor = seq[p["anchor_rate"]]
            gaps,bounds = {},{}
            if anchor["direct_pass"]:
                anchor_fields = self.case.reference_fields(self.solution(anchor),self.grids)
                for rate,item in seq.items():
                    if not item["direct_pass"]:
                        continue
                    delta = self.case.score(self.solution(item),anchor_fields,self.grids)
                    gaps[rate] = self.case.maxima(delta)
                    bounds[rate] = {}
                    for ref,errors in anchor["scores"].items():
                        upper = []
                        for x,y in zip(delta,errors,strict=True):
                            if (x["grid"],x["time"])!=(y["grid"],y["time"]):
                                raise RuntimeError("Time/grid sequence changed")
                            upper.append(dict(grid=x["grid"],time=x["time"],
                                **{k:x[k]*(1+y[k])+y[k] for k in gaps[rate]}))
                        bounds[rate][ref] = self.case.maxima(upper)
            for rate in p["rates"]:
                refined = [q for q in p["rates"] if q>=rate]
                monotone = all(q in gaps for q in refined) and all(
                    gaps[b][k]<=gaps[a][k]+p["trend_slack"]
                    for a,b in zip(refined,refined[1:]) for k in gaps[a])
                bound_ok = rate in bounds and all(
                    max(metrics.values())<=p["gate"] for metrics in bounds[rate].values())
                result[self.tag(seq[rate]["job"],p["backend"])] = dict(job=seq[rate]["job"],
                    direct_pass=seq[rate]["direct_pass"],anchor_gap=gaps.get(rate),
                    triangle_worst=bounds.get(rate),trend_pass=monotone,
                    cancellation_guard_pass=bound_ok,eligible=bool(monotone and bound_ok))
        return result

    def cpu_match(self, gpu, cpu):
        if not cpu["direct_pass"]:
            return dict(passed=False,reason="CPU direct gate or run failed")
        a,b = self.solution(cpu),self.solution(gpu)
        fields = self.case.maxima(self.case.score(b,self.case.reference_fields(a,self.grids),self.grids))
        probes = {name:float(self.np.linalg.norm(getattr(a,name)-getattr(b,name))/
                   max(self.np.linalg.norm(getattr(a,name)),self.np.finfo(float).tiny))
                  for name in ("transverse","vorticity")}
        return dict(passed=bool(max(*fields.values(),*probes.values())<=self.p["implementation_gate"]),
                    field_errors=fields,probe_errors=probes)

    def confirm(self, gpu, trends):
        p = self.p
        reports = []
        for name in p["methods"]:
            tested,choice = [],None
            for rate in p["rates"]:
                rows = [r for r in gpu if r["job"]["name"]==name and r["job"]["rate"]==rate]
                if not all(trends[self.tag(r["job"],p["backend"])]["eligible"] for r in rows):
                    continue
                checks = [dict(seed=r["job"]["seed"],**self.cpu_match(r,self.trial(r["job"],"cpu")))
                          for r in rows]
                ok = all(r["passed"] for r in checks)
                tested.append(dict(rate=rate,passes=ok,checks=checks))
                if ok:
                    choice = rate
                    break
            record = dict(name=name,selected_rate=choice,cpu_confirmation=tested)
            if choice is not None:
                counts = self.work(dict(name=name,rate=choice))
                anchor = self.work(dict(name=name,rate=p["anchor_rate"]))
                record.update(work=counts,savings_vs_rate128={k:anchor[k]-v for k,v in counts.items()})
            reports.append(record)
        return reports

    def sources(self):
        src = self.joint.hashes() | self.chain.fingerprint()
        src.update({f"rfm_v7/{path.name}":self.base.digest(path) for path in (PROTOCOL,Path(__file__))})
        return src

    def execute(self):
        p = self.p
        src = self.sources()
        rng = self.np.random.default_rng(p["order_seed"])
        order = [self.jobs[i] for i in rng.permutation(len(self.jobs))]
        freeze = dict(protocol=p,specs=self.specs,sources=src,references=self.reference_hashes,
            reference_gap=self.reference_gap,anchor=self.anchor,anchor_note=self.anchor_reason,
            anchor_provenance=self.anchor_provenance,order=order,
            timing_scope="Diagnostic wall times; other host work is allowed. No timing-based selection.")
        OUT.mkdir(exist_ok=True)
        manifest = OUT/"freeze.json"
        if manifest.exists():
            if self.base.read(manifest)!=freeze:
                raise RuntimeError("The frozen rate experiment changed")
        else:
            self.base.save(manifest,freeze)
        cpu = self.base.pin()
        gpu = [self.trial(job,p["backend"]) for job in order]
        cpu_anchors = [self.trial(job,"cpu") for job in order if job["rate"]==p["anchor_rate"]]
        trends = self.trends(gpu)
        selected = self.confirm(gpu,trends)
        if self.sources()!=src or any(self.base.digest(self.base.WORK/path)!=value
                                      for path,value in self.reference_hashes.items()):
            raise RuntimeError("Frozen source or reference changed during rate screening")
        self.base.save(OUT/"qualification.json",dict(status="complete",freeze_sha256=self.base.digest(manifest),
            logical_cpu=cpu,gpu=gpu,cpu_anchors=cpu_anchors,guard=trends,selected=selected,
            scope="Qualification and work counts, not an isolated wall-time ranking."))
        print("SELECTED",selected,flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--execute-screen",action="store_true",
                     help="Freeze then execute the diagnostic qualification, not a formal benchmark.")
    cli.add_argument("--remeasure-anchor",action="store_true",
                     help="Only if exact matched rate-128 reuse is impossible.")
    args=cli.parse_args()
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not args.execute_screen:
        print(json.dumps(dict(protocol=protocol,gpu_jobs=planned(protocol),
            planned_counts={n:{r:work_counts(r,protocol["horizon"],protocol["march_fft_layouts"][n])
                               for r in protocol["rates"]} for n in protocol["methods"]}),
            indent=2,ensure_ascii=False))
        return
    Screen(protocol,args.remeasure_anchor).execute()


if __name__=="__main__":
    main()
