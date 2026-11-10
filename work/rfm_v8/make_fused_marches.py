"""Generate audited same-march CUDA wrappers from unchanged parent functions."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
V7=WORK/"rfm_v7"
HEADER="""\"\"\"Generated matched warp-only and fused CUDA marches. See make_fused_marches.py.\"\"\"
from contextlib import ExitStack
import ctypes as C
import time

from fused_march_runtime import FusedMarchRuntime, nonlinear_metadata, nonlinear_nodes
from cuda_environment_native import I, P, bind, call
from galerkin_cuda_native import case, np, operators
from fd_cuda_native import FullFD, BandLaunch, EventTrace, radial_operators, precompute_dense
from galerkin_minimal_model import RecurrenceGalerkin as BufferedGalerkin
from sbdf2_schedule_minimal import SNAPSHOTS, bridge_weights, step_schedule
from galerkin_cuda_variable_minimal import bridge_launch
from galerkin_phase_shared_minimal import phase_operators_shared


"""


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def extract(path,name):
    source=path.read_text(encoding="utf-8")
    match=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name==name]
    require(len(match)==1,"Expected one frozen parent function")
    node=match[0]
    return "\n".join(source.splitlines()[node.lineno-1:node.end_lineno])+"\n"


class Transformation:
    def __init__(self,text):
        self.original=text
        self.text=text
        self.rules=[]

    def change(self,label,old,new):
        require(self.text.count(old)==1,f"Parent source changed at {label}")
        self.text=self.text.replace(old,new,1)
        self.rules.append(dict(label=label,before=old,after=new))

    def span(self,label,left,right,replacement):
        start=self.text.index(left)
        end=self.text.index(right,start+len(left))
        self.change(label,self.text[start:end],replacement)

    def verify(self):
        restored=self.text
        for rule in reversed(self.rules):
            require(restored.count(rule["after"])==1,f"Cannot invert {rule['label']}")
            restored=restored.replace(rule["after"],rule["before"],1)
        require(restored==self.original,"Generated march does not restore the complete parent function")
        return dict(parent_function_sha256=sha(self.original),generated_function_sha256=sha(self.text),
                    restored_parent_sha256=sha(restored),replacement_whitelist=self.rules)


def uniform(t):
    t.change("function entry","def solve_native(*, steps,","def solve_uniform_fused(*, engine, steps,")
    t.change("owned runtime argument","profile_steps=0, runtime=None, toolkit=None","profile_steps=0, toolkit=None")
    t.change("same-warp one-module runtime","Runtime(toolkit)) if runtime is None else runtime",
             'FusedMarchRuntime("uniform",engine,n,toolkit))')
    t.change("remove only fused physical staging","np.zeros((3, q, n)),\n            np.zeros((2, q, n)),",
             "None if rt.fused else np.zeros((3, q, n)),\n            None if rt.fused else np.zeros((2, q, n)),")
    t.change("optional staging upload","uploaded = [rt.upload(resources, a) for a in (",
             "uploaded = [rt.upload(resources, a) if a is not None else None for a in (")
    t.span("original FFT plan replaced by matched angular node","        inverse = rt.plan(","        form = rt.launch(",
           '        angular_nodes, prime, angular_names = nonlinear_nodes(rt,resources,"gal",n,k,q,spectrum,physical,\n'
           '                                                              products,transformed,previous)\n')
    t.span("original physical product now owned by angular node","        multiply = rt.launch(","        advance = rt.launch(",
           "        # Product and history run within the registered angular nodes.\n")
    t.span("nonlinear stage substitutions only","        stage_names = (","        def body():",
           '        stage_names = ("radial_reconstruction", *angular_names, "projection_and_advance", "probes_checkpoints_and_step")\n'
           "        stages = (form, *angular_nodes, advance, observe)\n\n")
    t.span("zero warm-up nodes","        # Materialize both","        first()\n",
           "        for node in prime:\n            node()\n")
    t.change("angular operation audit",'    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)',
             '    row.update(nonlinear_metadata(rt,steps-1,q,"gal"))\n'
             '    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)')


def combined(t):
    t.change("function entry",'def solve_cuda_prepared(*, preparation="combined",','def solve_combined_fused(*, engine, preparation="combined",')
    t.span("same existing shared preparation","    if preparation not in PREPARATIONS:","    if graph_steps < 0",
           '    if preparation != "combined":\n        raise ValueError("Only the qualified combined preparation is registered")\n'
           "    phase_builder = phase_operators_shared\n")
    t.change("same-warp one-module runtime","runtime_type(toolkit)",'FusedMarchRuntime("combined",engine,n,toolkit)')
    t.change("remove only fused physical staging","np.zeros((3, q, n)),\n                  np.zeros((2, q, n)),",
             "None if rt.fused else np.zeros((3, q, n)),\n                  None if rt.fused else np.zeros((2, q, n)),")
    t.change("optional staging upload","[rt.upload(resources, a) for a in arrays]",
             "[rt.upload(resources, a) if a is not None else None for a in arrays]")
    t.span("original FFT plan replaced by matched angular node","        inverse = rt.plan(","        form = rt.launch(",
           '        angular_nodes, prime, _ = nonlinear_nodes(rt,resources,"gal",n,k,q,spectrum,physical,\n'
           '                                                 products,transformed,previous)\n')
    t.span("original products now owned by angular node","        multiply = rt.launch(","        observe = rt.launch(",
           "        # Product and history run within the registered angular nodes.\n")
    t.change("nonlinear stage substitution only",
             "            call(rt.z2d, inverse, spectrum, physical)\n            multiply()\n            call(rt.d2z, forward, products, transformed)",
             "            for node in angular_nodes:\n                node()")
    t.change("zero warm-up nodes","        call(rt.z2d, inverse, spectrum, physical)\n        call(rt.d2z, forward, products, transformed)",
             "        for node in prime:\n            node()")
    t.change("angular operation audit","    return case.Solution(model, checkpoints, snapshots, schedule.probe_times, probes[:, 0], probes[:, 1], row)",
             '    row.update(nonlinear_metadata(rt,executed,q,"gal"))\n'
             "    return case.Solution(model, checkpoints, snapshots, schedule.probe_times, probes[:, 0], probes[:, 1], row)")


def fd(t):
    t.change("function entry","def solve_fd_native(*, steps,","def solve_fd_fused(*, engine, steps,")
    t.change("owned runtime argument","runtime=None, toolkit=None, **config","toolkit=None, **config")
    t.change("same-warp one-module runtime","FDRuntime(toolkit)) if runtime is None else runtime",
             'FusedMarchRuntime("fd",engine,n,toolkit))')
    t.change("remove only fused physical staging","np.zeros((4, w, n)), np.zeros((w, n)),",
             "None if rt.fused else np.zeros((4, w, n)), None if rt.fused else np.zeros((w, n)),")
    t.change("optional staging upload","[rt.upload(resources, a) for a in arrays]",
             "[rt.upload(resources, a) if a is not None else None for a in arrays]")
    t.span("original FFT plan replaced by matched angular node","        plan_inverse = rt.plan(","        form = rt.launch(",
           '        angular_nodes, prime, angular_names = nonlinear_nodes(rt,resources,"fd",n,modes-1,w,spectrum,physical,\n'
           '            products,transformed,previous,mean_terms=mean_terms,step=index,mean=mean)\n')
    t.span("original product and mean now owned by angular node","        product = rt.launch(","        assemble = rt.launch(",
           "        # Products, history and FD mean run within the registered angular nodes.\n")
    t.span("nonlinear stage substitutions only","        stage_names = (","        def body():",
           '        stage_names = ("radial_stencils", *angular_names, "rhs", "linear_solve", "probe_snapshot_step")\n'
           "        stages = (form, *angular_nodes, assemble, advance, observe)\n\n")
    t.span("zero warm-up nodes","        # Prime cuFFT","        first()\n","        for node in prime:\n            node()\n")
    t.change("angular operation audit","    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)",
             '    row.update(nonlinear_metadata(rt,steps-1,w,"fd"))\n'
             "    return case.Solution(model, checkpoints, times, probe_times, probes[:, 0], probes[:, 1], row)")


def build():
    entries=((V7/"galerkin_cuda_native.py","solve_native",uniform),
             (HERE/"galerkin_cuda_prepared_minimal.py","solve_cuda_prepared",combined),
             (V7/"fd_cuda_native.py","solve_fd_native",fd))
    code=[HEADER]
    audit=[]
    for path,name,mutate in entries:
        t=Transformation(extract(path,name))
        mutate(t)
        code.append(t.text+"\n\n")
        audit.append(dict(parent=path.relative_to(WORK).as_posix(),parent_file_sha256=sha256(path.read_bytes()).hexdigest(),
                          parent_function=name,**t.verify()))
    target="".join(code)
    ast.parse(target)
    return target,dict(status="passed",scope="Exact inverse restoration of parent solver functions after declared replacements",
                       target_sha256=sha(target),generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),functions=audit)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check",action="store_true")
    args=cli.parse_args()
    target,audit=build()
    source=HERE/"fused_marches.py"
    record=HERE/"fused_marches_generation.json"
    if args.check:
        require(source.read_bytes()==target.encode("utf-8") and json.loads(record.read_bytes())==audit,"Generated march evidence differs")
    else:
        require(not source.exists() and not record.exists(),"Preserve earlier march source and evidence")
        with source.open("xb") as handle:
            handle.write(target.encode("utf-8"))
        with record.open("x",encoding="utf-8",newline="\n") as handle:
            json.dump(audit,handle,indent=2)
            handle.write("\n")
    print("passed",audit["target_sha256"])
