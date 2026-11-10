"""Generate a solver-only Galerkin import path from frozen v7 source."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V7 = HERE.parent/"rfm_v7"
AUDIT = HERE/"galerkin_minimal_source_audit.json"


def source(name):
    return (V7/name).read_text(encoding="utf-8")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old}")
    return text.replace(old,new)


def definitions(text):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            result[node.name] = node
        elif isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            result[node.targets[0].id] = node
    return result


def extract(text, names):
    by_name = definitions(text)
    lines = text.splitlines(keepends=True)
    fragments = []
    for name in names:
        node = by_name[name]
        begin = min([node.lineno]+[d.lineno for d in getattr(node,"decorator_list",[])])
        fragments.append("".join(lines[begin-1:node.end_lineno]).rstrip())
    return "\n\n\n".join(fragments)+"\n"


def same_definition(reference, candidate, name):
    return ast.dump(definitions(reference)[name],include_attributes=False) == ast.dump(
        definitions(candidate)[name],include_attributes=False)


def generate():
    recurrence = source("basis_recurrence.py")
    stride = source("cpu_implicit_stride.py")
    model = source("basis_recurrence_model.py")
    sbdf = source("cpu_sbdf2_switch.py")
    variable = source("galerkin_cuda_variable.py")
    prepared = source("galerkin_cuda_prepared.py")
    shared = source("galerkin_phase_shared.py")
    model_header = '''"""Numerical recurrence model with no CPU profiling or qualification imports."""

from contextlib import contextmanager
from pathlib import Path
import sys

V7 = Path(__file__).resolve().parents[1] / "rfm_v7"
V6 = V7.with_name("rfm_v6")
for directory in (V7, V6):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import full_case as case
import full_galerkin_buffered as buffered
import cylinder_rfm
from radial_layers import layer_jets

np = case.np


'''
    basis_names = ("unit_legendre_jets","polynomial","mean_jets")
    basis = extract(recurrence,basis_names).replace("base.case.LENGTH","case.LENGTH")
    model_names = ("RecurrenceGalerkin","native_model")
    model_fragment = extract(model,model_names)
    aliases = (("rec.cylinder_rfm.case.LENGTH","cylinder_rfm.case.LENGTH"),
               ("rec.base.case.LENGTH","case.LENGTH"),("rec.polynomial(","polynomial("),
               ("rec.mean_jets(","mean_jets("))
    for old,new in aliases:
        model_fragment = replace_once(model_fragment,old,new)
    lean_model = model_header+basis+"\n\n"+extract(stride,("StridedGalerkin",))+"\n\n"+model_fragment
    schedule_header = '''"""Numerical SBDF2 grid and bridge weights without CPU solver imports."""

from dataclasses import dataclass
from pathlib import Path
import sys

V6 = Path(__file__).resolve().parents[1] / "rfm_v6"
if str(V6) not in sys.path:
    sys.path.insert(0, str(V6))

import full_case as case

np = case.np


'''
    schedule_names = ("EARLY","BRIDGE","LATE","SNAPSHOTS","StepSchedule","step_schedule","bridge_weights")
    lean_schedule = schedule_header+extract(sbdf,schedule_names)
    old_schedule = "from cpu_sbdf2_switch import SNAPSHOTS, bridge_weights, step_schedule"
    lean_variable = replace_once(variable,old_schedule,
        "from sbdf2_schedule_minimal import SNAPSHOTS, bridge_weights, step_schedule")
    lean_variable = replace_once(lean_variable,"from basis_recurrence_model import RecurrenceGalerkin",
        "from galerkin_minimal_model import RecurrenceGalerkin, V7")
    lean_variable = replace_once(lean_variable,'source = Path(__file__).with_suffix(".cu")',
        'source = V7 / "galerkin_cuda_variable.cu"')
    lean_prepared = replace_once(prepared,old_schedule,
        "from sbdf2_schedule_minimal import SNAPSHOTS, bridge_weights, step_schedule")
    lean_prepared = replace_once(lean_prepared,"from galerkin_cuda_variable import ",
        "from galerkin_cuda_variable_minimal import ")
    lean_prepared = replace_once(lean_prepared,"from galerkin_phase_shared import ",
        "from galerkin_phase_shared_minimal import ")
    lean_shared = shared[:shared.index("\n\ndef compare_arrays")].rstrip()+"\n"
    lean_shared = replace_once(lean_shared,"from basis_recurrence_model import RecurrenceGalerkin",
        "from galerkin_minimal_model import RecurrenceGalerkin")
    lean_shared = replace_once(lean_shared,"from cpu_sbdf2_switch import bridge_weights",
        "from sbdf2_schedule_minimal import bridge_weights")
    check = {}
    for name in basis_names:
        check[name] = same_definition(recurrence.replace("base.case.LENGTH","case.LENGTH"),lean_model,name)
    check["StridedGalerkin"] = same_definition(stride,lean_model,"StridedGalerkin")
    for name in model_names:
        check[name] = same_definition(model_fragment,lean_model,name)
    for name in schedule_names:
        check[name] = same_definition(sbdf,lean_schedule,name)
    for name in ("SwitchRuntime","same_coordinates","phase_operators","bridge_launch","solve_cuda_switch"):
        check[name] = same_definition(variable.replace('source = Path(__file__).with_suffix(".cu")',
            'source = V7 / "galerkin_cuda_variable.cu"'),lean_variable,name)
    for name in ("PREPARATIONS","solve_cuda_prepared"):
        check[name] = same_definition(prepared,lean_prepared,name)
    for name in ("spatial_blocks","pack_step","phase_operators_shared"):
        check[name] = same_definition(shared,lean_shared,name)
    if not all(check.values()):
        raise ArithmeticError("A generated numerical definition differs from its approved source")
    generated = dict(galerkin_minimal_model=lean_model,sbdf2_schedule_minimal=lean_schedule,
        galerkin_cuda_variable_minimal=lean_variable,galerkin_cuda_prepared_minimal=lean_prepared,
        galerkin_phase_shared_minimal=lean_shared)
    dependencies = ("basis_recurrence.py","basis_recurrence_model.py","cpu_implicit_stride.py",
        "cpu_sbdf2_switch.py","galerkin_cuda_variable.py","galerkin_cuda_prepared.py","galerkin_phase_shared.py",
        "galerkin_cuda_native.py","galerkin_cuda_module_bundle.py","galerkin_cuda_variable.cu","galerkin_cuda_native.cu")
    evidence = dict(status="passed",definition_ast_exact_after_allowed_references=check,
        allowed_references=["basis_recurrence base.case.LENGTH → full_case.LENGTH",
            "RecurrenceGalerkin rec.* aliases → equivalent numerical globals",
            "CUDA variable source resolved against the original rfm_v7 directory"],
        omitted="CPU profiles, native CPU integration, qualification, archive statistics, wake diagnostics and plotting.",
        original_sha256={name:sha256((V7/name).read_bytes()).hexdigest() for name in dependencies},
        generated_sha256={name+".py":sha256(text.encode("utf-8")).hexdigest() for name,text in generated.items()})
    return generated,evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    generated,evidence = generate()
    evidence["generator_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    for name,text in generated.items():
        path = HERE/(name+".py")
        if args.check:
            if path.read_bytes() != text.encode("utf-8"):
                raise ValueError(f"Generated source differs: {name}")
        else:
            with path.open("xb") as stream:
                stream.write(text.encode("utf-8"))
    if args.check:
        if json.loads(AUDIT.read_text(encoding="utf-8")) != evidence:
            raise ValueError("Generated source audit differs")
    else:
        with AUDIT.open("x",encoding="utf-8") as stream:
            json.dump(evidence,stream,indent=2,ensure_ascii=False)
            stream.write("\n")
    print("PASS",len(evidence["definition_ast_exact_after_allowed_references"]),flush=True)


if __name__ == "__main__":
    main()
