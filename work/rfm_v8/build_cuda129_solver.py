"""Add an explicit runtime injection point to the unchanged lean Galerkin march."""

import ast
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE/"galerkin_cuda_prepared_minimal.py"
TARGET = HERE/"galerkin_cuda_prepared_cuda129.py"
AUDIT = HERE/"cuda129_solver_source_audit.json"
SUBSTITUTIONS = (
    ("graph_steps=24, toolkit=None, **config):","graph_steps=24, toolkit=None, runtime_factory=None, **config):"),
    ("        rt = resources.enter_context(runtime_type(toolkit))",
     "        runtime = runtime_type(toolkit) if runtime_factory is None else runtime_factory(toolkit)\n"
     "        rt = resources.enter_context(runtime)"),
)


def main():
    original = SOURCE.read_bytes()
    body = original.decode("utf-8")
    for old,new in SUBSTITUTIONS:
        if body.count(old)!=1:
            raise ValueError("The lean CUDA template changed")
        body = body.replace(old,new)
    ast.parse(body)
    restored = body
    for old,new in reversed(SUBSTITUTIONS):
        if restored.count(new)!=1:
            raise ValueError("Ambiguous runtime-injection whitelist")
        restored = restored.replace(new,old)
    if restored.encode("utf-8")!=original:
        raise ValueError("CUDA march did not restore to original source")
    encoded = body.encode("utf-8")
    with TARGET.open("xb") as output:
        output.write(encoded)
    with AUDIT.open("x",encoding="utf-8") as output:
        json.dump(dict(status="passed",original_sha256=sha256(original).hexdigest(),target_sha256=sha256(encoded).hexdigest(),
                       restored_byte_exact=True,change="Optional runtime factory only; march, outputs and timing boundaries unchanged."),
                  output,indent=2)
        output.write("\n")


if __name__=="__main__":
    main()
