"""Generate the combined Galerkin observation-only adapter from the frozen march."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/"rfm_v7"/"galerkin_cuda_prepared.py"
TARGET = HERE/"galerkin_cuda_prepared_observe.py"
MANIFEST = HERE/"cuda_observe_solver_generation.json"


def main():
    if TARGET.exists() or MANIFEST.exists():
        raise FileExistsError("Preserve the generated observation adapter")
    raw = SOURCE.read_bytes()
    body = raw.decode("utf-8")
    replacements = (
        ('import time\n','import time\nfrom warp_observe_cuda import ObservationRuntime\n'),
        ('def solve_cuda_prepared(*, preparation="combined",','def solve_cuda_prepared_observe(*, warp=False, preparation="combined",'),
        ('    runtime_type, phase_builder = PREPARATIONS[preparation]\n',
         '    if preparation != "combined":\n        raise ValueError("Observation ablation registers only combined preparation")\n'
         '    _, phase_builder = PREPARATIONS[preparation]\n'),
        ('runtime_type(toolkit)','ObservationRuntime("combined", warp, toolkit)'),
        ('               preparation=preparation, phase_preparation_shared=',
         '               observation_reduction="warp" if warp else "serial",\n'
         '               preparation=preparation, phase_preparation_shared='),
    )
    for old,new in replacements:
        if body.count(old) != 1:
            raise ValueError(f"Expected exactly one prepared observation substitution {old!r}")
        body = body.replace(old,new)
    with TARGET.open("x",encoding="utf-8",newline="\n") as handle:
        handle.write(body)
    result = dict(source=SOURCE.relative_to(HERE.parent).as_posix(),source_sha256=hashlib.sha256(raw).hexdigest(),
                  target=TARGET.name,target_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                  replacements=[dict(before=a,after=b) for a,b in replacements])
    with MANIFEST.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2)


if __name__ == "__main__":
    main()
