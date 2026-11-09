"""Generate a standalone march with explicit preparation injection; no runtime patching."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'galerkin_cuda_variable.py'
TARGET = HERE / 'galerkin_cuda_prepared.py'


def main():
    if TARGET.exists():
        raise FileExistsError('Preserve the prepared solver once generated')
    raw = SOURCE.read_bytes()
    source = raw.decode('utf-8')
    body = source[source.index('def solve_cuda_switch('):]
    replacements = [
        ('def solve_cuda_switch(*, startup_rate=',
         'def solve_cuda_prepared(*, preparation="combined", startup_rate='),
        ('    start = time.perf_counter()\n',
         '    start = time.perf_counter()\n'
         '    if preparation not in PREPARATIONS:\n'
         '        raise ValueError("Choose module_only, shared_only, or combined")\n'
         '    runtime_type, phase_builder = PREPARATIONS[preparation]\n'),
        ('phase_operators(startup_rate, late_rate, config)',
         'phase_builder(startup_rate, late_rate, config)'),
        ('SwitchRuntime(toolkit)', 'runtime_type(toolkit)'),
        ('implementation="native-fp64-cufft-precontracted-variable-sbdf2",',
         'implementation="native-fp64-cufft-precontracted-variable-sbdf2-prepared",\n'
         '               preparation=preparation, phase_preparation_shared=(preparation != "module_only"),\n'
         '               compiled_cuda_modules=(2 if preparation == "shared_only" else 1),'),
    ]
    for old, new in replacements:
        if body.count(old) != 1:
            raise ValueError(f'Frozen march substitution is not unique: {old!r}')
        body = body.replace(old, new)
    prefix = '''"""Unchanged single-bridge CUDA march with explicit preparation ablations."""
from contextlib import ExitStack
import ctypes as C
import time

from cpu_sbdf2_switch import SNAPSHOTS, bridge_weights, step_schedule
from cuda_environment_native import call
from galerkin_cuda_native import case, np
from galerkin_cuda_variable import SwitchRuntime, phase_operators, bridge_launch
from galerkin_cuda_module_bundle import BundledSwitchRuntime
from galerkin_phase_shared import phase_operators_shared

PREPARATIONS = {
    "module_only": (BundledSwitchRuntime, phase_operators),
    "shared_only": (SwitchRuntime, phase_operators_shared),
    "combined": (BundledSwitchRuntime, phase_operators_shared),
}

'''
    TARGET.write_text(prefix+body, encoding='utf-8', newline='\n')
    audit = dict(source=SOURCE.name, source_sha256=sha256(raw).hexdigest(), target=TARGET.name,
                 target_sha256=sha256(TARGET.read_bytes()).hexdigest(),
                 allowed_replacements=[dict(old=a, new=b) for a,b in replacements],
                 unchanged='March ordering, buffers, FFT calls, graph scheduling, observations and synchronization.')
    (HERE/'cuda_prepared_solver_generation.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
