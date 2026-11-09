"""Qualified CPU/CUDA comparison with recurrence Galerkin and resident FD4.

This is a new frozen experiment. The earlier 25-row comparison is preserved.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import cpu_formal as base
import cpu_formal_joint as prior
from basis_recurrence_model import RecurrenceGalerkin, native_model
from fd_cuda_native import solve_fd_native
from fd_cpu_buffered import BufferedFD
from galerkin_cuda_native import solve_native

np = base.case.np
OUT = base.HERE/'hardware_output'
PATH = OUT/'timing.json'
NAMES = ('rfm37_k72', 'leg37_k72', 'fd176_k80')
BACKENDS = ('cpu', 'cuda_graph')
ORDER_SEED = 116288
GRAPH_STEPS = 24
IMPLEMENTATION_GATE = 1e-5


def hashes():
    names = ('joint_hardware.py', 'basis_recurrence.py', 'basis_recurrence_model.py',
             'fd_cuda_native.py', 'fd_cuda_native.cu', 'fd_cpu_buffered.py',
             'fd_cpu_buffered.c', 'fd_cpu_buffered.dll')
    return prior.hash_set() | {f'rfm_v7/{name}':base.digest(base.HERE/name) for name in names}


def solve(spec, protocol, seed, backend):
    cls, settings = base.config(spec, protocol, seed)
    steps = protocol['horizon']*spec['rate']
    if cls is base.FullFD:
        if backend == 'cpu':
            result = base.case.solve(BufferedFD, steps=steps, **settings)
        else:
            result = solve_fd_native(steps=steps, graph_steps=GRAPH_STEPS,
                                     cuda_linear_solve='dense', profile_steps=0, **settings)
    elif backend == 'cpu':
        result = base.case.solve(RecurrenceGalerkin, steps=steps, **settings)
    else:
        with native_model():
            result = solve_native(steps=steps, graph_steps=GRAPH_STEPS, profile_steps=0, **settings)
    return result, settings


def controls(qualification):
    result = {}
    grids = base.grid_set()
    for r in qualification['rows']:
        if r['name'] not in NAMES:
            continue
        path = base.OUT/r['artifact']
        if base.digest(path)!=r['sha256']:
            raise RuntimeError('Frozen qualification changed')
        cls = base.FullFD if r['method']=='fd4' else base.BufferedGalerkin
        sol = base.case.load_solution(path, cls)
        result[(r['name'],r['seed'])] = dict(fields=base.case.reference_fields(sol, grids),
            transverse=sol.transverse, vorticity=sol.vorticity,
            times=sol.times, probe_times=sol.probe_times)
    return result


def row(job, spec, protocol, reference, control):
    name, seed, backend = job['name'], job['seed'], job['backend']
    path = OUT/f"block{job['block']}_{backend}_{name}_{seed}.npz"
    record_path = path.with_suffix('.json')
    if record_path.exists():
        existing = base.read(record_path)
        if any(existing[key]!=value for key,value in job.items()):
            raise RuntimeError('Saved job differs from schedule')
        if 'sha256' in existing and base.digest(path)!=existing['sha256']:
            raise RuntimeError('Saved trajectory changed')
        return existing
    if path.exists():
        raise FileExistsError(f'Uncommitted artifact requires inspection: {path.name}')
    item = dict(job)
    start = time.perf_counter()
    try:
        sol, settings = solve(spec, protocol, seed, backend)
        seconds = time.perf_counter()-start
        if not np.array_equal(sol.times,control['times']) or not np.array_equal(sol.probe_times,control['probe_times']):
            raise RuntimeError('Physical checkpoints changed')
        grids = base.grid_set()
        scores = base.case.score(sol, reference, grids)
        errors = base.case.maxima(scores)
        implementation = base.case.maxima(base.case.score(sol, control['fields'], grids))
        probes = {key:float(np.linalg.norm(getattr(sol,key)-control[key])/np.linalg.norm(control[key]))
                  for key in ('transverse','vorticity')}
        diagnostics = base.wake(sol)
        temp = path.with_suffix('.tmp.npz')
        base.case.store_solution(temp, sol, settings)
        temp.replace(path)
        item.update(seconds=seconds, metadata=sol.row, errors=errors, scores=scores,
            implementation_field_errors=implementation, implementation_probe_errors=probes,
            implementation_pass=max(*implementation.values(),*probes.values())<=IMPLEMENTATION_GATE,
            wake=diagnostics, boundary_errors=base.boundary_errors(sol),
            artifact=path.name, sha256=base.digest(path),
            numerical_pass=max(errors.values())<=protocol['gate'] and diagnostics['sustained_alternating_probe_signs'],
            reconstruction_class='FullFD' if spec['method']=='fd4' else 'RecurrenceGalerkin')
    except (ArithmeticError,ValueError,np.linalg.LinAlgError,RuntimeError,OSError) as exc:
        item.update(failure=str(exc), failure_seconds=time.perf_counter()-start,
                    numerical_pass=False, implementation_pass=False)
    base.save(record_path, item)
    return item


def summary(rows):
    result = []
    for backend in BACKENDS:
        for name in NAMES:
            selected = [r for r in rows if r['backend']==backend and r['name']==name]
            good = [r for r in selected if 'seconds' in r]
            samples = [r['seconds'] for r in good]
            result.append(dict(backend=backend, name=name, count=len(selected),
                failure_count=sum('failure' in r for r in selected),
                passed=all(r['numerical_pass'] and r['implementation_pass'] for r in selected),
                samples_seconds=samples, median_seconds=statistics.median(samples) if samples else None,
                minimum_seconds=min(samples) if samples else None,
                maximum_seconds=max(samples) if samples else None,
                worst={key:max(r['errors'][key] for r in good) for key in (
                    'velocity','vorticity','near_velocity','near_vorticity')} if good else None))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-only',action='store_true')
    cli.add_argument('--confirm-isolated',action='store_true')
    args = cli.parse_args()
    if not args.freeze_only and not args.confirm_isolated:
        cli.error('Stop competing research workloads before the formal run')
    protocol = base.read(base.PROTOCOL)
    reference, truth = base.reference_audit(protocol)
    qualification, specs = prior.frozen_qualification(protocol,reference)
    constructor_check = base.read(base.OUT/'basis_recurrence.json')
    if constructor_check['status']!='complete' or not all(r['passes'] for r in constructor_check['trajectories']):
        raise RuntimeError('Full-trajectory recurrence validation is missing')
    pilot = base.read(base.HERE/'fd_cuda_native_dense_t72.json')
    if max(pilot['fields_vs_reference'].values())>protocol['gate']:
        raise RuntimeError('FD CUDA pilot must pass the identical gate')
    rng = np.random.default_rng(ORDER_SEED)
    methods = [(backend,name) for backend in BACKENDS for name in NAMES]
    schedule = []
    for block, seed in enumerate(protocol['seeds'],1):
        for i in rng.permutation(len(methods)):
            backend,name = methods[i]
            schedule.append(dict(block=block,backend=backend,name=name,
                seed=seed if specs[name]['method']=='rfm' else protocol['deterministic_seed']))
    fixed = dict(protocol=protocol, reference=reference, schedule=schedule, source_hashes=hashes(),
        order_seed=ORDER_SEED, graph_steps=GRAPH_STEPS, cuda_fd_linear='dense',
        implementation_tolerance=IMPLEMENTATION_GATE,
        scope='Fresh in-process end-to-end solves. Includes all model/linear preparation, '
              'NVRTC compilation, transfers, FFT plans, graph setup, synchronized marching, '
              'probes/checkpoints and cleanup. Python import and driver disk caches are not reset.',
        source_scope='Earlier 25-run results and their source files are preserved unchanged.')
    manifest = OUT/'protocol.json'
    if manifest.exists():
        if base.read(manifest)!=json.loads(json.dumps(fixed)):
            raise RuntimeError('Frozen hardware protocol or implementation changed')
    else:
        base.save(manifest, fixed)
    if args.freeze_only:
        print('Frozen', manifest, flush=True)
        return
    cpu = base.pin()
    if PATH.exists():
        result = base.read(PATH)
        if result['frozen']!=json.loads(json.dumps(fixed)):
            raise RuntimeError('Timing manifest differs')
        if result['cpu']!=cpu:
            raise RuntimeError('Pinned CPU differs from the saved measurement')
    else:
        result = dict(status='running', frozen=fixed, rows=[], cpu=cpu,
                      gpu=prior.nvidia(), platform=platform.platform())
        base.save(PATH,result)
    control = controls(qualification)
    known = {(r['block'],r['backend'],r['name']) for r in result['rows']}
    for job in schedule:
        if (job['block'],job['backend'],job['name']) in known:
            continue
        item = row(job, specs[job['name']], protocol, truth, control[(job['name'],job['seed'])])
        result['rows'].append(item)
        base.save(PATH,result)
        print(job['block'],job['backend'],job['name'],item.get('seconds'),
              item.get('errors',item.get('failure')),flush=True)
    if fixed['source_hashes']!=hashes():
        raise RuntimeError('Implementation changed during timing')
    if any(base.digest(base.WORK/name)!=sha for name,sha in reference['reference_hashes'].items()):
        raise RuntimeError('Reference changed during timing')
    result.update(status='complete',summary=summary(result['rows']))
    base.save(PATH,result)


if __name__=='__main__':
    main()
