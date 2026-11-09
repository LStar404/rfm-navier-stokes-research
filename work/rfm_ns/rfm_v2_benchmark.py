"""Fresh-seed steady NS comparison; does not change any earlier experiment.

API: SolverSpec(name, config, solve_fn=None, projection='dense', source_files=()).
Custom solve_fn(problem, seed=..., **kwargs) must return (op, coefficients, row).
op.fields(coefficients, xy) must return u, v and the actual velocity curl omega;
it is responsible for any internal coefficient transform, exactly as solve().

run_suite(specs, output=..., seeds=SEEDS, thresholds=(.003, .001), include_fd=True)
freezes configuration/provenance before running. Timings enclose the entire fresh
solver call; reference loading, field evaluation and reporting are outside it.
Run only in an agreed isolated CPU window. There is no import-time solve.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import scipy
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq
import rfm_solver as base
from cfd_reference import interpolate_solution
from cfd_reference_compare import compute
from cfd_reference_multilevel import two_level
from tensor_rfm import tensor_quadrature

ROOT = Path(__file__).resolve().parent
SEEDS = (401, 503, 607, 709, 811)
CORE_SOURCES = ('rfm_v2_benchmark.py', 'rfm_solver.py', 'tensor_rfm.py',
                'experiments.py', 'cfd_reference.py', 'cfd_reference_fourth.py',
                'cfd_reference_compare.py', 'cfd_reference_multilevel.py')


@dataclass
class SolverSpec:
    name: str
    config: dict
    solve_fn: object = None
    projection: str = 'dense'
    source_files: tuple = field(default_factory=tuple)

    def serializable(self):
        fn = self.solve_fn or base.solve
        return dict(name=self.name, config=self.config, projection=self.projection,
                    solver=f'{fn.__module__}:{fn.__qualname__}',
                    source_files=[str(Path(p)) for p in self.source_files])


def default_specs(reynolds=400):
    """Prior dense/tensor RFM and deterministic Legendre controls, no retuning."""
    frozen = read_json(ROOT/'frozen_cavity.json')
    labels = ('rfm496', 'legendre496') if reynolds == 400 else ('rfm325', 'legendre325')
    configs = [c for c in frozen if c['reynolds'] == reynolds and c['label'] in labels]
    if len(configs) != 2:
        raise ValueError('Default controls require the two original frozen configurations')
    return [SolverSpec(f"{c['label']}_{projection}", dict(c), projection=projection)
            for c in configs for projection in ('dense', 'tensor')]


def read_json(path):
    def invalid(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def json_safe(value):
    """Preserve failed observations without emitting invalid NaN/Infinity JSON."""
    if isinstance(value,np.generic):
        return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return f'nonfinite:{value}'
    if isinstance(value,np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value,dict):
        return {str(key):json_safe(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_safe(item) for item in value]
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def machine_info():
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                logical_cpu_count=os.cpu_count(), numpy=np.__version__, scipy=scipy.__version__,
                threadpools=pools, thread_environment={k: os.environ[k] for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def one_thread():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        # Environment is set before NumPy import. If imported by another driver,
        # that driver should install threadpoolctl or ensure one thread itself.
        return nullcontext()


def score_grids(problem, gauss_order=113):
    z = (np.arange(81)+.5)/81
    xy = problem.lo+problem.length*np.stack(np.meshgrid(z,z,indexing='ij'),-1).reshape(-1,2)
    a, w = np.polynomial.legendre.leggauss(gauss_order)
    a = (a+1)/2
    gauss = problem.lo+problem.length*np.stack(np.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)
    weights = np.outer(w,w).ravel()*(problem.length/2)**2
    return {'midpoint81': (xy, np.full(len(xy),problem.length**2/len(xy))),
            f'gauss{gauss_order}': (gauss, weights)}


def weighted_errors(pred, truth, weights):
    """Area-weighted relative L2 and sampled, not rigorous supremum, errors."""
    arrays = [np.asarray(d[k]) for d in (pred, truth) for k in ('u','v','omega')]
    if any(a.shape != weights.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Non-finite field or incompatible score-grid shape')
    du, dv, dw = pred['u']-truth['u'], pred['v']-truth['v'], pred['omega']-truth['omega']
    vel_denom = np.sum(weights*(truth['u']**2+truth['v']**2))
    vort_denom = np.sum(weights*truth['omega']**2)
    if min(vel_denom, vort_denom) <= 0:
        raise ValueError('Relative error is undefined for a zero reference norm')
    vmax = float(np.hypot(du,dv).max())
    wmax = float(np.abs(dw).max())
    return dict(velocity_rel_l2=float(np.sqrt(np.sum(weights*(du*du+dv*dv))/vel_denom)),
        vorticity_rel_l2=float(np.sqrt(np.sum(weights*dw*dw)/vort_denom)),
        velocity_sampled_linf=vmax, vorticity_sampled_linf=wmax,
        velocity_sampled_rel_linf=vmax/float(np.hypot(truth['u'],truth['v']).max()),
        vorticity_sampled_rel_linf=wmax/float(np.abs(truth['omega']).max()))


def interpolated_velocity_curl(fields, xy):
    u = RectBivariateSpline(fields['x'], fields['y'], fields['u'], kx=3, ky=3, s=0)
    v = RectBivariateSpline(fields['x'], fields['y'], fields['v'], kx=3, ky=3, s=0)
    return v.ev(xy[:,0],xy[:,1],dx=1)-u.ev(xy[:,0],xy[:,1],dy=1)


def curl_closure(fields, xy, pred, weights):
    curl = interpolated_velocity_curl(fields,xy)
    diff = curl-pred['omega']
    return dict(relative_l2=float(np.sqrt(np.sum(weights*diff**2)/np.sum(weights*pred['omega']**2))),
                sampled_linf=float(np.max(np.abs(diff))),
                definition='Curl of bicubic interpolated velocities minus interpolated discrete omega; diagnostic only')


def load_fields(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in ('x','y','psi','omega','u','v')}


class ScoreContext:
    """All reference arrays and interpolants are prepared before timed runs."""
    def __init__(self, case, reynolds, gauss_order=113):
        self.problem = base.Problem(case,reynolds)
        self.grids = score_grids(self.problem,gauss_order)
        self.truth = {}
        self.files = []
        self.audit = dict(case=case,reynolds=reynolds,grids={})
        fine = coarse = None
        if case == 'cavity':
            if reynolds not in (100,400):
                raise ValueError('Cavity reference grids are frozen for Re100/Re400 only')
            fine_n, coarse_n = (256,128) if reynolds == 100 else (384,256)
            self.files = [ROOT/f'cfd_reference4_cavity_Re{int(reynolds)}_n{n}.npz'
                          for n in (fine_n,coarse_n)]
            fine,coarse = (load_fields(path) for path in self.files)
            self.audit['reference_files'] = [dict(name=p.name,sha256=digest(p)) for p in self.files]
        elif case != 'kovasznay':
            raise ValueError(f'Unknown case {case}')
        for name,(xy,weights) in self.grids.items():
            truth = self.problem.exact(xy) if fine is None else interpolate_solution(fine,xy[:,0],xy[:,1])
            self.truth[name] = truth
            entry = dict(points=len(xy),points_sha256=array_digest(xy),weights_sha256=array_digest(weights),
                         weight_sum=float(weights.sum()),minimum_distance_to_wall=float(np.minimum(
                         xy-self.problem.lo,self.problem.lo+self.problem.length-xy).min()))
            if fine is not None:
                prior = interpolate_solution(coarse,xy[:,0],xy[:,1])
                entry['reference_grid_change'] = weighted_errors(prior,truth,weights)
                entry['fine_reference_velocity_curl_closure'] = curl_closure(fine,xy,truth,weights)
                entry['coarse_reference_velocity_curl_closure'] = curl_closure(coarse,xy,prior,weights)
            self.audit['grids'][name] = entry
        self.audit['caveat'] = ('Gauss points approach walls more closely than the reference spacing; '
            'coarse/fine differences are empirical error-floor diagnostics, not certified bounds. '
            'Native CFD discrete omega is the primary reference; spline-velocity curl is a separate diagnostic.')

    def evaluate(self, evaluator, fd_fields=None, keep_fields=False):
        scores, artifacts = {}, {}
        for name,(xy,weights) in self.grids.items():
            pred = {}
            for block in np.array_split(xy,max(1,int(np.ceil(len(xy)/256)))):
                values = evaluator(block)
                for key in ('u','v','omega','psi','div'):
                    if key in values:
                        pred.setdefault(key,[]).append(np.asarray(values[key]))
            pred = {key:np.concatenate(values) for key,values in pred.items()}
            scores[name] = weighted_errors(pred,self.truth[name],weights)
            if 'div' in pred:
                scores[name]['divergence_rms'] = float(np.sqrt(np.sum(weights*pred['div']**2)/weights.sum()))
            if fd_fields is not None:
                scores[name]['velocity_curl_closure'] = curl_closure(fd_fields,xy,pred,weights)
            if keep_fields:
                artifacts.update({f'{name}_{key}':value for key,value in pred.items()})
                artifacts[f'{name}_xy'],artifacts[f'{name}_weights'] = xy,weights
        return scores,artifacts

    def reference_change(self):
        return max((max(entry['reference_grid_change'][key] for key in
                        ('velocity_rel_l2','vorticity_rel_l2'))
                    for entry in self.audit['grids'].values() if 'reference_grid_change' in entry),default=0.)


def default_fd_configs(specs, thresholds):
    """Frozen prior strategies plus predeclared Re400 n104/n112 gap controls."""
    wanted = {float(s.config['reynolds']) for s in specs if s.config['case']=='cavity'}
    prior = read_json(ROOT/'cfd_fd4_multilevel.json')
    result = {}
    for selected in prior['selected_by_tier']:
        if selected['Re'] not in wanted or not any(abs(selected['tier']-t)<1e-14 for t in thresholds):
            continue
        row = selected['best']
        key = (row['Re'],row['n'],row['strategy'])
        result[key] = dict(case='cavity',reynolds=row['Re'],n=row['n'],strategy=row['strategy'],
                           name=f"fd4_Re{row['Re']}_n{row['n']}_{row['strategy']}")
    if 400 in wanted:
        for n in (104,112):
            result[(400,n,'two_level_64')] = dict(case='cavity',reynolds=400,n=n,
                strategy='two_level_64',name=f'fd4_Re400_n{n}_two_level_64')
    return list(result.values())


def summarize(rows, specs, fd_configs, thresholds, reference_audits):
    summaries = []
    identities = [(s.name,s.config['case'],s.config['reynolds'],'rfm_or_polynomial') for s in specs]
    identities += [(s['name'],s['case'],s['reynolds'],'fd4') for s in fd_configs]
    for name,case,reynolds,kind in identities:
        group = [row for row in rows if row['name']==name]
        good = [row for row in group if row['status']=='completed']
        metrics = {key:max((grid[key] for row in good for grid in row['scores'].values()),default=None)
                   for key in ('velocity_rel_l2','vorticity_rel_l2','velocity_sampled_linf','vorticity_sampled_linf')}
        times = [row['wall_seconds'] for row in good]
        refchange = reference_audits[(case,float(reynolds))].reference_change()
        passed = {str(t):bool(len(good)==len(group) and good and
                  max(metrics['velocity_rel_l2'],metrics['vorticity_rel_l2'])<=t) for t in thresholds}
        summaries.append(dict(name=name,case=case,reynolds=reynolds,kind=kind,runs=len(group),
            completed=len(good),median_wall_seconds=statistics.median(times) if times else None,
            min_wall_seconds=min(times) if times else None,max_wall_seconds=max(times) if times else None,
            worst_across_seeds_and_both_score_grids=metrics,passes=passed,
            reference_change_max_rel_l2=refchange,
            reference_change_below_20pct_of_tier={str(t):refchange<=.2*t for t in thresholds}))
    comparisons = []
    for case,reynolds in sorted({(x[1],x[2]) for x in identities}):
        for tier in thresholds:
            eligible = [row for row in summaries if row['case']==case and row['reynolds']==reynolds and
                        row['passes'][str(tier)] and row['reference_change_below_20pct_of_tier'][str(tier)]]
            fd = [row for row in eligible if row['kind']=='fd4']
            best = min(fd,key=lambda row:row['median_wall_seconds']) if fd else None
            comparisons.append(dict(case=case,reynolds=reynolds,tier=tier,
                fastest_tested_passing_fd=best['name'] if best else None,
                comparisons=[dict(name=row['name'],median_seconds=row['median_wall_seconds'],
                    fd_over_method_time=best['median_wall_seconds']/row['median_wall_seconds'] if best else None)
                    for row in eligible if row['kind']!='fd4']))
    return summaries,comparisons


def run_suite(specs, output=None, seeds=SEEDS, thresholds=(.003,.001), include_fd=True,
              fd_configs=None, gauss_order=113, save_first_seed=True, overwrite=False):
    """Sequential, randomized paired runs. Call only after timing authorization.

    fd_configs may explicitly specify e.g. dict(name='fd112',case='cavity',
    reynolds=400,n=112,strategy='two_level_64'); all such choices freeze first.
    Do not tune on these held-out seeds then describe the run as confirmation.
    """
    specs,seeds,thresholds = list(specs),tuple(seeds),tuple(float(t) for t in thresholds)
    if not specs or not seeds or len(set(seeds))!=len(seeds) or not thresholds or min(thresholds)<=0:
        raise ValueError('Require nonempty specs, unique seeds and positive thresholds')
    if gauss_order < 82:
        raise ValueError('Independent Gauss score must be finer than the original 81-point grid')
    for spec in specs:
        if spec.projection not in ('dense','tensor','native'):
            raise ValueError('Projection must be dense, tensor or native (solver-owned)')
        if spec.projection=='tensor' and spec.config.get('cells')!=1:
            raise ValueError('Tensor baseline supports a single global test cell')
    fd_configs = (default_fd_configs(specs,thresholds) if fd_configs is None else list(fd_configs)) if include_fd else []
    names = [s.name for s in specs]+[c['name'] for c in fd_configs]
    if len(names)!=len(set(names)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',n) for n in names):
        raise ValueError('Unique filesystem-safe names are required')
    output = Path(output) if output else ROOT/'rfm_v2_formal.json'
    frozen_path = output.with_name(output.stem+'_freeze.json')
    if not overwrite and (output.exists() or frozen_path.exists()):
        raise FileExistsError('Choose a new output name; old evidence is preserved by default')
    output.parent.mkdir(parents=True,exist_ok=True)
    sources = {ROOT/name for name in CORE_SOURCES}
    sources.update((ROOT/'frozen_cavity.json',ROOT/'cfd_fd4_multilevel.json'))
    for spec in specs:
        source = inspect.getsourcefile(spec.solve_fn or base.solve)
        if source is None:
            raise ValueError('Custom solver must have a source file for provenance')
        sources.add(Path(source).resolve())
        sources.update((ROOT/Path(p)).resolve() for p in spec.source_files)
    with one_thread():
        contexts = {(case,float(reynolds)):ScoreContext(case,reynolds,gauss_order)
                    for case,reynolds in sorted({(s.config['case'],s.config['reynolds']) for s in specs}|
                    {(c['case'],c['reynolds']) for c in fd_configs})}
        sources.update(path for context in contexts.values() for path in context.files)
        hashes = {str(path.resolve()):digest(path) for path in sorted(sources)}
        jobs = [('solver',i) for i in range(len(specs))]+[('fd',i) for i in range(len(fd_configs))]
        schedule = []
        for i,seed in enumerate(seeds):
            shuffled = list(jobs)
            random.Random(29031+i).shuffle(shuffled)
            schedule.extend(dict(seed=seed,kind=kind,index=index) for kind,index in shuffled)
        freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            specs=[s.serializable() for s in specs],fd_configs=fd_configs,seeds=seeds,
            thresholds=thresholds,gauss_order=gauss_order,schedule=schedule,source_sha256=hashes,
            machine=machine_info(),references=[c.audit for c in contexts.values()],
            timing='Outer complete fresh solver call, including all feature/assembly/linear/nonlinear/initialization work. Reference loading, scoring and serialization excluded.',
            selection='Every seed and both weighted-L2 grids must meet both velocity and vorticity thresholds. Reference-grid change <=20% of tier is an empirical resolution screen, not an error bound.',
            scope='Steady 2D NS; five seed/solve pairs per neural method, five fresh repeats per deterministic/FD control. No training-data reference state enters a solve.')
        write_json(frozen_path,freeze)
        result = dict(status='running',freeze=freeze,freeze_sha256=digest(frozen_path),rows=[],summaries=[],comparisons=[])
        write_json(output,result)
        # Common small LAPACK warmup; no feature matrix or physical solution reused.
        rng=np.random.default_rng(734)
        lstsq(rng.normal(size=(24,12)),rng.normal(size=24),lapack_driver='gelsy')
        for job in schedule:
            # Release the preceding solver's large feature arrays before timing
            # another solver. The references remain outside all solve callbacks.
            op=c=fields=evaluator=artifacts=None
            seed,kind,index = job['seed'],job['kind'],job['index']
            config = specs[index].config if kind=='solver' else fd_configs[index]
            name = specs[index].name if kind=='solver' else config['name']
            context = contexts[(config['case'],float(config['reynolds']))]
            row = dict(name=name,seed=seed,kind=kind,case=config['case'],reynolds=config['reynolds'])
            tick = None
            phase = 'solve'
            try:
                if kind=='solver':
                    spec=specs[index]
                    kwargs={k:v for k,v in config.items() if k not in ('case','reynolds','label')}
                    fn=spec.solve_fn or base.solve
                    manager=patch.object(base,'weak_quadrature',tensor_quadrature) if spec.projection=='tensor' else nullcontext()
                    with manager:
                        tick=time.perf_counter()
                        op,c,observation=fn(context.problem,seed=seed,**kwargs)
                        elapsed=time.perf_counter()-tick
                    evaluator=lambda xy:op.fields(c,xy)
                    fields=None
                else:
                    strategy=config['strategy']
                    tick=time.perf_counter()
                    if strategy.startswith('two_level_'):
                        fields,observation=two_level(config['n'],config['reynolds'],int(strategy.rsplit('_',1)[1]))
                    else:
                        fields,observation=compute(config['n'],config['reynolds'],strategy)
                    elapsed=time.perf_counter()-tick
                    evaluator=lambda xy:interpolate_solution(fields,xy[:,0],xy[:,1])
                row.update(wall_seconds=elapsed,observation=observation)
                if observation.get('success') is False or observation.get('converged') is False or observation.get('finite') is False:
                    raise ArithmeticError('Solver explicitly reported failure/nonconvergence')
                if kind=='fd' and not all(stage['success'] for stage in observation['stages']):
                    raise ArithmeticError('A finite-difference initialization/solve stage did not converge')
                phase = 'independent_score'
                scores,artifacts=context.evaluate(evaluator,fields,save_first_seed and seed==seeds[0])
                row.update(status='completed',scores=scores,
                    worst_grid_velocity_rel_l2=max(v['velocity_rel_l2'] for v in scores.values()),
                    worst_grid_vorticity_rel_l2=max(v['vorticity_rel_l2'] for v in scores.values()))
                row['passes']={str(t):max(row['worst_grid_velocity_rel_l2'],row['worst_grid_vorticity_rel_l2'])<=t for t in thresholds}
                if artifacts:
                    if kind=='solver':
                        artifacts['coefficients_internal']=np.asarray(c)
                        artifacts['coefficients_original']=op.transform@c if hasattr(op,'transform') else np.asarray(c)
                    path=output.with_name(f'{output.stem}__{name}_seed{seed}.npz')
                    np.savez_compressed(path,**artifacts)
                    row['artifact']=dict(path=path.name,sha256=digest(path))
            except Exception as exc:
                row.update(status='failed',failure_phase=phase,error=f'{type(exc).__name__}: {exc}')
                if 'wall_seconds' not in row:
                    row['wall_seconds']=time.perf_counter()-tick if tick is not None else 0.
            result['rows'].append(row)
            write_json(output,result)
            print(name,seed,row['status'],f"{row['wall_seconds']:.6f}s",row.get('worst_grid_vorticity_rel_l2'),flush=True)
        if any(digest(path)!=expected for path,expected in hashes.items()):
            result['status']='source_changed_during_run'
            write_json(output,result)
            raise RuntimeError('Frozen source/reference changed during run; results are not a frozen comparison')
        result['summaries'],result['comparisons']=summarize(result['rows'],specs,fd_configs,thresholds,contexts)
        result.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                      all_runs_completed=all(row['status']=='completed' for row in result['rows']))
        write_json(output,result)
        return result


def resolve_solver(name):
    module,attribute=name.split(':',1)
    value=importlib.import_module(module)
    for part in attribute.split('.'):
        value=getattr(value,part)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'rfm_v2_formal.json')
    parser.add_argument('--extra-specs',type=Path,help='JSON list: name, config, solver="module:function", optional projection/source_files')
    parser.add_argument('--no-defaults',action='store_true')
    parser.add_argument('--no-fd',action='store_true')
    parser.add_argument('--fd-configs',type=Path,help='Explicit predeclared FD config list, e.g. n96/n128 or additional n104/n112 pilots')
    parser.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    parser.add_argument('--thresholds',default='0.003,0.001')
    parser.add_argument('--gauss-order',type=int,default=113)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    specs=[] if args.no_defaults else default_specs()
    for entry in read_json(args.extra_specs) if args.extra_specs else []:
        specs.append(SolverSpec(entry['name'],entry['config'],resolve_solver(entry['solver']),
                                entry.get('projection','dense'),tuple(entry.get('source_files',()))))
    run_suite(specs,args.output,tuple(map(int,args.seeds.split(','))),
              tuple(map(float,args.thresholds.split(','))),not args.no_fd,
              read_json(args.fd_configs) if args.fd_configs else None,args.gauss_order,
              overwrite=args.overwrite)


if __name__=='__main__':
    main()
