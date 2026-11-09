"""Build the same Legendre trial/mean spaces with a differentiated recurrence.

No basis, boundary condition, seed, quadrature or time integrator is changed.
The context patches imported functions only within this process, then restores
them, keeping every frozen source file intact.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import statistics
import time

import cpu_formal as base
import full_galerkin
import cylinder_rfm

np = base.case.np


def unit_legendre_jets(x, count):
    """P_j(2x-1) and its first two x derivatives for 0 <= j < count."""
    if count < 1:
        raise ValueError('Require a positive basis count')
    x = np.asarray(x)
    xi = 2*x-1
    jets = np.zeros((3, len(x), count))
    jets[0, :, 0] = 1.
    if count > 1:
        jets[0, :, 1] = xi
        jets[1, :, 1] = 2.
    for j in range(1, count-1):
        a, b = (2*j+1)/(j+1), j/(j+1)
        jets[0, :, j+1] = a*xi*jets[0, :, j] - b*jets[0, :, j-1]
        jets[1, :, j+1] = a*(2*jets[0, :, j]+xi*jets[1, :, j])-b*jets[1, :, j-1]
        jets[2, :, j+1] = a*(4*jets[1, :, j]+xi*jets[2, :, j])-b*jets[2, :, j-1]
    return jets


def polynomial(s, count, warp, mapping='inner', outer='farfield', trial=False):
    z = np.asarray(s)/cylinder_rfm.case.LENGTH
    x, d, dd = cylinder_rfm.mapped_coordinate(z, warp, mapping)
    value, first, second = unit_legendre_jets(x, count)
    edge = None
    if trial and outer == 'farfield':
        edge_d = cylinder_rfm.mapped_coordinate(np.array([1.]), warp, mapping)[1][0]
        j = np.arange(count)
        edge = (1., edge_d*j*(j+1))
    return cylinder_rfm.clamped((value, first*d[:, None],
        second*d[:, None]**2+first*dd[:, None]), z, outer, edge)


def mean_jets(s, m):
    z = np.asarray(s)/base.case.LENGTH
    values, derivative, _ = unit_legendre_jets(z, m-1)
    weight = z*(1-z)**2
    dw = (1-z)*(1-3*z)
    return np.stack((
        np.column_stack((z*((base.case.LENGTH+2)-(base.case.LENGTH+1)*z), weight[:, None]*values)),
        np.column_stack((base.case.LENGTH+2-2*(base.case.LENGTH+1)*z,
                         dw[:, None]*values+weight[:, None]*derivative))/base.case.LENGTH))


@contextmanager
def optimized_basis():
    original_polynomial = cylinder_rfm.polynomial
    original_mean = full_galerkin.mean_jets
    try:
        cylinder_rfm.polynomial = polynomial
        full_galerkin.mean_jets = mean_jets
        yield
    finally:
        cylinder_rfm.polynomial = original_polynomial
        full_galerkin.mean_jets = original_mean


def jet_check():
    s = np.linspace(0., cylinder_rfm.case.LENGTH, 151)
    rows = []
    for count in (17, 37, 61):
        for warp, mapping in ((0., 'inner'), (.8, 'inner'), (.8, 'both')):
            old = cylinder_rfm.polynomial(s, count, warp, mapping, 'farfield', trial=True)
            new = polynomial(s, count, warp, mapping, 'farfield', trial=True)
            relative = [float(np.linalg.norm(x-y)/np.linalg.norm(x)) for x, y in zip(old, new)]
            row = dict(count=count, warp=warp, mapping=mapping, relative_jet_errors=relative,
                       maximum_absolute_jet_errors=np.max(abs(old-new), axis=(1,2)).tolist())
            rows.append(row)
            if max(relative) > 2e-12:
                raise ArithmeticError(f'Recurrence changed raw trial jets: {row}')
    old = full_galerkin.mean_jets(s, 24)
    new = mean_jets(s, 24)
    mean_error = float(np.linalg.norm(old-new)/np.linalg.norm(old))
    if mean_error > 2e-12:
        raise ArithmeticError('Recurrence changed the mean basis')
    return dict(trial_rows=rows, mean_relative_error=mean_error)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=base.OUT/'basis_recurrence.json')
    cli.add_argument('--full-trajectory', action='store_true')
    cli.add_argument('--repeats', type=int, default=5)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = base.read(base.PROTOCOL)
    specs = {s['name']:s for s in protocol['variants']}
    result = dict(status='running', jets=jet_check(), constructor_rows=[], trajectories=[],
        source_hashes=base.source_hashes() | {'rfm_v7/basis_recurrence.py':base.digest(Path(__file__))},
        policy='Fresh model construction comparison. Recurrence preserves the mathematical space; '
               'physical fields, not transformed coefficients, must be compared across SVD basis rotations.')
    qualification = base.read(base.OUT/'qualification.json')
    grids = base.grid_set()
    for name in ('rfm37_k72', 'leg37_k72'):
        spec = specs[name]
        seed = protocol['seeds'][0] if spec['method']=='rfm' else protocol['deterministic_seed']
        cls, settings = base.config(spec, protocol, seed)
        samples = {key:[] for key in ('original', 'recurrence')}
        for repetition in range(args.repeats):
            for backend in (('original','recurrence') if repetition%2==0 else ('recurrence','original')):
                if backend == 'original':
                    start = time.perf_counter()
                    model = cls(dt=1/spec['rate'], **settings)
                    elapsed = time.perf_counter()-start
                else:
                    with optimized_basis():
                        start = time.perf_counter()
                        model = cls(dt=1/spec['rate'], **settings)
                        elapsed = time.perf_counter()-start
                samples[backend].append(elapsed)
        result['constructor_rows'].append(dict(name=name, samples_seconds=samples,
            median_seconds={key:statistics.median(values) for key, values in samples.items()}))
        if args.full_trajectory:
            control = next(r for r in qualification['rows'] if r['name']==name and r['seed']==seed)
            path = base.OUT/control['artifact']
            if base.digest(path)!=control['sha256']:
                raise RuntimeError('Frozen trajectory changed')
            old = base.case.load_solution(path, cls)
            truth = base.case.reference_fields(old, grids)
            with optimized_basis():
                new = base.case.solve(cls, steps=protocol['horizon']*spec['rate'], **settings)
                errors = base.case.maxima(base.case.score(new, truth, grids))
                probe_errors = {key:float(np.linalg.norm(getattr(new,key)-getattr(old,key))/
                    np.linalg.norm(getattr(old,key))) for key in ('transverse','vorticity')}
                base.case.store_solution(base.OUT/f'recurrence_{name}_{seed}.npz', new, settings)
                wake = base.wake(new)
            result['trajectories'].append(dict(name=name, seed=seed, errors=errors,
                probe_errors=probe_errors, wake=wake, metadata=new.row,
                passes=max(*errors.values(),*probe_errors.values())<=1e-7))
        base.save(args.output, result)
        print(name, result['constructor_rows'][-1], flush=True)
    result['status'] = 'complete'
    base.save(args.output, result)


if __name__ == '__main__':
    main()
