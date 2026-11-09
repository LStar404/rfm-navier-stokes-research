"""Audit the actual Galerkin energy forms and native CUDA precontractions.

This is a numerical algebra audit, not a proof of SBDF2 stability or PDE error.
Run after official timings; it loads frozen full-trajectory checkpoints.
"""
import argparse
from pathlib import Path

import cpu_formal as base
from galerkin_cuda_native import operators
from radial_layers import radial_quadrature

np = base.case.np


def scaled_condition(a):
    scale = np.sqrt(np.diag(a))
    return float(np.linalg.cond(a / scale[:, None] / scale))


def relative_defect(a, b, target):
    """Backward residual relative to the requested operator, in Frobenius norm."""
    return float(np.linalg.norm(a @ b - target) / max(np.linalg.norm(target), 1e-300))


def algebra(model, dt):
    _, weights = radial_quadrature(len(model.s), model.metadata['split'])
    f, d, d2 = model.samples
    swr = np.sqrt(weights) / model.r
    linear, radial, angular, *unused = operators(model, dt)
    scale = dt / model.kernel.n
    rows = []
    matrices = []
    for i, mode in enumerate(model.modes):
        mass = model.M0 + mode**2 * model.M1
        lap = (d2 - mode**2 * f) * swr[:, None]
        diffusion = base.case.NU * lap.T @ lap
        system = 1.5 * mass + dt * diffusion
        inverse = model.implicit[i]
        b = linear[:, :, mode]
        g = radial[:, :, mode] / scale
        h = angular[:, :, mode] / (mode * scale)
        rows.append(dict(mode=int(mode),
            scaled_mass_condition=scaled_condition(mass),
            scaled_implicit_condition=scaled_condition(system),
            minimum_mass_eigenvalue=float(np.linalg.eigvalsh(mass)[0]),
            minimum_implicit_eigenvalue=float(np.linalg.eigvalsh(system)[0]),
            inverse_identity_defect=relative_defect(system, inverse, np.eye(model.width)),
            mass_precontract_defect=relative_defect(system, b, mass),
            radial_precontract_defect=relative_defect(system, g, model.test_s),
            angular_precontract_defect=relative_defect(system, h, model.test)))
        matrices.append(mass)
    mv, md = model.mean_samples
    mean_strain = (mv + md) * np.sqrt(weights)[:, None]
    mean_diffusion = base.case.NU * mean_strain.T @ mean_strain
    mean_system = 1.5 * model.mean_mass + dt * mean_diffusion
    mw = model.mean_width
    mean_row = dict(scaled_mass_condition=scaled_condition(model.mean_mass),
        scaled_implicit_condition=scaled_condition(mean_system),
        mass_precontract_defect=relative_defect(mean_system, linear[:mw, :mw, 0], model.mean_mass),
        radial_precontract_defect=relative_defect(mean_system, -radial[:mw, :, 0] / scale, model.mean_test))
    return rows, mean_row, matrices


def work_audit(solution, matrices):
    model = solution.model
    mw = model.mean_width
    inverse_mass = np.stack([np.linalg.inv(a) for a in matrices])
    inverse_mean = np.linalg.inv(model.mean_mass)
    result = []
    for t, state in zip(solution.times, solution.states):
        _, flux = model.terms(state, 0.)
        mean = state[:mw, 0].real
        mean_load = flux[:mw, 0].real
        modal = state[:, 1:].T
        load = flux[:, 1:].T
        velocity_energy = float(mean @ model.mean_mass @ mean +
            2 * np.einsum('ki,kij,kj->', modal.conj(), matrices, modal).real)
        dual_load = float(mean_load @ inverse_mean @ mean_load +
            2 * np.einsum('ki,kij,kj->', load.conj(), inverse_mass, load).real)
        mean_work = float(mean @ mean_load)
        mode_work = float(2 * np.vdot(modal, load).real)
        combined = mean_work + mode_work
        _, lifted = model.terms(state, base.case.speed(t))
        lifted_work = float(mean @ lifted[:mw, 0].real +
                            2 * np.vdot(state[:, 1:], lifted[:, 1:]).real)
        result.append(dict(time=float(t), homogeneous_work=combined,
            mean_work=mean_work, nonzero_mode_work=mode_work,
            normalized_homogeneous_work=abs(combined) / np.sqrt(velocity_energy * dual_load),
            lifted_work=lifted_work, energy_without_lift=velocity_energy,
            note='The lift is omitted only in this algebra audit; all production solves retain it.'))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=base.OUT / 'operator_audit.json')
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = base.read(base.PROTOCOL)
    qualification = base.read(base.OUT / 'qualification.json')
    names = ('rfm37_k72', 'leg37_k72', 'rfm41_k80', 'leg39_k80')
    result = dict(status='running', rows=[], source_hashes=base.source_hashes(),
        policy='Numerical residuals on frozen states; no new flow or timing claims. '
               'Positive quadrature and common trial/test evaluation imply semidiscrete '
               'homogeneous convection cancellation in exact arithmetic. Explicit SBDF2 '
               'and nonzero boundary lifting require separate stability analysis.')
    for name in names:
        record = next(r for r in qualification['rows'] if r['name'] == name)
        source = base.OUT / record['artifact']
        if base.digest(source) != record['sha256']:
            raise RuntimeError('Qualification artifact changed')
        sol = base.case.load_solution(source, base.BufferedGalerkin)
        modes, mean, matrices = algebra(sol.model, sol.row['dt'])
        energy = work_audit(sol, matrices)
        row = dict(name=name, seed=record['seed'], artifact=record['artifact'],
            sha256=record['sha256'], modes=modes, mean=mean, energy=energy,
            worst_operator_defect=max(r[key] for r in modes for key in (
                'mass_precontract_defect', 'radial_precontract_defect', 'angular_precontract_defect')),
            maximum_scaled_implicit_condition=max(r['scaled_implicit_condition'] for r in modes),
            worst_normalized_homogeneous_work=max(r['normalized_homogeneous_work'] for r in energy))
        result['rows'].append(row)
        base.save(args.output, result)
        print(name, row['worst_operator_defect'], row['maximum_scaled_implicit_condition'],
              row['worst_normalized_homogeneous_work'], flush=True)
    result['status'] = 'complete'
    base.save(args.output, result)


if __name__ == '__main__':
    main()
