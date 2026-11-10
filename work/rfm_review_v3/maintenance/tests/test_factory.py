"""Numerical compatibility and solve-local factory isolation (CPU only)."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from .. import rfm_solver as base
from .. import rfm_v2_solver as solver
from .. import rfm_v2_warp_legendre as mapped
from .snapshot import collect

np = base.np
DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = DIRECTORY.parents[2]


class FactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = base.Problem('cavity', 400)
        cls.options = dict(m=15, scale=3., order=5, q=11, maxiter=4,
                           metric='h2_parity', wall=dict(warp='sinh', strength=[1.3, 1.3]))
        cls.global_factory = solver.trial_basis
        cls.base_features = base.Features

    def assert_solve_equal(self, actual, expected):
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[0].stokes, expected[0].stokes, rtol=1e-12, atol=1e-12)
        for key in ('history', 'coarse_history', 'rank', 'final_residual'):
            self.assert_metadata_equal(actual[2][key], expected[2][key])

    def assert_metadata_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_metadata_equal(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_metadata_equal(a, b)
        elif isinstance(expected, (float, np.floating)):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        else:
            self.assertEqual(actual, expected)

    def assert_globals_unchanged(self):
        self.assertIs(solver.trial_basis, type(self).global_factory)
        self.assertIs(base.Features, self.base_features)

    def test_01_frozen_source_hashes_unchanged(self):
        manifest = json.loads((DIRECTORY/'provenance.json').read_text(encoding='utf8'))
        for path, expected in manifest['frozen_sources'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), expected)
        for path, record in manifest['archive_test_support'].items():
            with self.subTest(archive_support=path):
                self.assertEqual(hashlib.sha256((DIRECTORY/path).read_bytes()).hexdigest(), record['sha256'])

    def test_02_small_cpu_solves_match_archived_code(self):
        # The archived wrapper is allowed to patch only inside this child process.
        # It cannot contaminate imports or factories in the maintenance process.
        with tempfile.TemporaryDirectory(prefix='rfm_factory_equivalence_') as tmp:
            output = Path(tmp)/'archive.json'
            completed = subprocess.run([sys.executable, str(DIRECTORY/'tests/snapshot.py'),
                                        '--archived', '--output', str(output)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0,
                             'Archived child process failed:\n'+completed.stdout+completed.stderr)
            expected = json.loads(output.read_text(encoding='utf8'))
        actual = collect(base, solver, mapped)
        self.assertEqual(actual.keys(), expected.keys())
        metrics = []
        for name, want in expected.items():
            with self.subTest(case=name):
                for field in ('coefficients', 'fields', 'stokes'):
                    a, b = np.asarray(actual[name][field]), np.asarray(want[field])
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                    metrics.append(dict(case=name, quantity=field,
                                        max_abs_difference=float(np.max(abs(a-b)))))
                self.assert_metadata_equal(actual[name]['metadata'], want['metadata'])
        self.assert_globals_unchanged()
        self.equivalence_metrics = metrics

    def test_03_explicit_default_factory_preserves_default_call(self):
        implicit = solver.solve(self.problem, seed=17, **self.options)
        explicit = solver.solve(self.problem, seed=17, basis_factory=solver.trial_basis,
                                **self.options)
        self.assert_solve_equal(explicit, implicit)
        self.assert_globals_unchanged()

    def test_04_concurrent_distinct_factories_are_isolated(self):
        factories = (solver.trial_basis, mapped.mapped_legendre_factory)
        serial = [solver.solve(self.problem, seed=17, basis_factory=f, **self.options)
                  for f in factories]
        # Synchronize inside the factories to force overlapping customization.
        # Both factory bodies also assert that the public default is unchanged.
        for repeat in range(3):
            barrier = threading.Barrier(2)
            def run(factory):
                def synchronized(*args):
                    barrier.wait(timeout=15)
                    self.assert_globals_unchanged()
                    return factory(*args)
                return solver.solve(self.problem, seed=17, basis_factory=synchronized,
                                    **self.options)
            with ThreadPoolExecutor(max_workers=4) as executor:
                jobs = [executor.submit(run, factory) for factory in factories]
                default_job = executor.submit(solver.solve, self.problem, seed=17, **self.options)
                wrapper_job = executor.submit(mapped.solve, self.problem, seed=17, **self.options)
                actual = [job.result(timeout=30) for job in jobs]
                actual.extend((default_job.result(timeout=30), wrapper_job.result(timeout=30)))
            for got, want in zip(actual, (*serial, *serial)):
                self.assert_solve_equal(got, want)
            self.assert_globals_unchanged()

    def test_05_factory_and_assembly_exceptions_do_not_pollute_default(self):
        baseline = solver.solve(self.problem, seed=17, **self.options)
        def broken_factory(*args):
            raise RuntimeError('intentional factory failure')
        with self.assertRaisesRegex(RuntimeError, 'intentional factory failure'):
            solver.solve(self.problem, seed=17, basis_factory=broken_factory, **self.options)
        self.assert_globals_unchanged()

        class BrokenJets(base.LegendreFeatures):
            def jet(self, xy):
                raise RuntimeError('intentional jet assembly failure')
        def assembly_failure(problem, m, *args):
            return BrokenJets(problem, m)
        with self.assertRaisesRegex(RuntimeError, 'intentional jet assembly failure'):
            solver.solve(self.problem, seed=17, basis_factory=assembly_failure, **self.options)
        self.assert_globals_unchanged()
        self.assert_solve_equal(solver.solve(self.problem, seed=17, **self.options), baseline)

    def test_06_concurrent_failure_does_not_change_successful_solve(self):
        expected = mapped.solve(self.problem, seed=17, **self.options)
        barrier = threading.Barrier(2)
        def failing(*args):
            barrier.wait(timeout=15)
            raise RuntimeError('concurrent deliberate failure')
        def successful(*args):
            barrier.wait(timeout=15)
            return mapped.mapped_legendre_factory(*args)
        with ThreadPoolExecutor(max_workers=2) as executor:
            bad = executor.submit(solver.solve, self.problem, seed=17,
                                  basis_factory=failing, **self.options)
            good = executor.submit(solver.solve, self.problem, seed=17,
                                   basis_factory=successful, **self.options)
            with self.assertRaisesRegex(RuntimeError, 'concurrent deliberate failure'):
                bad.result(timeout=30)
            self.assert_solve_equal(good.result(timeout=30), expected)
        self.assert_globals_unchanged()

    def test_07_maintenance_has_no_archived_driver_imports(self):
        for name in ('rfm_solver', 'rfm_v2_solver', 'rfm_v2_warp_legendre',
                     'rfm_v2_benchmark', 'experiments', 'operator_opt', 'tensor_rfm', 'wall_trial'):
            self.assertNotIn(name, sys.modules)


if __name__=='__main__':
    unittest.main()
