"""Run the CPU maintenance checks and save their observed results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

from .tests.test_factory import FactoryTests

import numpy
import scipy


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []
        self.metrics = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())
        self.metrics.extend(getattr(test, 'equivalence_metrics', []))


def main():
    directory = Path(__file__).resolve().parent
    begun = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FactoryTests))
    report = dict(
        status='passed' if result.wasSuccessful() else 'failed',
        created_utc=datetime.now(timezone.utc).isoformat(),
        command='python -m work.rfm_review_v3.maintenance.run_validation',
        scope='CPU correctness/maintenance only; no GPU work, accuracy certificate, or speed comparison',
        legacy_path='work/rfm_ns',
        maintenance_path='work/rfm_review_v3/maintenance',
        canonical_annulus_path='work/rfm_v6/full_galerkin.py (used by later v7/v8 runs; not modified)',
        tests_run=result.testsRun, tests_passed=result.passed,
        failures=[dict(test=test.id(), traceback=tb) for test, tb in result.failures],
        errors=[dict(test=test.id(), traceback=tb) for test, tb in result.errors],
        equivalence_cases=8, equivalence_field_points_per_case=25,
        equivalence_compared_fields=['u', 'v', 'omega', 'psi', 'div'],
        equivalence_rtol=1e-12, equivalence_atol=1e-12,
        equivalence_max_abs_differences=result.metrics,
        equivalence_history_and_metadata='floating values: rtol=atol=1e-12; other values: exact equality; two elapsed-time fields excluded',
        elapsed_seconds_diagnostic_only=time.perf_counter()-begun,
        python=sys.version, platform=platform.platform(), numpy=numpy.__version__, scipy=scipy.__version__,
        source_sha256={str(path.relative_to(directory)).replace('\\', '/'):
                       hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(directory.rglob('*.py'))})
    (directory/'validation_results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    main()
