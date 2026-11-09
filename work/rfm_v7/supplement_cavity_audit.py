"""Read-only audit of the archived smooth-lid Re1000 supplementary case.

No old trajectory, source, frozen protocol or timing is overwritten. Only the
first seed's sampled fields were saved by v4; other seeds and v5 can be checked
against their recorded scores, not independently reconstructed from this audit.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
TREE = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent/'rfm_v3'))
from cavity_study import Score, errors, np
from rfm_v2_benchmark import weighted_errors


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    v4p = ROOT.parent/'rfm_v4'/'cavity_formal.json'
    v5p = ROOT.parent/'rfm_v5'/'cavity_paired.json'
    v4, v5 = read(v4p), read(v5p)
    f4p = v4p.with_name('cavity_formal_freeze.json')
    f5p = v5p.with_name('cavity_paired_freeze.json')
    f4, f5 = read(f4p), read(f5p)
    assert v4['status'] == v5['status'] == 'complete'
    assert digest(f4p) == v4['freeze_sha256']
    assert digest(f5p) == v5['freeze_sha256']
    frozen_sources = []
    for collection, mapping in [('v4', f4['source_sha256']), ('v5', f5['hashes'])]:
        for name, sha in mapping.items():
            path = Path(name) if Path(name).is_absolute() else TREE/name
            actual = digest(path)
            frozen_sources.append(dict(collection=collection,
                                       path=path.relative_to(TREE).as_posix(),
                                       sha256=actual, matches=actual == sha))
    assert all(x['matches'] for x in frozen_sources), 'Archived dependency changed'
    score = Score()
    recomputed_reference = score.grid_change
    score_keys = ('velocity_rel_l2', 'vorticity_rel_l2')
    for grid in score.grids:
        for key in score_keys:
            assert math.isclose(recomputed_reference[grid][key],
                                v4['reference_grid_change'][grid][key], rel_tol=1e-10, abs_tol=1e-15)
    assert len(v4['rows']) == 50 and len(v5['rows']) == 20
    expected4 = {(s['name'], seed) for s in f4['specs'] for seed in f4['seeds']}
    expected5 = {(int(m), method, seed) for m in f5['gates']
                 for method in ('v4', 'v5') for seed in f5['seeds']}
    assert {(r['name'], r['seed']) for r in v4['rows']} == expected4
    assert {(r['m'], r['method'], r['seed']) for r in v5['rows']} == expected5
    checked_rows = []
    sampled_field_replay = []
    for collection, rows in [('v4', v4['rows']), ('v5', v5['rows'])]:
        for row in rows:
            assert row['status'] == 'completed'
            assert row['seconds'] > 0 and math.isfinite(row['seconds'])
            assert set(row['scores']) == set(score.grids)
            worst = errors(row['scores'])
            assert all(math.isfinite(v) and v >= 0 for v in worst.values())
            stored = row['errors'] if collection == 'v4' else row['worst']
            assert worst == stored
            gate_checks = ({str(t): max(worst.values()) <= t for t in f4['thresholds']}
                           if collection == 'v4' else
                           {str(f5['gates'][str(row['m'])]): max(worst.values()) <= f5['gates'][str(row['m'])]})
            if collection == 'v4':
                assert gate_checks == row['passes']
            else:
                assert next(iter(gate_checks.values())) == row['passed']
            checked_rows.append(dict(collection=collection,
                                     name=row.get('name', f"{row.get('method')}_m{row.get('m')}"),
                                     seed=row['seed'], seconds=row['seconds'],
                                     worst=worst, gate_checks=gate_checks))
            if 'artifact' in row:
                path = v4p.parent/row['artifact']['file']
                assert digest(path) == row['artifact']['sha256']
                checked = {}
                with np.load(path, allow_pickle=False) as data:
                    for grid, (xy, weights) in score.grids.items():
                        assert np.array_equal(data[f'{grid}_xy'], xy)
                        assert np.array_equal(data[f'{grid}_weights'], weights)
                        fields = {k: data[f'{grid}_{k}'] for k in ('u', 'v', 'omega')}
                        checked[grid] = weighted_errors(fields, score.truth[grid], weights)
                        for key in score_keys:
                            assert math.isclose(checked[grid][key], row['scores'][grid][key],
                                                rel_tol=1e-8, abs_tol=1e-12)
                sampled_field_replay.append(dict(name=row['name'], seed=row['seed'],
                                                artifact=path.relative_to(TREE).as_posix(),
                                                sha256=digest(path), scores=checked))
    report = dict(
        status='passed',
        problem=dict(case='stationary smooth-lid unit square cavity', Re=1000,
                     top_u='16*x^2*(1-x)^2', top_v=0,
                     other_walls='no slip', time_integration=False,
                     note='This is not the discontinuous constant-lid benchmark.'),
        sources=frozen_sources,
        inputs={p.relative_to(TREE).as_posix(): digest(p) for p in (v4p, v5p, f4p, f5p)},
        reference_grid_change=recomputed_reference,
        reference_screen={str(t): max(errors(recomputed_reference).values()) <= .2*t for t in f4['thresholds']},
        checked_rows=checked_rows, sampled_field_replay=sampled_field_replay,
        archived_v4_comparisons=v4['comparisons'], archived_v5_summary=v5['summary'],
        limitations=[
            'Archived CPU timings are not newly measured and cannot be paired with v7 cylinder timings.',
            'Both archived campaigns define cold as fresh in-process solves, not fresh OS processes.',
            'v4 compared fastest passing configurations from its frozen tested candidate set, not all possible solvers.',
            'Only the first v4 seed per configuration has a saved field artifact (10 total).',
            'The remaining v4 and all v5 scores can be checked internally but were not independently regenerated.',
            'A two-grid empirical reference screen is not a certified continuum error bound.',
            'The v5 m595 pair is not a same-accuracy speed comparison: v4 passes only four of five seeds.',
            'Stationary smooth-lid results do not establish turbulent or general transient advantages.',
        ])
    out = ROOT/'supplement_cavity_audit.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], source_records=len(frozen_sources),
                         score_rows=len(checked_rows), sampled_field_replays=len(sampled_field_replay),
                         reference_screen=report['reference_screen']), indent=2))


if __name__ == '__main__':
    main()
