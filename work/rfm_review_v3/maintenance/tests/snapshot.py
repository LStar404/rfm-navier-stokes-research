"""Emit small deterministic CPU solves, using either archived or maintained code."""
import argparse
import json
from pathlib import Path
import sys


CASES = [
    dict(name='iid_diagonal', case='cavity', reynolds=100, seed=17,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4)),
    dict(name='iid_h2_parity', case='cavity', reynolds=400, seed=33,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4, metric='h2_parity')),
    dict(name='iid_h2_dense', case='cavity', reynolds=400, seed=33,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4, metric='h2_dense')),
    dict(name='legendre_prefix_reuse', case='cavity', reynolds=100, seed=17,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4,
                      family='legendre', coarse=6, reuse=2, metric='h2_parity')),
    dict(name='sobol', case='kovasznay', reynolds=40, seed=17,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4, family='sobol')),
    dict(name='pairs', case='kovasznay', reynolds=40, seed=17,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4, family='pairs')),
    dict(name='wall_rfm', case='cavity', reynolds=400, seed=17,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4,
                      metric='h2_parity', wall=dict(warp='sinh', strength=[1.3, 1.3]))),
    dict(name='mapped_legendre', case='cavity', reynolds=400, seed=17,
         options=dict(m=15, scale=3., order=5, q=11, maxiter=4,
                      metric='h2_parity', wall=dict(warp='sinh', strength=[1.3, 1.3]))),
]


def collect(base, solver, mapped):
    records = {}
    for case in CASES:
        problem = base.Problem(case['case'], case['reynolds'])
        solve = mapped.solve if case['name']=='mapped_legendre' else solver.solve
        op, coefficients, row = solve(problem, seed=case['seed'], **case['options'])
        z = base.np.linspace(.13, .87, 5)
        xy = problem.lo+problem.length*base.np.stack(base.np.meshgrid(z, z), -1).reshape(-1, 2)
        fields = op.fields(coefficients, xy)
        records[case['name']] = dict(
            coefficients=coefficients.tolist(),
            fields=[fields[key].tolist() for key in ('u', 'v', 'omega', 'psi', 'div')],
            stokes=op.stokes.tolist(),
            metadata={key: value for key, value in row.items()
                      if key not in ('setup_seconds', 'seconds')})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archived', action='store_true')
    parser.add_argument('--archive-root', type=Path,
                        help='Repository root supplying the unchanged archived work/rfm_ns modules')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    if args.archived:
        archive_root = root if args.archive_root is None else args.archive_root.resolve()
        # The published archive omits certify_rfm.py, an import-time dependency
        # of operator_opt's unused audit driver. Supply an exact-byte archival
        # copy in this child process, not a stub or a substitute numerical kernel.
        support = Path(__file__).resolve().parent/'archive_support'
        sys.path[:0] = [str(support), str(archive_root/'work/rfm_ns')]
        import rfm_solver as base
        import rfm_v2_solver as solver
        import rfm_v2_warp_legendre as mapped
    else:
        sys.path.insert(0, str(root))
        from work.rfm_review_v3.maintenance import rfm_solver as base
        from work.rfm_review_v3.maintenance import rfm_v2_solver as solver
        from work.rfm_review_v3.maintenance import rfm_v2_warp_legendre as mapped
    result = collect(base, solver, mapped)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False,
                                          default=lambda value: value.item())+'\n', encoding='utf8')


if __name__=='__main__':
    main()
