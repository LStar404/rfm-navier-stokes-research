"""Validate the release manifest and optionally run a short solver replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    seen=set()
    for row in manifest['files']:
        name=row['path']
        path=ROOT/name
        if name in seen or path.resolve().relative_to(ROOT).as_posix()!=name:
            raise ValueError(f'Unsafe or repeated manifest path {name!r}')
        seen.add(name)
        with path.open('rb') as handle:
            checksum=hashlib.file_digest(handle,'sha256').hexdigest()
        if path.stat().st_size!=row['bytes'] or checksum!=row['sha256']:
            raise RuntimeError(f'Snapshot integrity failure: {name}')
    print(f'{len(seen)} snapshot files verified')


def smoke(portable):
    directory=ROOT/'work/rfm_v6'
    sys.path.insert(0,str(directory))
    import full_case as case
    from full_formal import ENGINES, SEEDS, boundary_errors

    specs=json.loads((directory/'full_specs.json').read_text(encoding='utf-8'))
    for kind in ('rfm','legendre','fd'):
        spec=next(item for item in specs if item['kind']==kind)
        config=dict(spec['config'])
        if kind=='rfm':
            config['seed']=SEEDS[0]
        if kind=='fd' and portable:
            config['linear']='scipy'
        result=case.solve(ENGINES[spec['engine']],steps=spec['rate']//2,
                          horizon=.5,snapshots=(.25,.5),**config)
        residual=max(boundary_errors(result).values())
        if (not case.np.isfinite(result.states).all() or not case.np.any(result.states)
                or not case.np.isfinite(residual) or residual>1e-7):
            raise RuntimeError(f'{kind} short replay failed, boundary residual={residual:g}')
        print(f'{kind} replay passed at t=0.5, max boundary residual {residual:.3g}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--portable',action='store_true',help='Use SciPy for the FD smoke replay.')
    args=parser.parse_args()
    verify()
    if args.smoke:
        smoke(args.portable)


if __name__=='__main__':
    main()
