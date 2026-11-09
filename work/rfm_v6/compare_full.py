"""Score saved runs on identical full-annulus, time-synchronized grids."""
import argparse
import json
from pathlib import Path

from full_galerkin import FullGalerkin, case
from fd_full import FullFD


def load(path):
    return case.load_solution(path,FullFD if path.name.startswith('fd_') else FullGalerkin)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidates',type=Path,nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--layer-grid',action='store_true')
    args=parser.parse_args()
    ref=load(args.reference)
    grids=case.evaluation_grids()
    if args.layer_grid:
        grids['layer144']=case.layer_scoring_grid()
    truth=case.reference_fields(ref,grids)
    results=[]
    for candidate in args.candidates:
        other=load(candidate)
        if not case.np.array_equal(other.times,ref.times):
            raise ValueError('Expected identical physical checkpoint times')
        rows=case.score(other,truth,grids)
        worst=case.maxima(rows)
        results.append(dict(file=candidate.name,worst=worst,scores=rows,metadata=other.row))
        print(candidate.name,{key:round(100*value,5) for key,value in worst.items()},flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(reference=args.reference.name,rows=results),indent=2),encoding='utf8')


if __name__=='__main__':
    main()
