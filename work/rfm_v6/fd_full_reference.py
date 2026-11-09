"""Generate and compare frozen FD4 full-domain reference trajectories."""
import argparse
import json
from pathlib import Path

from fd_full import FullFD, ROOT, case


def tag(config, rate):
    suffix='' if config['grid']=='inner' else f"_{config['grid']}{config['stretch']:g}"
    return f"fd_full_n{config['n']}_k{config['k']}_rate{rate}{suffix}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int)
    parser.add_argument('--k',type=int)
    parser.add_argument('--rate',type=int)
    parser.add_argument('--stretch',type=float,default=2.1)
    parser.add_argument('--grid',default='inner')
    parser.add_argument('--time-order',type=int,choices=(2,3),default=2)
    parser.add_argument('--angular',default='fft')
    parser.add_argument('--linear',default='native')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--compare',nargs=2,type=Path)
    args = parser.parse_args()
    if args.compare:
        coarse,fine = [case.load_solution(p,FullFD) for p in args.compare]
        if not (coarse.times==fine.times).all():
            raise ValueError('Checkpoint times differ')
        grids = case.evaluation_grids()
        scores = case.score(coarse,case.reference_fields(fine,grids),grids)
        result = dict(coarse=str(args.compare[0]),fine=str(args.compare[1]),errors=case.maxima(scores),
                      scores=scores)
        print(json.dumps(result,indent=2),flush=True)
        if args.output:
            args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
        return
    config = dict(n=args.n,k=args.k,stretch=args.stretch,grid=args.grid,angular=args.angular,linear=args.linear)
    default = tag(config,args.rate)
    if args.time_order != 2:
        default = default.removesuffix('.npz')+f'_sbdf{args.time_order}.npz'
    target = args.output or ROOT/default
    if target.exists():
        raise FileExistsError(target)
    print('START',config,args.rate,flush=True)
    solution = case.solve(FullFD,steps=72*args.rate,time_order=args.time_order,**config)
    case.store_solution(target,solution,config)
    print('DONE',solution.row,'probe max',abs(solution.transverse[-192:]).max(),flush=True)


if __name__=='__main__':
    main()
