"""Lower-cost spatial and temporal candidates for the full-circle run."""
import argparse
import json
from pathlib import Path

from full_galerkin_buffered import BufferedGalerkin
from full_galerkin import case, np
from fd_full import FullFD
from full_screen_high import LAYERS
from diagnostic_wake import analyze

ROOT=Path(__file__).resolve().parent


def spec(m,k,q,rate,**overrides):
    return rate,dict(m=m,k=k,q=q,scale=62.,mean_order=24,layers=LAYERS,split=True,**overrides)


SPECS={
    'rfm35_q60_k64':spec(35,64,60,128),
    'rfm37_q66_k72':spec(37,72,66,128),
    'rfm35_k72':spec(35,72,72,128),
    'rfm37_k76':spec(37,76,74,144),
    'rfm39_k80':spec(39,80,76,144),
    'rfm41_k80':spec(41,80,76,152),
    'rfm37_warp':spec(37,76,74,144,warp=1.35,mapping='both'),
    'leg33_k76':spec(33,76,74,144,family='legendre'),
    'leg35_q60_k64':spec(35,64,60,128,family='legendre'),
    'leg37_q66_k72':spec(37,72,66,128,family='legendre'),
    'leg39_k80':spec(39,80,76,152,family='legendre'),
    'rfm43_k88':spec(43,88,82,160),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',choices=tuple(SPECS),nargs='+')
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    reference=case.load_solution(ROOT/'fd_full_n200_k144_rate256_outer1.npz',FullFD)
    grids=case.evaluation_grids()
    truth=case.reference_fields(reference,grids)
    for name in args.names:
        rate,config=SPECS[name]
        settings=dict(config,seed=args.seed)
        print('START',name,args.seed,flush=True)
        try:
            with np.errstate(over='ignore',invalid='ignore'):
                sol=case.solve(BufferedGalerkin,steps=72*rate,**settings)
            scores=case.score(sol,truth,grids)
            wake=analyze(sol.probe_times,sol.transverse,sol.vorticity)
            row=dict(name=name,config=settings,rate=rate,metadata=sol.row,worst=case.maxima(scores),scores=scores,wake=wake)
            case.store_solution(ROOT/f'weak_candidate_{name}_{args.seed}.npz',sol,settings)
            print('DONE',name,sol.row['seconds'],row['worst'],wake['strouhal'],flush=True)
        except (ArithmeticError,ValueError,np.linalg.LinAlgError) as exc:
            row=dict(name=name,config=settings,rate=rate,failure=str(exc))
            print('FAILED',name,str(exc),flush=True)
        (ROOT/f'candidate_{name}_{args.seed}.json').write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    main()
