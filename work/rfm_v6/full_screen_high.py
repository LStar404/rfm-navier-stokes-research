"""Refine time and space independently for the long-wake Galerkin pilots."""
import argparse
import json
from pathlib import Path

from full_galerkin import FullGalerkin, case, np
from diagnostic_wake import analyze

ROOT=Path(__file__).resolve().parent
LAYERS=(16.,36.,80.,165.,340.,660.)
SPECS={
    'leg_ref':(256,dict(m=61,k=120,q=96,family='legendre',mean_order=30)),
    'leg_ref_dt':(192,dict(m=61,k=120,q=96,family='legendre',mean_order=30)),
    'rfm_ref':(224,dict(m=63,k=108,q=88,scale=92.,mean_order=30)),
    'rfm_high_dt':(224,dict(m=51,k=88,q=74,scale=69.,mean_order=28)),
    'leg_ref_warp':(224,dict(m=60,k=112,q=94,family='legendre',mean_order=30,warp=1.0)),
    'layer_rfm35':(160,dict(m=35,k=88,q=78,scale=50.,mean_order=21,layers=LAYERS,split=True)),
    'layer_rfm45':(192,dict(m=45,k=100,q=90,scale=65.,mean_order=24,layers=LAYERS,split=True)),
    'layer_rfm51':(224,dict(m=51,k=112,q=100,scale=78.,mean_order=28,layers=LAYERS,split=True)),
    'layer_leg39':(160,dict(m=39,k=88,q=78,family='legendre',mean_order=21,layers=LAYERS,split=True)),
    'layer_leg49':(192,dict(m=49,k=100,q=90,family='legendre',mean_order=24,layers=LAYERS,split=True)),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',choices=tuple(SPECS),nargs='+')
    args=parser.parse_args()
    for name in args.names:
        rate,config=SPECS[name]
        print('START',name,rate,config,flush=True)
        with np.errstate(over='ignore',invalid='ignore'):
            sol=case.solve(FullGalerkin,steps=72*rate,**config)
        wake=analyze(sol.probe_times,sol.transverse,sol.vorticity)
        case.store_solution(ROOT/f'weak_{name}.npz',sol,config)
        (ROOT/f'weak_{name}.json').write_text(json.dumps(dict(wake=wake,metadata=sol.row),indent=2),encoding='utf8')
        print('DONE',name,sol.row['seconds'],wake['strouhal'],flush=True)


if __name__=='__main__':
    main()
