"""Independent Re1000 FD4 reference. Warm starts are used only for this reference."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'rfm_ns'))
from cfd_reference import np, solve_cavity, comparison
from cfd_reference_fourth import CavityFourth


def load(path):
    with np.load(path,allow_pickle=False) as a:
        return {k:a[k] for k in ('x','y','u','v','omega','psi')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',default='64,128,192,256,384')
    args = p.parse_args()
    initial = load(ROOT.parent/'rfm_ns'/'cfd_reference4_cavity_Re400_n128.npz')
    records = []
    for n in map(int,args.grids.split(',')):
        target = ROOT/f'cavity1000_ref{n}.npz'
        if target.exists():
            current = load(target)
            with np.load(target,allow_pickle=False) as a:
                row = json.loads(str(a['record']))
        else:
            if n==64:
                initial,_ = solve_cavity(n,700,initial=initial,_model_class=CavityFourth)
            current,row = solve_cavity(n,1000,initial=initial,_model_class=CavityFourth,maxiter=35)
            np.savez_compressed(target,**current,record=json.dumps(row))
        if records:
            row['grid_change'] = comparison(initial,current,nq=160)
        records.append(row)
        (ROOT/'cavity1000_reference.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        print(n,row['total_seconds'],row['top_wall_max_abs_vorticity_201'],row.get('grid_change'),flush=True)
        initial = current


if __name__=='__main__':
    main()
