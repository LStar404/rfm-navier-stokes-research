"""Screen SBDF3 and quantify late phase sensitivity at the same resolution."""
import json
from pathlib import Path

from full_galerkin_buffered import BufferedGalerkin
from full_galerkin import case, np
from full_screen_high import SPECS
from diagnostic_wake import analyze

ROOT=Path(__file__).resolve().parent


def main():
    config=SPECS['layer_rfm35'][1]
    rows=[]
    for order,rate in ((3,160),(3,192),(2,256)):
        print('START',order,rate,flush=True)
        try:
            with np.errstate(over='ignore',invalid='ignore'):
                sol=case.solve(BufferedGalerkin,steps=72*rate,time_order=order,**config)
            wake=analyze(sol.probe_times,sol.transverse,sol.vorticity)
            case.store_solution(ROOT/f'weak_layer_rfm35_bdf{order}_rate{rate}.npz',sol,config)
            row=dict(order=order,rate=rate,wake=wake,metadata=sol.row)
            print('DONE',order,rate,wake['strouhal'],sol.row['seconds'],flush=True)
        except (ArithmeticError,ValueError,np.linalg.LinAlgError) as exc:
            row=dict(order=order,rate=rate,failure=str(exc))
            print('FAILED',order,rate,str(exc),flush=True)
        rows.append(row)
        (ROOT/'time_full_pilot.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    main()
