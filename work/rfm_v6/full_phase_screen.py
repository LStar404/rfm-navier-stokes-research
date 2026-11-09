"""Probe-only time-lag diagnosis, separate from the unshifted accuracy gates."""
import argparse
import json
from pathlib import Path

from full_nonlinear import np
from scipy.optimize import minimize_scalar


def probes(path):
    with np.load(path,allow_pickle=False) as record:
        return record['probe_times'],record['transverse']


def compare(time,value,reference_time,reference):
    rows=[]
    for lo,hi in ((14,24),(24,36),(36,48),(48,60),(60,69)):
        keep=(time>=lo)&(time<=hi)
        t=time[keep]
        series=value[keep]
        def error(lag):
            shifted=np.column_stack([np.interp(t+lag,reference_time,r) for r in reference.T])
            return float(np.linalg.norm(series-shifted)/np.linalg.norm(shifted))
        grid=np.linspace(-2.75,2.75,111)
        j=int(np.argmin([error(lag) for lag in grid]))
        estimate=minimize_scalar(error,bounds=(grid[max(0,j-1)],grid[min(j+1,len(grid)-1)]),
                                 method='bounded',options={'xatol':1e-8})
        rows.append(dict(window=[lo,hi],lag=float(estimate.x),unshifted=error(0.),shifted=float(estimate.fun)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidates',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    args=parser.parse_args()
    ref_t,ref=probes(args.reference)
    results={}
    for path in args.candidates:
        t,v=probes(path)
        results[path.name]=compare(t,v,ref_t,ref)
        print(path.name,[(r['window'],round(r['lag'],4),round(r['unshifted'],4),round(r['shifted'],4))
                         for r in results[path.name]],flush=True)
    args.output.write_text(json.dumps(dict(reference=args.reference.name,rows=results),indent=2),encoding='utf8')


if __name__=='__main__':
    main()
