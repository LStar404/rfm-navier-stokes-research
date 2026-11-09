"""Independent FD4 sensitivity of the established wake to outer radius."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V6 = ROOT.parent/'rfm_v6'
sys.path.insert(0, str(V6))
from fd_full import FullFD, case
from full_case import np
from diagnostic_wake import analyze

OUT = ROOT/'domain_output'
REF6 = V6/'parallel_refinement'/'fd320_ref_tk.npz'
R8_CASES = ((256,112,256),(320,144,320))


def set_outer(radius):
    case.R = radius
    case.LENGTH = np.log(radius/case.A)


def run_refinement(spec):
    n,k,rate = spec
    target = OUT/f'fd_R8_n{n}_k{k}_rate{rate}.npz'
    if target.exists():
        return target
    set_outer(8.)
    config = dict(n=n,k=k,grid='outer',stretch=1.,angular='fft',linear='native')
    print('START',target.name,flush=True)
    solution = case.solve(FullFD,steps=72*rate,**config)
    temporary = target.with_suffix('.pending.npz')
    case.store_solution(temporary,solution,config)
    temporary.replace(target)
    print('DONE',target.name,flush=True)
    return target


def common_grid():
    x,w = case.leggauss(144)
    length = np.log(4/case.A)
    s = (x+1)*length/2
    theta = 2*np.pi*(np.arange(433)+.23)/433
    return {'common_r_le_4':(s,theta,w*(length/2)*case.radius(s)**2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    result_path = OUT/'domain_sensitivity.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(run_refinement,R8_CASES))
    grids_common = common_grid()
    set_outer(6.)
    old_domain = case.load_solution(REF6,FullFD)
    old_truth = case.reference_fields(old_domain,grids_common)
    old_wake = analyze(old_domain.probe_times,old_domain.transverse,old_domain.vorticity)
    set_outer(8.)
    coarse,fine = [case.load_solution(path,FullFD) for path in files]
    fine_wake = analyze(fine.probe_times,fine.transverse,fine.vorticity)
    coarse_wake = analyze(coarse.probe_times,coarse.transverse,coarse.vorticity)
    scores = case.score(fine,old_truth,grids_common)
    r8_grids = case.evaluation_grids()
    r8_grids['layer144'] = case.layer_scoring_grid()
    refinement = case.score(coarse,case.reference_fields(fine,r8_grids),r8_grids)
    data = dict(
        reference_R6=dict(file=REF6.name,wake=old_wake,setup=old_domain.row),
        reference_R8=dict(file=files[1].name,wake=fine_wake,setup=fine.row),
        predecessor_R8=dict(file=files[0].name,wake=coarse_wake,setup=coarse.row),
        common_unshifted=case.maxima(scores), common_scores=scores,
        R8_refinement=case.maxima(refinement), R8_refinement_scores=refinement,
        policy='R=6 and R=8 share finite-boundary conditions, perturbation and checkpoint times. Common r<=4 field comparisons are phase-sensitive. This is not an infinite-domain limit or a timing comparison.')
    result_path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')


if __name__ == '__main__':
    main()
