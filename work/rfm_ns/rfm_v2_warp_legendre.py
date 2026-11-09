"""Supplementary basis control with the same nonlinear coordinate map."""
import argparse
from unittest.mock import patch

from rfm_v2_benchmark import ROOT, SolverSpec, run_suite
import rfm_solver as base
import rfm_v2_solver as v2
from wall_trial import coordinate_jet


class MappedLegendre(base.LegendreFeatures):
    def __init__(self,problem,count,wall):
        super().__init__(problem,count)
        self.warp,self.strength = wall['warp'],wall['strength']

    def jet(self,xy):
        lo,length = self.problem.lo,self.problem.length
        coords = (xy-lo)/length
        maps = [coordinate_jet(z,k,self.warp) for z,k in zip(coords.T,self.strength)]
        mapped = lo+length*base.np.column_stack([a[0] for a in maps])
        composed = []
        for raw,(_,first,second) in zip(self.jets_1d(mapped,2),maps):
            first = first[:,None]
            composed.append(base.np.stack((raw[0],raw[1]*first,
                             raw[2]*first**2+raw[1]*second[:,None]/length)))
        return base.separable(*composed)

    def clamped_derivatives(self,xy):
        raise NotImplementedError('Mapped Legendre supports second jets for the weak solver')


def solve(problem,seed=17,**kwargs):
    def basis_factory(problem,m,seed,family,scale,wall):
        basis = MappedLegendre(problem,m,wall)
        perm = base.np.argsort(basis.ix+basis.iy,kind='stable')
        basis.ix,basis.iy = basis.ix[perm],basis.iy[perm]
        return basis
    with patch.object(v2,'trial_basis',basis_factory):
        return v2.solve(problem,seed,**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    common = dict(case='cavity',reynolds=400,scale=10,metric='h2_parity',
                  wall=dict(warp='sinh',strength=[1.3,1.3]))
    sources=('rfm_v2_warp_legendre.py','rfm_v2_solver.py','operator_opt.py','wall_trial.py')
    specs=[]
    for m,order,q,coarse in ((276,23,36,120),(325,25,38,136)):
        config=dict(common,m=m,order=order,q=q,coarse=coarse)
        specs.extend((SolverSpec(f'mapped_iid{m}',config,v2.solve,'native',sources),
                      SolverSpec(f'mapped_leg{m}',dict(config,family='legendre'),solve,'native',sources)))
    run_suite(specs,output=ROOT/'rfm_v2_basis_control.json',thresholds=(.001,.0005),include_fd=False,
              overwrite=args.overwrite)


if __name__=='__main__':
    main()
