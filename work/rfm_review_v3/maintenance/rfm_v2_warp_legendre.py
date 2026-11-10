"""Mapped-Legendre control for the archived steady 2-D weak solver.

This maintenance interface has no import-time benchmark dependencies and never
patches another module.  See README.md for scope and historical call sites.
"""
from . import rfm_solver as base
from . import rfm_v2_solver as v2
from .wall_trial import coordinate_jet

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



def mapped_legendre_factory(problem, m, seed, family, scale, wall):
    """Build the same degree-ordered mapped basis as the archived wrapper."""
    basis = MappedLegendre(problem, m, wall)
    perm = base.np.argsort(basis.ix+basis.iy, kind='stable')
    basis.ix, basis.iy = basis.ix[perm], basis.iy[perm]
    return basis


def solve(problem, seed=17, **kwargs):
    return v2.solve(problem, seed, basis_factory=mapped_legendre_factory, **kwargs)
