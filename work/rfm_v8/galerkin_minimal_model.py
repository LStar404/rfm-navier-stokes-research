"""Numerical recurrence model with no CPU profiling or qualification imports."""

from contextlib import contextmanager
from pathlib import Path
import sys

V7 = Path(__file__).resolve().parents[1] / "rfm_v7"
V6 = V7.with_name("rfm_v6")
for directory in (V7, V6):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import full_case as case
import full_galerkin_buffered as buffered
import cylinder_rfm
from radial_layers import layer_jets

np = case.np


def unit_legendre_jets(x, count):
    """P_j(2x-1) and its first two x derivatives for 0 <= j < count."""
    if count < 1:
        raise ValueError('Require a positive basis count')
    x = np.asarray(x)
    xi = 2*x-1
    jets = np.zeros((3, len(x), count))
    jets[0, :, 0] = 1.
    if count > 1:
        jets[0, :, 1] = xi
        jets[1, :, 1] = 2.
    for j in range(1, count-1):
        a, b = (2*j+1)/(j+1), j/(j+1)
        jets[0, :, j+1] = a*xi*jets[0, :, j] - b*jets[0, :, j-1]
        jets[1, :, j+1] = a*(2*jets[0, :, j]+xi*jets[1, :, j])-b*jets[1, :, j-1]
        jets[2, :, j+1] = a*(4*jets[1, :, j]+xi*jets[2, :, j])-b*jets[2, :, j-1]
    return jets


def polynomial(s, count, warp, mapping='inner', outer='farfield', trial=False):
    z = np.asarray(s)/cylinder_rfm.case.LENGTH
    x, d, dd = cylinder_rfm.mapped_coordinate(z, warp, mapping)
    value, first, second = unit_legendre_jets(x, count)
    edge = None
    if trial and outer == 'farfield':
        edge_d = cylinder_rfm.mapped_coordinate(np.array([1.]), warp, mapping)[1][0]
        j = np.arange(count)
        edge = (1., edge_d*j*(j+1))
    return cylinder_rfm.clamped((value, first*d[:, None],
        second*d[:, None]**2+first*dd[:, None]), z, outer, edge)


def mean_jets(s, m):
    z = np.asarray(s)/case.LENGTH
    values, derivative, _ = unit_legendre_jets(z, m-1)
    weight = z*(1-z)**2
    dw = (1-z)*(1-3*z)
    return np.stack((
        np.column_stack((z*((case.LENGTH+2)-(case.LENGTH+1)*z), weight[:, None]*values)),
        np.column_stack((case.LENGTH+2-2*(case.LENGTH+1)*z,
                         dw[:, None]*values+weight[:, None]*derivative))/case.LENGTH))


class StridedGalerkin(buffered.BufferedGalerkin):
    """Keep both input and output in radial-first complex storage."""

    def advance(self, rhs, amplitude, first):
        if first or rhs.dtype != np.complex128 or not rhs.flags.c_contiguous:
            return super().advance(rhs, amplitude, first)
        rhs[:, 1] -= amplitude * self.bdf_lift
        out = np.empty_like(rhs)
        # Adjacent real and imaginary columns remain one real matrix RHS.
        shape = (self.width, self.k + 1, 2)
        values = rhs.view(float).reshape(shape)[:, 1:].transpose(1, 0, 2)
        result = out.view(float).reshape(shape)[:, 1:].transpose(1, 0, 2)
        np.matmul(self.implicit, values, out=result)
        out[:, 0] = 0.
        out[:self.mean_width, 0] = self.mean_inverse @ rhs[:self.mean_width, 0].real
        return out


class RecurrenceGalerkin(StridedGalerkin):
    def raw_jets(self, s):
        if self.trial.family != 'legendre':
            return super().raw_jets(s)
        scale = cylinder_rfm.case.LENGTH / case.LENGTH
        raw = polynomial(s*scale, self.trial.m, self.trial.warp,
            self.trial.mapping, self.trial.outer, trial=True)
        raw = raw * np.array([1., scale, scale**2])[:, None, None]
        return np.concatenate((raw, layer_jets(s, self.layers)), axis=-1) if self.layers else raw

    def raw_mean(self, s):
        raw = mean_jets(s, self.mean_order)
        return np.concatenate((raw, layer_jets(s, self.layers, mean=True)), axis=-1) if self.layers else raw

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata['basis_evaluation'] = 'differentiated-recurrence'


@contextmanager
def native_model():
    """Select a constructor; returned models retain their own evaluation methods."""
    import galerkin_cuda_native as native
    old = native.BufferedGalerkin
    try:
        native.BufferedGalerkin = RecurrenceGalerkin
        yield
    finally:
        native.BufferedGalerkin = old
