"""Persistent model class for the validated recurrence-based basis evaluation."""
from contextlib import contextmanager

import basis_recurrence as rec
from cpu_implicit_stride import StridedGalerkin
from radial_layers import layer_jets

np = rec.np


class RecurrenceGalerkin(StridedGalerkin):
    def raw_jets(self, s):
        if self.trial.family != 'legendre':
            return super().raw_jets(s)
        scale = rec.cylinder_rfm.case.LENGTH / rec.base.case.LENGTH
        raw = rec.polynomial(s*scale, self.trial.m, self.trial.warp,
            self.trial.mapping, self.trial.outer, trial=True)
        raw = raw * np.array([1., scale, scale**2])[:, None, None]
        return np.concatenate((raw, layer_jets(s, self.layers)), axis=-1) if self.layers else raw

    def raw_mean(self, s):
        raw = rec.mean_jets(s, self.mean_order)
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
