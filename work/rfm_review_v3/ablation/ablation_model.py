"""Instance-local feature ablation; frozen v6/v7 solver implementations are unmodified."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import sys
WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "rfm_v7"))
from basis_recurrence_model import RecurrenceGalerkin
from cpu_profile import case, np
from radial_layers import radial_quadrature


class AblationGalerkin(RecurrenceGalerkin):
    """Change frequency jitter and nonzero-mode enrichments only.

    The six mean-block layers are deliberately retained in every arm.  The
    cutoff is unchanged; hence retained ranks may differ and are reported.
    """
    def __init__(self, *args, midpoint=False, nonzero_layers=True, **kwargs):
        self.midpoint = bool(midpoint)
        self.nonzero_layers = bool(nonzero_layers)
        self._frequency_ready = False
        super().__init__(*args, **kwargs)
        raw = self.raw_jets(self.s)
        _, weights = radial_quadrature(len(self.s), kwargs.get("split", False))
        values = (raw[2] - raw[0]) * (np.sqrt(weights) / self.r)[:, None]
        scale = np.linalg.norm(values, axis=0)
        singular = np.linalg.svd(values / scale[None, :], compute_uv=False)
        cutoff = float(kwargs.get("cutoff", 1e-10))
        kept = singular[singular > cutoff * singular[0]]
        self.metadata.update(midpoint=self.midpoint, nonzero_layers=self.nonzero_layers,
            frequency_design="fixed-midpoint" if self.midpoint else "registered-uniform-jitter",
            nonzero_layer_parameters=list(self.layers) if self.nonzero_layers else [],
            mean_layer_parameters=list(self.layers), raw_nonzero_width=raw.shape[-1],
            normalized_singular_values=singular.tolist(), svd_relative_cutoff=cutoff,
            retained_scaled_condition=float(kept[0] / kept[-1]),
            frequency_values=self.trial.frequency.tolist(), mean_block_unchanged=True)

    def raw_jets(self, s):
        if not self._frequency_ready:
            if self.midpoint:
                count = (self.trial.m - 1) // 2
                bins = (np.arange(count) + .5) * self.trial.scale / count
                self.trial.frequency = np.concatenate(([0.], np.repeat(bins, 2)))
            self._frequency_ready = True
        raw = super().raw_jets(s)
        return raw if self.nonzero_layers else raw[..., :self.trial.m]
