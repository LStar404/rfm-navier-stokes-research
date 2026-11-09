"""Opt-in FD4 row equilibration with exact FP64 power-of-two scales."""
import time
from unittest.mock import patch

from fd_equilibration_precision import base, fd_full, maxima, np, scale_for, sparse

NativeBand = fd_full.BandSolve


class PowerOfTwoBand(NativeBand):
    """Store factors of D A and solve (D A) x = D b."""
    def __init__(self, matrices, backend):
        start = time.perf_counter()
        self.row_scale = np.stack([scale_for(maxima(a,1),True) for a in matrices])
        scaled = [sparse.diags(d)@a for d,a in zip(self.row_scale,matrices)]
        self.equilibration_seconds = time.perf_counter()-start
        super().__init__(scaled,backend)

    def __call__(self, rhs):
        if self.native is None:
            return super().__call__(self.row_scale*rhs)
        # Scale directly into the same independent C-order work array that
        # the original native factor application would otherwise copy.
        result = np.empty(self.row_scale.shape,dtype=np.complex128)
        np.multiply(self.row_scale,rhs,out=result)
        self.native(self.factors.ctypes.data,self.pivots.ctypes.data,
                    result.ctypes.data,self.modes,self.width,self.lower,self.upper)
        return result


class PowerOfTwoFD(base.FullFD):
    def __init__(self,*args,**kwargs):
        with patch.object(fd_full,"BandSolve",PowerOfTwoBand):
            super().__init__(*args,**kwargs)
        self.metadata.update(equilibration="row_power_of_two",
            equilibration_setup_seconds=self.bdf.equilibration_seconds+self.first.equilibration_seconds)
