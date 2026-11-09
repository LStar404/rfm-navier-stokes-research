"""Independent CPU FD4 k=1 Taylor row evaluation with frozen pow2 bands."""

import ctypes as C
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess
import sys

from fd_pow2_long import Pow2EquilibratedBufferedFD, np


HERE = Path(__file__).resolve().parent
LIBRARY = HERE / ("fd_cpu_row_moment.dll" if sys.platform == "win32" else "fd_cpu_row_moment.so")
CLASS = "fd_cpu_row_moment.Pow2RowMomentFD"
VARIANT = "k1_taylor_d2_centered_d1"
TABLES = ("d1_mu0", "d2_mu0", "d2_mu1", "d2_delta", "span")


def moment_tables(model):
    """Use the exact binary64 weights and offsets, rounding each moment once."""
    width = model.d1.shape[0]
    table = {name: np.empty(width) for name in ("d1_mu0", "d2_mu0", "d2_mu1")}
    table["d2_delta"] = np.empty_like(model.d2.data)
    table["span"] = model.s[2:] - model.s[:-2]
    for row in range(width):
        center = row+1
        first = slice(model.d1.indptr[row], model.d1.indptr[row+1])
        second = slice(model.d2.indptr[row], model.d2.indptr[row+1])
        a1 = tuple(map(Fraction.from_float, model.d1.data[first]))
        a2 = tuple(map(Fraction.from_float, model.d2.data[second]))
        delta = model.s[model.d2.indices[second]] - model.s[center]
        table["d2_delta"][second] = delta
        table["d1_mu0"][row] = float(sum(a1))
        table["d2_mu0"][row] = float(sum(a2))
        table["d2_mu1"][row] = float(sum(a*Fraction.from_float(dx) for a, dx in zip(a2, delta)))
    return table


def moment_metadata(tables):
    return {name: dict(sha256=hashlib.sha256(tables[name].tobytes()).hexdigest(),
                       shape=list(tables[name].shape)) for name in TABLES}


def compile_library(compiler):
    if LIBRARY.exists():
        raise FileExistsError(LIBRARY)
    flags = ["-O3", "-std=c99", "-shared", "-mavx2", "-ffp-contract=off"]
    if sys.platform == "win32":
        flags += ["-ffreestanding", "-nostdlib", "-nodefaultlibs", "-Wl,--entry,0"]
    else:
        flags += ["-fPIC"]
    subprocess.run([compiler, *flags, str(HERE/"fd_cpu_row_moment.c"), "-o", str(LIBRARY)], check=True)


class Pow2RowMomentFD(Pow2EquilibratedBufferedFD):
    """Use the old buffer, FFT and solver paths with independent k=1 radial arithmetic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._moments = moment_tables(self)
        library = C.CDLL(str(LIBRARY))
        self._row_moment_fields = library.fd_cpu_row_moment_fields
        self._row_moment_fields.argtypes = [C.c_void_p]*21 + [C.c_int]*3
        self._row_moment_fields.restype = None
        self._moment_arguments = tuple(self._moments[name].ctypes.data for name in TABLES)
        self.metadata.update(implementation="cpu-fused-fd4-csr-avx2-row-moment",
                             reconstruction_class=CLASS, radial_variant=VARIANT,
                             moment_tables=moment_metadata(self._moments))

    def _fill_radial(self, state, lift_c1):
        self._psi[1:-1] = state.T
        self._psi[-1, 0] = self.right_mean @ state[0, -4:]
        self._psi[-1, 1] = self.r[-1] * lift_c1
        self._row_moment_fields(*self._arguments[:-3], *self._moment_arguments, *self._arguments[-3:])
