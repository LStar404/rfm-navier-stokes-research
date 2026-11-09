"""CPU FD4 with buffered, fused radial and physical product kernels."""
import argparse
import ctypes as C
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V6 = HERE.parent / "rfm_v6"
sys.path.insert(0, str(V6))

from fd_full import FullFD, case, np
from scipy.fft import irfft, rfft

LIBRARY = HERE / ("fd_cpu_buffered.dll" if sys.platform == "win32" else "fd_cpu_buffered.so")
SETTINGS = dict(n=176, k=80, stretch=1., grid="outer", angular="fft", linear="native")
BASELINE = HERE / "cpu_formal_output" / "qualify_00_fd176_k80_17.npz"
REFERENCE = V6 / "parallel_refinement" / "fd320_ref_tk.npz"


def compile_library(compiler):
    """Rebuild a local shared library with no fast-math or fused rounding."""
    if LIBRARY.exists():
        raise FileExistsError(LIBRARY)
    flags = ["-O3", "-std=c99", "-shared", "-mavx2", "-ffp-contract=off"]
    if sys.platform == "win32":
        flags.extend(("-ffreestanding", "-nostdlib", "-nodefaultlibs", "-Wl,--entry,0"))
    else:
        flags.append("-fPIC")
    subprocess.run([compiler, *flags, str(HERE / "fd_cpu_buffered.c"), "-o", str(LIBRARY)],
                   check=True)


def csr_arrays(matrix):
    return (np.ascontiguousarray(matrix.indptr, dtype=np.int32),
            np.ascontiguousarray(matrix.indices, dtype=np.int32),
            np.ascontiguousarray(matrix.data, dtype=np.float64))


class BufferedFD(FullFD):
    """Reuse radial and spectral work arrays, with independent mass/flux outputs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.plan.backend != "fft":
            raise ValueError("BufferedFD implements the dealiased real FFT path")
        library = C.CDLL(str(LIBRARY))
        self._radial = library.fd_cpu_fields
        self._radial.argtypes = [C.c_void_p]*16 + [C.c_int]*3
        self._radial.restype = None
        self._products = library.fd_cpu_products
        self._products.argtypes = [C.c_void_p, C.c_int]
        self._products.restype = None
        modes, width = self.rhs_shape
        half = self.plan.n//2+1
        self._psi = np.zeros((width+2, modes), dtype=np.complex128)
        self._omega = np.zeros_like(self._psi)
        self._spectrum = np.zeros((4, width, half), dtype=np.complex128)
        self._mean = np.empty(width)
        self._negative_inverse = -self.inv_r[1:-1, None]
        # Retain the arrays for the lifetime of their native pointers.
        self._kernel_arrays = (*csr_arrays(self.d1), *csr_arrays(self.d2),
                               *csr_arrays(self.wall_d2[0]), self._psi, self.inv_r,
                               self.inv_r**2, self.left_d1, self._omega,
                               self._spectrum, self._mean)
        self._arguments = (*(a.ctypes.data for a in self._kernel_arrays), width, modes, half)
        self.metadata["implementation"] = "cpu-fused-fd4-csr-avx2"

    def _fill_radial(self, state, lift_c1):
        self._psi[1:-1] = state.T
        self._psi[-1, 0] = self.right_mean @ state[0, -4:]
        self._psi[-1, 1] = self.r[-1] * lift_c1
        self._radial(*self._arguments)

    def terms(self, state, lift_c1):
        self._fill_radial(state, lift_c1)
        mass = self._omega[1:-1].T.copy()
        mass[0] = state[0]
        physical = irfft(self._spectrum, n=self.plan.n, norm="forward", workers=1)
        self._products(physical.ctypes.data, physical[0].size)
        flux = rfft(physical[0], axis=-1, norm="forward", workers=1,
                    overwrite_x=True)[:, :self.rhs_shape[0]]
        flux *= self._negative_inverse
        flux[:, 0] = self._mean
        return mass, flux.T


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.finfo(float).tiny))


def fixed_audit():
    """Check radial fields, advective terms and the live multistep histories."""
    rng = np.random.default_rng(934567)
    records = []
    for n, k in ((152, 80), (176, 80), (320, 112)):
        cfg = dict(SETTINGS, n=n, k=k, dt=1/128)
        old, new = FullFD(**cfg), BufferedFD(**cfg)
        maxima = {}
        retained = []
        for i in range(5):
            state = rng.normal(size=old.rhs_shape)+1j*rng.normal(size=old.rhs_shape)
            if i % 2:
                state[0] = state[0].real
            amplitude = .14-.43j+.013j*i
            a_mass, a_flux = old.terms(state, amplitude)
            b_mass, b_flux = new.terms(state, amplitude)
            ur, ut, omega = old.native(state, amplitude, False)
            actual = (new._spectrum[0, :, :k+1], new._spectrum[1, :, :k+1],
                      new._omega, new._spectrum[2, :, :k+1], new._spectrum[3, :, :k+1])
            expected = (ur, ut, omega, old.d1@omega, 1j*old.k*omega[1:-1])
            for name, a, b in zip(
                    ("ur", "ut", "curl", "curl_s", "curl_theta", "mass", "flux", "mean_flux"),
                    (*expected, a_mass, a_flux, a_flux[0]),
                    (*actual, b_mass, b_flux, b_flux[0])):
                maxima[name] = max(maxima.get(name, 0.), relative(a, b))
            retained.append((b_mass, b_flux, b_mass.copy(), b_flux.copy()))
        survives = all(np.array_equal(a, a_saved) and np.array_equal(b, b_saved)
                       for a, b, a_saved, b_saved in retained)
        if not survives or max(maxima.values()) > 5e-13:
            raise ArithmeticError((cfg, maxima, survives))
        records.append(dict(n=n, k=k, max_relative_l2=maxima, retained_history_exact=survives))
    return records


def short_screen(horizon=2, repeats=3):
    times = [[], []]
    agreement = []
    for i in range(repeats):
        sol = [None, None]
        for index in ((0, 1) if i % 2 == 0 else (1, 0)):
            cls = (FullFD, BufferedFD)[index]
            sol[index] = case.solve(cls, steps=horizon*128, horizon=horizon, snapshots=(horizon,),
                                    **SETTINGS)
            times[index].append(sol[index].row["seconds"])
        agreement.append(relative(sol[0].states, sol[1].states))
    return dict(horizon=horizon, repeats=repeats, original_seconds=times[0], fused_seconds=times[1],
                original_median=median(times[0]), fused_median=median(times[1]),
                relative_state_l2=agreement, timing="non-isolated diagnostic")


def long_validation():
    from diagnostic_wake import analyze
    baseline = case.load_solution(BASELINE, FullFD)
    ref = case.load_solution(REFERENCE, FullFD)
    target = HERE / "fd_cpu_buffered_t72.npz"
    if target.exists():
        raise FileExistsError(target)
    if baseline.row["steps"] != 72*128 or any(
            baseline.row[key] != value for key, value in SETTINGS.items()):
        raise ValueError("Expected the matching, frozen FD176 baseline")
    grids = case.evaluation_grids()
    grids["layer144"] = case.layer_scoring_grid()
    start = time.perf_counter()
    sol = case.solve(BufferedFD, steps=72*128, **SETTINGS)
    elapsed = time.perf_counter() - start
    vs_cpu = case.score(sol, case.reference_fields(baseline, grids), grids)
    vs_ref = case.score(sol, case.reference_fields(ref, grids), grids)
    wake = analyze(sol.probe_times, sol.transverse, sol.vorticity)
    probe = {name: relative(getattr(baseline, name), getattr(sol, name))
             for name in ("transverse", "vorticity")}
    pass_accuracy = (max(case.maxima(vs_ref).values()) <= .005
                     and wake["sustained_alternating_probe_signs"])
    pass_equivalence = max(*probe.values(), *case.maxima(vs_cpu).values()) <= 1e-8
    case.store_solution(target, sol, SETTINGS)
    return dict(elapsed=elapsed, model=sol.row, comparison_to_cpu=case.maxima(vs_cpu),
                fields_vs_reference=case.maxima(vs_ref), score_rows_vs_cpu=vs_cpu,
                score_rows_vs_reference=vs_ref, wake=wake, probe_relative_l2=probe,
                accuracy_gate=.005, pass_accuracy=pass_accuracy,
                pass_equivalence=pass_equivalence, timing="non-isolated diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--compiler", default="x86_64-w64-mingw32-gcc" if sys.platform == "win32" else "cc")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "fd_cpu_buffered_pilot.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.build:
        compile_library(args.compiler)
    audit = fixed_audit()
    print("fixed audit", audit, flush=True)
    short = short_screen()
    print("short screen", short, flush=True)
    result = dict(fixed_audit=audit, short_screen=short)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if args.long:
        result["t72"] = long_validation()
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(args.output)
        print("t72", {name: result["t72"][name] for name in
                      ("elapsed", "comparison_to_cpu", "fields_vs_reference", "probe_relative_l2",
                       "pass_accuracy", "pass_equivalence")}, flush=True)


if __name__ == "__main__":
    main()
