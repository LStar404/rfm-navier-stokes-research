"""Experimental BDF2 with extrapolated vorticity and implicit Lamb velocity.

This changes the time discretization. It is not a bitwise optimization of SBDF2.
The homogeneous frozen-vorticity operator is skew in the real Fourier pairing.
"""
import time
from galerkin_minimal_model import RecurrenceGalerkin, case, np
from full_galerkin import real_apply
from scipy.sparse.linalg import LinearOperator, gmres


class LambGalerkin(RecurrenceGalerkin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_omega = None
        self.frozen_evaluations = 0
        self.velocity_products = 0
        self.sqrt_two = np.sqrt(2.)
        self.real_size = self.mean_width + 2*self.width*self.k

    def pack(self, state):
        return np.concatenate((state[:self.mean_width, 0].real,
                               self.sqrt_two*np.ascontiguousarray(state[:, 1:]).view(float).ravel()))

    def unpack(self, vector):
        state = np.zeros(self.rhs_shape, dtype=np.complex128)
        state[:self.mean_width, 0] = vector[:self.mean_width]
        state[:, 1:] = np.ascontiguousarray(vector[self.mean_width:]/self.sqrt_two).view(np.complex128).reshape(self.width,self.k)
        return state

    def mass_only(self, state, amplitude=0.):
        mass = np.zeros_like(state)
        values = np.ascontiguousarray(state[:, 1:])
        mass[:, 1:] = real_apply(self.M0, values) + real_apply(self.M1, values)*self.k2
        mass[:, 1] += amplitude*self.M_lift
        mass[:self.mean_width, 0] = self.mean_mass@state[:self.mean_width, 0].real
        return mass

    def freeze_vorticity(self, coefficients):
        self.frozen_omega = self.kernel._values((coefficients,))[0]
        self.frozen_evaluations += 1

    def flux_frozen(self, state, amplitude=0.):
        # Reuse the exact velocity reconstruction; unused curl reconstruction is
        # retained in this correctness-first CPU prototype and counted as cost.
        ur, ut, _ = self.fields_on(state, amplitude, self.samples, self.lift, self.r, self.mean_samples)
        values = self.kernel._values((ur, ut))
        values *= self.frozen_omega
        radial, angular = self.kernel._coefficients(values)
        flux = np.zeros_like(state)
        flux[:, 1:] = real_apply(self.test_s, np.ascontiguousarray(radial[:, 1:])) - self.ik*real_apply(
            self.test, np.ascontiguousarray(angular[:, 1:]))
        flux[:self.mean_width, 0] = -self.mean_test@radial[:, 0].real
        self.velocity_products += 1
        return flux

    def inverse(self, rhs):
        return self.advance(rhs.copy(), 0., False)

    def implicit_step(self, state, prior, amplitudes, h, rtol=1e-11):
        gprev, gnow, gnext = amplitudes
        # BufferedGalerkin returns borrowed arrays. Reconstruct the extrapolated
        # state once, then consume its curl before any other reconstruction.
        omega_star = self.fields_on(2*state-prior, 2*gnow-gprev, self.samples, self.lift, self.r, self.mean_samples)[2]
        self.freeze_vorticity(omega_star)
        rhs = 2*self.mass_only(state, gnow)-.5*self.mass_only(prior, gprev)
        rhs[:, 1] -= gnext*self.bdf_lift
        rhs += h*self.flux_frozen(np.zeros_like(state), gnext)
        b = self.pack(self.inverse(rhs))
        calls_before = self.velocity_products
        history = []

        def matvec(value):
            return value-h*self.pack(self.inverse(self.flux_frozen(self.unpack(value))))

        operator = LinearOperator((self.real_size,self.real_size), matvec=matvec, dtype=np.float64)
        answer, info = gmres(operator, b, x0=self.pack(2*state-prior), rtol=rtol, atol=0., restart=24,
                             maxiter=10, callback=history.append, callback_type="pr_norm")
        residual = np.linalg.norm(matvec(answer)-b)/max(np.linalg.norm(b), np.finfo(float).tiny)
        if info != 0 or not np.isfinite(answer).all() or residual > 2*rtol:
            raise ArithmeticError(f"GMRES did not converge: info={info}, residual={residual:.3e}")
        return self.unpack(answer), dict(iterations=len(history), operator_calls=self.velocity_products-calls_before,
                                       preconditioned_residual=float(residual))


def solve_lamb(*, rate, horizon, snapshots, rtol=1e-11, **config):
    started = time.perf_counter()
    h = 1/rate
    steps = int(round(rate*horizon))
    if steps != rate*horizon:
        raise ValueError("Expected an integer step count")
    model = LambGalerkin(dt=h, **config)
    setup = time.perf_counter()-started
    times = np.asarray(snapshots, dtype=float)
    snapshot_steps = set(case.step_indices(times, h))
    probe_times = np.arange(0., horizon+.001, .125)
    probe_steps = dict(zip(case.step_indices(probe_times,h), range(len(probe_times)), strict=True))
    transverse = np.zeros((len(probe_times),len(case.PROBES)))
    vorticity = np.zeros_like(transverse)
    amplitude = case.speed(h*np.arange(steps+1))
    zero = np.zeros(model.rhs_shape, dtype=np.complex128)
    prior = zero.copy()
    state = model.advance(zero.copy(), amplitude[1], True)
    saved, work = [], []
    for i in range(1,steps+1):
        if i > 1:
            newer, row = model.implicit_step(state,prior,amplitude[i-2:i+1],h,rtol)
            work.append(row)
            prior, state = state, newer
        if i in snapshot_steps:
            saved.append(state.copy())
        if i in probe_steps:
            transverse[probe_steps[i]],vorticity[probe_steps[i]] = model.probes(state,amplitude[i])
    row = dict(model.metadata,dt=h,steps=steps,horizon=horizon,time_order=2,setup_seconds=setup,
               seconds=time.perf_counter()-started,implementation="experimental-bdf2-frozen-vorticity-lamb",
               linear_tolerance=rtol,frozen_vorticity_evaluations=model.frozen_evaluations,
               velocity_product_evaluations=model.velocity_products,
               inverse_real_transforms=model.frozen_evaluations+2*model.velocity_products,
               forward_real_transforms=2*model.velocity_products,
               gmres_iterations=sum(w["iterations"] for w in work),
               max_preconditioned_residual=max(w["preconditioned_residual"] for w in work))
    return case.Solution(model,np.asarray(saved),times,probe_times,transverse,vorticity,row),work
