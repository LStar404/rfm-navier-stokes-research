"""Exactly dealiased full-circle products for normalized real Fourier modes."""
import os

# Standalone kernel and its pilots use the same thread controls as the solvers.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
from scipy.fft import fft, irfft, next_fast_len, rfft


class FullNonlinear:
    """Coefficients are c_0 and c_k in f=c_0+2Re(sum c_k exp(i k θ)).

    The radial index is first. The plan owns work buffers and is not reentrant.
    Results own fresh transform outputs and survive subsequent evaluations.
    """

    def __init__(self, q, k, backend='fft'):
        if q < 1 or k < 1 or backend not in ('basic', 'fft', 'packed', 'dense'):
            raise ValueError('Require positive q and k and a supported backend')
        self.q, self.k, self.backend = q, k, backend
        # Products have degree 2k. The first wrapped frequency has magnitude
        # N-2k, which must be strictly greater than the retained cutoff k.
        self.n = next_fast_len(3*k+1, real=True)
        self.count = k+1
        self.ik = 1j*np.arange(self.count)
        self.work = {}
        if backend == 'dense':
            theta = (2*np.pi/self.n)*np.arange(self.n)
            phase = np.arange(self.count)[:, None]*theta
            cosine, sine = np.cos(phase), -np.sin(phase)
            self.synthesis = np.empty((2*self.count, self.n))
            self.synthesis[0::2], self.synthesis[1::2] = 2*cosine, 2*sine
            self.synthesis[0] = 1.
            self.projection = np.empty((self.n, 2*self.count))
            self.projection[:, 0::2], self.projection[:, 1::2] = cosine.T/self.n, sine.T/self.n

    def _workspace(self, count):
        if count not in self.work:
            width = self.count if self.backend == 'dense' else self.n//2+1
            self.work[count] = np.zeros((count, self.q, width), dtype=complex)
        return self.work[count]

    def _values(self, fields, differentiated=None):
        count = len(fields) + (differentiated is not None)
        if self.backend == 'basic':
            coefficients = list(fields)
            if differentiated is not None:
                coefficients.append(differentiated*self.ik)
            return irfft(np.stack(coefficients), n=self.n, norm='forward', workers=1)
        spectrum = self._workspace(count)
        for i, field in enumerate(fields):
            spectrum[i, :, :self.count] = field
        if differentiated is not None:
            np.multiply(differentiated, self.ik, out=spectrum[-1, :, :self.count])
        if self.backend == 'dense':
            return spectrum.view(float) @ self.synthesis
        return irfft(spectrum, n=self.n, norm='forward', workers=1)

    def _coefficients(self, values):
        if self.backend == 'dense':
            return (values @ self.projection).view(complex)
        return rfft(values, axis=-1, norm='forward', workers=1,
                    overwrite_x=True)[..., :self.count]

    def __call__(self, ur, ut, omega):
        values = self._values((ur,ut,omega))
        np.multiply(values[:2], values[2], out=values[:2])
        if self.backend == 'packed':
            # One complex FFT of P+iQ. Hermitian decomposition recovers the
            # separate real Fourier series from positive and negative modes.
            packed = fft(values[0]+1j*values[1], norm='forward', workers=1,
                         overwrite_x=True)
            reflected = np.conj(packed[:, (-np.arange(self.count)) % self.n])
            positive = packed[:, :self.count]
            return .5*(positive+reflected), -.5j*(positive-reflected)
        a, b = self._coefficients(values[:2])
        return a, b

    def advect(self, ur, ut, omega, omega_s, r):
        """Return −(ur ω_s + ut ω_θ)/r and −mean(ur ω).

        The mean tangential momentum source follows from the rotational
        velocity equation. Parseval obtains it without a fifth inverse FFT.
        """
        radial, angular, grad_s, grad_theta = self._values((ur,ut,omega_s), omega)
        np.multiply(radial, grad_s, out=radial)
        np.multiply(angular, grad_theta, out=angular)
        radial += angular
        advection = self._coefficients(radial)
        advection *= (-1/np.asarray(r).reshape(self.q, 1))
        mean = (ur[:, 0]*omega[:, 0].conj()).real
        mean += 2*np.einsum('qk,qk->q',ur[:,1:],omega[:,1:].conj()).real
        return advection, -mean
