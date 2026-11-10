"""FP64 CPU model and tables for one-block fused angular advection."""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class FFTPlan:
    n: int
    radices: tuple
    permutation: np.ndarray
    roots: np.ndarray


def plan(n):
    if type(n) is not int or n < 2:
        raise ValueError("Require a positive integer FFT length greater than one")
    remaining = n
    radices = []
    for radix in (2, 3, 5):
        while remaining % radix == 0:
            radices.append(radix)
            remaining //= radix
    if remaining != 1:
        raise ValueError("The fused FFT supports factors 2, 3 and 5")
    digits = np.arange(n, dtype=np.int32)
    permutation = np.zeros_like(digits)
    stride = n
    for radix in radices:
        stride //= radix
        permutation += (digits % radix)*stride
        digits //= radix
    roots = np.exp((-2j*pi/n)*np.arange(n))
    roots[0] = 1
    if n % 2 == 0:
        roots[n//2] = -1
    permutation.flags.writeable = False
    roots.flags.writeable = False
    return FFTPlan(n, tuple(radices), permutation, roots)


def butterfly(values, radix, sign):
    """Small unnormalized DFT in the same pair-symmetric form as the CUDA design."""
    a, b = values[0], values[1]
    if radix == 2:
        return a+b, a-b
    c = values[2]
    if radix == 3:
        summed, differenced = b+c, b-c
        middle = a-.5*summed
        skew = (sign*1j*sqrt(3)/2)*differenced
        return a+summed, middle+skew, middle-skew
    d, e = values[3:]
    sum_one, sum_two = b+e, c+d
    diff_one, diff_two = b-e, c-d
    c1, c2 = cos(2*pi/5), cos(4*pi/5)
    s1, s2 = sin(2*pi/5), sin(4*pi/5)
    m1, m2 = a+c1*sum_one+c2*sum_two, a+c2*sum_one+c1*sum_two
    j1, j2 = sign*1j*(s1*diff_one+s2*diff_two), sign*1j*(s2*diff_one-s1*diff_two)
    return a+sum_one+sum_two, m1+j1, m2+j2, m2-j2, m1-j1


def transform(values, fft, inverse=False):
    values = np.asarray(values, dtype=np.complex128)
    if values.shape[-1] != fft.n:
        raise ValueError("FFT data and plan lengths differ")
    out = values[..., fft.permutation].copy()
    roots = fft.roots.conj() if inverse else fft.roots
    prefix = out.shape[:-1]
    stride = 1
    for radix in fft.radices:
        length = radix*stride
        twiddles = roots[np.arange(radix)[:,None]*np.arange(stride)*(fft.n//length)]
        groups = out.reshape(*prefix, fft.n//length, radix, stride)
        prepared = np.moveaxis(groups*twiddles, -2, 0)
        groups[...] = np.stack(butterfly(prepared,radix,1 if inverse else -1),axis=-2)
        stride = length
    return out


def hermitian(spectrum, n):
    """Expand independent real signals without conjugating the packed complex pair."""
    if spectrum.shape[-1] != n//2+1:
        raise ValueError("Wrong half spectrum width")
    positive = spectrum.copy()
    positive[...,0] = positive[...,0].real
    if n % 2 == 0:
        positive[...,-1] = positive[...,-1].real
    last = -1 if n % 2 == 0 else None
    return np.concatenate((positive,positive[...,1:last][...,::-1].conj()),axis=-1)


def inverse_pair(first, second, fft):
    z = transform(hermitian(first,fft.n)+1j*hermitian(second,fft.n),fft,inverse=True)
    return z.real, z.imag


def forward_pair(first, second, fft):
    z = transform(first+1j*second,fft)
    k = np.arange(fft.n//2+1)
    opposite = z[...,(-k)%fft.n].conj()
    return .5*(z[...,k]+opposite), -.5j*(z[...,k]-opposite)


def fd_mean(mean_terms):
    result = np.zeros(mean_terms.shape[:-1],dtype=np.float64)
    for mode in range(1,mean_terms.shape[-1]):
        result += mean_terms[...,mode]
    return result


def nonlinear(spectrum, old_flux, fft, k, mean_terms=None):
    """Match the existing pre-projection Galerkin or FD flux/history contract."""
    fd = mean_terms is not None
    inputs, outputs = (4,1) if fd else (3,2)
    if (fft.n <= 3*k or k < 1 or spectrum.ndim != 3 or
            spectrum.shape[0] != inputs or spectrum.shape[-1] != fft.n//2+1 or
            old_flux.shape != (outputs,*spectrum.shape[1:]) or
            (fd and mean_terms.shape != (spectrum.shape[1],k+1)) or
            np.any(spectrum[...,k+1:] != 0)):
        raise ValueError("Expected the registered dealiased field and history layout")
    ur, ut = inverse_pair(spectrum[0],spectrum[1],fft)
    if fd:
        ds, dtheta = inverse_pair(spectrum[2],spectrum[3],fft)
        product = transform(ur*ds+ut*dtheta,fft)[...,:fft.n//2+1].copy()
        product[...,0] = product[...,0].real
        if fft.n % 2 == 0:
            product[...,-1] = product[...,-1].real
        return product[None],old_flux.copy(),fd_mean(mean_terms)
    curl = transform(hermitian(spectrum[2],fft.n),fft,inverse=True).real
    return np.stack(forward_pair(ur*curl,ut*curl,fft)),old_flux.copy(),None


def cost(n, rows):
    fft=plan(n)
    return dict(ntheta=n,radial_rows=rows,radices=list(fft.radices),block_threads=128,
                shared_bytes_per_block=2*n*16,logical_field_transforms_per_row=5,
                internal_complex_transforms_per_row=3,block_barriers_per_nonlinear=2*len(fft.radices)+3,
                old_physical_buffer_global_bytes_per_nonlinear=80*n*rows,
                scope="Byte traffic for separate physical arrays. Cache effects and timing are unmeasured.")
