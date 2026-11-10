"""Scalar consistency and sampled two-step stability for late nonlinear subsampling."""

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).with_suffix(".json")


def propagation(z_implicit, z_explicit, phase):
    a = 1.5-z_implicit
    if phase == "ordinary":
        first = [(2+2*z_explicit)/a,(-.5-z_explicit)/a]
    elif phase == "sampled":
        first = [(2+1.5*z_explicit)/a,-.5/a,-.5*z_explicit/a,0]
    else:
        first = [2/a,(-.5+2*z_explicit)/a,0,-z_explicit/a]
    matrix = np.eye(len(first),k=-1,dtype=complex)
    matrix[0] = first
    return matrix


def two_step(z_implicit, z_explicit, stride):
    if stride == 1:
        ordinary = propagation(z_implicit,z_explicit,"ordinary")
        return ordinary@ordinary
    return propagation(z_implicit,z_explicit,"skipped")@propagation(z_implicit,z_explicit,"sampled")


def evolve(rate, implicit, explicit, stride):
    h = 1/rate
    length = 2 if stride == 1 else 4
    state = np.exp(-(implicit+explicit)*h*np.arange(length))
    pair = two_step(h*implicit,h*explicit,stride)
    for _ in range(rate//2):
        state = pair@state
    return float(abs(state[0]-np.exp(implicit+explicit)))


def radius(alpha,beta,stride):
    return float(max(abs(np.linalg.eigvals(two_step(-alpha,1j*beta,stride)))))


def stability_band(alpha,stride):
    betas = np.linspace(0,1.25,2501)
    radii = np.array([radius(alpha,b,stride) for b in betas])
    exceeds = np.flatnonzero(radii>1+1e-12)
    return dict(alpha=alpha,first_sampled_unstable_beta=float(betas[exceeds[0]]) if len(exceeds) else None,
                sample_spacing=float(betas[1]),sampled_radii={str(beta):radius(alpha,beta,stride)
                    for beta in (0,.025,.05,.1,.2,.3,.4,.5,.75,1.)})


def run():
    assert np.allclose(np.sort(abs(np.linalg.eigvals(two_step(0,0,2)))), [0,0,1/9,1])
    rates = [32,64,128,256,512]
    examples = []
    for implicit,explicit in ((0j,1j),(-.3+0j,1j),(-3+0j,3j)):
        values = {}
        for stride in (1,2):
            error = [evolve(r,implicit,explicit,stride) for r in rates]
            order = [float(np.log2(a/b)) for a,b in zip(error,error[1:])]
            assert 1.9 < order[-1] < 2.1
            values[str(stride)] = dict(absolute_error=error,successive_order=order)
        examples.append(dict(implicit_real=implicit.real,explicit_imaginary=explicit.imag,rates=rates,by_stride=values))
    result = dict(status="passed",scope="Scalar split test equation, exact past values, sampled spectral radii; no NS stability claim",
        formula={"ordinary":[2,-1],"sampled_even":[1.5,-.5],"skipped_odd":[2,-1]},
        explicit_extrapolation_remainder_coefficient=[1,1.5,4],
        stability={str(s):[stability_band(a,s) for a in (0,.001,.01,.05,.1,.2,.5,1.,2.,4.)] for s in (1,2)},
        consistency=examples)
    with OUT.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(OUT)


if __name__ == "__main__":
    run()
