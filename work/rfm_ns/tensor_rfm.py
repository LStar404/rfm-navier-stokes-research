"""Factor global weak projections into two one-dimensional contractions.

Run this file to reproduce the paired global RFM/Legendre timing experiment.
The dense implementation is kept as the numerical control.
"""
import hashlib
import json
from unittest.mock import patch

from experiments import ROOT, evaluate, np
import rfm_solver as s


class TensorProjection:
    def __init__(self, factors, inverse_norm):
        self.factors = factors
        self.inverse_norm = inverse_norm

    def __rmul__(self, scalar):
        return TensorProjection(self.factors, scalar*self.inverse_norm)

    def __matmul__(self, data):
        order, q = self.factors[0][0].shape
        count = data.size//(q*q)
        blocks = data.reshape(q,q*count)
        result = np.zeros((order,order,count))
        for left, right in self.factors:
            first = (left@blocks).reshape(order,q,count).transpose(0,2,1)
            second = first.reshape(order*count,q)@right.T
            result += second.reshape(order,count,order).transpose(0,2,1)
        result *= self.inverse_norm[:,:,None]
        return result.reshape((order*order,)+data.shape[1:])


def tensor_quadrature(problem, cells, order, q):
    if cells!=1:
        raise ValueError('Tensor benchmark uses a single global test domain')
    z, w = s.leggauss(q)
    z, w = (z+1)/2, w*(problem.length/2)
    bubble = s.polynomial_jet(s.Poly([0,0,1,-2,1]),z,problem.length)
    modes = []
    for degree in range(order):
        p = s.polynomial_jet(s.Legendre.basis(degree,domain=[0,1]),z,problem.length)
        b0,b1,b2 = bubble
        p0,p1,p2 = p
        modes.append((b0*p0,b1*p0+b0*p1,b2*p0+2*b1*p1+b0*p2))
    value, first, second = np.asarray(modes).transpose(1,0,2)
    v2 = (value**2)@w
    d2 = (second**2)@w
    vd = (value*second)@w
    invnorm = 1/np.sqrt(np.outer(d2,v2)+np.outer(v2,d2)+2*np.outer(vd,vd))
    v, d, h = value*w, first*w, second*w
    nodes = problem.lo+problem.length*np.stack(np.meshgrid(z,z,indexing='ij'),-1).reshape(-1,2)
    tx = TensorProjection(((d,v),),invnorm)
    ty = TensorProjection(((v,d),),invnorm)
    lap = TensorProjection(((h,v),(v,h)),invnorm)
    return nodes,(tx,ty,lap)


def operator_checks():
    rng = np.random.default_rng(211)
    rows = []
    for case, re, order, q in [('cavity',400,29,43),('kovasznay',40,17,25)]:
        p = s.Problem(case,re)
        dense_xy,dense = s.weak_quadrature(p,1,order,q)
        fast_xy,fast = tensor_quadrature(p,1,order,q)
        assert np.array_equal(dense_xy,fast_xy)
        for name,a,b in zip(('tx','ty','lap'),dense,fast):
            for data in (rng.normal(size=q*q),rng.normal(size=(q*q,13))):
                want = a@data
                actual = b@data
                error = np.linalg.norm(actual-want)/np.linalg.norm(want)
                if error>1e-12:
                    raise ArithmeticError(f'{name} projection disagrees with dense: {error}')
                rows.append(dict(case=case,order=order,q=q,operator=name,relative_difference=float(error),
                                 right_hand_sides=1 if data.ndim==1 else data.shape[1]))
    return rows


def main():
    configs = json.loads((ROOT/'frozen_cavity.json').read_text(encoding='utf8'))
    selected = {100:{153,325},400:{231,496}}
    configs = [c for c in configs if c['method']=='weak' and c['cells']==1 and not c['continuation']
               and c['m'] in selected[c['reynolds']]]
    raw = s.weak_quadrature
    rows = []
    checks = operator_checks()
    for i, seed in enumerate((101,211,307)):
        schedule = [(c,variant) for c in configs for variant in ('dense','tensor')]
        for j in np.random.default_rng(3107+i).permutation(len(schedule)):
            config,variant = schedule[j]
            projection = tensor_quadrature if variant=='tensor' else raw
            with patch.object(s,'weak_quadrature',projection):
                row = evaluate(config,seed)
            row['projection'] = variant
            rows.append(row)
    for config in configs:
        for seed in (101,211,307):
            pair = [r for r in rows if r['label']==config['label'] and r['seed']==seed
                    and r['reynolds']==config['reynolds']]
            for metric in ('velocity_rel_l2','vorticity_rel_l2'):
                if abs(pair[0][metric]-pair[1][metric])>1e-7:
                    raise ArithmeticError(f"{config['label']} at seed {seed} changed {metric}")
    data = dict(configs=configs,rows=rows,operator_checks=checks,
        timing='Same one-thread process and timing boundaries as experiments.py. Each run starts from scratch.',
        hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('rfm_solver.py','tensor_rfm.py','experiments.py')})
    (ROOT/'tensor_rfm_formal.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    main()
