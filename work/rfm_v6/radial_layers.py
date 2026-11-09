"""Analytic outer-layer trial functions and split Gauss quadrature."""
import full_case as case
from full_case import np


def layer_jets(s, rates, mean=False):
    """Return homogeneous boundary jets for localized exponentials."""
    d=(case.LENGTH-s)[:,None]
    z=(s/case.LENGTH)[:,None]
    beta=np.asarray(rates)[None,:]
    p=1 if mean else 2
    c=beta+1+1/case.LENGTH if mean else beta+2/case.LENGTH
    if mean:
        h,dh,ddh=1+c*d,c,0.
    else:
        h,dh,ddh=d+c*d*d,1+2*c*d,2*c
    exponential=np.exp(-beta*d)
    a=z**p
    da=p*z**(p-1)/case.LENGTH
    value=a*h*exponential
    first=(da*h+a*(beta*h-dh))*exponential
    if mean:
        return np.stack((value,first))
    dda=2/case.LENGTH**2
    second=(dda*h+2*da*(beta*h-dh)+a*(beta**2*h-2*beta*dh+ddh))*exponential
    return np.stack((value,first,second))


def radial_quadrature(q,split):
    if not split:
        x,w=case.leggauss(q)
        return (x+1)*case.LENGTH/2,w*case.LENGTH/2
    short=max(8,q//6)
    edges=(0.,case.LENGTH-.12,case.LENGTH-.018,case.LENGTH)
    nodes,weights=[],[]
    for left,right,count in zip(edges,edges[1:],(q-2*short,short,short)):
        x,w=case.leggauss(count)
        nodes.append(left+(x+1)*((right-left)/2))
        weights.append(w*((right-left)/2))
    return np.concatenate(nodes),np.concatenate(weights)
