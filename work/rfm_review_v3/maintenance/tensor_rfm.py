"""Original tensor projection arithmetic; benchmark drivers are excluded."""
from . import rfm_solver as s
np = s.np

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
