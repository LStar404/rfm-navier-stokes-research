"""Original wall-coordinate jets and features; pilot monkeypatch driver excluded."""
import math
from . import rfm_solver as s
np = s.np

def coordinate_jet(z, strength, kind='sinh'):
    """A monotone map [0,1] -> [0,1] and its first two derivatives."""
    if abs(strength) < 1e-14 or kind == 'identity':
        return z, np.ones_like(z), np.zeros_like(z)
    k = float(strength)
    if kind == 'sinh':
        t = k*(2*z-1)
        return (.5+np.sinh(t)/(2*np.sinh(k)),
                k*np.cosh(t)/np.sinh(k),
                2*k*k*np.sinh(t)/np.sinh(k))
    if kind == 'top':
        t = np.exp(k*z)
        denom = np.expm1(k)
        return np.expm1(k*z)/denom, k*t/denom, k*k*t/denom
    if kind == 'cubic':
        return (z+k*(z-.5)*z*(z-1),
                1+k*(3*z*z-3*z+.5), k*(6*z-3))
    raise ValueError(kind)


class WallFeatures(s.Features):
    """Global random features, optionally supplemented by wall-normal layers."""
    def __init__(self, problem, count, seed, activation='cos', patches=1, scale=14.,
                 warp='identity', strength=(0., 0.), wall_fraction=0.,
                 wall_beta=(.5,1.5), wall_scale=1., anisotropy=(1.,1.)):
        if patches != 1:
            raise ValueError('Wall-aware pilot uses one global trial domain')
        extra = int(round((count-1)*wall_fraction))
        super().__init__(problem, count-extra, seed, activation, patches, scale)
        self.count, self.warp = count, warp
        self.strength = tuple(strength)
        self.anisotropy = np.asarray(anisotropy)
        self.layers = []
        rng = np.random.default_rng(seed+7919)
        # Equal allocation to all four walls; corner values remain hard-clamped.
        for wall, indices in enumerate(np.array_split(np.arange(extra),4)):
            n = len(indices)
            k = rng.uniform(-3*scale,3*scale,n)*wall_scale
            beta = rng.uniform(*wall_beta,n)*math.sqrt(problem.reynolds)
            phase = rng.uniform(-np.pi,np.pi,n)
            self.layers.append((wall,k,beta,phase))

    def jet(self, xy):
        L = self.problem.length
        t = (xy-self.problem.lo)/L
        mapped = [coordinate_jet(z,k,self.warp) for z,k in zip(t.T,self.strength)]
        x,y = mapped
        out = []
        for w,b,center in self.blocks:
            w = w*self.anisotropy
            z = (np.column_stack((x[0],y[0]))-center)@w.T+b
            if self.activation == 'cos':
                f = np.cos(z); df = -np.sin(z); d2f = -f
            else:
                f = np.tanh(z); df = 1-f*f; d2f = -2*f*df
            ax,ay = x[1][:,None]*w[:,0]/L, y[1][:,None]*w[:,1]/L
            laparg = (x[2][:,None]*w[:,0]+y[2][:,None]*w[:,1])/(L*L)
            out.append(np.stack((f,df*ax,df*ay,d2f*(ax*ax+ay*ay)+df*laparg)))
        for wall,k,beta,phase in self.layers:
            if not len(k):
                continue
            normal_axis = 0 if wall < 2 else 1
            tangential_axis = 1-normal_axis
            side = wall % 2
            normal = t[:,normal_axis,None]
            tangent = t[:,tangential_axis,None]
            rate = beta*(1 if side else -1)
            weight = np.exp(beta*(normal-1) if side else -beta*normal)
            argument = tangent*k+phase
            value = weight*np.cos(argument)
            dnormal = rate*value/L
            dtangent = -weight*k*np.sin(argument)/L
            dx,dy = (dnormal,dtangent) if normal_axis == 0 else (dtangent,dnormal)
            out.append(np.stack((value,dx,dy,(beta*beta-k*k)*value/(L*L))))
        constant = np.zeros((4,len(xy),1));constant[0] = 1
        return np.concatenate((constant,*out),axis=-1)

    def clamped_derivatives(self, xy):
        raise NotImplementedError('WallFeatures currently supports analytic second jets for weak form only')
