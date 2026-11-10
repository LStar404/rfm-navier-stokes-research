"""Steady 2D spatial RFM with analytic derivatives and independent scoring.

Primitive, hard-wall mixed, and hard-wall weak formulations all freeze their
hidden features. Only output coefficients are solved. No reference fields
enter the nonlinear iteration.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import math
import time
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial as Poly, Legendre
from numpy.polynomial.legendre import leggauss
from scipy.linalg import lstsq, svd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Problem:
    case: str
    reynolds: float

    @property
    def lo(self): return -1. if self.case=='kovasznay' else 0.

    @property
    def length(self): return 2. if self.case=='kovasznay' else 1.

    @property
    def nu(self): return 1/self.reynolds

    @property
    def lam(self):
        return -(2*np.pi)**2/(.5*self.reynolds+np.hypot(.5*self.reynolds,2*np.pi))

    def exact(self,xy):
        if self.case!='kovasznay':
            raise ValueError('An exact field is available only for Kovasznay')
        x,y=xy.T
        lam,k=self.lam,2*np.pi
        a=np.exp(lam*x)
        s,c=np.sin(k*y),np.cos(k*y)
        return dict(u=1-a*c,v=lam/k*a*s,p=(1-a*a)/2,
                    psi=y-a*s/k,omega=(lam*lam-k*k)/k*a*s)

    def boundary_velocity(self,xy):
        if self.case=='kovasznay':
            f=self.exact(xy)
            return f['u'],f['v']
        x,y=xy.T
        return np.where(y==1,16*x*x*(1-x)**2,0.),np.zeros(len(x))


def polynomial_jet(poly,z,L):
    return np.stack([poly.deriv(i)(z)/L**i for i in range(3)])


def separable(f,g):
    return np.stack((f[0]*g[0],f[1]*g[0],f[0]*g[1],f[2]*g[0]+f[0]*g[2]))


def hermite(f0,d0,f1,d1,length):
    return (f0*Poly([1,0,-3,2])+length*d0*Poly([0,1,-2,1])
        +f1*Poly([0,0,3,-2])+length*d1*Poly([0,0,-1,1]))


def lift_and_envelope(problem,xy):
    x,y=xy.T
    L=problem.length
    X,Y=(xy-problem.lo).T/L
    clamped=Poly([0,0,1,-2,1])
    envelope=256*separable(polynomial_jet(clamped,X,L),polynomial_jet(clamped,Y,L))
    if problem.case=='cavity':
        lift=16*separable(polynomial_jet(clamped,X,L),polynomial_jet(Poly([0,0,-1,1]),Y,L))
    else:
        lam,k=problem.lam,2*np.pi
        am,ap=np.exp(-lam),np.exp(lam)
        hx=polynomial_jet(hermite(am,lam*am,ap,lam*ap,L),X,L)
        hy=polynomial_jet(hermite(0,k,0,k,L),Y,L)
        ax=np.exp(lam*x)*np.array([[1],[lam],[lam*lam]])
        sy=np.stack([np.sin(k*y),k*np.cos(k*y),-k*k*np.sin(k*y)])
        # Hermite transfinite extension from the two pairs of boundary jets.
        # exp(lambda*x) and sin(k*y) here are tangential boundary traces.
        lift=np.stack((y,np.zeros_like(x),np.ones_like(x),np.zeros_like(x)))
        lift-=(separable(hx,sy)+separable(ax-hx,hy))/k
    return lift,envelope


def multiply(jet,weight):
    F,Fx,Fy,lap=jet
    b,bx,by,blap=weight[:, :,None]
    return np.stack((b*F,bx*F+b*Fx,by*F+b*Fy,blap*F+2*bx*Fx+2*by*Fy+b*lap))


class Features:
    def __init__(self,problem,count,seed,activation='tanh',patches=1,scale=3.):
        self.problem=problem
        self.count=count
        self.activation=activation
        self.patches=patches
        rng=np.random.default_rng(seed)
        self.blocks=[]
        # One exact constant and frozen, independently sampled hidden units.
        counts=[len(ix) for ix in np.array_split(np.arange(count-1),patches*patches)]
        for i,n in enumerate(counts):
            w=rng.uniform(-scale,scale,(n,2))*(2 if patches==2 else 1)
            if activation=='cos':
                w*=3.
                b=rng.uniform(-np.pi,np.pi,n)
            else:
                b=rng.uniform(-scale,scale,n)
            center=np.array([.5,.5]) if patches==1 else np.array([.25+.5*(i//2),.25+.5*(i%2)])
            self.blocks.append((w,b,center))

    def jet(self,xy):
        L=self.problem.length
        xy=(xy-self.problem.lo)/L
        if self.patches==2:
            factors=[]
            for coord in xy.T:
                z=np.clip(2*(coord-.25),0,1)
                p=Poly([0,0,0,0,0,126,-420,540,-315,70])
                s=polynomial_jet(p,z,L/2)
                factors.append((np.stack((1-s[0],-s[1],-s[2])),s))
        out=[]
        for i,(w,b,center) in enumerate(self.blocks):
            z=(xy-center)@w.T+b
            if self.activation=='cos':
                f=np.cos(z);df=-np.sin(z);d2f=-f
            else:
                f=np.tanh(z);df=1-f*f;d2f=-2*f*df
            wx,wy=(w/L).T
            local=np.stack((f,df*wx,df*wy,d2f*(wx*wx+wy*wy)))
            if self.patches==2:
                local=multiply(local,separable(factors[0][i//2],factors[1][i%2]))
            out.append(local)
        constant=np.zeros((4,len(xy),1));constant[0]=1
        return np.concatenate([constant,*out],axis=-1)

    def clamped_derivatives(self,xy):
        """Spatial derivatives of B*feature up to total order four."""
        L=self.problem.length
        t=(xy-self.problem.lo)/L
        envelope=Poly([0,0,1,-2,1])
        base=[np.stack([envelope.deriv(i)(x)/L**i for i in range(5)]) for x in t.T]
        factors=[]
        if self.patches==2:
            p=Poly([0,0,0,0,0,126,-420,540,-315,70])
            for coord,b in zip(t.T,base):
                z=np.clip(2*(coord-.25),0,1)
                s=np.stack([p.deriv(i)(z)/(L/2)**i for i in range(5)])
                right=np.stack([sum(math.comb(i,j)*b[j]*s[i-j] for j in range(i+1)) for i in range(5)])
                factors.append((b-right,right))
        orders=[(i,j) for i in range(5) for j in range(5-i)]
        out={ij:[(256*base[0][ij[0]]*base[1][ij[1]])[:,None]] for ij in orders}
        for pidx,(w,b,center) in enumerate(self.blocks):
            z=(t-center)@w.T+b
            if self.activation=='cos':
                diffs=[np.cos(z+i*np.pi/2) for i in range(5)]
            else:
                f=np.tanh(z)
                d=1-f*f
                diffs=[f,d,-2*f*d,(6*f*f-2)*d,(16*f-24*f**3)*d]
            px,py=(factors[0][pidx//2],factors[1][pidx%2]) if self.patches==2 else base
            wx,wy=(w/L).T
            for i,j in orders:
                value=0.
                for a in range(i+1):
                    for b in range(j+1):
                        value+=(256*math.comb(i,a)*math.comb(j,b)*(px[a]*py[b])[:,None]
                                *diffs[i+j-a-b]*(wx**(i-a)*wy**(j-b)))
                out[i,j].append(value)
        return {ij:np.concatenate(parts,axis=1) for ij,parts in out.items()}


class LegendreFeatures:
    """Deterministic control with a complete total-degree tensor basis."""
    def __init__(self, problem, count):
        degree = (math.isqrt(8*count+1)-3)//2
        self.orders = [(i,j) for i in range(degree+1) for j in range(degree+1-i)]
        if len(self.orders) != count:
            raise ValueError('Legendre control requires a complete total-degree basis')
        self.problem, self.count = problem, count
        self.polys = [Legendre.basis(i, domain=[0,1]) for i in range(degree+1)]
        self.ix, self.iy = np.asarray(self.orders).T

    def jets_1d(self, xy, order, clamped=False):
        length = self.problem.length
        coords = (xy-self.problem.lo)/length
        result = []
        bubble = Poly([0,0,1,-2,1])
        for z in coords.T:
            p = np.asarray([np.column_stack([f.deriv(k)(z)/length**k for f in self.polys])
                            for k in range(order+1)])
            if clamped:
                b = [bubble.deriv(k)(z)[:,None]/length**k for k in range(order+1)]
                p = np.asarray([sum(math.comb(k,j)*b[j]*p[k-j] for j in range(k+1))
                                for k in range(order+1)])
            result.append(p)
        return result[0][:,:,self.ix], result[1][:,:,self.iy]

    def jet(self, xy):
        return separable(*self.jets_1d(xy, 2))

    def clamped_derivatives(self, xy):
        x, y = self.jets_1d(xy, 4, clamped=True)
        return {(i,j):256*x[i]*y[j] for i in range(5) for j in range(5-i)}


def interior_boundary(problem,n):
    # Midpoints for PDE residual; extra independent points are used for scoring.
    lo,L=problem.lo,problem.length
    x=lo+L*(np.arange(n)+.5)/n
    pts=np.stack(np.meshgrid(x,x,indexing='ij'),-1).reshape(-1,2)
    z=np.linspace(lo,lo+L,n+2)[1:-1]
    edge=np.concatenate([np.column_stack((np.full(n,t),z)) for t in (lo,lo+L)]
        +[np.column_stack((z,np.full(n,t))) for t in (lo,lo+L)])
    return pts,edge


def scaled_lstsq(A,b,scaled=True,cond=1e-12):
    s=np.linalg.norm(A,axis=0) if scaled else np.ones(A.shape[1])
    s=np.maximum(s,max(s.max()*1e-15,1e-30))
    z,_,rank,_=lstsq(A/s,b,cond=cond,lapack_driver='gelsy',check_finite=False)
    return z/s,int(rank)


class Primitive:
    def __init__(self,problem,basis,n):
        self.problem,self.basis=problem,basis
        x,e=interior_boundary(problem,n)
        self.f,self.fx,self.fy,self.lap=basis.jet(x)
        self.b=basis.jet(e)[0]
        self.bu,self.bv=problem.boundary_velocity(e)
        self.m=basis.count
        self.size=3*self.m
        self.wp=1/np.sqrt(len(x))
        self.wb=4/np.sqrt(len(e))
        self.bound_jac=np.block([[self.b,np.zeros_like(self.b),np.zeros_like(self.b)],
            [np.zeros_like(self.b),self.b,np.zeros_like(self.b)]])*self.wb
        self.div_jac=np.column_stack((self.fx,self.fy,np.zeros_like(self.f)))*self.wp
        self.gauge=np.r_[np.zeros(2*self.m),self.f.mean(axis=0)][None,:]

    def evaluate(self,c,alpha,jac=False,picard=False):
        cu,cv,cp=c.reshape(3,self.m)
        F,Fx,Fy,L=self.f,self.fx,self.fy,self.lap
        u,v=F@cu,F@cv
        ux,uy,vx,vy=Fx@cu,Fy@cu,Fx@cv,Fy@cv
        r1=alpha*(u*ux+v*uy)+Fx@cp-self.problem.nu*(L@cu)
        r2=alpha*(u*vx+v*vy)+Fy@cp-self.problem.nu*(L@cv)
        r=np.r_[self.wp*r1,self.wp*r2,self.wp*(ux+vy),self.wb*(self.b@cu-self.bu),
                self.wb*(self.b@cv-self.bv),self.gauge@c]
        if not jac: return r
        z=np.zeros_like(F)
        D=alpha*(u[:,None]*Fx+v[:,None]*Fy)-self.problem.nu*L
        if picard:
            j1,j2=np.column_stack((D,z,Fx)),np.column_stack((z,D,Fy))
        else:
            j1=np.column_stack((D+alpha*ux[:,None]*F,alpha*uy[:,None]*F,Fx))
            j2=np.column_stack((alpha*vx[:,None]*F,D+alpha*vy[:,None]*F,Fy))
        return r,np.vstack((self.wp*j1,self.wp*j2,self.div_jac,self.bound_jac,self.gauge))

    def fields(self,c,xy):
        cu,cv,cp=c.reshape(3,self.m)
        F,X,Y,_=self.basis.jet(xy)
        return dict(u=F@cu,v=F@cv,p=F@cp,omega=X@cv-Y@cu,div=X@cu+Y@cv)


class HardMixed:
    def __init__(self,problem,basis,n):
        self.problem,self.basis=problem,basis
        x,e=interior_boundary(problem,n)
        self.m=basis.count
        self.size=2*self.m
        self.f,self.fx,self.fy,self.lap=basis.jet(x)
        self.lift,b=lift_and_envelope(problem,x)
        self.psi=multiply(np.stack((self.f,self.fx,self.fy,self.lap)),b)
        # Include Poisson compatibility on the wall, without imposing ω=0.
        self.wall_lift,be=lift_and_envelope(problem,e)
        self.wall_w=basis.jet(e)[0]
        self.wall_psi=multiply(basis.jet(e),be)
        self.w=1/np.sqrt(len(x))
        self.wb=.35/np.sqrt(len(e))
        self.compat=np.vstack((np.column_stack((-self.psi[3],-self.f))*self.w,
            np.column_stack((-self.wall_psi[3],-self.wall_w))*self.wb))

    def evaluate(self,c,alpha,jac=False,picard=False):
        a,b=c.reshape(2,self.m)
        u=self.lift[2]+self.psi[2]@a
        v=-self.lift[1]-self.psi[1]@a
        wx,wy=self.fx@b,self.fy@b
        compat=self.compat@c-np.r_[self.w*self.lift[3],self.wb*self.wall_lift[3]]
        r=np.r_[compat,self.w*(alpha*(u*wx+v*wy)-self.problem.nu*(self.lap@b))]
        if not jac: return r
        left=alpha*(wx[:,None]*self.psi[2]-wy[:,None]*self.psi[1])
        right=alpha*(u[:,None]*self.fx+v[:,None]*self.fy)-self.problem.nu*self.lap
        return r,np.vstack((self.compat,self.w*np.column_stack((left,right))))

    def fields(self,c,xy):
        a,b=c.reshape(2,self.m)
        F=self.basis.jet(xy)
        lift,envelope=lift_and_envelope(self.problem,xy)
        psi=lift+multiply(F,envelope)@a
        return dict(u=psi[2],v=-psi[1],psi=psi[0],omega=-psi[3],omega_independent=F[0]@b,
                    div=np.zeros(len(xy)))


def weak_quadrature(problem,cells,order,q):
    z,w=leggauss(q)
    z,w=(z+1)/2,w/2
    h=problem.length/cells
    points=[]
    for i in range(cells):
        for j in range(cells):
            points.append(problem.lo+h*np.stack(np.meshgrid(i+z,j+z,indexing='ij'),-1).reshape(-1,2))
    local=[]
    bubble=Poly([0,0,1,-2,1])
    b=polynomial_jet(bubble,z,h)
    modes=[]
    for degree in range(order):
        # Keep the Legendre representation when evaluating high-degree modes.
        # Expanding the shifted modes into powers of z causes cancellation.
        p=polynomial_jet(Legendre.basis(degree,domain=[0,1]),z,h)
        modes.append(np.stack((b[0]*p[0],b[1]*p[0]+b[0]*p[1],
            b[2]*p[0]+2*b[1]*p[1]+b[0]*p[2])))
    weight=h*h*w[:,None]*w[None,:]
    for bx in modes:
        for by in modes:
            t=separable(bx[:,:,None],by[:,None,:])
            denom=np.sqrt(np.sum(weight*t[3]**2))
            local.append((t[1:]*weight/denom).reshape(3,-1))
    local=np.asarray(local)
    m=order*order*cells*cells
    cols=(np.arange(cells*cells)[:,None,None]*q*q+np.arange(q*q)[None,None,:])
    cols=np.broadcast_to(cols,(cells*cells,order*order,q*q)).ravel()
    rows=np.repeat(np.arange(m),q*q)
    matrices=tuple(csr_matrix((np.tile(local[:,k,:].ravel(),cells*cells),(rows,cols)),
        shape=(m,cells*cells*q*q)) for k in range(3))
    return np.concatenate(points),tuple(a.toarray() for a in matrices) if cells==1 else matrices


class HardWeak:
    def __init__(self,problem,basis,cells,order=3,q=9):
        self.problem,self.basis=problem,basis
        self.size=basis.count
        xy,(self.tx,self.ty,lap)=weak_quadrature(problem,cells,order,q)
        self.lift,envelope=lift_and_envelope(problem,xy)
        self.psi=multiply(basis.jet(xy),envelope)
        self.U,self.V,self.W=self.psi[2],-self.psi[1],-self.psi[3]
        self.nu_lap=problem.nu*lap
        self.linear=self.nu_lap@self.W

    def whiten_stokes(self,threshold):
        """Filter weak Stokes directions once using training operators only."""
        scales=np.linalg.norm(self.linear,axis=0)
        scales=np.maximum(scales,scales.max()*1e-15)
        _,s,vh=svd(self.linear/scales,full_matrices=False,check_finite=False)
        keep=s>s[0]*threshold
        self.transform=vh[keep].T/scales[:,None]/s[keep]
        self.size=int(np.count_nonzero(keep))
        self.psi=self.psi@self.transform
        self.U,self.V,self.W=self.psi[2],-self.psi[1],-self.psi[3]
        self.linear=self.nu_lap@self.W
        return dict(relative_cutoff=threshold,retained=self.size,raw_count=len(scales),
            retained_scaled_singular_ratio=float(s[keep][-1]/s[0]),
            full_scaled_singular_ratio=float(s[-1]/s[0]))

    def evaluate(self,c,alpha,jac=False,picard=False):
        psi=self.lift+self.psi@c
        u,v,omega=psi[2],-psi[1],-psi[3]
        r=self.nu_lap@omega+alpha*(self.tx@(omega*u)+self.ty@(omega*v))
        if not jac: return r
        J=self.linear.copy()
        J+=alpha*(self.tx@(u[:,None]*self.W+omega[:,None]*self.U)
                  +self.ty@(v[:,None]*self.W+omega[:,None]*self.V))
        return r,J

    def fields(self,c,xy):
        lift,b=lift_and_envelope(self.problem,xy)
        if hasattr(self,'transform'):
            c=self.transform@c
        psi=lift+multiply(self.basis.jet(xy),b)@c
        return dict(psi=psi[0],u=psi[2],v=-psi[1],omega=-psi[3],div=np.zeros(len(xy)))


def lifted_stream_derivatives(problem,xy):
    """Use boundary data only, with the same lift as lift_and_envelope."""
    x,y=xy.T
    L=problem.length
    tx,ty=(xy-problem.lo).T/L
    if problem.case=='cavity':
        f,g=Poly([0,0,1,-2,1]),Poly([0,0,-1,1])
        return {(i,j):16*f.deriv(i)(tx)*g.deriv(j)(ty)/L**(i+j) for i in range(5) for j in range(5-i)}
    lam,k=problem.lam,2*np.pi
    hx=hermite(np.exp(-lam),lam*np.exp(-lam),np.exp(lam),lam*np.exp(lam),L)
    hy=hermite(0,k,0,k,L)
    out={}
    for i in range(5):
        h=hx.deriv(i)(tx)/L**i
        for j in range(5-i):
            t=h*k**j*np.sin(k*y+j*np.pi/2)+(lam**i*np.exp(lam*x)-h)*hy.deriv(j)(ty)/L**j
            out[i,j]=-t/k+(y if (i,j)==(0,0) else 1. if (i,j)==(0,1) else 0.)
    return out


def transport_components(j):
    return np.stack((j[0,1],-j[1,0],-j[3,0]-j[1,2],-j[2,1]-j[0,3],
        -j[4,0]-2*j[2,2]-j[0,4]))


class HardStrong:
    """Pressure-free collocation of the exact curl, using analytic fourth jets."""
    def __init__(self,problem,basis,n):
        self.problem,self.basis=problem,basis
        self.size=basis.count
        x,_=interior_boundary(problem,n)
        self.lift=transport_components(lifted_stream_derivatives(problem,x))
        self.basis_ops=transport_components(basis.clamped_derivatives(x))
        self.scale=1/np.sqrt(len(x))

    def evaluate(self,c,alpha,jac=False,picard=False):
        u,v,wx,wy,lap=self.lift+self.basis_ops@c
        r=self.scale*(alpha*(u*wx+v*wy)-self.problem.nu*lap)
        if not jac: return r
        U,V,X,Y,L=self.basis_ops
        J=alpha*(u[:,None]*X+v[:,None]*Y+wx[:,None]*U+wy[:,None]*V)-self.problem.nu*L
        return r,self.scale*J

    fields=HardWeak.fields


def solve(problem,method,m=120,n=31,seed=17,activation='tanh',patches=1,scale=3.,
          scaled=True,maxiter=12,continuation=True,cells=None,q=9,cond=1e-12,order=3,whiten=None):
    begun=time.perf_counter()
    basis=LegendreFeatures(problem,m) if activation=='legendre' else Features(problem,m,seed,activation,patches,scale)
    if method in ('picard','primitive'):
        op=Primitive(problem,basis,n)
    elif method=='mixed':
        op=HardMixed(problem,basis,n)
    elif method=='weak':
        cells=cells or math.ceil(math.sqrt(1.8*m/9))
        op=HardWeak(problem,basis,cells,q=q,order=order)
    elif method=='strong':
        op=HardStrong(problem,basis,n)
    else: raise ValueError(method)
    if whiten is not None and method!='weak':
        raise ValueError('Stokes whitening is implemented for the weak formulation')
    whitening=op.whiten_stokes(whiten) if whiten is not None else None
    setup=time.perf_counter()-begun
    c=np.zeros(op.size)
    r,J=op.evaluate(c,0.,True)
    c,rank=scaled_lstsq(J,-r,scaled,cond)
    history=[]
    for alpha in ((.25,.5,.75,1.) if continuation else (1.,)):
        for it in range(maxiter):
            r,J=op.evaluate(c,alpha,True,picard=method=='picard')
            old=np.linalg.norm(r)
            d,rank=scaled_lstsq(J,-r,scaled,cond)
            rate=1.
            for trial in range(12):
                newc=c+rate*d
                rn=np.linalg.norm(op.evaluate(newc,alpha))
                if np.isfinite(rn) and rn<=old: break
                rate*=.5
            else:
                history.append(dict(alpha=alpha,it=it,residual=float(old),step=0.,rank=rank))
                break
            c=newc
            history.append(dict(alpha=alpha,it=it,residual=float(rn),step=rate,rank=rank))
            if old-rn<=max(1e-12,1e-7*old): break
    elapsed=time.perf_counter()-begun
    return op,c,dict(case=problem.case,reynolds=problem.reynolds,method=method,m=m,n=n,seed=seed,
        activation=activation,patches=patches,scale=scale,cells=cells,q=q,order=order,unknowns=op.size,setup_seconds=setup,
        seconds=elapsed,scaled=scaled,cond=cond,continuation=continuation,maxiter=maxiter,history=history,whitening=whitening,
        nonlinear_steps=len(history),final_residual=float(np.linalg.norm(op.evaluate(c,1.))),rank=rank)


def field_errors(pred,ref):
    vel=np.sqrt((np.sum((pred['u']-ref['u'])**2)+np.sum((pred['v']-ref['v'])**2))
        /(np.sum(ref['u']**2)+np.sum(ref['v']**2)))
    vort=np.linalg.norm(pred['omega']-ref['omega'])/np.linalg.norm(ref['omega'])
    return dict(velocity_rel_l2=float(vel),vorticity_rel_l2=float(vort))
