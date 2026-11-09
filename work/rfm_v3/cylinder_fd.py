"""Fourth-order radial differences, Fourier angle, SBDF2 cylinder solver."""
import cylinder_case as case
from cylinder_case import np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.interpolate import make_interp_spline
from math import factorial


def difference_matrix(nodes,order):
    """Five-node stencils (six at the first and last second-derivative rows)."""
    n=len(nodes)-1
    rows,cols,values=[],[],[]
    for i in range(1,n):
        if order==2 and i in (1,n-1):
            ix=np.arange(6) if i==1 else np.arange(n-5,n+1)
        else:
            ix=np.arange(i-2,i+3) if 2<=i<=n-2 else (np.arange(5) if i==1 else np.arange(n-4,n+1))
        width=np.max(abs(nodes[ix]-nodes[i]))
        z=(nodes[ix]-nodes[i])/width
        b=np.zeros(len(ix)); b[order]=factorial(order)/width**order
        weights=np.linalg.solve(z**np.arange(len(ix))[:,None],b)
        rows.extend([i-1]*len(ix));cols.extend(ix);values.extend(weights)
    return sparse.csr_matrix((values,(rows,cols)),shape=(n-1,n+1))


def wall_second(nodes):
    """Fourth-order Hermite closure from wall value/slope and four neighbors."""
    n=len(nodes)-1
    rows,cols,vals,slopes=[],[],[],[]
    for side,ix in enumerate((np.arange(5),np.arange(n,n-5,-1))):
        width=abs(nodes[ix[1]]-nodes[ix[0]])
        powers=np.arange(6)
        z=(nodes[ix]-nodes[ix[0]])/width
        mat=np.column_stack((z**powers[:,None],powers==1))
        target=np.zeros(6);target[2]=2/width**2
        weights=np.linalg.solve(mat,target)
        rows.extend([side]*5);cols.extend(ix);vals.extend(weights[:5]);slopes.append(weights[-1]*width)
    return sparse.csr_matrix((vals,(rows,cols)),shape=(2,n+1)),np.asarray(slopes)


class FiniteDifference:
    def __init__(self,dt,n,k,stretch=0.,grid='inner',outer='farfield',linear='sparse'):
        if n<8 or k<1:
            raise ValueError('FD4 needs n>=8 and at least one sine mode')
        t=np.linspace(0,1,n+1)
        if grid=='both':
            t=(1-stretch)*t+stretch*.5*(1-np.cos(np.pi*t))
        elif stretch and grid=='inner':
            t=-np.log1p(t*np.expm1(-stretch))/stretch
        elif grid!='inner':
            raise ValueError(grid)
        t[-1]=1.
        self.s=case.LENGTH*t
        self.k=np.arange(1,k+1)
        self.r=case.radius(self.s)
        self.ntheta=case.angular_size(k)
        self.dt=dt
        if outer not in ('farfield','dirichlet'):
            raise ValueError(outer)
        self.outer=outer
        if linear not in ('sparse','batch'):
            raise ValueError(linear)
        self.linear=linear
        self.rhs_shape=(k,n-1)
        self.metadata=dict(method='fd4',n=n,k=k,stretch=stretch,grid=grid,outer=outer,ntheta=self.ntheta,linear=linear)
        self.d1=difference_matrix(self.s,1)
        self.d2=difference_matrix(self.s,2)
        self.wall_d2,slope=wall_second(self.s)
        self.wall_slope=np.array([0.,slope[1]*case.R if outer=='dirichlet' else 0.])
        if outer=='farfield':
            nodes=self.s[-5:]-self.s[-1]
            width=abs(nodes[0]);target=np.array([0.,1/width,0.,0.,0.])
            self.outer_d1=np.linalg.solve((nodes/width)**np.arange(5)[:,None],target)
        eye=sparse.eye(n+1,format='csr')
        interior=eye[1:-1,:]
        ends=eye[[0,n],:]
        p=interior.T
        inv=sparse.diags(1/self.r[1:-1]**2)
        inverse=1/self.r[[0,-1]]**2
        if outer=='farfield':
            inverse[1]=0.
        inv_wall=sparse.diags(inverse)
        lift_psi=np.zeros(n+1); lift_psi[-1]=case.R
        self.bdf=[]
        for j in self.k:
            lap=self.d2-j*j*interior
            w_int=-inv@lap
            lap_wall=self.wall_d2-j*j*ends
            w_wall=-inv_wall@lap_wall
            omega_all=sparse.vstack(((w_wall@p)[0,:],w_int@p,(w_wall@p)[1,:]),format='csr')
            mass=omega_all[1:-1,:]
            diff=case.NU*inv@lap@omega_all
            self.bdf.append(splu((1.5*mass-dt*diff).tocsc(),permc_spec='NATURAL'))
            if j==1:
                wall_lift=w_wall@lift_psi-inv_wall@self.wall_slope
                lift_omega=np.r_[wall_lift[0],w_int@lift_psi,wall_lift[1]]
                diff_lift=case.NU*(inv@lap@lift_omega)
                self.bdf_lift=1.5*lift_omega[1:-1]-dt*diff_lift
                self.start_lift=lift_omega[1:-1]-dt*diff_lift
                self.start=splu((mass-dt*diff).tocsc(),permc_spec='NATURAL')
        if linear=='batch':
            identity=np.eye(n-1)
            self.inverse=np.stack([a.solve(identity) for a in self.bdf])

    def advance(self,rhs,speed,first):
        result=np.empty_like(rhs)
        if first:
            result.fill(0.)
            result[0]=self.start.solve(rhs[0]-speed*self.start_lift)
        elif self.linear=='batch':
            rhs[0]-=speed*self.bdf_lift
            result=np.einsum('kij,kj->ki',self.inverse,rhs)
        else:
            result[0]=self.bdf[0].solve(rhs[0]-speed*self.bdf_lift)
            for j in range(1,len(result)):
                result[j]=self.bdf[j].solve(rhs[j])
        return result

    def native(self,state,speed):
        psi=np.zeros((len(self.s),len(self.k)))
        psi[1:-1]=state.T
        psi[-1,0]=case.R*speed
        derivative=np.zeros_like(psi)
        derivative[1:-1]=self.d1@psi
        if self.outer=='farfield':
            derivative[-1]=self.outer_d1@psi[-5:]
        else:
            derivative[-1,0]=case.R*speed
        omega=np.empty_like(psi)
        omega[1:-1]=-(self.d2@psi-psi[1:-1]*self.k**2)/self.r[1:-1,None]**2
        wall=self.wall_d2@psi-psi[[0,-1]]*self.k**2
        wall[:,0]+=self.wall_slope*speed
        omega[[0,-1]]=-wall/self.r[[0,-1],None]**2
        if self.outer=='farfield':
            omega[-1]=0.
        return psi,derivative,omega

    def terms(self,state,speed):
        psi,derivative,omega=self.native(state,speed)
        p=case.cosine_values(psi[1:-1]*self.k,self.ntheta)
        q=case.sine_values(derivative[1:-1],self.ntheta)
        wr=case.sine_values(self.d1@omega,self.ntheta)
        wa=case.cosine_values(omega[1:-1]*self.k,self.ntheta)
        forcing=case.sine_coeff((q*wa-p*wr)/self.r[1:-1,None]**2,len(self.k))
        return omega[1:-1].T.copy(),forcing.T

    def modal(self,state,speed,s):
        return tuple(make_interp_spline(self.s,x,k=5,axis=0)(s) for x in self.native(state,speed))
