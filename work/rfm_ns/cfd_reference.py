"""Independent steady cavity CFD reference: sparse finite differences/Newton.

Domain [0,1]^2, lid u=16*x^2*(1-x)^2, v=0, other walls no-slip.
psi_y=u, -psi_x=v, omega=v_x-u_y=-Delta psi.  Second-order
central interiors and a second-order three-point wall-vorticity closure.
This code does not use or import the RFM implementation.
"""
import os
for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_var]='1'
import json,time,argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parent
SOURCES={
 'regularized_cavity':{'title':'Martinez and Esperanca (2007), A Chebyshev collocation spectral method for numerical simulation of incompressible flow problems',
 'url':'https://www.scielo.br/j/jbsmse/a/3tFP6DFZwdN9fZkGLFkFMZR/?lang=en',
 'doi':'10.1590/S1678-58782007000300013','published_top_wall_max_abs_vorticity_on_201_points':{'100':13.4444,'400':24.9109}},
 'kovasznay':{'title':'Lin, Yang and Dong (2019), Numerical approximation of incompressible Navier-Stokes equations based on an auxiliary energy variable, JCP388 pp1-22, eq46',
 'url':'https://www.math.purdue.edu/~sdong/PDF/NSE_ES_JCP2019.pdf'},
}

def lid(x):return 16*np.asarray(x)**2*(1-np.asarray(x))**2

def kovasznay(x,y,Re=40.):
    """Exact steady unforced NS, usable on any rectangle with exact Dirichlet data."""
    x,y=np.broadcast_arrays(x,y);lam=Re/2-np.sqrt(Re**2/4+4*np.pi**2)
    ex=np.exp(lam*x);s=np.sin(2*np.pi*y);c=np.cos(2*np.pi*y)
    u=1-ex*c;v=lam/(2*np.pi)*ex*s;p=(1-np.exp(2*lam*x))/2
    psi=y-ex*s/(2*np.pi)
    omega=(lam**2/(2*np.pi)-2*np.pi)*ex*s
    return dict(u=u,v=v,p=p,psi=psi,omega=omega)

class CavityFD:
    def __init__(self,n,Re):
        self.n=int(n);self.m=n-1;self.d=self.m**2;self.h=1/n;self.Re=float(Re);self.nu=1/self.Re
        m=self.m;h=self.h;eye=sparse.eye(m,format='csr')
        dd=sparse.diags([np.ones(m-1),-2*np.ones(m),np.ones(m-1)],[-1,0,1],format='csr')/h**2
        d1=sparse.diags([-np.ones(m-1),np.ones(m-1)],[-1,1],format='csr')/(2*h)
        self.L=sparse.kron(dd,eye,format='csr')+sparse.kron(eye,dd,format='csr')
        self.Dx=sparse.kron(d1,eye,format='csr');self.Dy=sparse.kron(eye,d1,format='csr')
        self.I=sparse.eye(self.d,format='csr')
        entries={name:[[],[],[]] for name in ['lap','x','y']}
        const={name:np.zeros(self.d) for name in entries}
        def ind(i,j):return (i-1)*m+j-1
        def add(row,near,far,bc,lf,xf,yf):
            for name,fac in [('lap',lf),('x',xf),('y',yf)]:
                if fac:
                    rr,cc,vv=entries[name]
                    rr.extend([row,row]);cc.extend([ind(*near),ind(*far)])
                    vv.extend([-4*fac/h**2,.5*fac/h**2]);const[name][row]+=fac*bc
        for j in range(1,n):
            add(ind(1,j),(1,j),(2,j),0,1/h**2,-1/(2*h),0)
            add(ind(n-1,j),(n-1,j),(n-2,j),0,1/h**2,1/(2*h),0)
        for i in range(1,n):
            add(ind(i,1),(i,1),(i,2),0,1/h**2,0,-1/(2*h))
            add(ind(i,n-1),(i,n-1),(i,n-2),-3*lid(i*h)/h,1/h**2,0,1/(2*h))
        mats={name:sparse.csr_matrix((vals,(rr,cc)),shape=(self.d,self.d)) for name,(rr,cc,vals) in entries.items()}
        self.BL,self.BX,self.BY=mats['lap'],mats['x'],mats['y']
        self.bL,self.bX,self.bY=const['lap'],const['x'],const['y']
    def residual_jacobian(self,z,jac=True):
        psi,w=z[:self.d],z[self.d:];u=self.Dy@psi;v=-(self.Dx@psi)
        wx=self.Dx@w+self.BX@psi+self.bX;wy=self.Dy@w+self.BY@psi+self.bY
        wl=self.L@w+self.BL@psi+self.bL
        r=np.r_[self.L@psi+w,self.nu*wl-u*wx-v*wy]
        if not jac:return r
        diag=lambda a:sparse.diags(a,format='csr')
        jw=self.nu*self.L-diag(u)@self.Dx-diag(v)@self.Dy
        jp=self.nu*self.BL-diag(u)@self.BX-diag(v)@self.BY-diag(wx)@self.Dy+diag(wy)@self.Dx
        jj=sparse.bmat([[self.L,self.I],[jp,jw]],format='csc')
        return r,jj
    def reconstruct(self,z):
        n=self.n;h=self.h;m=self.m
        psi=np.zeros((n+1,n+1));w=psi.copy();u=psi.copy();v=psi.copy()
        psi[1:-1,1:-1]=z[:self.d].reshape(m,m);w[1:-1,1:-1]=z[self.d:].reshape(m,m)
        w[0,:]=(-4*psi[1,:]+.5*psi[2,:])/h**2
        w[-1,:]=(-4*psi[-2,:]+.5*psi[-3,:])/h**2
        w[:,0]=(-4*psi[:,1]+.5*psi[:,2])/h**2
        x=np.linspace(0,1,n+1)
        w[:,-1]=(-4*psi[:,-2]+.5*psi[:,-3])/h**2-3*lid(x)/h
        u[1:-1,1:-1]=(psi[1:-1,2:]-psi[1:-1,:-2])/(2*h)
        v[1:-1,1:-1]=-(psi[2:,1:-1]-psi[:-2,1:-1])/(2*h)
        u[:,-1]=lid(x)
        return dict(x=x,y=x.copy(),psi=psi,omega=w,u=u,v=v)

def interpolate_solution(fields,x,y):
    x,y=np.broadcast_arrays(x,y);flatx=x.ravel();flaty=y.ravel();out={}
    for key in ('u','v','omega','psi'):
        spl=RectBivariateSpline(fields['x'],fields['y'],fields[key],kx=3,ky=3,s=0)
        out[key]=spl.ev(flatx,flaty).reshape(x.shape)
    # Enforce the known velocity data exactly if evaluation lies on a wall.
    wall=(x==0)|(x==1)|(y==0)|(y==1)
    out['u'][wall]=0;out['v'][wall]=0;out['u'][y==1]=lid(x[y==1])
    return out

def solve_cavity(n,Re,initial=None,tol=1e-9,maxiter=25,_model_class=CavityFD):
    tick=time.perf_counter();model=_model_class(n,Re);setup=time.perf_counter()-tick
    if initial is None:z=np.zeros(2*model.d)
    else:
        xx,yy=np.meshgrid(np.linspace(0,1,n+1)[1:-1],np.linspace(0,1,n+1)[1:-1],indexing='ij')
        pp=interpolate_solution(initial,xx,yy);z=np.r_[pp['psi'].ravel(),pp['omega'].ravel()]
    trace=[];success=False
    for it in range(maxiter):
        r,j=model.residual_jacobian(z);rms=float(np.sqrt(np.mean(r*r)));rinf=float(abs(r).max())
        trace.append(dict(iteration=it,residual_rms=rms,residual_max=rinf))
        if rms<tol:success=True;break
        step=spsolve(j,-r)
        if not np.isfinite(step).all():raise FloatingPointError('Nonfinite Newton correction')
        alpha=1.
        for backtrack in range(16):
            zz=z+alpha*step;rr=model.residual_jacobian(zz,jac=False)
            if np.linalg.norm(rr)<(1-1e-4*alpha)*np.linalg.norm(r):break
            alpha*=.5
        else:
            if rms<10*tol:success=True;break
            raise RuntimeError(f'Newton line search failed at n={n},Re={Re},rms={rms:g}')
        z=zz;trace[-1].update(step_scale=alpha,step_max=float(abs(alpha*step).max()))
    fields=model.reconstruct(z);elapsed=time.perf_counter()-tick
    r=model.residual_jacobian(z,jac=False)
    if hasattr(model,'divergence'):div=model.divergence(fields)
    else:div=(fields['u'][2:,1:-1]-fields['u'][:-2,1:-1])/(2*model.h)+(fields['v'][1:-1,2:]-fields['v'][1:-1,:-2])/(2*model.h)
    topx=np.linspace(0,1,201);topw=interpolate_solution(fields,topx,np.ones(201))['omega']
    rec=dict(n_intervals=n,grid_nodes=n+1,Re=Re,nu=1/Re,spatial_order=getattr(model,'order',2),success=success,
        setup_seconds=setup,total_seconds=elapsed,iterations=len(trace)-1,trace=trace,
        residual_rms=float(np.sqrt(np.mean(r*r))),residual_max=float(abs(r).max()),
        discrete_divergence_max=float(abs(div).max()),streamfunction_min=float(fields['psi'].min()),
        top_wall_max_abs_vorticity_201=float(abs(topw).max()))
    if not success:raise RuntimeError(f'Newton did not converge: {rec}')
    return fields,rec

def comparison(coarse,fine,nq=100):
    z,w=leggauss(nq);z=(z+1)/2;w=w/2;x,y=np.meshgrid(z,z,indexing='ij');weights=w[:,None]*w[None,:]
    a=interpolate_solution(coarse,x,y);b=interpolate_solution(fine,x,y)
    return dict(velocity_relative_l2=float(np.sqrt(np.sum(weights*((a['u']-b['u'])**2+(a['v']-b['v'])**2))/np.sum(weights*(b['u']**2+b['v']**2)))),
        vorticity_relative_l2=float(np.sqrt(np.sum(weights*(a['omega']-b['omega'])**2)/np.sum(weights*b['omega']**2))),
        velocity_absolute_max=float(np.max(np.sqrt((a['u']-b['u'])**2+(a['v']-b['v'])**2))),
        quadrature_points_per_axis=nq)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--grids',default='32,64,128,256');parser.add_argument('--res',default='100,400');args=parser.parse_args()
    grids=[int(x) for x in args.grids.split(',')];res=[int(x) for x in args.res.split(',')]
    records=[];bycase={}
    for Re in res:
        previous=None
        for n in grids:
            path=ROOT/f'cfd_reference_cavity_Re{Re}_n{n}.npz'
            if path.exists():
                dat=np.load(path);fields={key:dat[key] for key in ('x','y','psi','omega','u','v')};rec=json.loads(str(dat['record']))
            else:
                if previous is None:
                    current=None;continuation=[]
                    for rr in ([10,25,50,100] if Re==100 else [10,25,50,100,200,400]):
                        current,cr=solve_cavity(n,rr,initial=current);continuation.append(cr)
                    fields,rec=current,continuation[-1];rec['continuation_seconds']=sum(cr['total_seconds'] for cr in continuation)
                    rec['continuation_Re']=[cr['Re'] for cr in continuation]
                else:fields,rec=solve_cavity(n,Re,initial=previous)
                np.savez_compressed(path,**fields,record=json.dumps(rec))
            if previous is not None:rec['difference_from_previous_grid']=comparison(previous,fields)
            records.append(rec);bycase[(Re,n)]=fields;previous=fields
            print('SOLVED',Re,n,rec['total_seconds'],rec['residual_rms'],rec['top_wall_max_abs_vorticity_201'],rec.get('difference_from_previous_grid'),flush=True)
            (ROOT/'cfd_reference_summary.json').write_text(json.dumps(dict(sources=SOURCES,rows=records),indent=2))
    print('DONE',flush=True)
if __name__=='__main__':main()
