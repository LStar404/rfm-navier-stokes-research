"""Independent FD4 radial discretization for the full-domain cylinder."""
from pathlib import Path
import ctypes
import sys

import full_case as case
from full_case import np
from full_nonlinear import FullNonlinear
from scipy import sparse
from scipy.interpolate import make_interp_spline
from scipy.linalg.lapack import dgbtrf, zgbtrs

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'rfm_v3'))
from cylinder_fd import difference_matrix, wall_second


def end_derivative(nodes):
    x = nodes - nodes[0]
    scale = np.max(np.abs(x))
    target = np.array([0.,1./scale,0.,0.,0.])
    return np.linalg.solve((x/scale)**np.arange(5)[:,None],target)


class BandSolve:
    """Reuse LU factors across every step, using both real/imaginary RHS."""

    def __init__(self, matrices, backend):
        coords = [a.tocoo() for a in matrices]
        offsets = [a.row-a.col for a in coords]
        self.lower = max(int(z.max()) for z in offsets)
        self.upper = -min(int(z.min()) for z in offsets)
        self.width = matrices[0].shape[0]
        self.modes = len(matrices)
        self.native = None
        if backend == 'native':
            path = ROOT/('fd_full_band.dll' if sys.platform=='win32' else 'fd_full_band.so')
            library = ctypes.CDLL(str(path))
            solve = library.fd_solve_bands
            solve.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*4
            solve.restype = None
            self.native = solve
        elif backend != 'scipy':
            raise ValueError(f'Unknown linear backend {backend}')
        factors = []
        pivots = []
        for a,z in zip(coords,offsets):
            ab = np.zeros((2*self.lower+self.upper+1,self.width),order='F')
            ab[self.lower+self.upper+z,a.col] = a.data
            factor,pivot,info = dgbtrf(ab,self.lower,self.upper,overwrite_ab=True)
            if info:
                raise np.linalg.LinAlgError(f'DGBTRF failed ({info})')
            factors.append(factor)
            pivots.append(pivot)
        self.factors = np.ascontiguousarray(factors) if self.native else [
            np.asfortranarray(a,dtype=np.complex128) for a in factors]
        self.pivots = np.ascontiguousarray(pivots,dtype=np.intc)

    def __call__(self, rhs):
        result = np.array(rhs,dtype=np.complex128,order='C',copy=True)
        if self.native:
            self.native(self.factors.ctypes.data,self.pivots.ctypes.data,
                        result.ctypes.data,self.modes,self.width,self.lower,self.upper)
        else:
            for i,(a,pivot) in enumerate(zip(self.factors,self.pivots)):
                values,info = zgbtrs(a,self.lower,self.upper,result[i,:,None],pivot,overwrite_b=True)
                if info:
                    raise np.linalg.LinAlgError(f'ZGBTRS failed ({info})')
                result[i] = values[:,0]
        return result


class FullFD:
    """Mode 0 solves tangential momentum; modes 1..k solve streamfunction."""

    def __init__(self, dt, n, k, stretch=2.1, angular='fft', linear='native', grid='inner'):
        if n<8 or k<1:
            raise ValueError('FD4 requires n>=8 and k>=1')
        t = np.linspace(0.,1.,n+1)
        if grid=='inner':
            if stretch:
                t = -np.log1p(t*np.expm1(-stretch))/stretch
        elif grid in ('both','outer'):
            if not 0<=stretch<=1:
                raise ValueError('Cosine-grid blend must lie in [0,1]')
            cosine = .5*(1-np.cos(np.pi*t)) if grid=='both' else np.sin(.5*np.pi*t)
            t = (1-stretch)*t+stretch*cosine
        else:
            raise ValueError(f'Unknown radial grid {grid}')
        t[-1] = 1.
        self.s = case.LENGTH*t
        self.r = case.radius(self.s)
        self.inv_r = 1/self.r
        self.k = np.arange(k+1)
        self.k2 = self.k*self.k
        self.rhs_shape = (k+1,n-1)
        self.plan = FullNonlinear(n-1,k,angular)
        self.d1 = difference_matrix(self.s,1)
        self.d2 = difference_matrix(self.s,2)
        self.wall_d2,_ = wall_second(self.s)
        self.left_d1 = end_derivative(self.s[:5])
        self.right_d1 = end_derivative(self.s[-5:][::-1])[::-1]
        self.right_mean = -self.right_d1[:-1]/(1.+self.right_d1[-1])
        eye = sparse.eye(n+1,format='csr')
        interior = eye[1:-1]
        homogeneous = interior.T
        inv_r2 = sparse.diags(self.inv_r[1:-1]**2)
        mean_extension = homogeneous.tolil()
        mean_extension[-1,-4:] = self.right_mean
        mean_extension = mean_extension.tocsr()
        mean_diffusion = case.NU*inv_r2@(self.d2-interior)@mean_extension
        mean_mass = sparse.eye(n-1,format='csr')
        mats = [1.5*mean_mass-dt*mean_diffusion]
        first = [mean_mass-dt*mean_diffusion]
        lift = np.zeros(n+1)
        lift[-1] = self.r[-1]
        for mode in self.k[1:]:
            lap = self.d2-mode*mode*interior
            omega_int = -inv_r2@lap
            omega_wall = -(self.inv_r[0]**2)*self.wall_d2[0]
            omega = sparse.vstack((omega_wall@homogeneous,omega_int@homogeneous,
                                   sparse.csr_matrix((1,n-1))),format='csr')
            mass = omega[1:-1]
            diffusion = case.NU*inv_r2@lap@omega
            mats.append(1.5*mass-dt*diffusion)
            first.append(mass-dt*diffusion)
            if mode==1:
                omega_lift = np.r_[float((omega_wall@lift)[0]),omega_int@lift,0.]
                diffuse_lift = case.NU*inv_r2@lap@omega_lift
                self.lift_bdf = 1.5*omega_lift[1:-1]-dt*diffuse_lift
                self.lift_first = omega_lift[1:-1]-dt*diffuse_lift
        self.bdf = BandSolve(mats,linear)
        # At a rest start, the boundary only drives mode 1. Retain the
        # general first-step solver so nonzero first RHS are also valid.
        self.first = BandSolve(first,linear)
        d1_full = sparse.vstack((sparse.csr_matrix((1,n+1)),self.d1,
                                 sparse.csr_matrix((self.right_d1,(np.zeros(5),np.arange(n-4,n+1))),
                                                   shape=(1,n+1))))
        probe_weights = make_interp_spline(self.s,np.eye(n+1),k=5,axis=0)(case.PROBE_S)
        self.probe_azimuthal = -(probe_weights*self.inv_r)@d1_full
        vorticity_d2 = sparse.vstack((self.wall_d2[0],self.d2,sparse.csr_matrix((1,n+1))))
        self.probe_vort = -(probe_weights*self.inv_r**2)@vorticity_d2
        self.probe_psi = probe_weights*self.inv_r**2
        self.probe_psi[:,-1] = 0. # The outer vorticity is prescribed.
        mean_d1 = d1_full.tolil()
        mean_d1[0,:5] = self.left_d1
        mean_curl = sparse.vstack(((mean_d1.tocsr()+eye)[:-1],sparse.csr_matrix((1,n+1))))
        self.probe_mean_vort = (probe_weights*self.inv_r)@mean_curl@mean_extension
        self.probe_mean_velocity = probe_weights@mean_extension
        self.metadata = dict(method='fd4',n=n,k=k,stretch=stretch,grid=grid,angular=angular,linear=linear,
                             ntheta=self.plan.n,band_lower=self.bdf.lower,band_upper=self.bdf.upper)

    def advance(self, rhs, lift_c1, first):
        rhs[1] -= lift_c1*(self.lift_first if first else self.lift_bdf)
        return (self.first if first else self.bdf)(rhs)

    def native(self, state, lift_c1, boundary_velocity):
        psi = np.zeros((len(self.s),len(self.k)),dtype=np.complex128)
        psi[1:-1] = state.T
        psi[-1,1] = self.r[-1]*lift_c1
        psi[-1,0] = self.right_mean@state[0,-4:]
        d1 = self.d1@psi
        d2 = self.d2@psi
        inv = self.inv_r[:,None]
        omega = np.zeros_like(psi)
        omega[1:-1] = -(d2-self.k2*psi[1:-1])*inv[1:-1]**2
        omega[1:-1,0] = (d1[:,0]+psi[1:-1,0])*self.inv_r[1:-1]
        omega[0] = -(self.wall_d2[0]@psi)[0]*self.inv_r[0]**2
        omega[0,0] = (self.left_d1@psi[:5,0])*self.inv_r[0]
        if boundary_velocity:
            derivative = np.zeros_like(psi)
            derivative[1:-1] = d1
            derivative[-1] = self.right_d1@psi[-5:]
            ur = (1j*self.k)*psi*inv
            ut = -derivative*inv
            ut[:,0] = psi[:,0]
        else:
            ur = (1j*self.k)*psi[1:-1]*inv[1:-1]
            ut = -d1*inv[1:-1]
            ut[:,0] = psi[1:-1,0]
        return ur,ut,omega

    def terms(self, state, lift_c1):
        ur,ut,omega = self.native(state,lift_c1,False)
        mass = omega[1:-1].T.copy()
        mass[0] = state[0]
        advect,mean = self.plan.advect(ur,ut,omega[1:-1],self.d1@omega,self.r[1:-1])
        advect[:,0] = mean
        return mass,advect.T

    def modal(self, state, lift_c1, s):
        fields = self.native(state,lift_c1,True)
        return tuple(make_interp_spline(self.s,x,k=5,axis=0)(s) for x in fields)

    def probes(self, state, lift_c1):
        modes = 2*state[1:].real
        psi = np.r_[0.,modes.sum(axis=0),2*self.r[-1]*lift_c1.real]
        weighted = self.k2[1:]@modes
        velocity = self.probe_azimuthal@psi + self.probe_mean_velocity@state[0].real
        omega = self.probe_vort@psi+self.probe_psi[:,1:-1]@weighted
        omega += self.probe_mean_vort@state[0].real
        return velocity,omega
