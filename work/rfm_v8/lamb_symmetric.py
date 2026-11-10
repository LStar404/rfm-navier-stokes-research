"""Symmetrically precondition the real frozen-vorticity Lamb system."""

from lamb_linearized import LambGalerkin, LambStepError, case, np
from radial_layers import radial_quadrature
from scipy.sparse.linalg import LinearOperator, gmres


class SymmetricLambGalerkin(LambGalerkin):
    def __init__(self,dt,**config):
        super().__init__(dt=dt,**config)
        _,w = radial_quadrature(self.metadata["q"],self.metadata["split"])
        f,_,d2 = self.samples
        rw = np.sqrt(w)/self.r
        j,u = (d2-f)*rw[:,None],f*rw[:,None]
        cross = j.T@u
        k2_minus_one = (self.k2-1)[:,None,None]
        k = case.NU*((j.T@j)-k2_minus_one*(cross+cross.T)+k2_minus_one**2*(u.T@u))
        m = self.M0+self.k2[:,None,None]*self.M1
        strain = (self.mean_samples[0]+self.mean_samples[1])*np.sqrt(w)[:,None]
        mean = 1.5*self.mean_mass+dt*case.NU*(strain.T@strain)
        self.lower = np.linalg.cholesky(1.5*m+dt*k)
        self.lower_mean = np.linalg.cholesky(mean)
        # Form these one time per model, then use two batched block products per
        # matrix-vector operation. No per-step block factorization is introduced.
        self.inverse_lower = np.linalg.solve(self.lower,np.broadcast_to(np.eye(self.width),self.lower.shape))
        self.inverse_lower_mean = np.linalg.solve(self.lower_mean,np.eye(self.mean_width))

    def apply_lower(self,state,*,inverse,transpose):
        modes = self.inverse_lower if inverse else self.lower
        mean = self.inverse_lower_mean if inverse else self.lower_mean
        if transpose:
            modes,mean = modes.swapaxes(-1,-2),mean.T
        out = np.zeros_like(state)
        values = np.ascontiguousarray(state[:,1:].T).view(float).reshape(self.k,self.width,2)
        out[:,1:] = (modes@values).view(np.complex128)[...,0].T
        out[:self.mean_width,0] = mean@state[:self.mean_width,0].real
        return out

    def implicit_step(self,state,prior,amplitudes,h,rtol=1e-9,*,enforce=True):
        gprev,gnow,gnew = amplitudes
        calls_at_start = self.velocity_products
        omega = self.fields_on(2*state-prior,2*gnow-gprev,self.samples,self.lift,self.r,self.mean_samples)[2]
        self.freeze_vorticity(omega)
        rhs = 2*self.mass_only(state,gnow)-.5*self.mass_only(prior,gprev)
        rhs[:,1] -= gnew*self.bdf_lift
        rhs += h*self.flux_frozen(np.zeros_like(state),gnew)
        b = self.pack(self.apply_lower(rhs,inverse=True,transpose=False))
        guess = self.pack(self.apply_lower(2*state-prior,inverse=False,transpose=True))
        calls_before = self.velocity_products
        history = []

        def matvec(y):
            physical = self.apply_lower(self.unpack(y),inverse=True,transpose=True)
            flux = self.flux_frozen(physical)
            return y-h*self.pack(self.apply_lower(flux,inverse=True,transpose=False))

        system = LinearOperator((self.real_size,self.real_size),matvec=matvec,dtype=np.float64)
        answer,info = gmres(system,b,x0=guess,rtol=rtol,atol=0.,restart=24,maxiter=10,
                            callback=history.append,callback_type="pr_norm")
        final = np.linalg.norm(matvec(answer)-b)/max(np.linalg.norm(b),np.finfo(float).tiny)
        solution = self.apply_lower(self.unpack(answer),inverse=True,transpose=True)
        converged = bool(info == 0 and np.isfinite(solution).all() and np.isfinite(final) and final <= 2*rtol)
        record = dict(iterations=len(history),operator_calls=self.velocity_products-calls_before,
                      velocity_product_calls=self.velocity_products-calls_at_start,gmres_info=int(info),
                      requested_rtol=float(rtol),preconditioned_residual=float(final),
                      final_reported_relative_residual=float(history[-1]) if history else None,
                      converged=converged,preconditioner="symmetric_block_cholesky")
        if enforce and not converged:
            raise LambStepError(record)
        return solution,record
