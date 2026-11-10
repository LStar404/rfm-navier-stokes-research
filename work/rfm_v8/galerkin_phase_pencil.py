"""Fixed-FP64 phase pencil candidates with the original dense pack interface."""

from dataclasses import dataclass
from pathlib import Path
import sys

V7 = Path(__file__).resolve().parents[1] / "rfm_v7"
sys.path.insert(0,str(V7))

from galerkin_phase_shared import case, np, pack_step, spatial_blocks
from scipy.linalg import cho_factor, cho_solve, eigh


def upper_symmetric(matrix):
    """Use the same authoritative triangle as frozen inverse_spd."""
    upper = np.triu(matrix)
    return upper + np.triu(matrix,1).T


@dataclass(frozen=True)
class AnchorPencil:
    transform: np.ndarray
    eigenvalues: np.ndarray
    anchor_dt: float

    @classmethod
    def factor(cls, mass, diffusion, anchor_dt, mean_strain=None):
        anchor = assembled_matrix(mass,diffusion,anchor_dt,mean_strain)
        scale = np.sqrt(np.diag(anchor))
        values,vectors = eigh(mass/scale[:,None]/scale,anchor/scale[:,None]/scale,
                              lower=False,driver="gvd",check_finite=False)
        return cls(vectors/scale[:,None],values,anchor_dt)

    def diagonal(self, alpha, dt):
        ratio = dt/self.anchor_dt
        values = ratio+(alpha-1.5*ratio)*self.eigenvalues
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise ArithmeticError("The phase pencil has a nonpositive diagonal denominator")
        return values

    def project(self, rhs):
        return self.transform.T @ rhs

    def apply_projected(self, alpha, dt, rhs):
        diagonal = self.diagonal(alpha,dt)
        return self.transform @ (rhs/diagonal.reshape((-1,)+(1,)*(rhs.ndim-1)))

    def solve(self, alpha, dt, rhs):
        return self.apply_projected(alpha,dt,self.project(rhs))


def scaled_direct(matrix, rhs):
    scale = np.sqrt(np.diag(matrix))
    factor = cho_factor(matrix/scale[:,None]/scale,lower=False,check_finite=False)
    if rhs.ndim == 1:
        return cho_solve(factor,rhs/scale,check_finite=False)/scale
    return cho_solve(factor,rhs/scale[:,None],check_finite=False)/scale[:,None]


def assembled_matrix(mass, diffusion, dt, mean_strain=None):
    if mean_strain is not None:
        # Preserve the v7 mean expression's order: multiply dt*NU first.
        return 1.5*mass+dt*case.NU*mean_strain
    return 1.5*mass+dt*diffusion


def mode_matrices(model, blocks):
    mass,diffusion,_,mean_strain = blocks
    return [(model.mean_mass,case.NU*mean_strain),*zip(mass,diffusion)]


def right_hand_sides(model, blocks):
    mass,_,lift_diffusion,_ = blocks
    zero_lift = np.empty((model.width,0))
    lift = np.column_stack((model.M_lift,lift_diffusion))
    return [np.column_stack((model.mean_mass,model.mean_test)),*(
        np.column_stack((m,model.test_s,model.test,lift if k==1 else zero_lift))
        for k,m in enumerate(mass,1))]


def pack_projected(model, dt, modes):
    """Keep original mode layout, normalization, mean and future lift semantics."""
    width,mean_width,q = model.width,model.mean_width,len(model.r)
    linear = np.zeros((width,width,model.k+1))
    radial = np.zeros((width,q,model.k+1))
    angular = np.zeros_like(radial)
    scale = dt/model.kernel.n
    linear[:mean_width,:mean_width,0] = modes[0][:,:mean_width]
    radial[:mean_width,:,0] = -scale*modes[0][:,mean_width:]
    for k,rhs in enumerate(modes[1:],1):
        linear[...,k] = rhs[:,:width]
        radial[...,k] = scale*rhs[:,width:width+q]
        angular[...,k] = (scale*k)*rhs[:,width+q:width+2*q]
    mass_lift,diffusion_lift = modes[1][:,-2:].T
    return linear,radial,angular,mass_lift,1.5*mass_lift+dt*diffusion_lift


def direct_packs(model, blocks, phase_dts):
    matrices = mode_matrices(model,blocks)
    rhs = right_hand_sides(model,blocks)
    return [pack_projected(model,dt,[scaled_direct(assembled_matrix(m,k,dt,blocks[3] if i==0 else None),b)
                                     for i,((m,k),b) in enumerate(zip(matrices,rhs,strict=True))])
            for dt in phase_dts]


def prepare_pencils(model, blocks, phase_dts):
    anchor_dt = sorted(phase_dts)[len(phase_dts)//2]
    matrices = mode_matrices(model,blocks)
    rhs = right_hand_sides(model,blocks)
    pencils = [AnchorPencil.factor(m,k,anchor_dt,blocks[3] if i==0 else None) for i,(m,k) in enumerate(matrices)]
    projected = [p.project(b) for p,b in zip(pencils,rhs,strict=True)]
    return pencils,projected


def pencil_packs(model, phase_dts, pencils, projected):
    return [pack_projected(model,dt,[p.apply_projected(1.5,dt,b)
                                     for p,b in zip(pencils,projected,strict=True)])
            for dt in phase_dts]
