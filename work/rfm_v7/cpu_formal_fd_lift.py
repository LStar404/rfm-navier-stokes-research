"""Exact discrete homogeneous lift for FD4, with short arithmetic audits."""
import argparse
from pathlib import Path
from unittest.mock import patch

import cpu_formal as base
from cpu_formal_fd_equilibrate import CaptureBandSolve, high_precision_rows, relative
from cpu_formal_reference_point1 import ROOT, PROTOCOL
import fd_full
from scipy import sparse
from scipy.interpolate import make_interp_spline

case = base.case
np = case.np
REPRESENTATION = "fd4_mode1_homogeneous_streamfunction"


class HomogeneousLiftFD(base.FullFD):
    """Store the homogeneous part of mode 1 and leave the FD stencils unchanged."""
    def __init__(self, dt, *args, lift_form="smooth_stencils", **settings):
        if lift_form not in ("smooth_stencils","stored_matrix"):
            raise ValueError("Expected a registered homogeneous lift form")
        with patch.object(fd_full,"BandSolve",CaptureBandSolve):
            super().__init__(dt,*args,**settings)
        self.point_bdf=self.lift_bdf
        self.point_first=self.lift_first
        radial=case.lift(self.s)[0].copy()
        radial[0],radial[-1]=0.,self.r[-1]
        self.lift_state=radial[1:-1]
        lap=(self.d2-sparse.eye(len(self.s),format="csr")[1:-1]).tocsr()
        lap_lift=high_precision_rows(lap,radial)[0].real
        wall_lift=high_precision_rows(self.wall_d2[0].tocsr(),radial)[0].real[0]
        omega=-np.r_[wall_lift,lap_lift,0.]*self.inv_r**2
        diffuse=case.NU*self.inv_r[1:-1]**2*high_precision_rows(lap,omega)[0].real
        self.stencil_bdf=1.5*omega[1:-1]-dt*diffuse
        self.stencil_first=omega[1:-1]-dt*diffuse
        self.matrix_bdf=high_precision_rows(self.bdf.matrices[1],self.lift_state,
                                            (1.+0j,self.point_bdf))[0].real
        self.matrix_first=high_precision_rows(self.first.matrices[1],self.lift_state,
                                              (1.+0j,self.point_first))[0].real
        if lift_form=="smooth_stencils":
            self.lift_bdf,self.lift_first=self.stencil_bdf,self.stencil_first
        else:
            self.lift_bdf,self.lift_first=self.matrix_bdf,self.matrix_first
        derivative=high_precision_rows(self.d1,radial)[0].real
        right=sparse.csr_matrix(self.right_d1[None,:])
        outer=high_precision_rows(right,radial[-5:])[0].real[0]
        self.lift_ur=1j*radial*self.inv_r
        self.lift_ut=-np.r_[0.,derivative,outer]*self.inv_r
        self.lift_omega=omega
        self.lift_omega_s=high_precision_rows(self.d1,omega)[0].real
        self.lift_probes=make_interp_spline(self.s,np.column_stack(
            (self.lift_ut,self.lift_omega)),k=5,axis=0)(case.PROBE_S)
        self.metadata.update(state_representation=REPRESENTATION,lift_form=lift_form)

    def physical_state(self, homogeneous, amplitude):
        result=homogeneous.copy()
        result[1]+=amplitude*self.lift_state
        return result

    def homogeneous_state(self, physical, amplitude):
        return self.physical_state(physical,-amplitude)

    def add_lift(self, fields, amplitude, boundary_velocity):
        ur,ut,omega=fields
        sl=slice(None) if boundary_velocity else slice(1,-1)
        ur[:,1]+=amplitude*self.lift_ur[sl]
        ut[:,1]+=amplitude*self.lift_ut[sl]
        omega[:,1]+=amplitude*self.lift_omega
        return ur,ut,omega

    def native(self, state, lift_c1, boundary_velocity):
        return self.add_lift(super().native(state,0.,boundary_velocity),lift_c1,boundary_velocity)

    def terms(self, state, lift_c1):
        fields=super().native(state,0.,False)
        omega_s=self.d1@fields[2]
        omega_s[:,1]+=lift_c1*self.lift_omega_s
        ur,ut,omega=self.add_lift(fields,lift_c1,False)
        mass=omega[1:-1].T.copy()
        mass[0]=state[0]
        advect,mean=self.plan.advect(ur,ut,omega[1:-1],omega_s,self.r[1:-1])
        advect[:,0]=mean
        return mass,advect.T

    def probes(self, state, lift_c1):
        velocity,vorticity=super().probes(state,0.)
        extras=2*float(np.real(lift_c1))*self.lift_probes
        return velocity+extras[:,0],vorticity+extras[:,1]


def load_lift_solution(path):
    """Load an artifact in its declared coordinates and reconstruct physical fields."""
    import json
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data["metadata"]))
    if metadata.get("state_representation")!=REPRESENTATION:
        raise ValueError("Artifact does not declare the homogeneous lift coordinates")
    return case.load_solution(path,HomogeneousLiftFD)


def report_delta(a,b):
    return dict(relative=relative(a,b),max_abs=float(np.max(abs(a-b))),
                outer12_relative=relative(a[-12:],b[-12:]))


def short_audit(spec,protocol,horizon,target):
    with patch.object(fd_full,"BandSolve",CaptureBandSolve):
        source=base.open_reference(spec,protocol)
    _,settings=base.config(spec,protocol)
    dt=1/spec["rate"]
    stable=HomogeneousLiftFD(dt=dt,**settings)
    ref=source.model
    physical=source.states[-1]
    amplitude=case.speed(source.times[-1])
    homogeneous=stable.homogeneous_state(physical,amplitude)
    a=ref.bdf.matrices[1]
    original_rhs,point_abs=high_precision_rows(a,physical[1],(amplitude,ref.lift_bdf))
    smooth_rhs,smooth_abs=high_precision_rows(a,homogeneous[1],(amplitude,stable.stencil_bdf))
    stored_rhs,_=high_precision_rows(a,homogeneous[1],(amplitude,stable.matrix_bdf))
    rhs=np.stack([m@state for m,state in zip(ref.bdf.matrices,physical)])
    rhs[1]=original_rhs
    direct=ref.advance(rhs.copy(),amplitude,False)
    lifted=stable.physical_state(stable.advance(rhs.copy(),amplitude,False),amplitude)
    before=ref.native(physical,amplitude,True)
    after=stable.native(homogeneous,amplitude,True)
    # Exercise the real part of the pulse without introducing an artificial
    # outer jump in the stored physical state.
    probe_amplitude=.07-.43j
    probe_physical=stable.physical_state(homogeneous,probe_amplitude)
    old_terms=ref.terms(physical,amplitude)
    new_terms=stable.terms(homogeneous,amplitude)
    summary=dict(spec=spec,smooth_vs_stored_lift_bdf=report_delta(stable.stencil_bdf,stable.matrix_bdf),
        smooth_vs_stored_lift_first=report_delta(stable.stencil_first,stable.matrix_first),
        original_point_lift_vs_smooth_norm_ratio=float(np.linalg.norm(ref.lift_bdf)/
                                                        np.linalg.norm(stable.stencil_bdf)),
        reconstructed_original_rhs_vs_smooth=report_delta(smooth_rhs,original_rhs),
        reconstructed_original_rhs_vs_stored=report_delta(stored_rhs,original_rhs),
        original_reconstructed_rhs_outer_cancellation=float(
            max(point_abs[-12:]/np.maximum(abs(original_rhs[-12:]),np.finfo(float).tiny))),
        smooth_reconstructed_rhs_outer_cancellation=float(
            max(smooth_abs[-12:]/np.maximum(abs(smooth_rhs[-12:]),np.finfo(float).tiny))),
        same_rhs_advance=report_delta(lifted[1],direct[1]),
        mass_and_explicit_term={name:relative(x,y) for name,x,y in zip(
            ("mass","nonlinear"),new_terms,old_terms)},
        full_native={name:relative(x,y) for name,x,y in zip(("ur","ut","omega"),after,before)},
        probes={name:relative(x,y) for name,x,y in zip(("velocity","vorticity"),
            stable.probes(homogeneous,probe_amplitude),ref.probes(probe_physical,probe_amplitude))})
    baseline=case.solve(base.FullFD,steps=horizon*spec["rate"],horizon=horizon,
                       snapshots=(float(horizon),),**settings)
    candidate=case.solve(HomogeneousLiftFD,steps=horizon*spec["rate"],horizon=horizon,
                        snapshots=(float(horizon),),**settings)
    grids=base.grid_set()
    scores=case.score(candidate,case.reference_fields(baseline,grids),grids)
    path=target/f"fd_lift_short_n{spec['n']}.npz"
    temp=path.with_suffix(".tmp.npz")
    case.store_solution(temp,candidate,settings)
    temp.replace(path)
    replay=load_lift_solution(path)
    if not np.array_equal(replay.states,candidate.states):
        raise RuntimeError("Homogeneous artifact changed on replay")
    summary.update(short_horizon=horizon,short_worst=case.maxima(scores),
        short_scores=scores,short_boundaries=base.boundary_errors(candidate),
        short_physical_state_difference=relative(candidate.model.physical_state(candidate.states[-1],
            case.speed(float(horizon))),baseline.states[-1]),
        short_artifact=path.name,short_sha256=base.digest(path))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon",type=int,default=1)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"fd_lift_short_actual")
    args=parser.parse_args()
    if not 1<=args.horizon<=2:
        parser.error("This command is only the short validation")
    target=args.output_dir
    target.mkdir(parents=True,exist_ok=False)
    protocol=base.read(PROTOCOL)
    result=dict(status="running",representation=REPRESENTATION,rows=[],
                note="Exact-arithmetic variable substitution with the same FD stencils. The two "
                     "combined-lift calculations differ only from stored coefficient and arithmetic rounding. "
                     "Short tests compare numerical sensitivity, not long-reference convergence.")
    base.save(target/"audit.json",result)
    for spec in protocol["reference_path"][:2]:
        row=short_audit(spec,protocol,args.horizon,target)
        result["rows"].append(row)
        base.save(target/"audit.json",result)
        print(spec["n"],row["short_worst"],row["smooth_vs_stored_lift_bdf"],flush=True)
    result["status"]="complete"
    base.save(target/"audit.json",result)


if __name__=="__main__":
    main()
