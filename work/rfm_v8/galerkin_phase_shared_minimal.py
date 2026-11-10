"""One shared Galerkin spatial assembly for the three CUDA SBDF2 phases."""

from galerkin_minimal_model import RecurrenceGalerkin
from sbdf2_schedule_minimal import bridge_weights
from full_galerkin import inverse_spd
from galerkin_cuda_native import case, np, operators
from radial_layers import radial_quadrature


def spatial_blocks(model):
    """Recover the step-independent matrices in the frozen assembly order."""
    # The frozen constructor exposes the transformed basis, but keeps the
    # weights and diffusion Gram blocks local. Rebuild them once for both phases.
    s, w = radial_quadrature(model.metadata["q"], model.metadata["split"])
    if not np.array_equal(s, model.s):
        raise ArithmeticError("The shared quadrature differs from the startup model")
    sw = np.sqrt(w)
    f, _, d2 = model.samples
    lf, _, ld2 = model.lift
    mass = model.M0+model.k2[:, None, None]*model.M1
    l1 = (d2-f)*(sw/model.r)[:, None]
    fw = f*(sw/model.r)[:, None]
    cross = l1.T@fw
    delta = (model.k2-1)[:, None, None]
    diffusion = case.NU*((l1.T@l1)-delta*(cross+cross.T)+delta**2*(fw.T@fw))
    lift_diffusion = case.NU*l1.T@((ld2-lf)*sw/model.r)
    mv, md = model.mean_samples
    strain = (mv+md)*sw[:, None]
    return mass, diffusion, lift_diffusion, strain.T@strain


def pack_step(model, blocks, dt):
    """Rebuild just the inverse and the five step-dependent device operators."""
    mass, diffusion, lift_diffusion, mean_strain = blocks
    inverse = np.stack([inverse_spd(a) for a in 1.5*mass+dt*diffusion])
    mean_inverse = inverse_spd(1.5*model.mean_mass+dt*case.NU*mean_strain)
    lift = 1.5*model.M_lift+dt*lift_diffusion
    width, modes, q = model.width, model.k+1, len(model.r)
    mean_width = model.mean_width
    linear = np.zeros((width, width, modes))
    radial = np.zeros((width, q, modes))
    angular = np.zeros_like(radial)
    linear[..., 1:] = (inverse @ mass).transpose(1, 2, 0)
    linear[:mean_width, :mean_width, 0] = mean_inverse @ model.mean_mass
    scale = dt/model.kernel.n
    radial[..., 1:] = scale*(inverse @ model.test_s).transpose(1, 2, 0)
    radial[:mean_width, :, 0] = -scale*(mean_inverse @ model.mean_test)
    angular[..., 1:] = scale*model.modes*(inverse @ model.test).transpose(1, 2, 0)
    return linear, radial, angular, inverse[0] @ model.M_lift, inverse[0] @ lift


def phase_operators_shared(startup_rate, late_rate, config):
    """Return the frozen phase interface using one recurrence basis and one observation pack.

    A_eff = (3/2) M + s*h_new K = s A_bridge for s=(3/2)/a0.
    Multiply the bridge linear and history mass-lift packs by s. The flux
    already includes s*h_new and the future lift already uses A_eff.
    """
    old_dt, new_dt = 1/startup_rate, 1/late_rate
    a0, b0, b1, e0, e1 = bridge_weights(new_dt/old_dt)
    coefficients = b0, -b1, e0, -e1
    old = RecurrenceGalerkin(dt=old_dt, **config)
    initial = operators(old, old_dt)
    if old_dt == new_dt:
        return old, initial[5:], (initial[:5],)*3, coefficients
    blocks = spatial_blocks(old)
    scale = 1.5/a0
    linear, radial, angular, mass_lift, bdf_lift = pack_step(old, blocks, scale*new_dt)
    bridge = scale*linear, radial, angular, scale*mass_lift, bdf_lift
    return old, initial[5:], (initial[:5], bridge, pack_step(old, blocks, new_dt)), coefficients
