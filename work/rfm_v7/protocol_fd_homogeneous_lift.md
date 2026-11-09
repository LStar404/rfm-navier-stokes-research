# FD4 mode-one homogeneous coordinates

The baseline stores the interior streamfunction in each nonzero mode,
with the outer streamfunction value entering mode one through a
boundary lift. Write the stored mode-one interior vector as \(x\), its
known complex boundary amplitude as \(a\), and the sampled smooth
radial lifting function as \(L\). \(L\) has the same outer value and
inner wall constraints as the baseline lift.

Let \(M_0,D_0\) be the homogeneous finite-difference mass and diffusion
maps. Let \(M_p,D_p\) be the action on the outer point lift. The
implicit matrix is \(A=\tfrac32M_0-\Delta tD_0\), and the original solve is

\[
A x^{n+1}=\mathcal R^n-a^{n+1}\ell_p,
\qquad \ell_p=\tfrac32M_p-\Delta tD_p .
\]

In the new coordinates \(y=x-aL_{\mathrm{int}}\), the exact-arithmetic
equation is

\[
A y^{n+1}=\mathcal R^n-a^{n+1}
  \left(A L_{\mathrm{int}}+\ell_p\right).
\]

The first-step version replaces the mass factor \(3/2\) with \(1\).
The same discrete boundary traces are preserved. Both solves use the
original band matrices. Mode zero and modes above one are unaffected.

## Two computations of the combined lift

The matrix calculation evaluates the right-hand coefficient
\(A L_{\mathrm{int}}+\ell_p\) from the stored FP64 entries with 50-digit
arithmetic. The stencil calculation evaluates the existing discrete
mass and diffusion operators on the smooth full sampled \(L\) in
separate derivative stages and forms the same combination. These
agree algebraically in exact arithmetic. Their measured differences
also include the effects of rounding in the stored matrix and its
boundary column.

The homogeneous implementation evaluates velocity, vorticity and the
explicit vorticity derivative as homogeneous contributions plus
precomputed smooth-lift contributions. It preserves the same centerline
probe locations and interpolation. It does not add the full
nonhomogeneous streamfunction back before applying a fine-grid
derivative. The stored state is explicitly marked
`fd4_mode1_homogeneous_streamfunction`. Use
`cpu_formal_fd_lift.load_lift_solution` for physical-field replay.

## Short audit and limitation

The audit uses actual saved \(t=72\) states to measure the cancellation
in the reconstructed original and homogeneous equations. The full
previous-step histories are not available in those snapshots. A
manufactured real part of the boundary amplitude separately exercises
the probe contribution, using a consistent transformed physical state.
Short independent marches validate output semantics and boundaries.
They cannot establish long-time accuracy or radial convergence.
