# Outer-layer rate tangents

The existing outer-layer rate family is evaluated in distance
\(d=L-s\) from the outer boundary and \(z=s/L\). For a nonzero angular
mode the trial function is

\[
\phi_\beta=z^2\left[d+(\beta+2/L)d^2\right]e^{-\beta d}.
\]

Its rate derivative is
\(\partial_\beta\phi_\beta=-(\beta+2/L)z^2d^3e^{-\beta d}\).
The factor \(z^2\) sets value and first derivative to zero at the inner
wall. At the outer wall, the factor \(d^3\) sets the value and second
derivative to zero. This is the homogeneous streamfunction trace
corresponding to the prescribed radial velocity and zero vorticity.

For the mean angular-momentum mode the existing velocity layer is
\(\chi_\beta=z[1+(\beta+1+1/L)d]e^{-\beta d}\).
Its rate derivative is
\(\partial_\beta\chi_\beta=-(\beta+1+1/L)zd^2e^{-\beta d}\).
The inner value is zero. The outer value and derivative are zero,
preserving the homogeneous Robin condition on the angular mean.
The implementation divides away the nonzero scalar factors in the
derivatives before the usual column scaling.

This adds first-order rate information near the chosen outflow decay
scales. It can represent nearby rates with a local Taylor expansion.
The largest scale is guided by \(R\,|u_r|/\nu\) near outflow, of order
600 in the fixed radius-six case. Finite-dimensional approximation
quality and conditioning are empirical.

The frozen comparison keeps the original six functions and compares
three additional ordinary layers with three tangent layers. Both RFM
and Legendre get each intervention. The angular cutoff, number of
quadrature nodes, steps and seed are fixed. Report the retained rank
as well as the nominal dimension, including the independent mean
space. The sampled oracle fits velocity and vorticity separately
under the unchanged lift. It is followed by a full \(t=72\) trajectory
for every variant, including failures. Timings from concurrent
correctness runs are not used to rank implementations.
