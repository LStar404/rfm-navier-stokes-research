# Full-annulus v7 CPU comparison

This preregistration preserves the v6 equation, geometry, boundaries, forcing,
initial state, SBDF2 method and scoring. The domain is the finite annulus
0.5 <= r <= 6 at Re = 100 with cylinder diameter and target streamwise speed
equal to 1. Both parity classes and the mean angular-momentum mode are
advanced. The transverse pulse acts only on 4 < t < 12. The final state is
sampled through t = 72.

## Reference check

The independent finite-difference ladder follows radial 280 to 320 at
K = 104 and 256 steps per unit time, angle 104 to 112 at n = 320 and the
same step, and time 256 to 320 at n = 320 and K = 112. Three members are
from the immutable v6 addendum. The missing n320/K112/256 intermediate
has been specified before it is solved.

For each ladder edge, the four empirical relative discrepancies are
computed on all score grids and times. Let e_i be their maximum. The
combined screen is prod(1 + e_i) - 1 <= 0.00125. This composes observed
pairwise discrepancies with the correct changing denominators by the
triangle inequality. It bounds only the finite path from the coarsest
tested member to the final reference. It is an empirical screen for
reference sensitivity.

The discrepancy with v6's FD240 reference remains a separate reported
diagnostic. The v6 0.5% timing transfer was rejected. v7 is a new cold
comparison against the independently refined ladder, including fresh
finite-difference timing. The historical FD240 case is not a rung of
the new ladder because it changes angular resolution as well as radial
resolution compared with the new reference.

The domain influence remains a separate question. A follow-on domain
study should vary only the outer radius, resolve the outer vorticity
layer, compare on shared inner radii, and report both absolute-time
differences and separate late-time periodic diagnostics. Moving the
inflow boundary can alter when the perturbation reaches the wake.

## Qualification

Compare velocity and curl at the absolute times 24, 36, 48, 54, 60, 66
and 72. Use the v6 Gauss123, midpoint129 and outer-layer144 grids and
the same weighted relative L2 errors over the full domain and r <= 2.5.
All four errors must be at most 0.005 on every grid and time. The wake
probe check must also pass. No phase shift enters this gate.

Each of RFM, Legendre and FD4 has three preregistered resolution levels.
RFM uses seeds 7331, 7687, 8011, 8353 and 9041. The first three are
historical and the last two are new. Legendre and FD4 each require one
deterministic qualification trajectory. A configuration qualifies only
if its complete seed set passes. Accuracy jobs may run concurrently.
Their recorded elapsed times are diagnostic and play no role in
selecting a timing configuration.

## Isolated timing

Select the first qualifying configuration in each family's fixed
low-to-high resolution order. The three selected configurations are
timed in five randomized blocks, using all five RFM seeds and five
repetitions of each deterministic method. CPU frequency, background
work and OS conditions still contribute to measured variability.
Run only after competing experiments are stopped and use the same
logical processor and single-thread settings for all methods.

Cold in-process timings include setup, every integration step and
checkpoint/probe generation. This protocol excludes Python import,
reference loading, score evaluation and serialization from that time.
Every timed trajectory is scored again. Report all failures and the
distribution of elapsed times. A speed claim requires the reference
screen and all timed trajectories to qualify.

## Commands

Use the existing Windows Python environment with NumPy, SciPy and the
v6 native FD library. Commands are given relative to the workspace
root.

```powershell
python work/rfm_v7/cpu_formal.py smoke
python work/rfm_v7/cpu_formal.py reference
python work/rfm_v7/cpu_formal.py audit
python work/rfm_v7/cpu_formal.py qualify --workers 7
python work/rfm_v7/cpu_formal.py timing
```

Reference and accuracy jobs may overlap. Timing runs alone. The
journals and per-job records permit restart with the same inputs.
Archive all raw records, artifacts, command logs and source hashes.

The optional throughput sweep runs a shorter mixed-family CPU workload
with distinct process and NumPy BLAS thread counts. It uses the actual
solver path and includes process startup. It does not tune FFT or
native FD threading and does not supply speed-ranking data.
