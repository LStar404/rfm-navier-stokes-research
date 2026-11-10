Review revision: matched-time profiles and a later SBDF2 switch
============================================================

Scope
-----
Four new CPU trajectories were actually integrated through T=72, with switch=12:
  RFM37 seed7331 / Legendre35: retained rank39, K72, q66, rates128->112.
  RFM41 seed7331 / Legendre36: retained rank40, K80, q76, rates152->136.
All use original spatial operators, four CPU workers with BLAS threads=1,
and the frozen 19 physical snapshots and 577 probe observations. No GPU was
used and no new reference was generated. Parallel elapsed times in raw
metadata are diagnostic and must not be presented as isolated speed results.

The original CPU phase clock accepts an arbitrary switch time. Its default
observation list contains t=8 bridge checkpoints, so the separate driver
explicitly sets the original 19 common physical snapshots. No old solver,
reference, qualification record, or manuscript was modified by this task.

Artifacts
---------
profiles_bridge_en.tex / profiles_bridge_zh.tex: two parallel manuscript inserts.
figures/review_radial_errors.{pdf,png}: same-time radial RMS profiles.
figures/review_angular_errors.{pdf,png}: same-time angular error spectra.
bridge_summary.json: all 12 control/new rows, all per-cell physical scores,
  uniform deltas and guards, argmax reference/grid/time, wake checks, work.
profiles_summary.json: exact input hashes, all profile energy fractions and
  separate physical argmax for each field.
radial_profiles.csv / angular_error_spectra.csv: every mode/node, 19 times,
  two existing references and five archived CPU trajectories (3 switched/FD
  plus two uniform controls). The profiles use the original layer144 radial
  quadrature, exact angular Parseval sums, no phase alignment. Velocity
  spectra use polar components (ur,utheta); total vector-error energy is
  invariant under the Cartesian transformation, but modal allocation is not.
physical_errors_all_times.csv: original physical-grid scores on all 3 grids.
profile_energy_summary.csv: radial-region and angular-band energy fractions.
*_switch12.npz / *_switch12.json: four new raw trajectories and metadata.
bridge_protocol.json: pre-execution configurations and source/control hashes.
verification.json: 380 Parseval-vs-physical score checks (max difference
  2.13e-17), four trajectory hashes, exact t<=12 uniform prefix, physical and
  triangle guards, alternating wake and actual nonlinear evaluation counts.

Reproduction
------------
Run from the repository/workspace root with the archived work/rfm_v* inputs:
  python work/rfm_review_v3/profiles_bridge/reproduce_analysis.py
This verifies the frozen execution sources and stored trajectory hashes,
then rebuilds the summaries/figures from existing data. It creates no new
marches or references. The initial CPU execution driver is run_bridge.py;
its first-run path was executed with four workers. A fresh integration can
be made in a clean copy of this task directory with its generated protocol,
four switch12 NPZ/JSON pairs and summaries absent. Preserve the published
artifacts before doing so; do not delete or overwrite the archive in place.

Interpretation constraints
--------------------------
1. Maxima are across references/grids/times, separately for each field. For
   high RFM41 seed7331 switch8, full u max is t72 and omega max is t60;
   Legendre36 u/omega maxima are t60/t54. Compare simultaneous profiles
   before interpreting differing rankings.
2. Error energy outside r>=4 does not prove the layer enrichment caused a
   gain. At t72 RFM41 has 61.14% of omega-error energy there, but only 1.37%
   in r>=5.9. Layer attribution requires the other controlled ablation.
3. At t12 the sin^4 transverse pulse is C3 with a possible fourth derivative
   jump. No temporal refinement series was run; no observed order or removal
   of transient order reduction is established.
4. The archived 1e-5 gate compares different implementations of the SAME
   schedule (CPU recurrence/native and GPU), not uniform versus mixed time
   discretizations. The latter uses field errors and the 0.5% per-cell guard.
   Sources: rfm_v7/cpu_sbdf2_switch_pilot.py execute() compares MODELS;
   rfm_v7/switch_qualification.py score_all() compares sol against control
   for implementation, then separate delta against baseline for guards.
5. The mid RFM velocity increase 0.029428% ->0.044672% equals 0.015244
   percentage points, about51.8% relative. In this particular seed both
   maxima are at t72. It is not alone evidence of order reduction.

Both reference families are existing hash-bound FD448/K144 runs: angular_fine
at rate384 and time_fine at rate512. The early (through t16) and late (t24..72)
saved observations are concatenated; both source trajectories start from
rest and are not newly stitched dynamical histories. Full hashes are in the
two summary JSONs and bridge_protocol.json. No analysis claims that the two
references establish exact continuum error.
