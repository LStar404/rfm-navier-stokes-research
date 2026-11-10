# Revision 4: saved-probe phase diagnostics and switch-error vectors

This is post-review analysis of previously stored arrays. No PDE trajectory, reference solve, GPU test, or performance measurement was added. All original physical scores retain their original time coordinates.

## Frequency and crossing diagnostics

`analyze_saved_probes.py` reads all 26 revision-3 ablation/control trajectories plus the two existing fine FD reference trajectories. Every input has 577 samples on the same `t=0:0.125:72` grid and three probes at `(1.5,0)`, `(2.5,0)`, `(3.5,0)`. Both transverse velocity and vorticity are analyzed. Hashes are checked against the previous execution records before analysis.

The primary window is `[24,72]`, with least-squares linear detrending. Upward zero crossings are located by linear interpolation, then crossing time is regressed against integer crossing index. Inverse slope gives frequency. `[36,72]` and `[48,72]`, mean-only detrending, and both fine references provide sensitivity checks, not confidence intervals. The analysis produces 1,008 frequency/crossing records and 1,872 reference contrasts. It does not infer an unbounded-cylinder Strouhal number.

The two midpoint no-layer cases have vorticity-probe frequency biases of approximately -4.193% and -3.975% at `(2.5,0)`. Their crossing delays grow to 1.75259 and 1.67221 at the matched reference crossing `t=66.797754`. These observations support accumulated phase drift as one contributor to the long-time mismatch. They do not isolate what fraction of the physical field error is phase, amplitude, or shape. The deleted analytic features represent the outer boundary; no inner-wall separation angle was measured, and retained rank changes simultaneously. The primary-window all-no-layer bias range is about -4.27% to -3.35%; the full window/detrending/reference sensitivity range is -5.20% to -1.88%.

`review_phase_probes.pdf` and `.png` show original-time vorticity waveforms and crossing delay. Crossings are paired once using the nearest initial event, then their indices stay fixed. Detrending is for event detection only. It does not shift the displayed raw traces or any qualification field.

## Switch-error vector identity

`analyze_switch_error_vectors.py` reconstructs the existing uniform, switch-at-8 and switch-at-12 fields at `T=72` on layer144 for four configurations and both references. The reconstruction builds spatial operators but never advances a time step. For both velocity and vorticity, it checks the deterministic common-denominator identity

`||e8||^2 = ||e12||^2 + 2<e12, delta> + ||delta||^2`, where `delta = e8 - e12`.

For RFM37 seed7331 velocity versus the time-fine reference, the terms are `1.9956194141e-7`, `1.9202943907e-7`, `7.4475738974e-9`, and `8.4928450335e-11`. These are unscaled squared relative norms, not percent squared. The unrounded identity residual is about `2.04e-22`. The cross term constitutes about 98.87% of the squared-norm change. The reported 5.584% recovery of the maximum velocity-error increase is merely an arithmetic ratio. Its complement cannot be assigned to an independently isolated coarse-step error.

Moving the switch changes the fine/coarse integration over four time units, the bridge's state and time, and subsequent error history. The pulse is analytic at `t=8`; `t=12` is its C3 junction. A causal bridge-only/coarse-step error partition requires further controlled interventions and timestep refinement.

## Reproduce from the repository root

```powershell
python work/rfm_review_v4/phase_diagnostics/analyze_saved_probes.py
python work/rfm_review_v4/phase_diagnostics/analyze_switch_error_vectors.py
python work/rfm_review_v4/phase_diagnostics/verify_diagnostics.py
```

Dependencies are NumPy, SciPy and Matplotlib, plus the preserved `work/rfm_v*` and revision-3 input layout. The scripts overwrite only their own derived diagnostics. Original arrays and solver sources are read-only inputs. Bilingual inserts are `phase_diagnostics_{zh,en}.tex` and `switch_attribution_{zh,en}.tex`.
