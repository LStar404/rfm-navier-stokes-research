# Reviewer-requested feature ablation

This is an accuracy-only supplement. It does not modify frozen v6–v8 code or extend the original 78 GPU qualification runs. The execution plan and source hashes were written to `protocol.json` before any T72 job ran.

- Two original configurations: `(m,K,q,rate)=(37,72,66,128)` and `(41,80,76,152)`.
- Registered seeds: 7331, 7687, 8011, 8353, 9041.
- Factors: paired-frequency uniform jitter versus the fixed midpoint; six analytic nonzero-mode layers present versus absent.
- Mean mode always retains its 24 base functions and six layer functions. It has rank 30 in all cells.
- Every cell uses the original quadrature, bandwidth 62, SVD relative cutoff `1e-10`, nonlinearity, time integration, forcing, and observation schedule.
- Six concurrent processes maximum, each with BLAS/OpenMP threads fixed to one. Elapsed seconds are diagnostics and must not be compared as performance evidence.

There are 16 new CPU trajectories (14 ablation trajectories and two random-enriched controls). Ten original random-enriched GPU trajectories are reused and rescored. The deterministic midpoint does not depend on the seed, so there is one trajectory per deterministic cell, not five fictitious repeats.

All 26 trajectories are scored at 19 physical times on three grids against two frozen FD references. Each score contains full-domain and near-region velocity/vorticity errors. There are 2,964 score rows and 11,856 scalar errors. The gate is 0.5% for every metric. All 16 new numerical runs completed; the 12 without nonzero-mode enrichment failed the accuracy gate. Their failed accuracy results remain in the archive.

The midpoint-plus-layer cells pass at both resolutions, with errors comparable to the random enriched cells. This does not support claiming that randomness is necessary. Removing nonzero-mode layers lowers retained rank from 39/40 to 34/35. Therefore this fixed-raw-width experiment cannot separate enrichment geometry from effective dimension, or assign a causal percentage to the layers.

A subsequent construction-only audit increased deterministic raw width to 41, 45, 49, 53, 57, and 61 at fixed bandwidth/cutoff/quadrature. Both resolutions retained ranks 35, 35, 36, 36, 36, and 36. No exactly equal-rank candidate was found in this bounded list, so no additional T72 solve was run. The direct SVD diagnostic intentionally does not use the old constructor's `m+6<q` guard: with nonzero layers absent, that guard is overly restrictive for the last mid-resolution diagnostic. This does not change any production solver.

## Files

- `protocol.json`: immutable job list, parameters, source hashes and archived-input hashes.
- `ablation_model.py`: instance-local subclass changing only frequency jitter and nonzero-mode columns.
- `runs/*/trajectory.npz`: new CPU states, probes, model parameters and metadata.
- `runs/*/result.json`: execution status, ranks, singular values, hashes and diagnostic elapsed seconds.
- `summary.json`: every physical score, failure and CPU/GPU control difference.
- `cell_summary.json`: eight table cells and worst-case errors; controls are separate.
- `adapter_verification.json`: bitwise checks of the unchanged mean blocks, baseline operators, nonlinear terms and implicit step; independence of midpoint columns from unused seeds.
- `coverage_verification.json`: coverage, source preservation and reproduction of archived score maxima.
- `capacity_construction.json`: bounded post-ablation construction-only rank check, including all singular values.
- `ablation_zh.tex`, `ablation_en.tex`: directly includable manuscript sections.

## Reproduce the audit from the repository root

```powershell
python work/rfm_review_v3/ablation/verify_adapter.py
python work/rfm_review_v3/ablation/run_ablation.py assess
python work/rfm_review_v3/ablation/check_capacity.py
python work/rfm_review_v3/ablation/finalize_ablation.py
```

The execution command is `python work/rfm_review_v3/ablation/run_ablation.py run --workers 6`. It resumes only missing jobs and never overwrites a completed run, including failed attempts. `freeze` refuses to replace an existing protocol.

For a genuinely new numerical reproduction while preserving all archived attempts, make a separate sibling copy containing only the scripts and the frozen protocol (no `runs` folder), then run there. The original source files remain available for the registered hash check:

```powershell
$replica = 'work/rfm_review_v3/ablation_reproduction'
New-Item -ItemType Directory -Path $replica -ErrorAction Stop
Copy-Item -Path 'work/rfm_review_v3/ablation/*.py' -Destination $replica
Copy-Item -LiteralPath 'work/rfm_review_v3/ablation/protocol.json' -Destination $replica
python work/rfm_review_v3/ablation_reproduction/run_ablation.py run --workers 6
python work/rfm_review_v3/ablation_reproduction/run_ablation.py assess
python work/rfm_review_v3/ablation_reproduction/verify_adapter.py
python work/rfm_review_v3/ablation_reproduction/check_capacity.py
python work/rfm_review_v3/ablation_reproduction/finalize_ablation.py
```

`New-Item -ErrorAction Stop` prevents an existing reproduction directory from being silently reused. New timestamps and diagnostic elapsed times will differ. Compare physical fields, scores and coverage rather than NPZ whole-file hashes across separate reproductions.
