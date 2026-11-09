# Operator and step-rate checkpoint

This additive checkpoint keeps the earlier 521-file release intact. The new report is `outputs/RFM_NS_v7_算子改写与步长消融_研究报告.pdf`. The previous full report remains the source for the three-scale GPU study and the supplementary smooth-lid cavity experiment.

The new work contains:

- A precomputed Galerkin projection/update with mode-contiguous C/AVX2 kernels, qualified on all five registered RFM seeds and the Legendre control.
- Separate step-rate screening at 80, 96, 112 and 128. The selected Galerkin trajectories use 112; FD176 stays at 128. Failed screens and three independent CPU failure controls are included.
- Five randomized ten-arm blocks that separate fixed-rate CPU operator changes from the CUDA step-rate change. Both the first 50-run cohort and a separate full 50-run replication encountered external computation and are preserved as diagnostic timing data. The replication uses the same configurations and schedule, with one preregistered 600-second external-activity observation window for the entire cohort. The journals contain every run; the cohorts are not pooled into ten repeats. Neither cohort supports a new isolated speed claim. Numerical acceptance and exact FFT work counts remain valid.
- Power-of-two-scaled high-resolution CUDA FD solves: the dense trajectory fails the implementation tolerance; the band solve passes. Both are retained.
- A k=1 row-moment derivative evaluation, independent CPU and GPU full trajectories, fixed-state high-precision comparisons, and the local algebra/roundoff derivation. This is a reliability change, not a claimed overall PDE accuracy gain or a timing victory.

## Verification

Run from the repository root, with the existing scientific environment installed. The recorded environment is Windows x64, Python 3.13, NumPy 2.4 and SciPy 1.16. Native CPU binaries require AVX2. The CUDA path uses the existing CUDA/cuFFT environment described by the previous release; it is not a PyTorch training dependency.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B work/rfm_v7/verify_v7_operator_round.py
python -B work/rfm_v7/verify_v7_operator_round.py --replay
```

The default verification checks frozen hashes, both 50-trajectory cohorts, randomized schedules and paired statistics, then independently recomputes 1,008 score values from six representative replication trajectories against both references. It also recomputes the accepted and rejected FD CUDA comparisons and the external-load rejection. `--replay` additionally runs six short trajectories (RFM/Legendre, original CPU/new CPU/CUDA). These short runs verify implementation consistency; they do not replace the full t=72 physical-accuracy acceptance.

The repository contains frozen results. Qualification and timing drivers intentionally reuse or reject existing records, rather than overwriting them. Executing them in this archived checkout does not create a new independent timing study. New measurements require a separate output directory and a newly frozen protocol, with the original files retained.

The C sources and build provenance are included. Rebuilding a DLL changes its hash and therefore requires a new qualification record; keep the bundled binary and original evidence intact. The provided build helper is specific to the recorded Windows compiler path and will need configuration on another machine. Font files used for the PDF are system resources and are not redistributed.

## Interpretation limits

The case is a two-dimensional finite annulus at Re=100, not an unlimited exterior-flow domain. Every physical score is evaluated at unshifted absolute times. All current time integrators are SBDF2; “fourth order CFD” refers to the radial finite-difference stencil.

The 0.5% comparison uses fixed velocity and vorticity criteria on three grids, seven times, and full/near-field regions. The archived independent-axis reference chain does not yet satisfy the 0.1% reference budget. CPU/GPU agreement at the same configuration is not an independent continuum-accuracy certificate.

Legendre-Galerkin is itself a deterministic spectral CFD method. Shared basis enrichment, weak discretization, precontraction, and CUDA implementation must not be attributed solely to randomized features. The matched input dimensions do not imply equal retained rank.

The row-moment identity uses the actual stored stencil coefficients and offsets, including their nonzero discrete moments. Its fixed-state improvement does not remove input quantization or prove an unconditional long-time stability result. All failed implementations and qualifying limitations are retained in the report and raw records.

`theory_sbdf2_scalar_stability.txt` derives an exact Schur boundary for the constant-coefficient scalar advection/diffusion test equation. `audit_sbdf2_scalar_stability.py` checks characteristic roots at 80 decimal digits. The scalar result does not cover the noncommuting, nonlinear, nonnormal operators of the full cylinder problem.
