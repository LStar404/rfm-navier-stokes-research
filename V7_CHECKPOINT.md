# V7 measured checkpoint

This repository is a local staging snapshot. No remote publication is implied.

The completed 25-run study uses five predeclared RFM seeds, all seven absolute
checkpoints, three area-weighted scoring grids, and both full/near velocity and
vorticity. The 0.5% claims are relative to the frozen FD320 numerical reference.
Later FD384-to-FD448 refinement changed curl by 0.14192%, so high-resolution
reference reliability is being audited. No 0.1% speed result is claimed.

Measured end-to-end medians in seconds: CPU RFM 8.563701, CPU Legendre 8.618241,
CPU FD176 28.191620, CUDA RFM 1.303755, CUDA Legendre 1.355915. These are fresh
in-process solves including initialization and cleanup; Python imports and
driver disk caches are not reset. All slower samples are retained.

The corresponding sources and raw measurements are under `work/rfm_v7`.
`cpu_formal_output/joint_isolated.json` records the exact source and data hashes.
`outputs/RFM_NS_v7_0.5门槛与CUDA复测_阶段报告.pdf` explains limitations and theory.
The saved RFM/Legendre formal NPZs use `BufferedGalerkin` for reconstruction.
The newer recurrence class and buffered CPU/CUDA FD paths are qualified
implementation candidates, not entries in the old frozen timing table.

From the repository root, with the scientific Python environment available:

```text
python verify_snapshot.py
python verify_v7.py
python verify_v7.py --cuda
```

The optional CUDA replay expects a supported NVIDIA driver and the tested
CUDA 11.8 toolkit on Windows. CUDA_PATH may select the toolkit installation.
Native Windows AVX2 FD libraries are bundled with their C sources. The CPU
FD build command is documented in `work/rfm_v7/fd_cpu_buffered_notes.md`.
The replay includes the nonzero transverse pulse through t=8; the raw formal
study runs to t=72. Re-run full cohorts in a fresh output directory, preserving
the frozen data and predeclared protocol, with competing workloads stopped.

The 1.5 GB exploratory Python/CUDA environment, caches, account metadata,
personal paths, credentials, unrelated files and raw console logs are excluded.
