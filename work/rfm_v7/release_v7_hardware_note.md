# V7 equilibrated reference and hardware comparison

The six-way timing cohort contains five isolated randomized blocks, with RFM37,
Legendre37 and FD176 on both CPU and resident native CUDA Graph. All 30 solves
passed the common 0.5% full-field and near-field velocity and vorticity gate
on seven fixed physical times and three scoring grids. CPU and GPU FD use the
same FD4/SBDF2 equations with independently checked band and dense solvers.

Four row-equilibrated references give an independent-axis combined difference
of 0.0290144%. This passes the 0.125% reference budget for the 0.5% comparison.
It exceeds the 0.025% budget for 0.1% and supports no 0.1% timing claim. The
unscaled reference trajectories and their failed audit are also preserved.
The scale and graph timing studies are implementation diagnostics and separate
from the five-seed matched-accuracy comparison.

Use the saved fields to verify the published reference and error records:

```text
python work/rfm_v7/verify_v7_hardware.py
python work/rfm_v7/verify_released_cavity.py
```

The first command scores six representative cylinder trajectories from disk,
without integrating in time. For a short CUDA solver replay on compatible
Windows/NVIDIA hardware, run `python verify_v7.py --cuda` separately.

The historical v4 smooth-lid cavity archive and its original freeze contained
machine-specific dependency paths. The release provides sanitized derived
metadata, the original SHA-256 provenance, the 70 checked score records and the
ten saved first-seed field arrays. The released verification checks the
published summaries and re-scores those ten fields. It cannot re-hash the
excluded v4 originals or reconstruct unsaved historical fields.
