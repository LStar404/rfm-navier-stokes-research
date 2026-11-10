# RFM Galerkin Navier–Stokes paper — revision 2

修订日期：2026-11-10。中文全文31页，英文全文29页；两份稿件均含4幅图、12张表、44个带标签公式及19条文献。作者和单位保留空白。本文定位为数值计算与应用高性能计算研究，尚未证明稳定超过同等优化传统CFD，也没有完成先进PINN的同门槛计时。

Revision date: 2026-11-10. The Chinese manuscript has 31 pages and the English manuscript 29 pages. Each contains four figures, twelve tables, 44 labeled equations, and 19 references. Author and affiliation fields remain unset. This is a numerical-computation and application-HPC study; it does not establish a stable speed advantage over comparably optimized CFD or a matched-accuracy timing comparison with advanced trained PINNs.

## What changed / 主要修订

- Expanded the feature construction, differential SVD normalization, weak forms, mean block, boundary lifting, SBDF2 bridge, and phase-coordinate proofs.
- Added source-level FFT packing, normalization, layout, shared-memory synchronization, nominal memory-access accounting, warp observations, CPU C kernels, and amortization analysis.
- Added controlled-comparison and hardware tables, seed/rank attribution, tolerance margins, and precise treatment of rejected timing campaigns.
- Corrected the distinction between 19 physical scoring times and two extra implementation checkpoints, and between 39 configurations and 78 cache-policy trajectories.
- Extracted a standalone operator example and verified 36 synthetic fixed-state cases on the local GPU plus nine malformed-input rejection cases. These checks are separate from the frozen Navier–Stokes results. They add no solver timing result.

数学、算子和运行工程贡献已分别说明；保留未成功分支及证据边界。原版本另备份在工作目录的 `work/rfm_paper/revision_v1/`，不打包到本文源文件。

## Manuscript build / 论文构建

`main.tex` 为中文全文。`english.tex` 为英文全文。两份稿件采用通用单栏论文版式。

`main.tex` is the Chinese manuscript and `english.tex` is the full English manuscript. Both use a general single-column article layout.

在项目目录下分别编译。中文依赖 `ctex` 和 Fandol 字体。两份稿件共用 `references.bib` 和 `figures/`，误差表位于 `tables/`。

Build either entry point from the project directory. The Chinese edition uses `ctex` with the Fandol fonts. Both editions share `references.bib` and `figures/`. The corresponding accuracy tables are under `tables/`.

```sh
tectonic main.tex
tectonic english.tex
```

```sh
latexmk -xelatex main.tex
latexmk -xelatex english.tex
```

Numerical records and figures correspond to archive commit `6f2b3500964d3d5153b8400018236c4702ef94f8`. Appendix C identifies the evidence files in that archive.

The statement above applies to the archived Navier–Stokes data and its three original figures. The new dataflow diagram and standalone correctness checks are identified separately. `supplement/evidence_revision.json` records numerical provenance; it is not a replacement for the restricted archive's raw flow arrays.

## Standalone operator / 独立算子示例

See `supplement/operator_sample/README.md`. The folder contains the original project-local CUDA kernel and CPU model, a new solver-independent NumPy/CUDA verification harness, the actual validation receipt, and copy hashes. From that folder:

```powershell
python verify.py --output cpu_validation.json
python verify.py --gpu --cuda-path 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8' --output validation.json
```

The ordinary cases are near FP64 roundoff. A field-scale imbalance of `1e10` exposes a weak-output relative error of approximately `4.97e-6`; the new input guard rejects such nonzero packed pairs by default. This is a conservative policy, not an error theorem or an implemented automatic fallback.

该示例是本地候选，不是已提交PR。还缺少等价融合cuFFTDx、分批cuFFT与本地核的同精度性能比较，以及外部干净环境的构建验证。本包不作新的授权许可或DCO声明。

This is a local contribution candidate, not a submitted PR. Equivalent cuFFTDx/cuFFT performance comparisons and a clean external build remain outstanding. No new license grant or DCO certification is made by this package.
