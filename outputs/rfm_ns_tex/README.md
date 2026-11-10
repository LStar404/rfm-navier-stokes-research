# RFM Galerkin Navier–Stokes paper — revision 3

修订日期：2026-11-10。中文38页，英文36页，另有5页中英逐条审稿回复。两份全文各含6幅图、14张表、49个带标签公式和20条文献。作者和单位留空，未指定投稿期刊。

The revised Chinese and English manuscripts have 38 and 36 pages, with a separate five-page bilingual point-by-point response. Each full manuscript contains six figures, fourteen tables, 49 labeled equations, and twenty references. Author/affiliation and journal fields remain unset.

## Revision 3 evidence

- Added 16 CPU T72 ablation/control trajectories and four delayed-switch trajectories. Ten frozen enriched-random GPU controls were independently rescored; the original 78 GPU qualification records are unchanged.
- Both fixed-midpoint frequency configurations pass with analytic layers. All twelve no-layer trajectories fail accuracy, but removing layers also reduces retained rank. The conclusion does not assign a causal percentage to layers or randomness.
- Added matched-time radial and polar-component angular error profiles, with 380 Parseval checks, and a t=12 versus t=8 sensitivity comparison. The latter improves margins modestly at 64 extra nonlinear evaluations per trajectory.
- Expanded nonseparable-geometry complexity/conditioning limits and a conditional preparation-cost crossover model. The latter is not an observed 15,000-step performance threshold.
- Added an explicit solve-local basis factory for the early steady solver. Seven CPU tests pass, including eight legacy-equivalence configurations and concurrent/exception isolation.
- Corrected the review's geometry, frequency, boundary-envelope, retained-rank and FFT-unit transcriptions. No new isolated GPU timing, full PINN comparison, or third-party-library PR is claimed.

## Build

`main.tex` is Chinese, `english.tex` is English, and `response.tex` is the bilingual response. Compile from this folder with Tectonic or XeLaTeX/latexmk. Chinese documents require ctex and Fandol. All three use `common.tex`; manuscripts share the bibliography and figures.

```sh
tectonic main.tex
tectonic english.tex
tectonic response.tex
```

```sh
latexmk -xelatex main.tex
latexmk -xelatex english.tex
latexmk -xelatex response.tex
```

## Reproduction and provenance

The original numerical evidence remains tied to archive commit `6f2b3500964d3d5153b8400018236c4702ef94f8`. Appendix C and `supplement/evidence_revision.json` identify its source records. The new experiments live separately at `work/rfm_review_v3/` in the repository; that directory contains the raw arrays, protocols, scripts, all field scores and subdirectory reproduction instructions. The LaTeX archive includes selected summary receipts and the inventory under `supplement/review_v3/`; it does not duplicate the complete flow archive.

`supplement/revision_manifest.json` records the final PDFs and source hashes. It excludes itself from its hash list. Frozen evidence is not overwritten, and failed accuracy cases are retained. Concurrent CPU elapsed times are diagnostics, not performance measurements. The repository is publicly accessible.

## Standalone operator from revision 2

The previous `supplement/operator_sample/` remains available with the project-local CUDA kernel, independent verification harness, 36 actual GPU correctness cases, and nine invalid-input checks. Its findings concern fixed-state correctness, including weak-channel sensitivity, not a new full-flow speed result. See its README for commands and outstanding equivalent cuFFTDx/cuFFT comparisons. No new license grant or DCO certification is made.
