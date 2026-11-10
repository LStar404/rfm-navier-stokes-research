# RFM Galerkin Navier–Stokes paper — revision 4

修订日期：2026-11-10。中文42页，英文39页，另有4页中英逐条二审回复。两份全文各含7幅图、15张表、54个带标签公式和21条文献。作者和单位留空，未指定投稿期刊。

The Chinese and English manuscripts have 42 and 39 pages, with a separate four-page bilingual response to the second review. Each full manuscript contains seven figures, fifteen tables, 54 labeled equations, and twenty-one references. Author/affiliation and journal fields remain unset.

## Revision 4: verified corrections and additions

- The three alleged Chinese formula errors are absent from the reviewed commit `353b9cf`. Both editions now share the same equation sources, checked against the audited mathematics. Four checker fixtures include identical corruption of both languages, which must be rejected.
- Saved-probe diagnostics show about −4.19% and −3.98% frequency biases for the main deterministic no-layer controls, with explicit lag and sensitivity analyses. They do not identify an inner-wall separation mechanism or explain all field error as phase. Qualification still uses the original checkpoint times, without phase alignment.
- Both Legendre and RFM contain the same six analytic outer layers and share quadrature. The paper corrects the review's contrary premise and its boundary-envelope interpretation. Removing layers also changes retained rank.
- A same-field squared-error identity retains the cross term. The descriptive 5.6% recovery ratio does not establish a 5.6%/94.4% causal bridge/coarse-step decomposition.
- The actual FD4 baseline uses nonuniform logarithmic nodes and recomputed Hermite weights. The constant-coefficient uniform-grid formula is only a special case; its general wall-value term and truncation-error sign are included.
- Radial reparameterization and angle-dependent coupling are distinguished. The PCG convergence bound is conditional on SPD operators, a fixed SPD preconditioner, and spectral equivalence; it does not mandate PCG or an FFT at every matrix product.
- The CPU–GPU discrepancy bound applies only to the two supported ablation controls. It is not assigned to CPU-only delayed-switch trajectories.

No new PDE integration, reference solution, GPU run, or isolated timing was performed in revision 4. The original 78 GPU qualification trajectories and revision 3's 20 CPU trajectories are unchanged. No new speed ranking, trained-PINN comparison, or upstream library PR is claimed.

## Build

`main.tex` is Chinese, `english.tex` is English, and `response.tex` is the bilingual response. Compile from this folder with Tectonic or XeLaTeX/latexmk. Chinese documents require ctex and Fandol. All three use `common.tex`; manuscripts share the bibliography, figures, and the three audited equations.

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

Original numerical evidence remains tied to archive commit `6f2b3500964d3d5153b8400018236c4702ef94f8`. Appendix C and `supplement/evidence_revision.json` identify its source records. Revision 3's experiments remain separately at `work/rfm_review_v3/` in the repository, including raw arrays and its maintained basis-factory package. Frozen historical wrappers have not been overwritten.

Revision 4's audit, saved-array diagnostics, spatial-operator checks, and reproduction commands are at `work/rfm_review_v4/`. Selected receipts and inventories are included under `supplement/review_v3/` and `supplement/review_v4/`; the source ZIP does not duplicate the full flow archive. The mathematical equations in `equations/` are the single manuscript source for both languages.

`supplement/revision_manifest.json` records final PDFs and source hashes, excluding itself from its source hash list. Frozen evidence and failed cases remain intact. The repository is public. A review recommendation is not a journal acceptance.

## Earlier standalone operator and maintenance work

The previous `supplement/operator_sample/` remains available with the project-local CUDA kernel, independent verification harness, 36 GPU fixed-state correctness cases, and nine invalid-input checks. Its findings concern fixed-state correctness, including weak-channel sensitivity, rather than a new full-flow speed result. Revision 3's seven CPU maintenance tests cover eight legacy-equivalence configurations and concurrent/exception isolation. These checks were not rerun as new experiments in revision 4. No new license grant or DCO certification is made.
