# Second-review revision: factual audit and saved-data diagnostics

Revision 4 responds to the review of commit `353b9cf4daafc4a87ccb752bafc3c6e7c9278a7a`. It does not integrate new Navier–Stokes trajectories, create reference solutions, run new GPU experiments, or establish a new timing result. The original 78 GPU qualification runs and the 20 CPU trajectories added in revision 3 remain unchanged.

## What was verified and added

| Review topic | Evidence and response |
|---|---|
| Three alleged Chinese formula mistakes | All three were already correct in the reviewed commit. Both editions now share equation files, checked against their audited mathematical definitions. |
| Long-time phase drift without layers | Saved crossing events show frequency mismatch and increasing lag. The main midpoint-control biases are about −4.19% and −3.98%, with explicit sensitivity ranges. This does not identify inner-wall separation or explain all field error as phase. |
| Unenriched Legendre explanation | The premise is false: both RFM and Legendre controls contain the same six analytic outer layers and use common quadrature. The removal ablation also changes retained rank. |
| 5.6% bridge versus 94.4% coarse-step error | The percentage is a descriptive norm-increment ratio, not a causal budget. A same-field squared-error identity explicitly retains its cross term. |
| Geometry and PCG complexity | Pure radial reparameterization preserves modal blocks under angularly invariant boundary operators. A conditional SPD PCG bound does not force a particular algorithm or an FFT at every product. |
| FD4 wall formula | The actual baseline uses nonuniform logarithmic nodes and recomputed Hermite weights. The uniform-grid formula is presented only as a special case, including its general wall-value term. |

## Reproduction from the repository root

The existing `work/rfm_v*` and `work/rfm_review_v3` data are required. Each subdirectory has a detailed README with definitions, references, source hashes, and scope.

```powershell
python work/rfm_review_v4/formula_audit/verify_bilingual_equations.py
python work/rfm_review_v4/formula_audit/checker_selftest.py
python work/rfm_review_v4/fd_geometry/verify_fd_closure.py
python work/rfm_review_v4/phase_diagnostics/verify_diagnostics.py
```

The first check expands the current shared LaTeX definitions and compares them with the already-correct reviewed mathematics. The FD command constructs spatial operators only. The probe verifier independently checks saved event fits and error identities; the analysis scripts rebuild the descriptive tables/figure from stored arrays without marching the flow. The historical Git audit additionally requires the reviewed commit object.

`revision_inventory.json` binds this revision's content. It excludes itself and Python bytecode caches. The manuscript's supplementary folder contains selected receipts; the full diagnostic CSV/JSON outputs and scripts are retained here. Physical scores remain at their original times, without phase alignment. Frequencies describe the finite annulus, not a new unbounded-cylinder Strouhal benchmark.

中文：这轮修订纠正了审稿中的若干误读，并补充可验证的数学与诊断说明；没有把不存在的公式错误记为已修复，也没有将相关性、范数差或条件性理论界写成已证明的物理因果结论或整体加速结论。
