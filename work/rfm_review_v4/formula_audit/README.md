# Second-review formula and factual audit

The review cites revision `353b9cf4daafc4a87ccb752bafc3c6e7c9278a7a`. The three alleged Chinese formula errors are absent from that commit. Their normalized mathematical bodies are identical in Chinese and English, including squared magnitudes in the radial/modal norms and the preparation penalty `Delta P -` in the net-time model. No numerical formula repair is justified by the source evidence.

The correct compiled revision-3 identifiers are:

| Label | Printed equation/table | Chinese page | English page |
|---|---:|---:|---:|
| `eq:review-profile-norm` | (9.1) | 24 | 23 |
| `eq:review-modal-error` | (9.2) | 24 | 23 |
| `eq:schedule-amortization` | (10.2) | 29 | 28 |
| `eq:geometry-equivalence` | (13.1) | 32 | 31 |
| `eq:geometry-condition` | (13.2) | 32 | 31 |
| `tab:review-ablation` | Table 8 | 23 | 22 |
| `tab:review-switch12` | Table 9 | 27 | 26 |

The manuscript has 49 labeled equations in total, with section-based printed numbers. The review's sequential “47–49” and tables13/14 do not match the compiled identifiers.

## Checks and artifacts

From the repository root:

```powershell
python work/rfm_review_v4/formula_audit/verify_bilingual_equations.py
```

The lightweight check reads both current manuscript editions, recursively expands local `\input` files, locates each equation by its label, and compares its mathematical body to the hash-bound correct baseline in `canonical_equations.json`. It ignores only whitespace, comments and equation labels; it does not discard arithmetic, powers, signs or normalization factors. An intentional mathematical change requires review of the canonical baseline rather than silently accepting identical mistakes in both languages. This is a focused regression check, not a general TeX parser or theorem prover.

`shared_equations/` supplies the three already-correct blocks as a suggested common source. The parent manuscript may use its own corresponding shared files under `equations/`; the recursive checker supports that arrangement.

`checker_selftest.py` verifies four synthetic fixture outcomes: shared-source correctness, rejection of a missing square in one language, rejection of a missing preparation term shared by both languages, and acceptance after restoring the correct source. All four outcomes pass; `checker_selftest.json` preserves the receipt. In particular, matching each other is insufficient if both editions depart from the audited mathematics.

```powershell
python work/rfm_review_v4/formula_audit/audit_revision3.py
```

The historical audit reads the cited Git objects, not a web rendering. A clone must contain commit353b9cf and the preserved original `work/rfm_v8` evidence. It records source SHA-256 values, line locations, exact equation bodies, 52 relevant qualification rows and factual verdicts in `audit.json`. Local revision-3 `.aux` files are used when available; otherwise the preserved `revision3_label_map.json` supplies the compiled label map and original `.aux` hashes. Current source files may legitimately differ after revision4; that is recorded separately from the historical finding.

## Corrections to the review, not to the frozen evidence

- Final geometry is `a=0.5`, `R=6`, physical span5.5 and logarithmic extent `L=log12`; it is not outer radius6.5 or linear extent6.
- Both RFM and Legendre controls in the cited error table include all six analytic layer functions. The review's no-layer Legendre explanation is not supported by those experiments.
- Far-field features use `16z²(1−z)` and the specified boundary correction. The squared envelope belongs to another boundary construction.
- The two main paired spaces have retained ranks39/40. Removing layers changes both the space and its retained dimension; a deterministic midpoint qualifying at the same threshold does not establish universally zero benefit from random features.
- The maintained factory interface is under `work/rfm_review_v3/maintenance/`; frozen `work/rfm_ns/` files were deliberately not overwritten. Actual tests are under `maintenance/tests/test_factory.py`.
- The `<2e-10` CPU/GPU field comparison belongs to two enriched-random ablation controls. It must not be attached as a measured implementation difference to the CPU-only switch12 runs.
- At `t=8` the pulse is smooth, not at a nonsmooth endpoint. Ratios of differences between maximum errors do not identify additive causal percentages for bridge error versus accumulated time-discretization error.

No existing source, manuscript, trajectory, reference, or timing record is modified by this audit. No new PDE solve or performance claim is made.
