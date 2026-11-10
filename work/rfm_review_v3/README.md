# Reviewer revision 3: additional evidence and maintenance

This directory adds evidence requested in the review without modifying the frozen v6–v8 sources or original 78 GPU qualification trajectories. The bilingual manuscripts and point-by-point response are under `outputs/rfm_ns_tex/`; rendered PDFs are under `outputs/`.

## Actual new work

| Work | New full NS trajectories | Main result | Limitation |
|---|---:|---|---|
| Frequency-jitter × analytic-layer ablation | 16 CPU, including 2 controls | Fixed midpoints with layers pass at both resolutions; all 12 no-layer cases fail accuracy | Removing layers also decreases retained rank; one deterministic frequency design |
| Switch time 12 versus 8 and uniform controls | 4 CPU | Small error reduction at 64 additional nonlinear evaluations per run | One RFM seed per resolution; no new GPU timing |
| Radial/angular error profiles | 0 | Common-time comparison and error-energy localization | Descriptive decomposition of evolved errors, not a unique causal explanation |
| Solve-local basis factory | 0 | 7 CPU tests pass, including 8 legacy-equivalence cases and concurrent/exception isolation | Applies to the early steady solver, not the final annulus GPU path |

All 20 new trajectories reach T=72. No new reference solution was generated. Field scoring uses the frozen angular/time-refined FD references, three grids, and 19 physical checkpoints. These are not independent solver families or certified continuum solutions. CPU workers ran concurrently; their elapsed times are diagnostics only. The original final isolated timing campaign remains rejected, and no new overall speedup is reported.

## Reproduction

- `ablation/README.md`: fixed protocol, execution/resumption, full rescoring, adapter checks, capacity diagnostics, and all 16 new trajectories.
- `profiles_bridge/README.txt`: four new switch trajectories, two publication figures, CSV data, reference/control hashes, and an analysis-only reproduction command.
- `maintenance/README.md`: maintained API, scope, compatibility tests, and factual audit of the review.

Use the preserved repository-relative layout with the original `work/rfm_v*` archive. The subdirectory instructions distinguish rereading stored trajectories from executing a new solve. Make fresh execution outputs in a separate copy rather than overwriting published evidence. The revision inventory records the new files and their hashes; the older root `MANIFEST.json` remains the historical v8 inventory.

## Conclusions changed by this revision

The present enriched configurations do not require random frequency jitter to meet the field tolerance. The layer-removal experiment shows sensitivity to enrichment and effective dimension together, not a dimension-controlled causal percentage. Moving the bridge to the pulse endpoint improves margins only modestly and does not recover constant-step accuracy. The geometry, frequency formula, boundary envelope, actual retained ranks, and FFT counting units follow canonical source and metadata rather than transcription errors in the review.

没有新增隔离 GPU 计时，没有确认约 15,000 步后必然出现性能优势，也没有提交第三方库 PR。新增结果收紧了随机性的贡献主张，不补齐最终性能排序或复杂非可分几何验证。
