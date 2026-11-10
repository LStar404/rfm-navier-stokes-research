# RFM for incompressible Navier–Stokes

<!-- paper-revision-3 -->
## 最新审稿修订 / Reviewer revision 3 (2026-11-10)

- [中文论文（38页）](outputs/RFM_NS_中文论文.pdf)
- [English manuscript (36 pages)](outputs/RFM_NS_English_Manuscript.pdf)
- [中英逐条审稿回复（5页） / Point-by-point response](outputs/RFM_NS_审稿回复.pdf)
- [完整 LaTeX 源码包](outputs/RFM_NS_LaTeX.zip) · [构建说明](outputs/rfm_ns_tex/README.md)
- [本轮实验、原始轨迹与维护源码](work/rfm_review_v3/README.md)
- [本次上传文件与校验和](PAPER_REVISION_MANIFEST.json)

本轮新增20条CPU T72轨迹：16条特征消融/复算和4条延后换步对照。固定频率中点加解析层的两档配置均达标；去掉非零模层的12条轨迹均精度失败，但实际秩也下降，因此不是等秩归因。新增同一时刻的径向/角谱误差图、复杂几何边界讨论、条件摊销模型，并用7项CPU测试验证旧稳态求解器的显式工厂注入。两份全文各含6幅图、14张表、49个带标签公式和20条文献。

The revision adds actual feature/switch controls and error decompositions, while correcting factual transcriptions in the review. All new parallel CPU durations remain diagnostic. The original 78 GPU qualification trajectories and rejected final timing campaign are unchanged; no new isolated speed ranking, trained-PINN comparison, or upstream library PR is claimed.

`MANIFEST.json` remains the frozen v8 inventory. `work/rfm_review_v3/revision_inventory.json` binds the new evidence; `PAPER_REVISION_MANIFEST.json` inventories the current manuscript upload. The previous standalone GPU operator example remains available under [outputs/rfm_operator_sample](outputs/rfm_operator_sample/README.md). Original source hashes apply to their original bytes; existing redaction maps still apply to any separately named historical derivatives.
<!-- /paper-revision-3 -->

This snapshot contains the verified v6 CPU baseline for the full-annulus Re=100 circular-cylinder wake and the subsequent parallel accuracy refinement. The cylinder radius is 0.5 and the outer radius is 6. The calculation retains both angular parities and the angular-mean tangential momentum. The outer boundary prescribes normal velocity and zero vorticity. A transverse pulse acts for (4<t<12), and the runs start from rest and continue to (t=72).

## Evidence

- [Main report](outputs/RFM_NS_v6_全域涡脱落复测.pdf)
- [Parallel refinement addendum](outputs/RFM_NS_v6_并行加密补充.pdf)
- [Frozen timing specifications](work/rfm_v6/full_specs.json) and [raw timing and error data](work/rfm_v6/full_formal.json)
- [Parallel refinement data](work/rfm_v6/parallel_refinement/parallel_refinement.json) and [original-run reassessment](work/rfm_v6/parallel_refinement/transfer_refinement.json)

The formal CPU baseline supports matched accuracy gates of 1% and 2% for both velocity and vorticity. The parallel refinement supplies additional results below 0.5% against a finer reference. Its CPU timings were collected concurrently and cannot be used in the isolated speed comparison. The original timed FD168 run misses 0.5% on the finer reference.

The empirical reference screen compares successive discrete solutions. It supplies evidence about resolution without serving as a rigorous bound on the continuum error. Gate scores use absolute checkpoint times and contain no phase alignment. The isolated Galerkin RFM and Legendre cases share the radial enrichment and the same operator implementation.

## Verify and reproduce

Run from the repository root in Python 3.13.5 with the packages listed in `requirements.txt`. The formal CPU-affinity runner and supplied FD DLL are Windows-specific. Rendering the original PDF scripts requires the Windows fonts in `work/report_v2/layout.py`.

```powershell
python -m pip install -r requirements.txt
python verify_snapshot.py
python verify_snapshot.py --smoke
```

The smoke run advances all three methods from rest to (t=0.5) and checks finite, nonzero states and boundary residuals. It is a packaging check. It does not cover wake shedding or reproduce the full timing comparison.

The full comparison creates new output and leaves the frozen evidence in place.

```powershell
New-Item -ItemType Directory -Force replay | Out-Null
python work/rfm_v6/full_formal.py --reference work/rfm_v6/fd_full_n240_k120_rate320_outer1.npz --predecessor work/rfm_v6/fd_full_n200_k144_rate256_outer1.npz --output replay/full_formal.json
python work/rfm_v6/diagnostic_figure.py work/rfm_v6/fd_full_dense_from54.npz --history work/rfm_v6/fd_full_n200_k144_rate256_outer1.npz --plot replay/wake.png --metrics replay/wake.json
```

Reference-generation commands, the portable FD backend, and numerical conventions are documented in the [v6 reproducibility notes](work/rfm_v6/README.txt). To regenerate the parallel batch from scratch, run `python work/rfm_v6/parallel_refine.py --workers 16` in a disposable copy with the supplied `work/rfm_v6/parallel_refinement/` directory moved out first. The script otherwise reuses its saved checkpoints by filename. Run `python work/rfm_v6/transfer_refinement.py` after the parallel batch to re-score the original timing configurations.
