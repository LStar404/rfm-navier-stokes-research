# RFM for incompressible Navier–Stokes

<!-- paper-revision-2 -->
## 最新论文 / Revised manuscripts (2026-11-10)

- [中文论文（31页）](outputs/RFM_NS_中文论文.pdf)
- [English manuscript (29 pages)](outputs/RFM_NS_English_Manuscript.pdf)
- [完整 LaTeX 源码包 / LaTeX source archive](outputs/RFM_NS_LaTeX.zip)
- [在线阅读源码与构建说明](outputs/rfm_ns_tex/README.md)
- [独立 GPU 算子示例](outputs/rfm_operator_sample/README.md) · [下载](outputs/RFM_NS_Operator_Sample.zip)
- [本次上传文件与校验和](PAPER_REVISION_MANIFEST.json)

中英文稿分别包含4幅图、12张表、44个带标签公式和19条文献。扩充了数学推导、底层CUDA/C实现、同精度比较与失败证据。本轮新增36个GPU固定态算子用例及9项非法输入检查；没有新增完整NS轨迹或计时排名。尚未证明相对同等优化传统CFD的稳定整体优势，独立算子示例也未提交上游PR。

Both manuscripts expand the numerical derivations, low-level CUDA/C implementations, matched-accuracy controls, and limitations. The new standalone checks are separate from the frozen NS evidence; no new full-flow timing result or upstream PR is claimed.

`MANIFEST.json` retains the frozen v8 inventory. `PAPER_REVISION_MANIFEST.json` inventories this manuscript addition; the underlying v8 evidence is unchanged. Supplemental evidence hashes identify the original source bytes used in the paper audit. Where the publication uses separately named redacted derivatives, consult the existing v8 path-redaction map rather than treating an original hash as authentication of different bytes.
<!-- /paper-revision-2 -->

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
