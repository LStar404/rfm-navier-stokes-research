# 不可压缩 Navier–Stokes 可复核研究

<!-- paper-revision-2 -->
## 最新论文 / Revised manuscripts (2026-11-10)

- [中文论文（31页）](../outputs/RFM_NS_中文论文.pdf)
- [English manuscript (29 pages)](../outputs/RFM_NS_English_Manuscript.pdf)
- [完整 LaTeX 源码包 / LaTeX source archive](../outputs/RFM_NS_LaTeX.zip)
- [在线阅读源码与构建说明](../outputs/rfm_ns_tex/README.md)
- [独立 GPU 算子示例](../outputs/rfm_operator_sample/README.md) · [下载](../outputs/RFM_NS_Operator_Sample.zip)
- [本次上传文件与校验和](../PAPER_REVISION_MANIFEST.json)

中英文稿分别包含4幅图、12张表、44个带标签公式和19条文献。扩充了数学推导、底层CUDA/C实现、同精度比较与失败证据。本轮新增36个GPU固定态算子用例及9项非法输入检查；没有新增完整NS轨迹或计时排名。尚未证明相对同等优化传统CFD的稳定整体优势，独立算子示例也未提交上游PR。

Both manuscripts expand the numerical derivations, low-level CUDA/C implementations, matched-accuracy controls, and limitations. The new standalone checks are separate from the frozen NS evidence; no new full-flow timing result or upstream PR is claimed.

`MANIFEST.json` retains the frozen v8 inventory. `PAPER_REVISION_MANIFEST.json` inventories this manuscript addition; the underlying v8 evidence is unchanged. Supplemental evidence hashes identify the original source bytes used in the paper audit. Where the publication uses separately named redacted derivatives, consult the existing v8 path-redaction map rather than treating an original hash as authentication of different bytes.
<!-- /paper-revision-2 -->

**最新入口：[v8：算子改良与同精度公平比较](../START_HERE_v8.md)。**

本仓库保存 Re=100 有限完整圆环绕流中的 RFM、Legendre–Galerkin 与径向四阶差分研究代码、理论、冻结协议、原始数据和失败记录。

v8 的 78/78 条 T72 数值资格通过原有门槛；最终强对照批次的正式计时接受数为 0。三次未通过接受条件的正式尝试均保留，本轮不形成新的隔离速度排名，也不据此宣称 RFM 稳定超过传统 CFD。数值资格、局部优化和完整公平计时的边界见最新入口。

- [最终研究报告](../outputs/RFM_NS_v8_算子改良与同精度公平比较_研究报告.pdf)
- [正式恢复分支最终结案](../work/rfm_v8/rank_fair_cache_recovery_formal_final_closure/final_closure.json)
- [完整文件清单](../MANIFEST.json) · [公开派生文件映射](../work/rfm_v8/publication_path_redaction_map.json)
- [上一轮 GPU 准备优化入口](../START_HERE_prepared_cuda.md) · [历史入口备份](../publication_history/v8_baseline_dcb69a05c60a4d2c18422b9018a8421145ced190/github_README.md)

旧报告和协议保留各自时间点的结论。公开派生文件具有独立哈希，不能替代原件的哈希证明；新的执行需在本机重新确认环境和协议，不能覆盖或拼接旧实验。
