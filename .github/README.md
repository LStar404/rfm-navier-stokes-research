# 不可压缩 Navier–Stokes 可复核研究

<!-- paper-revision-4 -->
## 最新审稿修订 / Second-review revision 4 (2026-11-10)

- [中文论文（42页）](../outputs/RFM_NS_中文论文.pdf)
- [English manuscript (39 pages)](../outputs/RFM_NS_English_Manuscript.pdf)
- [中英逐条二审回复（4页） / Point-by-point response](../outputs/RFM_NS_审稿回复.pdf)
- [完整 LaTeX 源码包](../outputs/RFM_NS_LaTeX.zip) · [构建说明](../outputs/rfm_ns_tex/README.md)
- [本轮核验、保存数据诊断与复现说明](../work/rfm_review_v4/README.md)
- [上一轮20条CPU轨迹与维护源码](../work/rfm_review_v3/README.md)
- [本次上传文件与校验和](../PAPER_REVISION_MANIFEST.json)

本轮核验发现：二审所称三处中文公式错误在基准提交353b9cf中并不存在，现改为中英共用公式源并增加基准一致性检查。补充保存探针的频率/过零诊断、同场误差向量恒等式、实际非均匀FD4壁面闭合及条件PCG界。主窗口无层中点配置频偏约−4.19%和−3.98%，但不据此推断内壁分离机理；5.6%恢复比例也不是独立误差来源的因果分摊。Legendre和RFM使用相同六个解析外边界层及共同求积。两份全文各含7幅图、15张表、54个带标签公式和21条文献。

This revision audits the review against the actual source and adds saved-array diagnostics and spatial-operator checks. It adds no PDE integration, reference solution, GPU run, or isolated timing. The original 78 GPU qualification trajectories, revision 3's 20 CPU trajectories, and rejected final timing campaign are unchanged. No new speed ranking, trained-PINN comparison, or upstream library PR is claimed. Qualification scores retain their original times, without phase alignment.

`MANIFEST.json` remains the frozen v8 inventory. The two review directories have separate inventories; `PAPER_REVISION_MANIFEST.json` binds this manuscript upload. The previous [standalone GPU operator example](../outputs/rfm_operator_sample/README.md) remains available. Original source hashes apply to their original bytes; existing redaction maps still apply to any separately named historical derivatives. The repository is public; a review recommendation is not a journal acceptance.

The unchanged [historical evidence-extraction note](../work/rfm_paper/evidence_notes.json), previously hash-referenced but absent from the uploaded tree, is now included so all 96 original evidence-source records resolve in the repository.
<!-- /paper-revision-4 -->

**最新入口：[v8：算子改良与同精度公平比较](../START_HERE_v8.md)。**

本仓库保存 Re=100 有限完整圆环绕流中的 RFM、Legendre–Galerkin 与径向四阶差分研究代码、理论、冻结协议、原始数据和失败记录。

v8 的 78/78 条 T72 数值资格通过原有门槛；最终强对照批次的正式计时接受数为 0。三次未通过接受条件的正式尝试均保留，本轮不形成新的隔离速度排名，也不据此宣称 RFM 稳定超过传统 CFD。数值资格、局部优化和完整公平计时的边界见最新入口。

- [最终研究报告](../outputs/RFM_NS_v8_算子改良与同精度公平比较_研究报告.pdf)
- [正式恢复分支最终结案](../work/rfm_v8/rank_fair_cache_recovery_formal_final_closure/final_closure.json)
- [完整文件清单](../MANIFEST.json) · [公开派生文件映射](../work/rfm_v8/publication_path_redaction_map.json)
- [上一轮 GPU 准备优化入口](../START_HERE_prepared_cuda.md) · [历史入口备份](../publication_history/v8_baseline_dcb69a05c60a4d2c18422b9018a8421145ced190/github_README.md)

旧报告和协议保留各自时间点的结论。公开派生文件具有独立哈希，不能替代原件的哈希证明；新的执行需在本机重新确认环境和协议，不能覆盖或拼接旧实验。
