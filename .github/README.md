# 不可压缩 Navier–Stokes 可复核研究

<!-- paper-revision-3 -->
## 最新审稿修订 / Reviewer revision 3 (2026-11-10)

- [中文论文（38页）](../outputs/RFM_NS_中文论文.pdf)
- [English manuscript (36 pages)](../outputs/RFM_NS_English_Manuscript.pdf)
- [中英逐条审稿回复（5页） / Point-by-point response](../outputs/RFM_NS_审稿回复.pdf)
- [完整 LaTeX 源码包](../outputs/RFM_NS_LaTeX.zip) · [构建说明](../outputs/rfm_ns_tex/README.md)
- [本轮实验、原始轨迹与维护源码](../work/rfm_review_v3/README.md)
- [本次上传文件与校验和](../PAPER_REVISION_MANIFEST.json)

本轮新增20条CPU T72轨迹：16条特征消融/复算和4条延后换步对照。固定频率中点加解析层的两档配置均达标；去掉非零模层的12条轨迹均精度失败，但实际秩也下降，因此不是等秩归因。新增同一时刻的径向/角谱误差图、复杂几何边界讨论、条件摊销模型，并用7项CPU测试验证旧稳态求解器的显式工厂注入。两份全文各含6幅图、14张表、49个带标签公式和20条文献。

The revision adds actual feature/switch controls and error decompositions, while correcting factual transcriptions in the review. All new parallel CPU durations remain diagnostic. The original 78 GPU qualification trajectories and rejected final timing campaign are unchanged; no new isolated speed ranking, trained-PINN comparison, or upstream library PR is claimed.

`MANIFEST.json` remains the frozen v8 inventory. `work/rfm_review_v3/revision_inventory.json` binds the new evidence; `PAPER_REVISION_MANIFEST.json` inventories the current manuscript upload. The previous standalone GPU operator example remains available under [outputs/rfm_operator_sample](../outputs/rfm_operator_sample/README.md). Original source hashes apply to their original bytes; existing redaction maps still apply to any separately named historical derivatives.
<!-- /paper-revision-3 -->

**最新入口：[v8：算子改良与同精度公平比较](../START_HERE_v8.md)。**

本仓库保存 Re=100 有限完整圆环绕流中的 RFM、Legendre–Galerkin 与径向四阶差分研究代码、理论、冻结协议、原始数据和失败记录。

v8 的 78/78 条 T72 数值资格通过原有门槛；最终强对照批次的正式计时接受数为 0。三次未通过接受条件的正式尝试均保留，本轮不形成新的隔离速度排名，也不据此宣称 RFM 稳定超过传统 CFD。数值资格、局部优化和完整公平计时的边界见最新入口。

- [最终研究报告](../outputs/RFM_NS_v8_算子改良与同精度公平比较_研究报告.pdf)
- [正式恢复分支最终结案](../work/rfm_v8/rank_fair_cache_recovery_formal_final_closure/final_closure.json)
- [完整文件清单](../MANIFEST.json) · [公开派生文件映射](../work/rfm_v8/publication_path_redaction_map.json)
- [上一轮 GPU 准备优化入口](../START_HERE_prepared_cuda.md) · [历史入口备份](../publication_history/v8_baseline_dcb69a05c60a4d2c18422b9018a8421145ced190/github_README.md)

旧报告和协议保留各自时间点的结论。公开派生文件具有独立哈希，不能替代原件的哈希证明；新的执行需在本机重新确认环境和协议，不能覆盖或拼接旧实验。
