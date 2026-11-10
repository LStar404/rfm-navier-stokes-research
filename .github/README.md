# 不可压缩 Navier–Stokes 可复核研究

**最新入口：[v8：算子改良与同精度公平比较](../START_HERE_v8.md)。**

本仓库保存 Re=100 有限完整圆环绕流中的 RFM、Legendre–Galerkin 与径向四阶差分研究代码、理论、冻结协议、原始数据和失败记录。

v8 的 78/78 条 T72 数值资格通过原有门槛；最终强对照批次的正式计时接受数为 0。三次未通过接受条件的正式尝试均保留，本轮不形成新的隔离速度排名，也不据此宣称 RFM 稳定超过传统 CFD。数值资格、局部优化和完整公平计时的边界见最新入口。

- [最终研究报告](../outputs/RFM_NS_v8_算子改良与同精度公平比较_研究报告.pdf)
- [正式恢复分支最终结案](../work/rfm_v8/rank_fair_cache_recovery_formal_final_closure/final_closure.json)
- [完整文件清单](../MANIFEST.json) · [公开派生文件映射](../work/rfm_v8/publication_path_redaction_map.json)
- [上一轮 GPU 准备优化入口](../START_HERE_prepared_cuda.md) · [历史入口备份](../publication_history/v8_baseline_dcb69a05c60a4d2c18422b9018a8421145ced190/github_README.md)

旧报告和协议保留各自时间点的结论。公开派生文件具有独立哈希，不能替代原件的哈希证明；新的执行需在本机重新确认环境和协议，不能覆盖或拼接旧实验。
