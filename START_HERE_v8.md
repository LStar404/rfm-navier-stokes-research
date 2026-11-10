# v8：算子改良与同精度公平比较

本轮完成了 **39 个数值配置、两种编译缓存政策下的 78/78 条 T72 数值资格验证**。完整公平计时未获接受，**最终强对照批次的正式计时接受数为 0**；本轮没有建立新的隔离速度排名，也没有证据宣称优化后的 RFM 稳定超过传统 CFD。

研究对象是 Re=100 的有限完整圆环非稳态绕流，不施加反射对称约束。比较包括 RFM、Legendre–Galerkin 和径向四阶差分对照，同时检查速度、涡量、近场和尾迹。有限域上的这些结果不能直接推广为无限域外流结论。

- [最终研究报告 PDF](outputs/RFM_NS_v8_算子改良与同精度公平比较_研究报告.pdf)
- [报告最终证据索引](work/rfm_v8/report_final_round.json) · [报告发布复核](work/rfm_v8/report_final_round_publication_review.json)
- [78 条数值资格汇总](work/rfm_v8/rank_fair_cache_qualification_output/qualification_summary.json)
- [正式计时恢复分支最终结案](work/rfm_v8/rank_fair_cache_recovery_formal_final_closure/final_closure.json)

数值资格与公平计时是两个不同结论。78 条资格保留了原精度门槛与 T72 范围；39 对缓存政策的字段和完整探针数据逐位一致。资格运行的耗时不用于速度排名，T9 入口等价检查也不能替代完整正式计时。

三次正式窗口未完成接受条件，记录全部保留：

| 尝试 | 已记录结果 | 当前结论 |
| --- | --- | --- |
| 原正式窗口 | 前 2 条完成，第 3 条有 READY/GO，缺少 RETURN 与关闭凭证 | 中断原因未知，不能恢复为完整计时窗口 |
| 第一次恢复 | 预检观察到未登记 Python 活动，0 条计时任务启动 | 预检拒绝；后续高档窗口未启动 |
| 预先限定的最后一次恢复 | 12 条尝试、11 条原始成功；第 12 条因末态记账错误仍为失败；148 条未运行 | 末态记账未闭合，且活动期外部 Python 超预算；两项分别足以否决隔离计时 |

最后一次尝试保存的 12 份 T72 数值数据均与已有资格对应，但这不等于 12 条正式计时成功。第 12 条仍保留失败状态。不能删去失败、修补记账后拼接样本，或把其他尝试的条目混入新的排名；预定恢复次数已耗尽。

- [原中断的独立复核](work/rfm_v8/formal_interruption_review_20261110/formal_interruption_review.json)
- [零条启动的预检失败复核](work/rfm_v8/rank_fair_cache_recovery_formal_v2_review/preflight_failure_review.json)
- [末态发布机制的独立 CPU 诊断公开副本](work/rfm_v8/rank_fair_slot_publish_cpu_review/independent_diagnosis.public.json)
- [相位矩阵铅笔方法的适用范围](work/rfm_v8/theory_phase_pencil_scope_note.txt) · [独立理论复核](work/rfm_v8/theory_phase_pencil_scope_independent_review.json)

公开材料包括理论、冻结协议、数值数据、源码、检查程序和失败证据。它们支持检查本轮具体实现与记录，不能自动证明随机特征具有普遍优势。局部内核或推进阶段的改良，也不能单独推出完整求解及新进程成本上的优势。

复核公开文件时，先用 [完整文件清单](MANIFEST.json) 验证字节数与 SHA-256，再使用 [公开派生文件映射](work/rfm_v8/publication_path_redaction_map.json) 和 [历史源码精确副本映射](work/rfm_v8/publication_historical_source_map.json)。公开副本与保留本机路径的原件具有不同文件名和不同哈希；原冻结记录中的哈希仍指向原件，不能用它冒充公开副本的哈希。精确历史副本则必须保持原始字节和哈希。

公开记录可用于离线复核，直接执行旧冻结协议可能因本机环境、未公开的原始路径记录或供应链依赖而失败。新的执行必须在本机重新确认环境、依赖与隔离条件，并生成独立协议和新输出目录；不得覆盖历史证据。项目自编译工具的源码、编译配方和历史二进制分别留有身份记录，不承诺跨机器编译逐位相同。必要依赖与明确限定的历史例外以发布清单中的范围说明为准。

此前的 [GPU 准备优化入口](START_HERE_prepared_cuda.md)、[变步长研究入口](START_HERE_variable_step.md)、[v7 入口](START_HERE_v7.md) 和 [原仓库说明](README.md) 继续保留各自时间点的结论。本次入口替换前的字节与原清单保存在 [历史入口备份](publication_history/v8_baseline_dcb69a05c60a4d2c18422b9018a8421145ced190/github_README.md) 与 [历史清单备份](publication_history/v8_baseline_dcb69a05c60a4d2c18422b9018a8421145ced190/MANIFEST.json)。
