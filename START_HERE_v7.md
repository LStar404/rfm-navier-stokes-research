# RFM–NS v7 最新入口

从修订报告开始阅读。它补充了旧报告缺少的启动段验收，并更新了步率选择。 [修订报告](outputs/RFM_NS_v7_算子改写与启动段复核_修订报告.pdf)

## 当前结论

- **步率 112 暂缓作为默认设置。** 原有 `t=24` 至 `72` 的验收通过。补充的 `t=0.5` 至 `8` 启动段检查中，四个实现配对均超出 10⁻⁵ 误差门槛。12 条短程轨迹同时覆盖步率 112 和 128。 [启动段审计](work/rfm_v7/startup_replay_output/audit.json)
- **保留步率 128 的算子改写。** RFM 的五个种子和 Legendre 对照在三条实现路径上共完成 18 条 `t=72` 轨迹，均通过资格检查。早期检查的结论限于同一离散的实现一致性，尚未建立早期连续 PDE 精度证书。双参考的 0.5% 物理精度门槛应用于原定的七个后期时刻。本轮保留步率 128，步率 112 的步数和 FFT 节省未计入采用结果。 [步率 128 完整资格](work/rfm_v7/r128_operator_qualification_output/qualification.json)
- 两批各 50 次计时都观察到了外部计算活动，分批保留为诊断。当前没有新的隔离速度结论。 [原批监控摘要](work/rfm_v7/cpu_native_joint_output/summary.json) · [复测监控摘要](work/rfm_v7/cpu_native_joint_repeat_output/summary.json)

## 验证归档

从归档根目录运行以下命令。第二条命令重评分已保存的启动段和步率 128 轨迹。 [独立重评分结果](V7_STARTUP_FOLLOWUP_VERIFICATION.json)

```powershell
python -B work/rfm_v7/verify_v7_operator_round.py
python -B work/rfm_v7/verify_v7_startup_followup.py
```

旧归档核验的默认命令通过。旧脚本附加 `--replay` 时会保留原来的失败行为，在步率 112 的启动段实现验收处返回退出码 1。对应结果分开记录。 [旧归档与启动段状态](work/rfm_v7/startup_replay_output/release_replay_status.json)
