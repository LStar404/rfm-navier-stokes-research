# GPU 准备优化与同等精度复测

本轮只针对已测到的初始化瓶颈：共享三个步长相位不变的 Galerkin 空间准备，并将两组未经修改的 CUDA 内核编译到一个模块。Python 保留实验组织职责，CPU 热点仍使用 C/AVX2，GPU 使用 CUDA/cuFFT。没有 Rust 迁移的性能声明。

**本轮已完成并独立审核。95/95 条计时轨迹通过数值资格，但两批均未取得完整隔离资格，所有新耗时仅作诊断。** 中档第490秒有一次进程识别缺口；高档第240秒有一次缺口，并有25个样本检测到超过阈值的外部 Python 活动。两批求解均按时完成，原始数据全部保留，没有重跑替换。

[阅读九页研究报告](outputs/RFM_NS_v7_GPU准备优化与公平复测_研究报告.pdf)。此前的研究与失败记录继续保留。

| 配置 | RFM 完整求解中位秒 | RFM 新进程至结果秒 | 同批 FD176 求解秒 | 同批 FD176 新进程秒 |
|---|---:|---:|---:|---:|
| 中档，两项合并 | 1.4810 | 3.5891 | 3.0618 | 5.1404 |
| 高档，两项合并 | 1.7519 | 3.8552 | 3.0780 | 5.1829 |

这些时间不能作为隔离速度排名。按同块配对，FD176/RFM 新进程时间比中位数分别为1.431和1.350；它们不是两组中位数直接相除。相对原恒步方案，组合实现的新进程配对收益约为1.002和1.016，未达到预登记的双时钟实用标准。没有证明随机特征优于同等优化的确定性谱方法。

单模块的诊断收益主要来自约0.15秒运行时初始化下降；共享空间准备约少0.01–0.02秒主机设置。新的准备工作、传输与图设置仍抵消不少推进节省，新进程额外开销约2秒。候选实现保留，尚不提升为已确认的最优默认方案。

## 实验范围

- Re=100、T=72、无反射对称约束的二维有限圆环绕流，内半径 0.5、外半径 6；不是无限域圆柱准确度结论。
- 速度和涡量同时达到 0.5%，在 19 个绝对时刻、三张独立评分网格、全域与近场检查；不作事后相位对齐。0.1%档的晚期参考预算尚未通过。
- 五个预登记 RFM 种子；相同优化用于 Legendre。低档保留失败控制，中、高档做新进程双时钟比较。
- 中档 55 条：恒步、原分段、仅合模块、仅共享主机准备、两项合并，加 FD176。
- 高档 40 条：恒步、原分段、两项合并，加 FD176 和 FD184 带状实现。
- 两批均只尝试一次；失败、外载与未覆盖窗口不能通过筛选记录消除。

## 数值复现

环境与原生依赖遵循既有仓库环境说明。以下命令在仓库根目录运行，并向新目录写入结果；不要覆盖冻结证据。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B work/rfm_v7/replay_prepared_cuda.py --method rfm37_k72 --preparation combined --horizon 9 --output replay/prepared/rfm37_T9.npz --json-output replay/prepared/rfm37_T9.json
python -B work/rfm_v7/replay_prepared_cuda.py --method rfm41_k80 --preparation combined --horizon 72 --output replay/prepared/rfm41_T72.npz --json-output replay/prepared/rfm41_T72.json
```

复现程序逐项比较同绝对时刻的状态与探针。短程复现不替代完整 T=72 物理资格；记录中的耗时仅供诊断，不能与正式隔离数据合并。注册的其他种子、Legendre 和中档单项准备方式也可以重放。

## 阅读顺序

1. 本轮报告：`outputs/RFM_NS_v7_GPU准备优化与公平复测_研究报告.pdf`。
2. 等价条件：`work/rfm_v7/theory_shared_phase_preparation.txt`。
3. 数值证据：`prepared_pilot_output`、`prepared_qualification_output`、`prepared_uniform_high_qualification_output`。
4. 独立计时：`work/rfm_v7/prepared_formal_timing_mid_output` 和 `prepared_formal_timing_high_output`，包含完整轨迹、协议、双时钟、身份记录与监测窗。
5. 历史失败与前一轮报告：`START_HERE_variable_step.md`。

冻结驱动的原汇总函数误把平坦记录当成嵌套字段，执行失败。新增 `work/rfm_v7/prepared_formal_summary.py` 仅在内存适配已核对的记录；冻结驱动、协议和原始计时均未修改。`verify_prepared_formal_timing.py` 独立复核全部95条记录、47项配对和两项因子交互，并从19条代表轨迹重新计算流场。复核证书为 `prepared_formal_timing_verification.json`。

所有发布文件以 `MANIFEST.json` 的字节数及 SHA256 为准。单模块和共享准备的收益属于所有相关 Galerkin 方法共享的实现改进，不能全部归因于随机特征。
