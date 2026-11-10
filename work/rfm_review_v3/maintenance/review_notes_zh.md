# 可用于逐条审稿回复的核查要点

本文本只记录本维护子任务核到的事实；新增流动消融、误差剖面及切换点实验由各自数据支持。

1. **代码维护建议：接受，修正适用路径。** 审稿指出的全局替换确实存在，实际目标是早期稳态方腔包装器中的 `v2.trial_basis`，不是 `base.basis_factory`。维护包给 `v2.solve` 增加可选、调用局部的 `basis_factory`，映射 Legendre 包装器显式传入。原 `work/rfm_ns` 文件不改，保留冻结哈希。最终非稳态圆环路径不使用这个包装器，不能把此项描述成圆环 GPU 性能优化。

2. **已验证的范围。** 七项 CPU 检查全部通过；八种小网格设置与归档函数的系数、Stokes 矩阵、25 点场值最大绝对差为 0（本机本次记录），统一比较容限相对和绝对各 `1e-12`。历史和残差的浮点量按同一容限比较，整数、标签等精确比较，排除耗时。并发工厂经过屏障强制重叠，另覆盖默认调用、映射包装器、工厂异常、组装异常和并发异常。局部工厂不再污染默认模块状态，不代表任意外部工厂均线程安全。

3. **半径：纠正审稿转录，不引入 6.5。** 原始算例 `a=0.5, R=6`，物理径向跨度 `R-a=5.5`；对数径向坐标 `s=log(r/a)`、区间长度 `L=log(R/a)=log(12)`。审稿建议将外半径改成 6.5、跨度改成 6.0 不符合代码。应加强符号区分，而不是改算例参数。

4. **边界算子：区分不同历史版本。** 最终 `farfield` 包络基础项为 `16z²(1-z)`，并有外端二阶导数修正；`16z²(1-z)²` 是较早 `dirichlet` 分支。最终算子来自 `rfm_v6/full_galerkin.py`、`rfm_v3/cylinder_rfm.py`、`radial_layers.py`，早期平面方腔核心不能作为最终圆环公式的出处。

5. **随机特征频率：更正分式。** 正确表达为 `b_j=62(j+ξ_j)/p`，`j=0,...,p-1`，并非 `62/[p(j+ξ_j)]`。审稿所提固定 `ξ_j=0.5` 的消融仍具有研究价值，但必须从正确频率规则出发。

6. **自由度：以增广、截断后的实际秩比较。** RFM37/K72 与 Leg35/K72 的保留非零模径向秩均为 39；RFM41/K80 与 Leg36/K80 均为 40；共同平均模秩均为 30。后缀是增广、截断前的特征数。审稿“额外两个自由度”“41 对 36 模态”的解释不成立。适合保留原始数、六个层函数和保留秩三列，避免二次混淆。

7. **微分 SVD 的表述：限定到具体度量。** 它规范化保留子空间在采样微分内积中的表示，使 `k=1` 扩散 Gram 约等于 `νI`。不能泛称消除了随机特征或所有求解算子的病态性；其他 `k`、质量矩阵、原始系数变换仍可能条件数较大。

8. **同一表不能拼接不同批次。** `1.4810 s`、`1.7519 s`、`3.5891 s`、`3.8552 s` 来自较早 prepared-CUDA 诊断批；最终 v8 同秩强对照误差属于另一批。历史两批计时隔离均未通过，最终 v8 正式计时接受数为 0。因此这些耗时不能成为最终精度行的同批速度结果。表中 `0.02943`、`0.40960` 等均为百分数，应显式加 `%` 或在表注中统一声明。

9. **FFT 数量应带统计单位。** 分段生产推进 8192 步、8191 次非线性求值、16382 次批量 cuFFT 调用。每次求值有三个逆变换场和两个正变换场，故 `q=66` 时逐径向行的场变换数为 `8191×5×66=2703030`。预热另计 330 个逐行场变换、两次批量调用。不能把批量调用数称为所有单独的一维 FFT 数。

10. **误差变化不自动说明降阶。** `0.02943%→0.04467%` 是跨评分点、时刻等取最大值后的汇总变化；它不证明相同状态在 `t=8` 附近发生局部降阶或分裂误差。可提出切换点与激励峰值关系的假说，再用配对实验检验。若没有时间收敛序列，避免声称确定的收敛阶下降。

11. **负结果的逻辑边界。** 可以写“在当前规则几何、精度约束和公平协议下尚未建立 RFM 稳定优势”，不能写“证明 RFM 在规则几何无法击败传统方法”。数值资格、非隔离计时诊断和真正接受的同精度隔离计时属于不同证据等级。

## 准备开销的条件摊销模型

设额外准备耗时为 `ΔP`，诊断批在 8192 步内的推进节省约为 `s₀=0.119 s`。将推进收益粗略假设为全步数线性时，

`Δt(N)=ΔP−s₀ N/8192`，因此 `N*=8192 ΔP/s₀`。

以四舍五入的 `ΔP∈[0.206,0.240] s` 代入，得约 **14181–16522 步**。但前 1024 步采用相同早期步率，实际改变的是之后 7168 步，因此建议用保留早期段的条件模型：

`Δt(N)=ΔP−s₀ max(N−1024,0)/7168`，

`N*=1024+7168 ΔP/s₀≈13432–15480 步`。

原始诊断差值逐对代入时，RFM37 为约 15437 步、旧 Leg37 为约 13444 步；这是同一历史诊断批的条件推算，不能误标成最终 Leg35 的性能数据。`facts_audit.json` 同时保留两种模型及未舍入原数。

这些数字**不是测得的盈亏平衡点，也不是已验证的性能预测**：原始批次隔离未通过；线性节省、固定准备成本、相同设备/精度/缓存策略和长时间精度持续合格均是假设。共享进程和缓存也只有在确实复用相应编译、上下文或空间算子时才可能跨轨迹摊销，独立新进程并不会自动达到该效果。不能把求解器准备差额模型当成已测得的新进程返回耗时模型。

## English response summary

We accept the explicit-factory recommendation, while correcting the affected path: the archived steady-cavity wrapper patched `rfm_v2_solver.trial_basis`. A separate maintained package now passes a solve-local factory and leaves frozen sources intact. Seven CPU checks pass, including eight archived-versus-maintained small configurations, forced concurrent overlap, and recovery after factory/assembly exceptions. This is a compatibility and state-isolation change, not an annular GPU acceleration.

We correct transcription errors in the domain (`a=0.5`, `R=6`, logarithmic `L=log(R/a)`), the final farfield envelope, the stratified frequencies, and retained ranks. We distinguish raw feature counts from the effective marching ranks and label all accuracy values as percentages. Earlier diagnostic timings are not assigned to the later rank-fair accuracy cohort. Conditional amortization estimates are explicitly separated from measured performance; the model preserving the unchanged first 1024 steps gives approximately 13432–15480 total steps from rounded diagnostic inputs, subject to unverified assumptions and failed original isolation.
