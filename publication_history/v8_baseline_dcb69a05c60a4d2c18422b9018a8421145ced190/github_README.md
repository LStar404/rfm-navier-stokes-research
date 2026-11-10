# Reproducible incompressible Navier–Stokes research

本仓库保存 RFM、Legendre–Galerkin 与优化径向四阶差分的研究代码、理论说明、冻结协议、原始轨迹、完整计时和失败记录。

**最新入口：[GPU 准备优化与同等精度复测](../START_HERE_prepared_cuda.md)。**

最新研究关注 Re=100 有限圆环非稳态绕流：不施加反射对称约束，同时检查速度与涡量，比较完整 GPU 求解和新进程返回时间。共享实现改进不等于随机特征本身有优势；有限域精度也不等于无限域外流精度。

- [最新研究报告](../outputs/RFM_NS_v7_GPU准备优化与公平复测_研究报告.pdf)
- [共享空间准备的等价条件](../work/rfm_v7/theory_shared_phase_preparation.txt)
- [上一轮入口与负面结果](../START_HERE_variable_step.md)
- [文件完整性清单](../MANIFEST.json)

旧报告和协议保留各自时间点的结论；请以最新入口核对当前结论及限制。原始实验从未通过删除失败或事后相位对齐来制造优势。
