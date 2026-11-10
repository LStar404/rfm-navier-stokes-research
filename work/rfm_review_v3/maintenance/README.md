# 旧稳态求解器的维护修订 / Maintenance of the early steady solver

本目录处理审稿意见中的基函数注入问题。它适用于早期二维稳态方腔和 Kovasznay 弱形式求解器；旧 `rfm_v2_warp_legendre.py` 的正式入口是 Re=400 方腔。最终 Re=100、T=72 圆环计算使用 `work/rfm_v6/full_galerkin.py` 及后续 v7/v8 驱动，**不经过这里的旧稳态包装器**。本修订不是圆环 GPU 算子加速，也不改变已冻结实验。

The review identifies a real maintainability problem but misidentifies its target: the archived wrapper patches `rfm_v2_solver.trial_basis`, not `rfm_solver.basis_factory`. This package replaces that wrapper's global patch with a call-local `basis_factory` argument. Omitting the argument preserves the default basis behavior and existing solver arguments. The archived source files remain untouched; their SHA-256 values are recorded in `provenance.json` and checked by the tests.

## API

从仓库根目录导入维护包；旧模块的数值参数保持不变。自定义工厂签名为 `(problem, m, seed, family, scale, wall)`，返回具有 `count=m` 和二阶解析 `jet(xy)` 的基函数对象。每次求解仅调用一次工厂。并发时每次求解独立构造对象，工厂自身也应避免共享可变状态。

```python
from work.rfm_review_v3.maintenance import rfm_solver as base
from work.rfm_review_v3.maintenance import rfm_v2_solver as solver
from work.rfm_review_v3.maintenance import rfm_v2_warp_legendre as mapped

problem = base.Problem('cavity', 400)
options = dict(m=15, scale=3., order=5, q=11, maxiter=4,
               metric='h2_parity', wall=dict(warp='sinh', strength=[1.3, 1.3]))
random_result = solver.solve(problem, seed=17, **options)
legendre_result = mapped.solve(problem, seed=17, **options)
explicit_result = solver.solve(problem, seed=17,
                               basis_factory=mapped.mapped_legendre_factory, **options)
```

旧基准驱动 `work/rfm_ns/rfm_v2_benchmark.py` 的 `SolverSpec(..., solve_fn=...)` 接口本身无需修改即可接收新的 `mapped.solve`；若正式复跑，必须生成新协议并将维护包全部源文件记录为新来源，不能覆盖历史输出或沿用旧哈希。本维护包不复制正式基准命令行入口，避免无意运行或覆盖历史实验。

The package is self-contained apart from NumPy and SciPy. `rfm_solver.py` is the archived core copied without numerical changes. `tensor_rfm.py`, `wall_trial.py` and `test_metric.py` contain the original projection, coordinate/feature and Gram arithmetic, extracted without their exploratory or benchmarking drivers. Imports are package-relative. The archived environment setting of one BLAS/OpenMP thread at core import is retained for compatibility; this change removes the factory patch and does not claim to make arbitrary user factories or all surrounding numerical libraries thread-safe.

## Verification

```powershell
python -m work.rfm_review_v3.maintenance.run_validation
```

`validation_results.json` records seven tests. Eight small CPU configurations are compared against the archived functions in a separate process. They cover IID, dense/parity H2 metrics, Legendre with prefix/reused Jacobian, Sobol, paired frequencies, wall-aware RFM, and mapped Legendre. Comparisons include coefficients, the Stokes matrix, fields at 25 independent points, nonlinear histories, ranks and residuals. Tolerances are `rtol=atol=1e-12`; the recorded differences are actual measurements, not assumed equivalence. Timing fields are excluded.

The archived `operator_opt.py` imports `certify_rfm.raw_fields` for its exploratory audit driver, even when only its original Gram function is needed. The historical publication omitted `certify_rfm.py`. To make the comparison runnable from that publication, `tests/archive_support/certify_rfm.py` contains the **exact original file bytes**, recorded in `provenance.json`. Only the archived-comparison child process adds this support directory to its import path. No stub, substituted Gram assembly, or rewritten archived solver is used. The maintained solver itself does not import this support file. `snapshot.py --archived --archive-root <repository-root> --output <new-json>` can independently test the archived modules from another checkout.

并发测试在两个工厂内部设置屏障，强制不同工厂重叠执行，并同时运行默认和映射 Legendre 包装器。另测试工厂异常、后续组装异常和一个线程失败时另一线程的正常结果。它们验证本次消除全局替换所针对的状态污染；不构成长时间数值稳定性证明。测试不使用 GPU，耗时不是性能基准。

## Scope of the factual audit

`facts_audit.json` separates early/legacy code from the canonical annular path, corrects the reviewer’s geometry, envelope, frequency, effective-rank and FFT-unit transcriptions, and marks historical diagnostic timings as separate cohorts. `maintenance_zh.tex` and `maintenance_en.tex` are short manuscript-ready implementation notes; `review_notes_zh.md` supplies concise response points and the conditional amortization model.
