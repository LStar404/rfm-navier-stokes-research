"""Check coverage and produce bilingual reviewer-response tables from saved scores."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, x: Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
p = read(HERE / "protocol.json")
s = read(HERE / "summary.json")
metrics = s["metrics"]
assert s["protocol_sha256"] == sha(HERE / "protocol.json")
assert all(sha(WORK / path)==expected for path,expected in p["sources_sha256"].items())
assert read(HERE / "adapter_verification.json")["status"] == "pass"
expected = {(g, t) for g in s["grid_names"] for t in p["times"]}
assert len(s["results"]) == 26
for row in s["results"]:
    if row["status"] != "complete":
        continue
    assert set(row["scores"]) == set(s["reference_names"])
    all_scores = []
    for ref, rows in row["scores"].items():
        assert len(rows)==57 and {(x["grid"], x["time"]) for x in rows}==expected
        all_scores.extend(rows)
    assert row["physical_worst"] == {m:max(r[m] for r in all_scores) for m in metrics}
    if row["provenance"] == "frozen-v8-GPU-trajectory-rescored":
        old = read(WORK / row["source"]["independent_score"])
        assert sha(WORK / row["source"]["independent_score"]) == row["source"]["independent_score_sha256"]
        assert all(abs(row["physical_worst"][m]-max(ref[m] for ref in old["physical_worst"].values())) < 1e-12 for m in metrics)
cells = []
for resolution in ("mid", "high"):
    for role in ("random-layers", "midpoint-layers", "random-no-layers", "midpoint-no-layers"):
        rows = [r for r in s["results"] if r["resolution"]==resolution and r["role"]==role]
        complete = [r for r in rows if r["status"]=="complete"]
        cells.append(dict(resolution=resolution, role=role, total=len(rows),
            complete=len(complete), numerical_failures=len(rows)-len(complete),
            accuracy_failures=sum(not r["physical_pass"] for r in complete),
            passes=sum(r["physical_pass"] for r in rows),
            retained_ranks=sorted(set(r["retained_rank"] for r in complete)),
            maximum={m:max(r["physical_worst"][m] for r in complete) for m in metrics},
            minimum={m:min(r["physical_worst"][m] for r in complete) for m in metrics}))
controls = [r for r in s["results"] if r["role"]=="cpu-control"]
max_control = max(v for r in controls for v in r["archived_gpu_control_difference"].values())
new_cpu = [r for r in s["results"] if r["provenance"]=="new-CPU-accuracy-only"]
score_rows = sum(len(rows) for r in s["results"] for rows in r.get("scores", {}).values())
summary = dict(registered_cpu_jobs=len(p["jobs"]), completed_cpu_jobs=sum(r["status"]=="complete" for r in new_cpu),
    cpu_numerical_failures=sum(r["status"]!="complete" for r in new_cpu),
    cpu_accuracy_failures=sum(r["status"]=="complete" and not r["physical_pass"] for r in new_cpu),
    reused_archived_trajectories=len(p["baselines"]), cell_unique_trajectories=sum(c["total"] for c in cells),
    controls=len(controls), physical_score_rows=score_rows, scalar_errors=score_rows*len(metrics),
    original_registered_source_hashes_unchanged=True, archived_score_maxima_reproduced=True,
    adapter_verification="adapter_verification.json", max_cpu_gpu_control_field_difference=max_control,
    cells=cells)
write(HERE / "cell_summary.json", summary)
write(HERE / "coverage_verification.json", dict(status="pass", **{k:v for k,v in summary.items() if k!="cells"}))
time_regions = []
for row in s["results"]:
    if row["role"] not in ("midpoint-no-layers", "midpoint-layers") or row["status"]!="complete":
        continue
    entries = [dict(ref=ref, **value) for ref,values in row["scores"].items() for value in values]
    time_regions.append(dict(id=row["id"],
        all_worst_locations={m:max(entries, key=lambda v:v[m]) for m in metrics},
        early_max={m:max(v[m] for v in entries if v["time"]<=16) for m in metrics},
        late_max={m:max(v[m] for v in entries if v["time"]>=24) for m in metrics}))
write(HERE / "time_region_summary.json", time_regions)

for language in ("zh", "en"):
    table_rows = []
    for cell in cells:
        role = cell["role"]
        if language=="zh":
            level = "中" if cell["resolution"]=="mid" else "高"
            method = ("随机" if role.startswith("random") else "定点") + ("+层" if "no-layers" not in role else "无层")
        else:
            level = "Mid" if cell["resolution"]=="mid" else "High"
            method = ("Random" if role.startswith("random") else "Midpoint") + ("+layer" if "no-layers" not in role else "/no layer")
        values = " & ".join(f"{100*cell['maximum'][m]:.4f}" for m in metrics)
        rank = "/".join(map(str, cell["retained_ranks"]))
        table_rows.append(f"{level}, {method} & {rank} & {values} & {cell['passes']}/{cell['total']} " + r"\\")
    rows_text = "\n".join(table_rows)
    if language=="zh":
        content = r"""\subsection{随机性与非零模解析层的补充消融}
\label{sec:review-ablation}
为区分随机频率与解析外边界层的作用，我们预先固定两档配置
$(m,K,q,\Delta t^{-1})=(37,72,66,128)$、$(41,80,76,152)$，
将配对频率中的随机偏移 $\xi_j\sim U[0.2,0.8]$ 与确定性中点
$\xi_j=0.5$ 交叉比较，并分别保留或移除非零模的六个解析层函数。
随机组使用原先注册的五个种子（7331、7687、8011、8353、9041）；
每个确定性配置仅有一个独立轨迹，不将未使用的种子视为重复实验。
平均模始终保留相同的24个基函数与六个层函数、同一组积分点及时间步长。
适配器验证确认四组的平均模离散数组与隐式算子逐位一致。

本次新增16条CPU $T=72$ 轨迹（14条消融及2条原配置复算控制），
复用并重新评分10条已冻结的随机加层GPU轨迹。
所有轨迹均在19个固定时刻、三个独立网格、两套细化FD参考上评价
全域与 $r\leq2.5$ 区域的速度和涡量；四项误差均须小于等于0.5\%。
表~\ref{tab:review-ablation} 报告每组跨种子、时刻、网格和参考的最坏相对误差。
16条CPU计算均完成，未出现数值运行失败；12条无层轨迹均为精度不达标。
两个CPU复算控制相对于归档GPU解的最坏场差异小于 $2.0\times10^{-10}$。

\begin{table}[htbp]
\centering\small
\caption{补充CPU精度消融。误差单位为\%；通过数不包含两条单列复算控制。
每行中的随机组为五个种子的最坏值，确定性组为一个配置。}
\label{tab:review-ablation}
\begin{tabular}{lrrrrrr}
\hline
配置 & 秩 & $e_u$ & $e_\omega$ & $e_{u,\mathrm{near}}$ & $e_{\omega,\mathrm{near}}$ & 通过 \\
\hline
__ROWS__
\hline
\end{tabular}
\end{table}

确定性中点加层的两档配置均通过门槛，其误差与随机加层组相当，
因此这些数据不支持将随机性视为当前精度优势的必要来源。
在固定原始 $m$ 与SVD截断阈值的设计下，去除非零模层函数使保留秩由
39/40降为34/35，并导致很大的速度与涡量误差；平均模的保留秩始终为30。
这表明现有紧凑配置高度依赖解析外边界层富集，但不能把其作用与增加的有效维数完全分开：
此处并非等秩层函数消融，也没有遍历所有确定性频率设计。
两档确定性无层配置的四项最坏误差均出现在 $T=72$；
其早期 $t\leq16$ 的最大涡量误差为2.820\%和2.592\%，随后分别增至73.784\%和71.871\%。
因此，表中的大误差并非由起始时刻近零参考场单独造成，但这些数据尚不能确定误差放大的具体动力学机制。
随后仅作构造检查：保持频率带宽62、积分点和截断阈值不变，
将无层确定性基的原始 $m$ 依次增加至41、45、49、53、57、61，
两档的保留秩均为35、35、36、36、36、36，未恢复至39或40。
因此没有追加声称等秩的 $T=72$ 对照，也没有为通过门槛而继续调整带宽或截断阈值。
并行CPU耗时只作为运行诊断，不构成新的隔离计时或GPU加速证据，
原78条GPU资格实验的边界保持不变。
"""
    else:
        content = r"""\subsection{Additional ablation of frequency jitter and nonzero-mode layers}
\label{sec:review-ablation}
We fixed two configurations,
$(m,K,q,\Delta t^{-1})=(37,72,66,128)$ and $(41,80,76,152)$,
before executing a two-by-two ablation: registered random paired-frequency
jitter $\xi_j\sim U[0.2,0.8]$ versus fixed midpoints $\xi_j=0.5$, and six
nonzero-mode analytic outer-layer features present versus absent.
The random cells use the five registered seeds (7331, 7687, 8011, 8353, 9041).
Each deterministic cell has one unique trajectory; unused seeds are not counted
as replicates. All cells retain the same 24 mean-mode functions, six mean-mode
layers, quadrature, and time step. Adapter checks confirm bitwise equality of
the mean-mode discretization arrays and implicit operators across the four cells.

We computed 16 new CPU trajectories to $T=72$: 14 ablation trajectories and
two enriched-random CPU controls. Ten frozen enriched-random GPU trajectories
were reused and independently rescored. Every trajectory was evaluated at the
same 19 physical checkpoints on three grids against both fine FD references,
using velocity and vorticity errors over the full domain and $r\leq2.5$.
All four errors must be at most 0.5\%.
Table~\ref{tab:review-ablation} gives maxima over seeds, times, grids, and references.
All 16 CPU runs completed without numerical failure; the 12 trajectories without
nonzero-mode layers failed the accuracy gate. The two CPU controls differ from
the archived GPU fields by less than $2.0\times10^{-10}$ in the worst field metric.

\begin{table}[htbp]
\centering\small
\caption{Additional CPU accuracy ablation. Errors are percentages. Pass counts
exclude the two separate CPU controls. Random rows report the worst of five
seeds; midpoint rows contain one deterministic configuration.}
\label{tab:review-ablation}
\begin{tabular}{lrrrrrr}
\hline
Configuration & Rank & $e_u$ & $e_\omega$ & $e_{u,\mathrm{near}}$ & $e_{\omega,\mathrm{near}}$ & Pass \\
\hline
__ROWS__
\hline
\end{tabular}
\end{table}

Both midpoint-plus-layer configurations pass, with errors comparable to the
random enriched cells. These data therefore do not establish randomness as
necessary for the observed accuracy. With raw $m$ and the SVD cutoff fixed,
removing the nonzero-mode layers reduces retained rank from 39/40 to 34/35 and
produces large velocity and vorticity errors; the mean rank remains 30.
The compact configurations depend strongly on analytic outer-layer enrichment,
but this design does not separate enrichment geometry from the increased
effective dimension. It is not an equal-rank enrichment ablation, and only one
deterministic frequency design was tested. All four error maxima for both
deterministic no-layer configurations occur at $T=72$. Their maximum vorticity
errors for $t\leq16$ are 2.820\% and 2.592\%, growing to 73.784\% and 71.871\%
at later times. The large errors therefore are not solely a consequence of a
near-zero startup reference field; the mechanism of error amplification is
not established by these data. A subsequent construction-only check increased raw $m$ to
41, 45, 49, 53, 57, and 61 while retaining bandwidth 62, quadrature, and the
SVD cutoff. Both resolutions produced ranks 35, 35, 36, 36, 36, and 36,
respectively, never recovering rank 39 or 40. We therefore did not add a
purported equal-rank $T=72$ control or tune the bandwidth/cutoff after failure.
Concurrent CPU elapsed times are
diagnostic only: they provide neither isolated timing nor new GPU speedup
evidence, and do not extend the original 78 GPU qualification trajectories.
"""
    (HERE / f"ablation_{language}.tex").write_text(content.replace("__ROWS__", rows_text), encoding="utf-8")
print(json.dumps(summary, indent=2))
