"""Build the completed operator-round supplement without altering older reports."""
import json
from pathlib import Path
import sys
from statistics import median

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'report_v2'))
import layout as doc


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def pct(x):
    return f'{100*x:.5f}%'


def footer(canvas, info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49,43,546,43)
    canvas.setFont('CN',8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50,29,'RFM / Legendre / FD4 | 算子改写与步长消融 | 2026-11-10')
    canvas.drawRightString(545,29,str(info.page))
    canvas.restoreState()


def main():
    timing=read('cpu_native_joint_repeat_output/timing.json')
    summary=read('cpu_native_joint_repeat_output/summary.json')
    first=read('cpu_native_joint_output/timing.json')
    historical=read('hardware_pow2_output/timing.json')
    step=read('cpu_formal_step_screen_output/qualification.json')
    native=read('cpu_native_qualification_output/qualification.json')
    native_fixed=read('cpu_native_qualification_output/fixed_state_audit.json')
    oracle=read('cpu_fd_row_moment_output/fixed_audit.json')
    paired=read('fd_cuda_rowcomp_verification.json')
    band=read('fd_cuda_pow2_replay_band.json')
    dense=read('fd_cuda_pow2_replay_dense.json')
    reference=read('cpu_formal_point1_pow2_chain_output/reference_audit.json')
    scalar=read('audit_sbdf2_scalar_stability.json')
    assert timing['status']=='complete' and len(timing['rows'])==50
    assert first['status']=='complete' and len(first['rows'])==50 and all(g['passes'] for g in first['groups'])
    assert historical['status']=='complete' and len(historical['rows'])==30
    assert not summary['isolated_speed_claim_supported'], 'This report describes the recorded contaminated replication'
    assert all(g['passes'] and g['count']==5 for g in timing['groups'])
    assert native['passes'] and paired['paired']['passed'] and band['passed'] and not dense['passed']
    assert scalar['status']=='passed' and scalar['checks']==86
    groups={(g['name'],g['arm']):g for g in timing['groups']}
    detail={(g['name'],g['arm']):g for g in summary['groups']}
    labels={'rfm37_k72':'RFM37','leg37_k72':'Legendre37','fd176_k80':'径向四阶 FD176'}
    p,table,sub=doc.p,doc.table,doc.sub

    def ratio(a,b):
        return next(r['aggregate'] for r in timing['ratios'] if tuple(r['numerator'])==a and tuple(r['denominator'])==b)

    def worst(g,kind):
        return max(v[k] for v in g['worst'].values() for k in (kind,'near_'+kind))

    def historical_ratio(backend):
        values=[]
        for block in range(1,6):
            def duration(name):
                return next(r['seconds'] for r in historical['rows'] if r['block']==block and r['name']==name and r['backend']==backend)
            values.append(duration('fd176_k80')/duration('rfm37_k72'))
        return median(values)

    p('计算研究报告 / 2026 年 11 月 10 日 / 第 7 轮算子优化补充','small')
    p('减少时间步，重写热点算子<br/>同精度验收与计算量消融','title')
    p('继续无反射对称约束的 Re=100 圆柱绕流，保留 Galerkin 径向离散、完整角向模态和交替涡脱落。'
      '本轮不更换整个技术栈，而是验证预收缩与连续内存访问、步长选择，以及高分辨率参考算子的浮点可靠性。')
    selected=[(n,a) for n in labels for a in (('cpu_buffered','cuda_graph') if n.startswith('fd') else ('cpu_native_contract','cuda_graph'))]
    table(['方法','硬件','步率','最坏速度','最坏涡量','数值验收'],[
        [labels[n],'CPU' if a.startswith('cpu') else 'GPU',128 if n.startswith('fd') else 112,
         pct(worst(groups[n,a],'velocity')),pct(worst(groups[n,a],'vorticity')),'5/5 通过']
        for n,a in selected],[108,36,37,112,112,91])
    p('本轮的两套50条完整队列均完成精度和实现验收，但计时期间均检测到外部计算。'
      '因此新增时间数据只作诊断，尚不能据此宣布新的隔离速度优势。第二套数据列于上表；两套记录不合并增加样本数。')
    p('已验证的新增成果是：在0.5%速度/涡量双门槛下减少1152个时间步与2304次FFT批调用；'
      '原生CPU投影/更新通过五种子验证；高分辨率参考导数的局部舍入误差显著减小。'
      '这些分别是计算量、实现一致性和数值可靠性的证据，不能相互替代。')
    p(f"上一轮30条已验收的同硬件比较仍单独保留：原配置同CPU的配对FD/RFM中位比约{historical_ratio('cpu'):.3f}，同GPU约{historical_ratio('cuda_graph'):.3f}。"
      '本轮不以受扰动样本更新这些速度结论，也不把胜过本例FD4推广为胜过全部传统CFD。','small')

    doc.new('01  验收条件与时间步选择')
    p('圆柱半径0.5、外半径6、ν=0.01、终点t=72。主流平滑启动，4&lt;t&lt;12施加横向脉冲；'
      '外边界指定流函数和零涡量，保留切向均值动量。评分时刻24、36、48、54、60、66、72，'
      '三个固定面积权重网格，全域及r≤2.5近场，两分量速度联合L²与物理涡量L²，共84项。无事后相位对齐。')
    table(['所选配置','最坏速度','最坏涡量','减少的非线性评价'],[
        [labels[n],pct(worst(groups[n,a],'velocity')),pct(worst(groups[n,a],'vorticity')),0 if n.startswith('fd') else 1152]
        for n,a in selected if a.startswith('cpu')],[150,106,106,134])
    p('同一空间离散分别检查步率80、96、112、128。RFM五个预定种子及Legendre选择112，FD176保留128。'
      '资格选择发生在正式计时之前；除双参考直接误差外，还要求相对于128档的变化与128档误差的三角组合低于门槛，'
      '并检查已测试更细时间档的变化趋势。该检查减少误差抵消风险，仍不是连续PDE认证误差界。')
    p('按每个范数独立使用：E(c,ref) ≤ E(c,anchor)[1+E(anchor,ref)]+E(anchor,ref)。'
      '原生CPU RFM的最坏组合约0.48029%，Legendre约0.41385%。同一网格/时刻的对应项组合后再取最大，'
      '不能混用不同归一化或用最大值的有利抵消。','small')
    table(['每条T=72轨迹','步率128','步率112'],[
        ['时间步数',9216,8064],['非线性评价',9215,8063],['正逆FFT批调用合计',18430,16126]
    ],[220,138,138])
    p('减少1152次非线性评价与2304次FFT批调用。Galerkin每次仍是3场批量逆FFT与2场批量正FFT；'
      '没有减少每次评价的变换数，也没有引入更高阶积分器。径向FD为四阶，所有方法时间推进仍为二阶SBDF2。')
    p('15个较粗步率GPU候选失败并保留。独立CPU控制复现：RFM96首个非有限通量t≈3.396，'
      'Legendre96为t≈3.365，FD112为t≈5.821；先出现有限值急剧增长再溢出。'
      '所测失败不能外推为普遍稳定界，亦没有靠修改初始扰动来避免失败。','small')

    doc.new('02  受扰动计时：保留观察，不作隔离结论')
    arms=[('cpu_recurrence','原CPU / 112'),('cpu_native_contract','原生融合 / 112'),
          ('cuda_graph_rate128','GPU图 / 128'),('cuda_graph','GPU图 / 112')]
    table(['实现与步率','RFM中位秒','Legendre中位秒'],[
        [label,f"{groups['rfm37_k72',a]['median']:.4f}",f"{groups['leg37_k72',a]['median']:.4f}"]
        for a,label in arms],[220,138,138])
    rows=[]
    for name in ('rfm37_k72','leg37_k72'):
        for old,new,desc in [('cpu_recurrence','cpu_native_contract','固定112步率的CPU算子改写'),
                             ('cuda_graph_rate128','cuda_graph','固定GPU实现的128→112')]:
            r=ratio((name,old),(name,new))
            rows.append([labels[name]+' / '+desc,f"{r['median']:.3f}",
                         f"{r['minimum']:.3f}-{r['maximum']:.3f}",f"{r['denominator_wins']}/5"])
    table(['受扰动配对 / 仅诊断','时间比中位','配对范围','分母更快'],rows,[241,74,109,72])
    p('本页数据来自第二次完整复测，记录实际观测，不把比值解释为已经确认的算子或步长提速。'
      f"同批FD176 CPU/GPU中位耗时为 {groups['fd176_k80','cpu_buffered']['median']:.4f} / "
      f"{groups['fd176_k80','cuda_graph']['median']:.4f} 秒；同样只作诊断。",'small')
    p('十个实验臂、五个随机顺序块。RFM按预定五个种子各运行一块，Legendre与FD每块重复；'
      'Ryzen 9 9955HX / RTX 5060 Laptop。CPU固定逻辑核4、计算库单线程。'
      '资格与高分辨率验证可并行。计时期间已停止本任务的其他数值计算，但外部项目中途再次活动。')
    p('最初50条队列在中途遇到外部计算，全部保留为诊断记录，没有并入本表或提高重复次数。'
      '外部计算退出后，另做一次相同配置/顺序的完整50条复测；整批采用预先登记的600秒共同外载观察窗，'
      '统一验收。这次也检测到外部活动，按预定次数限制结束，不追加第三批或筛选较快样本。','small')
    p('每次计时包括模型构造、预计算、GPU上下文/编译初始化、上传、FFT计划、图、完整推进、同步、返回与清理，'
      '不含评分和文件写入。同一进程内新建求解器，没有重启操作系统或清空驱动缓存；不能称为进程冷启动。','small')
    sub('随机特征的贡献仍须单独看')
    for arm,label in [('cpu_native_contract','CPU原生'),('cuda_graph','GPU')]:
        r=ratio(('leg37_k72',arm),('rfm37_k72',arm))
        p(f"本次受扰动{label}记录中，配对 Legendre/RFM 中位比 {r['median']:.3f}，RFM较快 {r['denominator_wins']}/5 次。",'small')
    p('共享的弱离散、边界层富集、预收缩与GPU实现同样提供给Legendre。RFM37与Legendre37匹配输入m、K、q，'
      '但截断后的非零模态秩为39与41，不能把此差异当作严格等秩随机性消融。'
      'Legendre-Galerkin本身也是确定性谱CFD；胜过本例FD4，不能被表述为已经胜过所有传统CFD。')

    doc.new('03  成本分项诊断与减少FFT的收益边界')
    p('下表保留第二次受扰动队列的分项观测，用于后续定位。外部负载可能改变各阶段耗时，'
      '这些数值不能替代新的空闲机器隔离复测。','small')
    def phase(n,a,key):
        value=detail[n,a]['phases_seconds'].get(key)
        return f"{value['median']:.4f}" if value else '未单列'
    cpu=[(n,a) for n,a in selected if a.startswith('cpu')]
    gpu=[(n,a) for n,a in selected if a=='cuda_graph']
    table(['CPU配置','构造/预计算秒','其余求解秒','端到端秒'],[
        [labels[n],phase(n,a,'setup_seconds'),phase(n,a,'solve_minus_setup_seconds'),f"{groups[n,a]['median']:.4f}"]
        for n,a in cpu],[160,112,112,112])
    p('CPU构造/预计算包含空间构造、分解或预收缩等设置。其余求解时间用内部总时间减设置时间，'
      '包含推进与在线观测；端到端另由外层计时。各列均独立取中位数，不能相加推导逐次精确分解。','small')
    table(['GPU配置','主机准备秒','额外线性预计算秒','GPU初始化秒','推进秒','总秒'],[
        [labels[n],phase(n,a,'host_setup_seconds'),phase(n,a,'host_linear_precompute_seconds'),
         phase(n,a,'runtime_init_seconds'),phase(n,a,'march_seconds'),f"{groups[n,a]['median']:.4f}"]
        for n,a in gpu],[106,75,92,85,69,69])
    p('Galerkin的主机侧预收缩包含在主机准备中，FD的额外密集线性预计算另列。'
      '完整JSON还记录上传、FFT计划、图构造和下载；上表不把未列出的项目当作零。'
      '推进时间含保存所需状态和在线探针处理，最终总时间包括清理。','small')
    sub('为什么12.5%的步数下降不保证12.5%的总耗时下降')
    p('一个解释成本的近似模型是 T(S)=T_setup+(S−1)T_nonlinear+S T_update+T_output。'
      '减少时间步只降低随S变化的部分；计划、预计算和返回的固定开销仍在。'
      '这个模型用于解释实测，不作为严格性能上界，实际缓存、调度和初始化也会变化。')
    p('本轮的GPU 128/112实验设计保持空间、图实现与硬件不变，旨在区分减少步数与前轮整条CUDA迁移的收益；'
      '当前受扰动耗时尚不足以完成这个速度归因。'
      'CPU新旧算子则固定112步率，不能把二者各自的速度比直接乘起来后声称未经实测的整体结果。')
    sub('冷启动的范围仍须明确')
    p('每次重建求解器并计入初始化，但同一宿主进程仍存活；驱动的磁盘缓存也未清空。'
      '因此这是包含设置成本的新建求解，不是全新进程或机器重启后的冷启动基准。'
      '本轮未采用初始化摊销口径。进程冷启动与跨机器复现仍待补齐。')
    p('计时期间仅做低频外部Python进程采样，后台服务的允许阈值在开跑前登记；共同观察窗检测到超阈值活动，因此拒绝新的隔离速度结论。'
      '采样不能排除两个采样点之间出现又结束的短暂活动，系统与图形桌面的背景负载也未被假定为零。','small')

    doc.new('04  CPU预收缩的代数与数据布局')
    p('固定步长下令 A=1.5M+dtK。将 B=A⁻¹M 与 G=A⁻¹T 预计算，可把每一步的通量投影与隐式更新合并。'
      '提升边界和平均模态必须分别按原方程处理；非线性始终由含提升的全场计算。')
    p('若 m<sub>n</sub>=M a<sub>n</sub>+g<sub>n</sub>m_lift，则原方程为<br/>'
      'A a<sub>n+1</sub>=2m<sub>n</sub>−0.5m<sub>n-1</sub>+dt(2f<sub>n</sub>−f<sub>n-1</sub>)−g<sub>n+1</sub>(1.5m_lift+dt k_lift)。'
      '当前静止启动的首步保留隐式Euler；不能直接套用BDF2或把这个首步简式推广到任意非静止初值。')
    p('直接用NumPy批量小矩阵实现，短程反而更慢：原来共享的较大矩阵乘被拆成各模态不同的小矩阵。'
      '保留的原生版本按[输出径向、输入径向、角模态]排布矩阵，最内层遍历连续模态，'
      '融合径向与角向通量投影，以改善访问和调用开销。')
    p('新增C/AVX2核使用FP64，关闭fast-math及乘加收缩；这不约束既有FFT/BLAS库内部编译。'
      'FFT仍使用同一个SciPy路径。收益来自预收缩、布局和融合的组合，不是单纯把Python改成C的语言实验。')
    p('16项固定状态检查覆盖所有模态、有限扰动、带间距输入与只读数组、历史量所有权和不合法布局拒绝。'
      '五个RFM种子加Legendre的完整T=72资格全部通过；原生与原CPU同档的场和探针差在既定实现门槛内。'
      '本轮低档CPU/GPU队列的实现一致性门槛为1e-5，最坏差3.88e-6；下一页高分辨率参考移植另用更严格的1e-6门槛。')
    sub('Rust 的适用位置')
    p('目前没有证据支持全量改写为Rust。GPU热点已在CUDA/cuFFT，CPU热点已有原生实现；'
      '更换宿主语言不会自动减少FFT次数、矩阵流量或离散自由度。若以后定位到管理开销或并行调度瓶颈，'
      '可单独试Rust扩展，但需要保持相同数值算法、编译与库配置后实测。')
    p('本轮没有Rust对照，因此不提供Rust/C速度排名。当前被保留或淘汰的是具体算子实现。','small')

    doc.new('05  CFD参考算子也同步改进')
    p('高分辨率FD448 / K144 / 步率512采用二次幂行缩放 DAx=Db。通过的带状CUDA路径显式缩放右端并使用带状分解；'
      '不能从(DA)⁻¹错误地直接作用到未缩放b，也不能将D重复乘两次。')
    table(['参考路径 / T=72','对匹配CPU的最坏场差','实现门槛'],[
        ['折叠缩放密集逆',f"{max(dense['worst_fields'].values()):.3e}",'未过 1e-6'],
        ['显式缩放带状解',f"{max(band['worst_fields'].values()):.3e}",'通过 1e-6'],
        ['带状解 + 行矩恢复',f"{max(paired['paired']['worst_fields'].values()):.3e}",'通过 1e-6']
    ],[201,175,120])
    p('密集版本失败被保留，不作为新的高分辨率参考。带状版本的实际设备线性解经65位固定方程复核，'
      '但正式评分参考仍是归档原CPU。这里的完整CPU/GPU验证在探索负载下执行，不能用其耗时推出正式参考加速比。')
    sub('为什么要改导数求值顺序')
    saved=[r for r in oracle['rows'] if r['state'].startswith('t')]
    old=max(r['staged_gradient']['relative_l2'] for r in saved)
    new=max(r['row_moment_gradient']['relative_l2'] for r in saved)
    old_outer=max(r['staged_gradient']['outer8_relative_l2'] for r in saved)
    new_outer=max(r['row_moment_gradient']['outer8_relative_l2'] for r in saved)
    table(['k=1固定状态D₁Ω审计','原求值','行矩恢复'],[
        ['全径向相对L²',f'{old:.3e}',f'{new:.3e}'],
        ['外侧8点相对L²',f'{old_outer:.3e}',f'{new_outer:.3e}']
    ],[220,138,138])
    p('上述为t=24、54、72的最坏值，对照把已存FP64系数与输入视为精确数的高精度计算。'
      '主要消去发生在先求D₂ψ再求D₁Ω的上游。只中心化D₁并不能解决上游污染；'
      '据此选择k=1的Taylor残差求值并恢复实际离散矩，其他模态保留原算术。')
    prior=max(paired['cpu_vs_prior']['worst_fields'].values())
    p(f'完整新CPU轨迹相对旧CPU的最坏物理场差仅 {prior:.3e}。这是对求值改动的敏感度，'
      '不是相对未知真解的误差；固定状态改善不能被解读为整体NS精度提高同样倍数。'
      '新旧均保持交替涡脱落，CPU/GPU配对也检查未移相的探针和过零时间。')

    doc.new('06  理论边界与本轮结论的范围')
    p('对实际浮点模板a<sub>j</sub>和实际存储位移δ<sub>j</sub>，取任意同一斜率d，令 μ₀=Σa<sub>j</sub>、μ₁=Σa<sub>j</sub>δ<sub>j</sub>，精确算术中有：<br/>'
      'Σa<sub>j</sub>x<sub>j</sub> = Σa<sub>j</sub>(x<sub>j</sub>−x_c−dδ<sub>j</sub>)+μ₀x_c+μ₁d。<br/>'
      '离散矩必须由同一组已装配系数与位移计算，不能强制设为解析零值而悄悄改变矩阵。')
    p('若实际残差为r<sub>j</sub>+e<sub>j</sub>、存储矩为μ<sub>i</sub>+η<sub>i</sub>，则最终积加前的差为 Σa<sub>j</sub>e<sub>j</sub>+η₀x_c+η₁d。'
      '后续顺序积加可用标准γ<sub>n</sub>=nu/(1−nu)乘绝对项和控制；还需没有溢出/下溢等假设。'
      '因此“精确残差O(h²)”不等于“浮点误差不随网格加密增长”。')
    p('O(h²)残差需要C²光滑场、局部位移O(h)、斜率误差O(h)和位移存储误差O(h²)，加权后的误差还取决于网格系数尺度。'
      '状态存储量化仍会经D₂放大；高精度矩计算也不能恢复已丢失的状态位。当前改写先计算原模板再覆盖k=1，'
      '增加了局部算术，主要目标是可靠性。非线性耦合会将k=1的变化传到其他模态和均值。')
    p('原参考链的独立径向、角向与时间加密合成差为'+pct(reference['composed_difference'])+
      '，通过0.5%档的0.125%预算，未通过0.1%档的0.025%预算。'
      '本轮同档CPU/GPU一致性不能代替新的独立加密链，因此仍不宣称0.1%速度优势。')
    p('既有三档规模、63条匹配实现数据及Re=1000平滑顶盖方腔结果保留在上一份完整报告。'
      '本轮50次消融只覆盖所选圆柱档位，不将其称为三档新实验。有限外域、秩不完全匹配、'
      '缺少跨机器与进程冷启动复测，仍是论文结论的重要限制。')
    doc.new('07  时间推进限制：严格标量结论与NS边界')
    p('对常系数方程 y′=(−a+ib)y，d=ah>0、v=bh。当前SBDF2把耗散隐式处理、输运二步外推，对应：<br/>'
      '(3/2+d)z²−(2+2iv)z+(1/2+iv)=0。')
    p('对复系数二次多项式使用Schur判据并直接化简，可得两根严格落在单位圆内的充要条件：<br/>'
      'v² &lt; (1+d)[(2/√3)√(d(1+d))−d]。<br/>'
      '其关键差式为 d(d+4)(1+d)²−6d(1+d)v²−3v⁴。完整代数步骤保留在理论附录。')
    p('等号处有一个单重单位圆根，另一根严格在圆内，对应线性幂有界而非严格衰减。'
      '当d=0且v≠0，当前格式不稳定；d=v=0则保留常数。'
      '在小d边界上 v⁴≈4d/3，即 h³≈4a/(3b⁴)，这个渐近式不能替代全范围的精确条件。')
    p('86项80位小数根复核覆盖10个耗散尺度、边界内/边界上/边界外及纯虚频率控制，均符合推导。'
      '该计算是标量推导的独立数值核对，没有新增流体实验，也不是用拟合公式解释既有成功样本。')
    p('若两个常系数线性算子在同一稳定基中可同时对角化，可以逐模态使用此条件。'
      '完整NS中的非正规性、时变边界、非线性模态耦合和不交换的径向输运/扩散均超出它的适用范围；'
      '不能用最大流速代入后声称精确预测本轮96/112步率的成败。')
    p('已有同状态对流功抵消与不可压恒等式，但当前SBDF2外推项并非与同一新状态配对，'
      '所以没有证明无条件全离散能量稳定。本轮有价值的结果是受精度约束的计算量下降、可审计的热点实现和参考可靠性改进。')
    sub('资料与复现')
    p('本轮协议及原始轨迹：cpu_native_joint_repeat_output与cpu_native_joint_output（两批均为受扰动计时）、cpu_formal_step_screen_output、'
      'cpu_native_qualification_output、cpu_fd_row_moment_output；配对验证：fd_cuda_rowcomp_verification.json。'
      '完整推导见operator_roundoff_theory.txt与theory_sbdf2_scalar_stability.txt。旧实验与失败记录均保留。','small')
    p('求和误差模型参考：<link href="https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf" color="#195783">'
      'Higham (1993), The Accuracy of Floating Point Summation</link>，第2节。'
      '行中心化是对现有模型的具体应用，未声称新的通用浮点理论。','small')
    p('IMEX多步格式的经典分析：<link href="https://www.cs.ubc.ca/tr/1993/tr-93-15" color="#195783">'
      'Ascher、Ruuth、Wetton：Implicit-Explicit Methods for Time-Dependent PDEs</link>。'
      '上面的标量推导是针对当前实现的直接复核，不主张新的通用IMEX稳定性理论。','small')
    target=HERE.parents[1]/'outputs'/'RFM_NS_v7_算子改写与步长消融_研究报告.pdf'
    pdf=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=44,bottomMargin=53,
                          title='RFM NS 算子改写与步长消融',author='Computational research study')
    pdf.build(doc.story,onFirstPage=footer,onLaterPages=footer)
    (HERE/'report_operator_text.txt').write_text('\n\n'.join(doc.transcript),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
