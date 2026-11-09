"""Build a new report from the audited reference and completed six-way study."""
import json
from pathlib import Path
import statistics
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'report_v2'))
import layout as doc


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def pct(v):
    return f'{100*v:.5f}%'


def link(url, title):
    return f'<link href="{url}" color="#195783">{title}</link>'


def footer(canvas, info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49,43,546,43)
    canvas.setFont('CN',8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50,29,'RFM / Legendre / FD4 | 参考解修复与同硬件复测 | 2026-11-10')
    canvas.drawRightString(545,29,str(info.page))
    canvas.restoreState()


def main():
    measured = read(ROOT/'hardware_pow2_output'/'timing.json')
    preflight = read(ROOT/'hardware_pow2_output'/'preflight.json')
    reference = read(ROOT/'cpu_formal_point1_pow2_chain_output'/'reference_audit.json')
    candidates = read(ROOT/'cpu_formal_point1_pow2_chain_output'/'saved_candidate_rescore.json')
    cavity = read(ROOT/'supplement_cavity_audit.json')
    scales = read(ROOT/'cuda_multiscale_pow2_output'/'timing.json')
    reuse = read(ROOT/'cuda_multiscale_pow2_output'/'reuse.json')
    assert measured['status']=='complete' and len(measured['rows'])==30
    assert preflight['passes'] and len(preflight['rows'])==6
    assert all(r['passes'] and r['count']==5 for r in measured['summary'])
    assert scales['status']=='complete' and len(scales['rows'])==63
    assert all(r['matched_implementation_pass'] for r in scales['summary'])
    assert reuse['status']=='complete' and cavity['status']=='passed'
    hardware_stats = read(ROOT/'hardware_pow2_output'/'summary.json')
    profiles = [read(ROOT/'cuda_multiscale_pow2_output'/f'graph_profile_{i}.json') for i in (1,2,3)]
    assert all(d['status']=='complete' for d in profiles)
    p, table, sub = doc.p, doc.table, doc.sub
    labels = {'rfm37_k72':'RFM37','leg37_k72':'Legendre37','fd176_k80':'径向四阶 FD176'}
    lookup = {(r['name'],r['backend']):r for r in measured['summary']}
    def median(name,backend): return lookup[name,backend]['median_seconds']
    def paired(a,b,backend):
        ratios=[]; wins=0
        for block in range(1,6):
            ta=next(r['seconds'] for r in measured['rows'] if r['block']==block and r['name']==a and r['backend']==backend)
            tb=next(r['seconds'] for r in measured['rows'] if r['block']==block and r['name']==b and r['backend']==backend)
            ratios.append(ta/tb);wins += tb<ta
        return statistics.median(ratios),wins
    p('计算研究报告 / 2026 年 11 月 10 日 / 本文更新第 7 轮阶段结论','small')
    p('参考解修复后的同精度比较<br/>RFM、Legendre 与优化 CFD','title')
    p('无反射对称约束的 Re=100 圆柱绕流，积分到交替涡脱落充分发展的 t=72。'
      '本轮同时给三类方法提供 CPU 优化和完整 CUDA 路径，再以速度、涡量共同达到0.5%为条件做隔离复测。')
    table(['方法','硬件','中位秒','五次范围 / 秒','最坏速度','最坏涡量'],[
        [labels[r['name']], 'CPU' if r['backend']=='cpu' else 'GPU',f"{r['median_seconds']:.3f}",
         f"{r['min_seconds']:.3f} - {r['max_seconds']:.3f}",
         pct(max(r['worst']['velocity'],r['worst']['near_velocity'])),
         pct(max(r['worst']['vorticity'],r['worst']['near_vorticity']))]
        for r in measured['summary']],[106,39,58,111,91,91])
    cpu_ratio=median('fd176_k80','cpu')/median('rfm37_k72','cpu')
    gpu_ratio=median('fd176_k80','cuda_graph')/median('rfm37_k72','cuda_graph')
    p(f'30/30 次满足精度和实现一致性验收。中位数之比：同 CPU 的 FD/RFM 为 {cpu_ratio:.2f} 倍，'
      f'同 GPU 为 {gpu_ratio:.2f} 倍。这里比较的是已登记且达标的具体配置，不代表优于全部传统 CFD。')
    pr, pw=paired('leg37_k72','rfm37_k72','cpu')
    gr, gw=paired('leg37_k72','rfm37_k72','cuda_graph')
    p(f'随机特征的独立贡献仍须谨慎：RFM 相对 Legendre 在 CPU 的五个配对块中快于对照 {pw} 次，'
      f'在 GPU 为 {gw} 次；配对 Legendre/RFM 中位比为 {pr:.3f} 和 {gr:.3f}。'
      '共享的径向弱离散、边界层富集和实现优化不能全部归功于随机性。')
    p('新的参考链足以通过0.5%档的经验收敛筛查；0.1%档仍未通过预定参考预算，且宽频 RFM49 '
      '五个种子实测涡量误差仍高于0.1%。本文不宣称这一精度等级的速度优势。','small')

    doc.new('01  冻结协议、边界与参考可靠性')
    p('圆柱半径0.5、外半径6、Re=100、ν=0.01。主流平滑启动，4&lt;t&lt;12 施加峰值0.12的横向脉冲。'
      '保留完整 Fourier 角向自由度及切向平均动量。外边界指定流函数且涡量为零；该问题属于有限环域。')
    p('固定三个面积加权评分网格 Gauss123、偏移中点129、外层加密144，七个绝对时刻 '
      't=24、36、48、54、60、66、72。全域和 r≤2.5 近场各比较两分量速度联合 L² 与实际涡量 L²，'
      '按同一参考场的范数归一化。84项误差全部过线；没有相位平移、挑时刻或改评分权重。')
    table(['独立加密边','保持不变的参数','最坏四指标差'],[
        ['n384 → 448','K128，步率384',pct(max(reference['edges'][0]['worst'].values()))],
        ['K128 → 144','n448，步率384',pct(max(reference['edges'][1]['worst'].values()))],
        ['步率384 → 512','n448，K144',pct(max(reference['edges'][2]['worst'].values()))],
        ['乘积合成','同一二次幂行缩放实现',pct(reference['composed_difference'])],
    ],[149,220,127])
    p('所有参考臂使用相同数值代数实现。合成差0.029014%低于0.5%档的0.125%预算，'
      '高于0.1%档的0.025%预算。最粗到最细直接差只有0.004158%，可能包含空间与时间误差抵消，'
      '因此没有用它替换独立边筛查。参考差异不是连续 PDE 的认证误差界。')
    sub('为何修复参考解')
    p('未缩放 FD 的 n384→448 径向涡量差曾达0.141921%。检查发现外边界聚点导致矩阵行量级悬殊。'
      '用二次幂行缩放后，65位方程对照下 n448 动态右端的线性解前向误差从5.47×10⁻⁹降到2.32×10⁻¹³；'
      '完整轨迹的径向差降到0.014116%。这是参考计算可靠性改善，不是修改方程后得到的精度提升。')
    p('FD 的四阶指径向模板；本轮三种方法的时间推进均为二阶 SBDF2。'
      '高分辨率参考采用缩放带状分解；正式低档 CFD 使用已逐位校验的 CPU 融合实现和已验收的 GPU 密集路径。','small')

    doc.new('02  资格、种子与0.1%失败记录')
    grouped={r['name']:r for r in candidates['summary']}
    rows=[]
    for group in measured['frozen']['all_candidate_decisions']:
        rows.append([group['name'],len(group['seeds']),pct(max(group['worst']['velocity'],group['worst']['near_velocity'])),
                     pct(max(group['worst']['vorticity'],group['worst']['near_vorticity'])),'通过' if group['qualifies'] else '未通过'])
    table(['已登记候选','种子/重复','最坏速度','最坏涡量','0.5%门槛'],rows,[145,79,98,98,76])
    p('RFM 每档使用预定种子7331、7687、8011、8353、9041。先按冻结候选顺序选择第一个五种子全达标档，'
      '再计时；选档没有利用试跑速度。确定性方法在正式计时中也重复五次，各块内随机顺序。')
    p('硬件为 Ryzen 9 9955HX（16核32线程）与 RTX 5060 Laptop（驱动596.13）。'
      '正式CPU固定逻辑核4、计算库单线程，并停止其他研究计算。RFM37/Legendre37匹配m、K、q与富集参数，'
      '但截断后非零模态秩为39/41，因此不是严格等秩的随机性因果消融。','small')
    sub('更严格精度的结果没有隐藏')
    wide=grouped['rfm49_bandwidth98']
    p(f'扩大随机频带后，RFM49 五种子的最坏涡量为 {pct(wide["worst"]["vorticity"])}，'
      '未达到0.1%。Legendre49 基础版实测约0.09765%，但由于参考预算不合格，仍不把它登记为可靠0.1%结果。')
    p('尺度切向增广由相容指数层对衰减参数求导得到，能保留边界迹并改善部分条件数。'
      '当前测试中，它没有优于所选普通指数层；增加普通层也可能让 Legendre 的长时误差变大。'
      '静态表示误差、代数条件数和完整轨迹精度不能互相替代。')
    p('切向与普通层测试还同时改变了指数位置，因此只能作为该配置的负面结果，不能据此证明一种形状普遍更差。'
      '下一次时间/空间优化应保留这些失败，避免只汇报最有利种子和配置。','small')

    doc.new('03  同一离散的三档规模与提交开销')
    scale_groups={}
    for r in scales['summary']:
        scale_groups.setdefault(r['spec']['name'],{})[r['backend']]=r
    rows=[]
    for name,group in scale_groups.items():
        cpu,eager,graph=(group[k]['wall_median'] for k in ('cpu','cuda_eager','cuda_graph'))
        rows.append([name,f'{cpu:.3f}',f'{eager:.3f}',f'{graph:.3f}',f'{cpu/graph:.2f}',f'{eager/graph:.2f}'])
    table(['固定空间配置','CPU 秒','GPU逐步 秒','GPU图 秒','CPU/图','逐步/图'],rows,[155,66,75,70,65,65])
    p('原生GPU逐步版本已经取得大部分加速。图执行在六个Galerkin配置中的中位比改善约1.07-1.18倍，'
      'FD176仅约1.009倍，后者不足以认定额外的稳定收益。应把完整GPU迁移与图提交的增量分开评价。','small')
    p('RFM与Legendre各有三档匹配m、K、q等参数的配置，截断后的有效秩可不同；FD在已达标中档作控制。'
      '每个组合三轮，RFM使用前三个登记种子；'
      '这63条数据检验相同离散的硬件与提交方式趋势，'
      '不替代上页五种子同精度方法比较。低档可以不达0.5%：它仍可用于同一离散的实现一致性与规模诊断，'
      '不能用于同精度获胜结论。所有硬件组合检查速度、涡量及探针对应一致性。')
    sub('初始化与重复使用')
    reuse_rows=[]
    for g in reuse['groups']:
        reuse_rows.append([g['spec']['name'],f"{g['runtime_initialization_seconds']:.4f}",
                           f"{g['matching_fresh_wall_median']:.4f}",f"{g['reused_wall_median']:.4f}",
                           g.get('break_even_solves_approx') or '无实测盈亏点'])
    table(['配置','运行时初始化秒','新建总秒','复用总秒','估算盈亏次数'],reuse_rows,[145,92,82,82,95])
    p('复用只保留 CUDA 上下文与编译模块，每次仍重建数学模型、数组上传、FFT 计划和图。'
      '一次性初始化与退出成本另计。RFM同种子的新建对照只有一个样本，其他两类各三个；'
      '估算的一次盈亏点对噪声敏感，不能视为稳健结论。主要比较继续采用完整新建求解时间。')
    p('正式总耗时包含构造、预计算、GPU初始化和编译、传输、计划、图、推进、最后同步、结果返回与清理；'
      '评分和文件序列化排除。新建求解器发生在同一进程内，不清空驱动磁盘缓存，不能称为操作系统级冷启动。','small')

    doc.new('04  初始化与实际瓶颈')
    stage_map=[('主机构造','host_setup_seconds'),('GPU初始化/编译','runtime_init_seconds'),
               ('算子与状态上传','upload_seconds'),('FFT计划','plan_seconds'),('图构建','graph_setup_seconds'),
               ('完整推进并同步','march_seconds'),('结果返回','download_seconds')]
    gpu_groups={g['family']:g for g in hardware_stats['groups'] if g['backend']=='cuda_graph'}
    table(['新建求解阶段','RFM秒','Legendre秒','FD4秒'],[
        [label,*[f"{gpu_groups[family]['phases_seconds'][key]['median']:.5f}" for family in ('rfm','legendre','fd4')]]
        for label,key in stage_map],[220,92,92,92])
    p('各列为五轮的阶段中位数；阶段中位数之和不必等于总耗时中位数。'
      '清理包含在总时间中，但没有单独测出。数据传输仅占毫秒级，初始化不可忽略，推进仍是主要耗时。')
    samples={}; controls=[]
    for data in profiles:
        for row in data['rows']:
            assert all(row['bitwise_trajectory_match'].values())
            for key,values in row['profile']['stage_samples_ms'].items():
                samples.setdefault((row['method'],key),[]).extend(1000*v['ms'] for v in values)
            controls.extend(1000*v for v in row['profile']['marker_control_ms'])
    def us(family,key): return f'{statistics.median(samples[family,key]):.2f}'
    table(['图内阶段 / 微秒','RFM','Legendre','FD4'],[
        ['径向重建/模板',us('rfm','radial_reconstruction'),us('legendre','radial_reconstruction'),us('fd4','radial_stencils')],
        ['逆FFT批',us('rfm','inverse_fft'),us('legendre','inverse_fft'),us('fd4','inverse_fft')],
        ['乘积与历史',us('rfm','products_and_history'),us('legendre','products_and_history'),us('fd4','products_history_mean')],
        ['正FFT批',us('rfm','forward_fft'),us('legendre','forward_fft'),us('fd4','forward_fft')],
        ['投影/更新或线性解',us('rfm','projection_and_advance'),us('legendre','projection_and_advance'),us('fd4','linear_solve')],
    ],[220,92,92,92])
    p(f'三次独立尾段剖析，每阶段有12个Galerkin样本或9个FD样本；FD另有右端组装。'
      f'空标记间隔中位{statistics.median(controls):.2f}微秒，最大{max(controls):.2f}微秒。'
      'WDDM调度和计时节点本身会扰动间隔；各阶段不发生在同一步，不能将中位数相加得到精确内核占比，'
      '也没有减去标记开销后制造净时间。')
    p('这组诊断支持继续关注FFT、径向算子和线性更新。当前还没有新的隔离CPU消融把缓冲、布局与递推收益'
      '全部拆开；不能将旧与新两轮中位数直接相减作因果归因。','small')

    doc.new('05  哪些收益有理论依据')
    sub('保持结构，而不是依赖训练惩罚')
    p('非零模态使用 u<sub>r</sub>=ψ<sub>θ</sub>/r、u<sub>θ</sub>=−ψ<sub>s</sub>/r，并演化切向均值。'
      '解析混合导数抵消给出离散试空间的不可压恒等式；RFM与Legendre都使用相容边界。'
      '正权求积形成质量/扩散 Gram 矩阵，在保留采样空间满秩时给出正定质量与半正定扩散。')
    p('同一状态、节点和权重下，非零模态的自对流功与均值动量功抵消；N&gt;3K 的角向求积防止二次项在保留模态的角向混叠。'
      '已有四组算子审计的预收缩残差最大9.42×10⁻¹¹，归一化齐次对流功最大4.46×10⁻¹⁶。'
      '非齐次外边界仍可做功，不能删除后声称总能量单调。')
    sub('预收缩与 CUDA 图各自做了什么')
    p('固定步长下，将 A⁻¹M 和 A⁻¹T 预计算，使投影与隐式更新合并。精确算术中等价；'
      '浮点下依靠原方程残差与完整轨迹验证。GPU保留状态、历史量和输出在显存，24步图减少主机提交。'
      '9216步仍评价9215次非线性。Galerkin每次将3个场的逆FFT合成一次批调用、2个场的正FFT合成另一次批调用；'
      '这一轮没有减少逻辑场变换或批调用次数。')
    sub('缩放不是精度保证，半离散抵消不是全离散稳定性')
    p('对正对角 D，(DA)x=Db 与 Ax=b 等价。若没有溢出或下溢，二次幂乘法不引入新舍入。'
      '但对固定近似解，分量后向误差在行缩放前后相同；改善来自重新分解后求得更准确的解，'
      '不能只缩放残差后宣布改善。该技术已有成熟文献，不是新的通用定理。')
    p('SBDF2 外推非线性与新状态的配对不享有同状态能量抵消，因此未证明无条件长期稳定。'
      '小代数误差、时间误差和频率漂移可能累积为相位偏差。速度含一阶导数，涡量含二阶导数，'
      '后者对薄层与高频更敏感；二者必须同时验收。')
    p('图内阶段剖析使用捕获事件并验证与无标记图逐位相同。Windows WDDM 下相邻空标记也出现间隔波动；'
      '阶段事件时间只能辅助定位，不能减掉一个常数后当作纯内核耗时。端到端结论使用无标记图。','small')

    doc.new('06  Re=1000 方腔补充验证')
    p('复核了已有平滑顶盖方腔：单位正方形，顶盖 u=16x²(1−x)²，其他壁面无滑移，Re=1000，'
      '求稳态。它不是不连续常速顶盖的经典基准，也不是长时间积分算例。')
    archive=read(ROOT/'cavity_release_metadata.json')
    keep=['rfm496_guard','rfm595_guard','strong_leg666_mild','fd152_c32','fd180_c32']
    table(['历史配置','新建求解中位秒','最坏速度','最坏涡量'],[
        [r['name'],f"{r['seconds']:.4f}",pct(r['worst']['velocity']),pct(r['worst']['vorticity'])]
        for r in archive['summary'] if r['name'] in keep],[168,116,106,106])
    p('在其独立冻结候选集内，0.1%门槛的 FD/RFM 中位比为7.96，0.05%为6.33。'
      '这些是历史单线程CPU实验，本轮只复核哈希与数据，不把它们和新的圆柱计时配成速度比。'
      '参考256→384的最坏涡量变化0.009337%，通过当时门槛20%的经验筛查。')
    sub('理论指导的延续求解改良')
    p('对二次非线性残差，沿给定方向的残差是二次多项式，平方范数为四次多项式；'
      '可以分析阻尼的驻点，但该形式并不保证更快收敛。实际保留了中间延续阶段的紧凑修正与原有最终验收。')
    p('五个新种子的496维配对中，新旧方法都达0.1%，新方法在4/5对更快，配对中位比1.050。'
      '595维的0.05%门槛下，旧方法仅4/5通过，新方法5/5通过；这里的主要证据是可靠性改善，'
      '没有计算不合格方法之间的同精度加速比。')
    p('本次审计检查30条依赖哈希记录、70条实验评分和10份保存的第一种子场数据。'
      '其余种子和v5仅有记录可作内部一致性核对，没有完整保存场可独立重建，报告保留这一复现限制。','small')

    doc.new('07  适用范围、负面结果与复现')
    p('R=8 的独立计算给出 St≈0.1737015，而R=6约0.17556；两域公共区域的未对齐涡量差曾达50.39%。'
      '因此当前0.5%数值门槛不能被解释为无限域物理预测达到0.5%。主要结论适用于这里明确的二维有限环域。')
    table(['证据类别','可以支持','仍不能支持'],[
        ['同硬件30次计时','所测三类达标配置的端到端比较','全体CFD或任意几何上的最优性'],
        ['三档63条轨迹','GPU驻留与图提交的规模趋势','用低档未达标配置声称同精度获胜'],
        ['行缩放参考修复','减少本例高分辨率代数污染','新的平衡定理或严格PDE误差界'],
        ['能量与边界结构','满足假设的有限维恒等式','当前SBDF2无条件长期稳定'],
        ['方腔历史复核','第二种几何/稳态求解的补充证据','湍流、三维、复杂工程几何的泛化'],
    ],[101,191,204])
    p('保留的失败包括：0.1%参考预算未过、RFM49五种子未过0.1%、切向富集未优于对照、'
      'SBDF3旧试跑发散、CuPy路径当前实现未取得优势。后续改进应针对实际误差与开销，不能靠更改门槛或移除失败取得结果。')
    p('论文仍需：对新的算子/空间组合补齐同轮因果消融、进程冷启动与跨机器复现，'
      '并评估更少时间步或更少非线性评价的可靠条件。当前较清晰的成果是可审计的结构离散与完整CUDA实现；'
      '随机特征的普遍优势和新的无条件稳定性理论均未建立。')
    sub('文献与完整材料')
    p(link('https://arxiv.org/abs/2207.13380','Chen等：Random Feature Method')+'；'+
      link('https://arxiv.org/abs/2304.06913','Chen等：Time-dependent RFM')+'。本文的相容Galerkin空间与SBDF2路线需与原框架区分。','small')
    p(link('https://netlib.org/lapack/explore-html/df/d0a/group__geequb_gaef3a3015d58122ab3d4d018422b93b90.html','LAPACK DGEEQUB 官方说明')+
      '给出机器基数幂平衡的既有原则及局限。'+link('https://arxiv.org/html/2602.03247v1','Jia、Wang：PD-OFM（2026）')+
      '使用物理预训练与正交正则，其NS测试为稳态Kovasznay；不能据此替代本文长时同精度对照。','small')
    p('主要数据：hardware_pow2_output/{protocol,preflight,timing}.json；'
      'cpu_formal_point1_pow2_chain_output/{reference_audit,saved_candidate_rescore}.json；'
      'cuda_multiscale_pow2_output/{protocol,timing,reuse}.json；supplement_cavity_audit.json。'
      '推导见theory_v7.txt与theory_reference_equilibration.txt。保留原始轨迹、种子、源码哈希和历史报告。','small')
    destination=ROOT.parents[1]/'outputs'/'RFM_NS_v7_参考修复与同硬件比较_研究报告.pdf'
    pdf=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=44,bottomMargin=53,
                          title='RFM NS 参考修复与同硬件比较',author='Computational research study')
    pdf.build(doc.story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'report_hardware_text.txt').write_text('\n\n'.join(doc.transcript),encoding='utf-8')
    print(destination)


if __name__=='__main__':
    main()
