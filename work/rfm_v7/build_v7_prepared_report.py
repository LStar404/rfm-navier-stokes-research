"""Prepared CUDA experiment report; tables are read from frozen cohort evidence."""
import json
from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'report_v2'))
import layout as doc

_original_glyphs = doc.glyphs
def scientific_glyphs(text):
    for symbol, markup in (('ᵀ', '<super>T</super>'), ('ₖ', '<sub>k</sub>'), ('ₛ', '<sub>s</sub>')):
        text = text.replace(symbol, markup)
    return _original_glyphs(text)
doc.glyphs = scientific_glyphs

def read(path):
    return json.loads((HERE / path).read_text(encoding='utf-8-sig'))

def pct(x):
    return f'{100*x:.4f}%'

def stat(x):
    return f"{x['median']:.3f} [{x['minimum']:.3f}, {x['maximum']:.3f}]" if x else '无合格数据'

def footer(canvas, info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49, 43, 546, 43)
    canvas.setFont('CN', 8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50, 29, '不可压 NS | GPU 准备优化与公平复测 | 2026-11-10')
    canvas.drawRightString(545, 29, str(info.page))
    canvas.restoreState()

def main():
    reports = {scale: read(f'prepared_formal_timing_{scale}_output/summary.json') for scale in ('mid', 'high')}
    protocols = {scale: read(f'prepared_formal_timing_{scale}_output/protocol.json') for scale in reports}
    monitors = {scale: read(f'prepared_formal_timing_{scale}_output/isolation.json') for scale in reports}
    qualification = read('prepared_qualification_output/qualification.json')
    uniform = read('prepared_uniform_high_qualification_output/qualification.json')
    pilot = read('prepared_pilot_output/summary.json')
    p, table, sub = doc.p, doc.table, doc.sub
    family = {'mid': ('rfm37_k72', 'leg37_k72'), 'high': ('rfm41_k80', 'leg39_k80')}
    names = {'rfm37_k72': 'RFM37', 'leg37_k72': 'Legendre37', 'rfm41_k80': 'RFM41',
             'leg39_k80': 'Legendre39', 'fd176_k80': 'FD176', 'fd184_k80': 'FD184 带状'}
    arms = {'uniform': '原恒步', 'switch_old': '原分段', 'switch_module': '仅合模块',
            'switch_host': '仅共享空间准备', 'switch_both': '两项合并', 'fd': '差分 GPU'}
    groups = {scale: {(r['name'], r['arm']): r for r in report.get('groups', [])}
              for scale, report in reports.items()}
    clean = {scale: report['isolated_speed_claim_supported'] for scale, report in reports.items()}

    def monitor_label(scale):
        m = monitors[scale]
        if clean[scale]:
            return '通过采样隔离验收'
        if m.get('solver_finished_within_window') and not m.get('sample_coverage_pass'):
            return '求解按时完成；监测覆盖不完整'
        if m.get('external_activity_detected'):
            return '观测到外部活动'
        return reports[scale]['monitor_status']

    def time_value(scale, name, arm, clock='solver_seconds'):
        value = groups[scale].get((name, arm), {}).get('phase_seconds', {}).get(clock)
        return f"{value['median']:.4f}" if value else '无合格数据'

    def pair(scale, baseline, candidate):
        return next((r for r in reports[scale].get('paired', [])
                     if r['baseline'] == list(baseline) and r['candidate'] == list(candidate)), None)

    def pair_row(label, row):
        if row is None:
            return [label, '无完整配对', '无完整配对', '未支持']
        clocks = row['clocks']
        ss, fresh = (clocks[k] for k in ('solver_seconds', 'fresh_process_return_seconds'))
        result = '支持' if row['isolated_practical_improvement_supported'] else '未支持'
        return [label, stat(ss) + (f"；{ss['candidate_wins']}/5" if ss else ''),
                stat(fresh) + (f"；{fresh['candidate_wins']}/5" if fresh else ''), result]

    def worst(rows, metric):
        return max(v[k] for row in rows for v in row['physical_worst'].values()
                   for k in (metric, 'near_' + metric))

    p('计算研究报告 / 2026 年 11 月 10 日', 'small')
    p('GPU 准备优化<br/>与同等精度复测', 'title')
    p('本轮针对上轮分段推进的初始化瓶颈，实施两项可独立核验的改动：'
      '把两组 CUDA 内核放入一个编译模块；共享三个步长相位不变的空间基和算子装配。'
      '保留无反射约束的 Re=100 有限圆环绕流、Galerkin 径向离散、Legendre 和优化四阶径向差分对照。')
    p('数值结果：24条短程消融轨迹逐位一致；完整时段新增26条分段资格和6条高档恒步资格。'
      '低档2条仍未达到速度与涡量同时0.5%的标准，中档18条、高档6条新分段轨迹和6条恒步控制通过。'
      '实现优化没有改善或放宽离散误差标准。')
    table(['独立批次', '登记 / 已完成', '数值资格', '隔离速度资格'], [
        [('中档' if scale == 'mid' else '高档'),
         f"{len(protocols[scale]['schedule'])} / {reports[scale]['rows']}",
         '全部通过' if reports[scale]['numerical_qualified'] else '未完整通过',
         '通过' if clean[scale] else '未通过；只作诊断'] for scale in reports], [82, 108, 125, 181])
    rows = []
    for scale in reports:
        for name in (*family[scale], 'fd176_k80'):
            arm = 'fd' if name.startswith('fd') else 'switch_both'
            rows.append([('中' if scale == 'mid' else '高') + ' / ' + names[name],
                         time_value(scale, name, arm), time_value(scale, name, arm, 'fresh_process_return_seconds')])
    table(['同批次候选 / 两项合并', '完整求解中位秒', '新进程至结果中位秒'], rows, [180, 145, 171])
    p('求解时钟包含每次独立构造、编译/加载、传输、FFT准备、图、推进、观测、同步、下载和释放；'
      '新进程时钟还包含解释器导入与身份握手。未摊销初始化。未通过隔离资格的批次，其时间不能用于确认提速。', 'small')
    p('这两项改动也用于Legendre。是否稳定更快由预先登记的双时钟配对规则判定；'
      '即使赢过本例FD4，也不能推论随机特征优于确定性谱方法或所有传统CFD。')
    p('本轮判断：单模块初始化显示可解释的诊断收益，但组合版相对原恒步方案的新进程收益很小，'
      '两档均未达到预登记的双时钟实用标准，也未取得完整隔离资格。保留候选实现与负面结果，不升级为稳定提速结论。', 'small')

    doc.new('01  空间不变量与共享准备的等价条件')
    p('固定几何、基族、种子、径向和角向分辨率、积分节点、边界提升与截断阈值后，'
      '基函数生成、列缩放和SVD坐标变换均发生在含步长的系统矩阵装配之前。'
      '因此三个时间相位可以共享这些空间量，而每个有效步长的线性系统仍独立构造。')
    p('RFM在此表示固定随机径向试验基，采用Galerkin投影求解时间系数；不是以逐点PDE残差训练网络的PINN。'
      'Legendre使用确定性多项式试验基，两族都保留相同边界层富集和角向Fourier模态。'
      '流函数表达的不可压结构、边界处理与原离散一致，准备复用不改变这些约束。')
    p('令s=ln(r/a)，F为变换后基函数，D=∂ₛF、D2=∂ₛ²F，W=diag(w)，k为角向模态。冻结弱式中的矩阵为：')
    p('Mₖ = DᵀWD + k²FᵀWF；<br/>'
      'J = diag(√w/r)(D2−F)，U = diag(√w/r)F；<br/>'
      'Kₖ = ν[JᵀJ − (k²−1)(JᵀU+UᵀJ) + (k²−1)²UᵀU]。')
    p('常规相位只改变A(h)=(3/2)M+hK，以及A(h)⁻¹作用后的历史、非线性和边界提升包。'
      '均值模态使用独立实值切向速度方程；其质量为Vᵀdiag(wr²)V，耗散由加权V+V′的Gram矩阵构成，V′也对s求导。'
      '共享实现保留其符号和提升处理，不能把均值模态当作复数k=0分量略去。')
    p('桥接使用A_bridge=a₀M+h_new K，a₀=(1+2ρ)/(1+ρ)、ρ=h_new/h_old。'
      '令c=(3/2)/a₀和h_eff=c h_new，则A(h_eff)=c A_bridge。历史质量及其提升乘c；'
      '通量包已含h_eff/N，未来提升也已按有效矩阵构造，不得再重复缩放。')
    p('实现只复用空间基、观测和不变的Gram装配。桥接、晚期仍计算原有带对角缩放的Cholesky逆及相位投影。'
      '起始模型仍由旧构造器装配；因其不保存局部Gram，桥接与晚段从已保存的基重建Gram一次并共用。'
      '保持原FP64表达式与求和分组的24组对照中，936项数组或系数的形状、类型和字节均相同。')
    p('这建立了固定离散的实现等价条件，不是新的NS稳定性定理。步长相关基、空间自适应或不同边界提升会破坏共享前提；'
      '换库和硬件后也须重新验证位级表现。完整推导见theory_shared_phase_preparation.txt。', 'small')

    doc.new('02  单模块 CUDA 与资源生命周期')
    p('原分段运行时把常规推进和桥接内核分别编译、加载。本轮把未经修改的两份CUDA源放入同一翻译单元，'
      '保留六个内核的名称、接口和计算表达式，NVRTC调用由2次变为1次。没有增加磁盘缓存，也没有把首次成本移到实验之前。')
    table(['验证层', '覆盖范围', '结果'], [
        ['设备内核', '6个内核、11个输出数组', '逐位一致'],
        ['主机相位准备', '24组配置、936项数组/系数', '逐位一致'],
        ['完整短程路径', '6种配置 × 4种准备方式，T=9', '24/24逐位一致'],
        ['资源释放', '模拟驱动的8处故障注入与正常退出', '通过'],
        ['推进源码', '只允许准备入口、类型和标记变化', '逆替换后字节一致']], [112, 252, 132])
    p('合模块运行时持有并释放主上下文的引用，恢复调用前上下文；退出按模块、流、原上下文和引用的顺序清理。'
      '没有重置共享设备上下文。模拟故障覆盖上下文、流、模块、符号查找、FFT和同步路径；'
      '真实GPU另做内核与完整轨迹检验。仅共享主机准备的消融仍使用旧运行时。')
    p('与原分段方案相比，每个时间步的非线性评价数、FFT批调用数、设备缓冲、图捕获和探针写入顺序均未改变。'
      '中档仍是8192步、8191次非线性评价、16382次推进FFT批调用；高档9920步、9919次评价、19838次推进批调用，'
      '另有不变的2次初始化FFT。')
    p('相对原恒步方案，中高档都少1024次非线性评价和2048次推进FFT批调用。这个收益来自此前固定分段调度，'
      '本轮解决的是为三相位准备付出的开销，不能再次把同一工作量下降算成新贡献。')
    p('API依据：<link href="https://docs.nvidia.com/cuda/archive/11.8.0/nvrtc/index.html" color="#195783">NVIDIA NVRTC 11.8</link>；'
      '<link href="https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html" color="#195783">CUDA Driver API 11.8主上下文管理</link>。'
      '文档支持接口语义；性能收益只能由本机实验给出。', 'small')

    doc.new('03  三档精度、独立参考与有限域边界')
    table(['配置 / 新准备方式', '条数', '最坏速度', '最坏涡量', '通过'], [
        [scale + ' / ' + name, str(len(rr)), pct(worst(rr, 'velocity')), pct(worst(rr, 'vorticity')),
         f"{sum(r['qualified'] for r in rr)}/{len(rr)}"]
        for scale, method_names in [('低档', ('rfm35_k64', 'leg35_k64')),
                                   ('中档', family['mid']), ('高档', family['high'])]
        for name in method_names
        for rr in [[r for r in qualification['rows'] if r['job']['name'] == name]]], [157, 42, 107, 107, 83])
    p('中档覆盖五个RFM种子和Legendre的三种新准备方式；高档覆盖五种子和Legendre的两项合并。'
      '低档只保留首种子与Legendre失败控制，不做低档五种子速度结论。高档六条原恒步GPU控制另行完整验收。')
    p('物理比较固定19个绝对时刻、3张独立评分网格、全域及r≤2.5近场的速度和涡量相对范数，两份参考均须通过。'
      '21时刻资格额外包含桥后的两个离格点；正式计时保存19个共同点。577点探针历史检查实现差和持续交替符号，不作相位对齐。')
    p('与CPU及原GPU的实现差须≤1e-5。中档和高档分段方案还检查相对于恒步控制的三角组合，'
      '避免仅凭误差抵消过关；最坏组合分别为0.47865%与0.36867%。这些都是相对于已计算参考和控制的数值差，不是PDE真误差证书。')
    p('独立参考分别加密径向、角向和时间：FD384/K128/r384 → FD448/K128/r384 → '
      'FD448/K144/r384 → FD448/K144/r512。早期、续算期和晚期的各轴合成最坏差约为0.00079%、0.01011%和0.02901%，'
      '满足0.5%档的0.125%预算。晚期尚未满足0.1%档的0.025%预算，不能报告0.1%速度优势。')
    p('差分FD152涡量约0.905%不合格；FD176稠密GPU约0.4088%，满足两门槛；FD184稠密路径虽满足物理门槛，'
      '却未通过1e-5实现差，保留为失败。既有带状FD184约0.3065%，通过并作为高档规模对照。')
    p('主问题为a=0.5、R=6、ν=0.01的有限圆环，外边界指定流函数、零涡量及均值Robin条件。'
      '历史R=8扩域产生明显场差，因此本报告的0.5%只能用于固定有限域离散问题。首步Euler启动，其后为二阶SBDF2；FD4指径向差分阶数。'
      't=4至12有固定横向触发，考察其结束后的持续交替脱落。', 'small')

    doc.new('04  计时协议与归因边界')
    p('使用Ryzen 9 9955HX和RTX 5060 Laptop，FP64与原工具链。每条求解新建进程，继承单逻辑核4和单线程数值库。'
      '5个完整块分别使用固定RFM种子7331、7687、8011、8353、9041；Legendre与FD使用17。块内顺序在取数前随机冻结。')
    table(['协议', '中档', '高档'], [
        ['求解次数', '55：每块11条', '40：每块8条'],
        ['Galerkin准备方式', '恒步、旧分段、仅模块、仅主机、组合', '恒步、旧分段、组合'],
        ['同批次差分', 'FD176稠密', 'FD176稠密 + FD184带状'],
        ['监测窗 / 采样间隔', str(protocols['mid']['isolation']['window_seconds'])+'秒 / 10秒',
                                    str(protocols['high']['isolation']['window_seconds'])+'秒 / 10秒'],
        ['整批监测结论', monitor_label('mid'), monitor_label('high')]], [113, 192, 191])
    p('同等精度允许不同方法使用不同空间配置与步长。两批都包含已测试达标的最低径向网格FD176，'
      '高档同时展示FD184，不以更细差分作为唯一基线。中高档之间不配对；所有速度比较只用同一批次、同块的记录。')
    p('求解时钟从进入选择的求解函数前开始，包含CUDA同步、下载、释放及模型释放；'
      '新进程时钟由父进程在启动前记录，到收到RETURN为止，额外包含导入、身份握手与空间指纹计算。'
      'RETURN先于轨迹写盘和评分，二者不计入两时钟。系统和驱动缓存未清空，因此不称为机器重启后的冷启动。')
    p('Windows监测用PID加100纳秒出生时间绑定真实进程，并保留退出CPU计数，避免把本队列刚退出的子进程错当外部任务。'
      '实际父子关系、READY/RETURN标记、顺序和终止时间共同验证；不能仅靠相同PID豁免。')
    p('整批只尝试一次。任何数值失败、未覆盖窗口或被监测到的外载，都使该批不能支持隔离速度结论，'
      '不删除受扰记录、不补跑较慢种子。采样不能证明10秒间隔内不存在短暂外载，监测主要覆盖Python和GPU端点。')
    for scale, monitor in monitors.items():
        unresolved = [s['target_elapsed_seconds'] for s in monitor.get('samples', [])
                      if s['snapshot'].get('unresolved_python') or s['snapshot'].get('snapshot_failure_type')]
        if unresolved:
            p(('中档' if scale == 'mid' else '高档') + '在监测起点后' + '、'.join(map(str, unresolved)) +
              '秒存在未解析完整身份的采样。求解是否按时完成与监测覆盖是否完整是两项不同检查。'
              '保留失败覆盖判定，不把原始状态名称解释为求解一定超时，也不事后删去缺口以升级速度结论。', 'small')
        active = [s['target_elapsed_seconds'] for s in monitor.get('samples', []) if s.get('external_activity_detected')]
        if active:
            p(('中档' if scale == 'mid' else '高档') + f'另有{len(active)}个样本检测到超阈值外部Python活动，'
              f'首末目标时刻为{min(active)}与{max(active)}秒。这也独立阻止隔离速度结论。', 'small')
    p('预登记“稳定且有实用幅度”的标准：每个时钟的5个配对均获胜，配对比中位数均≥1.05，且整批隔离合格。'
      '这是描述性验收规则，5个块不足以给出跨机器普遍概率保证。', 'small')

    doc.new('05  中档消融：模块、主机准备与组合')
    rows = []
    for arm in ('uniform', 'switch_old', 'switch_module', 'switch_host', 'switch_both'):
        rows.append([arms[arm], *[time_value('mid', name, arm, clock) for name in family['mid']
                                 for clock in ('solver_seconds', 'fresh_process_return_seconds')]])
    table(['实现', 'RFM求解秒', 'RFM进程秒', 'Leg求解秒', 'Leg进程秒'], rows, [148, 87, 87, 87, 87])
    comparisons = []
    for name in family['mid']:
        for tag, a, b in [('模块贡献', 'switch_old', 'switch_module'), ('主机贡献', 'switch_old', 'switch_host'),
                          ('组合贡献', 'switch_old', 'switch_both'), ('恒步→组合', 'uniform', 'switch_both')]:
            comparisons.append(pair_row(names[name] + ' / ' + tag, pair('mid', (name, a), (name, b))))
    table(['基线 / 候选', '求解比 [范围]；胜数', '进程比 [范围]；胜数', '隔离支持'], comparisons, [128, 151, 151, 66])
    p('比值均为同块“基线秒 / 候选秒”，大于1表示候选更快；先计算5个比值，再取中位数和范围。'
      '表中“未支持”可能来自不足5次获胜、幅度未达5%或隔离不合格，不能仅凭一列中位数越过1宣称稳定提速。', 'small')
    factorials = reports['mid'].get('factorials', [])
    if factorials:
        table(['交互项：旧−模块−主机+组合', '求解秒 [范围]', '进程秒 [范围]'], [
            [names[r['name']], stat(r['seconds']['solver_seconds']), stat(r['seconds']['fresh_process_return_seconds'])]
            for r in factorials], [198, 149, 149])
    p('交互项为0才符合两项时间收益简单相加。运行时初始化还包含上下文、编译、加载等操作，'
      '本轮只单列其总体阶段，不把整个差值强行解释为编译成本；主机准备同样不能细分为未测量的内部阶段。')

    doc.new('06  准备开销与推进开销分别改变多少')
    phase_rows = []
    for name in family['mid']:
        for label, baseline, candidate in [('仅模块', 'switch_old', 'switch_module'),
                                           ('仅主机', 'switch_old', 'switch_host'),
                                           ('组合', 'switch_old', 'switch_both'),
                                           ('恒步→组合', 'uniform', 'switch_both')]:
            contrast = pair('mid', (name, baseline), (name, candidate))
            phases = contrast['phase_delta_candidate_minus_baseline_seconds'] if contrast else {}
            phase_rows.append([names[name] + ' / ' + label, *[
                f"{phases[key]['median']:+.4f}" if key in phases and phases[key] else '无数据'
                for key in ('host_setup_seconds', 'runtime_init_seconds', 'march_seconds', 'solver_seconds')]])
    table(['候选相对基线', '主机设置', '运行时', '推进', '完整求解'], phase_rows,
          [146, 88, 88, 88, 86])
    p('单位为秒；每个数都是5个同块“候选−基线”的中位数，负值表示候选在该阶段少用时间。'
      '各阶段中位差之和未必等于总时间中位差，其余上传、计划、图、下载和释放仍包含在完整求解中。'
      '完整阶段原始值与配对范围保存在各批summary.json。')
    p('前三种比较使用完全相同的分段推进和FFT工作量，阶段差反映准备实现及测量波动。'
      '最后一种同时包含此前的分段调度，才有减少1024次非线性评价的工作量贡献；不能把这部分归因于新模块。')
    p('运行时阶段包括主上下文、NVRTC编译、模块加载和运行时对象等；计时没有进一步拆分，'
      '不将整个阶段差都解释成编译成本。主机准备保留三相位的不同有效矩阵，不是只构造一个矩阵用于所有步长。')
    p('这些是GPU完整路径内由CPU完成的准备工作。正式批次没有把CPU占用率作为优化目标；'
      '独立运行的目的是可靠测量同等精度下的完成时间。并行探索与隔离计时承担不同职责。')
    p('诊断上，合模块主要减少约0.146至0.149秒运行时初始化，单独共享空间准备减少约0.010至0.016秒主机设置。'
      '相对原恒步，组合仍额外花费约0.026秒主机设置、约0.050至0.052秒运行时准备；'
      '这些开销抵消了约0.118秒推进节省的大部分收益。新进程时钟相对求解时钟还多约2秒，'
      '含进程启动、导入、身份握手和空间指纹，未单独拆分，进一步压低相对收益。')

    doc.new('07  多档规模与同等精度 CFD 比较')
    comparisons = []
    for scale in reports:
        for name in family[scale]:
            comparisons.append(pair_row(('中档' if scale == 'mid' else '高档') + ' FD176 / ' + names[name],
                                        pair(scale, ('fd176_k80', 'fd'), (name, 'switch_both'))))
    for name in family['high']:
        comparisons.append(pair_row('高档 FD184 / ' + names[name],
                                    pair('high', ('fd184_k80', 'fd'), (name, 'switch_both'))))
    table(['同批次基线 / 候选', '求解比 [范围]；胜数', '进程比 [范围]；胜数', '隔离支持'], comparisons, [132, 149, 149, 66])
    table(['高档实现', '求解中位秒', '进程至结果中位秒'], [
        [names[name] + ' / ' + arms[arm], time_value('high', name, arm),
         time_value('high', name, arm, 'fresh_process_return_seconds')]
        for name in family['high'] for arm in ('uniform', 'switch_old', 'switch_both')]
        + [[names[name], time_value('high', name, 'fd'), time_value('high', name, 'fd', 'fresh_process_return_seconds')]
           for name in ('fd176_k80', 'fd184_k80')], [208, 130, 158])
    p('FD176是已测试且通过全部物理与实现检查的最低径向网格差分配置，两批都必须包含。'
      '本批诊断中，FD184带状实现的两钟中位均高于FD176稠密实现；因算法不同，仍不能只凭网格大小预判成本。'
      '这些是已测试配置之间的比较，不是对所有差分网格、线性求解器或谱方法的全局最优成本证明。')
    p('RFM与Legendre并非相同保留秩；它们是分别达到精度要求的实际配置。'
      '两者共享边界层富集、弱式、非线性算子、调度和准备优化。比较可以回答实际成本，'
      '还不能单独证明随机性本身的因果贡献。Legendre-Galerkin本身也是传统谱CFD。', 'small')

    doc.new('08  基族差异、工程判断与论文缺口')
    table(['组合：Legendre / RFM', '求解比 [范围]；胜数', '进程比 [范围]；胜数', '隔离支持'], [
        pair_row('中档' if scale == 'mid' else '高档',
                 pair(scale, (family[scale][1], 'switch_both'), (family[scale][0], 'switch_both')))
        for scale in reports], [132, 149, 149, 66])
    p('本轮不改写为Rust。已有CPU热点采用C/AVX2，GPU推进使用CUDA和cuFFT；'
      '新发现的重复空间装配和模块初始化可以在现有架构内消除。换一种主机语言不会自动减少FFT或改善离散误差。'
      'Rust可作为以后原生入口或调度层的候选，但当前没有其迁移收益的实测证据。')
    p('本轮新计时只比较完整GPU求解路径，包含CPU执行的空间准备。'
      '没有新增CPU单独隔离批次，也不能将旧协议CPU时间与新协议GPU时间相除当作新的加速比。'
      '既有隔离硬件批次曾测得RFM/Legendre/FD的CPU中位约8.689/8.693/21.064秒，GPU约1.297/1.317/2.835秒；'
      '它采用较少晚期观察点，作为历史工程证据另列，不并入本轮双时钟判断。')
    p('上轮60条分段队列虽然数值全部通过，但因外部活动只作诊断。'
      '其中GPU推进约少0.119秒，运行时初始化增加约0.164至0.198秒，总准备增加约0.206至0.240秒，完整求解反而更慢。'
      '本轮的两项改动正针对这一测到的准备瓶颈；旧失败结果和错误的进程分类诊断均保留。')
    p('尚未解决的数值问题包括：高角频显式逆应用的精度敏感性、0.1%档晚期参考预算、有限域边界影响，以及更一般几何的适用性。'
      '此前65位十进制审计提示条件数约1e7时，FP64显式逆应用比直接解更敏感；'
      '这不等于已证明其是早期误差放大的唯一原因。')
    p('Re=1000历史方腔使用光滑顶盖16x²(1−x)²，不是标准不连续恒速顶盖。'
      '本轮没有新增该例的隔离计时，不能用历史方腔结果弥补当前圆柱外域与跨几何证据的缺口。')
    sub('可复现贡献与下一步证据需求')
    p('可形成的明确贡献是：有等价条件和位级验证的准备复用、完整驻留GPU路径、'
      '同时约束速度与涡量的多档公平对照，以及保留失败的可审计测量。'
      '通用理论创新、随机特征独特优势、无限域标准圆柱准确度和跨硬件普适提速，均需要额外证据。')
    p('原研究归档不覆盖。新材料包含冻结协议、五种子、所有资格与计时轨迹、原始双时钟、进程身份与外载记录、'
      '共享准备推导、资源故障审计和报告生成程序。复现应写入新目录；短程重放验证实现，T72材料验证本轮数值结论。', 'small')
    p('冻结汇总函数错误地按嵌套job读取实际平坦记录，执行时失败。新增prepared_formal_summary.py先核对原始字段、'
      '协议与文件哈希，再仅在内存构造视图；未改计时数据或冻结代码。独立审核重算全部95条保存范数与47项配对，'
      '并从19条代表轨迹重新构造流场，复核两批诊断结论。', 'small')

    target = HERE.parents[1] / 'outputs' / 'RFM_NS_v7_GPU准备优化与公平复测_研究报告.pdf'
    pdf = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=49, leftMargin=49, topMargin=44, bottomMargin=53,
                            title='RFM NS GPU准备优化与公平复测', author='Computational research study')
    pdf.build(doc.story, onFirstPage=footer, onLaterPages=footer)
    (HERE / 'report_prepared_text.txt').write_text('\n\n'.join(doc.transcript), encoding='utf-8')
    print(target)

if __name__ == '__main__':
    main()
