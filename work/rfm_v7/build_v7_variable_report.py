"""Report the fixed single-bridge experiment, including all rejected controls."""
import json
from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'report_v2'))
import layout as doc


def read(path):
    return json.loads((HERE/path).read_text(encoding='utf-8'))


def percent(value):
    return f'{100*value:.4f}%'


def seconds(value):
    return f"{value['median']:.4f}" if value else '未合格'


def footer(canvas, info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49,43,546,43)
    canvas.setFont('CN',8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50,29,'不可压 NS | 分段步长与新进程计时 | 2026-11-10')
    canvas.drawRightString(545,29,str(info.page))
    canvas.restoreState()


def figure(name, caption):
    doc.story.append(Image(str(HERE/'variable_step_figures'/name),width=496,height=212))
    doc.p(caption,'cap')


def main():
    mid=read('switch_qualification_output/qualification.json')
    scales=read('multiscale_switch_qualification_output/qualification.json')
    fd=read('fd_early_multiscale_output/qualification.json')
    band=read('fd184_band_followup/qualification.json')
    early=read('early_reference_output/reference_audit.json')
    transition=read('early_reference_continuation_output/reference_audit.json')
    late=read('cpu_formal_point1_pow2_chain_output/reference_audit.json')
    timing=read('switch_formal_timing_output/timing.json')
    summary=read('switch_formal_timing_output/summary.json')
    monitor=read('switch_formal_timing_output/isolation.json')
    phases=read('switch_formal_timing_output/phase_diagnostics.json')
    assert mid['passes'] and len(mid['rows'])==18
    assert scales['summary']=={'low':{'qualified':0,'count':6},'high':{'qualified':18,'count':18}}
    assert len(fd['rows'])==6 and band['qualified']
    assert timing['status']=='complete' and len(timing['rows'])==60 and summary['rows']==60
    clean=summary['isolated_speed_claim_supported']
    groups={(r['name'],r['arm']):r for r in summary['groups']}
    labels={'rfm37_k72':'RFM37','leg37_k72':'Legendre37','fd176_k80':'FD176'}
    p,table,sub=doc.p,doc.table,doc.sub

    def pair(a,b,clock='solver_seconds'):
        return next(r[clock] for r in summary['paired'] if tuple(r['numerator'])==a and tuple(r['denominator'])==b)

    def physical(rows,metric):
        return max(v[k] for r in rows for v in r['physical_worst'].values() for k in (metric,'near_'+metric))

    def timed_worst(name,arm,metric):
        rows=[r for r in timing['rows'] if (r['name'],r['arm'])==(name,arm)]
        return max(v[k] for r in rows for v in r['physical_worst'].values() for k in (metric,'near_'+metric))

    def stat_text(value):
        if not value:
            return '未通过资格'
        return f"{value['median']:.3f} [{value['minimum']:.3f}, {value['maximum']:.3f}]"

    p('计算研究报告 / 2026 年 11 月 10 日','small')
    p('分段步长：更少 FFT<br/>与新进程计时','title')
    p('本轮保留无反射对称约束的 Re=100 有限圆环绕流、Galerkin 径向离散和完整角向模态。'
      '前8个时间单位沿用步率128，随后通过一次正确的变步SBDF2桥接转到112。'
      '所有登记物理时刻同时检查速度与涡量，不作事后相位对齐。')
    p('60条新进程队列已通过数值与完整观察窗验收，可以报告本机本协议下的速度比较。' if clean else
      '60条队列未获得新的隔离速度资格，以下时间均为诊断观察；不将其表述为已确认的提速。')
    selected=[(name,arm) for name in labels for arm in
              (('cpu_buffered128','cuda_graph128') if name.startswith('fd') else ('cpu_native_switch','cuda_graph_switch'))]
    table(['方法 / 硬件','最坏速度','最坏涡量','求解中位秒','进程至结果秒'],[
        [labels[n]+(' / CPU' if a.startswith('cpu') else ' / GPU'),percent(timed_worst(n,a,'velocity')),
         percent(timed_worst(n,a,'vorticity')),seconds(groups[n,a]['solver_seconds']),
         seconds(groups[n,a]['fresh_process_return_seconds'])] for n,a in selected],[128,88,88,96,96])
    p('表中误差取5次计时运行、两参考、19个时刻、三评分网格和全域/近场指标的最坏值；'
      '时间列分别取中位数。“求解”含设置、推进和资源释放；“进程至结果”还包含解释器与导入。'
      '轨迹写盘和评分放在完成标记之后。两种口径来自同一次求解。','small')
    p('中档每条轨迹由9216步减到8192步，非线性评价由9215次减到8191次，推进FFT批调用少2048次。'
      'CPU与GPU切换时都保留原始历史谱，没有新增非线性评价；CUDA另有不变的2次初始化FFT批调用。')
    p('该调度和算子改进同样用于Legendre。本轮不将共同改进归因于随机特征，也不把赢过本例FD4等同于赢过所有传统CFD。')

    doc.new('01  为什么保留细步启动，再做一次桥接')
    p('上轮直接全程用112步率时，晚期物理检查通过，但启动期不同实现的高角频误差被放大。'
      '本轮先独立建立早期参考，再固定t=8切换，未搜索多个切换时刻以挑选有利结果。')
    figure('implementation.png','CPU与CUDA的涡量实现差：每个时刻取三网格、全域与近场最大值。'
           '两种方法分别展示登记首种子/确定性种子；细步128与分段方案在启动期的曲线重合。虚线为1e-5门槛。')
    p('新增参考还说明：旧112的六个早期物理检查其实低于0.5%，但它仍未通过更严格的跨实现一致性门槛。'
      '两项标准回答不同问题，不能把“物理差尚小”改写为“数值实现已经稳健”，也不能把实现差直接称作PDE真误差。')
    p('分段方案的中档完整资格包含18条T72轨迹：RFM五个固定种子和Legendre，各用原CPU、预收缩CPU和CUDA。'
      '实现检查有21个快照，包含桥后的前两个离格时刻；物理误差只在与独立参考严格同一时刻的19点检查。'
      '还逐项检查相对于128步率控制的三角组合，避免仅靠误差抵消过关。')
    p('这是一种有条件通过的固定调度。标量零方程的桥接检验不证明完整NS无条件稳定，'
      '保存时刻通过也不是所有连续时间上的严格误差证书。')

    doc.new('02  变步公式、提升项与历史复用')
    p('设旧步长H、新步长h，r=h/H，m=M a+g M_l为含边界提升的质量，K为隐式耗散算子。'
      '一次桥接使用 a₀=(1+2r)/(1+r)，其方程为：')
    p('(a₀M+hK)a₊=(1+r)m₀−r²m₋/(1+r)+h[(1+r)f₀−r f₋]−g₊(a₀M_l+hK_l)。')
    p('中档 r=8/7、a₀=23/15；高档 r=19/17、a₀=55/36。桥后恢复等步SBDF2，'
      '使用真正相邻的两份历史。静止启动首步保留原隐式Euler；k=1提升与实值切向均值方程分别处理。')
    p('构造器原有质量系数3/2。令s=(3/2)/a₀，以有效步长s h构造，得到A_eff=s(a₀M+hK)。'
      '桥接右端和提升必须按同一比例处理。已有A_old⁻¹作用过的历史不能直接当作A_new⁻¹历史。')
    p('CPU预收缩路径在切换附近缓存两份原始非线性谱，并按桥接和新等步矩阵重投影。'
      'CUDA本来保留原始产品谱，三组系数表常驻显存，只在桥接步选择对应表；'
      'FFT计划与工作区复用。探针使用明确的时间槽，不沿用只适合恒步长的取模索引。')
    p("平滑条件下，导数近似的首项误差为−h(h+H)m'''/6，显式外推为−h(h+H)f''/2。"
      '对y′=0，差分满足Δ₊=[r²/(1+2r)]Δ₀，中档桥接因子64/161，其余步为1/3。'
      '这给出本网格的齐次差分有界性，未覆盖非正规输运、非线性耦合与时变强迫。')
    p('变步BDF2及其NS分析已有文献。本轮贡献是保持实际边界/均值和缓存一致的实现、'
      '可核对的工作量下降及扩展验收，不主张新的通用变步稳定性定理。','small')
    p('理论依据：<link href="https://link.springer.com/article/10.1007/s10543-018-0716-y" color="#195783">'
      'Söderlind等，BIT 2018</link>；<link href="https://arxiv.org/abs/2201.00527" color="#195783">'
      'Li与Liao，SIAM JNA 2022</link>；<link href="https://doi.org/10.1093/imanum/drac058" color="#195783">'
      'García-Archilla与Novo，IMA JNA 2023</link>。不同方法与扰动范数的结论不能直接套用于当前非线性离散。','small')

    doc.new('03  独立参考与完整时段的误差检查')
    table(['参考覆盖','独立轴合成最坏差','0.5%档预算','结果'],[
        ['t=0.5…8',percent(max(early['worst'].values())),'0.125%','通过'],
        ['t=0.5…16',percent(max(transition['worst'].values())),'0.125%','通过'],
        ['t=24…72',percent(late['composed_difference']),'0.125%','通过']],[145,147,102,102])
    p('每条链分别加密径向、角向和时间：FD384/K128/r384 → FD448/K128/r384 → '
      'FD448/K144/r384 → FD448/K144/r512。早期逐网格/时刻/指标组合各轴相对差，'
      '保留范数分子与分母；没有用小常数掩盖近零分母。T16续算从冻结的t6完整多步历史恢复，'
      '四臂均逐位重现原T8状态与探针。')
    figure('physical_accuracy.png','两参考、三网格及全域/近场中的最坏物理相对误差。'
           'RFM阴影为五种子范围、中线为中位数；Legendre为固定对照。按绝对时刻评分，无相位对齐。')
    p('中档RFM最坏参考差'+percent(physical([r for r in mid['rows'] if r['job']['name']=='rfm37_k72'],'vorticity'))+
      '，最坏保守三角组合'+percent(max(v for r in mid['rows'] if r['job']['name']=='rfm37_k72'
          for m in r['guard_worst'].values() for v in m.values()))+'。'
      '该组合是对已计算参考与控制的数值差约束，并非连续PDE严格误差界。')
    p('既有晚期参考链仍未满足0.1%档的0.025%预算。本轮早期参考更精细，并不自动补齐晚期0.1%资格。','small')

    doc.new('04  三档规模：通过项与失败项都保留')
    scale_rows=[]
    for label,records,names in [('低档',scales['rows'],('rfm35_k64','leg35_k64')),
                                 ('中档',mid['rows'],('rfm37_k72','leg37_k72')),
                                 ('高档',scales['rows'],('rfm41_k80','leg39_k80'))]:
        for name in names:
            rr=[r for r in records if r['job']['name']==name]
            assert rr, name
            scale_rows.append([label+' / '+name.split('_')[0],len({r['job']['seed'] for r in rr}),
                percent(physical(rr,'velocity')),percent(physical(rr,'vorticity')),
                f"{sum(r['qualified'] for r in rr)}/{len(rr)}"])
    table(['配置','种子数','最坏速度','最坏涡量','通过轨迹'],scale_rows,[162,50,99,99,86])
    p('低档为预先指定首种子与Legendre失败控制，不构成五种子的低档优势结论。'
      '中、高档均覆盖五个RFM种子。高档使用152→136调度，需9920步、9919次非线性评价；'
      '比全程152同样少1024次评价和2048次推进FFT批调用。外两档耗时是并发诊断，未用于速度排名。')
    fd_rows=[]
    for n in (152,176,184):
        rr=[r for r in fd['rows'] if r['job']['name']==f'fd{n}_k80']
        assert len(rr)==2
        worst=max(v[k] for r in rr for ref in r['physical'].values()
                  for v in (ref['early_worst'],ref['late_worst']) for k in ('vorticity','near_vorticity'))
        fd_rows.append([f'FD{n} / CUDA稠密',percent(worst),
                        '通过' if all(r['physical_pass'] for r in rr) else '未通过',
                        '通过' if all(r['implementation_pass'] for r in rr) else '未通过'])
    bw=max(v[k] for ref in band['physical'].values() for v in (ref['early_worst'],ref['late_worst'])
           for k in ('vorticity','near_vorticity'))
    fd_rows.append(['FD184 / CUDA带状',percent(bw),'通过','通过'])
    table(['四阶差分对照','最坏涡量','物理0.5%','实现1e-5'],fd_rows,[203,105,94,94])
    p('FD152物理不合格；FD184稠密CUDA虽达到物理门槛，但相对CPU的场差与探针差超过1e-5。'
      '单独冻结的既有带状解通过了FD184全部检查，稠密失败轨迹保留。中档正式计时仍用已通过的FD176稠密路径。','small')

    doc.new('05  随机顺序的新进程计时')
    p('硬件为Ryzen 9 9955HX与RTX 5060 Laptop。12个实验臂、5个固定种子块，块内随机顺序；'
      'CPU固定单逻辑核与单线程数值库。每臂单独新建Python进程，父进程等子进程退出和评分写盘完成后才启动下一臂。')
    arms=[('cpu_recurrence128','原CPU / 128'),('cpu_native128','融合CPU / 128'),
          ('cpu_native_switch','融合CPU / 分段'),('cuda_graph128','GPU / 128'),('cuda_graph_switch','GPU / 分段')]
    table(['实现','RFM求解秒','RFM进程秒','Leg求解秒','Leg进程秒'],[
        [label,*[seconds(groups[n,arm][clock]) for n in ('rfm37_k72','leg37_k72')
                  for clock in ('solver_seconds','fresh_process_return_seconds')]] for arm,label in arms],
        [172,81,81,81,81])
    p('求解时钟包括模型、基与系数构造，CUDA上下文、上传、FFT计划与初始化、图、推进、观测、同步、下载和释放。'
      '进程时钟从父进程创建子进程前，到收到求解完成标记为止；标记先于轨迹序列化。'
      '解释器导入期间已继承固定CPU亲和性。驱动与系统缓存未清空，因此属于新进程启动，不能称为机器重启后的冷启动。')
    p('全部计时轨迹保存同样的19个共同物理时刻，并逐条重新评分。两个离格桥接点由前面的21时刻资格覆盖，'
      '不冒称本计时的每一条都重新保存了这两个点。初始化没有跨求解摊销，随机特征没有跨任务训练成本被隐藏。')
    p('两段各10秒的安静检查后，固定监测900秒。记录协调器创建身份、子进程槽和串行关系，'
      '监测其他Python活动并检查GPU端点。只有整窗覆盖、全部数值资格及无登记的外部活动同时通过，才允许新的隔离速度结论。')
    p('本次观察窗状态：'+summary['monitor_status']+'。'
      '此前两批受扰动的50条队列不并入本次样本。本次也不替换较慢记录或追加复测。'
      '低频采样不能排除短于采样间隔的外载，也不构成所有系统进程完全空闲的证明。','small')

    doc.new('06  配对归因：算子、调度和端到端成本')
    comparisons=[]
    for n in ('rfm37_k72','leg37_k72'):
        for a,b,tag in [('cpu_recurrence128','cpu_native128','CPU算子'),
                        ('cpu_native128','cpu_native_switch','CPU分段'),
                        ('cuda_graph128','cuda_graph_switch','GPU分段')]:
            u=pair((n,a),(n,b)); v=pair((n,a),(n,b),'fresh_process_return_seconds')
            comparisons.append([labels[n]+' / '+tag,stat_text(u),stat_text(v),f"{u['denominator_wins']}/5" if u else '未合格'])
    table(['旧 / 新','求解比中位 [范围]','进程比中位 [范围]','新求解更快'],comparisons,[133,147,147,69])
    p('每个比值先在同块同种子配对，再报告5个比值的中位数与范围。它不是两组中位数相除，'
      '也不是统计置信区间。原CPU/融合CPU保持128步率；融合CPU与GPU各自的分段对照保持空间离散和算子实现不变。')
    rows=[]
    for n in ('rfm37_k72','leg37_k72'):
        for fdarm,arm,label in [('cpu_buffered128','cpu_native_switch','CPU'),('cuda_graph128','cuda_graph_switch','GPU')]:
            a=('fd176_k80',fdarm);b=(n,arm)
            rows.append(['FD176 / '+labels[n]+' '+label,stat_text(pair(a,b)),
                         stat_text(pair(a,b,'fresh_process_return_seconds'))])
    table(['同硬件 CFD / Galerkin','求解比中位 [范围]','进程比中位 [范围]'],rows,[190,153,153])
    p('以上'+('可用于本机本协议的速度判断。' if clean else '仅是受限诊断观察，不支持隔离速度判断。')+
      '新进程包含共同的解释器/导入固定成本，因而求解器级加速不会等比例变成完整启动加速。'
      '单纯替换宿主语言不能自动减少非线性评价、FFT次数或离散自由度。')
    legcpu=pair(('leg37_k72','cpu_native_switch'),('rfm37_k72','cpu_native_switch'))
    leggpu=pair(('leg37_k72','cuda_graph_switch'),('rfm37_k72','cuda_graph_switch'))
    p('随机特征的对照：分段CPU的Legendre/RFM求解比为'+stat_text(legcpu)+'，GPU为'+stat_text(leggpu)+'。'
      '两族共享边界层富集、弱离散、预收缩与GPU实现；中档截断秩39与41也不相等。'
      '这些差别不能被直接解释为随机性的因果收益。')

    doc.new('07  为什么 GPU 少做 FFT，却没有更快')
    p('以下均为受后台负载影响的诊断数据。按相同块配对后计算“分段减恒步”的阶段耗时差，'
      '再对五块取算术平均；正值代表分段多花时间。均值分解可相加，但不能由此断言干扰已被消除。')
    phases_keys=[('host_setup_seconds','主机模型与系数'),('runtime_init_seconds','运行时初始化'),
                 ('upload_seconds','上传'),('plan_seconds','FFT计划与初始化'),
                 ('graph_setup_seconds','计算图准备'),('march_seconds','推进'),
                 ('download_seconds','下载'),('model_release_seconds','释放'),
                 ('unattributed_solver_seconds','未单独计时部分'),('solver_seconds','完整求解'),
                 ('fresh_process_return_seconds','新进程至结果')]
    table(['分段减恒步 / 秒','RFM37 五块均差','Legendre37 五块均差'],[
        [label,*[f"{phases['contrasts'][n]['gpu_schedule']['mean_delta_seconds'][key]:+.5f}"
                  for n in ('rfm37_k72','leg37_k72')]] for key,label in phases_keys],[230,133,133])
    p('两族推进阶段各省约0.119秒；准备阶段合计却分别多约0.240秒和0.206秒，'
      '完整GPU求解在五个配对块中均更慢。该版本虽减少非线性评价次数，不能作为已经更快的单次GPU求解方案。')
    p('分段实现额外创建桥接CUDA模块、准备三套相位算子并捕获两段图。'
      '运行时初始化包含编译与加载等工作，当前计时没有将它们独立分开，不能把整段差值都算作编译成本。'
      '主机准备也同时包含模型、相位系数和预收缩，不能伪造更细的归因。')
    p('下一项有根据的工程方向是合并或缓存桥接内核的编译/加载，复用不同相位不变的算子部分。'
      '需要另行冻结协议并验证完整启动成本，尚未实施或取得提速证据。若缓存依赖重复运行，必须单列首次与后续成本。')
    p('CPU已经将重复的小规模算子收缩移到C/AVX2；GPU热路径使用CUDA与cuFFT。'
      'Rust可以承担将来适合的调度或原生接口，但改语言不会自动消除这些初始化、FFT和数值精度成本。'
      '当前证据支持针对算子及其生命周期继续优化，不支持全项目改写。')
    p('新进程比求解时钟多出的约2秒也值得研究。复用进程需要另列首次成本；'
      '若改用Rust或C++原生入口，还必须迁移当前Python负责的初始化才能避免其导入。'
      '这些策略应同等用于RFM、Legendre和差分基线，目前没有相应实测收益。','small')

    doc.new('08  可靠性发现、复现与尚未解决的问题')
    p('两个更严格的固定随机状态筛查曾失败，原记录保留。65位十进制对同一已装配FP64矩阵和右端复核后，'
      'RFM随机k=63的条件数约1.31e7；既有显式逆应用的相对前向差约5.72e-9，直接FP64解同一矩阵约1.33e-11。'
      '真实相邻启动状态的主要模态误差更小。这个审计定位了既有求逆应用的精度限制，未靠放宽完整轨迹1e-5门槛获得通过。')
    p('高条件数和高频误差增长具有相容的解释，但尚未证明它们就是旧112启动敏感性的唯一机制。'
      '以后若改线性求解顺序或系数缩放，应同时检查原方程后向误差、物理场与长期相位，不能只展示矩阵残差下降。')
    sub('当前结论的边界')
    p('主问题是半径0.5、外半径6的二维有限圆环，ν=0.01；外边界指定流函数和零涡量。'
      '历史R8域扩展已显示明显场差，因而0.5%只是这一固定域问题的离散比较，不能外推为无限域圆柱外流精度。'
      '所有时间推进仍为二阶SBDF2，“四阶CFD”指径向差分阶数。')
    p('Legendre-Galerkin本身属于确定性谱CFD。本轮验证了共享离散与实现的数值资格，新的效率幅度仍需合格的隔离计时。'
      '随机特征是否优于同等优化的确定性基也需单独证据。'
      '本次正式速度比较在中档；外两档给出准确度与实现边界，不能把并发耗时当成多档正式速度曲线。'
      '跨机器、其他非稳态几何及更高精度参考仍是论文结论的缺口。')
    p('先前Re=1000补充方腔采用光滑顶盖而非常见不连续恒速顶盖，原报告和失败记录继续保留，'
      '不将其历史耗时混入本次新进程圆柱比较。')
    sub('复现材料')
    p('数值协议、全部种子、原始轨迹、两级计时、外载采样和失败分支均按哈希归档。'
      'replay_switch_round.py提供不依赖任务管理文件的独立数值入口；原管理输入的历史哈希保留为来源记录，正文不发布。'
      '最小复现只核对短程实现，不能替代T72资格。用户应向新目录写重算结果，避免覆盖冻结证据。')
    p('本轮主要数据目录：early_reference_output、early_reference_continuation_output、switch_qualification_output、'
      'multiscale_switch_qualification_output、fd_early_multiscale_output、fd184_band_followup及switch_formal_timing_output。'
      '完整理论与固定方程审计见theory_sbdf2_single_bridge.txt和cpu_sbdf2_switch_pilot_output。','small')

    target=HERE.parents[1]/'outputs'/'RFM_NS_v7_分段步长与新进程计时_研究报告.pdf'
    pdf=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=44,bottomMargin=53,
                          title='RFM NS 分段步长与新进程计时',author='Computational research study')
    pdf.build(doc.story,onFirstPage=footer,onLaterPages=footer)
    (HERE/'report_variable_text.txt').write_text('\n\n'.join(doc.transcript),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
