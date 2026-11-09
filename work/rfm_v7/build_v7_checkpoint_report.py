"""Render the completed 0.5% study; distinguish later candidates from measured wins."""
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


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pct(x):
    return f'{100*x:.5f}%'


def link(url, title):
    return f'<link href="{url}" color="#195783">{title}</link>'


def footer(canvas, info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49,43,546,43)
    canvas.setFont('CN',8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50,29,'RFM / Legendre / FD4  |  第 7 轮阶段报告  |  2026-11-10')
    canvas.drawRightString(545,29,str(info.page))
    canvas.restoreState()


def main():
    out = ROOT/'cpu_formal_output'
    measured = read(out/'joint_isolated.json')
    summary = read(out/'joint_summary.json')
    qualified = read(out/'qualification.json')
    audit = read(out/'operator_audit.json')
    protocol = measured['frozen']['protocol']
    if measured['status']!='complete' or len(measured['rows'])!=25 or not all(r['passes'] for r in measured['summary']):
        raise RuntimeError('Require the entire qualified frozen 25-run study')
    p, table, sub = doc.p, doc.table, doc.sub
    p('计算研究报告 / 2026 年 11 月 10 日 / 已完成结果与在研工作分列','small')
    p('相对于冻结参考解的 0.5% 门槛<br/>CPU 与 CUDA 完整求解比较','title')
    p('无反射对称约束的 Re=100 圆柱绕流，保留角向平均动量和交替涡脱落。'
      '这一阶段完成了五轮随机交错的隔离计时。速度与涡量在三个评分网格、七个绝对时刻、全域和近场同时通过验收。')
    p('最新参考审计发现：在更高分辨率 n384→448 时，四阶 FD 的长期涡量差达到0.14192%，呈非单调行为。'
      '下面的数据相对于原冻结 FD320 参考成立；连续方程精度与0.1%档结论须等待该异常查明。','small')
    labels = {'rfm37_k72':'RFM37', 'leg37_k72':'Legendre37', 'fd176_k80':'四阶径向 FD176'}
    rows = []
    for r in measured['summary']:
        rows.append([labels[r['name']], 'CPU' if r['backend']=='cpu' else 'CUDA',
            f"{r['median_seconds']:.3f}", f"{r['min_seconds']:.3f} - {r['max_seconds']:.3f}",
            pct(max(r['worst']['velocity'],r['worst']['near_velocity'])),
            pct(max(r['worst']['vorticity'],r['worst']['near_vorticity']))])
    table(['方法','硬件','中位秒','5 次范围 / 秒','最坏速度','最坏涡量'],rows,[103,43,58,110,91,91])
    p('25/25 次通过，运行失败率为 0。RFM 使用五个预定种子，确定性方法等次数重复。'
      'CPU 上 RFM/Legendre 均比当前四阶 FD 基线快约 3.3 倍；RFM 仅在 3/5 个配对块快于 Legendre，随机特征没有显示稳健的额外优势。')
    p('同一 RFM 离散从 CPU 移至 CUDA，中位完整耗时改善 6.57 倍，Legendre 为 6.36 倍。'
      '这是已验证的实现收益。跨硬件 GPU-RFM / CPU-FD 的 21.62 倍不能当作随机特征的算法优势。','small')
    p('本表保留完成时的代码版本。后续更快的 Legendre 构造、CPU FD 融合和 GPU FD 路径已经进入验证，'
      '尚未纳入这张冻结计时表，不能把表中 FD 称为最终优化后的基线。','small')

    doc.new('01  比较协议与参考解可信度')
    p('圆柱半径 a=0.5，外边界 R=6，Re=100，ν=0.01。由静止启动，主流采用平滑三次起始爬升；'
      '4&lt;t&lt;12 施加峰值 0.12 的横向来流脉冲。之后保留完整角向自由度，均值流由切向动量方程演化。'
      '外边界指定流函数并令涡量为零。该有限环域问题与无限域圆柱绕流有不同边界。')
    p('评分时刻固定为 t=24、36、48、54、60、66、72。三套独立面积加权网格为 Gauss123、'
      '偏移中点129和外层加密144。速度采用两分量联合 L² 范数，涡量采用实际旋度的 L² 范数，'
      '分别除以同一区域同一时刻的参考范数。全域与 r≤2.5 近场四类误差均不超过0.5%；全程未作相位对齐。')
    ref = measured['frozen']['reference']
    table(['独立加密','配置变化','最大四指标变化'],[
        ['径向','n280 → n320；K104，rate256',pct(max(ref['edges'][0]['worst'].values()))],
        ['角向','K104 → K112；n320，rate256',pct(max(ref['edges'][1]['worst'].values()))],
        ['时间','rate256 → rate320；n320，K112',pct(max(ref['edges'][2]['worst'].values()))],
        ['组合筛查','相邻相对变化按乘积合成',pct(ref['composed_difference'])]
    ],[75,306,115])
    p('这一段局部加密链的组合差异低于预设0.125%筛查值，即门槛的四分之一。它是经验依据，'
      '不能充当相对连续 PDE 真解的严格误差上界。随后n384→448的非单调变化表明，必须继续审计高分辨率参考的数值可靠性。'
      '0.1%档的0.025%参考预算已经失败，当前不作这一档速度结论。')
    p('FD 的“四阶”指径向差分精度，角向为 Fourier；三种方法在本表中都用二阶 SBDF2 时间推进。'
      '各方法可独立选择达到误差门槛的网格与步长。这里的获胜范围是已登记的候选集合，未证明全局最优。','small')

    doc.new('02  三档分辨率与随机种子')
    specmap = {s['name']:s for s in protocol['variants']}
    candidates = []
    for r in qualified['summary']:
        s = specmap[r['name']]
        raw = next(x for x in qualified['rows'] if x['name']==r['name'])
        matches = [x for x in qualified['rows'] if x['name']==r['name'] and 'errors' in x]
        e = max(max(x['errors'].values()) for x in matches)
        rad = str(s['n']) if 'n' in s else f"{s['m']} + 6 / {s['q']}"
        candidates.append([r['name'],rad,s['k'],s['rate'],pct(e),'通过' if r['numerical_pass'] else '未通过'])
    table(['配置','n 或 m+层 / q','K','步/单位','最坏四指标','0.5%'],candidates,[123,103,37,57,101,75])
    p('RFM 在每一档都使用同一组预先固定种子：7331、7687、8011、8353、9041。'
      '较低档 RFM35、Legendre35、FD152 未通过；首个合格档分别为 RFM37、Legendre37、FD176。'
      '资格测试期间允许并行，其耗时不用于正式速度结论。')
    p('正式测量在同一 Ryzen 9 9955HX 上固定 CPU 逻辑核心并控制计算库线程。'
      '五个块内随机交错 CPU-RFM、CPU-Legendre、CPU-FD、CUDA-RFM、CUDA-Legendre。'
      '首个 CPU-RFM 测量较慢，也保留在范围和原始数据中。')
    sub('为何不能只看速度误差')
    p('流函数误差通过一阶导数进入速度，通过二阶导数及 k² 因子进入涡量。'
      '被截去的高频和外壁薄层会先在涡量指标暴露。相对采样参考场的离散 Fourier 投影下界表明：'
      'K72 的最坏涡量尾能量已有0.11198%，所以这个角向空间不可能达到0.1%。'
      'K88 与 K96 的对应下界约0.02741%与0.01319%，为下一轮选型提供依据。')

    doc.new('03  GPU 收益与计时范围')
    rfm = next(r for r in summary['median_stage_breakdowns'] if r['backend']=='cuda_graph' and r['name']=='rfm37_k72')
    stages = rfm['stages']
    titles = {'host_setup_seconds':'基函数与主机算子构造','runtime_init_seconds':'CUDA 初始化、编译和模块载入',
        'upload_seconds':'状态与算子上传','plan_seconds':'FFT 计划与准备','graph_setup_seconds':'图构建',
        'march_seconds':'同步后的完整时间推进','download_seconds':'检查点与探针返回'}
    table(['阶段','五轮中位秒'],[[title,f'{stages[key]:.6f}'] for key,title in titles.items() if key in stages],[360,136])
    p('每次重新构造模型、分解与预收缩算子，重新初始化运行时、上传数组、建立 FFT 计划和图；'
      '计时覆盖最后同步、结果返回和资源清理。各阶段中位数之和不必等于总耗时中位数。'
      '这是同一 Python 进程内的新建求解器，未清除解释器、驱动磁盘编译缓存；不能称为操作系统级冷进程测量。')
    p('FP64 状态、历史量、算子和输出常驻显存，24 步图执行批量提交。质量投影、非线性投影与隐式算子在设置阶段预收缩。'
      '完整9216步轨迹与对应 CPU 的状态和探针差异，RFM 最多1.09×10⁻⁹，Legendre 最多1.75×10⁻¹⁰。')
    sub('FFT 次数与同步的实际变化')
    p('本轮仍有9215次非线性评价，每次三组逆变换和两组正变换。'
      '图执行降低提交、分配和同步开销，没有减少数学上的 FFT 次数。'
      'Legendre 得到相同的驻留、预收缩和图执行优化，因此这项收益应归于实现。')
    p('CuPy 路径在完整轨迹上数值一致，但并发试跑约23.22秒；其当前批量矩阵调用拒绝图捕获。'
      '原生驱动、cuFFT 与自定义内核的完整路径因此被保留。该负面结果只适用于测试的软件版本和实现，不能推广为库的一般性能结论。','small')

    doc.new('04  理论与实际算子的对应')
    p('令 s=log(r/a)。非零模态速度取 u<sub>r</sub>=ψ<sub>θ</sub>/r、u<sub>θ</sub>=−ψ<sub>s</sub>/r，再加平均切向速度 v₀(s)。'
      '解析散度中的混合导数抵消。RFM 与 Legendre 都继承这个结构，不可压性不是随机特征独有。')
    p('离散质量形式为 M<sub>k</sub>=Σw(f<sub>s</sub>f<sub>s</sub><super>T</super>+k²ff<super>T</super>)，扩散形式为 '
      'K<sub>k</sub>=νΣw(f″−k²f)(f″−k²f)<super>T</super>/r²。正权求积与保留采样秩给出正定质量和半正定扩散；'
      'A<sub>k</sub>=1.5M<sub>k</sub>+ΔtK<sub>k</sub> 可解。正定性本身不保证求积精度或好的条件数。')
    p('齐次自对流配对中，非零模态留下 ∫r²v₀u<sub>r</sub>ω；均值动量的 −平均(u<sub>r</sub>ω) 配对恰好抵消。'
      '该恒等式要求公共节点、公共权重、相同状态、相容投影与 N&gt;3K 的二次去混叠。'
      '非齐次边界提升仍可做功。')
    table(['配置','最大预收缩相对残差','最大隐式缩放条件数','齐次功归一化残差'],[
        [r['name'],f"{r['worst_operator_defect']:.2e}",f"{r['maximum_scaled_implicit_condition']:.2e}",
         f"{r['worst_normalized_homogeneous_work']:.2e}"] for r in audit['rows']],[112,133,125,126])
    p('预收缩 B=A⁻¹M、G=A⁻¹T 将矩阵乘法和隐式更新合并；精确算术下等价，'
      '有限精度下检查 AB−M、AG−T 和完整轨迹。实际结果支持当前配置的等价性，未证明任意病态配置都安全。')
    sub('全离散稳定性尚未证明')
    p('SBDF2 使用外推非线性 2F(c<sub>n</sub>)−F(c<sub>n−1</sub>)，它与新状态的功不等于齐次自对流功。'
      '因此不能从半离散抵消推出无条件稳定。旧 SBDF3 两档试跑在 t=2 与2.75 产生非有限值，未被晋升。'
      '现阶段的稳定性证据来自指定步长下的完整轨迹和独立时间加密。')

    doc.new('05  CPU 吞吐量与计算域影响')
    throughput = []
    for path in out.glob('*throughput*.json'):
        data = read(path)
        if data.get('status')=='complete' and 'samples' in data:
            groups = {}
            for sample in data['samples']:
                key = (sample['workers'],sample['blas_threads'])
                groups.setdefault(key,[]).append(sample['jobs_per_second'])
            for (workers,threads),rates in groups.items():
                horizon = data.get('horizon',data.get('frozen',{}).get('horizon'))
                if horizon is None:
                    raise RuntimeError(f'Missing throughput horizon in {path.name}')
                throughput.append((horizon,workers,threads,statistics.median(rates),min(rates),max(rates)))
    kept = [r for r in throughput if r[0]==8 or (r[0]==2 and r[1] in (1,12) and r[2]==1)]
    table(['积分区间','进程数','BLAS线程','任务/秒中位','样本范围'],[
        [f'0 - {h}',w,t,f'{m:.3f}',f'{lo:.3f} - {hi:.3f}'] for h,w,t,m,lo,hi in sorted(kept)],
        [95,75,85,116,125])
    p('这些是每批12个真实 RFM、Legendre、FD 混合任务的完成率，含进程启动。'
      '12进程明显提高短批吞吐量；延长积分后，1与2个 NumPy BLAS 线程的样本范围重叠，'
      '最优线程数尚不稳健。正式单任务计时停止这些竞争工作。利用率只作辅助，不通过空转增加占用。')
    sub('外边界不确定性大于本轮数值门槛')
    p('独立加密的 R=8 计算中，涡脱落 Strouhal 数约0.1737015，R=6约0.1755584。'
      'R=8 自身加密的最坏全域涡量变化约0.02669%；两域在公共 r≤4 区域的未对齐涡量差最大约50.39%。'
      '长时相位差会放大这一指标，它同时提醒我们：0.5%的数值误差控制并未消除有限外边界影响。')
    p('本文的计时结论仅适用于这些明确的有限边界、二维层流和参数。未完成无限域外推，'
      '也未据此推断三维、高雷诺数湍流或一般复杂几何上的方法优势。','small')

    doc.new('06  在研改良与论文需要补齐的证据')
    table(['路线','已经验证','尚未进入正式结果'],[
        ['GPU 四阶 FD','两种线性路径通过短程；稠密路径完整轨迹同时过0.5%门槛',
         '同硬件、随机交错、端到端隔离计时；不能引用并发试跑速度作最终比值'],
        ['CPU FD 融合','随机复状态与历史缓冲核对；完整t=72场和探针与基线逐位一致',
         '融合作用的正式隔离消融，以及更快基线下的同精度复测'],
        ['基函数构造递推','保持原数学空间；完整轨迹校验通过',
         '减少Legendre初始化差异后的CPU/CUDA共同复测'],
        ['尺度切向增广','对相容指数层求衰减率导数，边界条件自动保持',
         '与普通层的误差/条件数权衡；不能因理论合理就宣布改良成功'],
        ['0.1%参考与候选','更细参考的径向变化0.14192%，未通过0.025%预算',
         '查明高分辨率非单调误差，再作参考与候选资格判断；暂不计时宣称获胜']
    ],[86,205,205])
    p('较有希望的贡献是：边界层相容的 Galerkin 离散、可核查的能量结构，以及完整 FP64 CUDA 求解器。'
      '要形成可信论文，还需补齐更快 FD 基线下的比较、多个规模的收益趋势、一个检验适用边界的补充算例、'
      '独立复现与完整版本归档。随机特征相对 Legendre 的优势仍应由数据决定。')
    p('额度规则：读取实际账户使用窗口，在第一次发现剩余低于50%时将当时一轮定为最后一轮；'
      '完成已冻结实验、分析和可复现性检查，再上传新的私有 GitHub 仓库。该报告不表示已经发布或完成整个研究。','small')
    sub('文献与复现依据')
    p(link('https://arxiv.org/abs/2207.13380','Chen、Chi、E、Yang：Random Feature Method（2022）')+'。'
      '原始框架以随机特征、配点与边界罚项为主；本文采用强制相容边界与 Galerkin，须明确区分。','small')
    p(link('https://arxiv.org/abs/2304.06913','Chen、E、Luo：Time-dependent Random Feature Method（2023）')+'。'
      '其时空随机特征误差分析不能直接替代本文的 SBDF2 长期稳定性证明。','small')
    p('核心数据：protocol_cpu.json、qualification.json、joint_isolated.json、joint_summary.json、'
      'operator_audit.json、domain_sensitivity.json，以及独立参考与原始轨迹NPZ。'
      '完整算法推导与假设见 theory_v7.txt；本报告只列影响结论的要点。','small')
    destination = ROOT.parents[1]/'outputs'/'RFM_NS_v7_0.5门槛与CUDA复测_阶段报告.pdf'
    destination.parent.mkdir(parents=True,exist_ok=True)
    pdf = SimpleDocTemplate(str(destination), pagesize=A4,
        rightMargin=49,leftMargin=49,topMargin=44,bottomMargin=53,
        title='RFM NS 第7轮：0.5%门槛与CUDA完整求解阶段报告', author='Computational research study')
    pdf.build(doc.story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'report_checkpoint_text.txt').write_text('\n\n'.join(doc.transcript),encoding='utf-8')
    print(destination)


if __name__=='__main__':
    main()
