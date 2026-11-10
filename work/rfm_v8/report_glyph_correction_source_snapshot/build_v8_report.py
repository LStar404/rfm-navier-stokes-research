"""Build the v8 scientific report from preserved evidence and reviewed narrative.

Run only after the final-round addendum exists. --draft produces a clearly
marked local preview, never the publication filename.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / 'report_v2'))
import layout as doc

original_glyphs = doc.glyphs
def scientific_glyphs(text):
    for symbol, markup in [('ᵀ', '<super>T</super>'), ('ₖ', '<sub>k</sub>'),
                           ('ₛ', '<sub>s</sub>'), ('ₙ', '<sub>n</sub>')]:
        text = text.replace(symbol, markup)
    return original_glyphs(text)
doc.glyphs = scientific_glyphs

INPUTS = {}
PUBLIC_RESOLUTIONS = []
PUBLIC_PACKAGE_VERIFIED = False
PUBLIC_MANIFEST_SHA256 = None


def resolve_review_source(relative, expected):
    """Accept an explicit public derivative only after full package verification."""
    global PUBLIC_PACKAGE_VERIFIED, PUBLIC_MANIFEST_SHA256
    original = ROOT / relative
    if not original.resolve().is_relative_to(ROOT):
        raise ValueError('Review input escaped research workspace')
    if original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest() == expected:
        return original, expected
    manifest_path = ROOT / 'MANIFEST.json'
    map_names = ('publication_path_redaction_map.json', 'publication_historical_source_map.json')
    if not manifest_path.is_file() or not any((HERE / n).is_file() for n in map_names):
        raise ValueError('Reviewed source changed or missing: ' + relative)
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if PUBLIC_PACKAGE_VERIFIED and manifest_digest != PUBLIC_MANIFEST_SHA256:
        raise ValueError('Public manifest changed after verification')
    manifest = json.loads(manifest_bytes)
    if not PUBLIC_PACKAGE_VERIFIED:
        seen = set()
        for row in manifest['files']:
            name = row['path']
            path = ROOT / name
            if (name in seen or not path.resolve().is_relative_to(ROOT) or path.is_symlink() or
                    path.stat().st_size != row['bytes'] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
                raise ValueError('Public snapshot integrity failed: ' + name)
            seen.add(name)
        PUBLIC_PACKAGE_VERIFIED = True
        PUBLIC_MANIFEST_SHA256 = manifest_digest
    manifested = {row['path']: row['sha256'] for row in manifest['files']}
    choices = []
    for name in map_names:
        path = HERE / name
        if not path.is_file():
            continue
        key = path.relative_to(ROOT).as_posix()
        if manifested.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Public evidence map lacks manifest authentication')
        mapping = json.loads(path.read_text(encoding='utf8'))
        if mapping.get('schema') not in ('rfm-v8-public-derivative-map-1', 'rfm-v8-historical-source-map-1'):
            raise ValueError('Unknown public evidence map schema')
        choices.extend((key, row) for row in mapping['records']
                       if row['original_path'] == relative and row['original_sha256'] == expected)
    if len(choices) != 1:
        raise ValueError('Public source resolution must be unique: ' + relative)
    map_name, row = choices[0]
    if (map_name.endswith('publication_historical_source_map.json') and
            row['original_sha256'] != row['public_sha256']):
        raise ValueError('Historical source alias must preserve exact original bytes')
    path = ROOT / row['public_path']
    if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
        raise ValueError('Public derivative escaped research workspace')
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if (manifested.get(row['public_path']) != observed or observed != row['public_sha256']):
        raise ValueError('Public derivative hash mismatch')
    PUBLIC_RESOLUTIONS.append(dict(original_path=relative, original_sha256=expected,
                                   public_path=row['public_path'], public_sha256=observed, mapping=map_name))
    return path, observed


def read(name):
    p = HERE / name
    INPUTS[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8-sig'))

def bind_review(name, field):
    review = read(name)
    for relative, expected in review[field].items():
        path, observed = resolve_review_source(relative, expected)
        INPUTS['../../' + path.relative_to(ROOT).as_posix()] = observed
    return review

def figure(name, width=496):
    path = ROOT / 'outputs' / 'rfm_v8_figures' / name
    INPUTS['../../outputs/rfm_v8_figures/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with PILImage.open(path) as im:
        height = width * im.height / im.width
    doc.story.append(Image(str(path), width=width, height=height))

def footer(canvas, info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49, 43, 546, 43)
    canvas.setFont('CN', 8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50, 29, '不可压 NS | 算子改良、理论审计与公平比较 | 2026-11-10')
    canvas.drawRightString(545, 29, str(info.page))
    canvas.restoreState()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()
    final_path = HERE / 'report_final_round.json'
    if not args.draft and not final_path.exists():
        raise SystemExit('Final-round evidence addendum required for a deliverable report.')
    final = read(final_path.name) if final_path.exists() else None
    if final and not args.draft and not final.get('reviewed_for_publication'):
        raise SystemExit('Final addendum has not passed independent review.')
    if final and not args.draft:
        review = read('report_final_round_publication_review.json')
        if (review.get('status') != 'passed' or
                review.get('addendum_sha256') != hashlib.sha256(final_path.read_bytes()).hexdigest() or
                review.get('builder_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise SystemExit('Final report content does not match its independent publication review.')
    thin = read('thin_formal_high_verification.json')
    mid = read('thin_formal_mid_verification.json')
    fused = read('fused_march_t9_output/summary.json')
    leg = read('legendre_rank_gpu_qualification_output/summary.json')
    read('galerkin_phase_pencil_audit_canonical_mean.json')
    read('nonlinear_stride2_scalar.json')
    bind_review('theory_independent_review.json', 'source_hashes')
    bind_review('operator_bottleneck_review.json', 'source_sha256')
    bind_review('cuda_ptx_cache_independent_review.json', 'source_and_evidence_sha256')
    if final:
        bind_review('report_final_round.json', 'evidence_sha256')
    p, table, sub = doc.p, doc.table, doc.sub

    p('计算研究报告 / 2026 年 11 月 10 日' + (' / 未完成草稿' if args.draft else ''), 'small')
    p('从随机特征到 GPU 算子<br/>理论审计与同精度比较', 'title')
    p('本轮继续无反射对称约束的 Re=100 有限圆环绕流研究，保留 Galerkin 径向离散、'
      '完整角向模态、交替尾迹以及速度和涡量同时达标的要求。重点是确认性能来自哪里，'
      '并检验更便宜的确定性基是否同样有效。')
    if final:
        for text in final['executive_paragraphs']:
            p(text)
    else:
        p('最终公平批次及编译产物复用分支仍在进行；此预览不代表最终速度排名。')
    table(['本轮已经确认的发现', '证据与边界'], [
        ['减少求解入口导入', '26条T72轨迹逐位等价；已通过隔离的高档50条批次中，新进程启动明显改善，求解本体没有相应提升。'],
        ['更强的Legendre对照', f'{len(leg["rows"])}条T72 GPU轨迹全部过门槛；新增与RFM同实际秩及更低秩的配置。旧大基数对照不能支持随机特征归因。'],
        ['全域非线性融合', 'T9数值通过；推进时间减少9%-17%，首次完整求解却增加23%-69%。编译与加载是直接瓶颈。'],
        ['理论上的负结果', '共同相位基改善局部条件性但准备更慢；冻结涡量BDF2的大步档精度失败且FFT工作量更高；隔步非线性外推失稳。'],
    ], [145, 351])
    p('研究定位：这里的RFM是固定随机试验空间与Galerkin时间推进的组合，未进行PINN残差训练。'
      '本轮没有新增先进PINN的同门槛比较。赢过所测FD4实现，也不等于胜过所有传统CFD；'
      'Legendre-Galerkin本身就是传统数值方法。', 'small')

    doc.new('01  算例、门槛与适用范围')
    p('二维不可压Navier-Stokes：∂ₜu+(u·∇)u=−∇p+νΔu，∇·u=0。'
      '圆柱半径a=0.5，外半径R=6，直径D=1，ν=0.01，Re=100，终止时间T=72。'
      '采用s=ln(r/a)坐标和流函数；实均值切向动量单独推进。外边界使用规定流函数、零涡量与均值Robin条件。')
    p('来流U(t)=1−exp[−(t/0.8)³]；4&lt;t&lt;12时叠加幅值0.12的sin⁴横向触发，之后取消。'
      '保留正负角向模态和全部实虚分量，没有施加反射对称、相位平移或尾迹对齐。'
      '三种算法均以同一边界和初始条件推进。FD4的“四阶”指径向空间离散；时间推进仍为二阶SBDF2，首步Euler。')
    table(['验收项', '预先固定的规则'], [
        ['物理误差', '速度、涡量及r≤2.5近场的相对L2误差，均≤0.5%；用两份细参考，逐时刻检查，不只看终点。'],
        ['独立重构网格', 'Gauss123、mid129、layer144三套径向节点；覆盖19个绝对时刻0.5至72。分段实现另查两次桥接邻近时刻。'],
        ['实现误差', 'GPU对同离散CPU的场和完整探针误差≤1e−5；保留577个等间隔探针时刻，未作相位修正。'],
        ['切换保护', '切换版对本空间恒步控制，再加恒步对参考的误差三角上界；不得借用更高秩空间代替自身控制。'],
        ['尾迹', '通过条件包含持续交替探针符号；幅值与频率作为诊断报告，不另设未登记的误差阈值。'],
    ], [100, 396])
    p('参考链覆盖径向384→448、角向K128→144以及时间率384→512。晚期分档差的组合值约0.0290%，'
      '低于0.5%门槛的四分之一0.125%；它不是连续PDE误差的严格上界。该参考链尚不能支撑0.1%门槛的四分之一预算。')
    p('有限外边界影响很大：R=8与R=6在共同r≤4区域、未作相位对齐时的最大涡量差约50.39%，脱落频率亦改变。'
      '因此本报告只针对明确的有限圆环边值问题，不宣称已达到无限外域标准圆柱绕流精度。'
      '既有Re=1000方腔采用光滑顶盖，也不能替代标准常速顶盖基准。', 'small')

    doc.new('02  交替尾迹与未对齐的比较')
    figure('finite_annulus_vorticity.png')
    p('图1：同一RFM41合格轨迹的t=24、48、72涡量。颜色范围固定，展示近尾迹；完整计算域的外半径为6。'
      '图像仅展示解的结构，精度由独立网格上的数值门槛决定。', 'cap')
    figure('unshifted_wake_probes.png', 470)
    p('图2：x=2.5D、y=0探针，四条曲线保留原绝对时间。触发结束后仍持续交替脱落。'
      'RFM41、同实际秩Legendre36与FD176均接近FD448参考。图中RFM/FD取已验收GPU轨迹，Legendre取对应CPU资格轨迹；'
      '它们不是一次隔离计时批次的截图，也不用于读图估计误差。', 'cap')

    doc.new('03  对照必须按实际空间和精度选择')
    p('输入基函数个数不等于最终保留秩。边界处理、列缩放和SVD截断后，旧高档RFM41实际秩为40，'
      '旧Legendre39为43；旧中档RFM37为39，旧Legendre37为41。直接比较原标签会把维度差误当成基函数优势。')
    table(['配置', '非零模秩', '分段最坏速度', '分段最坏涡量', '三角涡量界'], [
        ['中RFM37/K72/q66', '39', '0.0447%', '0.4317%', '0.4786%'],
        ['中Leg35/K72/q66', '39', '0.0410%', '0.4566%', '0.4977%'],
        ['高RFM41/K80/q76', '40', '0.0259%', '0.3403%', '0.3687%'],
        ['高Leg36/K80/q76', '40', '0.0197%', '0.3663%', '0.3940%'],
        ['高Leg35/K80/q76', '39', '0.0223%', '0.4191%', '0.4472%'],
        ['高Leg37/K80/q76', '41', '0.0189%', '0.3239%', '0.3506%'],
    ], [143, 65, 96, 96, 96])
    p('各行均值块秩均为30；表中秩指每个非零角模的径向秩。RFM行保留原五种子资格中的最坏值；'
      '新增Legendre为固定确定性基的全时段控制，高档Leg37仅作独立CPU资格记录，不属于下述12条GPU轨迹。'
      '同秩对照用于分离空间选择，低秩Legendre用于检验“同精度更便宜”的替代方案。中档Leg35的三角保护最接近阈值，余量仅约0.00227个百分点。', 'small')
    p('Leg33/K72/q66的恒步涡量误差约0.574%，不进入速度排名。较早低档RFM35的误差约0.642%，亦保留为失败。'
      'FD152未过门槛；FD176稠密版和FD184带状版通过。FD184稠密版曾有实现误差超限，不能因物理误差更小而混入排名。')
    p('三个新的Legendre配置分别运行恒步/分段及串行/warp观测，12条T72全部通过。'
      'warp与串行的保存状态、保存时刻逐位相同；完整探针最大相对差约2.48e−15。'
      '这证明观测优化可用在更强对照上，不会被错归因于随机基。')
    p('RFM种子为7331、7687、8011、8353、9041，确定性对照使用固定配置并重复计时。'
      '五个配对区块有助于判断本机稳健性，但不能证明随机特征在所有问题或所有基实现上具有普适优势。')

    doc.new('04  共同相位基：公式成立，成本未获益')
    p('固定角向模态后，线性系统为A(a,h)=aM+hK，M正定、K半正定。选择H=a*M+h*K，'
      '解广义特征问题，取P使PᵀHP=I、PᵀMP=Λ，则PᵀKP=(I−a*Λ)/h*。由此得到：')
    p('A(a,h)⁻¹ = P diag[dᵢ(a,h)]⁻¹Pᵀ；<br/>'
      'dᵢ = h/h* + (a−a*h/h*)λᵢ<br/>'
      '= (a/a*)(a*λᵢ)+(h/h*)(1−a*λᵢ)。')
    p('0≤a*λᵢ≤1，因此分母是a/a*与h/h*的凸组合。只有相位比值接近时，相对谱才接近1；'
      '仅取中间步长不构成一般保证。进一步对rⱼ=hⱼ/aⱼ，可将锚点比取为η=h*/a*=√(r最小·r最大)，'
      '最小化这组相位的最坏条件数上界，得到√(r最大/r最小)。这是上界的极小极大选择，'
      '不保证实际条件数最优，也不改变本轮已冻结实验。坐标变换P仍包含原系统的条件性。')
    p('跨模态共有一个基还需要更强条件。写M(x)=M₀+xM₁、K(x)=K₀+xK₁+x²K₂，x=k²。'
      '各系数在同一正定度量下白化后必须两两对易，才能同时正交对角化。实测归一交换子约0.051-0.076，'
      '不能只根据多项式形式合并所有角向模态。')
    table(['三个相位的准备 / 毫秒', '原方法', '直接三角解', 'H共同基'], [
        ['中档RFM', '34.27', '47.41', '51.96'],
        ['中档Legendre', '34.81', '44.72', '47.73'],
        ['高档RFM', '38.75', '56.39', '61.39'],
        ['高档Legendre', '45.71', '62.00', '66.22'],
    ], [196, 100, 100, 100])
    p('这些是局部准备诊断，不是隔离整轨迹计时。保持原GPU稠密包布局时，H基仍需为每相位生成宽右端投影，'
      '没有减少11.69-16.30MB的设备包；额外谱分解和矩阵乘法使准备增加37%-58%。直接三角解也没有更快。')
    p('65位定点参考下，前向误差最大值由原方法4.67e−9降到H基9.87e−11，'
      'H度量相对条件数约1.105，而原矩阵可达3.36e7。精度改善与速度改善是不同结论。'
      '若全程改用P坐标，需重写重构、提升、观测和存储；本轮未把局部优点外推成完整算法提速。', 'small')

    doc.new('05  冻结涡量BDF2：稳定性不是精度保证')
    p('令ω*=curl(2uₙ−uₙ₋₁)，以新速度构造Lamb项。固定ω*时，'
      'Fω的弱配对为积分ω*(uθvr−urvθ)，交换u与v即变号。实Fourier配对的均值权为1，'
      '非零模实部、虚部权各为2；固定求积及足够的角向去混叠保留这一反自伴结构。')
    p('齐次块为A−hF，A=(3/2)M+hK。固定步长、同一离散求积及精确线性求解下，齐次系数满足：<br/>'
      'G(aₙ₊₁,aₙ)−G(aₙ,aₙ₋₁)+‖aₙ₊₁−2aₙ+aₙ₋₁‖²_M<br/>'
      '+4h‖aₙ₊₁‖²_K=0，G(a,b)=‖a‖²_M+‖2a−b‖²_M。')
    p('真实驱动问题必须在右端保留边界提升做功和线性残差做功。该恒等式没有给出当前非齐次问题的无条件长期精度。'
      '形式二阶也不能替代本算例的步长收敛实验。文献中的速度-涡量方法还解独立涡量方程，不能直接搬用其稳定定理。')
    p('左预条件I−hA⁻¹F在普通系数范数下可非正规。令A=LLᵀ，对称变换得I−hL⁻¹FL⁻ᵀ；'
      '在正确实加权坐标中后项反对称，最小奇异值至少为1。固定态测试减少了目标步长的乘积次数，'
      '但新增块变换没有稳定降低单步时间。比较残差时仍检查原方程。')
    table(['T16试验', 'RFM37最坏涡量', 'Leg37最坏涡量', '结论'], [
        ['时间率128', '0.364950%', '0.364784%', '速度与涡量均过0.5%'],
        ['时间率64', '161.024%', '161.664%', '大步档失败'],
    ], [110, 118, 118, 150])
    p('大步档首次保存失败发生在t=4，早于横向触发。未观测到逐步GMRES门槛失败；'
      '12个保存检查点原方程残差≤7.59e−10，未发现线性求解未收敛的证据。未校正能量缺陷≤5.71e−10，'
      '加入实际残差做功后的代数闭合缺陷≤2.25e−15；这不排除早期扰动随后放大。事后观察到高角频动能显著增加，'
      '这定位了偏差，尚未单独证明其产生机制。')
    p('合格时间率128每条需要26765个FFT批、3397878条径向线变换，分别是原显式SBDF2的6.538与5.030倍。'
      '减半时间率后仍耗4.43倍以上线变换且失去精度。原方程审计的额外48批另列，不混入求解工作量。因此停止GPU晋升。', 'small')

    doc.new('06  减少非线性次数的失败与有效边界')
    p('隔步计算真实非线性项，并用相距2h的历史项外推，是直接减少FFT次数的候选。'
      '本轮在两步周期的标量分裂方程上建立四状态转移矩阵：采样步用(1.5,−0.5)，跳过步用(2,−1)权重。'
      '观察到的收敛阶仍约1.998-2.016，但三个标量试验在最细时间率512处的误差为原法3.29-3.62倍。'
      '多个α截面在β采样网格上更早失稳，例如α=0.1的首次超限β从0.558降至0.3355；这不是完整稳定域证明。')
    p('固定NS控制在t=8后启用该策略；RFM37/时间率128与RFM41/时间率152分别在t≈9.508和9.559溢出。'
      '原控制继续完成。标量二阶性不能保障非线性、刚性、边界驱动的长轨迹稳定；两条失败原样保留，未移植GPU。')
    table(['其它候选', '核验结果', '决定'], [
        ['通量历史双缓冲', '8对T9状态逐位相同；推进变化仅约−0.030/+0.018毫秒，不稳定。', '不晋升；名义字节减少不等于DRAM节省。'],
        ['CUDA图24→48/64/128步', '36条T72均逐位相同；没有方法同时满足预设推进、求解和新进程收益。', '保留24步。减少launch数不是充分指标。'],
        ['高版本CUDA工具链', 'T9兼容性通过；T72预登记36条中只16条完成，1条超时，1条无完整终态，18条未尝试。', '批次不完整，不作稳定速度结论。'],
    ], [122, 252, 122])
    p('现代工具链完整的16条另行数值复核均通过；改变cuFFT的轨迹并非全部逐位相同，仍按原物理与实现门槛评分。'
      '超时进程按准确身份结束后，旧CUDA11.8健康检查通过。现有证据无法确认是驱动挂起还是清理路径阻塞；没有重置驱动。')
    p('因此，本轮未发现可以进一步降低真实非线性求值次数且保持门槛的方案。'
      '既有分段步长减少的工作量来自调度，不来自随机特征。下文融合算子优化执行方式，仍保留每步原来的逻辑变换。')

    doc.new('07  融合非线性FFT的等价条件')
    p('Galerkin每次非线性计算含3个实逆变换，得到ur、uθ、ω，再形成urω、uθω并做2个正变换。'
      'FD含4个逆变换和1个乘积正变换；均值强迫保留原按角模升序的求和。两者每径向行仍是5个逻辑场变换。')
    p('将两组Hermitian系数A、B打包成A+iB，可用一次复逆FFT同时得到两个实场；'
      '两个实乘积的复正FFT结果Z，再用Zₖ与共轭Z₋ₖ分离。负模须先分别共轭A与B，'
      '不能错误地把整个打包谱当作Hermitian谱。原投影含1/N，融合核不得再归一化。')
    p('乘积最高频率为2K。对任意非零整数l，|k+lN|≥N−K；若N&gt;3K，'
      '则N−K&gt;2K，乘积别名不可能落入保留的|k|≤K模态。该条件只保护保留模态，不表示高频不存在。')
    table(['角向长度', '混合基分解', '共享内存/块', '屏障数'], [
        ['200', '2×2×2×5×5', '6400字节', '13'],
        ['225', '3×3×5×5', '7200字节', '11'],
        ['243', '3×3×3×3×3', '7776字节', '13'],
    ], [90, 160, 140, 106])
    p('一个线程块处理一条径向行，使用两块复共享数组完成逆变换、乘积和正变换。'
      '每次非线性原来的3个高层执行操作合为1个，去除全局物理暂存；原始通量历史与FD均值路径保持。'
      '这里的3→1不代表性能分析器测得的CUDA Graph内部实际内核节点数。'
      '名义全局数组流量最多减少80N字节/行，不把缓存命中后的实际设备流量等同于这个上限。')
    p('固定设备态对cuFFT的整体相对差≤6.58e−16。但打包会把大通道的舍入污染带入小通道：'
      '一条Galerkin通道缩小到1e−10时，其单通道相对差约2.71e−6。极端抵消还会放大输出相对误差。'
      '因此同时核验每通道相对误差、未抵消尺度绝对误差及完整物理轨迹，不能只报告合并向量的机器精度。', 'small')

    doc.new('08  小核收益与完整成本必须分开')
    table(['固定态图事件 / 微秒', '原cuFFT', '融合核'], [
        ['Galerkin N225/q66', '42.12', '22.64'],
        ['Galerkin N243/q76', '40.32', '25.48'],
        ['FD176 N243', '85.13', '61.00'],
        ['FD184 N243', '90.35', '69.38'],
    ], [236, 130, 130])
    p('上表为64次非线性图重放、9组事件计时的局部测量。它不包括模型构造、编译、计划与解的推进，'
      '不能作为整体求解提速。融合完整入口与原入口均使用相同warp观测、相同时间步、相同精度及相同真实非线性次数。', 'small')
    figure('t9_fusion_cost_breakdown.png')
    p('图3：四组T9顺序诊断的完整求解阶段。编译/加载从运行时初始化中单独拆出，未重复相加；'
      '其它准备与清理按完整时间扣除已分列的互斥阶段得到。此图不是隔离速度排名。', 'cap')
    p('8条T9全部通过原双参考、同离散CPU与两引擎配对门槛。融合对同warp观测的cuFFT最大场差约1.78e−9，'
      '对独立CPU的最大场差约2.26e−7，均须按各自规定评分。'
      '推进时间减少9%-17%，但编译与加载增加约0.25-0.62秒，超过该短程省下的0.017-0.030秒。'
      '单活跃融合入口已经编入同一模块，仍未消除启动瓶颈。按预设规则，原无显式缓存分支不晋升T72。')
    p('另一个独立优化是探针归约：将6条线程的串行点积改为6个warp共同完成。'
      '定点探针节点在Galerkin上约快19-22倍、FD上约快27倍；状态轨迹不变，探针舍入差需独立验收。'
      'T72有576次真实探针归约，按定点结果估计FD可节省约0.40-0.42秒，比Galerkin约0.064-0.081秒更多。'
      '这是预测，不是最终整程实测；必须同时优化FD，否则会夸大RFM优势。')

    doc.new('09  减少导入开销：已通过的历史批次')
    p('旧求解入口连带导入资格审计、信号与统计模块。最小入口保持23个求解定义的AST等价，'
      '24配置的2016个FP64数组逐位一致；新增26条T72轨迹的130个保存数组也全部逐位一致。'
      'FD保留实际需要的稀疏、分解和插值依赖，不人为增加对照负担。')
    groups = {tuple(x['group']): x for x in thin['groups']}
    def time_value(name, arm, path, clock):
        return f"{groups[(name,arm,path)]['clocks'][clock]['median']:.4f}"
    table(['高档同批次 / 秒', '完整求解', '新进程至结果'], [
        [label, time_value(name, arm, path, 'solver_seconds'),
         time_value(name, arm, path, 'fresh_process_return_seconds')]
        for label, name, arm, path in [
            ('RFM41恒步/旧入口', 'rfm41_k80', 'uniform', 'selected'),
            ('RFM41恒步/最小入口', 'rfm41_k80', 'uniform', 'minimal'),
            ('RFM41分段/最小入口', 'rfm41_k80', 'combined', 'minimal'),
            ('Leg39分段/最小入口', 'leg39_k80', 'combined', 'minimal'),
            ('FD176/最小入口', 'fd176_k80', 'fd', 'minimal'),
            ('FD184带状/最小入口', 'fd184_k80', 'fd', 'minimal'),
        ]
    ], [236, 130, 130])
    p('该高档50条批次通过数值与采样隔离验收。配对新进程时间，RFM恒步/分段入口分别改善1.293/1.301倍，'
      '五区块全部胜出；求解时钟约1倍。这是启动优化，不是更快的NS推进。')
    p('该批次FD176相对RFM分段的配对比为1.809倍完整求解、1.599倍新进程；'
      '但当时Legendre39秩更大，探针亦未统一改用warp。因此只能保留为当时配置的历史结果，'
      '不能用于最终随机特征归因，也不能与后来不同批次分母混合生成新比值。')
    p(f'同期中档登记{mid["registered_slots"]}条，外部进程在第一段预检活跃，计时槽实际尝试{mid["attempted_slots"]}条。'
      '该批次没有中档速度结论。失败窗口保留，没有删去失败槽后补跑成一个“干净批次”。')

    doc.new('10  隔离计时、故障与可复核性')
    p('比较采用独立新进程，完整构造求解器和CUDA上下文。求解时钟包括主机构造、编译/加载、'
      '传输、FFT准备、图捕获、推进、探针、同步、下载及资源释放；新进程时钟另含解释器导入、'
      '握手、身份与空间签名检查以及接收返回记录。历史thin采用刷新后的通信标记，最终rank-fair入口采用原子文件记录和轮询。'
      '结果序列化与独立评分位于返回记录之后，避免把审计成本算成某个方法的求解成本。')
    p('新进程不等于冷机器。资格测试后的系统与驱动缓存保持自然状态，不清全局缓存；'
      '显式编译产物缓存若启用，必须所有方法共同使用，首次构建和后续命中单独列出。')
    p('计时前有两段10秒CPU/GPU安静检查；正式区间使用每秒原生进程快照和PID+创建时间身份，'
      '保留已退出自有进程的终态CPU计数。按原10秒CPU预算与覆盖规则验收，未达到规则即降为诊断。'
      '已通过高档50条批次共有195个活动区间采样，最大间隔约1.014秒。')
    p('这些证据支持“通过采样隔离”，不是操作系统级完全独占证明。'
      '独立测试证实，生命周期完全处于采样间隙的短进程可能被漏掉；GPU利用率只作预检采样，'
      '其它非Python应用负载覆盖也有限。ETW能力探测被系统拒绝，未提升权限或改动系统安全设置。')
    p('已记录两类计时中断：旧工具链诊断的清理阻塞，以及后来正式批次第三个子进程持续运行超过两小时、'
      '原超时未形成可确认终态。后者的具体卡点未查明，不能归因于某个驱动函数或CPU调度。'
      '恢复版本把控制、求解、监控分核，并增加独立Job监控；测试覆盖失联与终止辅助进程阻塞。'
      '它只在确认精确受管进程树退出时记录清理成功；未确认则记pending，禁止续窗。'
      '这些测试不保证任意操作系统故障下都能及时清理，正式结果仍须逐项通过验收。')
    p('公平规则固定为五个配对区块、同一块使用同一RFM种子；报告基准/候选的配对比中位数、范围与胜出次数。'
      '“稳定实用收益”要求两时钟均五次胜出、配对中位比≥1.05。若只改善新进程入口，则明确标为启动收益。')

    if final:
        for page in final['pages']:
            doc.new(page['title'])
            for item in page['items']:
                if item['type'] == 'paragraph':
                    p(item['text'], item.get('style', 'body'))
                elif item['type'] == 'table':
                    table(item['headers'], item['rows'], item['widths'])
                elif item['type'] == 'figure':
                    figure(item['name'], item.get('width', 496))
                else:
                    raise ValueError(item['type'])
    else:
        doc.new('11  最后批次尚未写入')
        p('此页为本地审阅占位。最终报告要求完整记录强Legendre公平批次、编译产物复用分支、'
          '逐项贡献与未通过条件；没有最终审阅标记，生成器拒绝输出可交付文件。')

    doc.new('复现索引与文献依据')
    p('公开快照追加到既有私人GitHub仓库，保留旧提交及原实验。work/rfm_v8下的协议、'
      '逐槽原始计时、轨迹、资格、独立复核、失败记录与源码共同构成证据。不能只凭汇总PDF重做审计。')
    table(['证据入口', '作用'], [
        ['thin_formal_*_verification.json', '独立复核最小入口正式批次，保留中档预检拒绝与高档通过。'],
        ['legendre_rank_*_output', '实际秩、CPU与12条GPU资格，含较低秩失败及最紧三角保护。'],
        ['galerkin_phase_pencil_* / lamb_*', '共同相位基和Lamb路线的理论条件、局部审计、轨迹失败及工作量。'],
        ['fused_march_t9_output', '无显式缓存融合路线的全部数值与完整成本负结果。'],
        ['report_final_round.json', '最终批次、工程贡献及成本口径；与独立原始结果关联。'],
        ['publication_evidence_README.md', '公开副本、历史源码胶囊、供应链复建及隐私替换映射。'],
    ], [230, 266])
    p('两份合成故障日志、两份真实私有上下文故障日志，以及监控请求中的本地路径，'
      '只发布单独标记的脱敏副本及原/公开哈希映射。'
      '原冻结摘要中的旧哈希不能认证脱敏文件；原件保留在本地。第三方CUDA二进制不随仓库上传，'
      '只提供官方来源、版本、许可证与校验信息。', 'small')
    refs = [
        ('Chen、Chi、E、Yang：Random Feature Method (2022)', 'https://arxiv.org/abs/2207.13380',
         '原论文使用随机特征、配点与边界罚项；本工作的Galerkin变体和算例性能需要独立证明。'),
        ('Heister、Olshanskii、Rebholz：velocity-vorticity长时稳定性 (2015)', 'https://arxiv.org/abs/1511.08072',
         '包含独立涡量动力学方程，不能直接当成本报告冻结涡量算法的定理。'),
        ('LAPACK：Generalized Symmetric Definite Eigenproblems', 'https://netlib.org/lapack/lug/node54.html',
         '广义对称正定特征问题与Cholesky同余变换背景。'),
        ('NVIDIA：cuFFT transform conventions', 'https://docs.nvidia.com/cuda/archive/11.8.0/cufft/index.html',
         '实/复变换、输出布局及非归一化约定；实际实验锁定CUDA11.8组件。'),
        ('NVIDIA：NVRTC CUDA11.8 documentation', 'https://docs.nvidia.com/cuda/archive/11.8.0/nvrtc/index.html',
         '运行时源码编译与PTX生成/加载的边界。'),
    ]
    for title, url, note in refs:
        p(f'<b>{title}</b><br/><link href="{url}" color="#285F8A">{url}</link><br/>{note}', 'small')

    output = ROOT / 'outputs' / ('RFM_NS_v8_研究报告_草稿.pdf' if args.draft else 'RFM_NS_v8_算子改良与同精度公平比较_研究报告.pdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=49, leftMargin=49,
                            topMargin=45, bottomMargin=57,
                            title='不可压NS：算子改良与同精度公平比较', author='Computational research study')
    pdf.build(doc.story, onFirstPage=footer, onLaterPages=footer)
    stem = 'report_draft' if args.draft else 'report'
    (HERE / f'{stem}_text.txt').write_text('\n\n'.join(doc.transcript), encoding='utf-8')
    (HERE / f'{stem}_provenance.json').write_text(json.dumps({
        'status': 'draft' if args.draft else 'reviewed', 'inputs': INPUTS,
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pdf': output.name, 'pdf_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'public_package_manifest_verified': PUBLIC_PACKAGE_VERIFIED,
        'public_evidence_resolutions': PUBLIC_RESOLUTIONS,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(output)

if __name__ == '__main__':
    main()
