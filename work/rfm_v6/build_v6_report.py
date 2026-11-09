"""Build the full-annulus shedding report from frozen formal and diagnostic records."""
import json
from pathlib import Path
import sys

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image, SimpleDocTemplate, TableStyle

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'report_v2'))
import layout as doc


def read(name):
    return json.loads((ROOT/f'{name}.json').read_text(encoding='utf8'))


def pct(value):
    return f'{100*value:.4f}%'


def link(url,label):
    return f'<link href="{url}" color="#195783">{label}</link>'


def worst(row,key):
    return max(row['worst'][key],row['worst'][f'near_{key}'])


def best(rows,kind,gate):
    eligible=[r for r in rows if r['kind']==kind and r['passes'][str(gate)]]
    return min(eligible,key=lambda r:r['seconds'],default=None)


def compact(row):
    return '—' if row is None else f'{row["seconds"]:.2f} s'


def dense(headers,rows,widths):
    doc.table(headers,rows,widths)
    doc.story[-2].setStyle(TableStyle([
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))


def footer(canvas,info):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CFD8DF'))
    canvas.line(49,43,546,43)
    canvas.setFont('CN',8)
    canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50,29,'不可压 Navier–Stokes  |  全域圆柱涡脱落复测  |  2026-11-10')
    canvas.drawRightString(545,29,str(info.page))
    canvas.restoreState()


def build():
    p,sub,table=doc.p,doc.sub,doc.table
    data=read('full_formal')
    if data['status']!='complete':
        raise RuntimeError('The frozen formal matrix is incomplete')
    rows=data['summary']
    ref=data['reference_metadata']
    prev=data['predecessor_metadata']
    fig=ROOT/'figures'/'diagnostic_wake.png'
    st=data['reference_wake']['strouhal']
    p('计算研究报告　/　第 6 轮　/　2026 年 11 月 10 日','small')
    p('全域圆柱绕流的交替涡脱落<br/>Galerkin 随机特征法同精度复测','title')
    p('Re=100，圆柱直径 D=1，外半径 R=6。全圆周保留正弦、余弦与角向平均流。'
      '从静止开始，在 4&lt;t&lt;12 用峰值 0.12U 的平滑横向来流脉冲触发尾流失稳，之后横向外流速度为零。'
      f'正式四阶 FD 参考在 t≥36 的探针中位周期为 {data["reference_wake"]["median_period"]:.4f} D/U，'
      f'Strouhal 数为 {st:.5f}。')
    with PILImage.open(fig) as im:
        w,h=im.size
    doc.story.append(Image(str(fig),width=484,height=484*h/w))
    p('图使用另一份细网格 FD200 的冷启动探针和短时诊断重启密集帧。上图为三个尾流探针的横向速度。'
      '下图同色标涡量相隔约半周期。13 组半周期场在上下反射并翻转涡量符号后，中位相对差为 2.49%，'
      '直接比较为 142.1%。诊断重启只用于密集图帧。','cap')
    p('交替脱落和频率属于当前有限环域与外边界条件。圆柱文献中的近无界频率供量级参照。 '
      +link('https://doi.org/10.1017/jfm.2011.134','Chan 等，圆柱绕流对照')
      +'　'+link('https://doi.org/10.1016/0045-7825(94)00736-7','外计算域影响'),'small')

    doc.new('01　同精度门槛与完整求解计时')
    p(f'参考解为外壁加密四阶 FD，n={ref["n"]}、K={ref["k"]}、每单位时间 {round(1/ref["dt"])} 步。'
      f'前一级为 n={prev["n"]}、K={prev["k"]}、每单位时间 {round(1/prev["dt"])} 步。'
      '两者在独立评分网格的最大速度、涡量变化为 '
      f'{pct(max(data["reference_change"]["velocity"],data["reference_change"]["near_velocity"]))} 和 '
      f'{pct(max(data["reference_change"]["vorticity"],data["reference_change"]["near_vorticity"]))}。'
      '预设参考差不高于验收门槛的四分之一。该筛查是经验加密比较。')
    names={'rfm':'RFM','legendre':'Legendre','fd':'四阶 FD'}
    winners={}
    speed=[]
    for g in data['gates']:
        a,b,c=[best(rows,k,g) for k in names]
        winners[g]=(a,b,c)
        speed.append([f'{100*g:g}%',*map(compact,(a,b,c)),
                      f'{c["seconds"]/a["seconds"]:.2f}×' if a and c else '—',
                      '通过' if data['gate_screens'][str(g)] else '不足'])
    table(['双门槛','RFM','Legendre','四阶 FD','FD / RFM','参考加密'],speed,[74,88,89,88,79,76])
    for g,(a,b,c) in winners.items():
        if a and b and c:
            cf=sum(x<y for x,y in zip(a['times'],c['times']))
            pl=sum(x<y for x,y in zip(a['times'],b['times']))
            p(f'{100*g:g}% 门槛入选 {a["name"]}、{b["name"]}、{c["name"]}。RFM '
              f'在 {cf}/{len(data["seeds"])} 个顺序块快于四阶 FD，在 {pl}/{len(data["seeds"])} 个块快于 Legendre。','small')
    chosen=winners[.01][0]
    if chosen:
        p(f'1% 档入选 RFM 的最坏速度、涡量误差分别为 {pct(worst(chosen,"velocity"))} 和 '
          f'{pct(worst(chosen,"vorticity"))}。','small')
    raw_fine=[r['name'] for r in rows if r['numerical_pass']['0.005']]
    if raw_fine and not data['gate_screens']['0.005']:
        p('直接对 FD240 评分，达到 0.5% 的配置包括 '
          +'、'.join(raw_fine)+'。参考加密未通过这档筛查，因此表中留空。','small')
    p('误差按绝对物理时刻比较，三套独立面积权重网格包含一套外层加密网格。'
      't=24、36、48、54、60、66、72 的速度与实际旋度，在全域和 r≤2.5 近区均需达标。'
      '每档还要求所有运行出现持续交替脱落。')
    p(f'在同一 Ryzen 9 9955HX 的逻辑核心 {data["cpu"]} 上以单线程随机交错运行，共 {len(data["rows"])} 次。'
      f'新种子为 {", ".join(str(s) for s in data["seeds"])}。确定性方法等次数重复。'
      '时间包含构造、分解、从静止推进到 t=72 及探针和检查点。参考解与独立评分在计时之外。','small')

    doc.new('02　径向层分辨率与全域非线性算子')
    p('外边界固定 ω(R)=0，但脱落涡被输运至外圈。令 d=L−s，s=log(r/a)，在出流扇区将法向速度冻结为 u<sub>n</sub>&gt;0。'
      '对流扩散平衡给出向内衰减的修正 exp[−(R u<sub>n</sub>/ν)d]。其对数坐标厚度可到约 0.0017，'
      '只加密内壁会漏掉这层，继而造成长时相位漂移。')
    p('RFM 和 Legendre 都采用六个共享的解析外层函数，指数率 16、36、80、165、340、660。'
      '流函数增广项 z²d[1+(β+2/L)d]exp(−βd) 在内壁值与导数为零，外壁值和二阶导数为零。'
      '均值速度也加 Robin 相容的局部函数。径向求积在外层分段，FD 独立采用外侧余弦聚点。')
    p('在 FD160 同一网格上从每单位 192 步加密至 256 步，最坏涡量改变约 0.053%。'
      '以相同步长换到 FD200 外侧余弦网格时，涡量改变约 2.79%。'
      '这组对照把主要误差指向径向空间离散。','small')
    p('非零角模态保留同基 Galerkin 的正定质量和扩散矩阵。角向平均切向速度 v<sub>0</sub> 另解动量方程 '
      '∂<sub>t</sub>v<sub>0</sub>=−平均[u<sub>r</sub>ω]+ν(∂<sub>ss</sub>v<sub>0</sub>−v<sub>0</sub>)/r²，'
      '内壁 v<sub>0</sub>=0，外壁 v<sub>0</sub>+∂<sub>s</sub>v<sub>0</sub>=0。'
      '这样角向平均环量由方程演化。')
    sub('全圆周的去混叠与复用')
    p('对正频率复系数和直流项，用 N&gt;3K 的全圆周点数精确去除保留模态的二次混叠。'
      '将 u<sub>r</sub>、u<sub>θ</sub>、ω 的实数逆变换和两组乘积的正变换批量执行，复用补零缓冲区。'
      '径向求值预先吸收 r 的尺度并复用实虚列乘法的输出。直接卷积测试含直流、混合奇偶和 K=1 至 128，最大相对差为 3.31e−14。'
      '复数打包与提前径向投影在试跑中收益不稳，继续使用实数 FFT。')
    lookup={r['name']:r for r in rows}
    a,b=lookup['rfm35_k64_basic'],lookup['rfm35_k64']
    audit=read('full_energy_audit')
    p(f'固定 RFM35、K=64、q=60 时，原补零与径向求值、复用后的中位耗时分别为 '
      f'{compact(a)}、{compact(b)}。首种子全时刻场差最大为 {max(audit["trajectory_difference"].values()):.2e}。'
      f'同基 Galerkin 的齐次自对流做功抵消，在保存状态上的归一化残差至多为 '
      f'{max(r["relative_homogeneous_work"] for r in audit["energy"]):.2e}。'
      '外边界提升仍可做功，时间推进保持 SBDF2。','small')
    p('保存时刻中，角向平均切向速度的面积 L² 范数最多占全速度范数的 '
      f'{pct(max(r["mean_velocity_fraction"] for r in audit["energy"]))}。','small')
    sub('正式配置与分辨率')
    config_rows=[]
    for r in rows:
        c=r['config']
        radial=str(c['n']) if r['kind']=='fd' else f'{c["m"]}+{len(c.get("layers",[]))} / {c["q"]}'
        err='—' if not r['worst'] else pct(worst(r,'vorticity'))
        config_rows.append([r['name'],radial,c['k'],r['rate'],compact(r),err])
    dense(['配置','径向 n 或 m+层/q','K','步/单位','中位时间','最坏涡量'],
          config_rows,[146,107,38,54,74,75])

    path=ROOT.parents[1]/'outputs'/'RFM_NS_v6_全域涡脱落复测.pdf'
    pdf=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=44,bottomMargin=53,
                           title='全域圆柱绕流的交替涡脱落 · RFM 同精度复测')
    pdf.build(doc.story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    build()
