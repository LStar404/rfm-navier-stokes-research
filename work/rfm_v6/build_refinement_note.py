"""Build the v6 parallel-refinement addendum without changing frozen v6."""
import json
from pathlib import Path
import sys

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'report_v2'))
import layout as doc
from build_v6_report import footer, pct


def load(name):
    return json.loads((ROOT/'parallel_refinement'/f'{name}.json').read_text(encoding='utf8'))


def maximum(rows,key):
    return max(row['worst'][k] for row in rows for k in (key,f'near_{key}'))


def main():
    data=load('parallel_refinement')
    old=load('transfer_refinement')
    rows={r['name']:r for r in data['rows']}
    p,sub,table=doc.p,doc.sub,doc.table
    p('计算研究补充　/　第 6 轮　/　2026 年 11 月 10 日','small')
    p('并行加密补测<br/>全域圆柱的 0.5% 精度档','title')
    p(f'启动 {data["workers"]} 个单线程 CPU 进程，完成 {len(rows)} 个独立加密与候选算例，'
      f'批次墙钟耗时 {data["batch_elapsed"]:.0f} 秒。'
      '物理算例、横向脉冲、时间检查点和三套评分网格沿用 v6。并行耗时用于记录批次成本。')
    sub('分开检查径向与时间、角向误差')
    table(['比较','最大场差'],
          [['固定角向与步长，FD280 → FD320',pct(old['radial_change'])],
           ['固定 FD320 径向，提高角向截断和步数',pct(old['joint_time_angular_change'])],
           ['上面两段差的保守合并值',pct(old['bound_radial_and_joint'])],
           ['原报告 FD240 参考 → 新 FD320 参考',pct(old['old_to_new'])]],[324,170])
    p('最细参考为径向 320、角向 112、每单位时间 320 步。新细化路径的最大变化低于 0.125% 筛查值。'
      '原参考与新参考的差值超过该筛查值。','small')
    sub('对新参考解的同步误差')
    groups=[('RFM37，三种子',[r for n,r in rows.items() if n.startswith('rfm37_')]),
            ('RFM41，三种子',[r for n,r in rows.items() if n.startswith('rfm41_')]),
            ('Legendre37',[rows['leg37_q66_k72_17']]),('FD176',[rows['fd176_k80']]),
            ('FD168，原隔离计时配置',[r for r in old['rows'] if r['name']=='fd168_k88'])]
    table(['配置','最坏速度','最坏涡量'],
          [[n,pct(maximum(r,'velocity')),pct(maximum(r,'vorticity'))] for n,r in groups],[246,124,124])
    p('FD176 的本轮数据来自并行精度批次。原隔离计时配置 FD168 用新参考复评时超出 0.5%，'
      '因此隔离速度排名仍采用 v6 的 1% 和 2% 档。','small')
    target=ROOT.parents[1]/'outputs'/'RFM_NS_v6_并行加密补充.pdf'
    pdf=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=44,bottomMargin=53,
                           title='全域圆柱绕流 · 0.5% 并行加密补测')
    pdf.build(doc.story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    main()
