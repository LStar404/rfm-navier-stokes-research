"""Prepare a reviewable report addendum; never approve its own publication."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def paragraph(text, style='body'):
    return dict(type='paragraph', text=text, style=style)


def table(headers, rows, widths):
    return dict(type='table', headers=headers, rows=rows, widths=widths)


def prepare():
    evidence = [
        'rank_fair_cache_qualification_output/qualification_summary.json',
        'rank_fair_cache_design_independent_review.json',
        'rank_parity_expansion_output/summary.json',
        'cuda_private_scope_t9_output/summary.json',
        'cuda_private_context_independent_audit.json',
        'cuda_private_scope_solver_closure_audit.json',
        'stock_ptx_install_output/summary.json',
        'formal_interruption_review_20261110/formal_interruption_review.json',
        'rank_fair_outer_watchdog_independent_review.json',
    ]
    # Resolve a historical filename without silently dropping its evidence.
    if not (HERE / evidence[5]).exists():
        matches = list(HERE.glob('*private*closure*audit*.json'))
        if len(matches) != 1:
            raise RuntimeError('Unique production closure audit required')
        evidence[5] = matches[0].name
    pages = [dict(title='11  强对照与完整数值资格', items=[
        paragraph('最终候选使用相同的轻量求解入口、warp探针、独立新建上下文和原cuFFT非线性算子。39个不同数值配置分别验证首次编译与严格缓存命中，共78条T72轨迹；全部通过原来的速度、涡量、近场、实现误差和尾迹要求。39对缓存策略的场、时间和577点探针逐位一致。资格时间不参与速度排名。'),
        table(['空间 / 实际非零模态秩', '保留配置与限制'], [
            ['中档RFM37 / 39', 'K72、q66，5个固定种子；恒步与分段均保留。'],
            ['中档Legendre35 / 39', '与RFM同秩；恒步与分段均通过，最紧三角保护约0.49773%。'],
            ['高档RFM41 / 40', 'K80、q76，保留原基数作为清晰的同秩对照。'],
            ['高档RFM35 / 38或39', '5个种子全部通过；4个秩38，种子8353秩39。不能把5个种子都称为秩38。'],
            ['高档Legendre36、35 / 40、39', '恒步与分段均保留。'],
            ['高档Legendre34 / 38', '恒步通过；分段的三角保护0.50960%超标，不能纳入分段速度排名。'],
            ['FD176稠密、FD184带状', '同一物理门槛、同一观测优化与缓存政策。'],
        ], [205, 291]),
        paragraph('78条中最大直接物理误差约0.47754%；最大CPU/GPU场差约3.57e−6、完整探针差约3.88e−6，均按各自验收口径判定。更小的高档Legendre33在T72失败；中档RFM35分段也因三角保护失败。T16筛选通过不能替代T72晋升。'),
        paragraph('高档RFM35只降低径向空间的实际秩和矩阵工作；q、角向长度、实际非线性调用次数没有因此减少。它是待比较的空间压缩收益，不能直接写成已经获得的FFT提速。', 'small'),
    ]), dict(title='12  缓存与上下文：计入完整代价', items=[
        paragraph('缓存只保存项目内核源码编译得到的PTX。键包含源码、编译选项、目标架构、入口、编译器及依赖哈希；严格命中缺失或损坏时直接失败。显式缓存不含矩阵、解、CUDA上下文与cuFFT计划。驱动缓存保持自然状态，模块加载和可能发生的驱动JIT仍计费。所有方法执行同样政策。'),
        table(['政策', '实际计费范围'], [
            ['首次编译 empty_first', '每槽新空目录；实际编译、原子写入、读回验真、模块加载均在求解时钟内。'],
            ['严格命中 strict_hit', '每槽0次NVRTC编译；读取、哈希、模块加载及驱动工作仍计费。'],
            ['离线预构建一次', '3个stock键共147782字节；3次新进程完整安装为2.7292、2.6392、2.6487秒，中位2.6487秒。'],
        ], [150, 346]),
        paragraph('预构建成本必须按使用方式摊销。若独立安装成本为C、命中单次成本为H，N次总成本为C+NH；与每次首次编译的N次成本比较，不能把约0.93秒NVRTC子阶段冒充2.6487秒完整安装成本。安装包覆盖三种键；只用其中一种的最小安装成本本轮未测。'),
        paragraph('运行中曾出现主上下文获取停滞。改为新建私有上下文后，首次清理仍失败；进一步探针显示cuFFT销毁返回后当前上下文发生变化。窄范围修复要求入口当前上下文为空，准确追踪本任务拥有的上下文，恢复后销毁并确认最终为空。4次健康检查、24条T9及78条T72通过；不宣称支持任意外部上下文栈或所有驱动故障。'),
        paragraph('缓存后的融合FFT诊断中，四组严格命中求解时钟均改善，但新进程时钟只有两组胜出，未满足预先登记的晋升规则。融合仍停留在短程诊断，未纳入最终T72排名。其数值资格通过，不应把工程晋升失败写成数值失败。'),
    ]), dict(title='13  中断记录与恢复边界', items=[
        paragraph('原160槽正式设计的第一窗只完成2槽。第3槽是中档Legendre35，已有READY和GO，没有RETURN、精确终态与关闭环境检查。监控保存8327个采样；该求解进程在GO后仍持续约8295秒。原30秒返回、45秒整槽保护没有形成有效终止证据，这批不能生成配对速度排名。'),
        paragraph('证据无法定位当时阻塞在具体调用中。控制进程和求解进程原先同核，可能存在调度干扰；同步终止调用也缺少独立保障。这些是需要消除的风险，不是已经查明的根因。恢复版将求解固定CPU4、监控CPU6、控制CPU8，外层观察与Job所有者另用CPU10、CPU12。'),
        paragraph('求解前先在标准库入口核验继承亲和性，再改为求解核，随后才导入科学计算与CUDA模块。每槽在启动前提交原子截止记录，外层独立读取且不允许延期。C#外层把控制进程及其子进程在恢复运行前加入Windows Job；观察者不持有Job句柄，也不直接调用可能阻塞的终止操作。'),
        paragraph('11项CPU场景覆盖正常返回、控制失败、后代残留、终止辅助进程延迟、所有者异常退出、槽截止违规与输入变更。测试中的延迟终止场景约2.08秒返回pending；该记录没有被后来完成的清理改写。退出码0只代表已确认受管进程树清理，正式接受还要求控制正常结束、输入未变、完整槽位和独立数值隔离全部通过。'),
        paragraph('恢复批次必须从头冻结，不能把旧2槽拼进新结果。新进程时钟从提交槽监督记录前计起，包含该额外开销；所有方法口径一致，但与旧版本时钟不直接混除。外层仍调用同步文件与系统API，所以这不是任意操作系统故障下的硬实时保证。', 'small'),
    ]), dict(title='14  最终公平批次：待独立验收', items=[
        paragraph('最终恢复批次尚未完成；本页不得作为可发布结论。'),
    ]), dict(title='15  贡献归因与下一步判断', items=[
        paragraph('优化应分成空间、时间调度、非线性算子与启动代价分别判断。同秩RFM41与Legendre36可以检验随机特征是否提供额外收益；RFM35与Legendre34的对比则必须逐种子标明实际秩，并保留Legendre的合格恒步最优选项。分段对恒步只改变调度，缓存和轻量入口只改变实现成本。'),
        paragraph('当前非线性融合把三次高层操作合为一次，但每次非线性仍完成五个逻辑变换。隔步外推虽形式二阶，却在绕流轨迹中失稳；冻结涡量的隐式方案可以满足限定能量恒等式，仍可能在相同大步下不够准确，并因迭代增加FFT工作量。局部理论性质不能替代完整误差与工作量比较。'),
        paragraph('没有证据支持整套改写Rust。已有剖析中，主机空间准备约0.058秒，只占对应约1.70秒完整求解的3.4%；主要时间在运行时准备和GPU推进。Rust或C++只有替换已测到的主机瓶颈才可能改变这部分；不会自动减少cuFFT、CUDA矩阵运算或数值时间步。该比例只适用于所测历史配置，不是通用提速上限。'),
        paragraph('电脑有32个逻辑处理器。单个受控计时任务固定一核、限制BLAS线程，是为避免各方法受不同并发干扰；因此低总CPU占用不等于求解器没有运行。独立资格与评分可以多核并行，速度排名则需要安静的串行GPU窗口。多任务吞吐量与单任务时延是不同实验，本轮没有测并发吞吐优势。'),
        paragraph('只有同时满足速度和涡量精度门槛、五个配对区块和两种完整时钟的同批次结果，才能称为稳定实用优势。若最强Legendre持平或更快，只能判定本设计尚未证明随机特征有额外优势；这组对照不能定量归因全部收益来源。没有通过完整隔离批次时，结论就是尚未建立新的公平速度证据。'),
    ])]
    out = dict(schema='rfm-v8-final-report-addendum-1', reviewed_for_publication=False,
        executive_paragraphs=['数值资格已经完成：39个配置在两种编译缓存政策下共78条T72全部通过。更小的RFM空间和更强的Legendre对照都被保留。最终恢复计时尚待完成，不能据此宣布稳定胜过传统CFD。'],
        pages=pages, evidence_sha256={str((HERE / f).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in evidence})
    target = HERE / 'report_final_round.json'
    if target.exists():
        raise FileExistsError('Do not overwrite an existing final-round review')
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    prepare()
