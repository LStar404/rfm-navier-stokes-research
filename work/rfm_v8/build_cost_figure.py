"""Plot complete T9 costs without double-counting nested compile/load timings."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'fused_march_t9_output' / 'summary.json'
OUT = ROOT / 'outputs' / 'rfm_v8_figures'

def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert data['status'] == 'passed' and not data['isolated_speed_claim_supported']
    rows = {(x['job']['pair'], x['job']['engine']): x for x in data['rows']}
    names = ['RFM37 / uniform', 'RFM41 / switched', 'FD176 / dense', 'FD184 / banded']
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.6), sharey=True)
    colors = ['#59667A', '#DD8052', '#A7C3CF', '#267F89']
    labels = ['Host preparation', 'NVRTC + module load', 'Other setup / teardown', 'March including probes']
    proof = []
    for i, ax in enumerate(axes):
        for x, engine in enumerate(('warp_cufft', 'warp_fused')):
            m = rows[i, engine]['metadata']
            # These measured scopes are disjoint; compile/load is nested in runtime.
            h = m['host_setup_seconds']
            c = m['nvrtc_compile_load_seconds']
            t = m['march_seconds']
            rest = m['seconds'] - h - c - t
            assert min(h, c, t, rest) >= 0
            segments = [h, c, rest, t]
            base = 0
            for value, color in zip(segments, colors):
                ax.bar(x, value, bottom=base, color=color, width=.6)
                base += value
            ax.text(x, base + .025, f'{base:.3f}s', ha='center', fontsize=9)
            proof.append({'pair': i, 'engine': engine, 'total': m['seconds'], 'segments': dict(zip(labels, segments))})
        ax.set_xticks([0, 1], ['cuFFT', 'Fused'])
        ax.set_title(names[i], fontsize=10)
        ax.set_ylim(0, 1.75)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Complete solver wall time (s)')
    fig.suptitle('Faster marching did not offset added compilation at T = 9', fontsize=14)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels,
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.002))
    fig.subplots_adjust(bottom=.22, top=.82, left=.065, right=.99, wspace=.16)
    path = OUT / 't9_fusion_cost_breakdown.png'
    OUT.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=220, facecolor='white')
    plt.close(fig)
    (OUT / 'cost_provenance.json').write_text(json.dumps({
        'scope': 'Four sequential diagnostic pairs, same warp observer. Not isolated timing or T72 extrapolation.',
        'source': SOURCE.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figure_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'decomposition': proof,
    }, indent=2) + '\n', encoding='utf-8')
    print(path)

if __name__ == '__main__':
    main()
