"""Compact review figure: measured errors, with no interpolation or clipping."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC14_theory_solver_comparison.plot import COLORS, LABELS, NAMES


def draw(out):
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    rows = [r for n in cfg['N'] for r in json.loads((out / f'N{n}.json').read_text())]
    assert len(rows) == 6 * len(cfg['N'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 15, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    fig.subplots_adjust(left=.115, right=.98, bottom=.12, top=.85)
    for name, color in zip(NAMES, COLORS):
        rr = sorted((r for r in rows if r['target'] == name), key=lambda r: r['N'])
        for arm, style, marker in [('actual', '-', 'o'), (cfg.get('comparison_arm', 'theory'), (0, (4, 2)), 's')]:
            ax.plot([r['N'] for r in rr], [r[f'{arm}_error'] for r in rr],
                    color=color, linestyle=style, linewidth=1.9, marker=marker,
                    markersize=3.2, markeredgewidth=.9,
                    markerfacecolor=color if arm == 'actual' else 'white')
    labels = LABELS[:3] + ['Chirp', 'Mixture', 'Wave packet']
    fig.legend(handles=[Line2D([], [], color=c, linewidth=2.2, label=l)
                        for c, l in zip(COLORS, labels)],
               loc='upper center', bbox_to_anchor=(.54, .99), ncol=3,
               frameon=False, columnspacing=2.5, handlelength=2.0, labelspacing=.55)
    ax.legend(handles=[
        Line2D([], [], color='.2', linewidth=2, marker='o', markersize=4,
               label='Unscaled'),
        Line2D([], [], color='.2', linewidth=2, linestyle=(0, (4, 2)),
               marker='s', markerfacecolor='white', markersize=4,
               label='Reference-scaled')],
        loc='upper right', borderaxespad=.9,
        frameon=False, handlelength=2.8)
    ax.set_yscale('log')
    ymax = 2
    if cfg.get('plot_auto_ymax'):
        maximum = max(r[f'{arm}_error'] for r in rows for arm in ['actual', cfg['comparison_arm']])
        ymax = 10**np.ceil(np.log10(maximum * 1.1))
    ax.set_ylim(2e-16, ymax)
    ax.set_xlim(48, 784)
    ax.set_xticks(cfg['N'])
    ax.yaxis.set_major_locator(FixedLocator(10. ** np.arange(-16, 1, 2)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.minorticks_off()
    ax.set_xlabel(r'$N$', labelpad=9)
    ax.set_ylabel(r'Relative $L^2$ error', labelpad=8)
    ax.grid(axis='y', color='.9', linewidth=.7)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(direction='out', length=4, width=1, pad=6)
    ax.tick_params(axis='x', labelsize=9.5)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'comparison.{ext}', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    draw(parser.parse_args().output)
