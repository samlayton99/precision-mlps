"""Curated static figures from the persistence experiment's numerical tables."""
from __future__ import annotations
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = dict(full='#202020', frozen_tangent='#1f77b4', two_mode='#d62728',
              linear_features='#e68613', ten_mode='#27843b', constant_gradient='#888888')
LABELS = dict(full='Full GD', frozen_tangent='Frozen tangent', two_mode='Two slow modes',
              linear_features='Evolving readout / linear features', ten_mode='Ten-mode tanh',
              constant_gradient='Constant gradient')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f: return list(csv.DictReader(f))


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(root):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    comparison = read(root/'comparisons.csv'); mechanism = read(root/'mechanism.csv')
    generated = read(root/'generated_evolution.csv')
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for model in ('full', 'constant_gradient', 'frozen_tangent', 'two_mode'):
        r = sorted((v for v in comparison if v['model'] == model and v['seed'] == '0' and v['start'] == '600000'), key=lambda v: int(v['end']))
        ax[0].plot([int(v['end'])/1e6 for v in r], [float(v['force_norm']) for v in r],
                   color=COLORS[model], label=LABELS[model], linestyle='--' if model == 'two_mode' else '-',
                   linewidth=2.3 if model == 'full' else 1.5)
    ax[0].set(xlabel='GD update (millions)', ylabel='Effective slope-force norm', title='Force persistence, seed 0')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); ax[0].legend(fontsize=9)
    for mode, color in ((2, '#1f77b4'), (3, '#e68613')):
        r = sorted((v for v in generated if v['mode'] == str(mode) and v['seed'] == '0' and int(v['step']) >= 600000), key=lambda v: int(v['step']))
        yy = np.abs([float(v['residual']) for v in r]); yy /= yy[0]
        ax[1].plot([int(v['step'])/1e6 for v in r], yy, 'o-', color=color, label=f'Generated mode {mode}')
    r = sorted((v for v in mechanism if v['seed'] == '0' and int(v['step']) >= 600000), key=lambda v: int(v['step']))
    yy = np.abs([float(v['hard_residual']) for v in r]); yy /= yy[0]
    ax[1].plot([int(v['step'])/1e6 for v in r], yy, 's--', color='#202020', label='Hard mode 9')
    ax[1].set(xlabel='GD update (millions)', ylabel='Residual / its magnitude at 600k', title='Which error is being corrected?')
    ax[1].legend(fontsize=9); ax[1].set_ylim(bottom=0)
    fields = ('residual_effective', 'shape', 'readout', 'residual_tracking')
    values = [100*np.array([float(v['rate_'+k])/float(v['total_log_rate']) for v in mechanism]) for k in fields]
    med = np.array([np.median(v) for v in values]); low = np.array([v.min() for v in values]); high = np.array([v.max() for v in values])
    ax[2].bar(range(4), med, color=['#1f77b4', '#e68613', '#27843b', '#888888'])
    ax[2].errorbar(range(4), med, yerr=np.stack((med-low, high-med)), fmt='none', color='black', capsize=3)
    ax[2].set_xticks(range(4), ['Residual\nrelaxation', 'Geometry\nchange', 'Readout\nchange', 'Tracking'])
    ax[2].axhline(0, color='black', linewidth=.5)
    ax[2].set(ylabel='Share of log-force decline (%)', title='Median and range over audited states')
    finish(fig, root/'persistence_relaxation')

    initial = np.load(root/'predictions_600000.npz')['p0'][:, :177]
    initial_mean = np.mean(abs(initial), axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for model in ('constant_gradient', 'frozen_tangent', 'two_mode', 'linear_features', 'ten_mode', 'full'):
        r = [v for v in comparison if v['model'] == model and v['start'] == '600000']
        times = sorted(set(int(v['end']) for v in r)); force = []; gamma = []
        for step in times:
            group = [v for v in r if int(v['end']) == step]
            force.append([float(v['force_relative_error']) for v in group])
            gamma.append([float(v['mean_gamma'])-initial_mean[int(v['seed'])] for v in group])
        xx = np.array(times)/1e6
        style = '--' if model in ('two_mode', 'ten_mode') else '-'
        if model != 'full':
            mid = np.array([np.median(v) for v in force]); lower = np.array([min(v) for v in force]); upper = np.array([max(v) for v in force])
            ax[0].plot(xx, np.where(mid > 0, mid, np.nan), style, color=COLORS[model], label=LABELS[model])
            ax[0].fill_between(xx, np.where(lower > 0, lower, np.nan), upper, color=COLORS[model], alpha=.09)
        ax[1].plot(xx, [np.median(v) for v in gamma], style, color=COLORS[model], label=LABELS[model], linewidth=2 if model == 'full' else 1.5)
    ax[0].set(yscale='log', xlabel='GD update (millions)', ylabel='Relative effective-force vector error', title='Median and range across five seeds')
    ax[1].set(xlabel='GD update (millions)', ylabel='Change in mean gamma from 600k', title='Predicted and actual scale motion')
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); ax[1].axhline(0, color='black', linewidth=.5)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    finish(fig, root/'persistence_forecasts')

    summary = read(root/'bound_summary.csv'); tubes = read(root/'enclosures.csv.gz'); validation = read(root/'bound_validation.csv')
    energy = read(root/'energy_bounds.csv')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xx = np.arange(len(summary))
    ax[0].bar(xx-.25, [int(v['local_updates'])/1e6 for v in summary], width=.25, label='Starting-state gradient bound', color='#aaaaaa')
    ax[0].bar(xx, [int(v['enclosed_updates'])/1e6 for v in summary], width=.25, label='Frozen-forecast enclosure', color='#1f77b4')
    ax[0].bar(xx+.25, [int(v['updates'])/1e6 for v in energy], width=.25, label='Local hard-loss floor', color='#27843b')
    ax[0].set_xticks(xx, [f"{v['seed']}\n{int(v['start'])//1000}k" for v in summary])
    ax[0].set(xlabel='Seed / starting update', ylabel='Excluded horizon (million further updates)', title=r'Computed exclusion of $\gamma\geq1$')
    ax[0].set_yscale('log'); ax[0].set_ylim(.02, 80)
    ax[0].legend(fontsize=8)
    r = [v for v in tubes if v['seed'] == '0' and v['start'] == '100000' and v['closed'] == 'True']
    ax[1].plot([int(v['end'])/1e6 for v in r], [float(v['end_error']) for v in r], color='#1f77b4', label='Predictive parameter radius')
    v = [a for a in validation if a['seed'] == '0' and a['start'] == '100000']
    ax[1].plot([(int(a['step'])-100000)/1e6 for a in v], [float(a['parameter_error']) for a in v], 'o', color='black', markersize=3, label='Independent sampled error')
    ax[1].axvline(int(r[-1]['end'])/1e6, color='#888888', linestyle=':', label='Last enclosed block')
    ax[1].set(yscale='log', xlabel='Further GD updates (millions)', ylabel='Euclidean parameter error / radius', title='Seed 0 from 100k; ordinary FP64 evaluation')
    ax[1].legend(fontsize=8)
    finish(fig, root/'persistence_bounds')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    figures(p.parse_args().root)
