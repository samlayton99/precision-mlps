"""Plot reconstructed least-squares errors and coefficient amplification."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.source.read_text())
    cutoffs = [1e-12, 1e-14, 1e-16]
    lambdas = np.array([.03125, .0625, .09375, .125, .25, .5, 1.])
    targets = [('mixed_sine', 'Mixed sine', '#0072B2'),
               ('quadratic', 'Quadratic', '#D55E00')]
    metrics = [('eval_error', 'Relative output error'),
               ('coefficient_l1', r'Readout magnitude, $\|c\|_1$')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1))
    fig.subplots_adjust(left=.105, right=.98, bottom=.17, top=.74, wspace=.35)
    records = []
    for target, label, color in targets:
        for cutoff, ls, marker in zip(cutoffs, ['-', '--', ':'], ['o', '^', 'x']):
            selected = sorted([row for row in summary['rows']
                               if row['target'] == target and row['relative_cutoff'] == cutoff],
                              key=lambda row: row['lambda_value'])
            np.testing.assert_array_equal([row['lambda_value'] for row in selected], lambdas)
            for axis, (metric, _) in zip(axes, metrics):
                values = np.array([row[metric] for row in selected])
                assert np.all(np.isfinite(values)) and np.all(values > 0)
                axis.plot(lambdas, values, color=color, ls=ls, marker=marker,
                          lw=1.45 if cutoff == 1e-12 else 1.05,
                          ms=3.1, alpha=1 if cutoff == 1e-12 else .65,
                          zorder=3 if cutoff == 1e-12 else 2)
            records.extend(selected)
    for axis, (_, ylabel), title in zip(axes, metrics,
            ['A  Executed dense-grid fit', 'B  Coefficient amplification']):
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_xticks([1/32, 1/16, 1/8, 1/4, 1/2, 1],
                        ['1/32', '1/16', '1/8', '1/4', '1/2', '1'])
        axis.set_xlabel(r'Relative bandwidth, $\lambda$')
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc='left', pad=8)
        axis.grid(which='major', alpha=.15, linewidth=.5)
        axis.tick_params(length=3)
    axes[0].set_ylim(1e-14, 1e-3)
    axes[1].set_ylim(1, 1e10)
    axes[1].set_yticks(10.**np.arange(0, 11, 2))
    axes[1].tick_params(axis='y', which='minor', left=False)
    fig.legend(handles=[Line2D([], [], color=color, lw=1.8, label=label)
                        for _, label, color in targets], loc='upper center',
               bbox_to_anchor=(.5, 1.005), ncol=2, frameon=False)
    fig.legend(handles=[Line2D([], [], color='#444444', ls=ls, marker=marker,
                              ms=3.1, label=label)
                        for ls, marker, label in zip(['-', '--', ':'], ['o', '^', 'x'],
                        [r'$10^{-12}$ (primary)', r'$10^{-14}$', r'$10^{-16}$ (FP64 limit)'])],
               title='Relative SVD cutoff', title_fontsize=7,
               loc='upper center', bbox_to_anchor=(.5, .925), ncol=3,
               frameon=False, handlelength=2.3, columnspacing=1.1)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ['pdf', 'png']:
        fig.savefig(args.output / f'least_squares_attainability.{suffix}', dpi=300)
    plt.close(fig)
    (args.output / 'least_squares_figure_data.json').write_text(json.dumps(
        dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
             primary_cutoff=1e-12, evaluation_samples=8192,
             display='All cutoffs are drawn separately; no best-cutoff selection or capacity-floor inference.',
             rows=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
