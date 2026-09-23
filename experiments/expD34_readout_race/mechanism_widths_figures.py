"""Plot the fixed 20k-to-40k independent-width comparison from its CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['additional_updates'] == '20000']
    names = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
    labels = ('Degree 5', 'Mixed sine', 'Gaussian', 'Compact bump', 'Tanh step', 'Kink')
    ns = np.array([128, 512, 1024])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for color, (name, label) in enumerate(zip(names, labels)):
        for seed in (30, 31):
            selected = sorted((r for r in rows if r['target'] == name and int(r['seed']) == seed),
                              key=lambda r: int(r['nref']))
            axes[0].plot([int(r['nref']) for r in selected],
                         [float(r['actual_lambda_displacement']) for r in selected],
                         'o-' if seed == 30 else 's--', color=f'C{color}', markersize=4,
                         alpha=.85 if seed == 30 else .55, label=label if seed == 30 else None)
    axes[0].axhline(0, color='.7', linewidth=.7)
    axes[0].set_yscale('symlog', linthresh=1e-8)
    axes[0].set_ylabel(r'Change in mean $\lambda=(2/N_{\rm ref})|a|$')
    axes[0].set_title('Additional scale motion over 20k updates')
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    for model, label, color, shift in (
        ('constant_effective', 'Constant effective force', '#b86e18', .97),
        ('effective_pure', 'Evolving residual; fixed map', '#236fa5', 1.03)):
        medians = []
        for n in ns:
            values = np.array([float(r[model+'_relative_vector_error']) for r in rows if int(r['nref']) == n])
            axes[1].scatter(np.full(len(values), n*shift), values, s=18, color=color, alpha=.35)
            medians.append(np.median(values))
        axes[1].plot(ns*shift, medians, 'o-', color=color, label=label)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Slope-displacement error / actual displacement')
    axes[1].set_title('Checkpoint-only forecasts: all 12 cases per width')
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel(r'Reference resolution $N_{\rm ref}$')
        ax.grid(alpha=.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
