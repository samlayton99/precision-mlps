"""Render archived width diagnostics for the self-contained PI note."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/diagnostics/width_scaling.csv'
OUTPUT = ROOT/'docs/figures/d34_pi_width_scaling.png'
TARGETS = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
LABELS = ('Degree five', 'Mixed sine', 'Gaussian', 'Compact bump', 'Tanh step', 'Kink')
WIDTHS = np.array([177, 705, 1409])


def main():
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    records = {(int(row['W']), row['target'], int(row['seed'])): row for row in rows}
    expected = {(int(w), target, seed) for w in WIDTHS for target in TARGETS for seed in (30, 31)}
    if len(rows) != 36 or set(records) != expected:
        raise ValueError('Expected all 36 target/seed/width cases without duplicates')
    if any(int(row['start']) != 20000 or int(row['window_start']) != 20000 for row in rows):
        raise ValueError('This figure uses only the 20,000-update forks')
    if any((int(row['nref']), int(row['W'])) not in ((128, 177), (512, 705), (1024, 1409)) for row in rows):
        raise ValueError('Construction resolution and physical width must remain distinct')

    def values(target, seed):
        scaled_force = np.array([float(records[(int(w), target, seed)]['W15_rms_F_a']) for w in WIDTHS])
        scaled_slope = np.array([float(records[(int(w), target, seed)]['sqrtW_rms_a']) for w in WIDTHS])
        return scaled_force/WIDTHS**1.5, scaled_force, scaled_slope

    plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.7))
    colors = plt.get_cmap('tab10').colors
    all_values = []
    for i, target in enumerate(TARGETS):
        for seed in (30, 31):
            curves = values(target, seed)
            all_values.append(curves)
            offset = .985 if seed == 30 else 1.015
            for ax, curve in zip(axes, curves):
                ax.plot(WIDTHS*offset, curve, color=colors[i], alpha=.55,
                    linewidth=.8, marker='o' if seed == 30 else '^', markersize=4)
    medians = np.median(np.asarray(all_values), axis=0)
    for i, ax in enumerate(axes):
        ax.plot(WIDTHS, medians[i], color='#202020', marker='D', markersize=5,
                linewidth=1.7, zorder=5)
        ax.set_xscale('log')
        ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
        ax.minorticks_off()
        ax.set_xlim(148, 1700)
        ax.set_xlabel('Neuron count, $W$')
        ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
    axes[0].set_yscale('log'); axes[1].set_yscale('log')
    guide_x = np.geomspace(WIDTHS[0], WIDTHS[-1], 100)
    # Offset the reference so it remains visible; its amplitude is not a fit.
    axes[0].plot(guide_x, .12*medians[0, 0]*(guide_x/WIDTHS[0])**(-1.5),
        color='#606060', linestyle='--', linewidth=1.1, zorder=2)
    axes[0].set_title('Physical force')
    axes[0].set_ylabel('RMS$(F_a)$')
    axes[1].set_title('Scaled force')
    axes[1].set_ylabel(r'$W^{3/2}\,\mathrm{RMS}(F_a)$')
    axes[2].set_title('Scaled slopes')
    axes[2].set_ylabel(r'$\sqrt{W}\,\mathrm{RMS}(a)$')
    axes[2].set_ylim(1.25, 1.76)
    handles = [Line2D([], [], color=colors[i], marker='o', markersize=4,
                      linewidth=1, label=label) for i, label in enumerate(LABELS)]
    handles += [Line2D([], [], color='#202020', marker='D', label='Median of 12 cases'),
                Line2D([], [], color='#606060', linestyle='--', label=r'$W^{-3/2}$ guide only')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .065),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Fine slope forces shrink while rescaled slopes stay comparable', fontsize=10.5, y=.985)
    fig.text(.5, .025, '20,000 GD updates; six targets, two seeds per width.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.09, right=.985, top=.82, bottom=.32, wspace=.6)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, metadata={'Description': f'Archived modal diagnostics; source: {SOURCE.relative_to(ROOT)}'})
    plt.close(fig)
    print(json.dumps({'source': str(SOURCE.relative_to(ROOT)), 'output': str(OUTPUT.relative_to(ROOT)),
        'widths': WIDTHS.tolist(), 'physical_force_medians': medians[0].tolist(),
        'scaled_force_medians': medians[1].tolist(), 'scaled_slope_medians': medians[2].tolist()}))


if __name__ == '__main__':
    main()
