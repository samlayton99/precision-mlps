"""Restyle Figures 3 and 4 from their exact plotted data; see README.md."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Shared typography and axes. Panel-specific layout/legends are in the two functions.
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'axes.labelsize': 8, 'axes.labelpad': 3,
    'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.titlepad': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .7, 'ytick.major.width': .7,
    'legend.fontsize': 7, 'legend.frameon': False,
    'lines.linewidth': 1.25, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
OPTIMIZER_COLORS = {'adam': '#0072B2', 'gd': '#D55E00'}
BANDWIDTH_COLORS = ['#332288', '#0072B2', '#009E73', '#CC6677']
DPI = 600


def spectrum(data, output):
    """Figure 3: finite-kernel rate intervals and executed frozen-readout GD."""
    plt.rcdefaults()
    plt.rcParams.update(STYLE | {'savefig.pad_inches': .02})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), layout='constrained')
    bandwidths = data['bandwidths']
    for ax in axes:
        ax.set_yscale('log')
        ax.grid(axis='y', color='#D8D8D8', linewidth=.45, which='major')
        ax.set_axisbelow(True)
    for row, (rank, marker) in enumerate(zip(data['ranks'], ['o', 's', 'D'])):
        lower, rates, upper = [data[key][row] for key in ['rho_lower', 'rho', 'rho_upper']]
        axes[0].fill_between(bandwidths, lower, upper, color='#677381', alpha=.14, lw=0)
        axes[0].plot(bandwidths, lower, color='#677381', ls=(0, (3, 2)), lw=.8)
        axes[0].plot(bandwidths, upper, color='#677381', ls=(0, (3, 2)), lw=.8)
        axes[0].plot(bandwidths, rates, color='#242424', marker=marker, ms=4,
                     mfc='#D6DEE5', mec='#242424', mew=.8, zorder=4)
        axes[0].annotate(f'$i={rank}$', (bandwidths[-1], rates[-1]),
                         xytext=(5, 0), textcoords='offset points', va='center', fontsize=7.5)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xlim(.027, .43); axes[0].set_ylim(1e-9, .25)
    axes[0].set_xticks(bandwidths, ['1/32', '1/16', '1/8', '1/4'])
    axes[0].set_yticks([1e-8, 1e-5, 1e-2]); axes[0].minorticks_off()
    axes[0].set_title('a) Target-relevant rates')
    axes[0].set_xlabel(r'Bandwidth $\lambda=\gamma h$'); axes[0].set_ylabel(r'$\mu_i/\mu_1$')
    axes[0].legend(handles=[
        Line2D([], [], color='#242424', marker='o', ms=4, mfc='#D6DEE5',
               mec='#242424', mew=.8, label='Kernel'),
        Line2D([], [], color='#677381', ls=(0, (3, 2)), lw=.8, label='Theorem interval')],
        loc='lower right', handlelength=1.8, labelspacing=.25, borderaxespad=.45)

    for i, color in enumerate(BANDWIDTH_COLORS):
        axes[1].plot(data['gd_steps'], data['gd_error'][:, i], color=color, lw=1.25)
        axes[1].plot(data['bound_steps'], data['lower_error'][:, i], color=color,
                     ls=(0, (3, 2)), lw=1.05, zorder=3)
        axes[1].plot(data['marker_steps'], data['marker_error'][:, i], color=color,
                     ls='none', marker='o', ms=3.7, mfc='white', mec=color,
                     mew=.85, clip_on=False, zorder=4)
    bandwidth_legend = axes[1].legend(handles=[Line2D([], [], color=c, label=label)
        for c, label in zip(BANDWIDTH_COLORS, ['1/32', '1/16', '1/8', '1/4'])],
        title='Bandwidth λ', title_fontsize=7, loc='lower left', ncol=2,
        handlelength=1.5, columnspacing=.8, labelspacing=.25, borderaxespad=.4)
    axes[1].add_artist(bandwidth_legend)
    axes[1].legend(handles=[
        Line2D([], [], color='#242424', marker='o', ms=3.7, mfc='white', mew=.85, label='GD'),
        Line2D([], [], color='#242424', ls=(0, (3, 2)), label='Bound')],
        loc='upper right', ncol=2, handlelength=1.8, columnspacing=.9, borderaxespad=.45)
    axes[1].set_xscale('symlog', linthresh=10); axes[1].set_xlim(0, 5000000)
    axes[1].set_xticks([0, 100, 10000, 5000000], ['0', '$10^2$', '$10^4$', r'$5\!\times\!10^6$'])
    axes[1].set_ylim(1e-4, 1.5); axes[1].set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    axes[1].minorticks_off(); axes[1].set_title('b) Readout learning')
    axes[1].set_xlabel('Readout updates'); axes[1].set_ylabel('Relative training error')
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'spectrum_readout.{ext}', dpi=DPI)
    plt.close(fig)


def acquisition(data, output, interventions):
    """Figure 4: slope scaling, output error, and matched Adam pulse responses."""
    plt.rcdefaults()
    plt.rcParams.update(STYLE | {'savefig.pad_inches': .025, 'axes.labelpad': 1,
                                'xtick.major.pad': 2, 'ytick.major.pad': 2})
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85))
    fig.subplots_adjust(left=.074, right=.992, top=.85, bottom=.32, wspace=.40)
    widths = data['widths']; colors = OPTIMIZER_COLORS
    for optimizer, marker in [('adam', 'o'), ('gd', 's')]:
        color = colors[optimizer]
        seed_means = data[f'{optimizer}_slope_seed_means']
        for seed in range(seed_means.shape[1]):
            axes[0].plot(widths, seed_means[:, seed], color=color, alpha=.25, lw=.55,
                         marker=marker, ms=2.5, mec=color, mew=.45)
        axes[0].plot(widths, data[f'{optimizer}_slope_means'], color=color, lw=1.4,
                     marker=marker, ms=4.5, mfc='white', mec=color, mew=1, zorder=5)

        t = data[f'{optimizer}_error_steps']/1e6
        median = data[f'{optimizer}_error_median']
        axes[1].fill_between(t, data[f'{optimizer}_error_low'], data[f'{optimizer}_error_high'],
                             color=color, alpha=.075, lw=0)
        axes[1].plot(t, median, color=color, lw=1.35, zorder=4)
        marks = np.unique(np.linspace(0, len(t)-1, 8).astype(int))
        axes[1].plot(t[marks], median[marks], color=color, ls='none', marker=marker,
                     ms=3.1, mfc='white', mec=color, mew=.85, zorder=5, clip_on=False)

    construction = data['construction_slopes']
    axes[0].plot(widths, construction, color='#444444', ls=(0, (2, 2)), lw=1)
    axes[0].annotate(r'Supplied $\lambda=1/4$', (widths[-1], construction[-1]),
                     xytext=(0, 5), textcoords='offset points', ha='right', fontsize=6.6)
    axes[0].set_xscale('log', base=2); axes[0].set_yscale('log')
    axes[0].set_xticks(widths, [str(w) for w in widths]); axes[0].set_xlim(110, 1200)
    axes[0].set_ylim(.1, 260); axes[0].set_yticks([.1, 1, 10, 100])
    axes[0].set_xlabel('Total width $W$'); axes[0].set_ylabel(r'Mean slope $|\gamma|$')
    ft = data['frozen_adam_steps']/1e6
    axes[1].plot(ft, data['frozen_adam_error'], color=colors['adam'], ls=(0, (4, 2)), lw=1.05)
    axes[1].fill_between(ft, data['frozen_adam_low'], data['frozen_adam_high'],
                         color=colors['adam'], alpha=.07, lw=0)
    axes[1].plot(data['frozen_gd_steps']/1e6, data['frozen_gd_error'],
                 color=colors['gd'], ls=(0, (4, 2)), lw=1.05)
    highest = max(float(np.max(data['frozen_adam_high'])),
                  *(float(np.max(data[f'{o}_error_high'])) for o in colors))
    axes[1].set_yscale('log'); axes[1].set_ylim(1e-7, highest*1.3)
    axes[1].set_yticks([1e1, 1e-1, 1e-3, 1e-5, 1e-7]); axes[1].set_ylabel('Relative training error')
    axes[1].set_xlim(0, 5); axes[1].set_xticks([0, 2, 4, 5]); axes[1].set_xlabel('Updates (millions)')

    policy_styles = [
        ('gain_only', 'o', '#0072B2', 'Scalar amplification'),
        ('tracking_attenuated_variance', '^', '#AA4499', 'Tracking-attenuated\ndenominator'),
    ]
    for policy, marker, color, label in policy_styles:
        points = [r for r in interventions if r['policy'] == policy]
        axes[2].scatter([float(r['fine_path_ratio']) for r in points],
                        [float(r['slope_effect_percent']) for r in points],
                        marker=marker, s=17, facecolors='none', edgecolors=color,
                        linewidths=.85, zorder=4, label=label)
    x = np.array([float(r['fine_path_ratio']) for r in interventions])
    y = np.array([float(r['slope_effect_percent']) for r in interventions])
    axes[2].set_xscale('log'); axes[2].set_xlim(x.min()/1.2, x.max()*1.5)
    axes[2].set_ylim(y.min()-.08*np.ptp(y), y.max()+.1*np.ptp(y))
    axes[2].set_xticks([10, 100, 1000], ['$10^1$', '$10^2$', '$10^3$'])
    axes[2].set_yticks([0, 5, 10, 15])
    axes[2].axhline(0, color='#666666', ls=(0, (2, 2)), lw=.8, zorder=2)
    axes[2].set_xlabel('Fine-slope path / native')
    axes[2].set_ylabel('RMS slope change (%)')
    axes[2].legend(loc='upper right', fontsize=6.1, handlelength=.9,
                   handletextpad=.35, borderaxespad=.2, labelspacing=.5)
    for ax, title in zip(axes, ['(a) Slope scaling', '(b) Output accuracy',
                               '(c) Motion vs. scale']):
        ax.set_title(title); ax.minorticks_off()
        ax.grid(axis='y', color='#D8D8D8', lw=.45); ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[o], lw=1.3, marker=m, ms=3.5,
                       mfc='white', mew=.85, label=f'Joint {label}')
               for o, m, label in [('adam', 'o', 'Adam'), ('gd', 's', 'GD')]]
    handles += [Line2D([], [], color=colors[o], ls=(0, (4, 2)), lw=1.1, label=f'Frozen {label}')
                for o, label in [('adam', 'Adam'), ('gd', 'GD')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, -.005), ncol=4,
               handlelength=2, columnspacing=1.5, handletextpad=.5)
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'joint_acquisition.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'rendered')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, draw in [('spectrum_readout', spectrum), ('joint_acquisition', acquisition)]:
        with np.load(root/'data'/f'{name}.npz', allow_pickle=False) as data:
            if name == 'joint_acquisition':
                with (root/'data'/'adam_intervention.csv').open(newline='') as stream:
                    interventions = list(csv.DictReader(stream))
                draw(data, args.output, interventions)
            else:
                draw(data, args.output)
    print(f'Wrote PNG and PDF figures to {args.output.resolve()}')
