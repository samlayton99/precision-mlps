"""Render the paper figure from verified, exported observations and bounds.

Run after ``paper_training_failure_data``. Rendering needs only figure_data.json;
it does not load original experiments, train models, or calculate spectra.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/paper_training_failure/figure_data.json'
GAMMA_COLORS = {8: '#276492', 12: '#cd7338', 16: '#238778', 64: '#8564a3'}
OPTIMIZER_COLORS = {'gd': '#315E9B', 'adam': '#B45332'}
INK, GRAY = '#263442', '#68747D'


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9.5,
        'axes.titlelocation': 'left', 'axes.titlepad': 10,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#A3ABB1', 'axes.linewidth': .6,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def clean(ax):
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(length=3, pad=3)
    ax.set_axisbelow(True)


def gamma_handles():
    return [Line2D([], [], color=c, lw=1.7, label=rf'$\gamma={g}$')
            for g, c in GAMMA_COLORS.items()]


def main_figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.05),
                             gridspec_kw={'width_ratios': [1.12, 1, 1]})
    fig.subplots_adjust(left=.080, right=.972, bottom=.32, top=.87, wspace=.48)
    ax = axes[0]
    for row in data['gd']:
        color = GAMMA_COLORS[row['gamma']]
        steps, error = np.asarray(row['steps'])[1:], np.asarray(row['error'])[1:]
        markers = np.unique([np.argmin(abs(np.log(steps)-t))
                             for t in np.linspace(np.log(steps[0]), np.log(steps[-1]), 13)])
        ax.plot(steps, error, color=color, lw=1.1, alpha=.8)
        ax.plot(steps[markers], error[markers], lw=0, marker='o', ms=2.8,
                mfc='white', mec=color, mew=.75, zorder=4)
        bound_steps, bound = np.asarray(row['bound_steps']), np.asarray(row['bound_error'])
        keep = bound_steps >= steps[0]
        ax.plot(bound_steps[keep], bound[keep], color=color, lw=1.35, ls=(0, (3, 2)))
    ax.set(title='A  Readout learning', xlabel='GD updates', ylabel='Relative output error',
           xscale='log', yscale='log', xlim=(800, 2.2e7), ylim=(1e-3, 1))
    ax.set_xticks([1e3, 1e5, 1e7])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.legend(handles=gamma_handles(), loc='upper right', ncol=2,
              columnspacing=.7, handlelength=1.25, handletextpad=.35, borderaxespad=.15)
    ax.text(.5, -.37, 'Solid + markers: executed GD\nDashed: theorem lower bound',
            transform=ax.transAxes, ha='center', va='top', fontsize=8, linespacing=1.25)

    steps = np.asarray(data['joint']['steps'])
    keep = steps > 0
    for ax, metric in zip(axes[1:], ('train_error', 'lambda_rms')):
        for optimizer, color in OPTIMIZER_COLORS.items():
            values = np.asarray(data['joint'][optimizer][metric])[:, keep]
            median = np.median(values, axis=0)
            ax.fill_between(steps[keep], values.min(axis=0), values.max(axis=0),
                            color=color, alpha=.13, linewidth=0)
            ax.plot(steps[keep], median, color=color, lw=1.65,
                    label='GD' if optimizer == 'gd' else 'Adam')
            ax.plot(steps[-1], median[-1], marker='o', color=color, ms=3, zorder=4)
        ax.set(xscale='log', yscale='log', xlim=(.8, 9e5), xlabel='Joint-training updates')
        ax.set_xticks([1, 1e2, 1e4, 6e5], ['1', r'$10^2$', r'$10^4$', r'$6\!\times\!10^5$'])
    axes[1].set(title='B  Joint-training error', ylabel='Relative output error', ylim=(.02, 1.5))
    axes[1].set_yticks([.03, .1, .3, 1], ['0.03', '0.1', '0.3', '1'])
    axes[1].legend(loc='lower left', handlelength=1.7, borderaxespad=.2)
    witness = data['construction']['train_error']
    exponent = int(np.floor(np.log10(witness)))
    mantissa = witness/10**exponent
    axes[1].text(.5, -.37, 'Direct-fit reference (below axis)\n'
                 + rf'$e={mantissa:.1f}\times10^{{{exponent}}}$',
                 transform=axes[1].transAxes, ha='center', va='top', fontsize=8, linespacing=1.25)
    axes[2].set(title='C  Slope acquisition', ylabel=r'Normalized RMS slope, $\lambda_{\rm RMS}$',
                ylim=(1e-3, .42))
    axes[2].set_yticks([.001, .01, .1], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    axes[2].axhline(data['construction']['lambda_rms'], color=GRAY, lw=1, ls=(0, (4, 3)))
    axes[2].text(.04, .935, r'Constructed dictionary: $0.25$', color=GRAY,
                 transform=axes[2].transAxes, fontsize=8, va='bottom')
    axes[2].text(.5, -.37, 'Same runs as B\nMedian and full range of 5 seeds',
                 transform=axes[2].transAxes, ha='center', va='top', fontsize=8, linespacing=1.25)
    for ax in axes:
        clean(ax)
    return fig


def supporting_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.45), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.subplots_adjust(left=.087, right=.98, bottom=.32, top=.82, wspace=.33)
    for gi, row in enumerate(data['gd']):
        color = GAMMA_COLORS[row['gamma']]
        steps = np.asarray(row['steps'])[1:]
        ratio = np.asarray(row['lower_actual_ratio'])[1:]
        axes[0].plot(steps, ratio, color=color, lw=1.5)
        axes[0].plot(steps[-1], ratio[-1], color=color, marker='o', ms=3)
        positions = np.arange(len(row['crossings']))+(gi-1.5)*.11
        necessary = [c['necessary_updates'] for c in row['crossings']]
        axes[1].plot(positions, necessary, color=color, lw=1.25, ls=(0, (3, 2)))
        for x, c in zip(positions, row['crossings']):
            if c['executed_updates'] is not None:
                axes[1].plot(x, c['executed_updates'], 'o', color=color, ms=4)
            elif c['censored']:
                limit = c['bracket_lower']
                axes[1].plot(x, limit, marker='^', mfc='white', mec=color, ms=4)
                axes[1].annotate('', xy=(x, limit*2.5), xytext=(x, limit*1.1),
                                 arrowprops={'arrowstyle': '->', 'lw': .8, 'color': color})
            else:
                lower, upper = c['bracket_lower'], c['bracket_upper']
                axes[1].vlines(x, lower, upper, color=color, lw=1.4)
                axes[1].hlines([lower, upper], x-.055, x+.055, color=color, lw=1.2)
    axes[0].axhline(1, color=GRAY, lw=.8, ls=(0, (2, 3)))
    axes[0].set(title='A  Tightness across the trajectory', xscale='log',
                xlim=(800, 2.2e7), ylim=(.68, 1.02), xlabel='GD updates',
                ylabel=r'Lower bound / measured error')
    axes[0].set_xticks([1e3, 1e5, 1e7])
    axes[0].set_yticks([.7, .8, .9, 1])
    axes[1].set(title='B  Several tolerance crossings', yscale='log',
                ylim=(600, 2e9), xlim=(-.4, 4.4), xlabel=r'Relative-error tolerance $\varepsilon$',
                ylabel='Updates to cross tolerance')
    axes[1].set_xticks(range(5), ['0.1', '0.03', '0.01', '0.003', '0.001'])
    axes[1].set_yticks([1e3, 1e5, 1e7, 1e9])
    fig.legend(handles=gamma_handles(), loc='upper center', bbox_to_anchor=(.5, 1),
               ncol=4, columnspacing=2.2, handlelength=2)
    axes[0].text(.5, -.38, 'Every saved GD checkpoint is checked.\nNo clipping or fitted decay rates.',
                 transform=axes[0].transAxes, ha='center', va='top', fontsize=8)
    axes[1].text(.5, -.38, 'Dashed: necessary time; dot: exact observed hit\n'
                 'Bar: observed bracket; arrow: not reached',
                 transform=axes[1].transAxes, ha='center', va='top', fontsize=8)
    for ax in axes:
        clean(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=ROOT/'output/pdf')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    output_hashes, text_checks = {}, {}
    for name, plot in [('paper_training_failure_three_panel', main_figure),
                       ('paper_training_failure_validation', supporting_figure)]:
        fig = plot(data)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
        outside = [t.get_text() for t in texts
                   if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
                   or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
        assert not outside, (name, outside)
        text_checks[name] = dict(outside_canvas=outside, minimum_font_points=min(t.get_fontsize() for t in texts))
        # Preserve the stated physical width, rather than changing it by tight cropping.
        for suffix in ('pdf', 'png'):
            path = args.output/f'{name}.{suffix}'
            options = {'metadata': {'CreationDate': None, 'ModDate': None}} if suffix == 'pdf' else {}
            fig.savefig(path, dpi=300, **options)
            output_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    record = dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  outputs_sha256=output_hashes, width_inches=6.75, minimum_font_points=8,
                  text_checks=text_checks,
                  matplotlib_version=matplotlib.__version__)
    (args.output/'paper_training_failure_render.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
