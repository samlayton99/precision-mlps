"""Expose target energy in slow modes using the archived gamma-filter models.

Run as a module. Produces figures and numerical provenance only; no training,
trajectory fitting, or report generation. The single-cutoff bounds are FP64
diagnostics, separate from the existing interval-certified timing endpoints.
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

from .finite_gamma_gram import error, first_hit
from .pi_brief_figure import COLORS
from .review_figures import DEFAULT, GAMMAS, collect as collect_timing


def collect(folder):
    timing = collect_timing(folder)
    sources = dict(timing['source_sha256'])
    cutoff, comparison_step = 1e-6, 16013
    steps = np.unique(np.r_[np.geomspace(1000, 30000000, 360).astype(int), comparison_step])
    rows = []
    for gamma in GAMMAS:
        name = f'g{gamma}_Q2048.npz'
        sources[name] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        with np.load(folder/name) as saved:
            model = {k: saved[k] for k in ['rates', 'weights', 'floor']}
        rates, weights, floor = model['rates'], model['weights'][:, 0], float(model['floor'][0])
        assert np.all((rates >= 0) & (rates < 1))
        assert np.all(weights >= 0) and floor >= 0
        assert abs(floor+weights.sum()-1) < 1e-12
        order = np.argsort(rates)
        slow_mass = float(floor+weights[rates <= cutoff].sum())
        bound = np.sqrt(slow_mass)*np.exp(steps*np.log1p(-cutoff))
        residual = np.array([error(model, int(n))[0] for n in steps])
        assert np.all(bound <= residual+1e-14)
        crossing = first_hit(model)[0]
        certified = next(c for c in timing['cases']
                         if c['gamma'] == gamma and c['target'] == timing['cases'][0]['target'])
        assert certified['necessary'] <= crossing <= certified['sufficient']
        assert crossing == certified['executed']
        necessary = (int(np.ceil(np.log(np.sqrt(slow_mass)/.01)/(-np.log1p(-cutoff))))
                     if slow_mass > .01**2 else 0)
        rows.append(dict(gamma=gamma, rates=rates[order].tolist(),
            cumulative_target_energy=(floor+np.cumsum(weights[order])).tolist(),
            slow_mass=slow_mass, maximum_rate=float(rates.max()),
            comparison_residual=float(error(model, comparison_step)[0]),
            comparison_lower_bound=float(np.sqrt(slow_mass)*np.exp(comparison_step*np.log1p(-cutoff))),
            cutoff_necessary_steps=necessary, full_spectrum_hit=crossing,
            certified_necessary=certified['necessary'], certified_sufficient=certified['sufficient'],
            residual=residual.tolist(), cutoff_lower_bound=bound.tolist()))
    return dict(target='sine_mix_2_6_10', cutoff=cutoff, comparison_step=comparison_step,
        steps=steps.tolist(), rows=rows, source_sha256=sources,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='FP64 target-weighted spectra and single-cutoff lower bounds; existing primary timing endpoints independently interval-certified',
        gpu_hours=0)


def plot(data, output):
    ink, gray = '#263442', '#78838C'
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 10, 'axes.titlesize': 11,
        'axes.titlelocation': 'left', 'axes.titlepad': 10, 'text.color': ink,
        'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9AA3AA', 'axes.linewidth': .6, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.75, wspace=.34)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=rf'$\gamma={g}$')
                        for g, c in zip(GAMMAS, COLORS)],
               loc='upper center', ncol=4, frameon=False, columnspacing=2.2)
    for row, color in zip(data['rows'], COLORS):
        axes[0].step(row['rates'], 100*np.array(row['cumulative_target_energy']),
                     where='post', color=color, lw=1.6)
        axes[0].scatter(data['cutoff'], 100*row['slow_mass'], s=22, color=color, zorder=4)
        axes[1].plot(data['steps'], 100*np.array(row['residual']), color=color, lw=1.6)
        axes[1].plot(data['steps'], 100*np.array(row['cutoff_lower_bound']),
                     color=color, lw=1.2, ls=(0, (3, 2)))
    axes[0].axvline(data['cutoff'], color=gray, lw=.8, ls=':', zorder=0)
    axes[0].axhline(.01, color=gray, lw=.8, ls=':', zorder=0)
    axes[0].set(title='A  Target energy in slow modes',
        xlabel=r'Per-update rate cutoff $t$ ($a_i=\eta_\gamma\mu_i$)',
        ylabel=r'Target energy below cutoff, $S_\gamma(t)$ (%)',
        xscale='log', yscale='log', xlim=(1e-10, .6), ylim=(1e-6, 130))
    axes[0].set_xticks([1e-10, 1e-6, 1e-2])
    axes[0].set_yticks([1e-6, 1e-4, .01, 1, 100])
    axes[1].axhline(1, color=gray, lw=.8, ls=':', zorder=0)
    axes[1].axvline(data['comparison_step'], color=gray, lw=.8, ls=':', zorder=0)
    axes[1].set(title='B  What that slow mass guarantees', xlabel='Gradient updates',
        ylabel='Relative residual (%)', xscale='log', yscale='log',
        xlim=(1000, 30000000), ylim=(.01, 100))
    axes[1].set_xticks([1e3, 1e5, 1e7])
    axes[1].set_yticks([.01, 1, 100], ['0.01', '1', '100'])
    axes[1].legend(handles=[Line2D([], [], color=ink, lw=1.6, label='Full spectrum'),
        Line2D([], [], color=ink, lw=1.2, ls='--', label=r'Lower bound at $t=10^{-6}$')],
        loc='upper right', fontsize=8, frameon=False, handlelength=2)
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=3, pad=3)
    fig.savefig(output/'spectral_bridge.png', dpi=300)
    fig.savefig(output/'spectral_bridge.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = collect(args.source)
    output = args.output or args.source
    output.mkdir(parents=True, exist_ok=True)
    plot(data, output)
    (output/'spectral_bridge_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ['gamma', 'slow_mass', 'comparison_lower_bound',
        'comparison_residual', 'cutoff_necessary_steps']} for r in data['rows']]))


if __name__ == '__main__':
    main()
