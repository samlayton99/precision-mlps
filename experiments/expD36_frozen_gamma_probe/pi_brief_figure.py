"""Illustrate gamma attenuation and learning curves for one fixed target.

Run ``python -m experiments.expD36_frozen_gamma_probe.pi_brief_figure``.
The figure uses archived kernel forecasts and executed optimizer checkpoints.
It never trains a model, fits a decay rate, or smooths an observed trajectory.
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
from .review_figures import collect as collect_timing, DEFAULT, GAMMAS, TARGETS


COLORS = ['#315E9B', '#268B89', '#C58D2D', '#BE514B']
TARGET = TARGETS[0]


def collect(folder):
    timing = collect_timing(folder)
    root = folder.parent.parent
    sources = {str((folder/name).relative_to(root)): digest
               for name, digest in timing['source_sha256'].items()}

    def read(name):
        raw = (root/name).read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    summary = read('refinements/gamma_factorized_kernel/summary.json')
    campaign = read('refinements/capped_kernel/campaign_summary.json')
    cfg = read('manifest.json')['config']
    assert cfg['n'] == 512 and cfg['samples_per_cell'] == 16
    assert cfg['training_steps'] == 200000
    assert cfg['common_adam'] == dict(rate=.001, epsilon=1e-12)
    assert summary['targets'][0] == TARGET
    assert summary['width'] == 559 and summary['samples'] == 8193
    pilot_case = read('training/N512_raw_adam_pilot/case.json')
    cont_case = read('training/N512_raw_adam_continue/case.json')
    pilot = read('training/N512_raw_adam_pilot/evaluations.json')
    continuation = read('training/N512_raw_adam_continue/evaluations.json')
    selection = read('training/N512_raw_selection.json')
    assert pilot_case['gammas'] == cont_case['gammas']
    assert pilot_case['matrix_hashes'] == cont_case['matrix_hashes']
    assert pilot_case['map'] == cont_case['map'] == 'raw'
    assert pilot_case['n'] == cont_case['n'] == 512
    ci = next(i for i, c in enumerate(cont_case['columns'])
              if c['target'] == TARGET and c['view'] == 'common')
    gd, adam, attenuation = [], [], []
    frequency = np.linspace(0, 8, 241)
    for gamma in GAMMAS:
        dictionary = next(d for d in summary['dictionaries'] if d['gamma'] == gamma)
        actual = next(c for c in campaign['cases'] if c['id'] == dictionary['archived_case'])
        assert actual['matrix_hash'] == dictionary['matrix_hash']
        assert actual['training']['eta'] == dictionary['eta']
        hit = actual['training']['hits'][0][0]
        assert hit == dictionary['executed_hits'][0]
        certified = next(r for r in timing['cases'] if r['target'] == TARGET and r['gamma'] == gamma)
        assert certified['necessary'] <= hit <= certified['sufficient']
        path = folder/f'g{gamma}_Q2048.npz'
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            model = {k: saved[k] for k in ['rates', 'weights', 'floor']}
        forecast_hit = first_hit(model)[0]
        assert certified['necessary'] <= forecast_hit <= certified['sufficient']
        final_step = actual['training']['steps']
        final_error = actual['training']['final_train'][0]
        final_gap = abs(error(model, final_step)[0]-final_error)
        assert final_gap < 1e-10
        archive = f"refinements/capped_kernel/evidence/gd_trajectories/{actual['id']}"
        meta, training = read(archive+'/meta.json'), read(archive+'/training.json')
        assert training == actual['training']
        assert meta['matrix_hash'] == actual['matrix_hash']
        assert meta['target_hash'] == actual['target_hash']
        assert meta['eta'] == dictionary['eta']
        assert meta['map'] == 'raw' and meta['initialization'] == 'zero'
        checkpoints = [dict(step=c['step'], residual=c['train'][0])
                       for c in read(archive+'/curve.json')]
        assert all(a['step'] < b['step'] for a, b in zip(checkpoints, checkpoints[1:]))
        assert checkpoints[-1] == dict(step=final_step, residual=final_error)
        checkpoint_gap = max(abs(error(model, c['step'])[0]-c['residual'])
                             for c in checkpoints)
        assert checkpoint_gap < 1e-10
        # Select markers by update count only, never by agreement with the forecast.
        # Keep every checkpoint in the data export and the numerical comparison.
        log_steps = np.log([c['step'] for c in checkpoints])
        marker_indices = np.unique([int(np.argmin(abs(log_steps-s)))
                                    for s in np.linspace(log_steps[0], log_steps[-1], 18)])
        steps = np.unique(np.r_[0, np.geomspace(1000, final_step, 420).astype(int),
                                 forecast_hit, hit, final_step])
        residual = [float(error(model, int(n))[0]) for n in steps]
        assert np.isclose(residual[0], 1, rtol=0, atol=1e-12)
        gd.append(dict(gamma=gamma, steps=steps.tolist(), residual=residual,
            executed_hit=hit, predicted_hit=forecast_hit,
            necessary=certified['necessary'], sufficient=certified['sufficient'],
            eta=dictionary['eta'], observed_final_step=final_step,
            observed_final_residual=final_error, final_prediction_gap=float(final_gap),
            checkpoints=checkpoints, marker_indices=marker_indices.tolist(),
            maximum_checkpoint_prediction_gap=float(checkpoint_gap)))

        gi = cont_case['gammas'].index(gamma)
        pi = selection['indices'][gi][ci]
        col = pilot_case['columns'][pi]
        assert col['target'] == TARGET and col['initialization'] == 'zero'
        assert col['initial_rate'] == cont_case['rates'][gi][ci] == .001
        assert col['epsilon'] == cont_case['epsilons'][gi][ci] == 1e-12
        assert pilot[-1]['step'] == continuation[0]['step'] == 50000
        assert pilot[-1]['train'][gi][pi] == continuation[0]['train'][gi][ci]
        checkpoints = []
        for rows, column in [(pilot, pi), (continuation[1:], ci)]:
            for row in rows:
                assert not row['failed'][gi][column]
                checkpoints.append(dict(step=row['step'], residual=row['train'][gi][column]))
        assert checkpoints[0] == dict(step=0, residual=1.)
        assert checkpoints[-1]['step'] == 200000
        assert all(a['step'] < b['step'] for a, b in zip(checkpoints, checkpoints[1:]))
        adam.append(dict(gamma=gamma, checkpoints=checkpoints, initial_rate=.001, epsilon=1e-12))
        z = np.pi**2*frequency/gamma
        multiplier = np.ones_like(z)
        multiplier[1:] = z[1:]/np.sinh(z[1:])
        attenuation.append(dict(gamma=gamma, multiplier=multiplier.tolist()))
    return dict(target=TARGET, gammas=GAMMAS, frequency_cycles=frequency.tolist(),
        attenuation=attenuation, gd=gd, adam=adam, delay_ratio=timing['ratios'][0],
        source_sha256=sources,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='GD: kernel forecast, executed trajectory checkpoints, and executed first hits. Adam: saved common-recipe checkpoints.',
        gd_checkpoint_archive='/workspace/junmiaoh/experiments/precision-mlps/runs/frozen_gamma_cap_v1/cases',
        gd_marker_convention='Saved checkpoints nearest 18 equally spaced log-update counts per gamma; all checkpoints retained and checked.',
        adam_line_convention='Straight segments connect saved checkpoints; no resampling or smoothing.',
        gpu_hours=0)


def plot(data, output):
    ink, gray = '#263442', '#78838C'
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10.5,
        'axes.labelsize': 9, 'axes.titlelocation': 'left', 'axes.titlepad': 11,
        'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9AA3AA',
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35),
                             gridspec_kw={'width_ratios': [1, 1.65, 1]})
    fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.74, wspace=.48)
    handles = [Line2D([], [], color=c, lw=2, label=rf'$\gamma={g}$')
               for g, c in zip(GAMMAS, COLORS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.015),
               ncol=4, frameon=False, handlelength=2.5, columnspacing=2.4, fontsize=10)
    ax = axes[0]
    for row, color in zip(data['attenuation'], COLORS):
        ax.plot(data['frequency_cycles'], row['multiplier'], color=color, lw=1.9)
    for frequency in [1, 3, 5]:
        ax.axvline(frequency, color='#CCD1D5', lw=.65, ls=(0, (2, 3)), zorder=0)
    ax.set(title='A  Feature attenuation', xlabel=r'Frequency $\omega/(2\pi)$',
           ylabel=r'Feature amplitude retained, $M_\gamma$', xlim=(0, 8), ylim=(0, 1.03))
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, .5, 1], ['0', '0.5', '1'])

    ax = axes[1]
    for row, color in zip(data['gd'], COLORS):
        ax.plot(row['steps'], 100*np.array(row['residual']), color=color, lw=1.7)
        saved = [row['checkpoints'][i] for i in row['marker_indices']]
        ax.scatter([c['step'] for c in saved], [100*c['residual'] for c in saved],
                   s=15, facecolors='white', edgecolors=color, linewidths=.9, zorder=4)
        ax.vlines(row['predicted_hit'], .1, 1, color=color, lw=.8, ls=(0, (2, 2)))
        ax.scatter(row['executed_hit'], 1, s=23, marker='D', color=color,
                   edgecolors='white', linewidths=.5, zorder=5)
    ax.legend(handles=[Line2D([], [], color=ink, lw=1.7, label='Kernel prediction'),
                       Line2D([], [], color=ink, lw=0, marker='o', ms=4,
                              markerfacecolor='white', label='Measured GD')],
              loc='upper right', frameon=False, fontsize=7.5, handlelength=1.8,
              borderaxespad=.15, labelspacing=.4)
    for row, color, alignment in [(data['gd'][-1], COLORS[-1], 'left'),
                                   (data['gd'][0], COLORS[0], 'right')]:
        ax.text(row['predicted_hit'], .135, f"{row['predicted_hit']:,}", color=color,
                ha=alignment, va='bottom', fontsize=7.5)
    ax.text(1000, 1.12, '1% target', color=gray, fontsize=7.5)
    ax.set(title='B  GD: prediction and training', xlabel='Gradient updates',
           ylabel='Relative residual (%)', xscale='log', yscale='log',
           xlim=(800, 30000000), ylim=(.1, 100))
    ax.set_xticks([1e3, 1e5, 1e7], [r'$10^3$', r'$10^5$', r'$10^7$'])
    ax.set_yticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])

    ax = axes[2]
    for row, color in zip(data['adam'], COLORS):
        saved = [c for c in row['checkpoints'] if c['step'] > 0]
        ax.plot([c['step'] for c in saved], [100*c['residual'] for c in saved],
                color=color, lw=1.65, marker='o', ms=2.2)
    ax.set(title='C  Adam training', xlabel='Gradient updates',
           ylabel='Relative residual (%)', xscale='log', yscale='log',
           xlim=(800, 270000), ylim=(.01, 100))
    ax.set_xticks([1e3, 1e4, 1e5], [r'$10^3$', r'$10^4$', r'$10^5$'])
    ax.set_yticks([.01, .1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
    for ax in axes[1:]:
        ax.axhline(1, color=gray, ls=(0, (4, 3)), lw=.9, zorder=0)
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=3, pad=3)
    fig.savefig(output/'pi_brief_three_panel.png', dpi=300)
    fig.savefig(output/'pi_brief_three_panel.pdf', metadata={'CreationDate': None})
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
    (output/'pi_brief_figure_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(sources=len(data['source_sha256']), gd_crossings=len(data['gd']),
        gd_checkpoints=sum(len(r['checkpoints']) for r in data['gd']),
        adam_checkpoints=sum(len(r['checkpoints']) for r in data['adam']),
        max_gd_checkpoint_gap=max(r['maximum_checkpoint_prediction_gap'] for r in data['gd']))))


if __name__ == '__main__':
    main()
