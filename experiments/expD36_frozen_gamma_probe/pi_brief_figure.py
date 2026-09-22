"""Plot the gamma mechanism, certified GD delay, and archived GD/Adam evidence.

Run ``python -m experiments.expD36_frozen_gamma_probe.pi_brief_figure``.
This one-off figure reuses completed experiments; it never trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from .review_figures import collect as collect_timing, DEFAULT, TARGETS


def collect(folder):
    timing = collect_timing(folder)
    root = folder.parent.parent
    sources = {str((folder/name).relative_to(root)): digest
               for name, digest in timing['source_sha256'].items()}

    def read(name):
        raw = (root/name).read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    summary = read('summary.json')
    cfg = summary['config']
    assert summary['complete'] and cfg['targets'] == TARGETS
    assert cfg['n'] == 512 and cfg['samples_per_cell'] == 16
    assert cfg['training_steps'] == 200000
    assert cfg['common_adam'] == dict(rate=.001, epsilon=1e-12)
    assert cfg['validation_steps'] == [40000, 42500, 45000, 47500, 50000]
    selection = read('training/N512_raw_selection.json')
    assert selection == summary['selections']['N512_raw_selection']
    cases, endpoints = {}, []
    for optimizer in ['gd', 'adam']:
        run = 'N512_raw_gd' if optimizer == 'gd' else 'N512_raw_adam_continue'
        case = read(f'training/{run}/case.json')
        last = read(f'training/{run}/evaluations.json')[-1]
        assert case['n'] == 512 and case['map'] == 'raw'
        assert case['optimizer'] == optimizer and last['step'] == 200000
        cases[optimizer] = case
        for gamma in [8, 64]:
            gi = case['gammas'].index(gamma)
            meta = read(f'dictionaries/N512_raw_g{gamma}/meta.json')
            assert meta['width'] == 559 and meta['complete']
            assert meta['matrix_hash'] == case['matrix_hashes'][gi]
            for ci, col in enumerate(case['columns']):
                assert col['initialization'] == 'zero'
                view = col.get('view')
                matches = [r for r in summary['training'] if r['run'] == run
                           and r['gamma'] == gamma and r['target'] == col['target']
                           and r.get('view') == view and r['initialization'] == 'zero']
                assert len(matches) == 1
                row = matches[0]
                assert row['steps'] == 200000 and row['failed_at'] == 0
                assert not last['failed'][gi][ci]
                assert row['train_error'] == last['train'][gi][ci]
                assert row['rate'] == case['rates'][gi][ci]
                assert row['optimizer_epsilon'] == case['epsilons'][gi][ci]
                if optimizer == 'gd':
                    assert np.isclose(row['rate']*meta['L'], .5, rtol=1e-10)
                else:
                    assert row['rate'] == selection['rates'][gi][ci]
                    assert row['optimizer_epsilon'] == selection['epsilons'][gi][ci]
                    if view == 'common':
                        assert row['rate'] == .001 and row['optimizer_epsilon'] == 1e-12
                endpoints.append(dict(target=row['target'], gamma=gamma,
                    method=optimizer if view is None else 'adam_'+view,
                    residual=row['train_error'], initial_rate=row['rate'],
                    adam_epsilon=row['optimizer_epsilon'], run=run))
    assert cases['gd']['matrix_hashes'] == cases['adam']['matrix_hashes']
    assert len(endpoints) == 30
    ratios = []
    for target in TARGETS:
        for method in ['gd', 'adam_common', 'adam_selected']:
            pair = {r['gamma']: r['residual'] for r in endpoints
                    if r['target'] == target and r['method'] == method}
            assert len(pair) == 2 and all(np.isfinite(v) and v > 0 for v in pair.values())
            ratios.append(dict(target=target, method=method, ratio=pair[8]/pair[64]))
    # Include analytic curve samples in the provenance record, including omega=0.
    frequency = np.linspace(0, 8, 401)
    attenuation = []
    for gamma in [8, 64]:
        z = np.pi**2*frequency/gamma
        multiplier = np.ones_like(z)
        multiplier[1:] = z[1:]/np.sinh(z[1:])
        attenuation.append(dict(gamma=gamma, multiplier=multiplier.tolist()))
    return dict(frequency_cycles=frequency.tolist(), attenuation=attenuation,
        timing=[r for r in timing['cases'] if r['target'] == TARGETS[0]],
        delay_ratio=timing['ratios'][0], endpoints=endpoints, error_ratios=ratios,
        source_sha256=sources,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='Retrospective training evidence; only GD timing intervals are certified',
        gpu_hours=0)


def plot(data, output):
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9,
        'axes.titlelocation': 'left', 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    blue, orange, green = '#245c9f', '#c65525', '#278064'
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.9), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.05, 1.2]})
    ax = axes[0]
    for row, color in zip(data['attenuation'], [blue, orange]):
        ax.plot(data['frequency_cycles'], row['multiplier'], color=color,
                lw=1.8, label=rf"$\gamma={row['gamma']}$")
        for frequency in [1, 3, 5]:
            z = np.pi**2*frequency/row['gamma']
            ax.scatter(frequency, z/np.sinh(z), color=color, s=15, zorder=3)
    for frequency in [1, 3, 5]:
        ax.axvline(frequency, color='.75', lw=.6, ls=':', zorder=0)
    ax.set(title='A  Gamma filters fine scales', xlabel=r'Frequency $\omega/(2\pi)$',
           ylabel=r'Feature multiplier $M_\gamma(\omega)$', yscale='log',
           xlim=(0, 8), ylim=(8e-4, 1.35))
    ax.legend(frameon=False, loc='lower left', fontsize=8)
    ax.set_xticks([0, 1, 3, 5, 8])

    ax = axes[1]
    rows = data['timing']
    gamma = np.array([r['gamma'] for r in rows])
    low, high, hits = [np.array([r[k] for r in rows])
                       for k in ['necessary', 'sufficient', 'executed']]
    mid = (low+high)/2
    ax.plot(gamma, mid, color=orange, lw=1.5)
    ax.errorbar(gamma, mid, yerr=[mid-low, high-mid], fmt='none', color=orange,
                capsize=4, lw=1.7, label='Certified interval')
    ax.scatter(gamma, hits, color='.12', s=23, zorder=4, label='Executed GD')
    ax.set(title='B  Predicted acquisition', xlabel=r'Common slope $\gamma$',
           ylabel='Updates to 1% residual', xscale='log', yscale='log',
           xlim=(7, 77), ylim=(8e3, 4e7))
    ax.set_xticks(gamma, [str(g) for g in gamma])
    ax.text(.98, .97, '986.59x measured\n985.70-987.49x predicted',
            ha='right', va='top', transform=ax.transAxes, fontsize=7.5)
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.04, .76), fontsize=7.5)

    ax = axes[2]
    labels = ['Mixture', 'Exp. sine', 'Runge', 'Quadratic', 'Single sine']
    for method, color, marker, offset, label in [
            ('gd', blue, 'o', -.19, 'GD'),
            ('adam_common', green, 's', 0, 'Adam, common'),
            ('adam_selected', orange, '^', .19, 'Adam, selected')]:
        ratios = [next(r['ratio'] for r in data['error_ratios']
                       if r['target'] == t and r['method'] == method) for t in TARGETS]
        ax.scatter(ratios, np.arange(5)+offset, color=color, marker=marker,
                   s=23, label=label, zorder=3)
    ax.axvline(1, color='.45', lw=1, ls=':')
    ax.set(title='C  GD and Adam: five targets', xscale='log', xlim=(.8, 1600),
           ylim=(6.1, -.6), xlabel='Residual at 8 / residual at 64\nAfter 200,000 updates')
    ax.set_yticks(range(5), labels, fontsize=7.5)
    ax.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    ax.legend(frameon=False, loc='lower right', fontsize=7, handletextpad=.3,
              labelspacing=.25)
    for ax in axes:
        ax.grid(alpha=.16, lw=.6)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    fig.savefig(output/'pi_brief_three_panel.png', dpi=260)
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
    print(json.dumps(dict(sources=len(data['source_sha256']), endpoint_checks=len(data['endpoints']),
                         gd_intervals=len(data['timing']), delay_ratio=data['delay_ratio'])))


if __name__ == '__main__':
    main()
