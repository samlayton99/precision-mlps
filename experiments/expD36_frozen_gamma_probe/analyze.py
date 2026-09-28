"""Artifact-only tables and figures. Scientific prose is authored separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import core

COLORS = ['#6c757d', '#d55e00', '#0072b2', '#009e73']


def bank(root, name, gamma):
    path = root/f'analysis/{name}_g{gamma}.npz'
    if not path.exists():
        path = root/f'data/{name}_g{gamma}.npz'
    with np.load(path) as data:
        return {k: data[k] for k in data.files if not k.startswith('J')}


def first_hit(trace, endpoint, epsilon):
    hits = np.flatnonzero(trace <= epsilon)
    if len(hits):
        return int(hits[0])
    if endpoint <= epsilon:
        return len(trace)
    return None


def summarize(root, cfg):
    certificates = json.loads((root/'certificates.json').read_text())
    capacities = json.loads((root/'capacity.json').read_text())
    gd, adam, comparisons = [], [], []
    for name in cfg['maps']:
        folder = root/'training'/f'gd_{name}'
        trace = np.load(folder/'trace.npz')['trace']
        evaluations = json.loads((folder/'evaluations.json').read_text())
        final = evaluations[-1]
        for gi, gamma in enumerate(cfg['gammas']):
            for ti, target in enumerate(cfg['targets']):
                row = dict(map=name, gamma=gamma, target=target, executed_steps=len(trace),
                    final_train=final['train'][gi][ti], final_eval=final['evaluation'][gi][ti],
                    nonfinite_update=final['failed'][gi][ti],
                    evaluations=[dict(step=e['step'], train=e['train'][gi][ti],
                        evaluation=e['evaluation'][gi][ti]) for e in evaluations])
                row['hits'] = []
                capacity = next(c for c in capacities if c['map'] == name and c['gamma'] == gamma
                                and c['target'] == target and c['cutoff'] == min(cfg['cutoffs']))
                for eps in cfg['tolerances']:
                    hit = first_hit(trace[:, gi, ti, 0], row['final_train'], eps)
                    prediction = next(p for p in capacity['predictions'] if p['epsilon'] == eps)
                    bounds = [c for c in certificates if c['map'] == name and c['gamma'] == gamma
                              and c['target'] == target and c['epsilon'] == eps]
                    for c in bounds:
                        comparison = dict(map=name, gamma=gamma, target=target, epsilon=eps,
                            kind=c['kind'], k=c['k'], log10_bound=c['log10_bound'],
                            bound=c['bound'], first_hit=hit, executed_budget=len(trace),
                            hit_status='executed' if hit is not None else 'budget_censored',
                            capacity_refit=capacity['train_refit'], spectral_prediction=prediction,
                            resolution=c.get('resolution', 'fp64_estimate'))
                        if c['log10_bound'] is not None:
                            comparison['log10_actual_to_bound'] = (float(np.log10(hit)-c['log10_bound'])
                                                                  if hit else None)
                            comparison['log10_censored_ratio_lower'] = (float(np.log10(len(trace))-c['log10_bound'])
                                                                       if hit is None else None)
                            comparison['log10_spectral_to_bound'] = (prediction['log10_steps']-c['log10_bound']
                                if prediction['log10_steps'] is not None else None)
                        comparisons.append(comparison)
                    row['hits'].append(dict(epsilon=eps, step=hit,
                        status='executed' if hit is not None else 'budget_censored', prediction=prediction))
                row['max_spectral_abs_discrepancy'] = max(abs(e['train'][gi][ti]-e['spectral'][gi][ti]) for e in evaluations)
                gd.append(row)
        for rate in cfg['adam_rates']:
            folder = root/'training'/f'adam_{name}_lr{rate:g}'
            if not (folder/'evaluations.json').exists():
                continue
            evaluations = json.loads((folder/'evaluations.json').read_text())
            final = evaluations[-1]
            window = [e for e in evaluations if e['step'] in [40000, 42500, 45000, 47500, 50000]]
            for gi, gamma in enumerate(cfg['gammas']):
                adam.append(dict(map=name, gamma=gamma, rate=rate, executed_steps=final['step'],
                    final_train=final['train'][gi][0], final_eval=final['evaluation'][gi][0],
                    validation_score=float(np.median([e['validation'][gi][0] for e in window])) if len(window) == 5 else None,
                    selection_eligible=len(window) == 5 and final['failed'][gi][0] == 0,
                    nonfinite_update=final['failed'][gi][0]))
    selected = []
    for name in cfg['maps']:
        for gamma in cfg['gammas']:
            eligible = [r for r in adam if r['map'] == name and r['gamma'] == gamma and r['selection_eligible']]
            if eligible:
                selected.append(min(eligible, key=lambda r: (r['validation_score'], r['rate'])))
    summary = dict(gd=gd, adam_trials=adam, adam_selected=selected, comparisons=comparisons,
        maximum_spectral_abs_discrepancy=max(r['max_spectral_abs_discrepancy'] for r in gd))
    core.write_json(root/'summary.json', summary)
    return summary


def export(fig, root, name):
    directory = root/'figures'
    directory.mkdir(exist_ok=True)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(directory/f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def access_plot(root, cfg):
    fig = plt.figure(figsize=(10, 10), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[.8, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    reference = bank(root, 'raw', 4)
    k = np.arange(len(reference['E']))
    ax.semilogy(k, reference['E'][:, 0], color='#222222', label='Sine-mixture target tail')
    ax.semilogy(k[:2], reference['E'][:2, 1], 'o-', color='#9467bd', label='Quadratic target tail')
    ax.axhline(.01, color='#777777', ls=':', label='Primary tolerance: 1%')
    ax.text(13, .002, 'Quadratic: exact tail is zero for k ≥ 2', color='#9467bd', fontsize=10)
    ax.set(xlim=(0, 64), ylim=(1e-10, 1.4), ylabel='Relative target tail $E_k$', xlabel='Polynomial degree k',
           title='Necessary target tails and readout access • raw coordinates')
    ax.legend(loc='lower left', fontsize=9, ncol=3)
    for index, gamma in enumerate(cfg['gammas']):
        ax = fig.add_subplot(gs[1+index//2, index % 2])
        b = bank(root, 'raw', gamma)
        resolved = b['mu'][:, 0] > b['access_noise'][:, 0]
        ax.semilogy(k, np.where(resolved, b['mu'][:, 0], np.nan), color='#0072b2', label='Directional μ')
        ax.semilogy(k, np.where(~resolved, b['mu'][:, 0], np.nan), color='#0072b2', ls=':', label='μ below FP64 monitor')
        measured = np.isfinite(b['b'])
        ax.semilogy(k[measured], b['b'][measured], color='#009e73', marker='.', label='Subspace b')
        ax.semilogy(k, np.exp(b['log_B']), color='#d55e00', label='Analytic B')
        ax.set(xlim=(0, 64), ylim=(1e-36, 1e3), title=f'γ = {gamma}',
               xlabel='Polynomial degree k', ylabel='Squared access')
        ax.grid(alpha=.15)
        if index == 0:
            ax.legend(fontsize=8, loc='lower left')
    export(fig, root, 'access')


def curves_plot(root, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained', sharex=True, sharey=True)
    for mi, name in enumerate(cfg['maps']):
        trace = np.load(root/'training'/f'gd_{name}'/'trace.npz')['trace']
        indices = np.unique(np.geomspace(1, len(trace)-1, 450).astype(int))
        for gi, gamma in enumerate(cfg['gammas']):
            b = bank(root, name, gamma)
            predictions = np.array([core.spectral_error(int(n), b['singular'][b['keep']], b['loadings'],
                b['floor_sq'], b['norm_y'], .5/float(b['L'])) for n in indices])
            for ti, target in enumerate(cfg['targets']):
                ax = axes[mi, ti]
                ax.loglog(indices, trace[indices, gi, ti, 0], color=COLORS[gi], lw=2, label=f'γ = {gamma}')
                ax.loglog(indices[::15], predictions[::15, ti], 'o', color=COLORS[gi], ms=3, mfc='white')
                ax.set(title=f'{name.capitalize()} • {"sine mixture" if ti == 0 else "quadratic"}',
                       xlabel='GD updates', ylabel='Training relative error', ylim=(5e-4, 1.1))
                ax.axhline(.01, color='#777777', ls=':', lw=1)
                ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Executed GD trajectories; open circles are spectral predictions', fontsize=13)
    export(fig, root, 'gd_curves')


def certificates_plot(root, cfg, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained', sharey=True)
    for ax, name in zip(axes, cfg['maps']):
        rows = [r for r in summary['comparisons'] if r['map'] == name and r['target'] == 'sine_mix_2_6_10'
                and r['epsilon'] == .01 and r['gamma'] > 1]
        for kind, label, color, marker in [('analytic', 'Analytic lower bound', '#d55e00', 's'),
                                           ('directional', 'Directional lower bound', '#0072b2', '^')]:
            selected = [r for r in rows if r['kind'] == kind]
            ax.loglog([r['gamma'] for r in selected], [10**r['log10_bound'] for r in selected],
                      color=color, marker=marker, label=label)
        chosen = [r for r in rows if r['kind'] == 'directional']
        ax.loglog([r['gamma'] for r in chosen], [10**r['spectral_prediction']['log10_steps'] for r in chosen],
                  color='#333333', ls='--', marker='d', mfc='white', label='Spectral prediction')
        for r in chosen:
            ax.plot(r['gamma'], r['first_hit'] or r['executed_budget'],
                    marker='o' if r['first_hit'] else '^', color='#009e73', ms=8, ls='none')
        ax.plot([], [], 'o', color='#009e73', label='Executed first hit')
        ax.plot([], [], '^', color='#009e73', label='Censored above 100k')
        ax.set(title=name.capitalize(), xlabel='Frozen slope γ', ylabel='GD updates to 1% training error',
               xticks=[4, 16, 64], xticklabels=['4', '16', '64'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8, loc='center right')
    fig.suptitle('GD delay and bounds • η = 0.5/L\nγ = 1 omitted here: FP64 access unresolved; see precision audit', fontsize=12)
    export(fig, root, 'gd_certificates')


def precision_plot(root, cfg, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained', sharey=True)
    for ax, name in zip(axes, cfg['maps']):
        rows = [r for r in summary['gd'] if r['map'] == name and r['target'] == 'sine_mix_2_6_10']
        errors = [next(e['evaluation'] for e in r['evaluations'] if e['step'] == 50000) for r in rows]
        ax.loglog(cfg['gammas'], errors, 'o-', color='#0072b2', label='GD')
        for collection, label, style, color in [(summary['adam_trials'], 'Adam common LR 0.001', '--', '#d55e00'),
                                                (summary['adam_selected'], 'Adam validation-selected', '-', '#009e73')]:
            selected = [r for r in collection if r['map'] == name and
                        (collection is summary['adam_selected'] or r['rate'] == .001)]
            if selected:
                ax.loglog([r['gamma'] for r in selected], [r['final_eval'] for r in selected],
                          marker='s', ls=style, color=color, label=label)
        ax.set(title=name.capitalize(), xlabel='Frozen slope γ', ylabel='Independent-grid relative error',
               xticks=cfg['gammas'], xticklabels=[str(g) for g in cfg['gammas']])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('Trained sine-mixture precision • 50,000 updates • zero initialization', fontsize=12)
    export(fig, root, 'trained_precision')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.root/'manifest.json').read_text())['config']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    summary = summarize(args.root, cfg)
    access_plot(args.root, cfg)
    curves_plot(args.root, cfg)
    certificates_plot(args.root, cfg, summary)
    precision_plot(args.root, cfg, summary)
    print(json.dumps(dict(gd_cases=len(summary['gd']), adam_trials=len(summary['adam_trials']),
                         maximum_spectral_abs_discrepancy=summary['maximum_spectral_abs_discrepancy'])))


if __name__ == '__main__':
    main()
