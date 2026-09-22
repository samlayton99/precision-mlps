"""Compare learned Adam scales with frozen-gamma and fixed-center rescaling sweeps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import adam_analyze as aa, adam_forces as af, adam_summarize as summary, mechanism, targets

GAMMAS = (*mechanism.GAMMAS, 32., 64., 128., 256.)
FACTORS = (.25, .5, 1., 2., 4., 8., 16., 32.)


def measure(a, b, target, m=2048):
    x, y, mapping, scale = af.data(target, m)
    xe = targets.grid(4*m); ye = af.target_values(target, xe, mapping)/scale
    curves, _, _ = mechanism.frozen_curves(a, b, x, y, xe, ye)
    return curves


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    construction = []; scaled = []; selected = []; refinement = []; replay = []
    centers = -1+2*np.arange(-24, 153)/128
    archived = summary.read_csv(root/'geometry.csv.gz')
    old_construction = summary.read_csv(root/'analysis/construction.csv')
    for target in af.TARGETS:
        rows = []
        for gamma in GAMMAS:
            rows.extend(dict(target=target, gamma=gamma, **r) for r in measure(np.full(177, gamma), -gamma*centers, target))
        construction.extend(rows)
        for horizon in mechanism.HORIZONS[1:]:
            best = min((r for r in rows if r['updates']==horizon), key=lambda r:r['relative_train_mse'])
            selected.append(dict(family='construction', target=target, seed=-1, horizon=horizon,
                best_scale=best['gamma'], boundary=best['gamma'] in (GAMMAS[0], GAMMAS[-1]),
                best_train_error=best['relative_train_mse'], best_eval_error=best['relative_heldout_mse']))
        gamma = selected[-1]['best_scale']
        refined = measure(np.full(177, gamma), -gamma*centers, target, 4096)[-1]
        refinement.append(dict(family='construction', target=target, seed=-1, scale=gamma,
            original_error=selected[-1]['best_eval_error'], refined_error=refined['relative_heldout_mse']))
        for r in rows:
            if r['gamma'] in mechanism.GAMMAS:
                old = next(z for z in old_construction if z['target']==target and float(z['gamma'])==r['gamma'] and int(z['updates'])==r['updates'])
                replay.append(abs(r['relative_heldout_mse']-float(old['relative_heldout_mse'])))
    for folder in sorted((root/'curated').glob('primary_*')):
        cases = json.loads((folder/'manifest.json').read_text())['cases']
        f = np.load(folder/'snapshots.npz'); end = int(np.flatnonzero(f['steps']==600000)[0])
        for i, case in enumerate(cases):
            if case['optimizer']!='adam': continue
            a, b, _ = f['p'][i, end, :-1].reshape(3, -1)
            target, seed = case['target'], case['seed']; rows = []
            for factor in FACTORS:
                rows.extend(dict(target=target, seed=seed, factor=factor, mean_gamma=float(factor*np.mean(abs(a))),
                    max_gamma=float(factor*np.max(abs(a))), **r) for r in measure(factor*a, factor*b, target))
            scaled.extend(rows)
            for horizon in mechanism.HORIZONS[1:]:
                options = [r for r in rows if r['updates']==horizon]
                best = min(options, key=lambda r:r['relative_train_mse'])
                baseline = next(r for r in options if r['factor']==1.)
                selected.append(dict(family='adam_centers', target=target, seed=seed, horizon=horizon,
                    best_scale=best['factor'], boundary=best['factor'] in (FACTORS[0], FACTORS[-1]),
                    baseline_train_error=baseline['relative_train_mse'], baseline_eval_error=baseline['relative_heldout_mse'],
                    best_train_error=best['relative_train_mse'], best_eval_error=best['relative_heldout_mse'],
                    eval_improvement=baseline['relative_heldout_mse']/best['relative_heldout_mse'],
                    original_mean_gamma=float(np.mean(abs(a))), best_mean_gamma=best['mean_gamma']))
            factor = selected[-1]['best_scale']
            refined = measure(factor*a, factor*b, target, 4096)[-1]
            refinement.append(dict(family='adam_centers', target=target, seed=seed, scale=factor,
                original_error=selected[-1]['best_eval_error'], refined_error=refined['relative_heldout_mse']))
            for row in (r for r in rows if r['factor']==1.):
                old = next(r for r in archived if r['bundle']==folder.name and int(r['case_index'])==i and r['kind']=='learned' and int(r['step'])==600000 and int(r['updates'])==row['updates'])
                replay.append(abs(row['relative_heldout_mse']-float(old['relative_heldout_mse'])))
            print(json.dumps(dict(target=target, seed=seed, best_factor=factor)), flush=True)
    for name, rows in [('construction.csv', construction), ('scaled_adam.csv', scaled),
                       ('selected.csv', selected), ('refinement.csv', refinement)]:
        aa.write_csv(output/name, rows)
    error = max(replay)
    if error>1e-10: raise AssertionError(f'Archived frozen-readout assay changed: {error}')
    manifest = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root/'geometry.csv.gz', root/'analysis/construction.csv', *sorted((root/'curated').glob('primary_*/snapshots.npz'))]},
        gammas=GAMMAS, factors=FACTORS, horizons=mechanism.HORIZONS, samples=2048, evaluation_samples=8192,
        readout_optimizer='GD', eta=.002, readout_initialization='zero',
        selection='minimum training-grid relative MSE, separately for each horizon',
        evaluation_role='independent quadrature for this diagnostic sweep, not a blind generalization estimate',
        max_archive_replay_difference=error, scaled_cases=65*len(FACTORS), construction_cases=13*len(GAMMAS))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plot(output)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9})
    construction = summary.read_csv(output/'construction.csv')
    scaled = summary.read_csv(output/'scaled_adam.csv')
    fig, axes = summary.panels(plt, 'Frozen common gamma: same readout GD, fixed construction centers')
    for ax, target in zip(axes, af.TARGETS):
        for horizon, color in [(20000, '#93c5fd'), (100000, '#3b82f6'), (600000, '#1d4ed8')]:
            rr = [r for r in construction if r['target']==target and int(r['updates'])==horizon]
            ax.plot([float(r['gamma']) for r in rr], [float(r['relative_heldout_mse']) for r in rr], 'o-', color=color, ms=3, label=f'{horizon//1000}k readout updates')
        values = [float(r['relative_heldout_mse']) for r in scaled if r['target']==target and float(r['factor'])==1 and int(r['updates'])==600000]
        ax.axhspan(min(values), max(values), color='#b45309', alpha=.12)
        ax.axhline(np.median(values), color='#b45309', ls='--', label='Adam geometry / 600k readout GD')
        ax.set(xscale='log', yscale='log', xlabel='Common gamma', ylabel='Relative evaluation MSE')
    fig.axes[0].legend(fontsize=6)
    summary.save(fig, output, 'common_gamma', plt)
    fig, axes = summary.panels(plt, 'Scale Adam geometry while preserving centers; readout GD for 600k updates')
    for ax, target in zip(axes, af.TARGETS):
        for seed in range(5):
            rr = [r for r in scaled if r['target']==target and int(r['seed'])==seed and int(r['updates'])==600000]
            ax.plot([float(r['factor']) for r in rr], [float(r['relative_heldout_mse']) for r in rr], 'o-', lw=1, ms=3, alpha=.7, label=f'Seed {seed}')
        ax.axvline(1., color='black', ls=':', label='Learned scales')
        ax.set(xscale='log', yscale='log', xlabel='Multiplier of both a and b', ylabel='Relative evaluation MSE')
        ax.set_xticks([.25, 1, 4, 16, 32], ['0.25', '1', '4', '16', '32'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.axes[0].legend(fontsize=6, ncol=2)
    summary.save(fig, output, 'adam_rescaled', plt)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures-only', action='store_true')
    args = parser.parse_args()
    if args.figures_only: plot(args.output)
    else: analyze(args.root, args.output)
