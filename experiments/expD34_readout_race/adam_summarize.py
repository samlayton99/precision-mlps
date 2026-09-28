"""Tables and figures for the completed D34 Adam extension; no prose output."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from . import adam_forces as af, adam_analyze as aa, targets

LABELS = dict(sine='Sine', runge='Runge', moment3='Degree 3', moment4='Degree 4',
    moment5='Degree 5', moment9='Degree 9', mixed_sine='Mixed sine',
    localized_sine='Localized sine', chirp='Chirp', blend_m010='Mix s=-0.1',
    blend_p001='Mix s=0.01', blend_p010='Mix s=0.1', blend_p030='Mix s=0.3')
COLORS = dict(gd='#2563eb', adam='#b45309', momentum='#7c3aed', adaptive_only='#059669')


def read_csv(path):
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path, 'rt') as f:
        return list(csv.DictReader(f))


def num(row, key):
    return float(row[key]) if row.get(key, '') else np.nan


def primary(row):
    return row['bundle'].startswith('primary_')


def traces(root):
    result = []
    source = root/'curated' if (root/'curated').exists() else root/'raw'
    for folder in sorted(source.glob('primary_*')):
        manifest = json.loads((folder/'manifest.json').read_text())
        f = np.load(folder/'trace.npz')
        for i, case in enumerate(manifest['cases']):
            result.append((case, f['ends']-1, {k: f['values'][i, :, j] for j, k in enumerate(manifest['metrics'])}))
    return result


def panels(plt, title):
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for ax in axes.flat[len(af.TARGETS):]: ax.set_visible(False)
    for ax, target in zip(axes.flat, af.TARGETS): ax.set_title(LABELS[target], fontsize=10)
    fig.suptitle(title)
    return fig, axes.flat


def save(fig, root, name, plt):
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(root/f'{name}.png', dpi=150); plt.close(fig)


def spread(ax, x, values, color, label=None):
    values = np.asarray(values, dtype=float)
    if not len(values): return
    ax.scatter(np.full(len(values), x), values, color=color, alpha=.35, s=12)
    ax.plot([x-.08, x+.08], [np.median(values)]*2, color=color, lw=2, label=label)


def summarize(root, partial=False):
    folders = [p for p in (root/'analysis').iterdir() if p.is_dir() and (p/'audit.json').exists()]
    endpoints = [r for p in folders for r in read_csv(p/'endpoints.csv')]
    windows = [r for p in folders for r in read_csv(p/'windows.csv')]
    geometry = [r for p in folders for r in read_csv(p/'geometry.csv.gz')]
    states = [r for p in folders for r in read_csv(p/'states.csv.gz')]
    modal = [r for p in folders for r in read_csv(p/'modal.csv')]
    residuals = [r for p in folders for r in read_csv(p/'residuals.csv.gz')]
    if not partial and len(endpoints)!=184:
        raise ValueError(f'Expected 184 completed cases; found {len(endpoints)}')
    for name, rows in [('endpoints.csv', endpoints), ('windows.csv', windows), ('geometry.csv.gz', geometry),
                       ('states.csv.gz', states), ('modal.csv', modal), ('residuals.csv.gz', residuals)]:
        aa.write_csv(root/name, rows)
    summaries = []
    keys = sorted({(r['target'], r['optimizer'], r['eta'], r['epsilon'], primary(r)) for r in endpoints})
    for target, optimizer, eta, epsilon, main in keys:
        rr = [r for r in endpoints if (r['target'], r['optimizer'], r['eta'], r['epsilon'], primary(r))==(target, optimizer, eta, epsilon, main)]
        row = dict(target=target, optimizer=optimizer, eta=eta, epsilon=epsilon, primary=main, seeds=len(rr))
        for field in ('relative_eval_mse', 'frozen_relative_mse', 'mean_gamma', 'median_gamma', 'max_gamma', 'path',
                      'raw_tracking_ratio', 'step_tracking_ratio', 'preconditioned_raw_tracking_ratio'):
            values = [num(r, field) for r in rr]
            row.update({field+'_median': np.nanmedian(values), field+'_min': np.nanmin(values), field+'_max': np.nanmax(values)})
        summaries.append(row)
    aa.write_csv(root/'summary.csv', summaries)
    (root/'analysis_checks.json').write_text(json.dumps(dict(cases=len(endpoints),
        failed=sum(num(r, 'failed')>0 for r in endpoints),
        unresolved_steps=sum(num(r, 'unresolved_steps') for r in endpoints),
        max_modal_reconstruction=max(num(r, 'reconstruction_error') for r in modal),
        max_preconditioned_identity=max(num(r, 'preconditioned_coarse_identity_error') for r in states)), indent=2)+'\n')
    population_rows = []
    source = root/'curated' if (root/'curated').exists() else root/'raw'
    for folder in sorted(source.glob('primary_*')):
        cases = json.loads((folder/'manifest.json').read_text())['cases']
        f = np.load(folder/'snapshots.npz'); end = int(np.flatnonzero(f['steps']==600000)[0])
        for i, case in enumerate(cases):
            a = abs(f['p'][i, end, :177])
            population_rows.append(dict(**case, mean_gamma=a.mean(), max_gamma=a.max(),
                fraction_3p2=np.mean(a>=3.2), fraction_16=np.mean(a>=16)))
    aa.write_csv(root/'populations.csv', population_rows)
    plot(root, endpoints, windows, geometry, states, residuals)


def plot(root, endpoints, windows, geometry, states, residuals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9})
    trace = traces(root)
    fig, axes = panels(plt, 'Training targets on the common interval')
    for ax, target in zip(axes, af.TARGETS):
        x, y, _, _ = af.data(target)
        ax.plot(x, y, color='#111827'); ax.set_xlabel('x'); ax.set_ylabel('Target value')
    save(fig, root, 'targets', plt)
    for optimizer in ('gd', 'adam'):
        fig, axes = panels(plt, f'{optimizer.upper()}: raw slope force and its two contributions')
        for ax, target in zip(axes, af.TARGETS):
            rr = [(t, v) for c, t, v in trace if c['optimizer']==optimizer and c['target']==target]
            for field, color, label in [('raw_total_norm', '#111827', 'Full'), ('raw_effective_norm', '#2563eb', 'Effective fine'),
                                       ('raw_tracking_norm', '#d97706', 'Coarse tracking')]:
                if not rr: continue
                common = rr[0][0]; yy = np.array([np.interp(common, t, v[field]) for t, v in rr])
                ax.plot(common, np.median(yy, axis=0), color=color, label=label, lw=1)
                ax.fill_between(common, yy.min(axis=0), yy.max(axis=0), color=color, alpha=.1)
            ax.set(xscale='symlog', yscale='log', xlabel='Update', ylabel='Gradient norm', ylim=(1e-12, 10))
            ax.set_xscale('symlog', linthresh=10)
        fig.axes[0].legend(fontsize=7)
        save(fig, root, 'raw_forces_'+optimizer, plt)

    fig, axes = panels(plt, 'Adam: tracking share at sampled states after update 20,000')
    stages = ('raw', 'moment', 'scaled_current', 'step')
    for ax, target in zip(axes, af.TARGETS):
        for k, stage in enumerate(stages):
            values = []
            for c, t, v in trace:
                if c['optimizer']!='adam' or c['target']!=target: continue
                tracking = v[stage+'_tracking_norm']; effective = v[stage+'_effective_norm']
                values.append(np.median((tracking/(tracking+effective+1e-300))[t>=20000]))
            spread(ax, k, values, COLORS['adam'])
        ax.set(xticks=range(4), xticklabels=['Raw', 'Moment', 'P g', 'Step'],
               ylim=(0, 1.05), ylabel='Tracking / (tracking + effective)')
    save(fig, root, 'adam_stages', plt)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    for row, optimizer in enumerate(('gd', 'adam')):
        for col, target in enumerate(af.CONTROLS):
            ax = axes[row, col]
            rr = sorted([r for r in windows if primary(r) and r['target']==target and
                r['optimizer']==optimizer and int(r['start'])==20000 and int(r['end'])==600000], key=lambda r: int(r['seed']))
            pos = np.zeros(len(rr)); neg = pos.copy(); xx = np.arange(len(rr))
            for field, color, label in [('signed_channels_effective', '#2563eb', 'Effective'),
                ('signed_channels_tracking', '#d97706', 'Tracking'), ('crossing', '#9ca3af', 'Crossing remainder')]:
                yy = np.array([num(r, field) for r in rr]); base = np.where(yy>=0, pos, neg)
                ax.bar(xx, yy, bottom=base, color=color, label=label)
                pos += np.maximum(yy, 0); neg += np.minimum(yy, 0)
            ax.scatter(xx, [num(r, 'mean_gamma_change') for r in rr], color='black', marker='D', s=20, label='Net change', zorder=4)
            if len(rr):
                lower = min(0., neg.min()); upper = max(0., pos.max()); padding = .12*(upper-lower)
                ax.set_ylim(lower-padding, upper+padding)
            ax.axhline(0, color='black', lw=.6)
            ax.set(title=LABELS[target]+' / '+optimizer.upper(), xlabel='Seed', ylabel='Mean gamma change', xticks=xx)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Signed motion from update 20,000 to 600,000; each bar is one trajectory')
    save(fig, root, 'signed_motion', plt)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for k, target in enumerate(af.TARGETS):
        for optimizer, shift in [('gd', -.15), ('adam', .15)]:
            rr = [r for r in endpoints if primary(r) and r['target']==target and r['optimizer']==optimizer]
            for ax, field in zip(axes, ('relative_eval_mse', 'frozen_relative_mse', 'mean_gamma')):
                spread(ax, k+shift, [num(r, field) for r in rr], COLORS[optimizer], optimizer.upper() if k==0 else None)
        initial = [num(r, 'relative_heldout_mse') for r in geometry if primary(r) and r['target']==target and
            r['optimizer']=='gd' and int(r['step'])==0 and int(r['updates'])==600000 and r['kind']=='learned']
        axes[1].scatter([k]*len(initial), initial, marker='x', color='#9ca3af', s=15, label='Initial geometry' if k==0 else None)
    for ax, label in zip(axes, ('Actual relative MSE', 'Frozen geometry U(600k)', 'Mean gamma')):
        ax.set(yscale='log', ylabel=label); ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8)
    axes[-1].set(xticks=range(13), xticklabels=[LABELS[t] for t in af.TARGETS])
    axes[-1].tick_params(axis='x', rotation=35)
    fig.suptitle('Endpoints: learned fit, common readout assay, and slopes; all five seeds')
    save(fig, root, 'geometry_and_fit', plt)

    mixture = [('blend_m010', -.1), ('moment9', 0.), ('blend_p001', .01),
               ('blend_p010', .1), ('blend_p030', .3), ('moment3', 1.)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for optimizer, shift in [('gd', -.12), ('adam', .12)]:
        for k, (target, s) in enumerate(mixture):
            rr = [r for r in endpoints if primary(r) and r['target']==target and r['optimizer']==optimizer]
            for ax, field in zip(axes[:2], ('relative_eval_mse', 'mean_gamma')):
                spread(ax, k+shift, [num(r, field) for r in rr], COLORS[optimizer], optimizer.upper() if k==0 else None)
            rr = [r for r in residuals if primary(r) and r['target']==target and r['optimizer']==optimizer and int(r['step'])==600000]
            if abs(s)<1:
                spread(axes[2], k+shift, [num(r, 'e_9')**2/(.75*(1-s*s)) for r in rr], COLORS[optimizer])
    for ax, title in zip(axes, ('Total relative MSE', 'Mean gamma', 'Remaining degree-9 error / target degree-9 energy')):
        ax.set(yscale='log', title=title, xlabel='Cubic mixture coefficient s', xticks=range(6),
               xticklabels=['-0.1', '0', '0.01', '0.1', '0.3', '1']); ax.grid(axis='y', alpha=.2)
    axes[0].legend(); axes[2].axhline(1, color='gray', ls=':', lw=.8)
    fig.suptitle('Controlled mixtures: does easy-mode learning help the hard mode?')
    save(fig, root, 'mixtures', plt)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    optimizers = ('gd', 'momentum', 'adaptive_only', 'adam')
    for col, target in enumerate(af.CONTROLS):
        for k, optimizer in enumerate(optimizers):
            rr = [r for r in endpoints if r['target']==target and r['optimizer']==optimizer and int(r['seed'])<3 and
                (primary(r) or r['bundle'].startswith('controls_'))]
            for row, field in enumerate(('relative_eval_mse', 'mean_gamma')):
                spread(axes[row, col], k, [num(r, field) for r in rr], COLORS[optimizer])
        axes[0, col].set_title(LABELS[target])
        for row, ylabel in enumerate(('Relative MSE', 'Mean gamma')):
            axes[row, col].set(yscale='log', ylabel=ylabel, xticks=range(4), xticklabels=['GD', 'EMA', 'P only', 'Adam'])
    fig.suptitle('Matched controls: first-moment memory versus adaptive scaling, seeds 0–2')
    save(fig, root, 'controls', plt)

    fig, axes = panels(plt, 'Adam rate sensitivity: seed 0, fixed 600,000 updates')
    for ax, target in zip(axes, af.TARGETS):
        rr = sorted([r for r in endpoints if r['target']==target and r['optimizer']=='adam' and int(r['seed'])==0 and
            not r['bundle'].startswith('epsilon_')], key=lambda r: num(r, 'eta'))
        ax.plot([num(r, 'eta') for r in rr], [num(r, 'relative_eval_mse') for r in rr], 'o-', color=COLORS['adam'], label='Actual fit')
        ax.plot([num(r, 'eta') for r in rr], [num(r, 'frozen_relative_mse') for r in rr], 's--', color='#2563eb', label='Frozen assay')
        ax.set(xscale='log', yscale='log', xlabel='Learning rate', ylabel='Relative MSE')
        ax.set_xticks([.0002, .001, .002], ['0.0002', '0.001', '0.002'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.axes[0].legend(fontsize=7)
    save(fig, root, 'rates', plt)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--figures-only', action='store_true')
    args = parser.parse_args()
    if args.figures_only:
        plot(args.root, *[read_csv(args.root/name) for name in ('endpoints.csv', 'windows.csv',
            'geometry.csv.gz', 'states.csv.gz', 'residuals.csv.gz')])
    else:
        summarize(args.root, args.partial)
