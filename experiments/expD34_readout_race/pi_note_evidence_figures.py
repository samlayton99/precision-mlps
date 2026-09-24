"""Render two PI-note figures from existing per-case evidence; no training."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def points(ax, x, values, color, marker='o', label=None):
    values = np.asarray(values)
    jitter = np.linspace(-.09, .09, len(values))
    ax.scatter(x+jitter, values, s=16, color=color, marker=marker,
               alpha=.65, linewidths=0, label=label, zorder=3)
    ax.plot([x-.12, x+.12], [np.median(values)]*2, color=color, lw=2.5, zorder=4)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--inputs', type=Path, help='Optional flat staging directory for remote rendering')
    parser.add_argument('--output', type=Path, default=root/'docs/figures')
    args = parser.parse_args()
    evidence = root/'results/checkpoint_D_optimizers/expD34_readout_race'
    sources = {'heldout.csv': evidence/'effective_feedback/analysis/heldout/actual_vs_forecast.csv',
               'feedback.csv': evidence/'mechanism_persistence/evidence/feedback_summary/contrasts_20k.csv'}
    for cohort in ('development', 'confirmation'):
        sources[f'pulse_{cohort}.csv'] = evidence/f'mechanism_refinement/pulses/analysis_{cohort}.csv'
        sources[f'clone_{cohort}.csv'] = evidence/f'mechanism_refinement/splitting/{cohort}_analysis.csv'
    def table(name):
        return read(args.inputs/name if args.inputs else sources[name])
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 200})
    blue, orange = '#2563eb', '#d97706'
    rows = table('heldout.csv')
    # Multiple forecast models share each actual continuation: deduplicate cases.
    cases = {}
    for row in rows:
        if float(row['offset']) == 200000 and row['arm'] == 'joint':
            key = row['target'], row['seed'], row['start']
            pair = [float(row[f'actual_{stage}_remainder_to_effective_norm_ratio'])*100
                    for stage in ('initial', 'endpoint')]
            if key in cases:
                np.testing.assert_array_equal(cases[key], pair)
            cases[key] = pair
    assert len(cases) == 60, (len(cases), sorted({r['arm'] for r in rows}))
    data = np.asarray(list(cases.values()))
    assert np.all(data > 0)
    fig, ax = plt.subplots(figsize=(6.5, 3.0), layout='constrained')
    for pair in data:
        ax.plot([0, 1], pair, color='#94a3b8', alpha=.25, lw=.7)
    points(ax, 0, data[:, 0], blue)
    points(ax, 1, data[:, 1], orange)
    ax.axhline(100, color='#475569', ls='--', lw=1)
    ax.text(1.39, 115, 'Equal force', ha='right', fontsize=8, color='#475569')
    ax.set(yscale='log', xlim=(-.45, 1.45), xticks=[0, 1],
           xticklabels=['At continuation start', 'After 200k more updates'],
           ylabel='Remainder / effective slope force (%)',
           title='Effective fine force dominates on ten new functions')
    ax.set_ylim(float(data.min())*.55, 230)
    ax.grid(axis='y', alpha=.15)
    ax.text(.02, .85, '60 paired starts · bars mark medians', transform=ax.transAxes,
            fontsize=8, color='#475569')
    fig.savefig(args.output/'d34_pi_force_dominance.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), layout='constrained')
    stats = {'dominance_median_percent': np.median(data, axis=0).tolist(),
             'dominance_max_percent': data.max(axis=0).tolist()}
    feedback = table('feedback.csv')
    for cohort, color, marker, shift in [('development', blue, 'o', -.13),
                                         ('confirmation', orange, '^', .13)]:
        for j, arm in enumerate(('no_relaxation', 'double_geometry')):
            vals = [100*abs(float(r['positive_ratio_to_natural'])-1)
                    for r in feedback if r['cohort'] == cohort and r['arm'] == arm
                    and float(r['width']) == 1409 and float(r['horizon']) == 20000]
            assert len(vals) == 6
            points(axes[0], j+shift, vals, color, marker)
            stats[f'feedback_{cohort}_{arm}'] = vals
        pulse = [r for r in table(f'pulse_{cohort}.csv')
                 if float(r['horizon']) == 20000 and float(r['amplitude']) == .0025]
        assert len(pulse) == 23
        vals = [100*float(r['response_projection']) for r in pulse]
        points(axes[1], 0 if cohort == 'development' else 1, vals, color, marker)
        stats[f'pulse_{cohort}_range'] = [min(vals), np.median(vals), max(vals)]
        clones = [r for r in table(f'clone_{cohort}.csv')
                  if float(r['horizon']) == 20000]
        original = {(r['target'], r['seed'], r['start']): float(r['slope_displacement_norm'])
                    for r in clones if r['arm'] == 'original'}
        assert len(original) == 23
        for j, arm in enumerate(('k4_none', 'k4_readout', 'k4_hidden', 'k4_full')):
            vals = [float(r['slope_displacement_norm'])/original[r['target'], r['seed'], r['start']]
                    for r in clones if r['arm'] == arm]
            assert len(vals) == 23 and np.all(np.isfinite(vals))
            points(axes[2], j+shift, vals, color, marker,
                   cohort.capitalize() if j == 0 else None)
            stats[f'clone_{cohort}_{arm}_range'] = [min(vals), np.median(vals), max(vals)]
    axes[0].set(yscale='log', xticks=[0, 1], xticklabels=['Remove error\nrelaxation', 'Double geometry\nfeedback'],
                ylabel='Absolute change in outward travel (%)', title='A. Matched feedback\nWidth 1409 · 12 cases')
    axes[1].axhline(100, color='#475569', ls='--', lw=1)
    axes[1].set(xticks=[0, 1], xticklabels=['Development', 'Confirmation'],
                ylim=(0, 110), ylabel='Directional offset retained (%)', title='B. Small physical pulses\n23 functions per cohort')
    axes[2].axhline(1, color='#475569', ls='--', lw=1)
    axes[2].set(xticks=range(4), xticklabels=['None', 'Readout', 'Geometry', 'Both'],
                xlabel='Learning-rate correction',
                ylabel='L2 slope displacement / original', title='C. Fourfold cloning\n23 functions per cohort')
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(axis='x', labelsize=8)
    for label in axes[2].get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')
    fig.legend(*axes[2].get_legend_handles_labels(), loc='outside upper center',
               ncols=2, frameon=False)
    fig.savefig(args.output/'d34_pi_interventions.png')
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
