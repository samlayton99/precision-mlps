"""Paired fixed-feature/full-GD control facts and plots; no report generation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .population_output_summary import load, finite
from .mechanism_dilation_analysis import clean, digest, stats, write_csv


def identity(row):
    return tuple(row.get(k) for k in ('target', 'seed', 'start', 'arm'))


def improvement_fraction(initial, full, frozen):
    """Return a ratio only for positive, numerically distinguishable improvement."""
    reduction = initial-full
    tolerance = 1e-12*max(1., abs(initial), abs(full), abs(frozen))
    if reduction < -tolerance:
        return None, 'full_error_increased'
    if reduction <= tolerance:
        return None, 'nonpositive_or_tiny_full_improvement'
    return (initial-frozen)/reduction, 'positive_resolved_full_improvement'


def run(args):
    rows = load(args.input/'endpoints.csv'); bands = load(args.input/'slow_bands.csv')
    initial = {identity(r): r for r in rows if r['updates'] == 0}
    floor_lookup = defaultdict(list)
    for row in bands:
        floor_lookup[identity(row), row['updates']].append(row)
    for row in rows:
        start = initial[identity(row)]
        row['initial_train_error_difference'] = start['frozen_relative_l2']-start['full_relative_l2']
        row['initial_eval_error_difference'] = start['frozen_relative_eval_l2']-start['full_relative_eval_l2']
        for metric in ('relative_l2', 'relative_eval_l2'):
            full, frozen = row['full_'+metric], row['frozen_'+metric]
            row['frozen_over_full_'+metric] = frozen/full if full > 0 else None
            fraction, status = improvement_fraction(start['full_'+metric], full, frozen)
            row[metric+'_improvement_reproduced_fraction'] = fraction
            row[metric+'_improvement_ratio_status'] = status
        choices = [r for r in floor_lookup[identity(row), row['updates']]
                   if finite(r, 'relative_l2_lower_bound_through_budget')]
        if choices:
            best = max(choices, key=lambda r: r['relative_l2_lower_bound_through_budget'])
            row['best_frozen_budget_floor'] = best['relative_l2_lower_bound_through_budget']
            row['best_floor_cutoff_eta_eigenvalue'] = best['cutoff_eta_eigenvalue']
            row['best_floor_over_frozen_error'] = (row['best_frozen_budget_floor']/row['frozen_relative_l2']
                                                  if row['frozen_relative_l2'] > 0 else None)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['arm'], row['updates'])].append(row)
    metrics = ('frozen_relative_l2', 'full_relative_l2', 'frozen_relative_eval_l2', 'full_relative_eval_l2',
               'frozen_minus_full_relative_l2', 'frozen_minus_full_relative_eval_l2',
               'frozen_over_full_relative_l2', 'frozen_over_full_relative_eval_l2',
               'relative_l2_improvement_reproduced_fraction', 'relative_eval_l2_improvement_reproduced_fraction',
               'frozen_readout_rms_over_h', 'full_readout_rms_over_h',
               'best_frozen_budget_floor', 'best_floor_over_frozen_error',
               'spectral_residual_reconstruction_error')
    facts = []
    for (arm, step), group in groups.items():
        complete = [r for r in group if not r['full_failed'] and r['full_completed_updates'] == step]
        facts.append(dict(arm=arm, updates=step, cases=len(group), full_complete=len(complete),
                          targets=sorted({r['target'] for r in group}),
                          metrics={key: stats(complete, key) for key in metrics},
                          excluded_improvement_ratios={status: sum(r['relative_eval_l2_improvement_ratio_status'] == status for r in complete)
                              for status in ('full_error_increased', 'nonpositive_or_tiny_full_improvement')},
                          fp64_frozen_budget_failure_candidates={str(tol): sum(
                              r.get('best_frozen_budget_floor', 0) > tol for r in complete)
                              for tol in (.1, .01, .001, .0001, .000001)}))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'paired_cases.csv', rows)
    result = dict(groups=facts, cases=len(initial),
                  sources={str(args.input/name): digest(args.input/name) for name in ('endpoints.csv', 'slow_bands.csv', 'manifest.json')},
                  helper_sha256=digest(Path(__file__)),
                  maximum_initial_error_difference=max(abs(r['initial_train_error_difference']) for r in rows),
                  maximum_initial_eval_error_difference=max(abs(r['initial_eval_error_difference']) for r in rows),
                  interpretation=dict(improvement_fraction='(initial error - frozen error)/(initial error - full error); relative-L2 reduction, not squared loss',
                                      invalid_ratios='Undefined if full error increased or full improvement <= 1e-12*max(1,errors)',
                                      slow_floor='Maximum of six prelisted frozen-GD spectral lower bounds; FP64 audit, not interval certification',
                                      causality='Freezes a,b exactly at the shared post-repair state; readout GD uses the same eta and budget'))
    (args.output/'facts.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False)+'\n')
    plot(rows, args.output)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    labels = {arm: arm.replace('s100_', '100× ').replace('s10_', '10× ').replace('repaired', '1× repaired') for arm in arms}
    colors = {arm: f'C{i}' for i, arm in enumerate(arms)}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis_row, step in zip(axes, (20000, 100000)):
        selected = [r for r in rows if r['updates'] == step and not r['full_failed'] and r['full_completed_updates'] == step]
        for index, arm in enumerate(arms):
            group = [r for r in selected if r['arm'] == arm]
            axis_row[0].scatter([r['full_relative_eval_l2'] for r in group],
                                [r['frozen_relative_eval_l2'] for r in group], color=colors[arm], label=labels[arm], s=30)
            for key, offset, marker in (('full_readout_rms_over_h', -.12, 'o'), ('frozen_readout_rms_over_h', .12, '^')):
                values = [r[key] for r in group]
                axis_row[1].scatter(np.full(len(values), index+offset), values, color=colors[arm], marker=marker, s=25, alpha=.65)
            fractions = [r['relative_eval_l2_improvement_reproduced_fraction'] for r in group
                         if finite(r, 'relative_eval_l2_improvement_reproduced_fraction')]
            axis_row[2].scatter(index+np.linspace(-.12, .12, len(fractions)), fractions,
                                color=colors[arm], s=25, alpha=.65)
        values = [r[key] for r in selected for key in ('full_relative_eval_l2', 'frozen_relative_eval_l2') if r[key] > 0]
        if values:
            axis_row[0].plot([min(values), max(values)], [min(values), max(values)], '--', color='.4')
        axis_row[0].set(xscale='log', yscale='log', xlabel='Full-GD independent-grid relative L2',
                        ylabel='Frozen-geometry independent-grid relative L2', title=f'{step:,} updates: same initial network')
        axis_row[1].set(yscale='log', ylabel='Readout RMS / construction spacing h', title='Circles: full GD; triangles: frozen geometry')
        axis_row[2].axhline(1, color='.4', ls='--')
        axis_row[2].set(ylabel='Fraction of full-GD error reduction reproduced', title='Positive resolved improvements only')
        for ax in axis_row[1:]:
            ax.set_xticks(range(len(arms)), [labels[a].replace(' ', '\n') for a in arms], fontsize=8)
        for ax in axis_row:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fixed-feature readout GD isolates fitting from geometry evolution\n'
                 'FP64 spectral propagation of the same GD recurrence; no readout refit or singular-value truncation')
    fig.savefig(output/'frozen_geometry_comparison.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
