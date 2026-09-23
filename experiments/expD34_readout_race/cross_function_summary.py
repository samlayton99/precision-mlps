"""Summarize the archived-state audit remotely; emit evidence, never Markdown.

Counts are repeated starts/states, not independent function samples. Spectral
occupancy bounds apply to the frozen surrogate, not to ordinary GD.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

COHORTS = {'existing': 195, 'fresh': 78, 'heldout': 60}
HORIZONS = (1000, 10000, 50000, 200000)
IDENTITY = ('cohort', 'target', 'family', 'seed', 'start', 'horizon', 'threshold', 'mode')


def read_rows(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in ('cohort', 'target', 'family'):
                continue
            row[key] = (value == 'True' if value in ('True', 'False') else float(value))
    return rows


def safe_ratio(a, b):
    return a / b if b > 0 else math.nan


def case_id(row):
    return {k: row[k] for k in IDENTITY if k in row}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(finite_json(v)) if isinstance(v, (dict, list))
                             else finite_json(v) for k, v in row.items()})


def describe(rows, key):
    valid = [(float(r[key]), r) for r in rows if math.isfinite(float(r[key]))]
    values = [v for v, _ in valid]
    return dict(count=len(rows), finite_count=len(values), unresolved_count=len(rows)-len(values),
                positive_count=sum(v > 0 for v in values), negative_count=sum(v < 0 for v in values),
                zero_count=sum(v == 0 for v in values),
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                maximum_absolute=max(map(abs, values)) if values else None,
                maximum_case=case_id(max(valid, key=lambda p: p[0])[1]) if valid else None,
                maximum_absolute_case=case_id(max(valid, key=lambda p: abs(p[0]))[1]) if valid else None)


def summarize(rows, table):
    """Retain raw case summaries; label every pooling level explicitly."""
    results = []
    extra = tuple(k for k in ('threshold', 'mode') if k in rows[0])
    metrics = [k for k in rows[0] if k not in IDENTITY]
    for level, keys in [('cohort', ('cohort', 'horizon')+extra),
                        ('function', ('cohort', 'target', 'family', 'horizon')+extra),
                        ('family', ('cohort', 'family', 'horizon')+extra)]:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        for values, members in sorted(groups.items()):
            identity = dict(zip(keys, values))
            for metric in metrics:
                result = dict(table=table, level=level, cohort=identity['cohort'],
                              target=identity.get('target', ''), family=identity.get('family', ''),
                              horizon=identity['horizon'], threshold=identity.get('threshold'),
                              mode=identity.get('mode'), metric=metric,
                              functions=len({r['target'] for r in members}),
                              aggregation='repeated-case median and extrema', **describe(members, metric))
                # Equal-function summaries remain separate from pooled observations.
                functions = defaultdict(list)
                for r in members:
                    functions[r['target']].append(r)
                medians = [describe(rs, metric)['median'] for rs in functions.values()]
                resolved = [v for v in medians if v is not None]
                result['function_equal_mean_of_medians'] = statistics.mean(resolved) if resolved else None
                result['function_equal_median_of_medians'] = statistics.median(resolved) if resolved else None
                result['functions_with_unresolved_median'] = len(medians)-len(resolved)
                results.append(result)
    return results


def figures(output, forces, budgets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 8), layout='constrained')
    color_max = max(1., max(r['actual_new_ever_count'] for r in budgets if r['threshold'] == 1.))
    for ax, cohort in zip(axes, COHORTS):
        targets = sorted({r['target'] for r in budgets if r['cohort'] == cohort})
        cells = np.full((len(targets), len(HORIZONS)), np.nan)
        labels = {}
        for i, target in enumerate(targets):
            for j, horizon in enumerate(HORIZONS):
                rows = [r for r in budgets if r['cohort'] == cohort and r['target'] == target
                        and r['horizon'] == horizon and r['threshold'] == 1.]
                pred = describe(rows, 'predicted_new_ever_upper')['median']
                actual = describe(rows, 'actual_new_ever_count')['median']
                cells[i, j] = actual
                labels[i, j] = f'{pred:g}/{actual:g}' if pred is not None else f'?/{actual:g}'
        im = ax.imshow(cells, cmap='Blues', vmin=0, vmax=color_max, aspect='auto')
        for (i, j), label in labels.items():
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='white' if cells[i, j] > im.norm.vmax*.6 else 'black')
        ax.set(xticks=range(4), xticklabels=['1k', '10k', '50k', '200k'],
               yticks=range(len(targets)), yticklabels=targets, title=cohort,
               xlabel='Additional GD updates')
    fig.suptitle('Threshold 1: median frozen-surrogate additional-ever upper / actual new-ever count\n'
                 'Paired starts; color shows actual count. Surrogate bounds are not bounds on actual GD.', fontsize=12)
    fig.savefig(output/'01_surrogate_and_actual_acquisition.png', dpi=180)
    plt.close(fig)

    groups = [(cohort, target) for cohort in COHORTS
              for target in sorted({r['target'] for r in forces if r['cohort'] == cohort})]
    metrics = ('map_defect_a_relative_effective', 'residual_defect_a_relative_effective',
               'remainder_defect_a_relative_effective')
    values = np.full((len(groups), 3), np.nan)
    labels = {}
    for i, (cohort, target) in enumerate(groups):
        rows = [r for r in forces if r['cohort'] == cohort and r['target'] == target
                and r['horizon'] == 200000]
        for j, metric in enumerate(metrics):
            median = describe(rows, metric)['median']
            labels[i, j] = 'unresolved' if median is None else ('0' if median == 0 else f'{median:.2g}')
            if median is not None and median > 0:
                values[i, j] = math.log10(median)
    fig, ax = plt.subplots(figsize=(9, max(8, .30*len(groups))), layout='constrained')
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#ededed')
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect='auto')
    for (i, j), label in labels.items():
        ax.text(j, i, label, ha='center', va='center', fontsize=8,
                color='white' if math.isfinite(values[i, j]) and im.norm(values[i, j]) < .5 else 'black')
    ax.set(xticks=range(3), xticklabels=['Map defect', 'Residual defect', 'Remainder'],
           yticks=range(len(groups)), yticklabels=[f'{c}: {t}' for c, t in groups],
           title='At +200k: median pointwise defect norm / actual effective-force norm\n'
                 'Components can cancel; these are endpoint diagnostics, not path bounds.')
    fig.colorbar(im, ax=ax, label='log10(ratio); zero and unresolved cells are gray and labeled')
    fig.savefig(output/'02_endpoint_force_defects.png', dpi=180)
    plt.close(fig)


def report_metrics(forces, budgets, verification):
    """Compact evidence for hand-written reporting; no prose or new fits."""
    metrics = (
        'tracking_relative_slope', 'tracking_relative_residual',
        'omitted_relative_slope', 'omitted_relative_residual',
        'remainder_relative_slope', 'direct_relative_effective',
        'balanced_relative_effective', 'two_term_cancellation',
        'direct_balanced_signed_cancellation', 'force_defect_cancellation',
        'map_defect_a_relative_effective', 'residual_defect_a_relative_effective',
        'remainder_defect_a_relative_effective', 'gain_derivative_relative_residual',
        'fixed_motion_error', 'affine_motion_error',
    )
    drivers = tuple(f'{b}_gain_force_energy_rate' for b in ('a', 'b', 'c', 'd')) + (
        'residual_force_energy_rate', 'total_force_energy_rate',
        'gain_derivative_norm', 'residual_derivative_norm')

    def compact(rows, metric):
        stats = describe(rows, metric)
        # The group supplies cohort/target/horizon; retain a compact worst-state
        # locator instead of duplicating those long labels for every metric.
        worst = stats['maximum_absolute_case']
        locator = ([worst['target'], worst['seed'], worst['start']]
                   if worst is not None else None)
        return dict(count=stats['count'], unresolved=stats['unresolved_count'],
                    median=stats['median'], minimum=stats['minimum'],
                    maximum=stats['maximum'], maximum_absolute=stats['maximum_absolute'],
                    worst=locator)

    def group(cohort, horizon, target=None):
        rows = [r for r in forces if r['cohort'] == cohort and r['horizon'] == horizon
                and (target is None or r['target'] == target)]
        selected = metrics + (drivers if horizon == 200000 else ())
        record = dict(cohort=cohort, horizon=horizon, target=target, count=len(rows),
                      function_count=len({r['target'] for r in rows}),
                      opposed_signed_count=sum(r['opposed_direct_balanced_signed'] for r in rows),
                      metrics={k: compact(rows, k) for k in selected}, budgets=[])
        for threshold in (1., 3.2, 16.):
            rr = [r for r in budgets if r['cohort'] == cohort and r['horizon'] == horizon
                  and r['threshold'] == threshold and (target is None or r['target'] == target)]
            applicable = [r for r in rr if r['applicable']]
            exceed = [r for r in applicable if r['actual_exceeds_surrogate_upper']]
            record['budgets'].append(dict(
                threshold=threshold, cases=len(rr), nonoscillatory_applicable=len(applicable),
                inapplicable=len(rr)-len(applicable),
                total_initial=sum(r['initial_count'] for r in rr),
                total_actual_new=sum(r['actual_new_ever_count'] for r in rr),
                predicted_additional_upper_sum_applicable=sum(r['predicted_new_ever_upper'] for r in applicable),
                actual_new_sum_applicable=sum(r['actual_new_ever_count'] for r in applicable),
                cases_actual_exceeds_surrogate=len(exceed),
                cases_zero_additional_budget=sum(r['predicted_new_ever_upper'] == 0 for r in applicable),
                exceed_states=[[r['target'], r['seed'], r['start']] for r in exceed],
            ))
        return record

    return dict(
        starts=333, function_instances=23,
        scope='Descriptive repeated-start evidence. Frozen-surrogate upper counts are not ordinary-GD bounds.',
        worst_locator_order=['target', 'seed', 'fork_start'],
        maximum_identity_errors=verification,
        cohort_horizons=[group(c, h) for c in COHORTS for h in HORIZONS],
        function_endpoints=[group(c, 200000, t) for c in COHORTS
                            for t in sorted({r['target'] for r in forces if r['cohort'] == c})],
    )


def run(root, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run this numerical summary in a remote Slurm allocation')
    if (output/'summary.json').exists():
        raise ValueError('Use a fresh summary output directory')
    all_rows = {key: [] for key in ('forces', 'budgets', 'modes')}
    completions, hashes = {}, {}
    for cohort, count in COHORTS.items():
        path = root/cohort/'complete.json'
        done = json.loads(path.read_text())
        if not done['complete'] or not done['verification_passed'] or done['cases'] != count:
            raise ValueError(f'Incomplete or unverified cohort: {cohort}')
        completions[cohort] = done
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in all_rows:
            path = root/cohort/(name+'.csv')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = read_rows(path)
            expected = count*5*({'forces': 1, 'budgets': 3, 'modes': 64}[name])
            keys = [tuple(r[k] for k in IDENTITY if k in r) for r in rows]
            if (len(rows) != expected or len(set(keys)) != expected
                    or {r['horizon'] for r in rows} != {0, *HORIZONS}
                    or any(r['cohort'] != cohort for r in rows)):
                raise ValueError(f'Unexpected row coverage: {path}')
            all_rows[name].extend(rows)
    if len({r['target'] for r in all_rows['forces']}) != 23:
        raise ValueError('Expected all 23 original and new target functions')
    for r in all_rows['forces']:
        r['direct_relative_effective'] = safe_ratio(r['direct_slope_norm'], r['effective_slope_norm'])
        r['balanced_relative_effective'] = safe_ratio(r['balanced_slope_norm'], r['effective_slope_norm'])
        r['direct_balanced_signed_cancellation'] = safe_ratio(abs(r['effective_signed_velocity']),
                abs(r['direct_signed_velocity'])+abs(r['balanced_signed_velocity']))
        r['gain_derivative_relative_residual'] = safe_ratio(r['gain_derivative_norm'], r['residual_derivative_norm'])
        r['opposed_direct_balanced_signed'] = r['direct_signed_velocity']*r['balanced_signed_velocity'] < 0
    for r in all_rows['budgets']:
        r['predicted_new_ever_upper'] = (r['predicted_ever_upper']-r['initial_count']
                                       if r['applicable'] else math.nan)
        r['actual_exceeds_surrogate_upper'] = (r['actual_ever_count'] > r['predicted_ever_upper']
                                             if r['applicable'] else math.nan)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, rows in all_rows.items():
        summaries[name] = summarize(rows, name)
        write_csv(output/(name+'_combined.csv'), rows)
        write_csv(output/(name+'_summary.csv'), summaries[name])
    numerical = {key: max(c['maximum_identity_errors'][key] for c in completions.values())
                 for key in completions['existing']['maximum_identity_errors']}
    compact = report_metrics(all_rows['forces'], all_rows['budgets'], numerical)
    (output/'report_metrics.json').write_text(
        json.dumps(finite_json(compact), allow_nan=False, separators=(',', ':'))+'\n')
    result = dict(starts=333, function_instances=23, cohorts=COHORTS, horizons=[0, *HORIZONS],
                  scope='Repeated-start descriptive audit; no independent-function inference. '
                        'Frozen-surrogate travel/occupancy bounds are not actual-GD exclusions.',
                  aggregation='Cohorts separate. Family/cohort pooled case statistics and equal-function '
                              'means/medians of function medians are separately labeled. Original family '
                              'label is a bookkeeping group, not a scientific function family.',
                  verification_passed=True, maximum_identity_errors=numerical,
                  input_sha256=hashes,
                  summary_tables={name: name+'_summary.csv' for name in summaries})
    figures(output, all_rows['forces'], all_rows['budgets'])
    (output/'summary.json').write_text(json.dumps(finite_json(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), starts=333, functions=23, verification_passed=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == '__main__':
    main()
