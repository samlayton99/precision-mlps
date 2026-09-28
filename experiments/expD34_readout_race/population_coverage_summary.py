"""Compact numerical tables and figures for the retrospective population audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    try:
        value = float(row.get(key, ''))
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def stats(rows, key):
    values = [v for row in rows if (v := number(row, key)) is not None]
    if not values:
        return dict(n=0, min=None, median=None, max=None)
    return dict(n=len(values), min=float(min(values)), median=float(np.median(values)), max=float(max(values)))


def flag(row, key):
    return row.get(key, '').lower() == 'true'


def groupby(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return groups


STRUCTURAL = ('negative_product_fraction', 'negative_product_mass',
              'zero_slope_count',
              'readout_dominance_violation_fraction', 'dominance_defect_rms',
              'cone_distance_rms', 'cone_distance_max', 'beta_rms', 'beta_max',
              'target_odd_error', 'target_odd_relative', 'cubic_cancellation_relative',
              'm3_times_p', 'generated_margin', 'initial_moment_margin',
              'coarse_kappa', 'tracking_z')
REDUCED = ('relative_error', 'error_norm', 'cosine', 'positive_relative_error',
           'tracking_relative', 'exact_norm', 'exact_positive_norm')


def structural_summary(rows):
    return dict(states=len(rows), targets=sorted({r.get('target', '') for r in rows}),
                cone_exact=sum(flag(r, 'cone_exact') for r in rows),
                exact_bias_zero=sum(flag(r, 'exact_bias_zero') for r in rows),
                cone_and_bias_exact=sum(flag(r, 'cone_exact') and flag(r, 'exact_bias_zero') for r in rows),
                force_failures=sum(bool(r.get('force_status')) for r in rows),
                metrics={key: stats(rows, key) for key in STRUCTURAL})


def case_key(row):
    # Structural files use expanded arm index; source_index identifies the same
    # original state across structural and reduced audits.
    return row['panel'], row['source_index']


def regime(row):
    return 'late' if row['panel'].endswith('_late') else 'width'+row['width']


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    static = read(args.structural/'static.csv')
    paths = read(args.structural/'paths.csv')
    classes = read(args.structural/'classes.csv')
    reduced = read(args.reduced/'reduced_fields.csv')
    structural_manifest = json.loads((args.structural/'manifest.json').read_text())
    reduced_manifest = json.loads((args.reduced/'manifest.json').read_text())
    path_cases = defaultdict(list)
    group_cases = defaultdict(list)
    for row in paths:
        path_cases[case_key(row)].append(row)
    for row in classes:
        group_cases[case_key(row)].append(row)
    mechanism, endpoints = [], []
    for key, trajectory in sorted(path_cases.items()):
        trajectory.sort(key=lambda r: int(r['horizon']))
        first, last = trajectory[0], trajectory[-1]
        endpoint = {k: last.get(k) for k in ('panel', 'source_index', 'target', 'seed', 'width', 'horizon', 'actual_updates', 'failed')}
        for label in ('median', 'p90', 'p99', 'max'):
            initial, final = number(first, 'lambda_'+label), number(last, 'lambda_'+label)
            endpoint['lambda_'+label+'_initial'] = initial
            endpoint['lambda_'+label+'_final'] = final
            endpoint['lambda_'+label+'_change'] = final-initial if initial is not None and final is not None else None
        endpoints.append(endpoint)
        final_groups = [r for r in group_cases[key] if r['horizon'] == last['horizon']]
        total = sum(number(r, 'all_step_positive_sum_over_width') or 0. for r in final_groups)
        row = dict(endpoint)
        row['all_step_positive_mean'] = number(last, 'all_step_positive_mean')
        row['group_positive_reconstruction_error'] = total-row['all_step_positive_mean'] if row['all_step_positive_mean'] is not None else None
        for group in ('negative', 'dominance', 'compliant'):
            initial_rows = [r for r in group_cases[key] if r['group'] == group and r['horizon'] == first['horizon']]
            final_rows = [r for r in final_groups if r['group'] == group]
            if not final_rows:
                continue
            final = final_rows[0]
            row[group+'_fraction'] = number(final, 'fraction')
            positive = number(final, 'all_step_positive_sum_over_width')
            row[group+'_positive_share'] = positive/total if positive is not None and total > 0 else None
            for name in ('positive', 'negative', 'signed_effective', 'signed_tracking', 'crossing'):
                row[group+'_'+name] = number(final, 'all_step_'+name+'_sum_over_width')
            row[group+'_net'] = number(final, 'endpoint_lambda_sum_over_width')
            for phase, samples in (('initial', initial_rows), ('final', final_rows)):
                if samples:
                    for field in samples[0]:
                        if field.startswith(('compensation_', 'direct_raw_', 'ell', 'moment_')) or field == 'output_sign':
                            row[group+'_'+phase+'_'+field] = number(samples[0], field)
        for metric in ('fraction', 'positive_share', 'positive', 'negative', 'signed_effective', 'signed_tracking', 'crossing', 'net'):
            values = [row.get(group+'_'+metric) for group in ('negative', 'dominance')]
            row['unionviolations_'+metric] = sum(values) if all(v is not None for v in values) else None
        for phase in ('initial', 'final'):
            for receiver in ('all', 'upper'):
                suffix = receiver+'_signed_sum_over_width'
                bad = [row.get(group+'_'+phase+'_compensation_'+suffix) for group in ('negative', 'dominance')]
                components = [row.get(group+'_'+phase+'_'+component+'_'+suffix)
                              for group in ('negative', 'dominance', 'compliant')
                              for component in ('compensation', 'direct_raw')]
                bad_sum = sum(bad) if all(v is not None for v in bad) else None
                net = sum(components) if all(v is not None for v in components) else None
                prefix = phase+'_'+receiver+'_'
                row[prefix+'bad_source_compensation_signed'] = bad_sum
                row[prefix+'net_fine_signed_from_sources'] = net
                row[prefix+'bad_compensation_over_net_fine_signed'] = bad_sum/net if bad_sum is not None and net not in (None, 0.) else None
                row[prefix+'signed_aggregate_component_cancellation_ratio'] = sum(abs(v) for v in components)/abs(net) if net not in (None, 0.) else None
        mechanism.append(row)
    write_csv(args.output/'case_mechanisms.csv', mechanism)
    write_csv(args.output/'case_scale_changes.csv', endpoints)
    regular = [r for r in reduced if not r['panel'].endswith('_late')]
    late = [r for r in reduced if r['panel'].endswith('_late')]
    reduced_groups = groupby(regular, lambda r: (r['width'], r['degree'], r['subset']))
    late_groups = groupby(late, lambda r: (r['panel'], r['source_index'], r['degree'], r['subset']))
    per_case = []
    for _, rows in sorted(groupby(reduced, lambda r: (r['panel'], r['source_index'], r['degree'], r['subset'])).items()):
        first = rows[0]
        rec = {k: first[k] for k in ('panel', 'source_index', 'target', 'seed', 'width', 'degree', 'subset')}
        rec['horizons'] = sorted({int(r['horizon']) for r in rows})
        rec['metrics'] = {metric: stats(rows, metric) for metric in REDUCED}
        per_case.append(rec)
    source_groups = groupby(static, lambda r: r['source'])
    summary = dict(
        structural=structural_summary(static),
        structural_by_source={k: structural_summary(v) for k, v in source_groups.items()},
        structural_by_width={k: structural_summary(v) for k, v in groupby(static, lambda r: r['width']).items()},
        structural_by_target={k: structural_summary(v) for k, v in groupby(static, lambda r: r.get('target', '')).items()},
        natural_cases=len(path_cases), natural_path_states=len(paths),
        natural_case_cohorts=dict(Counter(rows[0].get('cohort', '') for rows in path_cases.values())),
        path_failed_states=sum(flag(r, 'failed') or r.get('status') == 'failed' for r in paths),
        path_force_failures=sum(bool(r.get('force_status')) for r in paths),
        counter_checks={key: stats(paths, key) for key in ('all_step_identity_max_error', 'all_step_endpoint_identity_max_error')},
        group_counter_check=stats(mechanism, 'group_positive_reconstruction_error'),
        compensation_reconstruction=stats(classes, 'compensation_reconstruction_error'),
        scale_changes={label: stats(endpoints, 'lambda_'+label+'_change') for label in ('median', 'p90', 'p99', 'max')},
        group_travel={g: {key: stats(mechanism, g+'_'+key) for key in ('fraction', 'positive_share', 'signed_effective', 'signed_tracking', 'net')} for g in ('negative', 'dominance', 'compliant')},
        group_travel_by_regime={name: {g: {key: stats(rows, g+'_'+key) for key in ('fraction', 'positive_share', 'signed_effective', 'signed_tracking', 'net')} for g in ('negative', 'dominance', 'compliant', 'unionviolations')} for name, rows in groupby(mechanism, regime).items()},
        group_moment_M3_by_regime={name: {g: {phase: stats(rows, g+'_'+phase+'_moment_M3') for phase in ('initial', 'final')} for g in ('negative', 'dominance', 'compliant')} for name, rows in groupby(mechanism, regime).items()},
        compensation_by_regime={name: {phase+'_'+receiver+'_'+metric: stats(rows, phase+'_'+receiver+'_'+metric) for phase in ('initial', 'final') for receiver in ('all', 'upper') for metric in ('bad_source_compensation_signed', 'net_fine_signed_from_sources', 'bad_compensation_over_net_fine_signed', 'signed_aggregate_component_cancellation_ratio')} for name, rows in groupby(mechanism, regime).items()},
        scale_changes_by_regime={name: {label: stats(rows, 'lambda_'+label+'_change') for label in ('median', 'p90', 'p99', 'max')} for name, rows in groupby(endpoints, regime).items()},
        reduced_by_width={k: {metric: stats(v, metric) for metric in REDUCED} for k, v in reduced_groups.items()},
        reduced_late_cases={k: {metric: stats(v, metric) for metric in REDUCED} for k, v in late_groups.items()},
        reduced_per_case=per_case,
        reduced_failed_states=sum(flag(r, 'failed') for r in reduced),
        reduced_nonfinite_rows=sum(not flag(r, 'finite_fields') for r in reduced),
        missing_structural_inputs=structural_manifest.get('missing_inputs', []),
        missing_reduced_snapshots=reduced_manifest.get('missing', []),
        structural_duplicates=structural_manifest.get('duplicates', []),
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [args.structural/'static.csv', args.structural/'paths.csv', args.structural/'classes.csv',
                  args.structural/'manifest.json', args.reduced/'reduced_fields.csv', args.reduced/'manifest.json', Path(__file__)]})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))

    targets = groupby(static, lambda r: r.get('target', ''))
    names = sorted(targets)
    fig, ax = plt.subplots(figsize=(10, max(5, .32*len(names))))
    y = np.arange(len(names))
    for offset, metric, label in ((-.15, 'negative_product_fraction', 'Negative product'), (.15, 'readout_dominance_violation_fraction', 'Readout dominance violation (includes negative)')):
        medians = [stats(targets[n], metric)['median'] for n in names]
        ax.barh(y+offset, medians, height=.28, label=label)
    ax.set_yticks(y, [f'{n} (n={len(targets[n])})' for n in names]); ax.set_xlim(0, 1)
    ax.set_xlabel('Median particle fraction across available states')
    ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(); fig.savefig(args.output/'structural_by_target.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, group in zip(axes, ('negative', 'dominance')):
        pairs = [(number(r, group+'_fraction'), number(r, group+'_positive_share')) for r in mechanism]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if pairs:
            ax.scatter(*zip(*pairs), s=24, alpha=.75)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--'); ax.set_title(group)
        ax.set_xlabel('Initial fixed-group population fraction'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Share of all-step outward travel')
    fig.tight_layout(); fig.savefig(args.output/'fixed_group_travel.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    widths = sorted({int(r['width']) for r in regular})
    late_keys = sorted({(r['panel'], r['source_index'], r['target']) for r in late})
    for degree, color in (('3', 'tab:blue'), ('5', 'tab:orange')):
        for subset, marker, shift in (('all', 'o', -.06), ('top10pct_abs_slope', '^', .06)):
            label = f'p{degree}, {subset}'
            for ax, groups in ((axes[0], [[r for r in regular if int(r['width']) == w] for w in widths]),
                               (axes[1], [[r for r in late if (r['panel'], r['source_index'], r['target']) == key] for key in late_keys])):
                for i, group_rows in enumerate(groups):
                    values = [number(r, 'relative_error') for r in group_rows if r['degree'] == degree and r['subset'] == subset]
                    values = [v for v in values if v is not None and v > 0]
                    if values:
                        position = i+shift+(-.12 if degree == '3' else .12)
                        ax.plot([position, position], [min(values), max(values)], color=color, alpha=.4)
                        ax.scatter(position, np.median(values), marker=marker, color=color, label=label if i == 0 else None)
    axes[0].set_xticks(range(len(widths)), widths); axes[0].set_xlabel('Physical width (late states excluded)')
    axes[1].set_xticks(range(len(late_keys)), [f'{key[0].split("_")[1]}\n{key[2]} #{key[1]}' for key in late_keys], fontsize=8)
    for ax in axes:
        ax.set_yscale('log'); ax.set_ylabel('Relative slope-vector error'); ax.axhline(1, color='gray', linestyle='--')
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.output/'polynomial_error.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(static_states=len(static), natural_cases=len(path_cases), reduced_rows=len(reduced))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structural', type=Path, required=True)
    parser.add_argument('--reduced', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
