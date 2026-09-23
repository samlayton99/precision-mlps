"""Summarize issued persistence experiments into CSV/JSON evidence only.

Run under the campaign CPU allocation. The contrast floor is a declared
reporting convention, not a rigorous bound on numerical error. No thresholds
are selected from outcomes, and failed/unresolved cases remain in artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    def parse(value):
        if value in ('True', 'False'):
            return value == 'True'
        if value == '':
            return None
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (TypeError, ValueError):
            return value
    with Path(path).open() as stream:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def divide(a, b):
    return a/b if a is not None and b is not None and b != 0 else None


def statistics(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return dict(count=len(values), minimum=min(values), median=float(np.median(values)), maximum=max(values)) if values else dict(count=0)


def normalized_diagnostics(row):
    result = dict(row)
    for term in ('D', 'C', 'C_generated', 'C_target', 'loaded'):
        result[term+'_over_q_squared'] = divide(row.get(term), row.get('q_squared'))
    return result


def row_key(row):
    return row['source_index'], row['arm']


def summarize(root, output, horizon=20000, absolute=1e-14, relative=1e-6):
    output.mkdir(parents=True, exist_ok=False)
    sources, prediction_map, audits = {}, {}, []
    for path in sorted(root.rglob('fork_diagnostics.csv')):
        manifest_path = path.parent/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        kind = manifest.get('kind', 'feedback')
        if kind not in ('feedback', 'physical'):
            continue
        records = read_csv(path)
        prediction_map[digest(manifest_path)] = (path.parent.name, kind, {row_key(r): r for r in records})
        sources[str(path.relative_to(root))] = digest(path)
        sources[str(manifest_path.relative_to(root))] = digest(manifest_path)
    for path in sorted(root.rglob('diagnostics.csv')):
        if not path.parent.name.startswith('audit_'):
            continue
        sources[str(path.relative_to(root))] = digest(path)
        audits.extend(dict(normalized_diagnostics(r), audit_panel=path.parent.name) for r in read_csv(path))

    states, contrasts, missing = [], [], []
    for path in sorted(root.rglob('states.csv')):
        manifest_path = path.parent/'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        match = prediction_map.get(manifest.get('predictions'))
        if match is None:
            missing.append(str(path.relative_to(root)))
            continue
        panel, kind, forks = match
        sources[str(path.relative_to(root))] = digest(path)
        sources[str(manifest_path.relative_to(root))] = digest(manifest_path)
        own = [r for r in read_csv(path) if r['horizon'] == horizon]
        by_key = {row_key(r): r for r in own}
        for row in own:
            initial = forks[row_key(row)]
            enriched = dict(normalized_diagnostics(initial), **row,
                            panel=panel, kind=kind, q0=initial['q'],
                            k0=initial.get('k_actual_arm', initial.get('k')))
            enriched['q_actual_over_q0'] = divide(row['q_actual'], initial['q'])
            enriched['q_forecast_relative_error'] = divide(abs(row['q_predicted']-row['q_actual']), row['q_actual']) if row['q_predicted'] is not None and row['q_actual'] is not None else None
            enriched['slope_scalar_beats_constant'] = (row['amplification_error'] < row['constant_error']) if row['amplification_error'] is not None and row['constant_error'] is not None else None
            states.append(enriched)
        contrast_path = path.parent/'contrasts.csv'
        if not contrast_path.exists():
            continue
        sources[str(contrast_path.relative_to(root))] = digest(contrast_path)
        for row in read_csv(contrast_path):
            if row['horizon'] != horizon:
                continue
            own_state = by_key[row_key(row)]
            baseline = by_key[(row['source_index'], 'natural')]
            width = row['width']
            motion_values = (own_state['motion_norm'], baseline['motion_norm'], own_state['mean_lambda_change'], baseline['mean_lambda_change'])
            finite_motion = all(v is not None for v in motion_values)
            rms_scale = row['h']*max(own_state['motion_norm'], baseline['motion_norm'])/np.sqrt(width) if finite_motion else 0.
            vector_floor = np.sqrt(width)*(absolute+relative*rms_scale)
            mean_floor = absolute+relative*max(abs(own_state['mean_lambda_change']), abs(baseline['mean_lambda_change'])) if finite_motion else absolute
            valid = not row['failed'] and finite_motion and all(row.get(k) is not None for k in ('actual_norm', 'actual_mean'))
            vector_resolved = valid and row['actual_norm'] > vector_floor
            mean_resolved = valid and abs(row['actual_mean']) > mean_floor
            forecast_resolved = valid and row['predicted_mean'] is not None and abs(row['predicted_mean']) > mean_floor
            sign_correct = bool(np.sign(row['actual_mean']) == np.sign(row['predicted_mean'])) if mean_resolved and forecast_resolved else False
            contrasts.append(dict(row, panel=panel, kind=kind,
                vector_resolution_floor=float(vector_floor), mean_resolution_floor=float(mean_floor),
                observed_vector_resolved=bool(vector_resolved), observed_mean_resolved=bool(mean_resolved),
                predicted_mean_resolved=bool(forecast_resolved), resolved_sign_correct=sign_correct,
                resolved_skill=1-row['error_norm']/row['actual_norm'] if vector_resolved and row['error_norm'] is not None else None))

    group_rows, contrast_groups = [], []
    groups = defaultdict(list)
    for row in states:
        groups[(row['kind'], row['panel'], row['cohort'], row['nref'], row['arm'])].append(row)
    metrics = ('q_actual_over_q0', 'k0', 'D_over_q_squared', 'C_over_q_squared', 'loaded_over_q_squared',
               'constant_relative_error', 'amplification_relative_error', 'q_forecast_relative_error',
               'mean_positive_travel', 'mean_lambda_change')
    for (kind, panel, cohort, nref, arm), rows in groups.items():
        good = [r for r in rows if not r['failed']]
        record = dict(kind=kind, panel=panel, cohort=cohort, nref=nref, arm=arm, cases=len(rows), failed=len(rows)-len(good),
                      scalar_beats_constant=sum(r['slope_scalar_beats_constant'] is True for r in good),
                      initial_positive_k=sum(r['k0'] is not None and r['k0'] > 0 for r in good),
                      finite_window_q_growth=sum(r['q_actual_over_q0'] is not None and r['q_actual_over_q0'] > 1 for r in good))
        for metric in metrics:
            record.update({metric+'_'+key: value for key, value in statistics([r.get(metric) for r in good]).items()})
        group_rows.append(record)
    groups = defaultdict(list)
    for row in contrasts:
        groups[(row['kind'], row['panel'], row['cohort'], row['nref'], row['arm'])].append(row)
    for (kind, panel, cohort, nref, arm), rows in groups.items():
        vector = [r for r in rows if r['observed_vector_resolved']]
        mean = [r for r in rows if r['observed_mean_resolved']]
        correct = sum(r['resolved_sign_correct'] for r in mean)
        record = dict(kind=kind, panel=panel, cohort=cohort, nref=nref, arm=arm, cases=len(rows),
            failed=sum(bool(r['failed']) for r in rows), observed_vector_resolved=len(vector),
            observed_mean_resolved=len(mean), observed_mean_unresolved=sum(not r['failed'] and not r['observed_mean_resolved'] for r in rows),
            predicted_mean_unresolved_among_observed_resolved=sum(not r['predicted_mean_resolved'] for r in mean),
            sign_correct=correct, sign_rate=divide(correct, len(mean)),
            invalid_vector_forecasts=sum(r['resolved_skill'] is None for r in vector),
            positive_skill=sum(r['resolved_skill'] is not None and r['resolved_skill'] > 0 for r in vector))
        record.update({'skill_'+k: v for k, v in statistics([r['resolved_skill'] for r in vector]).items()})
        contrast_groups.append(record)

    natural = [r for r in states if r['kind'] == 'feedback' and r['arm'] == 'natural']
    write_csv(output/'states_20k.csv', states)
    write_csv(output/'natural_20k.csv', natural)
    write_csv(output/'contrasts_20k.csv', contrasts)
    write_csv(output/'checkpoint_audits.csv', audits)
    write_csv(output/'group_summary.csv', group_rows)
    write_csv(output/'contrast_summary.csv', contrast_groups)
    audit_groups = []
    for panel in sorted({r['audit_panel'] for r in audits}):
        records = [r for r in audits if r['audit_panel'] == panel]
        record = dict(panel=panel, cases=len(records))
        for metric in ('k', 'k_pure', 'D_over_q_squared', 'C_over_q_squared', 'loaded_over_q_squared'):
            record.update({metric+'_'+k: v for k, v in statistics([r.get(metric) for r in records]).items()})
        audit_groups.append(record)
    write_csv(output/'audit_summary.csv', audit_groups)
    result = dict(source_sha256=digest(__file__), inputs=sources, horizon=horizon,
        resolution=dict(absolute_normalized_lambda=absolute, relative_to_motion=relative,
                        status='Predeclared reporting convention; not a floating-point certificate',
                        sign_denominator='All resolved observed means; unresolved predicted means count as unsuccessful'),
        counts=dict(states=len(states), natural=len(natural), contrasts=len(contrasts), audit_states=len(audits)),
        unmatched_analysis_files=missing, groups=group_rows, contrasts=contrast_groups, audit_groups=audit_groups,
        audit_statistics={key: statistics([r.get(key) for r in audits]) for key in ('k', 'k_pure', 'D_over_q_squared', 'C_over_q_squared', 'loaded_over_q_squared')})
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--absolute-threshold', type=float, default=1e-14)
    parser.add_argument('--relative-threshold', type=float, default=1e-6)
    args = parser.parse_args()
    summarize(args.root, args.output, absolute=args.absolute_threshold, relative=args.relative_threshold)
