"""Compact CSV/JSON comparison of scalar, tangent and coupled pulse models.

The primary comparison is fixed at amplitude .0025 and 20,000 updates.
Baseline movement is a separate quantity from the central pulse response.
Failed, truncated and missing cases remain explicit in the primary table.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .mechanism_persistence_summary import read_csv, write_csv, statistics, digest


PRIMARY_AMPLITUDE = .0025
PRIMARY_HORIZON = 20000


def panel_name(name, cohort):
    for prefix in ('physical_', 'coupled_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.startswith(cohort+'_'):
        name = name[len(cohort)+1:]
    for suffix in ('_paired', '_analysis'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def normalize(row, panel, model, block, quantity, prefix=''):
    cohort = row['cohort']
    supported = row.get('supported', row.get('forecast_supported', True))
    actual_failed = bool(row.get('actual_failed', row.get('failed', False)))
    surrogate_failed = bool(row.get('surrogate_failed', False))
    return dict(cohort=cohort, panel=panel_name(panel, cohort), target=row['target'], seed=row['seed'],
                width=row.get('width'), nref=row.get('nref'), horizon=row['horizon'],
                amplitude=row.get('amplitude', 0), model=model, block=block, quantity=quantity,
                available=True, supported=bool(supported) and not actual_failed,
                actual_failed=actual_failed, surrogate_failed=surrogate_failed,
                truncated=bool(row.get('truncated', False)),
                actual_norm=row.get(prefix+'actual_norm'), error_norm=row.get(prefix+'error_norm'),
                relative_error=row.get(prefix+'relative_error'),
                alignment=row.get(prefix+'alignment', row.get(prefix+'cosine')),
                halving_response_difference=row.get(prefix+'halving_response_difference'))


def case_key(row):
    return tuple(row[k] for k in ('cohort', 'panel', 'target', 'seed'))


def comparison(root, output):
    output.mkdir(parents=True, exist_ok=False)
    rows, sources, universe = [], {}, {}
    def read(path):
        sources[str(path.relative_to(root))] = digest(path)
        return read_csv(path)
    for path in sorted(root.glob('physical_*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest.get('kind') != 'physical':
            continue
        sources[str(path.relative_to(root))] = digest(path)
        for case in manifest['cases']:
            if float(case.get('amplitude', 0)) == 0:
                metadata = {k: case[k] for k in ('cohort', 'target', 'seed', 'width', 'nref')}
                metadata['panel'] = panel_name(path.parent.name, case['cohort'])
                universe[case_key(metadata)] = metadata
    for path in sorted(root.glob('linear*/responses.csv')):
        for row in read(path):
            rows.append(normalize(row, row['panel'], row['model'], row['block'], 'central_response'))
    for path in sorted(root.glob('physical_*_paired/paired.csv')):
        for row in read(path):
            for block in ('slope', 'readout', 'full'):
                rows.append(normalize(row, path.parent.name, 'scalar_amplification', block, 'central_response', block+'_'))
    for path in sorted(root.glob('physical_*_analysis/states.csv')):
        for row in read(path):
            if row.get('amplitude') != 0:
                continue
            item = normalize(row, path.parent.name, 'scalar_amplification', 'slope', 'baseline_movement')
            item.update(actual_norm=row['motion_norm'], error_norm=row['amplification_error'],
                        relative_error=row['amplification_relative_error'],
                        supported=not row['failed'] and row['amplification_error'] is not None)
            rows.append(item)
    for path in sorted(root.glob('coupled_*/paired.csv')):
        for row in read(path):
            for block in ('slope', 'readout'):
                rows.append(normalize(row, path.parent.name, row['model'], block, 'central_response', block+'_'))
    for path in sorted(root.glob('coupled_*/states.csv')):
        for row in read(path):
            if row.get('amplitude') != 0:
                continue
            for block in ('slope', 'readout'):
                rows.append(normalize(row, path.parent.name, row['model'], block, 'baseline_movement', block+'_'))
    for row in rows:
        row['source_case_known'] = case_key(row) in universe
        if not row['source_case_known']:
            raise ValueError(f'Comparison case absent from issued physical panel: {case_key(row)}')
    primary = [r for r in rows if r['horizon'] == PRIMARY_HORIZON and
               (r['quantity'] == 'baseline_movement' or r['amplitude'] == PRIMARY_AMPLITUDE)]
    expected = [('central_response', model, block) for model in
                ('full_hessian', 'effective_jacobian', 'scalar_amplification') for block in ('slope', 'readout', 'full')]
    expected += [('central_response', model, block) for model in ('anchored_fine', 'anchored_full') for block in ('slope', 'readout')]
    expected += [('baseline_movement', 'scalar_amplification', 'slope')]
    expected += [('baseline_movement', model, block) for model in ('anchored_fine', 'anchored_full') for block in ('slope', 'readout')]
    keys = [(case_key(r), r['quantity'], r['model'], r['block']) for r in primary]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate primary result; supply one version of each comparison')
    existing = set(keys)
    for key, case in universe.items():
        for quantity, model, block in expected:
            if (key, quantity, model, block) not in existing:
                primary.append(dict(case, quantity=quantity, model=model, block=block,
                    horizon=PRIMARY_HORIZON, amplitude=PRIMARY_AMPLITUDE if quantity == 'central_response' else 0,
                    available=False, supported=False, actual_failed=False, surrogate_failed=False,
                    truncated=False, relative_error=None, error_norm=None, actual_norm=None,
                    missing_reason='No source row; not counted as a successful prediction'))
    groups = defaultdict(list)
    for row in primary:
        groups[(row['cohort'], row['panel'], row['quantity'], row['model'], row['block'])].append(row)
        groups[('all', 'all', row['quantity'], row['model'], row['block'])].append(row)
    aggregates = []
    for key, records in sorted(groups.items()):
        aggregate = dict(zip(('cohort', 'panel', 'quantity', 'model', 'block'), key))
        good = [r for r in records if r['available'] and r['supported'] and r['relative_error'] is not None]
        aggregate.update(cases=len(records), available=sum(r['available'] for r in records),
            supported=sum(r['supported'] for r in records), actual_failed=sum(r['actual_failed'] for r in records),
            surrogate_failed=sum(r['surrogate_failed'] for r in records), truncated=sum(r['truncated'] for r in records),
            finite_relative_errors=len(good), missing=sum(not r['available'] for r in records),
            error_below_one=sum(r['relative_error'] < 1 for r in good))
        aggregate.update({'relative_error_'+k: v for k, v in statistics([r['relative_error'] for r in good]).items()})
        aggregates.append(aggregate)
    write_csv(output/'all_models_all_amplitudes_horizons.csv', rows)
    write_csv(output/'primary_per_target.csv', primary)
    write_csv(output/'primary_summary.csv', aggregates)
    slope_table = {key: dict(case) for key, case in universe.items()}
    for row in primary:
        if row['quantity'] == 'central_response' and row['block'] == 'slope':
            for metric in ('relative_error', 'supported', 'available', 'surrogate_failed', 'truncated'):
                slope_table[case_key(row)][row['model']+'_'+metric] = row[metric]
    write_csv(output/'primary_slope_response.csv', list(slope_table.values()))
    manifest = dict(source_sha256=digest(__file__), inputs=sources,
        primary=dict(horizon=PRIMARY_HORIZON, amplitude=PRIMARY_AMPLITUDE),
        cases=len(universe), all_rows=len(rows), primary_rows=len(primary),
        groups=aggregates, models=['scalar_amplification', 'full_hessian', 'effective_jacobian', 'anchored_fine', 'anchored_full'],
        quantities=dict(central_response='Derivative of own-initial displacement under paired physical pulses',
                        baseline_movement='Unpulsed ordinary-GD displacement'),
        availability='Frozen tangent models do not predict baseline movement; coupled CSVs have no full-block scores. Neither absence is treated as a model failure.',
        failure_policy='All issued baseline families retained. Summary medians use supported finite errors and always report coverage/failures alongside them.',
        retrospective=True, no_future_fit=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(dict(cases=len(universe), all_rows=len(rows), primary_rows=len(primary))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison(args.root, args.output)
