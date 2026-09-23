"""Locked held-out forecast assessment; writes evidence arrays/CSV/JSON only.

Specification started 2026-09-23T01:03:41Z, before inspecting held-out outcomes.
Run from the repository root with PYTHONPATH=. and the local Python environment.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.expD34_readout_race import effective_feedback_analysis as analysis
from experiments.expD34_readout_race.effective_feedback import load_inputs
from experiments.expD34_readout_race import effective_feedback_holdout as heldout
from experiments.expD34_readout_race import transport

ROOT = Path('results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback')
HORIZONS = (1000, 10000, 50000, 200000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(case):
    return case['target'], int(case['seed']), int(case['start'])


def stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(count=len(values), finite_count=len(finite),
                unresolved_or_nonfinite_count=int(len(values)-len(finite)),
                **{name: float(func(finite)) if len(finite) else None
                   for name, func in [('mean', np.mean), ('median', np.median),
                                      ('min', np.min), ('max', np.max)]})


def aggregate(rows, group_keys, metric_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)
    output = []
    for group, members in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(group_keys, group))
        record['case_count'] = len(members)
        record['metrics'] = {m: stats([r.get(m, np.nan) for r in members])
                             for m in metric_keys}
        output.append(record)
    return output


def summaries(rows, metrics, extra=('arm', 'model')):
    # Each family contains two functions. Stages and seeds are repeated views,
    # not independent functions; retain stage-specific and pooled summaries.
    function = aggregate(rows, ('target', 'family', 'start', 'offset')+extra, metrics)
    pooled = aggregate(rows, ('target', 'family', 'offset')+extra, metrics)
    macro = []
    for statistic in ('mean', 'median', 'min', 'max'):
        records = [{**{k: v for k, v in r.items() if k != 'metrics'},
                    **{m: r['metrics'][m][statistic] for m in metrics}}
                   for r in pooled]
        for record in records:
            for metric in metrics:
                if record[metric] is None:
                    record[metric] = np.nan
        grouped = aggregate(records, ('family', 'offset')+extra, metrics)
        for record in grouped:
            record['per_function_statistic'] = statistic
        macro.extend(grouped)
    return dict(function_by_stage=function, function_pooled_stages=pooled,
                family_macro=macro)


def score(prediction, actual):
    error = float(np.linalg.norm(prediction-actual))
    denominator = float(np.linalg.norm(actual))
    predicted_norm = float(np.linalg.norm(prediction))
    relative = error/denominator if denominator > 0 else np.nan
    return dict(absolute_error=error, actual_change_norm=denominator,
                predicted_change_norm=predicted_norm, relative_error=relative,
                skill=analysis._skill(relative),
                alignment=analysis._alignment(prediction, actual),
                zero_actual_change=denominator == 0,
                zero_predicted_change=predicted_norm == 0)


def regime_metrics(snapshot, i):
    """Endpoint force ratios and accumulated signed-vector ratios, not bounds."""
    result = {f'actual_endpoint_{k[7:]}': float(value[i])
              for k, value in snapshot.items()
              if k.startswith('metric_') and value.ndim == 1}
    force = snapshot['metric_effective_a'][i]
    tracking = snapshot['metric_tracking_a'][i]
    omitted = snapshot['metric_omitted_a'][i]
    denominator = np.linalg.norm(force)
    travel_denominator = np.linalg.norm(snapshot['effective'][i])
    for name, value in [('tracking', tracking), ('omitted', omitted),
                        ('remainder', tracking+omitted)]:
        result[f'actual_endpoint_{name}_to_effective_norm_ratio'] = analysis._ratio(
            np.linalg.norm(value), denominator)
    for name, value in [('tracking', snapshot['tracking'][i]),
                        ('omitted', snapshot['omitted'][i]),
                        ('remainder', snapshot['tracking'][i]+snapshot['omitted'][i])]:
        result[f'actual_integrated_signed_{name}_to_effective_vector_norm_ratio'] = analysis._ratio(
            np.linalg.norm(value), travel_denominator)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'raw/heldout')
    parser.add_argument('--predictions', type=Path, default=ROOT/'predictions/heldout_all')
    parser.add_argument('--inputs', type=Path, default=ROOT/'inputs/heldout_all.npz')
    parser.add_argument('--out', type=Path, default=ROOT/'analysis/heldout')
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True, exist_ok=True)
    parameters, x, y, cases = load_inputs(args.inputs)
    qh = np.asarray(transport.basis(x, 65))[:, 2:]
    forks = {}
    for case, p, target_y in zip(cases, parameters, y):
        a, b, c = p[:-1].reshape(3, -1)
        u = x[:, None]*a+b
        h = np.tanh(u)
        z = np.exp(-2*np.abs(u))
        derivative = 4*z/(1+z)**2
        jacobian = np.column_stack((x[:, None]*derivative*c,
                                   derivative*c, h, np.ones(len(x))))
        assert key(case) not in forks
        forks[key(case)] = (p, qh.T@jacobian/len(x),
                            qh.T@(h@c+p[-1]-target_y)/len(x),
                            float(np.mean(target_y**2)))
    analysis.HEADLINES = HORIZONS
    rows, contrasts = analysis.collect(args.raw, [args.predictions])
    manifests, snapshots, predictions, initial_diagnostics = {}, {}, {}, {}
    modal_rows, modal_vectors, coefficient_rows = [], [], []
    e0_defects = []
    for row in rows:
        row['family'] = heldout.FAMILIES[row['target']]
        row['additional_reference_updates'] = row['offset']
        row['total_reference_updates'] = int(row['start'])+int(row['offset'])
        row['total_actual_updates'] = int(row['start'])+int(row['updates'])
        run = Path(row['run'])
        if run not in manifests:
            manifest = json.loads((run/'manifest.json').read_text())
            manifests[run] = (manifest, {key(c): i for i, c in enumerate(manifest['cases'])})
            with np.load(run/'snapshots'/'000000000.npz') as data:
                initial_diagnostics[run] = {k: data[k].copy() for k in data.files
                    if k in ('tracking', 'omitted', 'effective') or k.startswith('metric_')}
        manifest, indices = manifests[run]
        i = indices[key(row)]
        snapshot_key = (run, row['offset'])
        if snapshot_key not in snapshots:
            with np.load(run/'snapshots'/f'{row["offset"]:09d}.npz') as data:
                snapshots[snapshot_key] = {k: data[k].copy() for k in data.files
                    if k in ('p', 'tracking', 'omitted', 'effective') or k.startswith('metric_')}
        snapshot = snapshots[snapshot_key]
        p0, jh, e0, target_energy = forks[key(row)]
        row['actual_train_relative_mse'] = float(snapshot['metric_relative_mse'][i])
        row['actual_train_loss'] = .5*target_energy*row['actual_train_relative_mse']
        row.update(regime_metrics(snapshot, i))
        row.update({k.replace('actual_endpoint_', 'actual_initial_'): v
                    for k, v in regime_metrics(initial_diagnostics[run], i).items()
                    if k.startswith('actual_endpoint_')})
        if not row['failed']:
            np.testing.assert_allclose(row['updates']*row['eta'], row['time'], rtol=2e-15, atol=0)
        if not row.get('forecast_supported', False):
            continue
        prediction_file = row['prediction_file']
        if prediction_file not in predictions:
            with np.load(prediction_file) as data:
                predictions[prediction_file] = {k: data[k].copy() for k in data.files
                    if k in ('steps', 'p0') or k.endswith(('_p', '_eH'))}
        prediction = predictions[prediction_file]
        np.testing.assert_array_equal(p0, prediction['p0'])
        step = int(np.flatnonzero(prediction['steps'] == row['updates'])[0])
        zero = np.flatnonzero(prediction['steps'] == 0)
        if len(zero):
            e0_defects.append(float(np.linalg.norm(e0-prediction['effective_pure_eH'][zero[0]])))
        if row['model'] == 'affine':
            ehat = e0+jh@(prediction[row['arm']+'_p'][step]-p0)
        else:
            ehat = prediction[row['model']+'_eH'][step]
        actual = snapshot['metric_eH'][i]
        identifier = len(modal_rows)
        base = {k: row[k] for k in ('target', 'family', 'seed', 'start', 'arm', 'model',
            'offset', 'time', 'additional_reference_updates',
            'total_reference_updates', 'total_actual_updates')}
        result = dict(base, modal_row=identifier, **score(ehat-e0, actual-e0))
        result['residual_level_absolute_error'] = float(np.linalg.norm(ehat-actual))
        nonzero_actual = (e0 != 0)&(actual != 0)
        nonzero_predicted = (e0 != 0)&(ehat != 0)
        actual_crossings = nonzero_actual&(np.sign(e0) != np.sign(actual))
        predicted_crossings = nonzero_predicted&(np.sign(e0) != np.sign(ehat))
        result.update(raw_actual_crossings=int(actual_crossings.sum()),
                      raw_predicted_crossings=int(predicted_crossings.sum()),
                      raw_crossing_disagreements=int(np.sum(actual_crossings != predicted_crossings)),
                      exact_zero_endpoint_count=int(np.sum((e0 == 0)|(actual == 0)|(ehat == 0))))
        modal_rows.append(result)
        modal_vectors.append(np.stack((e0, actual, ehat)))
        for k in range(len(e0)):
            coefficient_rows.append(dict(base, modal_row=identifier, degree=k+2,
                initial=float(e0[k]), actual=float(actual[k]), predicted=float(ehat[k]),
                actual_change=float(actual[k]-e0[k]), predicted_change=float(ehat[k]-e0[k]),
                absolute_error=float(abs(ehat[k]-actual[k])),
                relative_change_error=analysis._ratio(abs(ehat[k]-actual[k]), abs(actual[k]-e0[k])),
                raw_actual_crossing=bool(actual_crossings[k]),
                raw_predicted_crossing=bool(predicted_crossings[k]),
                sign_comparison_resolved=bool(nonzero_actual[k] and nonzero_predicted[k])))
    for row in contrasts:
        row['family'] = heldout.FAMILIES[row['target']]
        row['additional_reference_updates'] = row['offset']
        row['total_reference_updates'] = int(row['start'])+int(row['offset'])
    for name, records in [('actual_vs_forecast', rows), ('branch_contrasts', contrasts),
                          ('modal_scores', modal_rows), ('modal_coefficients', coefficient_rows)]:
        analysis.write_csv(args.out/f'{name}.csv', records)
    np.savez_compressed(args.out/'modal_vectors.npz',
                        vectors=np.asarray(modal_vectors),
                        axes=np.array(['row', 'initial_actual_predicted', 'degree2_through65']))
    slope_metrics = sorted({k for r in rows for k in r if k.startswith(
        ('actual_', 'slope_motion_', 'effective_force_', 'current_fraction_', 'ever_fraction_', 'new_ever_fraction_'))})
    contrast_metrics = ['contrast_absolute_error', 'contrast_relative_error', 'contrast_skill',
                        'contrast_alignment', 'actual_mean_gamma_contrast',
                        'predicted_mean_gamma_contrast', 'mean_gamma_contrast_sign_agrees']
    modal_metrics = ['absolute_error', 'relative_error', 'skill', 'alignment',
                     'actual_change_norm', 'predicted_change_norm',
                     'raw_actual_crossings', 'raw_predicted_crossings', 'raw_crossing_disagreements']
    metadata = dict(specification_started_utc='2026-09-23T01:03:41Z',
        executed_utc=started, script_sha256=digest(__file__), input_sha256=digest(args.inputs),
        prediction_manifest_sha256=digest(args.predictions/'manifest.json'),
        horizons=HORIZONS, expected_forks=60, input_forks=len(forks),
        update_convention='Primary horizons are additional updates after starts100000/400000/600000; '
            'total_reference_updates=start+offset. At eta0.002 these equal total actual updates. '
            'Million-step forecasts are outside this primary assessment.',
        expected_arm_horizon_rows=60*3*len(HORIZONS),
        observed_arm_horizon_rows=sum(r['model'] == 'affine' for r in rows),
        failed_arm_horizon_rows=sum(r['model'] == 'affine' and bool(r['failed']) for r in rows),
        unsupported_forecast_rows=sum(not r.get('forecast_supported', False) for r in rows),
        initial_modal_reconstruction_defects=stats(e0_defects),
        sign_count_caveat='Raw nonzero coefficient signs only; no numerical confidence threshold. '
            'Coefficient magnitudes are retained; sign counts are not numerically certified.',
        denominator_rule='Exact zero actual-change norm gives unresolved relative error and skill; no floor.',
        regime_rule='Endpoint force ratios are sampled diagnostics, not all-step bounds. '
            'Integrated ratios divide norms of accumulated signed per-neuron travel vectors; '
            'they may contain temporal cancellation and are not integrals of force norms.',
        family_rule='Macro summaries of per-function statistics; stages/seeds are not independent functions.',
        slope=summaries(rows, slope_metrics),
        contrasts=summaries(contrasts, contrast_metrics, extra=('arm',)),
        modal=summaries(modal_rows, modal_metrics))
    (args.out/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: metadata[k] for k in ('input_forks', 'observed_arm_horizon_rows',
                                             'unsupported_forecast_rows')}))


if __name__ == '__main__':
    main()
