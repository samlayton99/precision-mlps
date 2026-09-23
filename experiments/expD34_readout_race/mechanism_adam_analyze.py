"""CPU evidence tables for all Adam arms; no report generation or refitting."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import mechanism_adam as ma


def skill(actual, prediction):
    denominator = float(np.sum(actual**2))
    return None if denominator == 0 else 1-float(np.sum((actual-prediction)**2))/denominator


def ratio(a, b):
    return None if b == 0 else float(a/b)


def correlation(a, b):
    a, b = a-a.mean(), b-b.mean()
    return ratio(a@b, np.linalg.norm(a)*np.linalg.norm(b))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    pack = dict(np.load(args.inputs)); cases = json.loads(str(pack['cases']))
    forecasts = np.load(args.predictions/'forecasts.npz')
    phases = np.load(args.predictions/'driver_phases.npz')
    manifest = json.loads((args.predictions/'manifest.json').read_text())
    if manifest['input_sha256'] != ma.digest(args.inputs):
        raise ValueError('Prediction/input mismatch')
    width = (pack['p'].shape[-1]-1)//3
    baseline = {c['target']: i for i, c in enumerate(cases)
                if c['alpha_m'] == c['alpha_v'] == 1}
    endpoints, contrasts, phase_rows, dense_rows = [], [], [], []
    paths = sorted((args.run/'snapshots').glob('*.npz'))
    for path in paths:
        horizon = int(path.stem)
        if not horizon:
            continue
        state = np.load(path)
        if not np.isfinite(state['p']).all() or state['unresolved'].any():
            raise ValueError(f'Invalid state {path}')
        lam = ma.H*abs(state['p'][:, :width])
        lam0 = ma.H*abs(pack['p'][:, :width])
        for i, case in enumerate(cases):
            common = {k: case[k] for k in ('target', 'seed', 'alpha_m', 'alpha_v')}
            common.update(additional_updates=horizon, total_updates=600000+horizon)
            raw = state['force_activity'][i]; processed = state['step_activity'][i]
            endpoints.append(dict(common, mean_lambda=float(lam[i].mean()),
                mean_lambda_change=float((lam[i]-lam0[i]).mean()),
                mean_lambda_rate=float((lam[i]-lam0[i]).mean()/horizon),
                max_lambda=float(lam[i].max()), positive_travel=float(state['positive'][i].mean()),
                negative_travel=float(state['negative'][i].mean()),
                effective_signed=float(state['signed'][i, 0].mean()),
                tracking_signed=float(state['signed'][i, 1].mean()),
                crossing=float(state['crossing'][i].mean()),
                raw_tracking_share=ratio(raw[1], raw.sum()),
                step_tracking_share=ratio(processed[1], processed.sum()),
                moment_identity_max=float(state['identity_max'][i, 0]),
                step_identity_max=float(state['identity_max'][i, 1])))
            b = baseline[case['target']]
            if i == b:
                continue
            actual = lam[i]-lam[b]
            slope_actual = ma.H*(state['p'][i, :width]-state['p'][b, :width])
            for model in ('frozen', 'two_phase'):
                key = f'{model}_p_{horizon}'
                if key not in forecasts:
                    continue
                predicted = ma.H*(abs(forecasts[key][i, :width])-abs(forecasts[key][b, :width]))
                slope_predicted = ma.H*(forecasts[key][i, :width]-forecasts[key][b, :width])
                contrasts.append(dict(common, model=model, actual_mean_effect=float(actual.mean()),
                    predicted_mean_effect=float(predicted.mean()),
                    actual_mean_rate_effect=float(actual.mean()/horizon),
                    predicted_mean_rate_effect=float(predicted.mean()/horizon),
                    actual_vector_norm=float(np.linalg.norm(actual)),
                    vector_error_norm=float(np.linalg.norm(predicted-actual)),
                    lambda_vector_skill=skill(actual, predicted),
                    signed_slope_vector_skill=skill(slope_actual, slope_predicted),
                    mean_sign_match=bool(np.sign(actual.mean()) == np.sign(predicted.mean()))))
    # These are local forecast drivers, not observed future covariance.
    for i, case in enumerate(cases):
        f0, q0 = phases['phase_zero'][i, :2, :width]
        f1, q1 = phases['phase_one'][i, :2, :width]
        f, q = np.stack((f0, f1)), np.stack((q0, q1))
        av = case['alpha_v']
        phase_rows.append(dict(target=case['target'], seed=case['seed'],
            alpha_m=case['alpha_m'], alpha_v=av,
            effective_phase_cosine=ratio(f0@f1, np.linalg.norm(f0)*np.linalg.norm(f1)),
            tracking_phase_cosine=ratio(q0@q1, np.linalg.norm(q0)*np.linalg.norm(q1)),
            mean_F_squared=float(np.mean(f*f)), mean_Q_squared=float(np.mean(q*q)),
            variance_cross_term=float(2*av*np.mean(f*q)),
            attenuated_variance_driver=float(np.mean((f+av*q)**2)),
            baseline_variance_driver=float(np.mean((f+q)**2))))
    dense = []
    for path in sorted(args.run.glob('trace_*.npz')):
        trace = np.load(path)
        if 'dense' in trace:
            dense.append(trace['dense'])
    if dense:
        rows = np.concatenate(dense, axis=0)
        for i, case in enumerate(cases):
            for key in ('effective_signed_increment', 'tracking_signed_increment',
                        'raw_tracking_norm', 'raw_effective_norm'):
                v = rows[:, i, ma.METRICS.index(key)]
                dense_rows.append(dict(target=case['target'], seed=case['seed'],
                    alpha_m=case['alpha_m'], alpha_v=case['alpha_v'], metric=key,
                    updates=len(v), lag_one_correlation=correlation(v[:-1], v[1:]),
                    first_half_mean=float(v[:len(v)//2].mean()),
                    second_half_mean=float(v[len(v)//2:].mean()),
                    warning='Scalar temporal statistic, not raw force-vector phase stability.'))
    write_csv(args.output/'endpoints.csv', endpoints)
    write_csv(args.output/'contrasts.csv', contrasts)
    write_csv(args.output/'forecast_phases.csv', phase_rows)
    write_csv(args.output/'dense_scalar_diagnostics.csv', dense_rows)
    summaries = []
    for horizon in sorted({r['additional_updates'] for r in contrasts}):
        for model in ('frozen', 'two_phase'):
            for am, av in ma.ARMS[1:]:
                group = [r for r in contrasts if r['additional_updates'] == horizon
                         and r['model'] == model and (r['alpha_m'], r['alpha_v']) == (am, av)]
                values = [r['lambda_vector_skill'] for r in group if r['lambda_vector_skill'] is not None]
                if group:
                    summaries.append(dict(horizon=horizon, model=model, alpha_m=am, alpha_v=av,
                        cases=len(group), nonzero_effects=len(values),
                        median_lambda_vector_skill=float(np.median(values)) if values else None,
                        cases_beating_zero_effect=sum(v > 0 for v in values),
                        mean_sign_matches=sum(r['mean_sign_match'] for r in group)))
    ma.write_json(args.output/'summary.json', dict(cases=len(cases),
        targets=len(baseline), horizons=sorted({r['additional_updates'] for r in endpoints}),
        contrast_rows=len(contrasts), input_sha256=ma.digest(args.inputs),
        forecast_contrast_summaries=summaries,
        prediction_manifest_sha256=ma.digest(args.predictions/'manifest.json'),
        denominator_policy='Only exact zero is undefined; no empirical resolution cutoff was invented.',
        phase_scope='Two checkpoint-derived phases; dense traces do not retain raw force vectors.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'predictions', 'run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    analyze(p.parse_args())


if __name__ == '__main__':
    main()
