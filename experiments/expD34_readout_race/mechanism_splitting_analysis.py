"""Cached targetwise splitting forecasts and contrasts; training source unchanged."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import effective_feedback as ef, mechanism_splitting as ms


def analyze(args):
    with np.load(args.predictions/'predictions.npz') as archive:
        prediction = {key: archive[key] for key in ('horizons', 'p', 'effective_pure_p', 'effective_remainder_p')}
    driver_path = getattr(args, 'drivers', None)
    drivers = None
    if driver_path:
        with np.load(driver_path/'predictions.npz') as archive:
            drivers = {key: archive[key] for key in ('horizons', 'constant_full_p', 'constant_effective_p')}
    pp, _, _, cases = ef.load_inputs(args.inputs)
    w = (pp.shape[1]-1)//3
    rows = []
    cache = {}
    def snapshot(arm, horizon):
        key = (arm, int(horizon))
        if key not in cache:
            path = args.source/arm/'snapshots'/f'{int(horizon):09d}.npz'
            if path.exists():
                with np.load(path) as data:
                    cache[key] = {name: data[name] for name in data.files}
            else:
                cache[key] = None
        return cache[key]
    for ai, arm in enumerate(ms.ARMS):
        k, hidden, _ = ms.settings(arm)
        for hi, horizon in enumerate(prediction['horizons']):
            data = snapshot(arm, horizon)
            if data is None:
                continue
            baseline = snapshot('original', horizon)
            hidden_clock = float(horizon)*hidden/k
            clock_baseline = snapshot('original', int(hidden_clock)) if hidden_clock.is_integer() else None
            for i, case in enumerate(cases):
                actual = data['p'][i, :w]-pp[i, :w]
                magnitude_change = float(np.mean(abs(data['p'][i, :w])-abs(pp[i, :w]))/64)
                row = dict(**case, arm=arm, horizon=int(horizon), failed=int(data['failed'][i]),
                    relative_mse=float(data['relative_mse'][i]), mean_lambda=float(np.mean(abs(data['p'][i, :w]))/64),
                    actual_mean_lambda_change=magnitude_change, slope_displacement_norm=float(np.linalg.norm(actual)),
                    positive=float(data['positive'][i].mean()), negative=float(data['negative'][i].mean()),
                    effective=float(data['effective'][i].mean()), tracking=float(data['tracking'][i].mean()),
                    physical_readout_rms_over_h=float(np.sqrt(np.mean(data['p'][i, 2*w:3*w]**2))*64/k),
                    original_hidden_clock_offset=hidden_clock,
                    hidden_clock_comparison_available=clock_baseline is not None)
                if 'full_complement_tracking_to_effective_residual_forcing' in data:
                    row['residual_tracking_ratio'] = float(data['full_complement_tracking_to_effective_residual_forcing'][i])
                if baseline is not None:
                    actual_contrast = data['p'][i, :w]-baseline['p'][i, :w]
                    signed_contrast = float(np.mean(abs(data['p'][i, :w])-abs(baseline['p'][i, :w]))/64)
                    row.update(actual_mean_lambda_contrast=signed_contrast,
                               noeffect_contrast_vector_error=float(np.linalg.norm(actual_contrast)),
                               noeffect_contrast_signed_error=abs(signed_contrast))
                if clock_baseline is not None:
                    row['hidden_clock_mean_lambda_contrast'] = float(np.mean(abs(data['p'][i, :w])-abs(clock_baseline['p'][i, :w]))/64)
                for label, key in [('tangent', 'p'), ('effective_pure', 'effective_pure_p'),
                                   ('effective_remainder', 'effective_remainder_p')]:
                    predstate = prediction[key][ai, i, hi, :w]
                    predicted = predstate-pp[i, :w]
                    error = float(np.linalg.norm(predicted-actual))
                    row[label+'_supported'] = bool(np.isfinite(predicted).all())
                    row[label+'_displacement_error'] = error
                    row[label+'_mean_lambda_change'] = float(np.mean(abs(predstate)-abs(pp[i, :w]))/64)
                    if baseline is not None:
                        predref = prediction[key][0, i, hi, :w]
                        predcontrast = predstate-predref
                        signed_prediction = float(np.mean(abs(predstate)-abs(predref))/64)
                        row[label+'_contrast_vector_error'] = float(np.linalg.norm(predcontrast-actual_contrast))
                        row[label+'_contrast_signed_prediction'] = signed_prediction
                        row[label+'_contrast_signed_error'] = abs(signed_prediction-signed_contrast)
                if drivers is not None and horizon in drivers['horizons']:
                    di = int(np.flatnonzero(drivers['horizons'] == horizon)[0])
                    for label in ('constant_full', 'constant_effective'):
                        predstate = drivers[label+'_p'][ai, i, di, :w]
                        row[label+'_displacement_error'] = float(np.linalg.norm(predstate-pp[i, :w]-actual))
                        if baseline is not None:
                            predref = drivers[label+'_p'][0, i, di, :w]
                            signed_prediction = float(np.mean(abs(predstate)-abs(predref))/64)
                            row[label+'_contrast_vector_error'] = float(np.linalg.norm(predstate-predref-actual_contrast))
                            row[label+'_contrast_signed_prediction'] = signed_prediction
                            row[label+'_contrast_signed_error'] = abs(signed_prediction-signed_contrast)
                rows.append(row)
    if not rows:
        raise ValueError('No completed prediction horizons found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(dict(
        analyzer_sha256=ef.digest(__file__), input_sha256=ef.digest(args.inputs),
        prediction_sha256=ef.digest(args.predictions/'predictions.npz'),
        driver_sha256=ef.digest(driver_path/'predictions.npz') if driver_path else None,
        rows=len(rows), hidden_clock='n times quotient hidden mobility; matching this clock is not a full trajectory identity',
        full_compensation='Exact trajectory identity belongs only to k2_full and k4_full'), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--drivers', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()
