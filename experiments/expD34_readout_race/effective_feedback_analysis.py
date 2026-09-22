"""Offline forecast comparisons and exact-travel audits; never generate reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import adam_forces as af

HEADLINES = (0, 1, 2, 10, 100, 1000, 10000, 50000, 200000,
             500000, 2000000, 5400000)
THRESHOLDS = (1., 3.2, 16.)


def _key(case, degree, eta):
    return case['target'], int(case['seed']), int(case['start']), int(degree), float(eta)


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else np.nan


def _alignment(a, b):
    return _ratio(a @ b, np.linalg.norm(a)*np.linalg.norm(b))


def prediction_index(roots):
    """Index immutable per-case archives, checking their issued hashes."""
    index = {}
    seen = set()
    for root in map(Path, roots):
        paths = [root] if root.is_file() else sorted(root.rglob('manifest.json'))
        for path in paths:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            manifest = json.loads(path.read_text())
            for record in manifest.get('records', []):
                archive = path.parent/record['file']
                if hashlib.sha256(archive.read_bytes()).hexdigest() != record['sha256']:
                    raise ValueError(f'Changed prediction archive: {archive}')
                key = _key(record['case'], record['degree'], record['eta'])
                with np.load(archive) as data:
                    # Derivative matrices are large and are not needed for an
                    # independently issued endpoint forecast comparison.
                    arrays = {k: data[k].copy() for k in data.files if k in ('steps', 'p0')
                              or k.endswith(('_p', '_supported', '_effective_a', '_effective'))}
                entry = dict(path=str(archive), arrays=arrays, record=record)
                index.setdefault(key, []).append(entry)
    return index


def _matched_prediction(index, case, manifest, p0):
    matches = [item for item in index.get(_key(case, manifest['degree'], manifest['eta']), [])
               if np.array_equal(item['arrays']['p0'], p0)]
    if not matches:
        return None
    first = matches[0]
    for item in matches[1:]:
        if item['arrays'].keys() != first['arrays'].keys() or any(
                not np.array_equal(value, item['arrays'][key], equal_nan=True)
                for key, value in first['arrays'].items()):
            raise ValueError(f'Ambiguous predictions for {_key(case, manifest["degree"], manifest["eta"])}')
    return first


def evaluation_mse(p, target, samples=8192):
    """Evaluate with the original 2048-grid target mapping and normalization."""
    x, y, _, _ = af.data(target, samples)
    a, b, c = np.asarray(p[:-1]).reshape(3, -1)
    prediction = np.tanh(x[:, None]*a+b) @ c+p[-1]
    return float(np.mean((prediction-y)**2)/np.mean(y*y))


def actual_metrics(data, i, p0):
    p = data['p'][i]
    width = (len(p)-1)//3
    gamma0, gamma = np.abs(p0[:width]), np.abs(p[:width])
    change = gamma-gamma0
    signed = sum(data[k][i] for k in ('effective', 'tracking', 'omitted', 'crossing'))
    row = dict(actual_mean_gamma=float(gamma.mean()), actual_max_gamma=float(gamma.max()),
        actual_signed_mean_gamma_change=float(change.mean()),
        actual_positive_travel=float(data['positive'][i].mean()),
        actual_negative_travel=float(data['negative'][i].mean()),
        actual_effective_travel=float(data['effective'][i].mean()),
        actual_tracking_travel=float(data['tracking'][i].mean()),
        actual_omitted_travel=float(data['omitted'][i].mean()),
        actual_crossing=float(data['crossing'][i].mean()),
        motion_identity=float(np.max(np.abs(data['positive'][i]-data['negative'][i]-change))),
        signed_identity=float(np.max(np.abs(signed-change))))
    for ti, threshold in enumerate(THRESHOLDS):
        hit = data['first_hit'][i, ti]
        row[f'initial_fraction_{threshold:g}'] = float(np.mean(gamma0 >= threshold))
        row[f'new_ever_fraction_{threshold:g}'] = float(np.mean(hit > 0))
        row[f'ever_fraction_{threshold:g}'] = float(np.mean(hit >= 0))
        row[f'current_fraction_{threshold:g}'] = float(np.mean(gamma >= threshold))
    return row


def forecast_metrics(p, p0, effective, forecast_p, forecast_force):
    width = (len(p)-1)//3
    difference = forecast_p[:width]-p[:width]
    movement = p[:width]-p0[:width]
    gamma = np.abs(forecast_p[:width])
    force_error = np.linalg.norm(forecast_force-effective)
    return dict(predicted_mean_gamma=float(gamma.mean()), predicted_max_gamma=float(gamma.max()),
        predicted_signed_mean_gamma_change=float(np.mean(gamma-np.abs(p0[:width]))),
        slope_motion_absolute_error=float(np.linalg.norm(difference)),
        slope_motion_relative_error=_ratio(np.linalg.norm(difference), np.linalg.norm(movement)),
        effective_force_absolute_error=float(force_error),
        effective_force_relative_error=_ratio(force_error, np.linalg.norm(effective)),
        effective_force_alignment=_alignment(forecast_force, effective))


def collect(source, predictions, eval_samples=8192):
    """Return comparison rows and arm contrasts at preselected horizons only."""
    index = prediction_index(predictions)
    rows, branches = [], {}
    for path in sorted(Path(source).rglob('manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest.get('arm') not in ('joint', 'freeze_map', 'clamp_residual') or 'cases' not in manifest:
            continue
        folder = path.parent
        fork = folder/'snapshots'/'000000000.npz'
        if not fork.exists():
            raise ValueError(f'Missing initial snapshot: {folder}')
        with np.load(fork) as data:
            initial = data['p'].copy()
        matches = [_matched_prediction(index, case, manifest, p0)
                   for case, p0 in zip(manifest['cases'], initial)]
        for snapshot in sorted((folder/'snapshots').glob('*.npz')):
            with np.load(snapshot) as data:
                offset = int(data['offset'])
                if offset not in HEADLINES:
                    continue
                for i, case in enumerate(manifest['cases']):
                    arm, p0, p = manifest['arm'], initial[i], data['p'][i]
                    width = (len(p)-1)//3
                    failed = int(data['failed'][i])
                    updates = int(data['count'][i])
                    matched = matches[i]
                    base = dict(target=case['target'], seed=case['seed'], start=case['start'],
                        degree=manifest['degree'], eta=manifest['eta'], arm=arm, offset=offset,
                        updates=updates, time=offset*float(manifest.get('reference_eta', .002)),
                        failed=failed, run=str(folder), prediction_matched=matched is not None,
                        prediction_file='' if matched is None else matched['path'],
                        actual_eval_relative_mse=evaluation_mse(p, case['target'], eval_samples),
                        **actual_metrics(data, i, p0))
                    effective = data['metric_effective_a'][i]
                    arrays = {} if matched is None else matched['arrays']
                    locations = np.flatnonzero(arrays.get('steps', np.array([])) == updates)
                    hi = int(locations[0]) if len(locations) == 1 else None
                    models = [('affine', arm)]
                    if arm == 'joint':
                        models += [('effective_pure', 'effective_pure'),
                                   ('effective_with_remainder', 'effective_with_remainder')]
                    affine_p = None
                    for label, prefix in models:
                        force_key = prefix+('_effective_a' if label == 'affine' else '_effective')
                        supported = (not failed and hi is not None and prefix+'_supported' in arrays
                                     and bool(arrays[prefix+'_supported'][hi]))
                        if supported:
                            predicted = arrays[prefix+'_p'][hi]
                            force = arrays[force_key][hi][:width]
                            supported = bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite(force)))
                        record = base | dict(model=label, forecast_supported=supported)
                        if supported:
                            record.update(forecast_metrics(p, p0, effective, predicted, force))
                            if label == 'affine':
                                affine_p = predicted
                        rows.append(record)
                    key = (*_key(case, manifest['degree'], manifest['eta']), offset, arm)
                    entry = dict(p=p.copy(), p0=p0.copy(), predicted=affine_p,
                                 effective=effective.copy(), failed=failed, base=base)
                    if key in branches:
                        raise ValueError(f'Duplicate actual trajectory for {key}; select one run root')
                    branches[key] = entry
    contrasts = []
    for key, branch in branches.items():
        if key[-1] == 'joint':
            continue
        joint = branches.get((*key[:-1], 'joint'))
        if joint is None or not np.array_equal(joint['p0'], branch['p0']):
            continue
        width = (len(branch['p'])-1)//3
        delta = branch['p'][:width]-joint['p'][:width]
        mean_change = float(np.mean(abs(branch['p'][:width])-abs(joint['p'][:width])))
        row = {k: branch['base'][k] for k in ('target', 'seed', 'start', 'degree', 'eta', 'arm', 'offset', 'time')}
        row.update(actual_contrast_norm=float(np.linalg.norm(delta)),
                   actual_mean_gamma_contrast=mean_change,
                   actual_effective_contrast_norm=float(np.linalg.norm(branch['effective']-joint['effective'])),
                   contrast_supported=branch['predicted'] is not None and joint['predicted'] is not None,
                   valid=not (branch['failed'] or joint['failed']))
        if row['contrast_supported']:
            predicted = branch['predicted'][:width]-joint['predicted'][:width]
            predicted_mean = float(np.mean(abs(branch['predicted'][:width])-abs(joint['predicted'][:width])))
            row.update(predicted_contrast_norm=float(np.linalg.norm(predicted)),
                       contrast_alignment=_alignment(predicted, delta),
                       contrast_relative_error=_ratio(np.linalg.norm(predicted-delta), np.linalg.norm(delta)),
                       predicted_mean_gamma_contrast=predicted_mean,
                       mean_gamma_contrast_sign_agrees=bool(np.sign(predicted_mean) == np.sign(mean_change)))
        contrasts.append(row)
    return rows, contrasts


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_comparison(rows, output):
    """Fixed degree-9/sine panels; each available seed is a separate curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['model'] == 'affine' and r['degree'] == 65
                and r['eta'] == .002 and not r['failed']]
    colors = dict(joint='black', freeze_map='tab:blue', clamp_residual='tab:orange')
    for metric, ylabel, name in [('actual_signed_mean_gamma_change', 'Change in mean |a|', 'signed_scale_change'),
                                  ('actual_eval_relative_mse', 'Evaluation relative MSE', 'evaluation_error')]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for row_index, target in enumerate(('moment9', 'sine')):
            for column, start in enumerate((400000, 600000)):
                ax = axes[row_index, column]
                records = [r for r in selected if r['target'] == target and r['start'] == start]
                seeds = sorted({r['seed'] for r in records})
                for arm, color in colors.items():
                    for si, seed in enumerate(seeds):
                        values = sorted([r for r in records if r['arm'] == arm and r['seed'] == seed], key=lambda r: r['offset'])
                        if values:
                            ax.plot([r['offset'] for r in values], [r[metric] for r in values],
                                    color=color, alpha=.55, label=arm if si == 0 else None)
                if not records:
                    ax.text(.5, .5, 'No retained run', ha='center', transform=ax.transAxes)
                ax.set_title(f'{target}, fork {start:,}; {len(seeds)} seeds')
                ax.set_xscale('symlog', linthresh=10)
                ax.set_xlabel('Additional reference GD updates'); ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                if records:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(Path(output)/f'{name}.{suffix}', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, contrasts = collect(args.source, args.predictions)
    if not rows:
        raise ValueError('No headline experiment snapshots found')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'actual_vs_forecast.csv', rows)
    write_csv(args.output/'branch_contrasts.csv', contrasts)
    plot_comparison(rows, args.output)
    print(json.dumps(dict(rows=len(rows), contrasts=len(contrasts), output=str(args.output))))


if __name__ == '__main__':
    main()
