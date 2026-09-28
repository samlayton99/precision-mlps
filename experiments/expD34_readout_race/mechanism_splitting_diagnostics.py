"""Additional PSD block diagnostics and prospective independent-width scoring."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import effective_feedback as ef, transport, mechanism_splitting as ms
from .mechanism_splitting_baselines import matrices


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def psd(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    rows, forms = [], []
    selections = [(args.root/'inputs'/f'{cohort}.npz', None, cohort, ms.ARMS)
                  for cohort in ('development', 'confirmation')]
    selections += [(args.root/'width_inputs'/f'N{n}.npz',
                    args.root/'widths'/f'N{n}'/'snapshots/000020000.npz',
                    f'width{n}', ('original',)) for n in (128, 512, 1024)]
    for source, snapshot, cohort, arms in selections:
        pp, x, yy, cases = ef.load_inputs(source)
        if snapshot:
            with np.load(snapshot) as data:
                pp = data['p'].copy()
        q = transport.basis(x, 65)
        for arm in arms:
            for p, y, case in zip(pp, yy, cases):
                w = (len(p)-1)//3
                d = ms.mobility(w, arm)
                _, T, JH, eH = matrices(p, x, y, d, q)
                blocks = np.stack([T[lo:hi].T@(T[lo:hi]/d[lo:hi, None])
                    for lo, hi in ((0, w), (w, 2*w), (2*w, 3*w), (3*w, 3*w+1))])
                S = JH@T
                energies = np.einsum('i,kij,j->k', eH, blocks, eH)
                total = float(eH@S@eH)
                row = dict(case, cohort=cohort, arm=arm, start=case['start']+(20000 if snapshot else 0),
                    sum_to_S_error=float(np.linalg.norm(blocks.sum(axis=0)-S)), S_norm=float(np.linalg.norm(S)),
                    block_min_eigenvalue=float(min(np.linalg.eigvalsh(v).min() for v in blocks)),
                    total_effective_energy=total)
                for label, value in zip(('a', 'b', 'c', 'd'), energies):
                    row[label+'_energy'] = float(value)
                    row[label+'_share'] = float(value/total) if total > 0 else np.nan
                rows.append(row); forms.append(blocks)
    write_csv(out/'psd_blocks.csv', rows)
    np.savez_compressed(out/'psd_blocks.npz', Q_blocks=np.asarray(forms))
    (out/'psd_provenance.json').write_text(json.dumps(dict(
        source_sha256=ef.digest(__file__), degree=65, rows=len(rows),
        formula='Q_block=T_velocity,block.T diag(1/d_block) T_velocity,block; sum Q_block=S',
        interpretation='PSD allocation of instantaneous effective dissipation; differs from signed J_H,block T_velocity,block residual forcing'), indent=2))


def width(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (128, 512, 1024):
        folder = args.root/'width_predictions'/f'N{n}'
        manifest = json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'predictions.npz') as f:
            pred = {key: f[key] for key in f.files}
        h = 2/n
        for hi, offset in enumerate(pred['horizons']):
            path = args.root/'widths_post20k'/f'N{n}'/'snapshots'/f'{int(offset):09d}.npz'
            if not path.exists():
                continue
            with np.load(path) as f:
                actual = {key: f[key] for key in f.files}
            w = (pred['p0'].shape[1]-1)//3
            for i, case in enumerate(manifest['cases']):
                p0 = pred['p0'][i]
                displacement = actual['p'][i, :w]-p0[:w]
                motion = float(np.linalg.norm(displacement))
                actual_lambda = float(np.mean(abs(actual['p'][i, :w])-abs(p0[:w]))*h)
                row = dict(**case, additional_updates=int(offset), absolute_updates=case['start']+int(offset),
                    actual_lambda_displacement=actual_lambda, slope_displacement_norm=motion,
                    positive=float(actual['positive'][i].mean()), negative=float(actual['negative'][i].mean()),
                    everhit_lambda025=float(np.mean(actual['first_hit'][i] >= 0)),
                    relative_eval_mse=float(actual['metric_relative_eval_mse'][i]),
                    residual_tracking_ratio=float(actual['metric_full_complement_tracking_to_effective_residual_forcing'][i]),
                    slope_tracking_ratio=float(actual['metric_tracking_to_effective_slope'][i]),
                    projection_force_difference=float(pred['fine_projection_difference'][0, i]),
                    retained_effective_force_norm=float(pred['effective_norm'][0, i]))
                for model in ('constant_full', 'constant_effective', 'effective_pure', 'effective_remainder'):
                    state = pred[model+'_p'][0, i, hi, :w]
                    error = float(np.linalg.norm(state-p0[:w]-displacement))
                    row[model+'_vector_error'] = error
                    row[model+'_relative_vector_error'] = error/motion if motion > 0 else np.nan
                    row[model+'_lambda_prediction'] = float(np.mean(abs(state)-abs(p0[:w]))*h)
                    row[model+'_lambda_error'] = abs(row[model+'_lambda_prediction']-actual_lambda)
                rows.append(row)
    if not rows:
        raise ValueError('No width continuation checkpoints found')
    write_csv(out/'width_forecasts.csv', rows)
    selected = [r for r in rows if r['additional_updates'] == 20000]
    summary = {}
    for n in (128, 512, 1024):
        group = [r for r in selected if r['nref'] == n]
        if not group:
            continue
        pure = np.array([r['effective_pure_vector_error'] for r in group])
        constant = np.array([r['constant_effective_vector_error'] for r in group])
        summary[n] = dict(cases=len(group), pure_beats_constant=int(np.sum(pure < constant)),
            median_pure_over_constant=float(np.median(pure/constant)),
            median_pure_relative_error=float(np.median([r['effective_pure_relative_vector_error'] for r in group])),
            median_constant_relative_error=float(np.median([r['constant_effective_relative_vector_error'] for r in group])),
            max_projection_difference=float(max(r['projection_force_difference'] for r in group)),
            max_residual_tracking_ratio=float(max(r['residual_tracking_ratio'] for r in group)))
    (out/'width_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('psd', 'width'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
