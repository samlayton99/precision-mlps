"""Summarize fixed-degree polynomial predictions without selecting targets."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import effective_feedback as ef
from .mechanism_polynomial_anchor import coarse
from .mechanism_splitting_diagnostics import write_csv


def analyze(root):
    with (root/'scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    summary = {}
    for n in (128, 512, 1024):
        for degree in (3, 5):
            subset = [r for r in rows if int(r['nref']) == n and int(r['degree']) == degree]
            error = np.array([float(r['polynomial_error']) for r in subset])
            constant = np.array([float(r['constant_effective_error']) for r in subset])
            frozen = np.array([float(r['frozen_effective_error']) for r in subset])
            summary[f'N{n}_degree{degree}'] = dict(cases=len(subset),
                finite=sum(r['finite'] == 'True' for r in subset),
                median_relative_error=float(np.median([float(r['polynomial_relative_error']) for r in subset])),
                max_relative_error=float(max(float(r['polynomial_relative_error']) for r in subset)),
                beats_constant=int(np.sum(error < constant)), beats_frozen=int(np.sum(error < frozen)),
                median_error_over_constant=float(np.median(error/constant)),
                misses_vs_constant=[f'{r["target"]}/seed{r["seed"]}' for r in subset if float(r['polynomial_error']) >= float(r['constant_effective_error'])],
                targets={target: [float(r['polynomial_relative_error']) for r in subset if r['target'] == target]
                         for target in sorted({r['target'] for r in subset})})
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def confirmation(root):
    rows = []
    for n in (128, 512, 1024):
        pp, x, _, cases = ef.load_inputs(root/'fork_inputs'/f'N{n}.npz')
        with np.load(root/'forecasts'/f'N{n}.npz') as data:
            models = {name: data[name].copy() for name in ('poly3', 'poly5', 'anchored5')}
        with np.load(root/'baselines'/f'N{n}'/'predictions.npz') as data:
            models.update({name: data[name+'_p'][0, :, 0].copy() for name in ('constant_effective', 'effective_pure')})
        with np.load(root/'continuation'/f'N{n}'/'snapshots/000020000.npz') as data:
            actual = data['p'].copy()
            if np.any(data['failed']):
                raise ValueError('Nonfinite confirmation trajectory')
        w = (pp.shape[1]-1)//3
        for i, case in enumerate(cases):
            motion = np.linalg.norm(actual[i, :w]-pp[i, :w]); c0 = coarse(pp[i], x)
            row = dict(case, actual_motion=float(motion),
                actual_coarse_drift=float(np.linalg.norm(coarse(actual[i], x)-c0)))
            for name, states in models.items():
                row[name+'_relative_error'] = float(np.linalg.norm(states[i, :w]-actual[i, :w])/motion)
                row[name+'_coarse_drift'] = float(np.linalg.norm(coarse(states[i], x)-c0))
            rows.append(row)
    write_csv(root/'scores.csv', rows)
    summary = {}
    for n in (128, 512, 1024):
        part = [r for r in rows if r['nref'] == n]
        summary[n] = {name: dict(median_relative_error=float(np.median([r[name+'_relative_error'] for r in part])),
            maximum_relative_error=float(max(r[name+'_relative_error'] for r in part)),
            beats_constant=sum(r[name+'_relative_error'] < r['constant_effective_relative_error'] for r in part),
            misses_vs_constant=[f'{r["target"]}/seed{r["seed"]}' for r in part if r[name+'_relative_error'] >= r['constant_effective_relative_error']],
            max_coarse_drift=float(max(r[name+'_coarse_drift'] for r in part))) for name in models}
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def anchored(root):
    with (root/'scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (root.parent/'polynomial/scores.csv').open() as stream:
        baseline = {(r['target'], r['seed'], r['nref']): r for r in csv.DictReader(stream) if r['degree'] == '5'}
    summary = {}
    for n in (128, 512, 1024):
        part = [r for r in rows if int(r['nref']) == n]
        summary[n] = dict(median_relative_error=float(np.median([float(r['anchored_relative_error']) for r in part])),
            max_relative_error=max(float(r['anchored_relative_error']) for r in part),
            beats_constant=sum(float(r['anchored_relative_error']) < float(baseline[(r['target'],r['seed'],r['nref'])]['constant_effective_relative_error']) for r in part),
            max_exact_coarse_drift=max(float(r['anchored_exact_coarse_drift']) for r in part))
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def clamped(root):
    with (root/'scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    summary = {}
    for n in (128, 512, 1024):
        part = [r for r in rows if int(r['nref']) == n]
        summary[n] = dict(cases=len(part),
            clamp_beats_own=sum(float(r['clamped_relative_error']) < float(r['own_error_relative_error']) for r in part),
            clamp_worse_cases=[f'{r["target"]}/seed{r["seed"]}' for r in part if float(r['clamped_relative_error']) >= float(r['own_error_relative_error'])],
            median_clamped_relative_error=float(np.median([float(r['clamped_relative_error']) for r in part])),
            median_own_relative_error=float(np.median([float(r['own_error_relative_error']) for r in part])),
            max_actual_residual_change=max(float(r['actual_full_complement_residual_fraction_change']) for r in part))
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--anchored', action='store_true')
    parser.add_argument('--clamped', action='store_true')
    args = parser.parse_args()
    (confirmation if args.confirmation else anchored if args.anchored else clamped if args.clamped else analyze)(args.root)
