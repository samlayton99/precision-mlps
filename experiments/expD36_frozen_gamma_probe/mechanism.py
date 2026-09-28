"""Gamma-based finite-time predictions without dictionary spectral measurements.

Predictions use the target polynomial tails, slopes, center counts, and sample
grid only. Saved optimizer evaluations are read afterward for validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import core, full_core as f


def curvature_lower_bound(x, centers, gamma):
    """Raw-map Rayleigh lower bound on a symmetric grid; includes output bias."""
    count = int(np.count_nonzero(np.abs(centers) <= .5))
    fraction = float(np.mean(np.abs(x) >= .75))
    return max(1., count*fraction**2*np.tanh(gamma/4)**2)


def finite_error_floor(tails, log_access, steps, eta_upper):
    """Relative-residual floor from ||theta_n-theta_0||^2 <= n*eta*||r0||^2."""
    movement = (np.zeros_like(log_access) if steps == 0 else
                np.exp(.5*(np.log(steps*eta_upper)+log_access)))
    values = np.maximum(tails-movement, 0)
    degree = int(np.argmax(values))
    return dict(error=float(values[degree]), k=degree if values[degree] > 0 else None)


def forced_slow_mass(delta, access_over_cutoff):
    """Angle bound on residual energy below a spectral threshold; no spectrum input."""
    sine = np.sqrt(np.minimum(access_over_cutoff, 1.))
    return np.maximum(delta*np.sqrt(1-sine*sine)
                      - np.sqrt(np.maximum(1-delta*delta, 0))*sine, 0)**2


def predict(cfg):
    """All predictions are completed here, without reading experiment outputs."""
    rows = []
    for n in cfg['widths']:
        g = core.geometry(n)
        x = np.linspace(-1, 1, cfg['samples_per_cell']*n+1)
        names = cfg['targets'] if n == cfg['n'] else cfg['robust_targets']
        y = np.column_stack([f.target(x, name) for name in names])/np.sqrt(len(x))
        transform = core.polynomial_transform(x, cfg['k_max'])
        yh = core.transform(transform, y)
        squared = np.cumsum((yh*yh)[::-1], axis=0)[::-1]
        tails = np.sqrt(squared[1:cfg['k_max']+2]/squared[0])
        # Do not turn unresolved target roundoff into a tiny positive floor.
        tails[tails < 1e-12] = 0
        if 'quadratic' in names:
            tails[2:, names.index('quadratic')] = 0
        gammas = cfg['gammas'] if n == cfg['n'] else [1, 4, n/8]
        for gamma in gammas:
            lower = curvature_lower_bound(x, g.centers, gamma)
            log_b = np.log(g.width)+2*core.log_feature_envelope(gamma, np.arange(len(tails)))
            for ti, target in enumerate(names):
                time = f.bound(tails[:, ti], log_b, .01, lower)
                checkpoints = [dict(step=step, **finite_error_floor(tails[:, ti], log_b, step, .5/lower))
                               for step in cfg['checkpoints']]
                k = time['k']
                rows.append(dict(n=n, width=g.width, gamma=gamma, target=target,
                    curvature_lower=lower,
                    central_centers=int(np.count_nonzero(np.abs(g.centers) <= .5)),
                    outer_sample_fraction=float(np.mean(np.abs(x) >= .75)),
                    grid_hash=core.array_hash(x), geometry_hash=core.array_hash(g.centers),
                    c2_without_measured_curvature=time,
                    witness_tail=float(tails[k, ti]) if k is not None else None,
                    witness_log_access=float(log_b[k]) if k is not None else None,
                    checkpoint_error_floors=checkpoints))
    return rows


def analyze(root):
    cfg = f.config()
    rows = predict(cfg)
    sources = {}; checked = 0; positive = 0
    for n in cfg['widths']:
        folder = root/'training'/f'N{n}_raw_gd'
        for name in ['case.json', 'evaluations.json']:
            path = folder/name
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        case = json.loads((folder/'case.json').read_text())
        evaluations = json.loads((folder/'evaluations.json').read_text())
        assert case['optimizer'] == 'gd'
        for row in [r for r in rows if r['n'] == n]:
            bi = case['gammas'].index(row['gamma'])
            ci = next(i for i, column in enumerate(case['columns']) if column['target'] == row['target'])
            assert case['columns'][ci]['initialization'] == 'zero'
            assert case['rates'][bi][ci] <= .5/row['curvature_lower']*(1+1e-12)
            for prediction in row['checkpoint_error_floors']:
                observed = next(e for e in evaluations if e['step'] == prediction['step'])
                error = observed['train'][bi][ci]
                assert prediction['error'] <= error+1e-12, (row, prediction, error)
                prediction['observed_error'] = error
                checked += 1; positive += int(prediction['error'] > 0)
    checks = dict(predicted_cases=len(rows), checkpoint_comparisons=checked,
                  positive_error_floors=positive, violations=0)
    core.write_json(root/'refinements/gamma_mechanism.json', dict(
        scope='post-hoc raw-map analysis; no new training', chi=.5, epsilon=.01,
        target_tail_roundoff_threshold=1e-12,
        prediction_inputs='target samples, center geometry, gamma, width, budget; no measured spectrum or curvature',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation_inputs_sha256=sources, checks=checks, rows=rows))
    plot(root, rows)
    print(json.dumps(checks, indent=2))


def plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    primary = [r for r in rows if r['n'] == 512 and r['target'] == 'sine_mix_2_6_10']
    gamma = [r['gamma'] for r in primary]
    axes[0].plot(gamma, [r['checkpoint_error_floors'][-1]['observed_error'] for r in primary],
                 'o-', color='#777777', label='Executed GD error')
    axes[0].plot(gamma, [r['checkpoint_error_floors'][-1]['error'] for r in primary],
                 'o-', color='#186da7', label='Analytic lower bound')
    axes[0].set(xscale='log', xlabel='Frozen slope gamma', ylabel='Relative training error',
                title='Sine mixture after 200k updates')
    axes[0].legend(fontsize=8)
    width = sorted([r for r in rows if r['gamma'] == 4 and r['target'] == 'sine_mix_2_6_10'], key=lambda r:r['width'])
    axes[1].plot([r['width'] for r in width],
                 [r['c2_without_measured_curvature']['bound']/1e6 for r in width], 'o-', color='#186da7')
    axes[1].axhline(.2, color='#777777', linestyle='--', label='Executed budget')
    axes[1].set(xlabel='Hidden width W', ylabel='Necessary updates (millions)', ylim=(0, 1.7),
                title='Gamma 4: bound to 1% error')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Gamma-based predictions use no measured spectrum or curvature', fontsize=10)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(root/f'refinements/gamma_mechanism.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)
