"""Post-hoc analytic and spectral refinements using compact, saved sweep data.

No training or fitted constants. See the full-sweep report for the proofs.
Floating-point evaluations are estimates, not arithmetic enclosures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import core, full_core as f


def log_derivative_tail(gamma, degrees, terms=128):
    """Uniform Chebyshev-tail bound for d/dc tanh(gamma*(x-c)).

    Sum positive pole bounds and an explicit upper bound on the omitted tail.
    The real-axis derivative bound gamma supplies an additional valid cap.
    """
    k = np.asarray(degrees, dtype=float)
    terms = max(terms, int(np.ceil(gamma/np.pi)))
    v = np.pi*(np.arange(terms)+.5)/gamma
    beta = np.arcsinh(v)
    d = np.where(v < 1, np.sqrt(2*v), np.sqrt(1+v*v))
    bracket = ((k[:, None]+1+1/np.expm1(beta))/d**2
               + np.sqrt(d*d+1)/d**3)
    finite = (np.log(4/gamma) + logsumexp(-k[:, None]*beta
              - np.log(np.expm1(beta)) + np.log(bracket), axis=1))
    # v_terms >= 1; a decreasing-series first-term-plus-integral bound.
    remainder = (np.log(32)+(k+2)*np.log(gamma)-(k+3)*np.log(np.pi)
                 + np.log(k+2+np.sqrt(2))
                 + np.logaddexp(-(k+3)*np.log(2*terms+1),
                     -(k+2)*np.log(2*terms+1)-np.log(2*(k+2))))
    return np.minimum(np.log(gamma), np.logaddexp(finite, remainder))


def log_neighbor_envelope(gamma, degrees, centers, scales, terms=128):
    """Frobenius envelope retaining neighboring cancellation and final anchor."""
    u = core.log_feature_envelope(gamma, degrees)
    derivative = log_derivative_tail(gamma, degrees, terms)
    differences = np.minimum(np.log(2)+u[:, None],
                             derivative[:, None]+np.log(np.diff(centers)))
    columns = 2*(differences+np.log(scales[1:-1]))
    anchor = 2*(u+np.log(scales[-1]))
    return logsumexp(np.column_stack([columns, anchor]), axis=1)


def spectral_tail_bound(eigenvalues, weights, eta, epsilon):
    """Jensen bound over all slow spectral prefixes, omitting unknown energy.

    weights are squared initial-residual loadings / ||r0||^2. They may sum
    to less than one when only resolved eigenpairs are supplied. In particular,
    the residual of a truncated SVD is NOT assigned to an exact nullspace.
    """
    eigenvalues, weights = np.asarray(eigenvalues), np.asarray(weights)
    if (np.any(eigenvalues < 0) or np.any(weights < 0) or eta <= 0
            or np.any(eta*eigenvalues >= 1) or not 0 < epsilon < 1):
        raise ValueError('Require nonnegative spectrum/weights and 0 < eta*L, epsilon < 1')
    order = np.argsort(eigenvalues)
    values, weights = eigenvalues[order], weights[order]
    mass = np.cumsum(weights)
    moment = np.cumsum(weights*values)
    eligible = np.flatnonzero(mass > epsilon**2)
    if not len(eligible):
        return dict(bound=0., log10_bound=None, status='insufficient_retained_energy')
    mean = moment[eligible]/mass[eligible]
    if np.any(mean == 0):
        return dict(bound=None, log10_bound=None, status='exact_zero_mode_obstruction')
    times = np.log(np.sqrt(mass[eligible])/epsilon)/(-np.log1p(-eta*mean))
    best = int(np.argmax(times)); index = int(eligible[best]); time = float(times[best])
    return dict(bound=float(np.ceil(time)) if time < 2**53 else None,
                log10_bound=float(np.log10(time)), continuous_bound=time,
                mass=float(mass[index]), mean_eigenvalue=float(mean[best]),
                max_eigenvalue=float(values[index]), modes=index+1,
                status='retained_spectrum_estimate')


def analyze(root):
    cfg = f.config(); rows = []; sources = {}
    checks = dict(subspace_envelope_cases=0, directional_envelope_cases=0,
                  reached_cases=0, spectral_forecast_comparisons=0,
                  max_cutoff_relative_change_reached=0.)

    def record(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    for name in cfg['maps']:
        diagnostics = json.loads(record(root/'diagnostics'/f'N512_{name}_gd'/'certificates.json').read_text())
        g = core.geometry(cfg['n']); scales, neighbor = f.map_spec(g, name)
        for gamma in cfg['gammas']:
            folder = root/'dictionaries'/f'N512_{name}_g{gamma:g}'
            meta = json.loads(record(folder/'meta.json').read_text())
            assert core.array_hash(g.centers) == meta['geometry_hash']
            np.testing.assert_array_equal(scales, meta['scales'])
            certs = json.loads(record(folder/'certificates.json').read_text())
            capacity = json.loads(record(folder/'capacity.json').read_text())
            with np.load(record(folder/'access.npz')) as archive:
                access = dict(archive)
            with np.load(record(folder/'spectrum.npz')) as archive:
                spectrum = dict(archive)
            degrees = np.arange(len(access['log_B_used']))
            matrix_noise = np.max(access['noise'][0]*access['E_measured'][0]**2)
            log_b = np.minimum(access['log_B_used'], np.log(meta['L']))
            if neighbor:
                log_columns = log_neighbor_envelope(gamma, degrees, g.centers, scales)
                # This envelope bounds the full projected Frobenius norm too.
                resolved = access['frobenius'] > matrix_noise
                assert np.all(access['frobenius'][resolved] <= np.exp(log_columns[resolved])*(1+1e-10))
                log_b = np.minimum(log_b, log_columns)
            finite = np.isfinite(access['b']) & (access['b'] > matrix_noise)
            assert np.all(access['b'][finite] <= np.exp(log_b[finite])*(1+1e-10))
            checks['subspace_envelope_cases'] += int(finite.sum())
            resolved = access['mu'] > access['noise']
            assert np.all(access['mu'][resolved] <= np.broadcast_to(np.exp(log_b[:, None]), resolved.shape)[resolved]*(1+1e-10))
            checks['directional_envelope_cases'] += int(resolved.sum())
            for ti, target in enumerate(cfg['targets']):
                for epsilon in cfg['tolerances']:
                    old = {c['kind']: c for c in certs if c['target'] == target and c.get('epsilon') == epsilon}
                    executed = next(c for c in diagnostics if c['gamma'] == gamma
                                    and c['target'] == target and c['epsilon_target'] == epsilon
                                    and c['kind'] == 'directional')
                    spectral = {}
                    for cutoff in cfg['cutoffs']:
                        keep = spectrum['singular'] > cutoff*np.sqrt(meta['L'])
                        value = spectral_tail_bound(spectrum['singular'][keep]**2,
                            (spectrum['loadings'][keep, ti]/spectrum['norm_y'][ti])**2,
                            .5/meta['L'], epsilon)
                        spectral[str(cutoff)] = value
                        refit = next(c for c in capacity if c['target'] == target and c['cutoff'] == cutoff)
                        forecast = next(c for c in refit['predictions'] if c['epsilon'] == epsilon)
                        if forecast['log10_steps'] is not None and value['log10_bound'] is not None:
                            assert value['log10_bound'] <= forecast['log10_steps']+1e-10
                            checks['spectral_forecast_comparisons'] += 1
                    analytic = f.bound(access['E'][:, ti], log_b, epsilon, meta['L'])
                    assert analytic['log10_bound'] is None or analytic['log10_bound'] >= old['analytic']['log10_bound']-1e-10
                    hit = executed['first_hit']
                    if hit is not None:
                        checks['reached_cases'] += 1
                        for value in [analytic, *spectral.values()]:
                            assert value['bound'] is not None and value['bound'] <= hit, (name, gamma, target, epsilon, value, hit)
                        times = [v['continuous_bound'] for v in spectral.values() if 'continuous_bound' in v]
                        if times:
                            checks['max_cutoff_relative_change_reached'] = max(
                                checks['max_cutoff_relative_change_reached'], max(times)/min(times)-1)
                    rows.append(dict(map=name, gamma=gamma, target=target, epsilon=epsilon,
                        analytic_c2=old['analytic'], directional_c2=old['directional'],
                        refined_analytic_c2=analytic, spectral_tail=spectral, first_hit=hit))
    result = dict(analysis='post_hoc; zero-start primary sweep; no new training',
                  numerical_status='FP64 estimates, not interval certificates; discarded spectral energy omitted',
                  source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_artifacts_sha256=sources, pole_terms=128, checks=checks, rows=rows)
    core.write_json(root/'refinements/c2_tightening.json', result)
    plot(root, rows)
    print(json.dumps(checks, indent=2))


def plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, name, title in zip(axes, ['raw', 'collective_neighbor'], ['Raw coordinates', 'Collective neighboring']):
        selected = [r for r in rows if r['map'] == name and r['target'] == 'sine_mix_2_6_10'
                    and r['epsilon'] == .01 and r['first_hit'] is not None]
        gammas = [r['gamma'] for r in selected]
        for key, label, color in [('directional_c2', 'Original directional C2', '#7a7a7a'),
                                   ('spectral_tail', 'Spectral-tail Jensen', '#186da7')]:
            values = [r[key]['1e-14'] if key == 'spectral_tail' else r[key] for r in selected]
            ax.plot(gammas, [r['first_hit']/v['bound'] for r, v in zip(selected, values)],
                    'o-', label=label, color=color, markersize=4)
        ax.axhline(1, color='black', linestyle=':', linewidth=1)
        ax.set(xlabel='Frozen slope gamma', title=title, yscale='log')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Executed first hit / necessary updates')
    axes[0].legend(fontsize=8)
    fig.suptitle('1% sine-mixture error: reached cases only; closer to 1 is tighter', fontsize=10)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(root/f'refinements/c2_tightening.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)
