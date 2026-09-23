"""Evaluate the distribution theorem on saved hits and small CPU dictionaries.

Produces numerical artifacts and figures only. The report is authored separately.
No optimizer trajectory is launched, and no constants are fitted to saved hits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svd

from . import core, full_core as f
from .mechanism import curvature_lower_bound
from .slope_spectrum import (
    analytic_access, cdf_atoms, cdf_error, cdf_time_bound, distribution_bound,
    exceptional_target_tails, mean_spectrum_bound, target_cdf_bound,
)
from .tighten import spectral_tail_bound


def measured_cdf(values, weights, thresholds, curvature):
    assert weights.sum() <= 1+1e-10
    return np.minimum(1., [weights[values <= s*curvature].sum() for s in thresholds])


def common_slope(root, record, checks, curves):
    cfg = f.config(); g = core.geometry(cfg['n'])
    x = np.linspace(-1, 1, cfg['samples_per_cell']*cfg['n']+1)
    y = np.column_stack([f.target(x, name) for name in cfg['targets']])/np.sqrt(len(x))
    k_max = cfg['k_max']; raw_qr = core.polynomial_transform(x, k_max)
    thresholds = np.geomspace(1e-32, 1., 641)
    curves['thresholds'] = thresholds
    rows = []
    case = json.loads(record(root/'training/N512_raw_gd/case.json').read_text())
    assert case['optimizer'] == 'gd' and case['gammas'] == cfg['gammas']
    assert cfg['tolerances'][0] == .01
    assert [column['target'] for column in case['columns']] == cfg['targets']
    assert all(column['initialization'] == 'zero' for column in case['columns'])
    with np.load(record(root/'training/N512_raw_gd/hitting_audit.npz')) as archive:
        hits = archive['first'].copy()
    evaluations = json.loads(record(root/'training/N512_raw_gd/evaluations.json').read_text())
    for gi, gamma in enumerate(cfg['gammas']):
        started = time.monotonic()
        # Form feature/target witnesses before reading this dictionary's spectrum.
        predicted = analytic_access(x, g.centers, gamma, y, k_max, 512, raw_qr=raw_qr)
        predicted['tails'][2:, cfg['targets'].index('quadratic')] = 0
        l_star = curvature_lower_bound(x, g.centers, gamma)
        folder = root/'dictionaries'/f'N512_raw_g{gamma:g}'
        meta = json.loads(record(folder/'meta.json').read_text())
        assert meta['geometry_hash'] == core.array_hash(g.centers)
        with np.load(record(folder/'spectrum.npz')) as archive:
            spectrum = dict(archive)
        with np.load(record(folder/'access.npz')) as archive:
            access = dict(archive)
        curvature = meta['L']; values = spectrum['singular']**2
        assert l_star <= curvature*(1+1e-12)
        np.testing.assert_allclose(case['rates'][gi], .5/curvature, rtol=1e-12)
        matrix_noise = float(np.max(access['noise'][0]*access['E_measured'][0]**2))
        spectral_noise = (64*np.finfo(float).eps*np.sqrt(len(x)*(g.width+1)))**2
        tail_values = np.array([values[k+1:].sum() for k in range(k_max+1)])
        resolved = tail_values > matrix_noise
        for bound in [predicted['cap'], predicted['centered']]:
            assert np.all(tail_values[resolved] <= bound[resolved]*(1+1e-8)+matrix_noise)
            checks['spectral_tail_comparisons'] += int(resolved.sum())
        resolved_access = access['frobenius'] > matrix_noise
        assert np.all(access['frobenius'][resolved_access]
                      <= predicted['centered'][resolved_access]*(1+1e-8)+matrix_noise)
        checks['center_frobenius_comparisons'] += int(resolved_access.sum())
        curves[f'g{gamma:g}_spectrum_tail'] = tail_values
        curves[f'g{gamma:g}_cap'] = predicted['cap']
        curves[f'g{gamma:g}_centered'] = predicted['centered']
        for ti, target in enumerate(cfg['targets']):
            tails = predicted['tails'][:, ti]
            weights = (spectrum['loadings'][:, ti]/spectrum['norm_y'][ti])**2
            actual_cdf = measured_cdf(values, weights, thresholds, curvature)
            # Unknown retained-model remainder is not declared to be a null mode.
            possible_cdf = np.minimum(1., actual_cdf+max(0., 1-weights.sum()))
            resolved_cdf = thresholds > 10*spectral_noise
            resolved_mu = (access['mu'][:, ti] > access['noise'][:, ti]) & (tails > 1e-8)
            assert np.all(access['mu'][resolved_mu, ti]
                          <= predicted['directional'][resolved_mu, ti]*(1+1e-7)
                          + access['noise'][resolved_mu, ti])
            checks['center_directional_comparisons'] += int(resolved_mu.sum())
            variants = {
                'cap_geometric': (predicted['cap'], l_star),
                'cap_given_step': (predicted['cap'], curvature),
                'center_frobenius_given_step': (predicted['centered'], curvature),
                'center_direction_given_step': (predicted['directional'][:, ti], curvature),
                'center_direction_geometric': (predicted['directional'][:, ti], l_star),
                'measured_access_diagnostic': (access['mu'][:, ti], curvature),
            }
            times = {}; distributions = {}
            for name, (bound, normalizer) in variants.items():
                mass = target_cdf_bound(tails, bound, thresholds, normalizer)
                assert np.all(mass[resolved_cdf] <= possible_cdf[resolved_cdf]+1e-8), (gamma, target, name)
                checks['target_mass_comparisons'] += int(resolved_cdf.sum())
                times[name] = cdf_time_bound(thresholds, mass)
                distributions[name] = mass
                curves[f'g{gamma:g}_{target}_{name}'] = mass
            times['measured_cdf_diagnostic'] = cdf_time_bound(thresholds, actual_cdf)
            times['spectral_jensen_diagnostic'] = spectral_tail_bound(values, weights, .5/curvature, .01)
            # Preserve the old C2 comparator; multithreshold conversion need not dominate it.
            times['center_direction_c2_given_step'] = f.bound(tails, np.log(predicted['directional'][:, ti]), .01, curvature)
            hit = int(hits[gi, ti, 0]); hit = hit if hit >= 0 else None
            if hit is not None:
                for name, value in times.items():
                    assert value['bound'] is not None and value['bound'] <= hit, (gamma, target, name, value, hit)
                checks['reached_cases'] += 1
            checks['censored_cases'] += int(hit is None)
            floors = []
            mass = distributions['center_direction_given_step']
            rates, atom_weights = cdf_atoms(thresholds, mass)
            for evaluation in evaluations:
                lower = cdf_error(evaluation['step'], rates, atom_weights)
                actual = evaluation['train'][gi][ti]
                assert lower <= actual+1e-10, (gamma, target, evaluation['step'], lower, actual)
                checks['executed_checkpoint_comparisons'] += 1
                floors.append(dict(step=evaluation['step'], lower=lower, actual=actual))
            old = f.bound(tails, np.log(predicted['cap']), .01, l_star)
            rows.append(dict(gamma=gamma, target=target, width=g.width, n=cfg['n'],
                L=curvature, geometric_L=l_star, first_hit=hit, budget=cfg['training_steps'],
                bounds=times, previous_geometric_c2=old, checkpoint_floors=floors,
                degree=512, pole_pairs=predicted['terms'],
                roundoff_monitor=predicted['roundoff_monitor'],
                resolved_center_degrees=int(predicted['centered_resolved'].sum()),
                numerical_scope=('nominal-feature prediction; FP64 spectrum unresolved in relevant tail'
                                 if gamma <= 2 else 'FP64 resolved comparisons only; no interval enclosure'),
                witness_degree_30=dict(cap=float(predicted['cap'][30]),
                    centered=float(predicted['centered'][30]), directional=float(predicted['directional'][30, ti]),
                    measured_frobenius=float(access['frobenius'][30]), measured_direction=float(access['mu'][30, ti]),
                    remainder=float(predicted['remainder'][30])),
                spectral_mass_max_excess=float(max(0., np.max(mass[resolved_cdf]-possible_cdf[resolved_cdf])))))
            curves[f'g{gamma:g}_{target}_actual_cdf'] = actual_cdf
        print(f'common gamma {gamma:g}: {time.monotonic()-started:.1f}s', flush=True)
    return rows


def heterogeneous(checks, curves):
    g = core.geometry(128)
    x = np.linspace(-1, 1, 16*128+1)
    w = g.width
    target_names = ['sine_mix_2_6_10', 'runge_25']
    y = np.column_stack([f.target(x, name) for name in target_names])/np.sqrt(len(x))
    degrees = np.array([0, 2, 4, 8, 12, 16, 20, 24, 28, 30, 32, 40, 48, 64, 96, 128])
    thresholds = np.r_[0., np.geomspace(1e-24, 1., 481)]
    cases = [('common_4', np.full(w, 4.), [4.]),
             ('dispersed_1_to_7', np.linspace(1, 7, w), [4., 7.])]
    for count in [1, 4, 8]:
        for placement in ['spread', 'clustered']:
            if count == 1 and placement == 'clustered':
                continue
            indices = (np.array([w//2]) if count == 1 else
                np.linspace(0, w-1, count).round().astype(int) if placement == 'spread' else
                np.arange(w//2-count//2, w//2-count//2+count))
            slopes = np.full(w, (4*w-64*count)/(w-count))
            slopes[indices] = 64
            cases.append((f'mean4_outliers{count}_{placement}', slopes, [4., 64.]))
    inactive = np.zeros(w); inactive[np.argmin(np.abs(g.centers))] = 64
    cases.append(('inactive_neurons_control', inactive, [0., 64.]))
    rows = []
    for name, slopes, caps in cases:
        started = time.monotonic()
        j = core.design(x, g.centers, slopes)
        local_y = y; names = target_names
        if name == 'inactive_neurons_control':
            local_y = np.column_stack([y, j[:, 1+np.argmax(slopes)]])
            names = [*target_names, 'exception_feature']
        u, singular, _ = svd(j, full_matrices=False)
        values = singular**2; curvature = values[0]
        loadings = u.T@local_y/np.linalg.norm(local_y, axis=0)
        relative_noise = (64*np.finfo(float).eps*np.sqrt(j.size))**2
        mean_upper = mean_spectrum_bound(w, np.mean(slopes), np.arange(129),
                                         np.geomspace(.05, max(64., np.sum(slopes)), 201))
        eigen_resolved = values > curvature*relative_noise
        assert np.all(values[eigen_resolved] <= mean_upper[eigen_resolved]*(1+1e-9))
        checks['mean_only_eigenvalue_comparisons'] += int(eigen_resolved.sum())
        curves[f'hetero_{name}_mean_eigenvalue_upper'] = mean_upper
        curves[f'hetero_{name}_eigenvalues'] = values/curvature
        curves[f'hetero_{name}_slopes'] = slopes
        cap_rows = []
        for cap in caps:
            bound = distribution_bound(slopes, degrees, cap)
            actual_tail = np.array([values[r:].sum() for r in bound['rank']])
            resolved = actual_tail > curvature*relative_noise
            assert np.all(actual_tail[resolved] <= bound['bound'][resolved]*(1+1e-9))
            checks['heterogeneous_spectral_comparisons'] += int(resolved.sum())
            high = j[:, 1:][:, slopes > cap]
            tails, resolved_span = exceptional_target_tails(x, high, local_y, degrees)
            l_star = max(1., np.mean(np.abs(x) >= .75)**2
                         * np.sum(np.tanh(slopes[np.abs(g.centers) <= .5]/4)**2))
            assert l_star <= curvature*(1+1e-12)
            target_rows = []
            for ti, target in enumerate(names):
                delta = np.where(resolved_span, tails[:, ti], 0.)
                mass = target_cdf_bound(delta, bound['bound'], thresholds, curvature)
                actual_cdf = measured_cdf(values, loadings[:, ti]**2, thresholds, curvature)
                possible = actual_cdf+max(0., 1-np.sum(loadings[:, ti]**2))
                resolved_mass = thresholds > 10*relative_noise
                assert np.all(mass[resolved_mass] <= possible[resolved_mass]+1e-8), (name, cap, target)
                checks['heterogeneous_mass_comparisons'] += int(resolved_mass.sum())
                keep = singular > 1e-14*singular[0]
                spectral = spectral_tail_bound(values[keep], loadings[keep, ti]**2, .5/curvature, .01)
                time_bound = cdf_time_bound(thresholds, mass)
                geometric = cdf_time_bound(thresholds, target_cdf_bound(delta, bound['bound'], thresholds, l_star))
                target_rows.append(dict(target=target, bound=time_bound, geometric_bound=geometric,
                    spectral_jensen_unexecuted=spectral,
                    delta_at_30=float(delta[np.flatnonzero(degrees == 30)[0]]),
                    largest_certified_mass=float(mass.max())))
            cap_rows.append(dict(cap=cap, exceptions=bound['exceptions'],
                mean_count_upper=min(w, int(np.floor(np.sum(slopes)/cap))) if cap else w,
                resolved_span_witnesses=int(resolved_span.sum()), total_span_witnesses=len(degrees),
                degree=degrees.tolist(), rank=bound['rank'].tolist(),
                spectral_tail_bound=bound['bound'].tolist(), measured_tail=actual_tail.tolist(),
                targets=target_rows))
        rows.append(dict(case=name, n=128, width=w, mean=float(np.mean(slopes)),
            median=float(np.median(slopes)), maximum=float(np.max(slopes)), L=float(curvature),
            mean_only_bound_at_mode_40=float(mean_upper[39]), actual_eigenvalue_40=float(values[39]),
            geometry_hash=core.array_hash(g.centers), slope_hash=core.array_hash(slopes), caps=cap_rows))
        print(f'heterogeneous {name}: {time.monotonic()-started:.1f}s', flush=True)
    return rows


def refinement_check(rows):
    """Increase both polynomial degree and pole count; report sensitivity."""
    cfg = f.config(); g = core.geometry(cfg['n'])
    x = np.linspace(-1, 1, cfg['samples_per_cell']*cfg['n']+1)
    y = f.target(x, 'sine_mix_2_6_10')/np.sqrt(len(x))
    thresholds = np.geomspace(1e-32, 1., 641)
    output = []
    for gamma in [4, 16, 64, 96]:
        old = next(r for r in rows if r['gamma'] == gamma and r['target'] == 'sine_mix_2_6_10')
        refined = analytic_access(x, g.centers, gamma, y, cfg['k_max'], 1024, 256)
        mass = target_cdf_bound(refined['tails'][:, 0], refined['directional'][:, 0], thresholds, old['L'])
        bound = cdf_time_bound(thresholds, mass)
        if old['first_hit'] is not None:
            assert bound['bound'] <= old['first_hit']
        output.append(dict(gamma=gamma, degree=1024, pole_pairs=256, refined_bound=bound,
            original_bound=old['bounds']['center_direction_given_step'],
            refined_directional_at_30=float(refined['directional'][30, 0]),
            refined_remainder_at_30=float(refined['remainder'][30])))
    return output


def previous_geometric_audit(root, record):
    """Persist the earlier 73-case first-hit check, including width controls."""
    previous = json.loads(record(root/'refinements/gamma_mechanism.json').read_text())
    cases = {}; hits = {}; ratios = []; censored = 0; excluded = 0
    for n in sorted({r['n'] for r in previous['rows']}):
        folder = root/'training'/f'N{n}_raw_gd'
        cases[n] = json.loads(record(folder/'case.json').read_text())
        with np.load(record(folder/'hitting_audit.npz')) as data:
            hits[n] = data['first'].copy()
    for row in previous['rows']:
        case = cases[row['n']]
        gi = case['gammas'].index(row['gamma'])
        ti = next(i for i, column in enumerate(case['columns']) if column['target'] == row['target'])
        hit = int(hits[row['n']][gi, ti, 0]); bound = row['c2_without_measured_curvature']
        if hit >= 0:
            assert bound['bound'] is not None and bound['bound'] <= hit
            ratios.append(hit/bound['bound'])
        else:
            censored += 1
            excluded += int(bound['log10_bound'] is not None
                            and bound['log10_bound'] > np.log10(200000))
    return dict(cases=len(previous['rows']), reached=len(ratios), censored=censored,
        censored_with_bound_above_budget=excluded, median_slack=float(np.median(ratios)),
        min_slack=float(min(ratios)), max_slack=float(max(ratios)))


def plot(root, rows, hetero, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for gamma, color in [(4, '#346e9f'), (16, '#c78325'), (64, '#34835f')]:
        k = np.arange(len(curves[f'g{gamma}_cap']))
        actual = curves[f'g{gamma}_spectrum_tail']
        shown = actual > 1e-20
        axes[0].semilogy(k[shown], actual[shown], color=color, label=f'gamma {gamma}: measured')
        axes[0].semilogy(k, curves[f'g{gamma}_cap'], '--', color=color, alpha=.8)
        axes[0].semilogy(k, curves[f'g{gamma}_centered'], ':', color=color)
    axes[0].set(xlim=(0, 100), ylim=(1e-16, 1e3), xlabel='Polynomial degree k',
                ylabel='Spectral tail / upper bound', title='(a) Slopes constrain the spectrum')
    axes[0].legend(fontsize=7)
    axes[0].text(.97, .66, 'Solid: measured tail\nDashed: cap bound\nDotted: center bound',
                 transform=axes[0].transAxes, ha='right', fontsize=7,
                 bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    gamma = 16; key = f'g{gamma}_sine_mix_2_6_10_'; s = curves['thresholds']
    for suffix, label, style in [('actual_cdf', 'Measured target mass', '-'),
            ('cap_given_step', 'Uniform cap', '--'),
            ('center_direction_given_step', 'Center + target bound', ':')]:
        axes[1].loglog(s, curves[key+suffix], style, label=label)
    axes[1].axhline(1e-4, color='gray', lw=1)
    axes[1].set(xlim=(1e-10, 1), ylim=(1e-7, 1.1), xlabel='Normalized eigenvalue threshold',
                ylabel='Target energy in slow modes', title='(b) Gamma 16: required target mass')
    axes[1].legend(fontsize=7)
    axes[1].text(.04, .07, 'Uniform-cap witness gives zero mass here.',
                 transform=axes[1].transAxes, fontsize=7, color='#666666')
    primary = [r for r in rows if r['target'] == 'sine_mix_2_6_10']
    gammas = [r['gamma'] for r in primary]
    for method, label, style in [('cap_given_step', 'Uniform-cap CDF', '--'),
            ('center_direction_given_step', 'Center + target CDF', 'o-'),
            ('spectral_jensen_diagnostic', 'Measured spectral Jensen', ':')]:
        axes[2].semilogy(gammas, [10**min(20., r['bounds'][method]['log10_bound']) for r in primary],
                         style, label=label, markersize=3)
    reached = [r for r in primary if r['first_hit'] is not None]
    censored = [r for r in primary if r['first_hit'] is None]
    axes[2].semilogy([r['gamma'] for r in reached], [r['first_hit'] for r in reached],
                    'kx', label='Executed hit')
    axes[2].semilogy([r['gamma'] for r in censored], [r['budget'] for r in censored],
                    '^', color='black', fillstyle='none', label='No hit by 200k')
    axes[2].set(xscale='log', ylim=(1, 1e14), xlabel='Frozen slope gamma',
                ylabel='Updates to 1% error', title='(c) Necessary times versus actual hits')
    axes[2].legend(fontsize=6.5)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(root/f'refinements/slope_spectrum.{ext}', dpi=180)
    plt.close(fig)


def analyze(root):
    started = time.monotonic(); sources = {}; curves = {}
    checks = {k: 0 for k in ['spectral_tail_comparisons', 'center_frobenius_comparisons',
        'center_directional_comparisons', 'target_mass_comparisons', 'reached_cases',
        'censored_cases', 'executed_checkpoint_comparisons', 'heterogeneous_spectral_comparisons',
        'heterogeneous_mass_comparisons', 'mean_only_eigenvalue_comparisons']}

    def record(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    rows = common_slope(root, record, checks, curves)
    hetero = heterogeneous(checks, curves)
    refinement = refinement_check(rows)
    previous = previous_geometric_audit(root, record)
    slack = {}
    for name in rows[0]['bounds']:
        ratios = [r['first_hit']/r['bounds'][name]['bound'] for r in rows if r['first_hit'] is not None]
        slack[name] = dict(min=float(min(ratios)), median=float(np.median(ratios)), max=float(max(ratios)))
    core.save_arrays(root/'refinements/slope_spectrum_curves.npz', **curves)
    code = [Path(__file__).with_name(name) for name in [Path(__file__).name,
            'slope_spectrum.py', 'core.py', 'full_core.py', 'mechanism.py', 'tighten.py', 'full_config.yaml']]
    result = dict(scope='post-hoc raw readout analysis; no new training; CPU only',
        numerical_status='FP64 estimates and resolution monitors, not interval enclosures',
        prediction_inputs='slopes, centers, samples, targets; either geometric L lower bound or prescribed eta=.5/L',
        source_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        validation_inputs_sha256=sources, checks=checks, reached_case_slack=slack,
        common_slope=rows, heterogeneous=hetero, truncation_refinement=refinement,
        previous_geometric_first_hit_audit=previous,
        seconds=time.monotonic()-started)
    core.write_json(root/'refinements/slope_spectrum.json', result)
    plot(root, rows, hetero, curves)
    print(json.dumps(dict(checks=checks, slack=slack, seconds=result['seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)
