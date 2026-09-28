"""Evaluate the simplified Theorem 1 on the archived common-slope experiment.

This is an FP64 diagnostic, not an interval certificate. It evaluates the
actual analytic bound, records its slack, and preserves the exact low-wave
span through an algebraically equivalent, better-conditioned basis. It does
not train, replace the bound by measured kernel action, or generate prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import eigh, svd
from scipy.special import eval_legendre, hyp2f1
import numpy as np

from . import core, gamma_filter as gf
from .finite_gamma_gram import first_hit
from .pi_brief_figure import COLORS
from .review_figures import DEFAULT, GAMMAS


def low_span(x, k, half_period=8., independent=False):
    """Span of bias and first k odd Fourier harmonics, without rank truncation.

    Odd waves span odd polynomials in sin(theta). Even waves span cos(theta)
    times even polynomials. Subtracting the first k binomial terms from
    1/cos(theta) replaces the almost dependent bias by its scaled tail:
    cos(theta)*u**(2*k)*2F1(1,k+1/2;k+1;sin(theta)**2).
    The scale is nonzero, so the span is unchanged. See the note's diagnostic.
    An independent long-double recurrence checks special-function evaluation.
    """
    if k == 0:
        return np.ones((len(x), 1))/np.sqrt(len(x)), 1.
    dtype = np.longdouble if independent else np.float64
    theta = np.arccos(dtype(-1))*x.astype(dtype)/half_period
    u = np.sin(theta)/np.sin(np.max(np.abs(theta)))
    z = np.sin(theta)**2
    cosine = np.cos(theta)
    if independent:
        polynomials = [np.ones_like(u), u]
        for degree in range(2, 2*k):
            polynomials.append(((2*degree-1)*u*polynomials[-1]
                                -(degree-1)*polynomials[-2])/degree)
        term = np.ones_like(u)
        tail = term.copy()
        for n in range(1, 80):
            term *= (dtype(k)+n-dtype('.5'))/(dtype(k)+n)*z
            tail += term
    else:
        polynomials = [eval_legendre(degree, u) for degree in range(2*k)]
        tail = hyp2f1(1, k+.5, k+1, z)
    columns = [polynomials[2*i+1] for i in range(k)]
    columns += [cosine*polynomials[2*i] for i in range(k)]
    columns += [cosine*u**(2*k)*tail]
    b = np.asarray(np.column_stack(columns), dtype=float)
    b /= np.linalg.norm(b, axis=0)
    q, singular, _ = svd(b, full_matrices=False)
    return q, float(singular[0]/singular[-1])


def largest_eigenvalue(matrix):
    n = len(matrix)
    return float(eigh(matrix, subset_by_index=[n-1, n-1], eigvals_only=True)[0])


def collect(folder):
    root = folder.parent.parent
    sources = {}

    def source(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    arrays = np.load(source(root/'common/N512/arrays.npz'))
    x, centers, targets = [arrays[key] for key in ['x_train', 'centers', 'y_train']]
    summary = json.loads(source(folder/'summary.json').read_text())
    assert summary['targets'][0] == 'sine_mix_2_6_10'
    for key, values in [('grid_hash', x), ('centers_hash', centers), ('target_hash', targets)]:
        assert core.array_hash(values) == summary[key]
    y = targets[:, 0]
    norm = np.linalg.norm(y)
    geom = gf.geometry(x, centers, 2048)
    assert np.array_equal(x, np.linspace(-1, 1, 8193))
    period_points = int(round(2*geom['half_period']/(x[1]-x[0])))
    assert period_points == 65536 and 2*geom['harmonics']-1 < period_points/2
    # Odd harmonics are antiperiodic over T. On a full T-grid their sin/cos
    # columns are orthogonal with norm squared P/(4*m). Our rows are a subset.
    f_norm_sq_upper = period_points/(4*len(x))
    witness = np.exp(-.5*(x/.08)**2)*np.cos(100*x)
    witness /= np.linalg.norm(witness)
    correlations = geom['f'].T@witness
    models = {}
    for gamma in GAMMAS:
        dictionary = next(d for d in summary['dictionaries'] if d['gamma'] == gamma)
        forecast = np.load(source(folder/f'g{gamma}_Q2048.npz'))
        reference = np.load(source(folder/f'reference_g{gamma}.npz'))
        j = core.design(x, centers, gamma)
        assert core.array_hash(j) == dictionary['matrix_hash']
        remainder = gf.feature_remainder(gamma, 2048, 8., geom['radius'])
        delta = np.sqrt(len(centers))*sum(remainder.values())
        hit = first_hit(reference)[0]
        assert hit == dictionary['executed_hits'][0]
        models[gamma] = dict(j=j, eta=dictionary['eta'],
            delta=(2*forecast['singular'][0]+delta)*delta,
            reference=reference, executed_hit=hit)
    rows = []
    for k in range(13):
        basis, condition = low_span(x, k)
        check, _ = low_span(x, k, independent=True)
        assert condition < 2e6
        projector_gap = np.linalg.norm(basis-check@(check.T@basis), 'fro')
        assert projector_gap < 1e-8
        alpha = float(np.linalg.norm(y-basis@(basis.T@y))/norm)
        alpha_check = float(np.linalg.norm(y-check@(check.T@y))/norm)
        start = 1+2*k
        ch = geom['c'][start:]
        c_norm_sq = largest_eigenvalue(ch.T@ch)
        f_lower = float(np.sum(correlations[start:]**2))
        assert abs(f_lower/f_norm_sq_upper-1) < 1e-12
        b_upper = f_norm_sq_upper*c_norm_sq
        omega = float(geom['omega'][k])
        cases = []
        for gamma, model in models.items():
            a = model['eta']*(b_upper*float(gf.multiplier(gamma, omega))**2+model['delta'])
            rj = model['j']-basis@(basis.T@model['j'])
            beta = model['eta']*largest_eigenvalue(rj.T@rj)
            rj_check = model['j']-check@(check.T@model['j'])
            beta_check = model['eta']*largest_eigenvalue(rj_check.T@rj_check)
            assert beta <= a and abs(beta_check/beta-1) < 1e-6
            ordered_rates = np.sort(model['reference']['rates'])[::-1]
            next_rate = float(ordered_rates[1+2*k])
            assert next_rate <= beta*(1+1e-8)
            cases.append(dict(gamma=gamma, A_upper=a, measured_restricted_rate=beta,
                action_bound_ratio=a/beta, analytic_kernel_error=float(model['delta']),
                measured_next_rate=next_rate,
                largest_rate=float(ordered_rates[0]), spectral_scale_ratio=float(ordered_rates[0]/a),
                independent_action_relative_gap=abs(beta_check/beta-1)))
        rows.append(dict(low_harmonics=k, omega=omega, rank=1+2*k, alpha=alpha,
            scaled_basis_condition=condition, independent_projector_gap=float(projector_gap),
            independent_alpha_gap=abs(alpha-alpha_check), B_upper=b_upper,
            F_norm_sq_lower=f_lower, F_norm_sq_upper=f_norm_sq_upper, cases=cases))
    rate_grid = np.unique(np.r_[np.geomspace(1e-10, .999999, 1025), 1e-6])
    time_grid = np.geomspace(1e-10, .999999, 16385)
    spectra = []
    for gamma, model in models.items():
        alphas = np.array([r['alpha'] for r in rows])
        upper = np.array([next(c['A_upper'] for c in r['cases'] if c['gamma'] == gamma) for r in rows])
        guarantee = np.max(np.maximum(0., alphas[:, None]-np.sqrt(upper[:, None]/rate_grid))**2, axis=0)
        reference = model['reference']
        rates, weights, floor = reference['rates'], reference['weights'][:, 0], float(reference['floor'][0])
        observed = np.array([floor+weights[rates <= t].sum() for t in rate_grid])
        assert np.all(guarantee <= observed+1e-10)
        amplitude = alphas[:, None]-np.sqrt(upper[:, None]/time_grid)
        valid = amplitude > .01
        raw_time = np.zeros_like(amplitude)
        raw_time[valid] = (np.log(amplitude/.01, where=valid, out=np.zeros_like(amplitude))
                          /(-np.log1p(-time_grid)))[valid]
        ki, ti = np.unravel_index(np.argmax(raw_time), raw_time.shape)
        necessary = int(np.ceil(raw_time[ki, ti]))
        assert amplitude[ki, ti]*(1-time_grid[ti])**(necessary-1) > .01
        spectra.append(dict(gamma=gamma, rate_cutoffs=rate_grid.tolist(),
            measured_slow_mass=observed.tolist(), theorem_slow_mass=guarantee.tolist(),
            first_possible_positive_cutoff=float(np.min(upper/alphas**2)),
            theorem_necessary_updates=necessary, selected_low_harmonics=int(ki),
            selected_rate_cutoff=float(time_grid[ti]),
            true_spectrum_hit=model['executed_hit'], executed_hit=model['executed_hit']))
    return dict(target=summary['targets'][0], gammas=GAMMAS, low_harmonics=list(range(13)),
        half_period=8, harmonics=2048, epsilon=.01, action_panel_low_harmonics=12,
        rows=rows, spectra=spectra, source_sha256=sources,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='FP64 evaluation of simplified Theorem 1, original-tanh restricted action and spectrum; no interval certificate for these new quantities.',
        selection='Exploratory cutoffs k=0..12; action panel uses k=12. Temporal bound maximized on 16385 rate cutoffs; no claim of global optimization.',
        geometry_bound='Full antiperiod T-grid bounds ||F_H||^2 by P/(4m); a localized wave attains this bound to 1e-12 relative in the diagnostic.',
        projection='All 2k+1 directions retained using exact-span polynomial/hypergeometric re-expression; independent long-double recurrence checked.',
        gpu_hours=0)


def plot(data, output):
    ink, gray = '#263442', '#78838C'
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10,
        'axes.titlelocation': 'left', 'axes.titlepad': 10, 'text.color': ink,
        'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9AA3AA', 'axes.linewidth': .6, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.7))
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.79, wspace=.40)
    action = data['rows'][data['action_panel_low_harmonics']]
    ax = axes[0]
    ax.loglog(GAMMAS, [c['A_upper'] for c in action['cases']], 's--', color='#9B6B37', label='Theorem upper bound')
    ax.loglog(GAMMAS, [c['measured_restricted_rate'] for c in action['cases']], 'o-', color=COLORS[0], label='Measured restricted rate')
    ax.set(title='A  Restricted learning strength', xlabel=r'Common slope $\gamma$',
           ylabel='Maximum normalized kernel action', ylim=(1e-6, .02))
    ax.set_xticks(GAMMAS, labels=[str(g) for g in GAMMAS])
    ax.minorticks_off()
    ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    ax.text(.03, .97, r'$\Omega=25\pi/8$', transform=ax.transAxes, va='top', fontsize=7.5)
    ax.text(.03, .70, 'Target norm outside: 16.83%', transform=ax.transAxes, va='top', fontsize=7.5)
    ax = axes[1]
    for row, color in zip(data['spectra'], COLORS):
        rates = np.array(row['rate_cutoffs'])
        actual = np.array(row['measured_slow_mass'])
        lower = np.array(row['theorem_slow_mass'])
        ax.loglog(rates, actual, color=color, lw=1.6)
        ax.loglog(rates, np.where(lower > 0, lower, np.nan), '--', color=color, lw=1.4)
    ax.axvline(1e-6, color=gray, ls=':', lw=.8)
    ax.set(title='B  Slow target energy', xlabel=r'Per-update rate cutoff $t$',
           ylabel='Fraction of target energy', xlim=(1e-8, 1), ylim=(1e-7, 1.2))
    ax.text(.035, .965, 'Theorem bound = 0\n'+r'for $t\leq0.072$ (scan).', transform=ax.transAxes, va='top', fontsize=7.5)
    ax.legend(handles=[Line2D([], [], color=ink, label='True-kernel spectrum'),
                       Line2D([], [], color=ink, ls='--', label='Theorem 1 lower bound')],
              loc='lower right', fontsize=7.5, frameon=False)
    ax = axes[2]
    ax.loglog(GAMMAS, [r['true_spectrum_hit'] for r in data['spectra']], '-', color=ink, lw=1.6, label='True-spectrum prediction')
    for row, color in zip(data['spectra'], COLORS):
        ax.scatter(row['gamma'], row['executed_hit'], facecolors='white', edgecolors=color, s=35, zorder=3)
    ax.loglog(GAMMAS, [r['theorem_necessary_updates'] for r in data['spectra']], 's--', color='#9B6B37', label='Theorem 1 necessary time')
    ax.set(title='C  Necessary training time', xlabel=r'Common slope $\gamma$', ylabel='Updates to 1% residual', ylim=(1, 5e7))
    ax.set_xticks(GAMMAS, labels=[str(g) for g in GAMMAS])
    ax.minorticks_off()
    ax.legend(handles=[Line2D([], [], color=ink, label='True-spectrum prediction'),
        Line2D([], [], color=ink, marker='o', lw=0, markerfacecolor='white', label='Executed GD'),
        Line2D([], [], color='#9B6B37', ls='--', marker='s', label='Theorem 1 necessary time')],
        loc='upper right', fontsize=7.5, frameon=False)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=rf'$\gamma={g}$') for g, c in zip(GAMMAS, COLORS)],
               loc='upper center', ncol=4, frameon=False)
    for ax in axes:
        ax.tick_params(length=3, pad=3)
    fig.savefig(output/'theorem1_spectral_probe.png', dpi=300)
    fig.savefig(output/'theorem1_spectral_probe.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = collect(args.source)
    output = args.output or args.source
    output.mkdir(parents=True, exist_ok=True)
    plot(data, output)
    (output/'theorem1_spectral_probe.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{k: row[k] for k in ['gamma', 'first_possible_positive_cutoff',
        'theorem_necessary_updates', 'true_spectrum_hit']} for row in data['spectra']], indent=2))


if __name__ == '__main__':
    main()
