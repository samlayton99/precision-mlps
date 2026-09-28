"""V4 current-gamma finite eigenvalue intervals, using the p=0 specialization.

Whole-line mean-zero integral minus explicit absent lattice centers. Analytic
tails and signed-factor compression allowances are distinguished from empirical
quadrature/FP64 checks. No sample-by-sample dense kernel is diagonalized.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, svd

from .core import design, target as target_function

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'


def mean_zero_coordinates(values):
    """Q.T @ values using a Householder reflector; Q has m-1 columns."""
    values = np.asarray(values)
    m = len(values)
    direction = np.full(m, -1/np.sqrt(m)); direction[0] += 1
    direction /= np.linalg.norm(direction)
    return (values-2*np.outer(direction, direction@values))[1:]


def lattice_allowance(x, spacing, gamma):
    theta = np.arctan(np.pi/(2*gamma*spacing))
    exponent = 2*np.pi*theta/(gamma*spacing)
    return float(8*gamma*np.sum((x-np.mean(x))**2)/(3*spacing*len(x)*np.cos(theta)**4)
                 *np.exp(-exponent)/(-np.expm1(-exponent)))


def quadrature_factor(x, spacing, gamma, order=10, padding=20., panel_scale=1.):
    """Positive Gauss factor for the projected whole-line center integral."""
    radius = float(np.max(np.abs(x))+padding/gamma)
    panels = int(np.ceil(2*radius*gamma/panel_scale))
    edges = np.linspace(-radius, radius, panels+1)
    nodes, weights = leggauss(order)
    half = (edges[1]-edges[0])/2
    centers = ((edges[:-1]+edges[1:])[:, None]/2+half*nodes).ravel()
    weights = np.tile(half*weights, panels)
    values = np.tanh(gamma*(x[:, None]-centers))
    projected = mean_zero_coordinates(values)*np.sqrt(weights/(spacing*len(x)))
    tail = float(2*np.exp(-4*padding)/(spacing*gamma))
    return projected, dict(order=order, padding=padding, panel_scale=panel_scale,
        panels=panels, quadrature_columns=len(centers), integration_radius=radius,
        integration_upper_tail=tail)


def exterior_factor(x, centers, gamma, padding=18.):
    spacing = centers[1]-centers[0]
    count = int(np.ceil(padding/(gamma*spacing)))
    absent = np.r_[centers[0]-spacing*np.arange(1, count+1),
                   centers[-1]+spacing*np.arange(1, count+1)]
    # The tested geometries cover the whole sample interval; do not silently
    # omit interior lattice centers in a different geometry.
    if centers[0] > np.min(x) or centers[-1] < np.max(x):
        raise ValueError('This experiment requires centers covering the sample interval.')
    projected = mean_zero_coordinates(np.tanh(gamma*(x[:, None]-absent)))/np.sqrt(len(x))
    left, right = centers[0]-(count+1)*spacing, centers[-1]+(count+1)*spacing
    tail = float(4*(np.exp(-4*gamma*(right-np.max(x)))
                  +np.exp(-4*gamma*(np.min(x)-left)))/(-np.expm1(-4*gamma*spacing)))
    return projected, dict(exterior_count_per_side=count, exterior_lower_tail=tail,
                           exterior_first_unretained_left=left, exterior_first_unretained_right=right)


def padded_eigenvalues(reduced, dimension):
    """Put missing zero eigenvalues between positive and negative reduced ones."""
    reduced = np.sort(np.asarray(reduced))[::-1]
    if len(reduced) > dimension:
        raise ValueError('Reduced spectrum exceeds the restricted dimension.')
    return np.r_[reduced[reduced > 0], np.zeros(dimension-len(reduced)), reduced[reduced <= 0]]


def endpoint_arrays(beta, delta, lower_extra, upper_extra, ell, largest_upper, width):
    """Codimension-one interlacing and ratio normalization, descending indices."""
    m = len(beta)+1
    lower = np.zeros(m); upper = np.zeros(m)
    lower[0], upper[0] = ell, largest_upper
    lower[1:-1] = np.maximum(0., beta[1:]-delta-lower_extra)
    upper[1:] = np.maximum(0., beta+delta+upper_extra)
    lower[width+1:] = 0.; upper[width+1:] = 0.
    rho_lower = lower/largest_upper
    rho_upper = np.minimum(1., upper/ell)
    rho_lower[0] = rho_upper[0] = 1.
    return lower, upper, rho_lower, rho_upper


def error_curve(rates, weights, floor, steps):
    rates = np.asarray(rates)
    return np.sqrt(np.exp(2*np.outer(np.asarray(steps), np.log1p(-rates)))@weights+floor)


def first_crossing(rates, weights, floor, epsilon=.01, cap=10**18):
    def value(step):
        return float(error_curve(rates, weights, floor, [step])[0])
    if value(0) <= epsilon:
        return 0
    if floor >= epsilon**2:
        return None
    high = 1
    while high < cap and value(high) > epsilon:
        high = min(cap, high*2)
    if value(high) > epsilon:
        return None
    low = 0
    while high-low > 1:
        mid = (high+low)//2
        if value(mid) <= epsilon:
            high = mid
        else:
            low = mid
    return high


def calculate(x, centers, gamma, target, eta=None, order=10, padding=20.,
              compression_cutoff=1e-14, arithmetic_factor=64.):
    spacing = centers[1]-centers[0]
    if not np.allclose(np.diff(centers), spacing, rtol=1e-10, atol=1e-14):
        raise ValueError('Consecutive uniform centers required.')
    a, integral = quadrature_factor(x, spacing, gamma, order, padding)
    c, exterior = exterior_factor(x, centers, gamma)
    joined = np.column_stack((a, c))
    u, singular, vt = svd(joined, full_matrices=False, lapack_driver='gesvd')
    keep = singular > compression_cutoff*singular[0]
    count = int(np.count_nonzero(keep))
    tail = float(singular[count]) if count < len(singular) else 0.
    compression = float(2*singular[0]*tail+tail**2)
    signed = np.r_[np.ones(a.shape[1]), -np.ones(c.shape[1])]
    coordinate = singular[keep, None]*vt[keep]
    small = (coordinate*signed)@coordinate.T
    small = (small+small.T)/2
    reduced = eigh(small, eigvals_only=True)[::-1]
    beta = padded_eigenvalues(reduced, len(x)-1)
    arithmetic = float(arithmetic_factor*np.finfo(float).eps*np.max(np.abs(reduced)))
    delta = lattice_allowance(x, spacing, gamma)
    two_sided = delta+compression+arithmetic
    # Reference finite-feature SVD supplies actual target projections and validation
    # eigenvalues. It is not used to construct beta or its remainder allowances.
    features = design(x, centers, gamma)
    reference_u, reference_s, _ = svd(features, full_matrices=False, lapack_driver='gesvd')
    actual = reference_s**2
    if eta is None:
        eta = .5/actual[0]
    if not 0 < eta*actual[0] < 1:
        raise ValueError('This error sandwich assumes a nonoscillatory stable step.')
    constant = np.ones(len(x))/np.sqrt(len(x))
    loading = features.T@constant
    ell = float(loading@loading)
    b = float(np.linalg.norm(mean_zero_coordinates((features@loading)[:, None])))
    restricted_upper = float(beta[0]+two_sided+integral['integration_upper_tail'])
    largest_upper = float((ell+restricted_upper+np.hypot(ell-restricted_upper, 2*b))/2)
    endpoints = endpoint_arrays(beta, two_sided, exterior['exterior_lower_tail'],
        integral['integration_upper_tail'], ell, largest_upper, len(centers))
    lower, upper, rho_lower, rho_upper = endpoints
    target_loading = reference_u.T@target
    weights = target_loading**2/(target@target)
    remainder = float(np.linalg.norm(target-reference_u@target_loading)**2/(target@target))
    closure_gap = float(np.sum(weights)+remainder-1)
    actual_rho = actual/actual[0]
    resolved = actual_rho > 1e-18
    observed_step = eta*actual[0]
    lower_rates = observed_step*rho_lower[:len(actual)].copy()
    upper_rates = observed_step*rho_upper[:len(actual)]
    lower_rates[~resolved] = 0.
    resolved_weights = weights.copy(); resolved_weights[~resolved] = 0.
    necessary = first_crossing(upper_rates, resolved_weights, 0.)
    sufficient = first_crossing(lower_rates, weights, remainder)
    reference_hit = first_crossing(eta*actual, weights, remainder)
    steps = np.unique(np.r_[0, np.geomspace(1, max(10**8, (reference_hit or 10**15)*10), 360).astype(np.int64)])
    lower_curve = error_curve(upper_rates, resolved_weights, 0., steps)
    upper_curve = error_curve(lower_rates, weights, remainder, steps)
    reference_curve = error_curve(eta*actual, weights, remainder, steps)
    violation = np.maximum(rho_lower[:len(actual)]-actual_rho,
                            actual_rho-rho_upper[:len(actual)])
    summary = dict(gamma=gamma, samples=len(x), width=len(centers), spacing=float(spacing),
        eta=float(eta), eta_mu1=float(observed_step), quadrature=integral, exterior=exterior,
        lattice_terms=0, lattice_delta=delta, compression_relative_cutoff=compression_cutoff,
        retained_factor_rank=count, combined_factor_columns=joined.shape[1],
        first_discarded_factor_singular_value=tail, compression_two_sided=compression,
        empirical_arithmetic_factor=arithmetic_factor,
        empirical_arithmetic_allowance=arithmetic, total_two_sided_allowance=two_sided,
        ell=ell, largest_upper=largest_upper, actual_largest=float(actual[0]),
        necessary_updates=necessary, spectral_reference_updates=reference_hit,
        sufficient_updates=sufficient, search_cap=10**18,
        retained_actual_modes=int(np.count_nonzero(resolved)), actual_feature_modes=len(actual),
        unresolved_target_energy=float(np.sum(weights[~resolved])), projection_remainder=remainder,
        target_energy_closure_gap=closure_gap,
        max_resolved_ratio_violation=float(max(0., np.max(violation[resolved]))),
        lower_curve_max_violation=float(max(0., np.max(lower_curve-reference_curve))),
        upper_curve_max_violation=float(max(0., np.max(reference_curve-upper_curve))),
        numerical_status='FP64 checked theorem evaluation; analytic tails/compression separated from empirical arithmetic guard. Quadrature requires refinement; not interval arithmetic.')
    arrays = dict(beta=beta, eigenvalue_lower=lower, eigenvalue_upper=upper,
        rho_lower=rho_lower, rho_upper=rho_upper, actual_eigenvalues=actual,
        actual_rho=actual_rho, actual_target_weights=weights, resolved=resolved,
        steps=steps, lower_error=lower_curve, upper_error=upper_curve, reference_error=reference_curve)
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=['sanity', 'archive'], default='sanity')
    parser.add_argument('--gammas', type=float, nargs='+', default=[4., 8., 16., 64.])
    parser.add_argument('--order', type=int, default=10)
    parser.add_argument('--padding', type=float, default=20.)
    parser.add_argument('--output', type=Path, default=DEFAULT/'refinements/gamma_direct_ratio')
    args = parser.parse_args()
    source_paths = [Path(__file__), Path(__file__).with_name('core.py')]
    supplied_note = ROOT/'docs/gamma_ratio_note_v4.pdf'
    if supplied_note.exists():
        source_paths.append(supplied_note)
    if args.geometry == 'sanity':
        x = np.linspace(-1, 1, 263)
        centers = -1+np.arange(-12, 141)/64
        target = target_function(x, 'sine_mix_2_6_10')/np.sqrt(len(x))
        steps = {}
    else:
        source = DEFAULT/'common/N512/arrays.npz'
        archive = DEFAULT/'refinements/gamma_factorized_kernel/summary.json'
        data = np.load(source)
        x, centers, target = data['x_train'], data['centers'], data['y_train'][:, 0]
        steps = {r['gamma']: r['eta'] for r in json.loads(archive.read_text())['dictionaries']}
        source_paths.extend([source, archive])
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    args.output.mkdir(parents=True, exist_ok=True)
    for gamma in args.gammas:
        summary, arrays = calculate(x, centers, gamma, target, steps.get(gamma), args.order, args.padding)
        summary['source_sha256'] = sources
        summary['geometry'] = args.geometry
        stem = f'{args.geometry}_g{gamma:g}_q{args.order}_p{args.padding:g}'
        np.savez_compressed(args.output/f'{stem}.npz', **arrays)
        summary['arrays_sha256'] = hashlib.sha256((args.output/f'{stem}.npz').read_bytes()).hexdigest()
        (args.output/f'{stem}.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(stem, json.dumps({k: summary[k] for k in ['retained_factor_rank', 'lattice_delta',
            'compression_two_sided', 'empirical_arithmetic_allowance', 'necessary_updates',
            'spectral_reference_updates', 'sufficient_updates', 'max_resolved_ratio_violation']}), flush=True)


if __name__ == '__main__':
    main()
