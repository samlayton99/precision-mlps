"""Finite cross-gamma ratio bounds from explicit Toeplitz gain and corrections.

Run as a module. Numerical resolution checks are diagnostics, not interval
certificates. New full feature spectra are validation references only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, eigh, matmul_toeplitz, solve_triangular, svd

from .core import design

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'


def kernel_increment(distance, gamma_old, gamma_new, spacing, samples):
    """2[d*coth(gamma_old*d)-d*coth(gamma_new*d)]/(h*m)."""
    if not (0 < gamma_old <= gamma_new and spacing > 0 and samples > 0):
        raise ValueError('Require positive geometry and nondecreasing positive slopes.')
    d = np.abs(np.asarray(distance, dtype=float))
    result = np.zeros_like(d)
    if gamma_old == gamma_new:
        return result
    small = gamma_new*d < 1e-3
    z = d[small]
    result[small] = 2*((1/gamma_old-1/gamma_new)
        +(gamma_old-gamma_new)*z**2/3
        -(gamma_old**3-gamma_new**3)*z**4/45
        +2*(gamma_old**5-gamma_new**5)*z**6/945)/(spacing*samples)
    z = d[~small]
    # exp(-t)/(1-exp(-t)) avoids overflow in expm1(t).
    inv_old = np.exp(-2*gamma_old*z)/(-np.expm1(-2*gamma_old*z))
    inv_new = np.exp(-2*gamma_new*z)/(-np.expm1(-2*gamma_new*z))
    result[~small] = 4*z*(inv_old-inv_new)/(spacing*samples)
    return result


def increment_action(x, vectors, gamma_old, gamma_new, spacing):
    x = np.asarray(x)
    if len(x) < 2 or not np.allclose(np.diff(x), x[1]-x[0], rtol=1e-10, atol=1e-14):
        raise ValueError('Toeplitz application requires an increasing uniform sample grid.')
    if x[1] <= x[0]:
        raise ValueError('Sample grid must increase.')
    column = kernel_increment(x-x[0], gamma_old, gamma_new, spacing, len(x))
    return matmul_toeplitz((column, column), vectors, check_finite=True)


def kernel_row_sum_bound(features, block=256):
    """Maximum absolute row sum without storing the full sample kernel."""
    return max(float(np.max(np.sum(np.abs(features[start:start+block]@features.T), axis=1)))
               for start in range(0, len(features), block))


def compressed_ratio_bound(old_block, gain, new_block, denominator):
    """Trial-space theorem, including a legitimate zero when epsilon >= 1."""
    if denominator <= 0:
        raise ValueError('The eigenvalue denominator bound must be positive.')
    sym = lambda a: (a+a.T)/2
    c = sym(old_block+gain)
    r = sym(new_block-c)
    eigen = eigh(c, eigvals_only=True)
    tolerance = 64*np.finfo(float).eps*len(c)*np.linalg.norm(c, 2)
    result = dict(c_min=float(eigen[0]), c_max=float(eigen[-1]),
                  c_resolution_tolerance=float(tolerance))
    if eigen[0] <= tolerance:
        return dict(result, status='unresolved_positive_trial_matrix', lower_ratio=None)
    lower = cholesky(c, lower=True)
    temp = solve_triangular(lower, r, lower=True)
    whitened = solve_triangular(lower, temp.T, lower=True).T
    epsilon = float(np.max(np.abs(eigh(sym(whitened), eigvals_only=True))))
    independent = float(np.max(np.abs(eigh(r, c, eigvals_only=True))))
    defect = np.linalg.norm(lower@whitened@lower.T-r, 'fro')
    discrepancy = abs(epsilon-independent)/max(1., epsilon, independent)
    resolved = discrepancy < 1e-5
    return dict(result, epsilon=epsilon, epsilon_generalized=independent,
        whitening_relative_disagreement=float(discrepancy),
        whitening_reconstruction_frobenius=float(defect),
        lower_ratio=max(0., 1-epsilon)*float(eigen[0])/denominator if resolved else None,
        status=('resolved_zero_bound' if epsilon >= 1 else 'resolved_numerical_bound')
               if resolved else 'unresolved_whitening')


def evaluate_ratios(x, centers, old_features, new_features, gamma_old, gamma_new,
                    ranks=(8, 16, 32, 64), old_svd=None):
    """Evaluate the theorem before computing the new full validation spectrum."""
    x, centers = np.asarray(x), np.asarray(centers)
    h = centers[1]-centers[0]
    if h <= 0 or not np.allclose(np.diff(centers), h, rtol=1e-10, atol=1e-14):
        raise ValueError('Centers must be consecutive and uniformly spaced.')
    if old_features.shape != new_features.shape or len(old_features) != len(x):
        raise ValueError('Both slopes must use the same sample and feature geometry.')
    u, singular, _ = svd(old_features, full_matrices=False, lapack_driver='gesvd') \
        if old_svd is None else old_svd
    ranks = tuple(int(r) for r in ranks)
    if not ranks or min(ranks) < 1:
        raise ValueError('Ranks must be positive integers.')
    maximum = min(max(ranks), len(singular))
    trial = u[:, :maximum]
    gain_action = increment_action(x, trial, gamma_old, gamma_new, h)
    gain = trial.T@gain_action
    old_loading, new_loading = old_features.T@trial, new_features.T@trial
    old_block, new_block = old_loading.T@old_loading, new_loading.T@new_loading
    denominator = kernel_row_sum_bound(new_features)
    # Independent direct rows check FFT Toeplitz action without a dense m*m array.
    rows = np.unique(np.linspace(0, len(x)-1, min(9, len(x)), dtype=int))
    direct = kernel_increment(x[rows, None]-x[None, :], gamma_old, gamma_new, h, len(x))@trial
    action_error = float(np.max(np.abs(direct-gain_action[rows])))
    tol = 10*np.finfo(float).eps*max(old_features.shape)*singular[0]
    records = []
    for rank in ranks:
        record = dict(rank=rank)
        if rank > len(singular):
            records.append(dict(record, status='rank_exceeds_feature_dimension', lower_ratio=None))
            continue
        gap = singular[rank-1]-(singular[rank] if rank < len(singular) else 0.)
        record.update(old_singular_value=float(singular[rank-1]),
                      old_singular_gap=float(gap), old_resolution_tolerance=float(tol),
                      old_ratio=float((singular[rank-1]/singular[0])**2))
        if singular[rank-1] <= tol or gap <= tol:
            record.update(status='unresolved_old_trial_space', lower_ratio=None)
        else:
            record.update(compressed_ratio_bound(old_block[:rank, :rank],
                gain[:rank, :rank], new_block[:rank, :rank], denominator))
        records.append(record)
    # Reference only: no new spectrum enters the candidate bound above.
    reference = svd(new_features, compute_uv=False, lapack_driver='gesvd')**2
    for record in records:
        if record['rank'] <= len(reference):
            actual = float(reference[record['rank']-1]/reference[0])
            record['actual_new_ratio'] = actual
            if record['lower_ratio'] is not None:
                record['fraction_of_actual'] = record['lower_ratio']/actual if actual > 0 else None
                record['bound_gain_over_old'] = record['lower_ratio']/record['old_ratio']
    return dict(gamma_old=gamma_old, gamma_new=gamma_new, samples=len(x),
        width=len(centers), spacing=float(h), denominator_row_sum=denominator,
        reference_largest_eigenvalue=float(reference[0]),
        direct_row_action_max_error=action_error,
        gain_symmetry_defect=float(np.linalg.norm(gain-gain.T, 'fro')),
        trial_orthogonality_defect=float(np.linalg.norm(trial.T@trial-np.eye(maximum), 'fro')),
        records=records, status='FP64 diagnostic evaluations, not interval-certified bounds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT/'common/N512/arrays.npz')
    parser.add_argument('--output', type=Path, default=DEFAULT/'refinements/gamma_optimizer_access')
    args = parser.parse_args()
    arrays = np.load(args.source)
    x, centers = arrays['x_train'], arrays['centers']
    old = design(x, centers, 8.)
    old_svd = svd(old, full_matrices=False, lapack_driver='gesvd')
    results = []
    for gamma in (8., 12., 16., 64.):
        results.append(evaluate_ratios(x, centers, old, design(x, centers, gamma),
                                      8., gamma, old_svd=old_svd))
        print(gamma, [(r['rank'], r['status']) for r in results[-1]['records']], flush=True)
    paths = [args.source, Path(__file__), Path(__file__).with_name('core.py')]
    supplied_note = ROOT/'docs/gamma_slow_learning_alt_view.pdf'
    if supplied_note.exists():
        paths.append(supplied_note)
    output = dict(reference_gamma=8, prescribed_ranks=[8, 16, 32, 64], cases=results,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        theorem='Collaborator Theorem 1: lower new eigenvalue ratio from old trial space, explicit gamma gain, and relative finite correction.',
        scope='Selected numerical eigenvalue-ratio lower bounds. Not a target-specific necessary-time bound; reference spectra are validation only.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'ratio_bounds.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
