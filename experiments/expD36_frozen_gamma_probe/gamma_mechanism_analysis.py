"""Explicit gamma mechanism on one fixed, target-relevant sample direction.

This is a retrospective decomposition diagnostic, not an eigenvalue or timing
certificate. Selection uses initial target energy and reference features only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from .core import array_hash, design
from .gamma_ratio_bound import increment_action, kernel_increment
from .uniform_periodic import attenuation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'


def select_direction(features, target, ratio_cutoff=2e-6, singular_cutoff=1e-14):
    """Largest initial target energy among retained positive slow reference modes."""
    u, singular, _ = svd(features, full_matrices=False, lapack_driver='gesvd')
    rho = (singular/singular[0])**2
    weights = (u.T@target)**2/(target@target)
    eligible = np.flatnonzero((rho > 0)&(rho <= ratio_cutoff)
                             &(singular > singular_cutoff*singular[0]))
    if not len(eligible):
        raise ValueError('No retained positive reference mode satisfies the cutoff.')
    index = int(eligible[np.argmax(weights[eligible])])
    direction = u[:, index].copy()
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    rate = singular[index]**2
    residual = features@(features.T@direction)-rate*direction
    return direction, dict(rank=index+1, rho=float(rho[index]),
        initial_target_energy=float(weights[index]), eigenvalue=float(rate),
        reference_largest_eigenvalue=float(singular[0]**2),
        ratio_cutoff=ratio_cutoff, relative_singular_cutoff=singular_cutoff,
        eligible_count=len(eligible), retained_count=int(np.count_nonzero(singular > singular_cutoff*singular[0])),
        eigenvector_residual_norm=float(np.linalg.norm(residual)),
        sample_mean=float(direction.mean()), norm=float(np.linalg.norm(direction)))


def directional_response(x, centers, direction, reference_features, gamma, kernel_max,
                         reference_gamma=8.):
    """Actual finite action versus baseline plus the exact explicit bulk gain."""
    h = centers[1]-centers[0]
    features = design(x, centers, gamma)
    v = np.asarray(direction)
    reference_action = float(np.sum((reference_features.T@v)**2))
    actual = float(np.sum((features.T@v)**2))
    action = increment_action(x, v, reference_gamma, gamma, h)
    gain = float(v@action)
    predicted_bulk = reference_action+gain
    correction = actual-predicted_bulk
    indices = np.unique(np.linspace(0, len(x)-1, min(11, len(x)), dtype=int))
    direct = kernel_increment(x[indices, None]-x[None, :], reference_gamma, gamma, h, len(x))@v
    return dict(gamma=gamma, actual_action=actual, reference_action=reference_action,
        reference_action_plus_gain=predicted_bulk, explicit_gain=gain,
        finite_correction=correction, correction_fraction=correction/actual,
        largest_eigenvalue=kernel_max, normalized_actual_action=actual/kernel_max,
        normalized_reference_plus_gain=predicted_bulk/kernel_max,
        direct_row_max_absolute_error=float(np.max(np.abs(direct-action[indices]))),
        direct_row_indices=indices.tolist(), matrix_sha256=array_hash(features))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT/'refinements/gamma_optimizer_access')
    args = parser.parse_args()
    array_path = args.source/'common/N512/arrays.npz'
    diagnostics_path = args.source/'refinements/gamma_optimizer_access/analysis.json'
    arrays = np.load(array_path)
    x, centers, target = arrays['x_train'], arrays['centers'], arrays['y_train'][:, 0]
    diagnostics = json.loads(diagnostics_path.read_text())
    reference = design(x, centers, 8.)
    direction, selection = select_direction(reference, target)
    selection.update(reference_gamma=8, target='sine_mix_2_6_10',
        rule='Maximum initial squared target projection among gamma8 modes with0<rho<=2e-6 and sigma>1e-14*sigma1; descending-spectrum rank.',
        retrospective_status='Prescribed diagnostic rule applied retrospectively to the existing archive; no GD or Adam trajectory outcomes enter selection.',
        interpretation='The direction is held fixed at every gamma; it is an eigenvector only at the reference gamma8, not asserted to remain one.')
    gammas = [8, 12, 16, 64]
    rows = []
    for gamma in gammas:
        diagnostic = next(r for r in diagnostics['diagnostics'] if r['gamma'] == gamma)
        row = directional_response(x, centers, direction, reference, gamma,
                                   diagnostic['largest_eigenvalue'])
        assert row['matrix_sha256'] == diagnostic['matrix_hash']
        rows.append(row)
    omega = np.unique(np.r_[0., np.geomspace(.1, 256., 401), [2*np.pi, 6*np.pi, 10*np.pi]])
    target_omega = np.array([2*np.pi, 6*np.pi, 10*np.pi])
    frequency = dict(units='Angular frequency, radians per input-coordinate unit',
        omega=omega.tolist(), target_omega=target_omega.tolist(),
        target_coefficients=[1., .5, .25], rows=[dict(gamma=g,
            multiplier=attenuation(omega, g).tolist(),
            multiplier_squared=(attenuation(omega, g)**2).tolist(),
            target_multiplier_squared=(attenuation(target_omega, g)**2).tolist()) for g in gammas])
    paths = [array_path, diagnostics_path, Path(__file__),
        Path(__file__).with_name('core.py'), Path(__file__).with_name('gamma_ratio_bound.py'),
        Path(__file__).with_name('uniform_periodic.py')]
    result = dict(gammas=gammas, samples=len(x), tanh_features=len(centers),
        normalization='J has bias and tanh columns divided bysqrt(m); v has unit Euclidean norm; target is archived y_train[:,0].',
        selection=selection, rows=rows, frequency=frequency,
        maximum_absolute_correction_fraction=max(abs(r['correction_fraction']) for r in rows),
        scope='Exact finite-kernel decomposition evaluated in FP64. No timing/eigenvalue bounds, no eigenvector-invariance assumption, and no new-gamma eigensolves.',
        largest_eigenvalue_source='Previously measured actual finite-kernel values from analysis.json, used only for optional normalization.',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'mechanism_arrays.npz', x=x, centers=centers,
                        target=target, fixed_direction=direction)
    result['arrays_sha256'] = hashlib.sha256((args.output/'mechanism_arrays.npz').read_bytes()).hexdigest()
    (args.output/'mechanism_analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(selection=selection, rows=rows), indent=2))


if __name__ == '__main__':
    main()
