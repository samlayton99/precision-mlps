"""Reverse gamma comparison: trial-space upper rates and fixed-target delay.

All numerical results are FP64 diagnostics. Candidate selection never reads GD
hitting times. Gamma-dependent projected eigensolves are explicitly disclosed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.linalg import eigh, qr, svd

from .core import design
from .gamma_ratio_bound import increment_action

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'


def delay_bound(alpha, coupling, target_overlap, null_upper, cutoff, epsilon=.01):
    """Sylvester leakage + subspace-angle inequality; all rates include eta.

    alpha bounds ||V* A V||, coupling bounds ||(I-VV*) A V||,
    target_overlap=||V*y||/||y||. A is a PSD contraction.
    """
    if not (0 <= alpha and coupling >= 0 and 0 <= target_overlap <= 1
            and 0 <= null_upper <= 1 and 0 < cutoff < 1):
        raise ValueError('Invalid PSD rate, overlap, floor, or cutoff.')
    leakage = min(1., coupling/(cutoff-alpha)) if cutoff > alpha else 1.
    amplitude = max(0., target_overlap*np.sqrt(max(0., 1-leakage**2))
                    -leakage*np.sqrt(max(0., 1-target_overlap**2)))
    mass = max(0., amplitude**2-null_upper)
    updates = int(np.ceil(np.log(np.sqrt(mass)/epsilon)/(-np.log1p(-cutoff)))) \
        if mass > epsilon**2 else 0
    return dict(cutoff=float(cutoff), leakage=float(leakage), positive_mass=float(mass),
                necessary_updates=updates)


def trial_statistics(features, target, basis, eta):
    """Direct physical-coordinate action; no eigendirections assumed invariant."""
    action = eta*features@(features.T@basis)
    block = basis.T@action
    block = (block+block.T)/2
    outside = action-basis@block
    return dict(alpha=max(0., float(eigh(block, eigvals_only=True)[-1])),
        coupling=float(np.linalg.norm(outside, 2)),
        target_overlap=min(1., float(np.linalg.norm(basis.T@target)/np.linalg.norm(target))))


def scan_space(features, target, basis, eta, null_upper, kind):
    """Keep full projected action and its off-space coupling, scanning fixed bands."""
    action = eta*features@(features.T@basis)
    block = basis.T@action; block = (block+block.T)/2
    residual = action-basis@block
    _, small_residual = qr(residual, mode='economic')
    if kind == 'projected_spectrum':
        eigenvalues, rotation = eigh(block)
        action_block = rotation.T@block@rotation
        remainder = small_residual@rotation
        loading = rotation.T@(basis.T@target)/np.linalg.norm(target)
        index_sets = [np.flatnonzero(eigenvalues <= cutoff)
                      for cutoff in np.geomspace(1e-10, .05, 49)]
    elif kind == 'reference_tail':
        rotation = np.eye(len(block)); action_block = block
        remainder = small_residual; loading = basis.T@target/np.linalg.norm(target)
        index_sets = [np.arange(start, len(block)) for start in [2, 4, 8, 16, 24, 32, 48, 64]
                      if start < len(block)]
    else:
        raise ValueError(kind)
    records = []
    best = None
    seen = set()
    for indices in index_sets:
        key = tuple(indices)
        if not len(indices) or key in seen:
            continue
        seen.add(key)
        outside = np.setdiff1d(np.arange(len(block)), indices)
        trial = action_block[np.ix_(indices, indices)]
        alpha = max(0., float(eigh(trial, eigvals_only=True)[-1]))
        coupling = float(np.linalg.norm(np.vstack((remainder[:, indices],
            action_block[np.ix_(outside, indices)])), 2))
        overlap = min(1., float(np.linalg.norm(loading[indices])))
        # An explicitly heuristic arithmetic sensitivity check, not a certificate.
        guard = 64*np.finfo(float).eps*len(block)*max(np.linalg.norm(block, 2),
                                                          np.linalg.norm(action, 'fro'))
        for multiplier in [0., 1., 10.]:
            best_band = max((delay_bound(alpha+multiplier*guard, coupling+multiplier*guard,
                overlap, null_upper, b) for b in np.geomspace(1e-10, .5, 121)),
                key=lambda row: row['necessary_updates'])
            record = dict(best_band, dimension=len(indices), alpha=alpha, coupling=coupling,
                target_overlap=overlap, arithmetic_guard=float(guard), guard_multiplier=multiplier,
                selected_indices=indices.tolist())
            records.append(record)
            if multiplier == 1. and (best is None or record['necessary_updates'] > best['necessary_updates']):
                best = record
    selected = basis@rotation[:, best['selected_indices']]
    direct = trial_statistics(features, target, selected, eta)
    best = dict(best, direct_check=direct)
    return dict(kind=kind, parent_dimension=basis.shape[1], best=best, candidates=records), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT/'refinements/gamma_optimizer_access')
    args = parser.parse_args()
    array_path = args.source/'common/N512/arrays.npz'
    archive_path = args.source/'refinements/gamma_factorized_kernel/summary.json'
    floor_path = args.source/'refinements/structured_gamma/summary.json'
    arrays = np.load(array_path)
    x, centers, target = arrays['x_train'], arrays['centers'], arrays['y_train'][:, 0]
    archive = json.loads(archive_path.read_text())
    floors = json.loads(floor_path.read_text())
    reference = design(x, centers, 64.)
    reference_basis = svd(reference, full_matrices=False, lapack_driver='gesvd')[0][:, :128]
    polynomial = {degree: qr(legvander(x, degree), mode='economic')[0] for degree in (32, 64, 128)}
    output = dict(reference_gamma=64, gamma=[8, 12, 16, 64], cases=[],
        scope='Reverse finite comparison with full corrected trial action and off-space coupling; no training hits enter selection.',
        numerical_status='FP64 diagnostics, including heuristic guard sensitivities; not interval certificates.',
        trial_selection='Reference64 top128 tail spaces; or gamma-dependent projected eigenbands in fixed discrete Legendre spaces of degrees32,64,128.',
        action_source='Bounds use actual finite-tanh feature actions. The explicit gamma gain and finite remainder are audited afterward on the selected space. This is not an independent gamma-only approximation.',
        approximation_scope='Polynomial coordinates restrict the trial space, not the training model; the full outside-space action is retained through coupling. Each gamma requires a small projected eigensolve. Arithmetic guards are heuristic.',
        null_mass_source='Previously evaluated default1x explicit readout witness, not a spectrum-derived nullspace estimate.')
    for gamma in output['gamma']:
        features = design(x, centers, gamma)
        eta = next(d['eta'] for d in archive['dictionaries'] if d['gamma'] == gamma)
        floor = next(r['floor_upper'][1][0] for r in floors['rows'] if r['n'] == 512 and r['gamma'] == gamma)
        approaches = []
        for label, basis, kind in [('reference64', reference_basis, 'reference_tail')]+[
                (f'polynomial_degree{degree}', basis, 'projected_spectrum') for degree, basis in polynomial.items()]:
            result, selected = scan_space(features, target, basis, eta, floor, kind)
            result['label'] = label
            # Explicit reverse decomposition on the selected space only.
            gain_action = increment_action(x, selected, gamma, 64., centers[1]-centers[0])
            reference_block = (reference.T@selected).T@(reference.T@selected)
            gamma_gain = selected.T@gain_action
            bulk = (reference_block-gamma_gain)*eta
            corrected = eta*(features.T@selected).T@(features.T@selected)
            correction = corrected-bulk
            bulk_action = eta*(reference@(reference.T@selected)-gain_action)
            actual_action = eta*features@(features.T@selected)
            correction_action = actual_action-bulk_action
            reconstructed_action = bulk_action+correction_action
            coupled_action = reconstructed_action-selected@(selected.T@reconstructed_action)
            result['reverse_identity'] = dict(reference_gamma=64, actual_gamma=gamma,
                bulk_min=float(eigh((bulk+bulk.T)/2, eigvals_only=True)[0]),
                bulk_max=float(eigh((bulk+bulk.T)/2, eigvals_only=True)[-1]),
                correction_norm=float(np.linalg.norm(correction, 2)),
                corrected_max=float(eigh((corrected+corrected.T)/2, eigvals_only=True)[-1]),
                projected_correction_action_discrepancy=float(np.linalg.norm(
                    selected.T@correction_action-correction, 'fro')),
                reverse_action_reconstruction_frobenius=float(np.linalg.norm(
                    reconstructed_action-actual_action, 'fro')),
                reconstructed_offspace_coupling=float(np.linalg.norm(coupled_action, 2)),
                additive_upper=max(0., float(eigh((bulk+bulk.T)/2, eigvals_only=True)[-1]
                                              +np.linalg.norm(correction, 2))))
            approaches.append(result)
            print(gamma, label, result['best']['necessary_updates'], flush=True)
        simple = trial_statistics(features, target, (target/np.linalg.norm(target))[:, None], eta)
        jensen = int(np.ceil(np.log(100)/(-np.log1p(-simple['alpha']))))
        output['cases'].append(dict(gamma=gamma, eta=eta, null_upper=floor,
                                   jensen_target_direction_updates=jensen, approaches=approaches))
    paths = [array_path, archive_path, floor_path, Path(__file__),
             Path(__file__).with_name('gamma_ratio_bound.py'), Path(__file__).with_name('core.py')]
    output['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'reverse_bounds.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
