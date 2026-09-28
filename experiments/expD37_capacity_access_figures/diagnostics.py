"""Note equations (4)--(6), (11), and (16) on the actual D36 grid.

Work with B's SVD, never an explicitly formed B B.T. Truncated modes
remain residual, not declared exact nullspace. Save cutoff/driver audits.
"""
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq


def polynomial_tails(x, Y, maximum):
    P, _ = sla.qr(np.polynomial.legendre.legvander(x, maximum), mode='economic')
    D = np.zeros((maximum+1, Y.shape[1]))
    q = np.zeros((maximum+1, len(x), Y.shape[1]))
    for k in range(maximum+1):
        basis = P[:, :k+1]
        tail = Y-basis@(basis.T@Y)
        # Reprojection removes low-degree roundoff before normalizing small tails.
        tail -= basis@(basis.T@tail)
        D[k] = np.linalg.norm(tail, axis=0)/np.linalg.norm(Y, axis=0)
        valid = D[k] > 1e-13
        q[k, :, valid] = (tail[:, valid]/np.linalg.norm(tail[:, valid], axis=0)).T
    return P, D, q


def structural_envelope(gamma, k, width, map_norm=1.):
    logrho = np.arcsinh(np.pi/(2*gamma))
    U = 4*np.exp(-np.asarray(k)*logrho)/np.expm1(logrho)
    U *= 1/np.sqrt(gamma**2+np.pi**2/4)+1/(np.pi*(np.asarray(k)+1))
    return map_norm**2*width*np.minimum(np.tanh(gamma), U)**2


def spectral_parts(U, s, vector, cutoff):
    vector = vector/np.linalg.norm(vector)
    keep = s > cutoff*s[0]
    a = U[:, keep].T@vector
    outside = vector-U[:, keep]@a
    return (s[keep]/s[0])**2, a*a, float(outside@outside)


def hitting_time(U, s, vector, epsilon, cutoff):
    """tau = T * ||B||^2. Inf means unresolved with THIS cutoff, not impossible."""
    nu, weights, outside = spectral_parts(U, s, vector, cutoff)
    if outside >= epsilon**2:
        return np.inf
    def difference(logtau):
        return outside+weights@np.exp(-2*10.**logtau*nu)-epsilon**2
    return 10.**brentq(difference, -12, 40, xtol=1e-12)


def damping_remaining(U, s, vector, rho, cutoff):
    nu, weights, outside = spectral_parts(U, s, vector, cutoff)
    gains = rho[:, None]/(rho[:, None]+nu[None, :])
    return np.sqrt(outside+gains**2@weights)


def discrete_error(U, s, vector, steps, cutoff=1e-14):
    """Actual spectral discrete GD at eta=1/||B||^2; used to validate against training."""
    nu, weights, outside = spectral_parts(U, s, vector, cutoff)
    factors = np.maximum(1-nu, 0.)
    return np.sqrt(outside+weights@(factors[:, None]**(2*np.asarray(steps)[None, :])))


def measure(cfg, source, output):
    ref = np.load(source/'data/reference.npz')
    x, centers = ref['x_train'], ref['centers']
    Y = ref['y_train']/np.sqrt(len(x))
    names = ['sine', 'quadratic', 'mixed', 'runge']
    gammas = np.asarray(cfg['diagnostic_gammas'], dtype=float)
    ks = np.arange(cfg['polynomial_max_degree']+1)
    P, D, q = polynomial_tails(x, Y, ks[-1])
    eps = cfg['relative_tolerance']
    necessary = np.maximum(D-eps, 0.)**2
    cutoffs = np.asarray(cfg['svd_relative_cutoffs'])
    rho = np.logspace(cfg['relative_damping_log10_min'], cfg['relative_damping_log10_max'],
                      cfg['relative_damping_count'])
    ng, nf, nk = len(gammas), len(names), len(ks)
    access = np.full((ng, nk, nf), np.nan)
    access_resolved = np.zeros((ng, nk, nf), dtype=bool)
    envelope, norms = [], []
    times = np.zeros((ng, nf, len(cutoffs)))
    driver_times = np.zeros((ng, nf))
    remainder = np.full((ng, nf, len(cutoffs), len(rho)), np.nan)
    full_remainder = np.zeros_like(remainder)
    probe_span_remainders = np.full((ng, nf, len(cutoffs)), np.nan)
    singulars, audit = [], []
    primary = cfg['primary_svd_relative_cutoff']
    probe_k = cfg['common_tail_degree']
    for i, gamma in enumerate(gammas):
        B = np.c_[np.ones(len(x)), np.tanh(gamma*(x[:, None]-centers))]/np.sqrt(len(x))
        U, s, V = sla.svd(B, full_matrices=False, lapack_driver='gesdd')
        U2, s2, _ = sla.svd(B, full_matrices=False, lapack_driver='gesvd')
        norms.append(s[0]**2)
        singulars.append(s)
        envelope.append(structural_envelope(gamma, ks, len(centers))/s[0]**2)
        max_identity_discrepancy = 0.
        for k in ks:
            projected = B-P[:, :k+1]@(P[:, :k+1].T@B)
            direct = B.T@q[k]
            alternate = projected.T@q[k]
            amp = np.linalg.norm(direct, axis=0)
            discrepancy = np.linalg.norm(direct-alternate, axis=0)
            access[i, k] = amp**2/s[0]**2
            # Omit unresolved signals rather than plotting a roundoff plateau.
            access_resolved[i, k] = ((amp > 100*np.finfo(float).eps*s[0]) &
                                     (discrepancy < .01*amp) & (D[k] > 1e-13))
            max_identity_discrepancy = max(max_identity_discrepancy, float(discrepancy.max()/s[0]))
        for f in range(nf):
            driver_times[i, f] = hitting_time(U2, s2, Y[:, f], eps, primary)
            for j, cutoff in enumerate(cutoffs):
                times[i, f, j] = hitting_time(U, s, Y[:, f], eps, cutoff)
                full_remainder[i, f, j] = damping_remaining(U, s, Y[:, f], rho, cutoff)
                if D[probe_k, f] > 1e-13:
                    remainder[i, f, j] = damping_remaining(U, s, q[probe_k, :, f], rho, cutoff)
                    probe_span_remainders[i, f, j] = np.sqrt(spectral_parts(U, s, q[probe_k, :, f], cutoff)[2])
        audit.append({'gamma': float(gamma),
                      'relative_svd_reconstruction_error': float(np.linalg.norm(B-(U*s)@V, 2)/s[0]),
                      'direct_vs_projected_access_amplitude_max_discrepancy': max_identity_discrepancy})
        print(f'Measured theorem diagnostics at gamma={gamma:g}', flush=True)
    envelope = np.asarray(envelope)
    C = np.log(1/eps)/(1-eps)**2
    ratios = np.where(access_resolved, necessary[None, :, :]/np.maximum(access, 1e-300), 0.)
    directional_bound = C*np.max(ratios, axis=1)
    directional_k = np.argmax(ratios, axis=1)
    slope_ratios = necessary[None, :, :]/envelope[:, :, None]
    slope_bound = C*np.max(slope_ratios, axis=1)
    slope_k = np.argmax(slope_ratios, axis=1)

    # Independent check: exact spectral discrete-GD formula versus saved real GD,
    # at every saved lambda and three budgets. This does not validate Adam.
    history = np.load(source/'data/trajectory.npz')
    steps = np.asarray([1, 100, 20000])
    observed, predicted = [], []
    for gamma in ref['gammas']:
        B = np.c_[np.ones(len(x)), np.tanh(gamma*(x[:, None]-centers))]/np.sqrt(len(x))
        U, s, _ = sla.svd(B, full_matrices=False)
        predicted.append(np.stack([discrete_error(U, s, Y[:, f], steps) for f in range(nf)]))
    for step in steps:
        index = int(np.flatnonzero(history['steps'] == step)[0])
        observed.append(history['train_rel_l2'][index, 0])
    observed = np.asarray(observed).transpose(1, 2, 0)
    predicted = np.asarray(predicted)
    output.joinpath('data').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'data/theorem_diagnostics.npz',
        gammas=gammas, k=ks, D=D, necessary_energy=necessary, access=access,
        access_resolved=access_resolved, normalized_envelope=envelope,
        squared_operator_norm=np.asarray(norms), singular_values=np.asarray(singulars),
        flow_time=times, flow_time_second_driver=driver_times, relative_cutoffs=cutoffs,
        directional_bound=directional_bound, directional_maximizing_k=directional_k,
        bounded_slope_bound=slope_bound, bounded_slope_maximizing_k=slope_k,
        rho=rho, tail_damping_remainder=remainder, full_damping_remainder=full_remainder,
        probe_span_remainder_by_cutoff=probe_span_remainders,
        discrete_gd_validation_steps=steps, discrete_gd_observed=observed,
        discrete_gd_predicted=predicted)
    ii = int(np.argmin(abs(cutoffs-primary)))
    ratio = np.divide(times[:, :, ii], driver_times, where=np.isfinite(driver_times),
                      out=np.full_like(driver_times, np.nan))
    report = {'per_gamma': audit,
              'polynomial_Q_orthogonality_error': float(np.linalg.norm(P.T@P-np.eye(len(ks)), 2)),
              'quadratic_max_tail_degree_2_and_above': float(D[2:, 1].max()),
              'max_discrete_GD_absolute_error_vs_saved_training': float(np.max(abs(observed-predicted))),
              'max_spectral_time_relative_driver_discrepancy': float(np.nanmax(abs(ratio-1))),
              'finite_time_all_requested_targets': bool(np.isfinite(times[:, [2,3], :]).all()),
              'directional_bound_below_primary_flow_time': bool(np.all(directional_bound <= times[:, :, ii]*(1+1e-6))),
              'structural_bound_below_primary_flow_time': bool(np.all(slope_bound <= times[:, :, ii]*(1+1e-6))),
              'structural_envelope_covers_resolved_access': bool(np.all(access[access_resolved] <= np.broadcast_to(envelope[:, :, None], access.shape)[access_resolved]*(1+1e-6))),
              'limits': 'FP64 diagnostics, independent SVD drivers and cutoffs; no claim of an exact-arithmetic span floor. The time certificate maximizes only the plotted degree range; unresolved access values are omitted. No inference of actual Adam iteration counts.'}
    (output/'data/validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if not (report['directional_bound_below_primary_flow_time'] and report['structural_bound_below_primary_flow_time'] and report['structural_envelope_covers_resolved_access']):
        raise AssertionError(report)
    print(json.dumps(report, indent=2), flush=True)
