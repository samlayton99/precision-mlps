"""Slow-subspace bounds with non-invariant trial directions and null energy."""
import numpy as np
from scipy.linalg import eigh, qr

from experiments.expD36_frozen_gamma_probe.gamma_reverse_bound import delay_bound, trial_statistics, scan_space


def test_exact_slow_eigenvector_and_null_subtraction():
    result = delay_bound(.001, 0., .2, .01, .002)
    assert abs(result['positive_mass']-.03) < 1e-15
    assert result['necessary_updates'] > 0
    assert delay_bound(.1, 1., .5, 0., .2)['positive_mass'] == 0
    assert delay_bound(.001, 0., .2, .04, .002)['necessary_updates'] == 0


def test_rotated_trial_bound_below_true_positive_mass_and_residual():
    rng = np.random.default_rng(182)
    for _ in range(15):
        rotation = qr(rng.normal(size=(7, 7)))[0]
        rates = np.array([0., .001, .003, .008, .1, .2, .5])
        features = rotation@np.diag(np.sqrt(rates))
        target = rng.normal(size=7); target /= np.linalg.norm(target)
        trial = rotation[:, 1:4]+.001*rng.normal(size=(7, 3))
        trial = qr(trial, mode='economic')[0]
        stats = trial_statistics(features, target, trial, 1.)
        null_mass = float((rotation[:, 0]@target)**2)
        result = delay_bound(**stats, null_upper=null_mass, cutoff=.025)
        actual_mass = float(np.sum((rotation[:, 1:4].T@target)**2))
        assert result['positive_mass'] <= actual_mass+1e-13
        for step in (0, 100, 1000):
            actual_error = np.linalg.norm((1-rates)**step*(rotation.T@target))
            lower = np.sqrt(result['positive_mass'])*(1-.025)**step
            assert lower <= actual_error+1e-13


def test_reference_subtraction_is_not_globally_positive():
    from experiments.expD36_frozen_gamma_probe.gamma_ratio_bound import kernel_increment
    from experiments.expD36_frozen_gamma_probe.core import design
    x = np.linspace(-1, 1, 19); centers = np.linspace(-1, 1, 7)
    reference = design(x, centers, 8.)
    gain = kernel_increment(x[:, None]-x, 2., 8., centers[1]-centers[0], len(x))
    bulk = reference@reference.T-gain
    assert eigh(bulk, eigvals_only=True)[0] < -1e-4


def test_candidate_scan_matches_direct_action_and_true_spectral_mass():
    rotation = qr(np.random.default_rng(5).normal(size=(31, 31)))[0]
    rates = np.geomspace(1e-6, .5, 31)
    features = rotation@np.diag(np.sqrt(rates))
    target = rotation[:, 3]+.1*rotation[:, 25]
    result, selected = scan_space(features, target, rotation[:, :20], 1., 0., 'projected_spectrum')
    best = result['best']
    assert best['necessary_updates'] > 0
    true_mass = np.sum((rotation[:, rates <= best['cutoff']].T@target)**2)/(target@target)
    assert best['positive_mass'] <= true_mass+1e-12
    np.testing.assert_allclose(selected.T@selected, np.eye(selected.shape[1]), atol=1e-14)
    for key in ('alpha', 'coupling', 'target_overlap'):
        np.testing.assert_allclose(best[key], best['direct_check'][key], atol=1e-13)
