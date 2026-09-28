"""Independent checks of the explicit gain and finite trial-space theorem."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import eigh

from experiments.expD36_frozen_gamma_probe.core import design
from experiments.expD36_frozen_gamma_probe.gamma_ratio_bound import (
    kernel_increment, increment_action, kernel_row_sum_bound,
    compressed_ratio_bound, evaluate_ratios,
)


def test_increment_matches_independent_center_integral_and_limits():
    old, new, h, m = 2., 5., .2, 13
    for distance in (0., 1e-7, .3, 1.5):
        integrand = lambda c: np.tanh(new*(distance-c))*np.tanh(-new*c) \
            -np.tanh(old*(distance-c))*np.tanh(-old*c)
        expected = quad(integrand, -20, 20, epsabs=1e-12, points=[0., distance])[0]/(h*m)
        np.testing.assert_allclose(kernel_increment(distance, old, new, h, m), expected,
                                   rtol=1e-9, atol=2e-13)
    assert kernel_increment(0., old, new, h, m) == 2*(1/old-1/new)/(h*m)
    np.testing.assert_array_equal(kernel_increment([0., 1., 1e6], old, old, h, m), 0.)
    assert np.isfinite(kernel_increment(1e6, old, new, h, m))


def test_toeplitz_action_and_positive_increment_against_dense():
    x = np.linspace(-1, 1, 41)
    u = np.random.default_rng(12).normal(size=(len(x), 5))
    dense = kernel_increment(x[:, None]-x, 3., 7., .1, len(x))
    np.testing.assert_allclose(increment_action(x, u, 3., 7., .1), dense@u, atol=2e-14)
    assert eigh(dense, eigvals_only=True)[0] > -1e-13
    with pytest.raises(ValueError):
        increment_action([0., .1, .4], u[:3], 3., 7., .1)


def test_rowsum_bound_and_trial_theorem_include_zero_and_unresolved():
    rng = np.random.default_rng(5)
    j = rng.normal(size=(19, 8))
    assert kernel_row_sum_bound(j, block=4) == pytest.approx(np.max(np.sum(abs(j@j.T), axis=1)))
    c = np.diag([3., 1.])
    positive = compressed_ratio_bound(c, np.zeros((2, 2)), .8*c, 3.)
    assert positive['lower_ratio'] == pytest.approx(.8/3)
    zero = compressed_ratio_bound(c, np.zeros((2, 2)), 3*c, 9.)
    assert zero['status'] == 'resolved_zero_bound'
    assert zero['lower_ratio'] == 0
    unresolved = compressed_ratio_bound(np.diag([1., 0.]), np.zeros((2, 2)), c, 3.)
    assert unresolved['lower_ratio'] is None


@pytest.mark.parametrize('gamma', [4., 8., 16.])
def test_finite_ratio_bound_below_independent_spectrum(gamma):
    x = np.linspace(-1, 1, 65)
    centers = np.linspace(-1.25, 1.25, 31)
    result = evaluate_ratios(x, centers, design(x, centers, 4.), design(x, centers, gamma),
                            4., gamma, ranks=(2, 4, 8, 16, 40))
    assert result['denominator_row_sum'] >= result['reference_largest_eigenvalue']-1e-13
    assert result['direct_row_action_max_error'] < 1e-13
    assert result['records'][-1]['status'] == 'rank_exceeds_feature_dimension'
    for row in result['records'][:-1]:
        if row['lower_ratio'] is not None:
            assert row['lower_ratio'] <= row['actual_new_ratio']+1e-12
    assert result['records'][0]['lower_ratio'] > 0
