"""Small exact recurrences verify the frozen-model envelope, not GD transfer."""
import numpy as np
import pytest

from experiments.expD34_readout_race.mechanism_envelope import (
    geometric_gain, spectral_envelope, tracking_amplification,
)


def test_endpoint_and_prefix_bound_against_explicit_coupled_updates():
    T = np.array([[.3, -.2], [.1, .4], [.2, .1], [0., 0.], [.1, -.1], [0., .2], [0., 0.]])
    p = np.array([.15, -.4, 0., 0., 0., 0., 0.])
    e0 = np.array([.8, -.5])
    eta = .2
    result = spectral_envelope(p, T, e0, eta, [1, 10, 31], h=.5)
    point, error = p.copy(), e0.copy()
    maximum = abs(p[:2]).copy()
    for n in range(1, 32):
        point -= eta*T@error
        error -= eta*(T.T@T)@error
        maximum = np.maximum(maximum, abs(point[:2]))
        if n in (1, 10, 31):
            k = [1, 10, 31].index(n)
            np.testing.assert_allclose(result['pure_endpoint_a'][k], point[:2], atol=1e-15)
            assert np.all(maximum*.5 <= result['pure_prefix_lambda_upper'][k]+1e-15)


def test_zero_tiny_modes_and_nonoscillating_gate():
    gains = geometric_gain(np.array([0., 1e-200, .5, 1.]), 1., 50000000)
    assert gains[0] == 0
    assert gains[1] > 0
    np.testing.assert_allclose(gains[1], 5e-193)
    np.testing.assert_allclose(gains[2:], [2., 1.])
    np.testing.assert_array_equal(geometric_gain(np.array([0., 1.]), 1., 0), [0., 0.])
    with pytest.raises(ValueError, match='Nonoscillating'):
        geometric_gain(np.array([1.1]), 1., 10)


def test_tracking_dyadic_upper_sum_and_supplied_allowance():
    left_a = np.array([[.8, -.1], [.2, .7]])
    singular = np.array([.1, .8])
    eta, horizon = .1, 31
    upper, endpoint = tracking_amplification(left_a, singular, eta, horizon)
    exact = eta*sum(np.linalg.norm(left_a*geometric_gain(singular, eta, k), axis=1)
                    for k in range(horizon))
    assert np.all(upper >= exact-1e-14)
    np.testing.assert_allclose(endpoint, np.linalg.norm(left_a*geometric_gain(singular, eta, 30), axis=1))
    T = np.zeros((7, 2)); T[:2] = np.diag(singular)
    result = spectral_envelope(np.zeros(7), T, np.ones(2), eta, [31],
                               tracking_u=np.array([.01, .02]), tracking_v=.03)
    expected = eta*31*np.array([.01, .02])+.03*result['tracking_residual_amplification_upper'][0]
    np.testing.assert_allclose(result['tracking_only_allowance'][0], expected)


def test_initial_occupants_and_no_invented_tracking_bounds():
    p = np.array([16., 0., 0., 0., 0., 0., 0.])
    result = spectral_envelope(p, np.zeros((7, 2)), np.ones(2), .002, [1, 50000000])
    np.testing.assert_array_equal(result['initial_occupied'], [True, False])
    np.testing.assert_array_equal(result['pure_prefix_lambda_upper'], [[.25, 0.], [.25, 0.]])
    assert 'tracking_only_allowance' not in result
    with pytest.raises(ValueError, match='both tracking'):
        spectral_envelope(p, np.zeros((7, 2)), np.ones(2), .002, [1], tracking_v=.01)
