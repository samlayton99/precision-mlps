"""Independent discrete recurrences check the frozen-map budget and scope."""

import numpy as np

from experiments.expD34_readout_race.cross_function_budget import spectral_budget


def direct_history(p0, T, e0, eta, steps):
    p, e = p0.copy(), e0.copy()
    history = [p.copy()]
    for _ in range(steps):
        direction = T @ e
        p = p-eta*direction
        e = e-eta*(T.T @ direction)
        history.append(p.copy())
    return np.array(history)


def test_svd_prediction_matches_independent_full_recurrence():
    rng = np.random.default_rng(73)
    p0 = rng.normal(size=10)
    T = rng.normal(size=(10, 5))*.12
    e0 = rng.normal(size=5)
    eta = .2
    horizons = [0, 1, 7, 80]
    got = spectral_budget(p0, T, e0, eta, horizons)
    actual = direct_history(p0, T, e0, eta, max(horizons))
    assert got["applicable"]
    np.testing.assert_allclose(got["displacement"], actual[horizons]-p0,
                               rtol=2e-12, atol=3e-14)
    np.testing.assert_allclose(got["displacement_norm"],
                               np.linalg.norm(actual[horizons]-p0, axis=1), atol=3e-14)
    travel = np.cumsum(np.abs(np.diff(actual[:, :3], axis=0)), axis=0)
    assert np.all(travel[np.array(horizons[1:])-1] <= got["slope_travel_upper"][1:]+3e-14)


def test_exact_null_and_tiny_positive_mode_keep_their_distinct_response():
    p0 = np.zeros(7)
    T = np.zeros((7, 3))
    T[0, 0], T[1, 1] = .5, 1e-14
    e0 = np.array([2., -1e12, 9.])
    got = spectral_budget(p0, T, e0, .2, [0, 1, 13])
    expected = direct_history(p0, T, e0, .2, 13)
    np.testing.assert_allclose(got["displacement"], expected[[0, 1, 13]], atol=3e-15)
    np.testing.assert_allclose(got["displacement"][-1, 1], .026, atol=1e-16)
    assert got["singular_values"][-1] == 0
    assert got["displacement"][-1, 2] == 0


def test_step_boundary_and_outside_applicability_are_explicit():
    p0 = np.array([1.1, .2, 0., 0., 0., 0., 0.])
    T = np.zeros((7, 1))
    T[1, 0] = 1.
    got = spectral_budget(p0, T, np.array([.5]), 1., [0, 1, 9])
    assert got["applicable"]
    np.testing.assert_allclose(got["displacement"][[1, 2], 1], -.5)
    np.testing.assert_array_equal(got["initial_occupancy"], [1, 0, 0])
    np.testing.assert_array_equal(got["simultaneous_occupancy_upper"][0], [1, 0, 0])
    np.testing.assert_array_equal(got["ever_occupancy_upper"][0], [1, 0, 0])
    # Even a stable but oscillating step is outside this particular theorem.
    for eta in (1.01, 3.):
        unsupported = spectral_budget(p0, T, np.array([.5]), eta, [1, 9])
        assert not unsupported["applicable"]
        assert np.isnan(unsupported["displacement"]).all()
        assert (unsupported["simultaneous_occupancy_upper"] == -1).all()


def test_signed_cancellation_and_crossings_do_not_break_population_bounds():
    p0 = np.array([.2, 1.1, 0., 0., 0., 0., 0.])
    T = np.zeros((7, 2))
    T[:2] = np.array([[.8, .2], [.8, -.2]])/np.sqrt(2.)
    e0 = np.array([1., -4.])
    horizons = [0, 1, 10, 100]
    got = spectral_budget(p0, T, e0, .5, horizons, thresholds=(.5, 1., 3.2))
    actual = direct_history(p0, T, e0, .5, 100)
    # Two nonzero modal slope forces cancel initially for neuron zero.
    assert abs(actual[1, 0]-p0[0]) < 1e-15
    assert got["slope_travel_upper"][1, 0] > .1
    assert np.any(actual[:, 1] < 0)  # Include actual absolute-value crossings.
    for h, n in enumerate(horizons):
        magnitudes = np.abs(actual[:n+1, :2])
        positive_travel = np.maximum(np.diff(magnitudes, axis=0), 0).sum(axis=0)
        assert np.all(positive_travel <= got["slope_travel_upper"][h]+1e-13)
        for k, threshold in enumerate(got["thresholds"]):
            acquired = magnitudes >= threshold
            assert acquired.sum(axis=1).max() <= got["simultaneous_occupancy_upper"][h, k]
            assert acquired.any(axis=0).sum() <= got["ever_occupancy_upper"][h, k]
