import numpy as np
import pytest

from experiments.expD36_frozen_gamma_probe import core
from experiments.expD36_frozen_gamma_probe.slope_spectrum import (
    analytic_access, cdf_atoms, cdf_error, cdf_time_bound, distribution_bound,
    exceptional_target_tails, mean_spectrum_bound, pole_coefficients, target_cdf_bound,
)


def test_distribution_spectral_tail_with_large_slope_exceptions():
    x = np.linspace(-1, 1, 193)
    centers = np.linspace(-1.2, 1.2, 31)
    slopes = np.linspace(.2, 4, len(centers))
    slopes[[5, 15, 25]] = [30, 50, -70]
    j = core.design(x, centers, slopes)
    values = np.linalg.svd(j, compute_uv=False)**2
    degrees = np.array([0, 2, 4, 8, 12, 20])
    for threshold in [0, 2, 4, 70]:
        result = distribution_bound(slopes, degrees, threshold)
        assert np.all(result['bound'] <= result['cap_bound']*(1+1e-12))
        for rank, bound in zip(result['rank'], result['bound']):
            assert values[rank:].sum() <= bound*(1+1e-11)+1e-25
    mean_only = mean_spectrum_bound(len(slopes), np.mean(np.abs(slopes)),
                                   np.arange(32), np.geomspace(.1, 500, 101))
    assert np.all(values <= mean_only*(1+1e-10)+1e-25)


def test_center_poles_against_independent_projected_tanh_and_target_access():
    x = np.linspace(-1, 1, 257)
    centers = np.array([-1.4, -.8, -.31, 0, .12, .6, 1.3])
    slopes = np.array([0, .5, -2, 4, 8, 16, 32])
    y = np.column_stack([np.sin(6*np.pi*x), np.exp(x)])/np.sqrt(len(x))
    result = analytic_access(x, centers, slopes, y, 40, 192, 128)
    j = core.design(x, centers, slopes)
    for k in [0, 1, 4, 8, 16, 24, 40]:
        p, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, k))
        tail = j-p@(p.T@j)
        assert np.linalg.norm(tail, 'fro')**2 <= result['centered'][k]*(1+1e-10)+1e-25
        for ti in range(2):
            q = y[:, ti]-p@(p.T@y[:, ti])
            # Explicit subtraction loses relative accuracy for tiny exp tails;
            # only compare independently resolved directions to the bound.
            if np.linalg.norm(q) > 1e-5:
                q /= np.linalg.norm(q)
                assert np.linalg.norm(j.T@q)**2 <= result['directional'][k, ti]*(1+1e-9)+1e-25
    coefficients, _, _ = pole_coefficients(np.array([0.]), np.array([4.]), 40)
    np.testing.assert_allclose(coefficients[1::2], 0, atol=1e-16)
    assert np.all(result['centered'] <= result['cap']*(1+1e-12))


def test_cdf_combination_is_below_direct_gd_and_uses_more_than_one_threshold():
    rates = np.array([.001, .008, .06, .3, 1.])
    weights = np.array([.02, .08, .3, .2, .4])
    thresholds = np.array([.002, .01, .1, .4])
    p = np.array([weights[rates <= s].sum() for s in thresholds])
    bound_rates, bound_weights = cdf_atoms(thresholds, p)
    assert bound_weights.sum() == pytest.approx(1.)
    theta = np.zeros(5)
    j = np.diag(np.sqrt(rates)); y = np.sqrt(weights)
    hit = None
    for step in range(6000):
        error = np.linalg.norm(j@theta-y)
        lower = cdf_error(step, bound_rates, bound_weights)
        assert lower <= error+1e-12
        assert lower+1e-14 >= np.max(np.sqrt(p)*(1-.5*thresholds)**step)
        if error <= .01:
            hit = step
            break
        theta -= .5*j.T@(j@theta-y)
    assert hit is not None
    assert cdf_time_bound(thresholds, p)['bound'] <= hit
    assert cdf_time_bound([.01], [1.])['bound'] == np.ceil(np.log(.01)/np.log(.995))
    assert cdf_time_bound([0.], [.1])['status'] == 'proved_zero_mode_obstruction'
    np.testing.assert_allclose(target_cdf_bound([.5], [0.], [0., .1], 1), [.25, .25])


def test_exception_aligned_target_and_inactive_neuron_counterexample():
    x = np.linspace(-1, 1, 129)
    centers = np.linspace(-1, 1, 21)
    slopes = np.zeros(21); slopes[10] = 64
    j = core.design(x, centers, slopes)
    y = j[:, 11]
    tails, resolved = exceptional_target_tails(x, j[:, [11]], y, [0, 2, 8])
    assert np.all(resolved)
    np.testing.assert_array_equal(tails, 0)
    one = core.design(x, np.array([0.]), 64)
    np.testing.assert_allclose(j@j.T, one@one.T, atol=1e-15)
    assert np.mean(slopes) < 4 and np.median(slopes) == 0
    assert np.all(target_cdf_bound(tails[:, 0], np.ones(3), [.01, .1], 1) == 0)


def test_target_mass_witness_against_full_eigenvectors():
    rng = np.random.default_rng(95)
    j = rng.normal(size=(15, 8))
    y = rng.normal(size=15); y /= np.linalg.norm(y)
    p, _ = np.linalg.qr(rng.normal(size=(15, 4)))
    q = y-p@(p.T@y)
    delta = np.linalg.norm(q); q /= delta
    access = np.linalg.norm(j.T@q)**2
    eigenvalues, u = np.linalg.eigh(j@j.T)
    thresholds = np.geomspace(.01, .99, 40)
    lower = target_cdf_bound([delta], [access], thresholds, eigenvalues[-1])
    actual = np.array([np.sum((u[:, eigenvalues <= s*eigenvalues[-1]].T@y)**2)
                       for s in thresholds])
    assert np.all(lower <= actual+1e-12)
