import numpy as np
import pytest

from experiments.expD36_frozen_gamma_probe import fourier_law as p
from experiments.expD36_frozen_gamma_probe import finite_gamma_gram as f


@pytest.mark.parametrize('gamma', [.3, 1., 4., 16.])
@pytest.mark.parametrize('density,offset', [(1, 0.), (1, .5), (4, 0.), (4, .5)])
def test_periodic_independent_synthesis(gamma, density, offset):
    n = 16
    a = p.periodized_design(n, gamma, density, offset, images=80)
    b = p.sampled_design(n, gamma, density, offset, aliases=80)
    np.testing.assert_allclose(a, b, atol=3e-14, rtol=1e-11)
    np.testing.assert_allclose(a.sum(axis=1), 2/np.sqrt(n*density), atol=3e-14)
    fourier = np.exp(2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)
    h = fourier.conj().T@(a.T@a)@fourier
    np.testing.assert_allclose(h, np.diag(p.sampled_spectrum(n, gamma, density, offset, 80)), atol=3e-14)
    if density == 1 and offset == 0:
        np.testing.assert_allclose(a@((-1.)**np.arange(n)), 0, atol=3e-14)


def test_continuous_bracket_and_alias_tail():
    theta = np.linspace(-np.pi, np.pi, 101)
    for lam in [.1, 1., 8., 32.]:
        for aliases in [0, 2, 8]:
            q = np.r_[np.arange(-512, -aliases), np.arange(aliases+1, 513)]
            values = np.abs(p.amplitude(theta[:, None]+2*np.pi*q, lam))
            for power in [1, 2]:
                assert np.all(np.sum(values**power, axis=1) <= p.alias_tail(theta, lam, aliases, power)*(1+1e-12)+1e-300)
        spectrum, lo, hi = p.continuous_spectrum(64, 32*lam, aliases=512)
        assert np.all(spectrum*64/4 >= lo-2e-14)
        assert np.all(spectrum*64/4 <= hi+2e-14)


@pytest.mark.parametrize('gamma', [.01, 4., 64.])
def test_finite_gram_and_actual_gd(gamma):
    x = np.linspace(-1, 1, 129)
    centers = np.r_[np.linspace(-1.2, 1.2, 17), .3, .3, .3+1e-12]
    j = np.column_stack([np.ones(len(x)), np.tanh(gamma*(x[:, None]-centers))])/np.sqrt(len(x))
    h = f.gram(x, centers, gamma)
    np.testing.assert_allclose(h, j.T@j, atol=2e-13, rtol=2e-12)
    y = np.sin(2*np.pi*x)/np.sqrt(len(x))
    forecast = f.rectangular_forecast(j, y)
    eta = .5/forecast['L']
    theta = np.zeros(j.shape[1])
    for step in range(101):
        if step in [0, 1, 10, 100]:
            np.testing.assert_allclose(f.error(forecast, step)[0], np.linalg.norm(j@theta-y)/np.linalg.norm(y), atol=2e-13)
        theta -= eta*j.T@(j@theta-y)


def test_integer_first_hit_and_gram_forecast():
    h = np.diag([1., .1])
    model = f.gram_forecast(h, np.array([0., np.sqrt(.1)]), 1., .5)
    hit = f.first_hit(model, .01)[0]
    assert hit == int(np.ceil(np.log(.01)/np.log(.95)))
    assert f.error(model, hit)[0] <= .01 < f.error(model, hit-1)[0]


def test_whole_line_neighbor_integral_and_noncommuting_transfer():
    from scipy.integrate import quad
    from experiments.expD36_frozen_gamma_probe.fourier_validate import whole_line_neighbor_gram
    for gamma in [.5, 4., 16.]:
        h = .25
        matrix = whole_line_neighbor_gram(5, h, gamma)
        for k in range(5):
            def integrand(x):
                a = np.tanh(gamma*x)-np.tanh(gamma*(x-h))
                b = np.tanh(gamma*(x-k*h))-np.tanh(gamma*(x-(k+1)*h))
                return a*b/2
            actual, _ = quad(integrand, -40/gamma, 5*h+40/gamma, epsabs=1e-12)
            assert matrix[0, k] == pytest.approx(actual, abs=2e-12)
    rng = np.random.default_rng(4)
    a = rng.normal(size=(8, 5)); b = rng.normal(size=(8, 5))
    k, kt = a@a.T, b@b.T
    eta = .5/max(np.linalg.norm(k, 2), np.linalg.norm(kt, 2))
    y = rng.normal(size=8); y /= np.linalg.norm(y)
    actual, surrogate = y.copy(), y.copy()
    accumulated = 0.
    for step in range(50):
        assert np.linalg.norm(actual-surrogate) <= accumulated+1e-12
        accumulated += eta*np.linalg.norm((k-kt)@surrogate)
        actual = actual-eta*k@actual
        surrogate = surrogate-eta*kt@surrogate


def test_coefficient_rescaling_with_matching_clock():
    rng = np.random.default_rng(22)
    j, y = rng.normal(size=(12, 7)), rng.normal(size=12)
    original = f.rectangular_forecast(j, y)
    scaled = f.rectangular_forecast(13*j, y)
    for step in [0, 1, 20, 1000]:
        np.testing.assert_allclose(f.error(original, step), f.error(scaled, step), atol=2e-14)
