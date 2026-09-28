"""Independent raw-feature, lattice, and optimizer checks of the structure."""
import numpy as np
import pytest
from scipy.linalg import eigh

from experiments.expD36_frozen_gamma_probe.structured_gamma import (
    finite_geometry, whiten, spectrum, transformed_gd_step,
    kernel_error_bound, difference_profile, polyphase_symbol, toeplitz_boundary,
)


@pytest.mark.parametrize('n,q,halo,gamma', [(5, 1, 0, 1.), (6, 3, 2, 4.), (9, 2, 1, 12.)])
def test_exact_transform_metric_and_spectrum(n, q, halo, gamma):
    g = finite_geometry(n, q, halo, gamma)
    j, z, c, metric = [g[k] for k in ('j', 'z', 'transform', 'metric')]
    np.testing.assert_allclose(z, j@c, atol=2e-15)
    np.testing.assert_allclose(whiten(z, c), j, atol=5e-15)
    np.testing.assert_array_equal(np.diag(metric), np.r_[1., np.full(len(c)-2, 2.), 1.])
    expected = np.linalg.svd(j, compute_uv=False)**2
    np.testing.assert_allclose(spectrum(z, c)['eigenvalues'], expected, atol=2e-14)
    generalized = eigh(z.T@z, metric, eigvals_only=True)
    np.testing.assert_allclose(np.sort(generalized)[-len(expected):][::-1], expected, atol=3e-14)


def test_metric_gd_matches_raw_gd_and_euclidean_reparameterization_does_not():
    g = finite_geometry(7, 3, 2, 3.)
    j, z, c = [g[k] for k in ('j', 'z', 'transform')]
    target = np.sin(5*g['x'])/np.sqrt(len(j))
    raw = np.zeros(j.shape[1]); transformed = raw.copy(); wrong = raw.copy()
    step = .4/np.linalg.norm(j, 2)**2
    for _ in range(31):
        raw -= step*j.T@(j@raw-target)
        transformed = transformed_gd_step(z, c, transformed, target, step)
        wrong -= step*z.T@(z@wrong-target)
    np.testing.assert_allclose(c@transformed, raw, atol=1e-13)
    assert np.linalg.norm(z@wrong-j@raw) > 1e-3


@pytest.mark.parametrize('bandwidth,q', [(.3, 1), (.8, 3), (2., 4)])
def test_symbol_against_independent_lattice_sum(bandwidth, q):
    theta = np.linspace(-np.pi, np.pi, 53)
    symbol = polyphase_symbol(theta, bandwidth, q, aliases=12)
    k = np.arange(-300, 301)
    direct = np.array([(np.tanh(bandwidth*(k+s/q))-np.tanh(bandwidth*(k-1+s/q)))
                       @np.exp(-1j*k[:, None]*theta) for s in range(q)])
    np.testing.assert_allclose(symbol['polyphase'], direct, atol=3e-13)
    np.testing.assert_allclose(polyphase_symbol([0.], bandwidth, q)['polyphase'], 2., atol=2e-14)


def test_symbol_alias_tail_and_quadrature_gram():
    bandwidth, q, m = 3., 2, 17
    theta = -np.pi+2*np.pi*np.arange(4096)/4096
    coarse = polyphase_symbol(theta, bandwidth, q, aliases=0, samples=m)
    fine = polyphase_symbol(theta, bandwidth, q, aliases=20, samples=m)
    assert np.max(np.abs(coarse['polyphase']-fine['polyphase'])) <= coarse['amplitude_tail_bound']
    assert np.all(np.abs(coarse['energy']-fine['energy']) <= coarse['energy_error_bound'])
    k = np.arange(-100, 101)
    for lag in range(4):
        correlation = sum(np.dot(difference_profile(k+s/q, bandwidth),
                                 difference_profile(k+s/q+lag, bandwidth)) for s in range(q))/m
        integral = np.mean(fine['energy']*np.exp(1j*theta*lag))
        assert abs(integral-correlation) < 2e-13


def test_finite_boundary_identity_with_resolvable_tail():
    g = finite_geometry(8, 3, 2, 2.)
    coarse = toeplitz_boundary(g, padding=9)
    fine = toeplitz_boundary(g, padding=180)
    assert np.linalg.norm(coarse['approximation']-g['z'][:, 1:-1].T@g['z'][:, 1:-1], 2) <= coarse['error_bound']
    assert coarse['error_bound'] > 1e-7
    assert np.linalg.norm(coarse['toeplitz']-fine['toeplitz'], 2) <= coarse['toeplitz_tail_bound']
    assert np.linalg.norm(coarse['boundary']-fine['boundary'], 2) <= coarse['boundary_tail_bound']
    np.testing.assert_allclose(fine['approximation'], fine['finite'], atol=2e-15)
    assert np.linalg.norm(fine['boundary'], 2) > .01


def test_error_propagation_uses_raw_metric():
    g = finite_geometry(12, 2, 1, 3.)
    z, c = g['z'], g['transform']
    error = np.random.default_rng(391).normal(size=z.shape)*1e-4
    approximate = whiten(z-error, c)
    actual = whiten(z, c)
    measured = np.linalg.norm(actual@actual.T-approximate@approximate.T, 2)
    assert measured <= kernel_error_bound(z-error, error, c)
