"""Small independent mathematical fixtures for the frozen-gamma probe."""
import numpy as np
from scipy.linalg import qr

from experiments.expD36_frozen_gamma_probe import core


def test_projection_complement_and_independent_basis():
    x = np.linspace(-1, 1, 129)
    rng = np.random.default_rng(7)
    y = rng.normal(size=(len(x), 2))
    raw = core.polynomial_transform(x, 12)
    full, _ = qr(np.polynomial.chebyshev.chebvander(x, 12), mode='full')
    yh = core.transform(raw, y)
    np.testing.assert_allclose(core.transform(raw, yh, transpose=False), y, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(yh, axis=0), np.linalg.norm(y, axis=0))
    q = core.discrete_polynomials(x, 12)
    np.testing.assert_allclose(q.T@q, np.eye(13), atol=2e-14)
    for k in [0, 2, 8, 12]:
        residual = y-full[:, :k+1]@(full[:, :k+1].T@y)
        np.testing.assert_allclose(np.linalg.norm(yh[k+1:], axis=0), np.linalg.norm(residual, axis=0))
        residual2 = y-q[:, :k+1]@(q[:, :k+1].T@y)
        np.testing.assert_allclose(np.linalg.norm(residual2, axis=0), np.linalg.norm(residual, axis=0))
    assert np.linalg.norm(yh[13:]) > 1


def test_access_identity_and_polynomial_control():
    x = np.linspace(-1, 1, 129)
    j = core.design(x, np.linspace(-1.1, 1.1, 15), 4)
    y = np.column_stack([core.target(x, t) for t in ['sine_mix_2_6_10', 'quadratic']])/np.sqrt(len(x))
    raw = core.polynomial_transform(x, 20)
    yh, jh = core.transform(raw, y), core.transform(raw, j)
    e, mu, frob = core.access(yh, jh, 20)
    assert np.max(e[2:, 1]) < 2e-14
    for k in [0, 3, 8]:
        q = core.transform(raw, np.vstack([np.zeros_like(yh[:k+1]), yh[k+1:]]), transpose=False)
        q /= np.linalg.norm(q, axis=0)
        np.testing.assert_allclose(mu[k], np.sum((j.T@q)**2, axis=0), rtol=1e-9, atol=1e-25)
        b = np.linalg.norm(jh[k+1:], 2)**2
        assert np.max(mu[k]) <= b*(1+1e-12)
        assert b <= frob[k]*(1+1e-12)
        analytic = 15*np.exp(2*core.log_feature_envelope(4, np.array([k]))[0])
        assert frob[k] <= analytic*(1+1e-12)


def test_envelope_monotonic_and_repeated_columns():
    k = np.arange(80)
    logs = np.array([core.log_feature_envelope(g, k) for g in [1, 4, 16, 64]])
    assert np.all(np.diff(logs, axis=1) <= 0)
    assert np.all(np.diff(logs, axis=0) >= 0)
    x = np.linspace(-1, 1, 65)
    j = core.design(x, np.zeros(7), 2)
    jh = core.transform(core.polynomial_transform(x, 4), j)
    np.testing.assert_allclose(np.linalg.norm(jh[3:], 2)**2, 7*np.linalg.norm(jh[3:, 1])**2)


def test_one_mode_gd_and_mixed_tail_slack():
    nu, eps = .04, .01
    b = core.bound(np.array([1.]), np.log(np.array([nu])), eps, nu)
    assert b['bound'] == np.ceil(np.log(eps)/np.log(.5))
    pred = core.spectral_hit(np.sqrt([nu]), np.array([1.]), 0., 1., .5/nu, eps)
    assert pred['steps'] == b['bound']
    delta, eps, nu = .01, .0001, 1e-8
    directional_flow = np.log(1/eps)/(1-eps)**2*(delta-eps)**2/nu
    exact_slow_time = np.log(delta/eps)/nu
    np.testing.assert_allclose(exact_slow_time/directional_flow, 5100.5)


def test_null_mass_and_zero_witness():
    result = core.spectral_hit(np.array([1.]), np.array([.8]), .36, 1., .5, .1)
    assert result['status'] == 'retained_model_unattainable'
    assert core.bound(np.array([0.]), np.array([-np.inf]), .01, 1.)['bound'] == 0
    assert core.bound(np.array([1.]), np.array([-np.inf]), .01, 1.)['status'] == 'zero_access'
