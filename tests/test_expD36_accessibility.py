import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from experiments.expD36_ssb_geometry_switching import accessibility as a


@pytest.mark.parametrize('rank_deficient', [False, True])
def test_gain_matches_exact_residual_flow_and_svd(rank_deficient):
    rng = np.random.default_rng(81)
    matrix = rng.normal(size=(12, 5))*.3
    if rank_deficient:
        matrix[:, 3:] = matrix[:, :2]
    r = rng.normal(size=12)
    for tau in [.001, 1., 100.]:
        remain = expm(-tau*matrix@matrix.T)@r
        expected = 1-remain@remain/(r@r)
        assert float(a.gain(jnp.asarray(matrix), jnp.asarray(r), tau)) == pytest.approx(expected, abs=2e-12)
        assert float(a.gain_svd(jnp.asarray(matrix), jnp.asarray(r), tau)) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize('tiny', [False, True])
def test_frechet_derivative_at_repeated_and_zero_eigenvalues(tiny):
    rng = np.random.default_rng(8)
    matrix = np.diag([1., 1., .01, 0., 0.])
    if tiny: matrix *= 1e-8
    direction = rng.normal(size=matrix.shape)*(1e-8 if tiny else 1.)
    r = jnp.asarray(rng.normal(size=5))
    value, slope = jax.jvp(lambda x:a.gain(x, r, 3.), (jnp.asarray(matrix),), (jnp.asarray(direction),))
    epsilon = 1e-4 if tiny else 1e-5
    finite = (a.gain(jnp.asarray(matrix+epsilon*direction), r, 3.)-a.gain(jnp.asarray(matrix-epsilon*direction), r, 3.))/(2*epsilon)
    assert np.isfinite(slope)
    np.testing.assert_allclose(slope, finite, rtol=3e-5, atol=1e-22 if tiny else 1e-9)
    if tiny: assert 0 < float(value) < 1e-14


def test_frozen_residual_mode_displacement():
    jac = np.diag([2., .01])
    metric = np.diag([.5, 3.])
    residual = np.array([0., .4])
    curvature = .0003
    time = 170.
    expected = -(1-np.exp(-curvature*time))/curvature*metric@jac.T@residual
    solution = -np.linalg.solve(jac.T@jac, jac.T@(np.eye(2)-expm(-time*jac@metric@jac.T))@residual)
    np.testing.assert_allclose(solution, expected)


def test_exposure_switch_and_positive_metric():
    beta = a.exposure_beta(-2., 1., 1e-4, .5, .1, 1e-4)
    assert 2/3 < beta < 1
    assert (1-beta)*-2+beta > 0
    assert a.exposure_beta(-2., 1., 1e-4, -1., -2., 1e-4) is None
    h = jnp.diag(jnp.array([.02, 200.]))
    g = jnp.array([2., 1.])
    mixed, scale = a.mix_metric(h, g, beta)
    assert np.min(np.linalg.eigvalsh(mixed)) > 0
    assert float(g@(-mixed@g)) < 0
    np.testing.assert_array_equal(a.mix_metric(h, g, 0.)[0], h)
    np.testing.assert_allclose(np.linalg.norm(scale*g), np.linalg.norm(h@g))
