import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from experiments.expD36_frozen_gamma_probe.cap_search import powered_coefficients, objective


def test_powered_selection_surrogate_matches_ordinary_updates():
    rng = np.random.default_rng(88)
    j = rng.normal(size=(31, 9)); y = rng.normal(size=31)
    h = j.T@j; b = j.T@y; eta = .5/np.linalg.norm(h, 2)
    for steps in [0, 1, 2, 3, 7, 64, 177, 1000]:
        theta = np.zeros(9)
        for _ in range(steps):
            theta -= eta*j.T@(j@theta-y)
        powered = np.asarray(powered_coefficients(jnp.asarray(h), jnp.asarray(b), eta, steps))
        np.testing.assert_allclose(powered, theta, atol=2e-14)


def test_selection_gradient_matches_centered_difference():
    x = jnp.linspace(-1., 1., 33); centers = jnp.linspace(-1.2, 1.2, 7)
    y = jnp.sin(2*jnp.pi*x); slopes = jnp.linspace(.3, .8, 7)
    fun = lambda s:objective(s, 8., x, centers, y, 113)
    grad = np.asarray(jax.grad(fun)(slopes))
    for i in [0, 3, 6]:
        direction = np.eye(7)[i]*1e-5
        fd = float((fun(slopes+direction)-fun(slopes-direction))/(2e-5))
        assert grad[i] == pytest.approx(fd, rel=2e-5, abs=2e-6)
