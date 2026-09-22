"""Verify discrete-measure quadrature against its original sample sum."""
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD34_readout_race import persistence as pe, persistence_quadrature as pq, targets


def test_discrete_moments_and_not_continuous_gauss():
    x = targets.grid(2048); z, mass = pq.empirical_rule(x, 64)
    expected = np.polynomial.legendre.legvander(x, 127).mean(axis=0)
    actual = mass @ np.polynomial.legendre.legvander(z, 127)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=1e-10)
    assert abs(mass @ (z*z)-1/3) > 1e-8
    assert np.all(mass > 0)


@pytest.mark.parametrize('model', ('full', 'five_mode', 'ten_mode'))
def test_actual_small_slope_gradients(model):
    rng = np.random.default_rng(411); x = targets.grid(2048); z, mass = pq.empirical_rule(x)
    mapping = targets.polynomial_map(x); y = targets.values('moment9', x, mapping)
    yz = targets.values('moment9', z, mapping)
    columns = [0, 1, 2, 3, 9] if model == 'five_mode' else list(range(10))
    q = np.polynomial.legendre.legvander(x, 9) @ mapping
    qz = np.polynomial.legendre.legvander(z, 9) @ mapping
    if model == 'full': q, qz = np.empty((len(x), 0)), np.empty((len(z), 0))
    else: q, qz = q[:, columns], qz[:, columns]
    for _ in range(3):
        p = rng.uniform(-.3, .3, 532)
        full = pe.field(p, p, x, y, q, model)
        reduced = pe.field(p, p, z, yz, qz, model, mass)
        np.testing.assert_allclose(reduced[0], full[0], rtol=3e-11, atol=2e-14)
        np.testing.assert_allclose(reduced[2], full[2], rtol=3e-11, atol=2e-14)
        e = qz.T @ (mass*np.asarray(reduced[1]))
        assert pq.analytic_force_remainder(p, x, 64, model, e) < 1e-40
