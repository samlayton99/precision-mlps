"""Independent derivatives and directional identities for temporal windows."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import population_window as window
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel


def state():
    p = jnp.array([.15, -.22, .12, .04, .3, -.12, .07])
    x = jnp.linspace(-1, 1, 31)
    return p, x, jnp.sin(3*x)+.1*x*x


def test_loaded_operator_matches_independent_hessian_and_is_symmetric():
    p, x, y = state()
    s = kernel.decomposition(p, x, y)
    load = jax.lax.stop_gradient(s['eH']-s['basis']@s['balance'])
    hessian = jax.hessian(lambda z: jnp.mean(load*kernel.output(z, x)))(p)
    jh = s['J']-s['basis']@(s['basis'].T@s['J']/len(x))
    expected = jh.T@jh/len(x)+hessian
    actual = jax.vmap(lambda v: window.loaded_operator(p, x, y, v))(jnp.eye(len(p))).T
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(actual, actual.T, rtol=2e-13, atol=2e-15)


def test_directional_split_matches_derivative_of_log_force_and_tracking():
    p, x, y = state()
    s = kernel.decomposition(p, x, y)
    logq = lambda z: jnp.log(jnp.linalg.norm(kernel.effective(z, x, y)))
    k = lambda z: jax.jvp(logq, (z,), (-kernel.effective(z, x, y),))[1]
    measured = window.diagnostics(p, x, y)
    assert float(measured['kappa']) == pytest.approx(float(k(p)), rel=2e-12)
    for key, direction in [('kappa_dot', -s['F']), ('tracking_kappa_drift', -s['R'])]:
        expected = jax.jvp(k, (p,), (direction,))[1]
        assert float(measured[key]) == pytest.approx(float(expected), rel=2e-11, abs=2e-14)
    expected = jax.jvp(logq, (p,), (-s['R'],))[1]
    assert float(measured['tracking_log_rate']) == pytest.approx(float(expected), rel=2e-12)
    assert abs(float(measured['rate_identity_error'])) < 2e-14


def test_centered_flow_difference_converges_to_acceleration():
    p, x, y = state()
    expected = float(window.diagnostics(p, x, y)['kappa_dot'])
    errors = []
    for dt in (.2, .1, .05):
        before = window.advance(p, x, y, -dt, 1, 'effective')
        after = window.advance(p, x, y, dt, 1, 'effective')
        difference = (window.effective_rate(after, x, y)-window.effective_rate(before, x, y))/(2*dt)
        errors.append(abs(float(difference)-expected))
    assert errors[0]/errors[1] == pytest.approx(4, rel=.08)
    assert errors[1]/errors[2] == pytest.approx(4, rel=.08)


def test_zero_force_is_not_reported_as_zero_reinforcement():
    p, x, _ = state()
    result = window.diagnostics(p, x, kernel.output(p, x))
    assert not bool(result['force_resolved'])
    assert np.isnan(float(result['kappa']))
