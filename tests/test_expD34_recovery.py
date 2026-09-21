"""Check the new scientific accounting against independent derivatives."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import pytest
import torch

from experiments.expD34_readout_race.recovery import (
    acquisition_distance, concentration, population_upper, state_diagnostics,
)


def state():
    rng = np.random.default_rng(12)
    x = np.linspace(-1, 1, 25)
    return rng.normal(size=(3, 7))*.4, .17, x, np.sin(2*np.pi*x)+.2


def test_gradients_and_block_actions_against_torch():
    z, d, x, y = state()
    scalars, actions = state_diagnostics(z, d, x, y)
    params = torch.tensor(np.r_[z.ravel(), d], dtype=torch.float64, requires_grad=True)
    xx, yy = torch.tensor(x), torch.tensor(y)
    def loss(p):
        a, b, c = p[:-1].reshape(3, -1)
        r = torch.tanh(xx[:, None]*a+b) @ c+p[-1]-yy
        return .5*torch.mean(r*r)
    g = torch.autograd.functional.jacobian(loss, params).numpy()
    H = torch.autograd.functional.hessian(loss, params).numpy()
    np.testing.assert_allclose(actions["gradient"].ravel(), g[:-1], atol=2e-15)
    np.testing.assert_allclose(actions["gradient_d"], g[-1], atol=2e-15)
    np.testing.assert_allclose(actions["Hac_filter"]+actions["Hac_amplitude"], H[:7, 14:21]@g[14:21], atol=2e-15)
    np.testing.assert_allclose(actions["Had_gd"], H[:7, -1]*g[-1], atol=2e-15)
    np.testing.assert_allclose(actions["Haq_gq"], H[:7, :14]@g[:14], atol=2e-15)
    assert scalars["D_c"] == pytest.approx(scalars["D_c_filter"]+scalars["D_c_amplitude"]+scalars["D_c_normalization"])


def test_log_signal_derivative_and_tail_descent():
    z, d, x, y = state()
    s, v = state_diagnostics(z, d, x, y)
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        plus, _ = state_diagnostics(z-h*v["gradient"], d-h*v["gradient_d"], x, y)
        minus, _ = state_diagnostics(z+h*v["gradient"], d+h*v["gradient_d"], x, y)
        derivative = -(np.log(plus["xi"])-np.log(minus["xi"]))/(2*h)
        errors.append(abs(derivative-s["D_c"]-s["D_d"]-s["D_q"]))
    assert errors[-1] < 1e-8
    assert errors[1] < errors[0]/50
    h = 1e-5
    direction = np.zeros_like(z); direction[0] = v["gradient"][0]
    p, _ = state_diagnostics(z-h*direction, d, x, y)
    m, _ = state_diagnostics(z+h*direction, d, x, y)
    assert -(p["tail_loss"]-m["tail_loss"])/(2*h) == pytest.approx(s["tail_loss_slope_descent"], abs=1e-9)


def test_concentration_and_population_bounds():
    assert concentration(np.ones(20)) == dict(top10_share=.1, participation_fraction=1.)
    assert concentration(np.r_[1., np.zeros(19)]) == dict(top10_share=1., participation_fraction=.05)
    assert concentration(np.zeros(3))["top10_share"] is None
    assert acquisition_distance(np.array([.1, .5, 2.]), 1., 2/3) == .5
    assert acquisition_distance(np.array([.1, .5, 2.]), 1., 1.) == pytest.approx(np.sqrt(.9**2+.5**2))
    reference = np.array([.1, -.2, .8, 1.2])
    actual = np.array([1.1, -.5, 1., .9])
    upper = population_upper(reference, np.linalg.norm(actual-reference), 1., .3)
    assert np.mean(abs(actual) >= 1) <= upper <= 1


def test_zero_readout_unresolved_and_crossing_accounting():
    z, d, x, y = state()
    z[2] = 0
    s, _ = state_diagnostics(z, d, x, y)
    assert s["xi"] == 0 and s["D_c"] is None and not s["attribution_resolved"]
    z, d, x, y = state()
    s, v = state_diagnostics(z, d, x, y, eta=100.)
    assert s["crossing_remainder"] > 0
    assert s["delta_mean_gamma"] == pytest.approx(s["positive_step"]-s["negative_step"])
    assert s["delta_mean_gamma"] == pytest.approx(100*s["signed_velocity"]+s["crossing_remainder"])


def test_replay_matches_original_steps_and_accumulates_travel():
    import jax.numpy as jnp
    from experiments.expD34_readout_race import core, targets
    from experiments.expD34_readout_race.replay_recovery import advance_function, initial_state
    z, d, _, _ = state()
    x = targets.grid(32); y = np.sin(2*np.pi*x)
    inputs = dict(x=jnp.array(x), y=jnp.array(y), powers=jnp.array(x[:, None]**np.arange(11)))
    s = initial_state(z[None], np.array([d]))
    out = advance_function(32, .002)(s, jnp.array(y[None]), 30)
    zz, dd = jnp.array(z), jnp.array(d)
    positive, negative, path = np.zeros(7), np.zeros(7), 0.
    for _ in range(30):
        _, _, g, _, _ = core.field(zz, dd, inputs, 0)
        zn, dn = core.update(zz, dd, inputs, 0, .002, 1.)
        delta = np.abs(zn[0])-np.abs(zz[0])
        positive += np.maximum(delta, 0); negative += np.maximum(-delta, 0)
        path += .002*float(jnp.linalg.norm(g[0]))
        zz, dd = zn, dn
    np.testing.assert_allclose(out["z"][0], zz, atol=2e-15)
    np.testing.assert_allclose(out["positive"][0], positive, atol=2e-15)
    np.testing.assert_allclose(out["negative"][0], negative, atol=2e-15)
    assert float(out["path"][0]) == pytest.approx(path, abs=2e-15)
    np.testing.assert_allclose(out["positive"][0]-out["negative"][0], abs(zz[0])-abs(z[0]), atol=2e-15)
