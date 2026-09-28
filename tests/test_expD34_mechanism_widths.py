"""Focused checks of ordinary GD and independent-width measurements."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import adam_forces as af, targets
from experiments.expD34_readout_race import mechanism_widths as mw


def test_width_gradient_and_multiple_steps_match_ordinary_gd():
    p = jnp.asarray(np.random.default_rng(33).normal(size=16)*.3)
    x = jnp.asarray(targets.grid(32)); y = jnp.cos(2*x)+.1*x
    np.testing.assert_allclose(mw.gradient(p, x, y), af.field(p, x, y)[0], atol=2e-16)
    state = jax.vmap(lambda a: mw.initial(a, 1/64))(p[None, :])
    result = mw.advance_factory(x, 1/64)(state, y[None, :], 23)
    expected = jax.lax.fori_loop(0, 23, lambda _, a: a-mw.ETA*af.field(a, x, y)[0], p)
    np.testing.assert_allclose(result['p'][0], expected, atol=3e-15)
    movement = (np.abs(expected[:5])-np.abs(p[:5]))/64
    np.testing.assert_allclose(result['positive'][0]-result['negative'][0], movement, atol=1e-17)
    assert not bool(result['failed'][0])


def test_lambda_occupancy_and_readout_normalization():
    p = jnp.array([1., 20., -.2, .4, 2., -3., .1])
    x = jnp.asarray(targets.grid(32)); y = jnp.sin(x)
    state = mw.initial(p, 1/64)
    np.testing.assert_array_equal(state['first_hit'], [-1, 0])
    metric = mw.diagnostic(p, x, y, x, y, 1/64)
    np.testing.assert_allclose(metric['readout_rms_over_h'], np.sqrt(6.5)*64)
    assert float(metric['population_lambda025']) == .5


def test_target_evaluation_retains_original_normalization():
    for name in mw.TARGETS:
        _, _, _, train_scale = mw.data(name, 2048)
        _, _, _, eval_scale = mw.data(name, 8192)
        assert train_scale == eval_scale
