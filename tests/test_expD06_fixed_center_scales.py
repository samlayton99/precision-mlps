import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
import jax.numpy as jnp

from experiments.expD06_fixed_center_scales import core


def test_reference_envelopes_and_halo_slots():
    g = core.geometry(512)
    assert g.width == 559 and g.radius == 23
    assert g.corrected_halo.sum() == 24
    assert not np.any(g.core & g.corrected_halo)
    np.testing.assert_allclose(g.alpha[1:], g.alpha[:0:-1])
    assert g.alpha[0] == 1 + g.alpha[1:].sum()
    np.testing.assert_allclose(g.ordinary_alpha, 0.008662986733781007)
    np.testing.assert_allclose(g.alpha.max(), 10.763909685873035)


@pytest.mark.parametrize("family", ["xavier", "envelope"])
def test_paired_physical_initialization_and_fixed_centers(family):
    g = core.geometry(512)
    c, gamma = core.initial_physical(g, 3, family)
    x = jnp.linspace(-1, 1, 37)
    predictions = []
    for arm in ["raw", "both"]:
        cs, gs = core.coordinate_scales(g, arm)
        params = core.to_params(c, gamma, cs, gs)
        predictions.append(core.predict(params, x, g.centers, cs, gs))
        changed_gamma = gamma * 1.37
        beta = -changed_gamma * g.centers
        np.testing.assert_allclose(-beta / changed_gamma, g.centers, atol=1e-15)
    np.testing.assert_allclose(*predictions, rtol=2e-13, atol=2e-14)


def test_stable_tanh_derivative_and_finite_difference():
    np.testing.assert_allclose(jax.grad(core.tanh)(20.0), 4 * math.exp(-40), rtol=1e-14)
    g = core.geometry(512)
    c, gamma = core.initial_physical(g, 0, "envelope")
    cs, gs = core.coordinate_scales(g, "both")
    p = core.to_params(c, gamma, cs, gs)
    x = jnp.linspace(-1, 1, 65)
    objective = lambda params: core.loss(params, x, core.target(x, "sine"), g.centers, cs, gs)
    gradient = jax.grad(objective)(p)["slope"]
    j = 14
    eps = 1e-7
    plus = {**p, "slope": p["slope"].at[j].add(eps)}
    minus = {**p, "slope": p["slope"].at[j].add(-eps)}
    np.testing.assert_allclose(gradient[j], (objective(plus) - objective(minus)) / (2 * eps), rtol=2e-6)


@pytest.mark.parametrize("name", ["gd", "adam"])
def test_one_step_coordinate_identities(name):
    g = core.geometry(512)
    c, gamma = core.initial_physical(g, 2, "envelope")
    cs, gs = core.coordinate_scales(g, "both")
    params = core.to_params(c, gamma, cs, gs)
    x = jnp.linspace(-1, 1, g.n + 1)
    y = core.target(x, "sine")
    raw = core.to_params(c, gamma, np.ones_like(c), 1.0)
    grad = jax.grad(core.loss)(raw, x, y, g.centers, np.ones_like(c), 1.0)
    eta_r, eta_g = 1e-5, 1e-7
    state, _ = core.make_chunk(g, name, "sine", samples_per_cell=1, steps=1)(
        core.initial_state(params, core.optimizer(name)), cs, gs, eta_r, eta_g)
    actual_c, actual_gamma = core.physical(state["params"], cs, gs)
    if name == "gd":
        dc = -eta_r * cs**2 * grad["readout"]
        dg = -eta_g * gs**2 * grad["slope"]
    else:
        dc = -eta_r * cs * grad["readout"] / (jnp.abs(grad["readout"]) + 1e-8 / cs)
        dg = -eta_g * gs * grad["slope"] / (jnp.abs(grad["slope"]) + 1e-8 / gs)
    np.testing.assert_allclose(actual_c, c + dc, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(actual_gamma, gamma + dg, rtol=2e-13, atol=1e-15)
    if name == "gd":
        assert np.all(np.asarray(state["lambda_travel"]) <= np.asarray(state["gd_lambda_budget"]) + 1e-15)


def test_halo_metric_ablation_preserves_physical_initialization_and_bias_scale():
    g = core.geometry(512)
    c, gamma = core.initial_physical(g, 1, "envelope")
    full, gs = core.coordinate_scales(g, "both")
    ordinary, _ = core.coordinate_scales(g, "both", "ordinary")
    assert full[0] == ordinary[0]
    assert np.all(ordinary[1:][g.corrected_halo] == math.sqrt(g.ordinary_alpha))
    p = core.to_params(c, gamma, ordinary, gs)
    np.testing.assert_allclose(core.physical(p, ordinary, gs)[0], c)
