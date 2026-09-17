import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
import jax.numpy as jnp

from experiments.expD06_fixed_center_scales import core
from experiments.expD06_fixed_center_scales import diagnostics
from experiments.expD06_fixed_center_scales import run
from experiments.expD06_fixed_center_scales import campaign


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


def test_fourier_split_reconstructs_energy_and_signed_gradients():
    rng = np.random.default_rng(18)
    for size in [32, 33]:
        r = rng.normal(size=size)
        a = rng.normal(size=(size, 4))
        j = rng.normal(size=(size, 6))
        u, _ = np.linalg.qr(a)
        pj = u @ (u.T @ j)
        out = diagnostics.gradient_bands(r, a, j, pj)
        np.testing.assert_allclose(out["band_energy"].sum(), r @ r, rtol=1e-14)
        np.testing.assert_allclose(out["band_gradient_lambda"].sum(axis=0), j.T @ r, atol=1e-14)
        np.testing.assert_allclose(out["band_gradient_parallel"].sum(axis=0), pj.T @ r, atol=1e-14)
        np.testing.assert_allclose(out["band_gradient_readout"].sum(axis=0), a.T @ r, atol=1e-14)
        np.testing.assert_allclose(out["band_gradient_lambda"], out["band_gradient_parallel"]
                                   + out["band_gradient_perpendicular"], atol=1e-14)


def test_detached_refit_and_gradient_diagnostics():
    rng = np.random.default_rng(4)
    centers = np.linspace(-1.1, 1.1, 9)
    gamma = np.linspace(1, 5, 9)
    c = rng.normal(size=10)
    x = np.linspace(-1, 1, 65)
    y = core.target(x, "sine", np)
    d = np.linspace(0.5, 2, 10)
    dc, dl = rng.normal(size=10) * 1e-5, rng.normal(size=9) * 1e-5
    before = [v.copy() for v in (c, gamma, d)]
    out = diagnostics.checkpoint_arrays(x, y, centers, 0.25, d, c, gamma, {},
                                        delta_c=dc, delta_lambda=dl)
    for old, current in zip(before, (c, gamma, d)):
        np.testing.assert_array_equal(old, current)
    assert np.linalg.norm(out["residual_refit"]) < np.linalg.norm(out["residual_train"])
    np.testing.assert_allclose(out["residual_parallel"] + out["residual_perpendicular"],
                               out["residual_train"], atol=1e-14)
    np.testing.assert_allclose(out["prediction_change_measured"], out["prediction_change_readout"]
                               + out["prediction_change_geometry"] + out["prediction_change_interaction"], atol=1e-14)
    params = core.to_params(c, gamma, d, 4.0)
    grad = jax.grad(core.loss)(params, jnp.asarray(x), jnp.asarray(y), centers, d, 4.0)
    np.testing.assert_allclose(out["gradient_lambda"], grad["slope"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(out["gradient_readout"], grad["readout"] / d, rtol=1e-12, atol=1e-13)
    direction = np.sign(gamma) / np.sqrt(len(gamma))
    tangent = c[1:] * (x[:, None] - centers) / .25
    tangent *= 1 / np.cosh((x[:, None] - centers) * gamma)**2
    tangent = tangent @ direction / np.sqrt(len(x))
    phase = 2 * np.pi * 3 * np.arange(len(x)) / len(x)
    probe = np.sqrt(2 / len(x)) * np.sin(phase)
    np.testing.assert_allclose(out["probe_raw_all_signed_sin"][3], probe @ tangent, atol=1e-14)
    np.testing.assert_allclose(out["force_all"][:2].sum(), -out["gradient_lambda"] @ direction, atol=1e-13)


def test_scientific_runner_enforces_twenty_thousand_steps(tmp_path):
    with pytest.raises(ValueError, match="20,000"):
        run.run_batch([run.Case()], tmp_path, 19999)
    assert run.convergence_status([{"step": 19999, "finite": True}]) == "continuing"


@pytest.mark.parametrize("optimizer", ["gd", "adam"])
def test_checkpoint_resume_matches_uninterrupted(optimizer, tmp_path):
    g = core.geometry(128)
    c, gamma = core.initial_physical(g, 7, "xavier")
    cs, gs = core.coordinate_scales(g, "both")
    state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer(optimizer))
    chunk = core.make_chunk(g, optimizer, "sine", samples_per_cell=1, steps=7)
    after_seven, _ = chunk(state, cs, gs, 1e-5, 1e-6)
    path = tmp_path / "checkpoint.pkl"
    run.save_state(path, after_seven, 7)
    restored, step = run.load_state(path)
    assert step == 7
    continued, _ = chunk(restored, cs, gs, 1e-5, 1e-6)
    uninterrupted, _ = chunk(after_seven, cs, gs, 1e-5, 1e-6)
    for a, b in zip(jax.tree.leaves(continued), jax.tree.leaves(uninterrupted)):
        np.testing.assert_array_equal(a, b)


def test_batched_runs_are_independent():
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    states = []
    for seed in [0, 1]:
        c, gamma = core.initial_physical(g, seed, "xavier")
        states.append(core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam")))
    batch = core.make_chunk(g, "adam", "sine", 1, 5, batched=True)
    result, _ = batch(run.stack_states(states), jnp.stack([cs, cs]), jnp.array([gs, gs]),
                      jnp.array([1e-4, 1e-3]), jnp.array([1e-6, 1e-5]))
    single = core.make_chunk(g, "adam", "sine", 1, 5)
    for i, (rr, rg) in enumerate([(1e-4, 1e-6), (1e-3, 1e-5)]):
        expected, _ = single(states[i], cs, gs, rr, rg)
        for a, b in zip(jax.tree.leaves(run.unstack_state(result, i)), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-15)


def test_rate_manifest_covers_native_and_scale_matched_controls():
    for optimizer in ["gd", "adam"]:
        initial = campaign.pilot_manifest(optimizer)
        expanded = campaign.pilot_manifest(optimizer, expanded=True)
        assert len(initial) == 84 and len(expanded) == 212
        assert len({r["key"] for r in expanded}) == len(expanded)
        assert {r["key"] for r in initial} <= {r["key"] for r in expanded}
        g = core.geometry(512)
        raw = next(r for r in initial if r["case"]["arm"] == "raw" and r["grid"] == "ordinary_update_matched")
        expected = raw["base_lr"] * raw["bandwidth_to_readout_ratio"] * g.h**(-2 if optimizer == "gd" else -1)
        assert raw["case"]["rate_g"] == expected
        assert {r["case"]["seed"] for r in expanded} == {0, 1}


def test_nonfinite_checkpoint_is_recorded_as_failure(tmp_path):
    case = run.Case()
    g = core.geometry(case.n)
    c, gamma = core.initial_physical(g, 0, "xavier")
    cs, gs = core.coordinate_scales(g, "both")
    state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    arrays = {"c": c, "gamma": gamma, "prediction_train": np.full(3, 1e200)}
    row = run.record_checkpoint(tmp_path, case, state, 2, arrays)
    assert row == {"step": 2, "finite": False}
    assert run.convergence_status([row]) == "nonfinite"
    assert (tmp_path / case.key / "state_000000002.pkl").exists()
    with np.load(tmp_path / case.key / "checkpoint_000000002.npz") as stored:
        np.testing.assert_array_equal(stored["c"], c)
