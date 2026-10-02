"""Regression tests for the editable geometry and exact training resumption."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.expG02_geometry_reader_codex.engine import GeometryEngine, activation, target_function


def small_engine(**kwargs):
    return GeometryEngine({
        "n_centers": 7, "n_train": 41, "n_test": 61,
        "n_intervals": 6, "halo": 0, "reference_h": 1/3,
        "center_min": -1, "center_max": 1, "readout_init": "xavier",
        **kwargs,
    })


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "relu", "gelu", "swish"])
def test_analytic_gradient_matches_finite_difference(name):
    engine = small_engine(activation=name)
    # Avoid ReLU's nondifferentiable lattice knots.
    engine.params["b"] += 0.017
    _, analytic = engine._loss_and_gradients()
    for key in engine.params:
        for index in range(engine.params[key].size):
            original = engine.params[key][index]
            eps = 2e-6
            engine.params[key][index] = original + eps
            plus, _ = engine._loss_and_gradients()
            engine.params[key][index] = original - eps
            minus, _ = engine._loss_and_gradients()
            engine.params[key][index] = original
            assert analytic[key][index] == pytest.approx((plus-minus)/(2*eps), abs=2e-8, rel=2e-5)


def test_solve_recovers_realizable_target_with_negative_slopes():
    engine = small_engine(target="0.7*tanh(2*x-0.4)-0.2*tanh(-x+0.5)+0.3", rcond=1e-14)
    engine.initialize({"slopes": [2.0, -1.0], "hidden_bias": [-0.4, 0.5]})
    frame = engine.evaluate()
    assert frame["metrics"]["solved"]["linf"] < 2e-15
    assert frame["solved_readout"] == pytest.approx([0.7, -0.2], abs=2e-14)
    assert frame["solved_output_bias"] == pytest.approx(0.3, abs=2e-14)
    json.dumps(frame, allow_nan=False)


def test_geometry_edits_preserve_orthogonal_coordinate_and_sign():
    engine = small_engine()
    engine.params["w"][2] *= -1
    engine.params["b"][2] *= -1
    before_lambda = engine.lambdas.copy()
    engine.edit("center", 2, 0.73)
    np.testing.assert_allclose(engine.lambdas, before_lambda)
    assert engine.centers[2] == pytest.approx(0.73)
    before_centers = engine.centers.copy()
    engine.edit("lambda", 2, 0.91)
    np.testing.assert_allclose(engine.centers, before_centers)
    assert engine.lambdas[2] == pytest.approx(0.91)
    assert engine.signs[2] == -1
    engine.edit("mean_lambda", None, 0.08)
    assert engine.lambdas.mean() == pytest.approx(0.08, abs=1e-12)
    assert np.min(engine.lambdas) > 0
    np.testing.assert_allclose(engine.centers, before_centers)


def test_clean_interpolates_and_resize_keeps_jumbled_profile():
    engine = small_engine()
    engine.initialize({"centers": [-1, -0.4, 0.5, -0.3, 0.7, 0.2, 1],
                       "lambdas": [0.1, 0.15, 0.3, 0.7, 0.2, 0.35, 0.6]})
    old_centers, old_lambdas = engine.centers.copy(), engine.lambdas.copy()
    old_h = engine.config["reference_h"]
    engine.resize(13)
    new_h = engine.config["reference_h"]
    assert new_h == pytest.approx(old_h / 2)
    interpolated = np.interp(np.linspace(0, 1, 13), np.linspace(0, 1, 7), old_lambdas)
    height_scale = old_lambdas.mean() / interpolated.mean()
    np.testing.assert_allclose(engine.lambdas[::2], old_lambdas * height_scale)
    assert engine.lambdas.mean() == pytest.approx(old_lambdas.mean(), rel=2e-15)
    old_deviation = (old_centers - np.linspace(-1, 1, 7)) / old_h
    new_deviation = (engine.centers[::2] - np.linspace(-1, 1, 7)) / new_h
    np.testing.assert_allclose(new_deviation, old_deviation)
    before = engine.centers.copy()
    before_lam = engine.lambdas.copy()
    order = np.argsort(before)
    engine.transform("clean", 0.5)
    np.testing.assert_allclose(engine.centers[order], (before[order]+np.linspace(-1, 1, 13))/2)
    np.testing.assert_allclose(engine.lambdas, (before_lam+0.25)/2)
    engine.transform("clean", 1)
    np.testing.assert_allclose(np.sort(engine.centers), np.linspace(-1, 1, 13))
    np.testing.assert_allclose(engine.lambdas, 0.25)


def test_solve_feedback_cannot_erase_manual_readout_before_gradient():
    engine = small_engine(ls_feedback=True)
    engine.evaluate(feedback=True)
    engine.edit("readout", 2, 9.0)
    manual = engine.state_dict()
    frame = engine.evaluate(feedback=True)
    assert not frame["feedback_applied"]
    assert frame["adam_readout"][2] == 9.0
    assert engine.params["v"][2] == 9.0
    expected = small_engine().load_state(manual)
    expected.step()
    engine.step()
    np.testing.assert_array_equal(engine.params["w"], expected.params["w"])
    assert engine.params["v"][2] != 9.0
    frame = engine.evaluate(feedback=True)
    assert frame["feedback_applied"]
    np.testing.assert_array_equal(engine.params["v"], frame["solved_readout"])
    assert frame["adam_readout"][2] != frame["solved_readout"][2]


def test_exact_resume_including_moments_random_stream_and_feedback():
    engine = small_engine(noise=0.01, sampling="uniform")
    engine.step(8)
    frame = engine.evaluate(feedback=True)
    # JSON round-trip is the actual storage boundary, not a shared Python object.
    restored = small_engine().load_state(json.loads(json.dumps(frame["state"])))
    for candidate in (engine, restored):
        candidate.step(4)
        candidate.transform("jitter_centers", 0.18)
        candidate.transform("jitter_lambda", 0.12)
        candidate.step(3)
    assert restored.state_dict() == engine.state_dict()


def test_steps_do_not_implicitly_solve_and_fixed_groups_stay_fixed(monkeypatch):
    engine = small_engine(train_geometry=False)
    w, b = engine.params["w"].copy(), engine.params["b"].copy()
    def forbidden():
        raise AssertionError("A train iteration must not solve implicitly")
    monkeypatch.setattr(engine, "solve", forbidden)
    loss_before, _ = engine._loss_and_gradients()
    engine.step(20)
    loss_after, _ = engine._loss_and_gradients()
    np.testing.assert_array_equal(engine.params["w"], w)
    np.testing.assert_array_equal(engine.params["b"], b)
    assert loss_after < loss_before
    assert engine.step_count == 20


def test_catalog_preserves_original_reference_h_and_optional_problem_settings():
    preset = {
        "settings": {"N": 128, "h": 1/64, "halo": 70, "target": "exp(x)", "n_eval": 101},
        "geometry": {"centers": [-2, 0, 2], "slopes": [2, -3, 4], "biases": [4, 0, -8]},
        "readout": {"saved": [1, 2, 3], "saved_bias": 0.7},
    }
    engine = small_engine(readout_init="zero")
    engine.initialize(preset)
    assert engine.config["target"] == "sin(2*pi*x)"
    assert engine.config["reference_h"] == 1/64
    np.testing.assert_array_equal(engine.lambdas, np.array([2, 3, 4])/64)
    np.testing.assert_array_equal(engine.params["v"], [0, 0, 0])
    engine.initialize(preset, apply_settings=True)
    assert engine.config["target"] == "exp(x)"
    assert engine.test_x.size == 101


@pytest.mark.parametrize("text", ["__import__('os')", "x.__class__", "[x][0]", "sin(x, x)", "2**1000000000", "1/0"])
def test_target_language_rejects_python_and_nonfinite_math(text):
    with pytest.raises(ValueError):
        target_function(text)(np.array([0.2, 0.3]))


def test_invalid_configuration_does_not_destroy_current_session():
    engine = small_engine()
    engine.step(2)
    before = engine.state_dict()
    with pytest.raises(ValueError):
        engine.configure({"mask_enabled": True, "mask_min": -2, "mask_max": 2})
    assert engine.state_dict() == before


def test_saturated_hidden_units_keep_their_small_analytic_gradient():
    z = np.array([-30.0, 30.0])
    # 1-tanh(z)^2 has already rounded to zero here, although the derivative is nonzero.
    assert np.all(1-np.tanh(z)**2 == 0)
    np.testing.assert_allclose(activation(z, "tanh", True), 4*np.exp(-60), rtol=1e-14)
    assert activation(np.array([50.0]), "sigmoid", True)[0] > 0


def test_readout_reset_keeps_geometry_and_survives_initial_feedback():
    engine = small_engine(readout_init="zero")
    engine.evaluate(feedback=True)
    w, b = engine.params["w"].copy(), engine.params["b"].copy()
    engine.transform("reset_readout", 1)
    frame = engine.evaluate(feedback=True)
    assert not frame["feedback_applied"]
    np.testing.assert_array_equal(engine.params["v"], 0)
    np.testing.assert_array_equal(engine.params["w"], w)
    np.testing.assert_array_equal(engine.params["b"], b)


def test_ridge_solve_matches_augmented_least_squares():
    engine = small_engine(ridge=0.2, rcond=0)
    v, bias, _ = engine.solve()
    design = np.column_stack((engine._features(engine.train_x), np.ones(engine.train_x.size)))
    augmented = np.vstack((design, np.sqrt(0.2)*np.eye(design.shape[1])))
    target = np.concatenate((engine.train_y, np.zeros(design.shape[1])))
    expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
    np.testing.assert_allclose(np.r_[v, bias], expected, atol=5e-15)


def test_gamma_handles_preserve_centers_and_signed_slope_profile():
    engine = small_engine()
    engine.initialize({"slopes": [-0.5, 2.0, -4.0], "hidden_bias": [-0.2, 0.6, 0.8]})
    centers = engine.centers.copy()
    engine.edit("gamma", 0, 3.0)
    np.testing.assert_allclose(engine.centers, centers)
    np.testing.assert_array_equal(engine.gammas, [3, 2, 4])
    assert engine.params["w"][0] == -3
    profile = engine.gammas.copy()
    engine.edit("mean_gamma", None, 12.0)
    np.testing.assert_allclose(engine.gammas, profile * 4)
    np.testing.assert_allclose(engine.centers, centers)
    np.testing.assert_array_equal(engine.signs, [-1, 1, -1])
    gamma_before = engine.gammas.copy()
    engine.edit("center", 0, 0.37)
    np.testing.assert_array_equal(engine.gammas, gamma_before)
    assert engine.centers[0] == pytest.approx(0.37)
    frame = engine.evaluate()
    assert frame["gammas"] == pytest.approx(gamma_before)
    assert frame["mean_gamma"] == pytest.approx(12)
    assert frame["ideal_gamma"] == pytest.approx(engine.config["clean_lambda"] / engine.config["reference_h"])
    assert not frame["feedback_applied"]


def test_gamma_jitter_matches_lambda_relative_jitter():
    engine = small_engine()
    other = small_engine().load_state(engine.state_dict())
    engine.transform("jitter_gamma", 0.2)
    other.transform("jitter_lambda", 0.2)
    np.testing.assert_array_equal(engine.gammas, other.gammas)
    np.testing.assert_array_equal(engine.centers, other.centers)


def test_least_squares_feedback_is_opt_in():
    engine = small_engine()
    original = engine.params["v"].copy()
    frame = engine.evaluate(feedback=True)
    assert not frame["feedback_applied"]
    np.testing.assert_array_equal(engine.params["v"], original)


def test_resize_preserves_mean_lambda_and_scales_mean_gamma_with_density():
    engine = small_engine()
    engine.initialize({"centers": [-1, -0.1, -0.4, 0.3, 0.2, 0.8, 1],
                       "lambdas": [0.02, 0.13, 0.74, 0.06, 0.23, 1.2, 0.04]})
    original_mean_lambda = float(engine.lambdas.mean())
    original_mean_gamma = float(engine.gammas.mean())
    original_h = engine.config["reference_h"]
    for count in (12, 27, 5, 19, 7):
        engine.resize(count)
        assert engine.lambdas.mean() == pytest.approx(original_mean_lambda, rel=2e-15)
        expected_gamma = original_mean_gamma * original_h / engine.config["reference_h"]
        assert engine.gammas.mean() == pytest.approx(expected_gamma, rel=2e-15)
        if count > 7:
            assert engine.gammas.mean() > original_mean_gamma
        else:
            assert engine.gammas.mean() <= original_mean_gamma * (1 + 2e-15)


def test_explicit_reference_spacing_change_preserves_all_centers_and_lambdas():
    engine = small_engine()
    engine.step(3)
    centers, lambdas, gammas = engine.centers.copy(), engine.lambdas.copy(), engine.gammas.copy()
    original_h = engine.config["reference_h"]
    engine.configure({"reference_h": original_h/2}, reset=False)
    np.testing.assert_allclose(engine.centers, centers, rtol=2e-15)
    np.testing.assert_allclose(engine.lambdas, lambdas, rtol=2e-15)
    np.testing.assert_allclose(engine.gammas, 2*gammas, rtol=2e-15)
    assert engine.step_count == 3
    assert engine.optimizer_step == 0


def test_midpoint_samples_have_correct_offsets_and_exact_eval_count():
    engine = small_engine(sampling="midpoint", test_sampling="midpoint", n_test=60, prime_test_grid=False)
    assert engine.train_x[0] == pytest.approx(-1+1/41)
    assert engine.train_x[-1] == pytest.approx(1-1/41)
    assert engine.test_x.size == 60
    assert engine.test_x[0] == pytest.approx(-1+1/60)
    assert engine.test_x[-1] == pytest.approx(1-1/60)


def test_every_actual_catalog_preset_applies_research_settings_and_solves():
    catalog_path = Path(__file__).resolve().parents[1] / "experiments/expG02_geometry_reader_codex/presets/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    measured = {}
    for preset in catalog["presets"]:
        engine = small_engine()
        engine.initialize(preset, apply_settings=True)
        settings = preset["settings"]
        assert engine.config["target"] == "sin(2*pi*x)"
        assert engine.test_x.size == settings["n_eval"]
        assert engine.train_x.size == settings["n_train"]
        if settings["sampling"] == "uniform":
            np.testing.assert_array_equal(engine.train_x, np.linspace(-1, 1, settings["n_train"]))
        else:
            assert engine.config["sampling"] == "midpoint"
            assert engine.test_x[0] > -1
            assert engine.test_x[-1] < 1
        frame = engine.evaluate()
        json.dumps(frame, allow_nan=False)
        measured[preset["id"]] = frame["metrics"]["solved"]["relative_l2"]
    assert measured["xavier_sine_floor"] < 3e-14
    assert measured["clean_qi"] < 2e-13


def test_default_halo_is_square_root_with_minimum_ten():
    engine = GeometryEngine()
    assert engine.config["n_interior"] == 64
    assert engine.config["n_intervals"] == 63
    assert engine.config["halo"] == 10
    assert engine.config["n_centers"] == 84
    np.testing.assert_allclose(engine.centers[[0, -1]], [-1-20/63, 1+20/63])


def test_auto_halo_resize_obeys_rule_without_losing_lambda_or_jumble():
    engine = GeometryEngine()
    engine.edit("center", 40, 0.22)
    engine.edit("gamma", 40, 21)
    mean_lambda = engine.lambdas.mean()
    for count in (100, 101, 144, 2, 64):
        before_n = engine.config["n_centers"]
        before = (engine.centers - np.linspace(engine.config["center_min"],
                   engine.config["center_max"], before_n)) / engine.config["reference_h"]
        engine.resize(count)
        cfg = engine.config
        assert cfg["n_interior"] == count
        assert cfg["halo"] == max(10, int(np.ceil(np.sqrt(count))))
        assert cfg["n_centers"] == count + 2*cfg["halo"]
        assert cfg["n_intervals"] == count-1
        total = cfg["n_centers"]
        assert cfg["reference_h"] == pytest.approx(2/cfg["n_intervals"])
        assert engine.lambdas.mean() == pytest.approx(mean_lambda, rel=3e-15)
        actual = (engine.centers - np.linspace(cfg["center_min"], cfg["center_max"], total)) / cfg["reference_h"]
        np.testing.assert_allclose(actual, np.interp(np.linspace(0, 1, total),
                                   np.linspace(0, 1, before_n), before), atol=3e-14)
    engine.transform("clean", 1)
    np.testing.assert_allclose(np.sort(engine.centers), np.linspace(cfg["center_min"], cfg["center_max"], total))


def test_old_snapshot_keeps_its_recorded_halo():
    engine = small_engine()
    state = engine.state_dict()
    state["config"].pop("auto_halo")
    state["config"].pop("n_interior")
    loaded = GeometryEngine().load_state(state)
    assert loaded.config["auto_halo"] is False
    assert loaded.config["halo"] == 0
    assert loaded.config["n_interior"] == 7
    for key in state["params"]:
        np.testing.assert_array_equal(loaded.params[key], state["params"][key])


def test_clean_preset_uses_new_rule_and_historical_presets_keep_sources():
    path = Path(__file__).resolve().parents[1] / "experiments/expG02_geometry_reader_codex/presets/catalog.json"
    catalog = json.loads(path.read_text())["presets"]
    engine = GeometryEngine()
    engine.initialize(next(p for p in catalog if p["id"] == "clean_qi"))
    assert engine.config["auto_halo"] is True
    assert engine.config["n_centers"] == 168
    assert engine.config["n_interior"] == 144
    assert engine.config["halo"] == 12
    engine.initialize(next(p for p in catalog if p["id"] == "xavier_sine_floor"))
    assert engine.config["auto_halo"] is False
    assert engine.config["halo"] == 102
    assert engine.config["n_interior"] == 257


@pytest.mark.parametrize("count", [1, 100.5, True, 1959])
def test_auto_halo_rejects_invalid_interior_count_without_mutation(count):
    engine = GeometryEngine()
    before = engine.state_dict()
    with pytest.raises(ValueError, match="Interior center count"):
        engine.resize(count)
    assert engine.state_dict() == before
    custom = small_engine()
    custom.initialize("xavier")
    assert custom.config["n_centers"] == 7 and not custom.config["auto_halo"]


@pytest.mark.parametrize("count,expected_r,total", [(2, 10, 22), (64, 10, 84), (100, 10, 120), (101, 11, 123), (144, 12, 168), (256, 16, 288)])
def test_auto_halo_matches_both_sides_of_max_rule(count, expected_r, total):
    engine = GeometryEngine({"n_interior": count})
    assert engine.config["n_interior"] == count
    assert engine.config["n_intervals"] == count-1
    assert engine.config["halo"] == expected_r
    assert len(engine.centers) == total


def test_one_hundred_interior_points_has_gamma_twelve_point_three_seven_five():
    engine = GeometryEngine()
    engine.resize(100)
    assert engine.config["n_interior"] == 100
    assert engine.config["n_centers"] == 120
    assert engine.config["halo"] == 10
    assert engine.config["reference_h"] == pytest.approx(2/99)
    frame = engine.evaluate()
    assert frame["ideal_gamma"] == pytest.approx(12.375)
    np.testing.assert_allclose(engine.gammas, 12.375)
    np.testing.assert_allclose(engine.centers[[10, 109]], [-1, 1])
    assert np.count_nonzero((engine.centers >= -1-1e-14) & (engine.centers <= 1+1e-14)) == 100
