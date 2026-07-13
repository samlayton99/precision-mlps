import math

import numpy as np

from experiments.expD05_scale_init_story import run as exp


def test_all_initializers_are_finite_fp64():
    geometry = exp.geometry_for_resolution(16)
    for family in exp.DEFAULT_FAMILIES:
        state = exp.build_initial_state(family, "sine", geometry, seed=0)
        assert state.input_weights.dtype == np.float64
        assert state.input_biases.dtype == np.float64
        assert state.readout_weights.dtype == np.float64
        assert state.input_weights.shape == (geometry.width,)
        assert np.all(np.isfinite(state.input_weights))
        assert np.all(np.isfinite(state.input_biases))
        assert np.all(np.isfinite(state.readout_weights))
        pred = exp.predict_np(
            np.linspace(-1.0, 1.0, 9),
            state.input_weights,
            state.input_biases,
            state.readout_weights,
            state.readout_bias,
        )
        assert np.all(np.isfinite(pred))


def test_qi_scale_only_matches_requested_lambda():
    geometry = exp.geometry_for_resolution(32)
    state = exp.build_initial_state("qi_scale_only_l030", "sine", geometry, seed=0)
    _centers, gammas, valid = exp.canonical_semantic(state.input_weights, state.input_biases)
    assert np.all(valid)
    lambdas = gammas * geometry.h
    np.testing.assert_allclose(lambdas, 0.30, rtol=1e-13, atol=1e-13)
    assert set(state.requested_lambdas) == {0.30}


def test_scale_corrected_xavier_sets_median_lambda():
    geometry = exp.geometry_for_resolution(32)
    state = exp.build_initial_state("scale_corrected_xavier_l025", "sine", geometry, seed=1)
    _centers, gammas, valid = exp.canonical_semantic(state.input_weights, state.input_biases)
    assert np.all(valid)
    assert math.isclose(float(np.median(gammas * geometry.h)), 0.25, rel_tol=1e-13, abs_tol=1e-13)
    np.testing.assert_allclose(np.median(gammas * geometry.h), 0.25, rtol=1e-13, atol=1e-13)


def test_uniform_qi_geometry_refit_reaches_sine_floor():
    geometry = exp.geometry_for_resolution(32)
    state = exp.build_initial_state("qi_scale_only_l030", "sine", geometry, seed=0)
    x_train = np.linspace(-1.0, 1.0, 257)
    x_eval = np.linspace(-1.0, 1.0, 513)
    target = exp.get_target("sine")
    metrics = exp.refit_readout_metrics(
        state.input_weights,
        state.input_biases,
        x_train,
        target.fn_numpy(x_train),
        x_eval,
        target.fn_numpy(x_eval),
    )
    assert metrics["refit_eval_rel_l2"] < 1e-10


def test_run_matrix_row_count():
    config = exp.RunConfig(
        targets=("sine", "runge"),
        resolutions=(16, 32),
        seeds=(0, 1),
        families=("standard_xavier_affine", "qi_scale_only_l030"),
        steps=1,
        eval_interval=1,
        learning_rate=1e-3,
        n_train=33,
        n_eval=65,
        mode="test",
    )
    cases = exp.make_cases(config)
    assert len(cases) == 2 * 2 * 2 * 2
    rows = exp.run_matrix_rows(cases, config)
    assert len(rows) == len(cases)


def test_sharding_is_stable_and_disjoint():
    config = exp.RunConfig(
        targets=("sine", "runge"),
        resolutions=(16, 32),
        seeds=(0, 1),
        families=("standard_xavier_affine", "qi_scale_only_l030"),
        steps=1,
        eval_interval=1,
        learning_rate=1e-3,
        n_train=33,
        n_eval=65,
        mode="test",
    )
    cases = exp.make_cases(config)
    shard0 = exp.select_shard(cases, 0, 2)
    shard1 = exp.select_shard(cases, 1, 2)
    assert shard0 == cases[0::2]
    assert shard1 == cases[1::2]
    assert {exp.case_key(case) for case in shard0}.isdisjoint({exp.case_key(case) for case in shard1})
    assert len(shard0) + len(shard1) == len(cases)


def test_figure_html_uses_relative_paths():
    html = exp.figure_html("plot.png", "caption")
    assert 'src="figures/plot.png"' in html
    assert "/data/home" not in html
    assert "file://" not in html


def test_refit_does_not_mutate_state():
    geometry = exp.geometry_for_resolution(16)
    state = exp.build_initial_state("center_spread_xavier", "sine", geometry, seed=2)
    before = (
        state.input_weights.copy(),
        state.input_biases.copy(),
        state.readout_weights.copy(),
        state.readout_bias,
    )
    x_train = np.linspace(-1.0, 1.0, 65)
    x_eval = np.linspace(-1.0, 1.0, 129)
    target = exp.get_target("sine")
    _ = exp.refit_readout_metrics(
        state.input_weights,
        state.input_biases,
        x_train,
        target.fn_numpy(x_train),
        x_eval,
        target.fn_numpy(x_eval),
    )
    np.testing.assert_allclose(state.input_weights, before[0])
    np.testing.assert_allclose(state.input_biases, before[1])
    np.testing.assert_allclose(state.readout_weights, before[2])
    assert state.readout_bias == before[3]
