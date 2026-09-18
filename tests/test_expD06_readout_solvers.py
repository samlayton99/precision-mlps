import numpy as np
from experiments.expD06_fixed_center_scales import core, diagnostics, readout_solvers as rs, run


def test_neighbor_map_is_invertible_and_preserves_prediction():
    g = core.geometry(128)
    rng = np.random.default_rng(42)
    c = rng.normal(size=g.width + 1)
    gamma = rng.normal(size=g.width) / g.h
    x = np.linspace(-1, 1, 513)
    for coords in ("prescribed", "differences"):
        t = rs.coordinate_map(g, gamma, coords)
        z = rs.from_physical(c, g, gamma, coords)
        np.testing.assert_allclose(t @ z, c, atol=2e-14)
        np.testing.assert_allclose(rs.dictionary(x, g, gamma, coords) @ z,
                                   diagnostics.features(x, g.centers, gamma) @ c, atol=2e-13)
        np.testing.assert_allclose(rs.dictionary(x, g, gamma, coords),
                                   diagnostics.features(x, g.centers, gamma) @ t, atol=2e-14)
        assert np.linalg.matrix_rank(t) == g.width + 1


def test_first_order_armijo_descends_and_matches_gradient_descent():
    b = np.array([[1., 0.], [0., .01], [1., .01]])
    y = np.array([1., .1, 1.1])
    transform = np.eye(2)
    for algorithm in (0, 1, 2):
        state = rs.initial_state(np.zeros(2), algorithm)
        end, (trace, dense) = rs.linear_chunk(100, True, False)(state, b, y, transform, algorithm, 0)
        assert np.all(np.diff(np.asarray(trace[:, 0])) <= 1e-14)
        assert int(end["gradient_evaluations"]) == 100
        assert int(end["loss_evaluations"]) >= 200
        np.testing.assert_allclose(dense["delta_c"], np.diff(np.r_[dense["c"], end["z"][None]], axis=0), atol=1e-15)
        if algorithm == 0:
            np.testing.assert_allclose(dense["delta_c"][0], .1 * b.T @ y)


def test_warm_adam_moments_and_resume_match_live_frozen_readout(tmp_path):
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    c, gamma = core.initial_physical(g, 1, "envelope")
    state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    warm, _ = core.make_chunk(g, "adam", "sine", 1, 9)(state, cs, gs, .001, .001)
    c, gamma = core.physical(warm["params"], cs, gs)
    x = np.linspace(-1, 1, 129)
    b = rs.dictionary(x, g, np.asarray(gamma), "prescribed") / np.sqrt(len(x))
    y = core.target(x, "sine", np) / np.sqrt(len(x))
    moments = warm["opt"][0]
    linear = rs.initial_state(np.asarray(warm["params"]["readout"]), 4,
                              moments.mu["readout"], moments.nu["readout"], int(moments.count))
    t = rs.coordinate_map(g, np.asarray(gamma), "prescribed")
    expected, _ = core.make_chunk(g, "adam", "sine", 1, 8)(warm, cs, gs, 1e-6, 0.)
    actual, _ = rs.linear_chunk(8, False, False)(linear, b, y, t, 4, 0)
    np.testing.assert_allclose(actual["z"], expected["params"]["readout"], atol=1e-14)
    half, _ = rs.linear_chunk(4, False, False)(linear, b, y, t, 4, 0)
    run.save_state(tmp_path / "linear.pkl", half, 4)
    restored, at = run.load_state(tmp_path / "linear.pkl")
    resumed, _ = rs.linear_chunk(4, False, False)(restored, b, y, t, 4, at)
    np.testing.assert_array_equal(actual["z"], resumed["z"])
