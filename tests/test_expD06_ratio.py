import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from experiments.expD06_fixed_center_scales import core, run, ratio


def tree_close(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-14)


def test_schedule_endpoints_and_physical_factors():
    for arm, pair in ratio.ARMS.items():
        knots = ratio.primary_schedule(.01, arm)
        for step, expected in [(0, [.01]*2), (80_000, [.01]*2),
                               (160_000, [1e-6]*2), (180_000, pair), (500_000, pair)]:
            np.testing.assert_allclose(ratio.schedule(step, knots), expected, atol=1e-18)
    np.testing.assert_allclose(ratio.schedule(40_000, ratio.early_schedule()), [.01, .001])
    np.testing.assert_allclose(ratio.schedule(160_000, ratio.early_schedule()), [1e-6, 1e-7])
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    np.testing.assert_array_equal(cs, g.d)
    assert gs == 1 / g.h


def test_constant_schedule_matches_original_and_resume(tmp_path):
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    c, gamma = core.initial_physical(g, 0, "envelope")
    state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    knots = np.array([[0, .001, .001], [20_000, .001, .001]])
    expected, _ = core.make_chunk(g, "adam", "sine", 1, 8)(state, cs, gs, .001, .001)
    actual, _ = ratio.chunk(128, "sine", 1, 8, False, False)(state, 0, knots)
    tree_close(expected, actual)
    knots = ratio.primary_schedule(.01, "both_changes")
    whole, _ = ratio.chunk(128, "sine", 1, 8, True, False)(state, 159_996, knots)
    half, _ = ratio.chunk(128, "sine", 1, 4, False, False)(state, 159_996, knots)
    run.save_state(tmp_path / "state.pkl", half, 160_000)
    restored, step = run.load_state(tmp_path / "state.pkl")
    resumed, _ = ratio.chunk(128, "sine", 1, 4, True, False)(restored, step, knots)
    tree_close(whole, resumed)


def test_case_matrix(tmp_path):
    acquire, tails = ratio.main_groups(tmp_path, 0)
    assert len(acquire) == len(tails) == 6
    assert sum(map(len, tails)) == 30
    assert {b.case.n for group in tails for b in group} == {512, 1024}
    assert {b.case.target for group in tails for b in group} == {"sine", "quadratic", "mixed"}
    assert all(b.case.arm == "both" and b.case.initialization == "envelope" for group in tails for b in group)
    assert all(b.source_step == 160_000 for group in tails for b in group)
    indices = ratio.dense_sample_indices()
    np.testing.assert_array_equal(indices // 32, np.arange(64))
    np.testing.assert_array_equal(indices, ratio.dense_sample_indices())
    assert indices[0] == 0 and len(np.unique(indices % 32)) > 16


def test_dense_evidence_is_actual_physical_motion():
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    c, gamma = core.initial_physical(g, 0, "envelope")
    state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    knots = ratio.primary_schedule(.01, "slow_geometry")
    end, (_, dense) = ratio.chunk(128, "mixed", 1, 6, True, False)(state, 170_000, knots)
    final_c, final_g = core.physical(end["params"], cs, gs)
    np.testing.assert_allclose(np.diff(np.r_[dense["c"], final_c[None]], axis=0), dense["delta_c"], atol=2e-15)
    np.testing.assert_allclose(np.diff(np.r_[dense["gamma"] * g.h, final_g[None] * g.h], axis=0),
                               dense["delta_lambda"], atol=2e-15)


def test_worker_persists_rates_moments_and_resumes(tmp_path):
    case = run.Case(n=128, samples_per_cell=1, validation_points=256, initialization="envelope")
    branch = ratio.Branch(case, "check", tuple(map(tuple, ratio.primary_schedule(.001))))
    ratio.advance_group(tmp_path, [branch], 4)
    ratio.advance_group(tmp_path, [branch], 8)
    folder = branch.folder(tmp_path)
    saved, step = run.load_state(folder / "state_000000008.pkl")
    initial, _, _ = ratio.prepare(tmp_path / "fresh", branch)
    expected, _ = ratio.chunk(128, "sine", 1, 8, False, False)(initial, 0, np.asarray(branch.knots))
    tree_close(saved, expected)
    assert step == int(saved["opt"][0].count) == 8
    assert run.trace_window_losses(folder, 0, 8).shape == (8,)
    with np.load(folder / "checkpoint_000000008.npz") as arrays:
        np.testing.assert_allclose(arrays["physical_readout_rate_factor"], .001 * core.geometry(128).d)
        assert arrays["physical_gamma_rate_factor"] == .001 / core.geometry(128).h
