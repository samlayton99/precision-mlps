import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from experiments.expD06_fixed_center_scales import core, run
from experiments.expD06_fixed_center_scales import continue_stall as stall


def tree_close(a, b):
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-14)


def test_shared_schedule_and_dense_windows():
    np.testing.assert_allclose(stall.learning_rate(np.array([0, 40000, 80000, 100000]), True),
                               [1e-3, .0005005, 1e-6, 1e-6], atol=1e-18)
    np.testing.assert_array_equal(stall.learning_rate(np.array([0, 40000, 100000]), False), [1e-3]*3)
    assert all(stall.dense_step(t) for t in [0, 2047, 17952, 19999, 39999])
    assert not any(stall.dense_step(t) for t in [2048, 17951, 20000])


def test_continuation_matches_original_and_restores_moments(tmp_path):
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    c, gamma = core.initial_physical(g, 0, "envelope")
    initial = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    warm, _ = core.make_chunk(g, "adam", "sine", 1, 7)(initial, cs, gs, .001, .001)
    expected, _ = core.make_chunk(g, "adam", "sine", 1, 8)(warm, cs, gs, .001, .001)
    actual, (_, dense) = stall.scheduled_chunk(g, 1, 8, capture=True, batched=False)(warm, 0, False, False)
    tree_close(expected, actual)
    np.testing.assert_allclose(dense["delta_lambda"], np.diff(np.r_[np.asarray(dense["gamma"])*g.h,
                                                   np.asarray(actual["params"]["slope"])[None]], axis=0), atol=2e-16)
    path = tmp_path / "state.pkl"
    half, _ = stall.scheduled_chunk(g, 1, 4, batched=False)(warm, 39998, False, True)
    run.save_state(path, half, 40002)
    restored, step = run.load_state(path)
    resumed, _ = stall.scheduled_chunk(g, 1, 4, batched=False)(restored, step, False, True)
    uninterrupted, _ = stall.scheduled_chunk(g, 1, 8, batched=False)(warm, 39998, False, True)
    tree_close(uninterrupted, resumed)
    assert int(resumed["opt"][0].count) == 15


def test_freezing_geometry_preserves_slopes_but_trains_readout():
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    c, gamma = core.initial_physical(g, 1, "envelope")
    initial = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    warm, _ = core.make_chunk(g, "adam", "sine", 1, 7)(initial, cs, gs, .001, .001)
    frozen, (_, dense) = stall.scheduled_chunk(g, 1, 8, capture=True, batched=False)(warm, 79998, True, True)
    np.testing.assert_array_equal(warm["params"]["slope"], frozen["params"]["slope"])
    np.testing.assert_array_equal(dense["delta_lambda"], 0.)
    assert np.linalg.norm(np.asarray(frozen["params"]["readout"]-warm["params"]["readout"])) > 0
    assert int(frozen["opt"][0].count) == int(warm["opt"][0].count) + 8


def test_four_branches_do_not_share_optimizer_updates():
    g = core.geometry(128)
    cs, gs = core.coordinate_scales(g, "both")
    c, gamma = core.initial_physical(g, 0, "envelope")
    initial = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
    flags = [(False, False), (False, True), (True, False), (True, True)]
    batch, _ = stall.scheduled_chunk(g, 1, 8)(run.stack_states([initial]*4), 79998,
                                              np.array([f for f, _ in flags]), np.array([d for _, d in flags]))
    for i, (frozen, decay) in enumerate(flags):
        expected, _ = stall.scheduled_chunk(g, 1, 8, batched=False)(initial, 79998, frozen, decay)
        tree_close(expected, run.unstack_state(batch, i))


def test_settling_checks_only_compare_windows_after_decay():
    rows = []
    for step, begin in [(20000, 0), (40000, 20000), (80000, 40000),
                        (100000, 80000), (120000, 100000), (160000, 120000), (240000, 160000)]:
        rows.append({"step": step, "finite": True, "validation": {"rms": 1.}, "c_l1": 1.,
                     "band_energy": [1.], "lambda_quantiles": [.1]*5,
                     "window_loss": {"start_step": begin, "mean": .5, "std": .1}})
    assert run.convergence_status(rows[:-1], step_offset=80000) == "continuing"
    assert run.convergence_status(rows, step_offset=80000) == "oscillatory"
