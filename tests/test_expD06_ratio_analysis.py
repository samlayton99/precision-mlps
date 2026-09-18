import numpy as np
from experiments.expD06_fixed_center_scales import core, diagnostics
from experiments.expD06_fixed_center_scales.ratio_analysis import update_budget, read_dense
from experiments.expD06_fixed_center_scales.run import save_arrays


def test_exact_update_budget_closes_in_function_and_loss_space():
    rng = np.random.default_rng(31)
    centers = np.linspace(-1.1, 1.1, 15)
    x = np.linspace(-1, 1, 129)
    h = .1
    c, gamma = rng.normal(size=16), rng.normal(size=15) * 5
    dc, dl = rng.normal(size=16) * .01, rng.normal(size=15) * .001
    y = core.target(x, "mixed", np)
    residual, pieces, budget = update_budget(x, y, centers, h, c, gamma, dc, dl)
    before = diagnostics.prediction(x, centers, c, gamma)
    after = diagnostics.prediction(x, centers, c + dc, gamma + dl / h)
    np.testing.assert_allclose(pieces.sum(axis=0), after - before, atol=3e-15)
    np.testing.assert_allclose(budget["mse_change"][-1], np.mean((after-y)**2 - (before-y)**2), atol=3e-15)
    np.testing.assert_allclose(budget["mse_change"], budget["linear_mse_change"]+budget["quadratic_mse_cost"])
    assert np.all(budget["quadratic_mse_cost"] >= 0)


def test_dense_reader_handles_interrupted_overlapping_saves(tmp_path):
    for lo, hi in [(0, 1000), (0, 1500), (1500, 2048)]:
        steps = np.arange(lo, hi)
        save_arrays(tmp_path / f"dense_{lo:09d}_{hi:09d}.npz", step=steps, c=steps[:, None])
    dense = read_dense(tmp_path, 2048)
    np.testing.assert_array_equal(dense["step"], np.arange(2048))
    np.testing.assert_array_equal(dense["c"][:, 0], np.arange(2048))
