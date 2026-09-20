"""Checks of freeze timing, raw gradients, and evaluation-only readout solves."""
import numpy as np
import torch
from experiments.expD26_freeze_and_readout_spectrum import freeze


def test_raw_gradient_and_output_bias_freeze():
    torch.set_num_threads(2)
    rng = np.random.default_rng(14)
    state = dict(a=rng.normal(size=5), b=rng.normal(size=5), v=rng.normal(size=6))
    x = np.linspace(-1, 1, 19)
    y = np.sin(3*x)
    gradients = freeze.numpy_gradients(state, x, y)
    for frozen in (False, True):
        run = freeze.train_trajectory(state, x, y, 3, .002, freeze_readout=frozen)
        for key in freeze.FIELDS:
            expected = state[key] if frozen and key == "v" else state[key]-.002*gradients[key]
            np.testing.assert_allclose(run[key][1], expected, rtol=3e-14, atol=3e-15)
        if frozen:
            np.testing.assert_array_equal(run["v"], np.broadcast_to(state["v"], run["v"].shape))
            assert np.any(run["a"][1] != run["a"][0])


def test_branch_prefix_freeze_timing_and_refit_nonmutation():
    cfg = freeze.config()
    cfg.update(resolution=8, halo=2, n_train=48, n_eval=96, freeze_steps=[2, 4], frozen_steps=5)
    cases, discrepancy = freeze.train_branches("sine", cfg)
    assert discrepancy < 1e-14
    before = {name: {key: case[key].copy() for key in freeze.FIELDS} for name, case in cases.items()}
    freeze.evaluate_target(cases, "sine", cfg)
    for name, state in before.items():
        for key in freeze.FIELDS:
            np.testing.assert_array_equal(state[key], cases[name][key])
    for step in cfg["freeze_steps"]:
        branch = cases[f"freeze_{step}"]
        for key in freeze.FIELDS:
            np.testing.assert_array_equal(branch[key][:step+1], cases["joint"][key][:step+1])
        np.testing.assert_array_equal(branch["v"][step:], np.broadcast_to(branch["v"][step], branch["v"][step:].shape))
        assert branch["steps"][-1] == step+5
        assert 0 in branch["refit_steps"] and step in branch["refit_steps"]
        np.testing.assert_array_equal(branch["relative_l2"][:step+1], cases["joint"]["relative_l2"][:step+1])
        assert np.all(np.isfinite(branch["refit_relative_l2"]))
