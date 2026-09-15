"""Check paired trajectories independently and keep refits outside training."""
import numpy as np
import pytest
import torch

from experiments.expD24_gd_residual_spectrum import four_way_comparison as exp


@pytest.mark.parametrize("arm", ["xavier", "gamma_4"])
def test_paired_training_matches_direct_numpy_gradients(arm):
    torch.set_num_threads(1)
    cfg = exp.config() | dict(resolution=8, halo=2, n_train=64, n_eval=256, steps=5)
    case = exp.train_pair("gaussian_envelope", arm, cfg)
    initial = exp.initial_state(arm, cfg)
    x = exp.midpoint_grid(cfg["n_train"])
    y = exp.matched.target_values("gaussian_envelope", x, cfg)
    a, b, v = (initial[key].copy() for key in ("a", "b", "v"))
    frozen_v = v.copy()
    A0 = exp.design(x, a, b)
    for index in range(cfg["steps"]+1):
        for key, value in (("a", a), ("b", b), ("v", v), ("frozen_v", frozen_v)):
            np.testing.assert_allclose(case[key][index], value, atol=1e-14, rtol=1e-13)
        hidden = np.tanh(x[:, None]*a+b)
        residual = hidden @ v[:-1]+v[-1]-y
        np.testing.assert_allclose(case["joint_train_loss"][index], .5*np.mean(residual**2))
        if index < cfg["steps"]:
            sensitivity = residual[:, None]*v[:-1]*(1-hidden**2)
            ga = np.mean(x[:, None]*sensitivity, axis=0)
            gb = np.mean(sensitivity, axis=0)
            gv = np.r_[hidden.T @ residual/len(x), np.mean(residual)]
            frozen_v -= cfg["learning_rate"]*(A0.T @ (A0 @ frozen_v-y)/len(x))
            a -= cfg["learning_rate"]*ga
            b -= cfg["learning_rate"]*gb
            v -= cfg["learning_rate"]*gv

    original = {key: value.copy() for key, value in case.items()}
    evaluated = exp.evaluate_pair(case, "gaussian_envelope", cfg)
    for key in original:
        np.testing.assert_array_equal(case[key], original[key])
    np.testing.assert_array_equal(evaluated["errors"][:, 3], np.full(6, evaluated["errors"][0, 2]))
    np.testing.assert_allclose(evaluated["errors"][0, 0], evaluated["errors"][0, 1])
    # On the samples actually solved, the readout refit cannot have larger error.
    assert np.all(evaluated["train_errors"][:, 2] <= evaluated["train_errors"][:, 0]+1e-12)
    assert np.all(evaluated["train_errors"][:, 3] <= evaluated["train_errors"][:, 1]+1e-12)
    if arm != "xavier":
        np.testing.assert_array_equal(evaluated["errors"][0, 2], evaluated["errors"][1, 2])
