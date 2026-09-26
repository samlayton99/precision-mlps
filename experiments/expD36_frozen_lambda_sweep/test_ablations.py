"""Tests of span-preserving maps and optimizer gradients, independent of plots."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run
import ablations


def problem():
    cfg = run.configuration()
    cfg.update(n=64, halo_per_side=2, n_train=131, n_eval=263, lambda_count=3)
    return run.make_problem(cfg)


def test_neighbor_map_retains_bias_anchor_and_physical_predictions():
    raw = problem()
    p = ablations.transform_problem(raw, "neighbor_unscaled")
    generator = torch.Generator().manual_seed(7)
    theta = torch.randn((len(p.lambdas), p.A.shape[2], 4), generator=generator)
    physical = p.readout_map @ theta
    torch.testing.assert_close(p.mapped_A @ theta, p.A @ physical, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(p.mapped_E @ theta, p.E @ physical, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(p.mapped_A[:, :, 0], p.A[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(p.mapped_A[:, :, -1], p.A[:, :, -1], rtol=0, atol=0)
    assert int(torch.linalg.matrix_rank(p.readout_map)) == p.A.shape[2]
    np.testing.assert_array_equal(raw.reference["eval_rel_l2"], p.reference["eval_rel_l2"])


@pytest.mark.parametrize("arm", ["neighbor_unscaled", "sqrt_allowance"])
def test_map_gradient_is_chain_rule_and_rate_uses_mapped_features(arm):
    p = ablations.transform_problem(problem(), arm)
    theta = torch.linspace(-.01, .01, p.A.shape[2], requires_grad=True)
    prediction = p.A[1] @ (p.readout_map @ theta)
    (.5 * (prediction-p.y_train[:, 0]).square().mean()).backward()
    expected = p.B[1].T @ (p.B[1] @ theta.detach()-p.y_train[:, 0]/np.sqrt(len(p.x_train)))
    torch.testing.assert_close(theta.grad, expected, rtol=1e-11, atol=1e-14)
    s = torch.linalg.svdvals(p.B[1])[0]
    torch.testing.assert_close(p.gd_rates[1, 0, 0], 1/s.square(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("arm", ["neighbor_unscaled", "sqrt_allowance"])
def test_physical_snapshots_and_native_resume(tmp_path, arm):
    p = ablations.transform_problem(problem(), arm)
    state = run.zero_state(p)
    history = run.fresh_history(p, state)
    run.advance(p, state, 7)
    path = tmp_path / "state.npz"
    run.checkpoint(path, p, state, history)
    with np.load(path) as d:
        np.testing.assert_allclose(d["physical_c"], (p.readout_map @ state["c"]).numpy())
    restored, _ = run.resume(path, p)
    run.advance(p, state, 5)
    run.advance(p, restored, 5)
    for name in ("c", "adam_m", "adam_v"):
        torch.testing.assert_close(state[name], restored[name], rtol=0, atol=0)


def test_allowances_match_reviewed_pr_values_and_scale_only_coordinates():
    # Independent values recorded by the PR source review at its N512/H23 geometry.
    alpha = ablations.reference_allowances(512, 23)
    np.testing.assert_allclose([alpha[24], alpha[1:].max(), alpha[0]],
                               [.008663, 1.04521, 10.7639], rtol=6e-5, atol=1e-7)
    np.testing.assert_array_equal(alpha[1:], alpha[:0:-1])
    assert np.all(alpha > 0)
    raw = problem()
    p = ablations.transform_problem(raw, "sqrt_allowance")
    np.testing.assert_array_equal(raw.reference["eval_rel_l2"], p.reference["eval_rel_l2"])
    torch.testing.assert_close(p.mapped_A, p.A @ p.readout_map, rtol=0, atol=0)
    torch.testing.assert_close(p.mapped_E, p.E @ p.readout_map, rtol=0, atol=0)
    assert int(torch.linalg.matrix_rank(p.readout_map)) == p.A.shape[2]
