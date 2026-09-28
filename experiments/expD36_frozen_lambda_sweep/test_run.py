"""Scientific checks: exact optimizer, readout metric, and continuation semantics."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("d36_run", Path(__file__).with_name("run.py"))
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


def small_problem():
    cfg = run.configuration()
    cfg.update(n=64, halo_per_side=2, n_train=131, n_eval=263, lambda_count=3)
    return run.make_problem(cfg)


def test_batched_updates_match_pytorch_gd_and_adam():
    p = small_problem()
    state = run.zero_state(p)
    f = len(p.cfg["targets"])
    for method in range(2):
        idx = 1
        c = torch.zeros_like(state["c"][idx, :, :f], requires_grad=True)
        optimizer = (torch.optim.SGD([c], lr=float(p.gd_rates[idx])) if method == 0
                     else torch.optim.Adam([c], lr=p.cfg["adam_lr"], eps=p.cfg["adam_eps"],
                                           betas=(p.cfg["adam_beta1"], p.cfg["adam_beta2"])))
        for _ in range(13):
            optimizer.zero_grad()
            loss = .5 * (p.B[idx] @ c - p.y_train/np.sqrt(len(p.x_train))).square().sum()
            loss.backward()
            optimizer.step()
        local = run.zero_state(p)
        run.advance(p, local, 13)
        # BMM versus the single-case autograd GEMM changes reduction rounding.
        # Near symmetry-zero gradients, Adam divides that noise by epsilon.
        np.testing.assert_allclose(local["c"][idx, :, method*f:(method+1)*f].numpy(),
                                   c.detach().numpy(), rtol=3e-10, atol=1e-13)


def test_zero_error_and_loss_normalization():
    p = small_problem()
    state = run.zero_state(p)
    for error in run.evaluate(p, state):
        np.testing.assert_array_equal(error, np.ones_like(error))
    c = torch.linspace(-.01, .01, p.B.shape[2], requires_grad=True)
    residual = p.A[0] @ c-p.y_train[:, 0]
    (.5*residual.square().mean()).backward()
    expected = p.B[0].T @ (p.B[0] @ c.detach()-p.y_train[:, 0]/np.sqrt(len(p.x_train)))
    torch.testing.assert_close(c.grad, expected, rtol=1e-12, atol=1e-14)


def test_checkpoint_resume_is_identical(tmp_path):
    p = small_problem()
    whole = run.zero_state(p)
    run.advance(p, whole, 17)
    split = run.zero_state(p)
    history = run.fresh_history(p, split)
    run.advance(p, split, 7)
    train, evaluation = run.evaluate(p, split)
    history["steps"].append(7)
    history["train_rel_l2"].append(train)
    history["eval_rel_l2"].append(evaluation)
    path = tmp_path / "state.npz"
    run.checkpoint(path, p, split, history)
    restored, _ = run.resume(path, p)
    run.advance(p, restored, 10)
    for key in ("c", "adam_m", "adam_v"):
        torch.testing.assert_close(whole[key], restored[key], rtol=0, atol=0)
    assert restored["step"] == whole["step"] == 17


def test_reference_is_same_feature_matrix():
    p = small_problem()
    for i, c in enumerate(p.reference["coefficients"]):
        fitted = p.E[i] @ torch.from_numpy(c)
        actual = (torch.linalg.vector_norm(fitted-p.y_eval, dim=0)
                  / torch.linalg.vector_norm(p.y_eval, dim=0)).numpy()
        np.testing.assert_array_equal(actual, p.reference["eval_rel_l2"][i])
    assert .25 in p.lambdas
    assert .5 in p.lambdas
    assert p.lambdas[-1] == max([p.cfg["lambda_max"]]+p.cfg.get("additional_lambdas", []))
