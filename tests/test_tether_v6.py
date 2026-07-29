"""Tether v6 (continuous bounded auto-tether) unit tests:
1. self-removal on an exactly linear problem (q == 0 -> S == 1, floor hit);
2. S stays below the hard bound 1 + T_MAX on a nonlinear run, and T never
   exceeds T_MAX / never drops below 1.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "experiments" / "expD08_qi_init_nlcg"))
import tether_v6  # noqa: E402
import run as d08run  # noqa: E402


def _linear_problem(n=400, m=60, seed=0):
    rng = np.random.default_rng(seed)
    A = torch.tensor(rng.standard_normal((n, m)) / np.sqrt(m))
    x_true = torch.tensor(rng.standard_normal(m))
    y = A @ x_true
    theta0 = torch.zeros(m, dtype=torch.float64)

    def f_pred(th):
        return A @ th

    def f_resid(th):
        with torch.no_grad():
            return (A @ th - y).detach()

    def f_jvp(th, d):
        return (A @ d).detach()

    def f_vjp(th, s):
        return (A.t() @ s).detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(A @ th - y)
                         / torch.linalg.norm(y))

    return theta0, f_pred, f_resid, f_jvp, f_vjp, eval_rel, y


def test_linear_self_removal():
    torch.set_default_dtype(torch.float64)
    theta0, f_pred, f_resid, f_jvp, f_vjp, eval_rel, y = _linear_problem()
    losses, rels, snaps, theta, stats = tether_v6.train_v6(
        theta0, f_resid, f_jvp, f_vjp, f_pred, y.numel(), 400, {0},
        lambda th: {"w": th.numpy()}, eval_rel,
        float(torch.linalg.norm(y)), variant="A", snap_S=True)
    # linear problem: probe q == 0, so S == 1 in every snapshot
    for s in snaps.values():
        assert np.allclose(s["S"], 1.0)
    assert min(rels) < 1e-12


def test_bounds_on_nonlinear_toy():
    torch.set_default_dtype(torch.float64)
    C = tether_v6.toy_closures("sine", 32)
    losses, rels, snaps, theta, stats = tether_v6.train_v6(
        C["theta0"], C["f_resid"], C["f_jvp"], C["f_vjp"], C["f_pred"],
        d08run.N_TRAIN, 700, {0}, C["get_wbv"], C["eval_rel"],
        C["y_norm"], variant="C", snap_S=True)
    Ts = [t for _, t in stats["T_hist"]]
    assert Ts, "controller never ran"
    assert max(Ts) <= tether_v6.T_MAX + 1e-6
    assert min(Ts) >= 1.0
    for s in snaps.values():
        assert s["S"].max() <= 1.0 + tether_v6.T_MAX * (1 + 1e-12)
    # sanity: it actually optimizes
    assert min(rels) < 1e-4
