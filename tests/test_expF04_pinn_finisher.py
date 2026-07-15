import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF02_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF03_newton_burgers"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF04_pinn_finisher"))


def test_pinn_trains_and_loss_decreases():
    import pinn
    net, hist = pinn.train_pinn(nu=0.1, steps=60, batch=256, eval_every=30, seed=0)
    assert np.isfinite(hist[-1]["loss"])
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_pinn_fields_match_autograd_fd():
    import pinn
    net, _ = pinn.train_pinn(nu=0.1, steps=5, batch=64, eval_every=5, seed=0)
    fields = pinn.pinn_fields(net)
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.9, 0.9, (40, 2))
    f = fields(P)
    h = 1e-4
    Px_p, Px_m = P.copy(), P.copy()
    Px_p[:, 0] += h
    Px_m[:, 0] -= h
    fd_ux = (fields(Px_p)["u"] - fields(Px_m)["u"]) / (2 * h)
    assert np.max(np.abs(f["ux"] - fd_ux)) < 1e-3
    fd_lap_part = (fields(Px_p)["ux"] - fields(Px_m)["ux"]) / (2 * h)
    Py_p, Py_m = P.copy(), P.copy()
    Py_p[:, 1] += h
    Py_m[:, 1] -= h
    fd_lap = fd_lap_part + (fields(Py_p)["uy"] - fields(Py_m)["uy"]) / (2 * h)
    assert np.max(np.abs(f["lap_u"] - fd_lap)) < 1e-2
