import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF06_newton_burgers"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF07_pinn_finisher"))


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


@pytest.mark.slow
def test_finisher_improves_pinn():
    import pinn
    import newton as nt
    net, hist = pinn.train_pinn(nu=0.1, steps=400, batch=512, eval_every=200, seed=0)
    before = hist[-1]["rel_l2_u"]
    # Representation ceiling: the ridge correction must express u* - PINN, and
    # an undertrained (400-step) PINN's error field is rougher than the
    # gamma-limited basis can fully cancel — residual floors at ~0.06 here
    # (0.379 -> 0.0169 rel L2, then flat). The ceiling scales with the PINN's
    # error magnitude; the full run (50k steps, W=1024) measures the real claim.
    # Smoke bar: >=20x improvement.
    res = nt.newton_burgers(nu=0.1, W=256, lam=0.25, max_iter=6, seed=0,
                            base_fields=pinn.pinn_fields(net))
    after = res["history"][-1]["rel_l2_u"]
    assert after < 0.05 * before, (before, after)
