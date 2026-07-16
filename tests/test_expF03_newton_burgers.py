import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF02_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF03_newton_burgers"))


def test_burgers_manufactured_fd_verified():
    import burgers as bp
    bp.verify_all(nu=0.1)
    bp.verify_all(nu=0.01)


@pytest.mark.slow
def test_newton_converges_nu01():
    import burgers as bp
    import newton as nt
    res = nt.newton_burgers(nu=0.1, W=256, lam=0.25, max_iter=8, seed=0,
                            u_exact=bp.u_exact, v_exact=bp.v_exact)
    hist = res["history"]
    # damping guarantees monotone residual over accepted steps
    resids = [h["res_norm"] for h in hist]
    assert all(b <= a * 1.0001 for a, b in zip(resids, resids[1:]))
    assert hist[-1]["rel_l2_u"] < 1e-6, hist[-1]


@pytest.mark.slow
def test_init_sol_warm_start():
    import newton as nt
    base = nt.newton_burgers(nu=0.1, W=256, lam=0.25, max_iter=8, seed=0)
    warm = nt.newton_burgers(nu=0.1, W=256, lam=0.25, max_iter=1, seed=0,
                             init_sol=(base["sol_u"], base["sol_v"]))
    # a warm start at the converged coefficients is already at the floor on iter 0
    assert warm["history"][0]["rel_l2_u"] < 1e-5, warm["history"][0]
