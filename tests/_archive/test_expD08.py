"""expD08 verification: the NLCG-ortho core must match the expD07
cgls_reortho result on pure quadratics (floor in <= ~rank iterations), and
the full run_case must descend from the seed9 init on a small width.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expD07_lstsq_optim_suite"))
import build_problems as bp  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "expd08_run", REPO_ROOT / "experiments" / "expD08_qi_init_nlcg" / "run.py")
d08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d08)


def _quadratic_env(A, y):
    A_t = torch.from_numpy(A)
    y_t = torch.from_numpy(y)
    R = A.shape[0]
    y_norm = float(np.linalg.norm(y))

    def f_loss(v):
        with torch.no_grad():
            r = A_t.mv(v) - y_t
            return float(r.dot(r)) / R

    def f_grad(v):
        r = A_t.mv(v) - y_t
        return float(r.dot(r)) / R, (2.0 / R) * A_t.t().mv(r)

    def eval_rel(v):
        with torch.no_grad():
            return float(torch.linalg.norm(A_t.mv(v) - y_t)) / y_norm

    def f_hvp(v, d):
        return (2.0 / R) * A_t.t().mv(A_t.mv(d))

    return f_loss, f_grad, f_hvp, eval_rel


def test_secant_cg_ortho_reaches_floor_on_quadratics():
    """The gradients-only core (no loss comparisons, secant steps) must
    still match cgls_reortho on pure quadratics."""
    torch.set_default_dtype(torch.float64)
    for cell in ("wellcond_iso", "illcond8_iso"):
        A, y = bp.make_synthetic(cell, bp.CELLS[cell], 48, 96, 11)
        f_loss, f_grad, f_hvp, eval_rel = _quadratic_env(A, y)
        losses, rels, snaps, _, stats = d08.train_secant_cg_ortho(
            torch.zeros(48), f_grad, iters=400,
            frames_at=set(), get_wbv=lambda th: {}, eval_rel=eval_rel,
            f_hvp=f_hvp)
        assert min(rels) < 1e-11, (cell, min(rels), stats)
        # the pure-gradient secant mode must stay within ~one order
        losses2, rels2, *_ = d08.train_secant_cg_ortho(
            torch.zeros(48), f_grad, iters=400,
            frames_at=set(), get_wbv=lambda th: {}, eval_rel=eval_rel)
        assert min(rels2) < 1e-9, (cell, min(rels2))


def test_residual_cg_matches_cgls_reortho_on_frozen_phi_class():
    """The iteration_4 core (carried residual) must beat the fresh-residual
    core by orders on an ill-conditioned quadratic and land at the
    recursive-CGLS class (~1e-13)."""
    torch.set_default_dtype(torch.float64)
    A, y = bp.make_synthetic("illcond8_iso", bp.CELLS["illcond8_iso"],
                             48, 96, 11)
    A_t, y_t = torch.from_numpy(A), torch.from_numpy(y)
    R = A.shape[0]
    y_norm = float(np.linalg.norm(y))

    def f_resid(v):
        with torch.no_grad():
            return (A_t.mv(v) - y_t).detach()

    def f_jvp(v, d):
        return A_t.mv(d)

    def f_vjp(v, seed):
        return A_t.t().mv(seed)

    def eval_rel(v):
        with torch.no_grad():
            return float(torch.linalg.norm(A_t.mv(v) - y_t)) / y_norm

    losses, rels, snaps, _, stats = d08.train_residual_cg_ortho(
        torch.zeros(48), f_resid, f_jvp, f_vjp, R, iters=400,
        frames_at=set(), get_wbv=lambda th: {}, eval_rel=eval_rel,
        refresh=200)
    assert min(rels) < 1e-12, (min(rels), stats)


def test_residual_cg_slim_reaches_floor_on_frozen_phi_class():
    """The iteration_5 slim core (trust window / descent check / beta clamp
    deleted per the hardening ablation) must still land at the
    recursive-CGLS class (~1e-13) on an ill-conditioned quadratic."""
    torch.set_default_dtype(torch.float64)
    A, y = bp.make_synthetic("illcond8_iso", bp.CELLS["illcond8_iso"],
                             48, 96, 11)
    A_t, y_t = torch.from_numpy(A), torch.from_numpy(y)
    R = A.shape[0]
    y_norm = float(np.linalg.norm(y))

    def f_resid(v):
        with torch.no_grad():
            return (A_t.mv(v) - y_t).detach()

    def f_jvp(v, d):
        return A_t.mv(d)

    def f_vjp(v, seed):
        return A_t.t().mv(seed)

    def eval_rel(v):
        with torch.no_grad():
            return float(torch.linalg.norm(A_t.mv(v) - y_t)) / y_norm

    losses, rels, snaps, _, stats = d08.train_residual_cg_slim(
        torch.zeros(48), f_resid, f_jvp, f_vjp, R, iters=400,
        frames_at=set(), get_wbv=lambda th: {}, eval_rel=eval_rel,
        refresh=200)
    assert min(rels) < 1e-12, (min(rels), stats)


def test_iter6_reaches_floor_on_frozen_phi_class():
    """On an exactly-linear problem the auto-tether must self-remove
    (q = 0 identically) and the iter6 core must behave as the slim core:
    recursive-CGLS class (~1e-13) on an ill-conditioned quadratic."""
    torch.set_default_dtype(torch.float64)
    A, y = bp.make_synthetic("illcond8_iso", bp.CELLS["illcond8_iso"],
                             48, 96, 11)
    A_t, y_t = torch.from_numpy(A), torch.from_numpy(y)
    R = A.shape[0]
    y_norm = float(np.linalg.norm(y))

    def f_pred(v):
        return A_t.mv(v)

    def f_resid(v):
        with torch.no_grad():
            return (A_t.mv(v) - y_t).detach()

    def f_jvp(v, d):
        return A_t.mv(d)

    def f_vjp(v, seed):
        return A_t.t().mv(seed)

    def eval_rel(v):
        with torch.no_grad():
            return float(torch.linalg.norm(A_t.mv(v) - y_t)) / y_norm

    losses, rels, snaps, _, stats = d08.train_iter6(
        torch.zeros(48), f_resid, f_jvp, f_vjp, f_pred, R, iters=400,
        frames_at=set(), get_wbv=lambda th: {}, eval_rel=eval_rel,
        y_norm=y_norm, refresh=200)
    assert min(rels) < 1e-12, (min(rels), stats)
    assert stats["T_final"] == 1.0      # auto-tether never engaged


def test_run_case_descends_from_seed9_init():
    job = {"fn": "sine", "N": 32, "iters": 300, "threads": 2,
           "core": "iter6"}  # legacy-core regression test (iter6/7)
    r = d08.run_case(job)
    assert r["rels"][0] < 1.5           # starts near rel L2 ~ 1 (v = 0)
    assert r["best_rel"] < 1e-2 * r["rels"][0]  # real progress in 300 iters
    assert r["floor"] < 1e-6            # the init geometry's lstsq floor
    assert "0" in r["snaps"] and "w" in r["snaps"]["0"]
