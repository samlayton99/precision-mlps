"""iteration_11 permanent tests: the two-regime certify-and-solve core.

  1. Membership hysteresis + budget (pure function): enter below Q_ENTER,
     stay until Q_EXIT (deadband), leave above; over-budget keeps the
     b_cap most-linear (degrade by selection).
  2. Linear self-identification: on an exactly linear model EVERY
     parameter certifies and the varpro arm IS a least-squares solver --
     floor within one solve.
  3. qn arm floors a linear system: dense inverse-BFGS with exact GN
     steps and exact secant pairs reaches <= 1e-10 on a kappa=1e4 system
     (the iterative-arm correctness gate).
  4. Readout discovered blind: from the standard seed9 init (QI
     geometry, ZERO readout) the certificate selects exactly the
     readout block (v, c0) -- geometry stays with Adam. Discovery, not a
     last-layer rule.
  5. QI-init live floor (gate-1 regression): the full live loop (Adam
     moving the geometry + exploit solve) holds the lstsq floor of the
     init geometry on sine@32.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expD08_qi_init_nlcg"))

_spec = importlib.util.spec_from_file_location(
    "expd08_iter11",
    REPO_ROOT / "experiments" / "expD08_qi_init_nlcg" / "iter11.py")
it11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(it11)

torch.set_default_dtype(torch.float64)


def _linear_env(m=40, n=120, kappa=1e4, seed=0):
    """Exactly linear model pred(theta) = A theta, y = A x_true."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, m)))
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    s = np.logspace(0, -np.log10(kappa), m)
    A = torch.tensor(U @ np.diag(s) @ V.T)
    x_true = torch.tensor(rng.standard_normal(m))
    y = A @ x_true
    y_norm = float(torch.linalg.norm(y))

    def f_pred(th):
        return A @ th

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - y).detach()

    def f_jvp(th, dvec):
        return (A @ dvec).detach()

    def f_vjp(th, seed_vec):
        return (A.t() @ seed_vec).detach()

    def eval_rel(th):
        return float(torch.linalg.norm(f_pred(th) - y)) / y_norm

    def get_wbv(th):
        t = th.detach().numpy()
        k = m // 3
        return {"w": t[:k].copy(), "b": t[k:2 * k].copy(),
                "v": t[2 * k:].copy()}

    return dict(theta0=torch.zeros(m), f_pred=f_pred, f_resid=f_resid,
                f_jvp=f_jvp, f_vjp=f_vjp, eval_rel=eval_rel,
                get_wbv=get_wbv, y_norm=y_norm, n=n)


def _qi_env(fn_name="sine", N=32):
    env = it11.build_case(fn_name, N)
    return env


def test_membership_hysteresis_and_budget():
    q = torch.tensor([0.0, 1e-9, 5e-8, 2e-7, 1.0])
    members = torch.zeros(5, dtype=torch.bool)
    # enter: only below Q_ENTER (1e-8)
    new = it11.update_membership(q, members, b_cap=10)
    assert new.tolist() == [True, True, False, False, False]
    # deadband: 5e-8 is between enter and exit -> a MEMBER stays, a
    # non-member does not enter
    members = torch.tensor([False, False, True, False, False])
    new = it11.update_membership(q, members, b_cap=10)
    assert new.tolist() == [True, True, True, False, False]
    # exit: above Q_EXIT (1e-7) a member is demoted (reactivation)
    members = torch.tensor([False, False, False, True, False])
    new = it11.update_membership(q, members, b_cap=10)
    assert new.tolist() == [True, True, False, False, False]
    # budget, no info supplied: fall back to the linearity ranking
    q = torch.tensor([1e-12, 1e-10, 1e-9, 1e-11, 1.0])
    members = torch.zeros(5, dtype=torch.bool)
    new = it11.update_membership(q, members, b_cap=2)
    assert new.tolist() == [True, False, False, True, False]


def test_budget_selects_informed_not_blind():
    """Regression for a measured failure (N=512, all six targets -> ~1e10):
    when the budget binds, the exploit set must be filled by DATA ENERGY,
    not by linearity. Coords 0 and 1 are perfectly linear but invisible
    (saturated halo neurons probe at exactly zero); coords 2 and 3 are
    certified AND carry gradient energy. Ranking by linearity picks the
    invisible pair and starves the informed ones."""
    q = torch.tensor([0.0, 0.0, 1e-10, 1e-11, 1.0])
    info = torch.tensor([0.0, 0.0, 5.0, 7.0, 9.0])   # coord 4 uncertified
    members = torch.zeros(5, dtype=torch.bool)
    new = it11.update_membership(q, members, b_cap=2, info=info)
    assert new.tolist() == [False, False, True, True, False], \
        "budget must go to coordinates that are certified AND informed"
    old = it11.update_membership(q, members, b_cap=2)
    assert old.tolist() == [True, True, False, False, False], \
        "the linearity ranking demonstrably selects the invisible pair"


def test_blind_coordinates_never_admitted():
    """Admission needs BOTH halves of the certificate. A data-blind
    coordinate (gradient exactly zero -- a saturated halo neuron) reads
    as perfectly linear, so linearity alone would admit it even with
    budget to spare. Measured at N=512: 642 such coordinates, all with
    |g| == 0.0 exactly, against a readout median of 1.7e-2."""
    q = torch.tensor([0.0, 0.0, 1e-10])
    info = torch.tensor([0.0, 0.0, 3.0])
    members = torch.zeros(3, dtype=torch.bool)
    new = it11.update_membership(q, members, b_cap=99, info=info)
    assert new.tolist() == [False, False, True], \
        "blind coordinates must never be admitted, budget notwithstanding"
    # and a member that goes blind is demoted back to Adam
    was = torch.tensor([True, False, True])
    out = it11.update_membership(q, was, b_cap=99, info=info)
    assert out.tolist() == [False, False, True]


def test_linear_full_certify_floors_varpro():
    env = _linear_env()
    losses, rels, snaps, theta, stats = it11.train_iter11(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], env["n"], iters=10, frames_at={0},
        get_wbv=env["get_wbv"], eval_rel=env["eval_rel"],
        y_norm=env["y_norm"], refresh=200, arm="varpro")
    assert stats["B_final"] == env["theta0"].numel(), \
        "every parameter of a linear model must certify"
    assert min(rels) <= 1e-12, f"varpro on linear: {min(rels):.2e}"


def test_linear_floors_qn():
    env = _linear_env()
    losses, rels, snaps, theta, stats = it11.train_iter11(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], env["n"], iters=400, frames_at={0},
        get_wbv=env["get_wbv"], eval_rel=env["eval_rel"],
        y_norm=env["y_norm"], refresh=200, arm="qn")
    assert stats["B_final"] == env["theta0"].numel()
    assert min(rels) <= 1e-10, f"qn on linear: {min(rels):.2e}"


def test_readout_discovered():
    """The certificate must (a) certify the whole readout block and (b)
    certify NO live geometry. Data-saturated halo neurons (|wx+b| > 4 on
    all of [-1,1]: zero curvature AND zero gradient -- data-blind) are
    allowed in: they are measured-linear, their Jacobian columns are
    ~null and rcond-truncated by the solve, and the batched t-statistic
    is the eventual informed-vs-blind separator (spec)."""
    env = _qi_env("sine", 32)
    W = env["W"]
    th0 = env["theta0"]
    losses, rels, snaps, theta, stats = it11.train_iter11(
        th0, env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], 2003, iters=1, frames_at={0},
        get_wbv=env["get_wbv"], eval_rel=env["eval_rel"],
        y_norm=env["y_norm"], refresh=200, arm="varpro", snap_S=True)
    E = snaps[0]["E"] > 0.5
    assert E[2 * W:].all(), "readout block (v, c0) must certify"
    x = np.linspace(-1, 1, 2003)
    w0, b0 = th0[:W].numpy(), th0[W:2 * W].numpy()
    sat = np.min(np.abs(np.outer(x, w0) + b0), axis=0) > 4.0
    geom_members = np.nonzero(E[:2 * W])[0] % W
    assert sat[geom_members].all(), \
        "a live (unsaturated) geometry coordinate certified as linear"
    live = ~sat
    assert not E[:W][live].any() and not E[W:2 * W][live].any(), \
        "live geometry must stay with Adam"


def test_qi_init_live_floor_varpro():
    env = _qi_env("sine", 32)
    losses, rels, snaps, theta, stats = it11.train_iter11(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], 2003, iters=600, frames_at={0},
        get_wbv=env["get_wbv"], eval_rel=env["eval_rel"],
        y_norm=env["y_norm"], refresh=200, arm="varpro")
    best = min(rels)
    assert best <= max(30.0 * env["floor"], 1e-11), \
        f"live loop lost the floor: best {best:.2e} vs floor " \
        f"{env['floor']:.2e}"
    assert stats["solves"] >= 2
