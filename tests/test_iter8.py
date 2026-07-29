"""iteration_8 unit tests: self-removal on linear problems, hard bounds
(cap, readout untetherable), memoryless-T property, frozen-quadratic
floor, and the zero-blowup minibatch guard."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "expD08_qi_init_nlcg"))
import run as d08  # noqa: E402

torch.set_default_dtype(torch.float64)


def _linear_problem(m=40, n=200, cond=1e6, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, m)))
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    s = np.logspace(0, -np.log10(cond), m)
    A = torch.tensor(U * s @ V.T)
    theta_star = torch.tensor(rng.standard_normal(m))
    y = A @ theta_star
    return A, y, theta_star


def _closures(A, y):
    def f_resid(th, idx=None):
        with torch.no_grad():
            r = A @ th - y
            return r if idx is None else r[idx]

    def f_jvp(th, dvec, idx=None):
        jd = A @ dvec
        return jd if idx is None else jd[idx]

    def f_vjp(th, seed, idx=None):
        if idx is None:
            return A.t() @ seed
        return A[idx].t() @ seed

    def f_pred(th):
        return A @ th

    return f_resid, f_jvp, f_vjp, f_pred


def test_linear_self_removal_and_floor():
    A, y, _ = _linear_problem()
    f_resid, f_jvp, f_vjp, f_pred = _closures(A, y)
    ynorm = float(torch.linalg.norm(y))

    def eval_rel(th):
        return float(torch.linalg.norm(A @ th - y)) / ynorm

    theta0 = torch.zeros(A.shape[1])
    losses, rels, snaps, theta, stats = d08.train_iter8(
        theta0, f_resid, f_jvp, f_vjp, f_pred, y.numel(), 400,
        {0, 400}, lambda th: {"w": th.numpy()[:1], "b": th.numpy()[:1],
                              "v": th.numpy()[:1]},
        eval_rel, ynorm, snap_S=True)
    assert min(rels) < 1e-11, f"linear floor not reached: {min(rels):.2e}"
    # tether self-removal: a linear map probes q == 0, so S stays 1
    for s_ in snaps.values():
        assert float(np.max(s_["S"])) == 1.0
    assert stats["probes"] >= 1


def test_memoryless_T_and_bounds():
    A, y, _ = _linear_problem(cond=1e10, seed=1)
    f_resid, f_jvp, f_vjp, f_pred = _closures(A, y)
    ynorm = float(torch.linalg.norm(y))
    theta0 = torch.zeros(A.shape[1])
    _, rels, _, _, stats = d08.train_iter8(
        theta0, f_resid, f_jvp, f_vjp, f_pred, y.numel(), 300, {0},
        lambda th: {"w": th.numpy()[:1], "b": th.numpy()[:1],
                    "v": th.numpy()[:1]},
        lambda th: float(torch.linalg.norm(A @ th - y)) / ynorm, ynorm)
    T = np.array(stats["T_traj"])
    # controller properties: hard bounds, and the NO-STEPS invariant --
    # per-step log motion never exceeds the controller gain x clamp
    lnT = np.log(T)
    assert T.max() <= d08.T_CAP and T.min() >= 1.0
    assert np.max(np.abs(np.diff(lnT))) <= d08.ETA_T * 2.0 + 1e-12


def test_full_model_readout_untetherable():
    r = d08.run_case({"fn": "sine", "N": 32, "iters": 400, "threads": 2,
                      "save_snaps": True, "snap_S": True})
    assert np.isfinite(r["best_rel"]) and r["best_rel"] < 5e-2
    W = (len(r["snaps"]["0"]["w"]))
    for s_ in r["snaps"].values():
        S = np.asarray(s_["S"])
        assert S.max() <= 1.0 + d08.T_CAP
        assert float(np.max(S[2 * W:3 * W])) == 1.0, \
            "readout got tethered"


def test_minibatch_no_blowup_no_halo():
    # the halo-study failure shape in miniature: [-1,1] data (halo neurons
    # data-blind), tiny batches. iteration_8 must stay bounded.
    fn = d08.make_fn(d08.TARGETS["sine"])
    centers, gamma, _ = d08.geometry(32, d08.LAM)
    W = centers.size
    theta0 = torch.tensor(np.concatenate([
        np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))
    x = np.linspace(-1, 1, 400)
    xt = torch.tensor(x).unsqueeze(1)
    yt = torch.tensor(fn(x))

    def model(x_, th):
        return torch.tanh(x_ * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] \
            + th[-1]

    def f_resid(th, idx=None):
        with torch.no_grad():
            if idx is None:
                return (model(xt, th) - yt).detach()
            return (model(xt[idx], th) - yt[idx]).detach()

    def f_jvp(th, dvec, idx=None):
        x_ = xt if idx is None else xt[idx]
        _, jd = torch.func.jvp(lambda t: model(x_, t), (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed, idx=None):
        x_ = xt if idx is None else xt[idx]
        thr = th.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(model(x_, thr), thr, grad_outputs=seed)
        return g.detach()

    def f_pred(th):
        return model(xt, th)

    rng = np.random.default_rng(0)

    def gen():
        while True:
            perm = rng.permutation(400)
            for i in range(0, 400, 7):
                yield torch.from_numpy(perm[i:i + 7].copy())

    losses, rels, snaps, theta, stats = d08.train_iter8(
        theta0, f_resid, f_jvp, f_vjp, f_pred, 400, 300, {0},
        lambda th: {"w": th.numpy()[:W], "b": th.numpy()[W:2 * W],
                    "v": th.numpy()[2 * W:3 * W]},
        lambda th: 0.0, float(torch.linalg.norm(yt)), batch_iter=gen())
    assert np.all(np.isfinite(rels)) and np.all(np.isfinite(losses))
    assert float(theta.abs().max()) < 1e3, \
        f"blowup: max|theta| = {float(theta.abs().max()):.1e}"
