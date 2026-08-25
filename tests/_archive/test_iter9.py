"""iteration_9 permanent tests: the <10-vector core (no B) + ceiling-fixed
tether + staged recycling.

  1. Linear self-removal: on an exactly linear model the tether never
     engages (q floor + rounding-level drift) and T stays 1.
  2. Readout untetherable + T ceiling: even with an absurd ratchet, T is
     capped at T_CEIL and the readout block's S stays exactly 1.
  3. QI-polish floors WITHOUT B (the staged-gate result as a regression
     test): from the fp64 QI construction with frozen geometry, the
     memoryless core reaches <= 1e-11 eval rel L2.
  4. Recycler no-op guard: with no safely-recyclable capacity it returns
     None (the optimizer degrades to plain slim — dl_test safety).
  5. Recycler near-lossless absorb: recycling saturated low-utilization
     neurons changes the prediction by less than the residual scale.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "expd08_run", REPO_ROOT / "experiments" / "expD08_qi_init_nlcg" / "run.py")
d08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d08)

torch.set_default_dtype(torch.float64)


def _mlp_env(fn_name, N, theta0=None):
    f_vec = d08.make_fn(d08.TARGETS[fn_name])
    if theta0 is None:
        centers, gamma, _ = d08.geometry(N, d08.LAM)
        W = centers.size
        theta0 = torch.tensor(np.concatenate([
            np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))
    else:
        W = (theta0.numel() - 1) // 3
    x_tr = np.linspace(-1, 1, 601)
    x_ev = np.linspace(-1, 1, 1001)
    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(f_vec(x_tr))
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

    def f_pred(th):
        return model(xt, th)

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - yt).detach()

    def f_jvp(th, dv):
        _, jd = torch.func.jvp(f_pred, (th,), (dv,))
        return jd.detach()

    def f_vjp(th, seed):
        thr = th.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(f_pred(thr), thr, grad_outputs=seed)
        return g.detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(model(xe, th) - ye)
                         / torch.linalg.norm(ye))

    def get_wbv(th):
        t = th.detach().numpy()
        return {"w": t[:W].copy(), "b": t[W:2 * W].copy(),
                "v": t[2 * W:3 * W].copy()}

    return dict(theta0=theta0, f_pred=f_pred, f_resid=f_resid, f_jvp=f_jvp,
                f_vjp=f_vjp, eval_rel=eval_rel, get_wbv=get_wbv,
                n=len(x_tr), y_norm=float(torch.linalg.norm(yt)),
                W=W, xt=xt)


def test_linear_self_removal():
    rng = np.random.default_rng(0)
    A = torch.tensor(rng.normal(size=(200, 40)))
    y = A.mv(torch.tensor(rng.normal(size=40)))   # consistent system
    th0 = torch.zeros(40)

    def f_pred(th):
        with torch.no_grad():
            return A.mv(th)

    def f_resid(th):
        return f_pred(th) - y

    def f_jvp(th, dv):
        with torch.no_grad():
            return A.mv(dv)

    def f_vjp(th, seed):
        with torch.no_grad():
            return A.t().mv(seed)

    losses, rels, _, _, stats = d08.train_iter9(
        th0, f_resid, f_jvp, f_vjp, f_pred, 200, 800, {0},
        lambda th: {"w": np.zeros(1), "b": np.zeros(1), "v": np.zeros(1)},
        lambda th: float(torch.linalg.norm(f_resid(th))
                         / torch.linalg.norm(y)),
        float(torch.linalg.norm(y)))
    assert stats["trips"] == []
    assert stats["T_final"] == 1.0
    assert min(rels) < 1e-10       # plain CG floors a small linear system


def test_readout_untetherable_and_ceiling():
    env = _mlp_env("sine", 32)
    frames = {0, 400, 800}
    _, _, snaps, _, stats = d08.train_iter9(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], env["n"], 800, frames, env["get_wbv"],
        env["eval_rel"], env["y_norm"], ratchet=1e40, snap_S=True,
        recycle_fn=None)
    assert stats["T_final"] <= d08.T_CEIL
    W = env["W"]
    for s in snaps.values():
        if "S" in s:
            assert float(np.max(s["S"][2 * W:])) == 1.0   # readout + bias


def test_qi_polish_floors_without_B():
    from src.construction.qi_mpmath import construct_qi
    from src.data.targets import get_target
    t = get_target("sine")
    res = construct_qi(t.fn_numpy, t.deriv_numpy, 64, precision="fp64")
    W = res.centers.size
    theta0 = torch.tensor(np.concatenate([
        np.full(W, res.gamma), -res.gamma * res.centers,
        res.a_coeffs, [res.c0]]))
    env = _mlp_env("sine", 64, theta0=theta0)
    mask0 = torch.zeros(3 * W + 1)
    mask0[2 * W:] = 1.0            # geometry frozen: readout polish only
    _, rels, _, _, _ = d08.train_iter9(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], env["n"], 1500, {0}, env["get_wbv"],
        env["eval_rel"], env["y_norm"], recycle_fn=None, mask0=mask0)
    assert min(rels) <= 1e-11


def test_recycler_noop_when_all_used():
    # every neuron in-range (no saturated halo), O(1) readout, and a residual
    # far smaller than any neuron's content: nothing is safely recyclable
    W = 24
    xt = torch.linspace(-1, 1, 301).unsqueeze(1)
    centers = np.linspace(-0.9, 0.9, W)
    th = torch.tensor(np.concatenate([
        np.full(W, 2.0), -2.0 * centers, np.ones(W), [0.0]]))
    rhat = 1e-8 * torch.ones(301)
    rec = d08.make_recycler(W, xt, w_cap=100.0)
    assert rec(th, rhat) is None


def test_recycler_lossless_absorb():
    env = _mlp_env("sine", 32)
    W = env["W"]
    th = env["theta0"].clone()
    th[2 * W:3 * W] = 1.0
    # saturate a third of the neurons far outside the data and give them
    # tiny readout: near-constant columns, near-lossless to absorb
    idx = np.arange(0, W, 3)
    th[idx] = 50.0
    th[W + idx] = 200.0            # tanh(50x + 200) == 1 on [-1, 1]
    th[2 * W + idx] = 1e-6
    rhat = env["f_resid"](th)
    rec = d08.make_recycler(W, env["xt"], w_cap=100.0)
    out = rec(th, rhat)
    assert out is not None
    th_new, mask, info = out
    with torch.no_grad():
        before = env["f_pred"](th)

        def pred_frozen(t):
            return env["f_pred"](t)
        # zero the recycled readouts' new contribution (v=0 already) and
        # compare: absorb moved their constant content into the bias
        after = env["f_pred"](th_new)
    change = float(torch.linalg.norm(after - before))
    assert change <= 0.5 * float(torch.linalg.norm(rhat))
    assert info["n"] >= len(idx) * 0.5
    assert float(mask[-1]) == 1.0
    assert float(mask[:2 * W].max()) == 0.0       # geometry never active
