"""iteration_4 suite -- shared machinery.

`train_generic` is core4's assembled optimizer (Method C discovery +
alpha = r_entry damping + f_gen cooling) generalized to any env:
  - env["geo_mask"] replaces the hardcoded (w,b) block
  - mu and sigma1 are logged (the suite's signal plots want them)
  - snapshots for the parameter-movement gifs

Env builders:
  env_1d      core4.build_case (five cases, fit/holdout split)
  env_2d      expD15 core15's four 2-D ridge cases, torch-wrapped
  env_model   any nn.Module + (xtr, ytr, xte, yte): closures via autograd,
              full Jacobian via one vmap'd jacrev on the fit rows
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

_s4 = importlib.util.spec_from_file_location(
    "d14_core4", HERE.parent / "core4.py")
core4 = importlib.util.module_from_spec(_s4)
_s4.loader.exec_module(core4)
core1 = core4.core1

_s15 = importlib.util.spec_from_file_location(
    "d15_core", REPO / "experiments" / "expD15_inclusion_score" / "core15.py")
core15 = importlib.util.module_from_spec(_s15)
_s15.loader.exec_module(core15)

RESULTS = core4.RESULTS / "suite"
FIGS = RESULTS / "figures"
RCOND = 1e-15
COL_FLOOR = 1e-8
ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0

append, load = core4.append, core4.load
env_1d = core4.build_case


# ---------------------------------------------------------------- 2-D env --

def env_2d(case, seed=0, N=48, holdout_every=10):
    """expD15's four 2-D ridge cases (qi / clustered / datagap / random),
    torch-wrapped with a fit/holdout split. theta = (A.ravel()|b|v|c0)."""
    e = core15.build_case(case, dim=2, N=N, seed=seed)
    net = e["net"]
    W, dim = net.W, net.dim
    m = net.m
    th0 = torch.tensor(net.theta.copy())
    Xall, yall = e["Xtr"], e["ytr"]
    ho = np.zeros(Xall.shape[0], bool)
    ho[holdout_every // 2::holdout_every] = True
    Xf, yf = torch.tensor(Xall[~ho]), torch.tensor(yall[~ho])
    Xh, yh = torch.tensor(Xall[ho]), torch.tensor(yall[ho])
    Xe, ye = torch.tensor(e["Xev"]), torch.tensor(e["yev"])
    ye_norm = float(torch.linalg.norm(ye))

    def model(X, th):
        A = th[:W * dim].reshape(W, dim)
        return torch.tanh(X @ A.T + th[W * dim:W * dim + W]) \
            @ th[W * dim + W:W * dim + 2 * W] + th[-1]

    def jac(X, th):
        with torch.no_grad():
            A = th[:W * dim].reshape(W, dim)
            t = torch.tanh(X @ A.T + th[W * dim:W * dim + W])
            s = (1.0 - t * t) * th[W * dim + W:W * dim + 2 * W]
            JA = (s[:, :, None] * X[:, None, :]).reshape(X.shape[0], W * dim)
            return torch.cat([JA, s, t,
                              torch.ones(X.shape[0], 1, dtype=th.dtype)], 1)

    def f_residgrad(th):
        thr = th.detach().requires_grad_(True)
        pred = model(Xf, thr)
        r = (pred - yf).detach()
        (g,) = torch.autograd.grad(pred, thr,
                                   grad_outputs=(2.0 / Xf.shape[0]) * r)
        return r, g.detach()

    def floor_fn(th, idx=None):
        J = jac(Xf, th)[:, W * dim + W:]          # oracle readout block
        U, s, Vt = torch.linalg.svd(J, full_matrices=False)
        keep = s > RCOND * s[0]
        sol = Vt.T @ (torch.where(keep, 1.0 / torch.where(keep, s,
                      torch.ones_like(s)), torch.zeros_like(s)) * (U.T @ yf))
        Je = jac(Xe, th)[:, W * dim + W:]
        fl = float(torch.linalg.norm(Je @ sol - ye)) / ye_norm
        return fl, float("nan"), float("nan")

    groups = {"A": np.arange(0, W * dim),
              "b": np.arange(W * dim, W * dim + W),
              "v": np.arange(W * dim + W, W * dim + 2 * W),
              "c0": np.array([m - 1])}
    geo = torch.zeros(m, dtype=torch.bool)
    geo[:W * dim + W] = True

    return dict(theta0=th0, m=m, W=W, dim=dim, case=case, fn="target2d",
                n_train=int(Xf.shape[0]), geo_mask=geo, groups=groups,
                idx_readout=torch.arange(W * dim + W, m),
                floor0=floor_fn(th0)[0], floor_fn=floor_fn,
                f_residgrad=f_residgrad,
                f_resid=lambda th: (model(Xf, th) - yf).detach(),
                f_resid_ho=lambda th: (model(Xh, th) - yh).detach(),
                f_jvp=lambda th, d: torch.func.jvp(
                    lambda t: model(Xf, t), (th,), (d,))[1].detach(),
                f_vjp=lambda th, u: _vjp(model, Xf, th, u),
                f_jac_full=lambda th: jac(Xf, th),
                f_jac_cols=lambda th, cols: jac(Xf, th)[:, cols],
                f_jac_cols_ho=lambda th, cols: jac(Xh, th)[:, cols],
                eval_rel=lambda th: float(torch.linalg.norm(
                    model(Xe, th) - ye)) / ye_norm,
                y_norm=float(torch.linalg.norm(yf)),
                y_norm_ho=float(torch.linalg.norm(yh)))


def _vjp(model, X, th, u):
    thr = th.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(model(X, thr), thr, grad_outputs=u)
    return g.detach()


# ------------------------------------------------------------- model env --

def env_model(model_fn, theta0, groups, xtr, ytr, xte, yte,
              holdout_every=10):
    """Generic env from a functional model(X, theta)->pred. groups maps
    tensor names to index arrays into theta. Full Jacobian via vmap'd
    jacrev on the fit rows -- suite-scale only."""
    n = xtr.shape[0]
    ho = np.zeros(n, bool)
    ho[holdout_every // 2::holdout_every] = True
    hot = torch.tensor(ho)
    Xf, yf = xtr[~hot], ytr[~hot]
    Xh, yh = xtr[hot], ytr[hot]
    m = theta0.numel()
    yte_norm = float(torch.linalg.norm(yte))

    def jac(X, th):
        with torch.no_grad():
            return torch.func.vmap(
                torch.func.jacrev(lambda t, x: model_fn(x[None], t)[0]),
                in_dims=(None, 0))(th, X).detach()

    def f_residgrad(th):
        thr = th.detach().requires_grad_(True)
        pred = model_fn(Xf, thr)
        r = (pred - yf).detach()
        (g,) = torch.autograd.grad(pred, thr,
                                   grad_outputs=(2.0 / Xf.shape[0]) * r)
        return r, g.detach()

    def floor_fn(th, idx):
        """Exact solve on the DISCOVERED block from the current theta,
        scored on the test set (test relative L2)."""
        if idx is None or not len(idx):
            return float("nan"), float("nan"), float("nan")
        J = jac(Xf, th)[:, idx]
        r = (model_fn(Xf, th) - yf).detach()
        U, s, Vt = torch.linalg.svd(J, full_matrices=False)
        keep = s > RCOND * s[0]
        d = Vt.T @ (torch.where(keep, 1.0 / torch.where(keep, s,
                    torch.ones_like(s)), torch.zeros_like(s)) * (U.T @ (-r)))
        th2 = th.clone()
        th2[idx] += d
        fl = float(torch.linalg.norm(model_fn(xte, th2) - yte)) / yte_norm
        return fl, float("nan"), float("nan")

    return dict(theta0=theta0.clone(), m=m, groups=groups,
                geo_mask=None,      # set after first discovery
                idx_readout=None, floor_fn=floor_fn, floor0=float("nan"),
                n_train=int(Xf.shape[0]),
                f_residgrad=f_residgrad,
                f_resid=lambda th: (model_fn(Xf, th) - yf).detach(),
                f_resid_ho=lambda th: (model_fn(Xh, th) - yh).detach(),
                f_jvp=lambda th, d: torch.func.jvp(
                    lambda t: model_fn(Xf, t), (th,), (d,))[1].detach(),
                f_vjp=lambda th, u: _vjp(model_fn, Xf, th, u),
                f_jac_full=lambda th: jac(Xf, th),
                f_jac_cols=lambda th, cols: jac(Xf, th)[:, cols],
                f_jac_cols_ho=lambda th, cols: jac(Xh, th)[:, cols],
                eval_rel=lambda th: float(torch.linalg.norm(
                    model_fn(xte, th) - yte)) / yte_norm,
                test_mse=lambda th: float(torch.mean(
                    (model_fn(xte, th) - yte) ** 2)),
                train_mse=lambda th: float(torch.mean(
                    (model_fn(Xf, th) - yf) ** 2)),
                y_norm=float(torch.linalg.norm(yf)),
                y_norm_ho=float(torch.linalg.norm(yh)))


# --------------------------------------------------------------- trainer --

def train_generic(env, iters, *, cool="fgen", lr=1e-3, betas=(0.9, 0.999),
                  eps_adam=1e-8, refresh=200, record=5, floor_every=50,
                  snap_every=0, seed=0, verbose=False):
    """core4.train_v4 generalized: geo_mask from env (or complement of the
    discovered L), floor via env["floor_fn"] when present, mu and sigma1
    logged, optional snapshots."""
    b1, b2 = betas
    theta = env["theta0"].clone()
    m = env["m"]
    y_norm = env["y_norm"]

    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    t_adam = 0
    passes = 0

    members, ev = core1.discover_pertensor(env, theta, seed=seed)
    passes += ev
    idx = torch.nonzero(members, as_tuple=False).flatten()
    geo = env.get("geo_mask")
    if geo is None:
        geo = ~members
    n_member_changes = 0

    sigma1 = 1.0
    pw = torch.randn(int(idx.numel()), dtype=theta.dtype)

    hist = {k: [] for k in ("it", "rel", "alpha", "mu", "sigma1", "fgen",
                            "shat", "coolf", "B", "vnorm", "passes",
                            "train_mse", "test_mse")}
    fhist = {k: [] for k in ("it", "floor")}
    snaps = {}

    def masked_ops(th):
        def av(z):
            dv = torch.zeros(m, dtype=th.dtype)
            dv[idx] = z
            return env["f_jvp"](th, dv)

        def atu(u):
            return env["f_vjp"](th, u)[idx]
        return av, atu

    def log_floor(it):
        if "geometry_floor_split" in env:
            fl = env["geometry_floor_split"](theta)[0]
        else:
            fl = env["floor_fn"](theta, idx)[0]
        fhist["it"].append(it)
        fhist["floor"].append(fl)

    log_floor(0)
    if snap_every:
        snaps["0"] = theta.numpy().copy().tolist()
    mu = 0.0
    fgen = 1.0
    s_prev = 1.0
    for it in range(1, iters + 1):
        r, g = env["f_residgrad"](theta)
        passes += 2
        r_entry = float(torch.linalg.norm(r)) / y_norm
        alpha = float(np.clip(r_entry, ALPHA_MIN, ALPHA_MAX))

        t_adam += 1
        ma = b1 * ma + (1.0 - b1) * g
        va = b2 * va + (1.0 - b2) * g * g
        mh = ma / (1.0 - b1 ** t_adam)
        vh = va / (1.0 - b2 ** t_adam)
        adam_full = -lr * mh / (vh.sqrt() + eps_adam)
        shat = core1.snr_corrected(mh, vh, geo, b1)

        dmu = None
        fgen = 1.0
        if idx.numel():
            av, atu = masked_ops(theta)
            if it == 1 or (it - 1) % refresh == 0:
                sigma1, pw = core1.power_sigma1(av, atu, pw, iters=3)
                passes += 6
            J = env["f_jac_cols"](theta, idx)
            cn = torch.linalg.norm(J, dim=0)
            cmax = float(cn.max()) if cn.numel() else 0.0
            live = cn > COL_FLOOR * cmax if cmax > 0 else cn > 0
            dmu = torch.zeros(int(idx.numel()), dtype=theta.dtype)
            if bool(live.any()):
                Jl = J[:, live]
                U, s, Vt = torch.linalg.svd(Jl, full_matrices=False)
                keep = s > RCOND * s[0]
                utr = U.T @ (-r)
                if cool in ("fgen", "fgen2") or \
                        (cool == "fabsc" and (it == 1 or
                                              (it - 1) % refresh == 0)):
                    dstar = Vt.T @ (torch.where(keep, 1.0 / torch.where(
                        keep, s, torch.ones_like(s)),
                        torch.zeros_like(s)) * utr)
                    r_ho = env["f_resid_ho"](theta)
                    J_ho = env["f_jac_cols_ho"](theta, idx)[:, live]
                    passes += 1
                    r_ho_new = r_ho + J_ho @ dstar
                    nho = float(torch.linalg.norm(r_ho))
                    if cool == "fabsc":
                        s_prev = float(np.clip(
                            float(torch.linalg.norm(r_ho_new))
                            / env.get("y_norm_ho", y_norm), 0.0, 1.0))
                    else:
                        fgen = float(np.clip(
                            float(torch.linalg.norm(r_ho_new))
                            / max(nho, 1e-300), 0.0, 1.0))
                if cool == "fgen2":
                    alpha = float(np.clip(r_entry * fgen,
                                          ALPHA_MIN, ALPHA_MAX))
                elif cool == "fabsc":
                    fgen = s_prev
                    alpha = float(np.clip(s_prev, ALPHA_MIN, ALPHA_MAX))
                mu = (alpha * sigma1) ** 2
                dmu[live] = Vt.T @ (torch.where(keep, s / (s ** 2 + mu),
                                    torch.zeros_like(s)) * utr)
            passes += int(idx.numel())

        stepA = adam_full.clone()
        if idx.numel():
            stepA[idx] = 0.0
        coolf = fgen if cool in ("fgen", "fgen2", "fabsc") else 1.0
        dth = coolf * stepA
        if dmu is not None:
            dth = dth.clone()
            dth[idx] += dmu

        th_new = theta + dth
        if not np.isfinite(float(torch.linalg.norm(th_new))):
            ma.zero_()
            va.zero_()
            continue
        theta = th_new

        if it % refresh == 0:
            new, ev = core1.discover_pertensor(env, theta, seed=seed + it)
            passes += ev
            if not torch.equal(new, members):
                n_member_changes += 1
                members = new
                idx = torch.nonzero(members, as_tuple=False).flatten()
                if env.get("geo_mask") is None:
                    geo = ~members
                pw = torch.randn(int(idx.numel()), dtype=theta.dtype) \
                    if idx.numel() else None

        if snap_every and it % snap_every == 0:
            snaps[str(it)] = theta.numpy().copy().tolist()
        if it % floor_every == 0 or it == iters:
            log_floor(it)
        if it % record == 0 or it == iters:
            hist["it"].append(it)
            hist["rel"].append(env["eval_rel"](theta))
            hist["alpha"].append(float(alpha))
            hist["mu"].append(float(mu))
            hist["sigma1"].append(float(sigma1))
            hist["fgen"].append(float(fgen))
            hist["shat"].append(float(shat))
            hist["coolf"].append(float(coolf))
            hist["B"].append(int(idx.numel()))
            vn = float(torch.linalg.norm(theta[idx])) if idx.numel() else 0.0
            hist["vnorm"].append(vn)
            hist["passes"].append(int(passes))
            hist["train_mse"].append(env["train_mse"](theta)
                                     if "train_mse" in env else float("nan"))
            hist["test_mse"].append(env["test_mse"](theta)
                                    if "test_mse" in env else float("nan"))
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:5d}  rel {hist['rel'][-1]:.3e}  "
                  f"fgen {fgen:.2e}  floor {fhist['floor'][-1]:.3e}")

    if "geometry_floor_split" in env:
        fl = env["geometry_floor_split"](theta)[0]
    else:
        fl = env["floor_fn"](theta, idx)[0]
    return dict(hist=hist, fhist=fhist, snaps=snaps, theta=theta,
                best_rel=float(np.min(hist["rel"])), final_rel=hist["rel"][-1],
                floor_final=fl, passes=passes,
                n_member_changes=n_member_changes,
                L_size=int(idx.numel()))


def train_adam(env, iters, *, lr=1e-3, betas=(0.9, 0.999), eps_adam=1e-8,
               record=5, floor_every=50, snap_every=0):
    """Stock Adam under the same env, for baselines."""
    b1, b2 = betas
    theta = env["theta0"].clone()
    m = env["m"]
    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    hist = {k: [] for k in ("it", "rel", "train_mse", "test_mse")}
    fhist = {"it": [], "floor": []}
    snaps = {}

    def log_floor(it):
        if "geometry_floor_split" in env:
            fl = env["geometry_floor_split"](theta)[0]
        else:
            members, _ = core1.discover_pertensor(env, theta, seed=0)
            idx = torch.nonzero(members).flatten()
            fl = env["floor_fn"](theta, idx)[0]
        fhist["it"].append(it)
        fhist["floor"].append(fl)

    log_floor(0)
    if snap_every:
        snaps["0"] = theta.numpy().copy().tolist()
    for it in range(1, iters + 1):
        _, g = env["f_residgrad"](theta)
        ma = b1 * ma + (1 - b1) * g
        va = b2 * va + (1 - b2) * g * g
        theta = theta - lr * (ma / (1 - b1 ** it)) / \
            ((va / (1 - b2 ** it)).sqrt() + eps_adam)
        if snap_every and it % snap_every == 0:
            snaps[str(it)] = theta.numpy().copy().tolist()
        if it % floor_every == 0 or it == iters:
            log_floor(it)
        if it % record == 0 or it == iters:
            hist["it"].append(it)
            hist["rel"].append(env["eval_rel"](theta))
            hist["train_mse"].append(env["train_mse"](theta)
                                     if "train_mse" in env else float("nan"))
            hist["test_mse"].append(env["test_mse"](theta)
                                    if "test_mse" in env else float("nan"))
    if "geometry_floor_split" in env:
        fl = env["geometry_floor_split"](theta)[0]
    else:
        members, _ = core1.discover_pertensor(env, theta, seed=0)
        idx = torch.nonzero(members).flatten()
        fl = env["floor_fn"](theta, idx)[0]
    return dict(hist=hist, fhist=fhist, snaps=snaps, theta=theta,
                best_rel=float(np.min(hist["rel"])), final_rel=hist["rel"][-1],
                floor_final=fl)
