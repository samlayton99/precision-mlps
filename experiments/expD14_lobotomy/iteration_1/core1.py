"""expD14 iteration 1 -- clearing the feature-learning confound.

THE QUESTION
  iteration_0 found that solving L exactly in-training suppresses geometry
  learning from a random init. But every solving arm carried two throttles
  (the line-searched A step and the gain-allocated energy split) that were
  never run with them off, so "the solve stuns Adam" is indistinguishable
  from "the throttles stunned Adam". Those imply opposite designs.

THE RUN
  Five arms separate the mechanisms, all scored on the FLOOR TRAJECTORY
  (the error one terminal exact solve would give from the current geometry),
  which is the only metric that distinguishes "solved the readout" from
  "learned the geometry":

    adam    stock Adam, no solve                        (baseline)
    full    exact solve + Adam's own step on A, no split (fully unthrottled)
    ls      exact solve + line-searched A step, no split (one throttle)
    lobo0   exact solve + line search + gain split       (iteration_0's arm)
    tau1    tau=1 damped LSQR + Adam's own step, no split (cheap under-solve)

  NOTE the `full` and `tau1` arms are EXPECTED to sawtooth or diverge in
  reached error (the ||v||*eta re-injection); that is irrelevant here because
  the score is the floor trajectory, not the reached error.

  Cases are Sam's four-regime matrix (expD15) plus the deployment init:
    qi          correct uniform geometry, data everywhere
    clustered   centers pushed to the edges (r -> r^0.45), data everywhere
    datagap     correct geometry, 96% of data removed from |x| < 0.4
    rand        stock PyTorch init (w,b ~ U(-1,1))  -- iteration_0's regime
    rand_scale  gamma-regime slopes, random centers -- the deployment regime
                (motivation.md: "initialization in the right gamma regime")

  qi/clustered/datagap/rand_scale start with expD15's half-solved readout
  protocol (damped half-solve + noise, rescaled to explain half the target).
  v = 0 is a measured trap: every geometry Jacobian column is proportional
  to v_k, so at v = 0 they all vanish and any probe admits the geometry.

DISCOVERY
  Method C (expD15 per-tensor sampling with verification) replaces the
  iteration-11 dense curvature probe -- roadmap item 2. Two J evaluations
  per tensor, counted honestly in `passes`.

NEW INSTRUMENTS (both O(d), both from ORIENTATION section 4)
  coherent travel  D(W) = ||sum dtheta|| / sum ||dtheta|| on the (w,b) block
                   over a 100-step window; ~1 coherent, ~1/sqrt(W) random walk
  corrected SNR    shat = clip((mean(mhat^2/vhat) - q)/(1 - q), 0, 1) on the
                   (w,b) block, q = (1-b1)/(1+b1); the noise-floor correction
                   matters: pure noise reads 0.053 uncorrected
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

_spec0 = importlib.util.spec_from_file_location(
    "d14_core0", HERE.parent / "iteration_0" / "core0.py")
core0 = importlib.util.module_from_spec(_spec0)
_spec0.loader.exec_module(core0)
d08run = core0.d08run

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD14_lobotomy" / "iteration_1"
FIGS = RESULTS / "figures"

N_TRAIN, N_EVAL = 2003, 4001
LAM = 0.25
RCOND = 1e-15
GAP_RADIUS, GAP_KEEP = 0.4, 0.04
COL_FLOOR = 1e-8
ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0
PROBE_DELTA, PROBE_TOL = 1e-2, 1e-10   # analytic J: tol above its noise floor

append, load = core0.append, core0.load
lsqr_damped_op, power_sigma1 = core0.lsqr_damped_op, core0.power_sigma1


# =========================================================== the problem ====

def build_case(fn_name, N, case="qi", seed=0):
    """1-D tanh MLP, theta = (w | b | v | c0), m = 3W+1, five cases."""
    f_vec = d08run.make_fn(d08run.TARGETS[fn_name])
    centers, gamma, _halo = d08run.geometry(N, LAM)
    W = centers.size
    m = 3 * W + 1
    rng = np.random.default_rng(1000 + seed)
    edge = 1.0

    cen = centers.copy()
    if case == "clustered":
        # expD15's geometry defect: push centers away from the middle,
        # r -> r^0.45, count preserved, middle emptied. Data untouched.
        r = np.abs(cen)
        cen = np.sign(cen) * edge * (np.maximum(r, 1e-12) / edge) ** 0.45

    if case in ("qi", "clustered", "datagap"):
        w0 = np.full(W, gamma)
        b0 = -gamma * cen
    elif case == "rand":
        # stock PyTorch nn.Linear init, iteration_0's protocol exactly
        rng_r = np.random.default_rng(1234 + seed)
        w0 = rng_r.uniform(-1, 1, W)
        b0 = rng_r.uniform(-1, 1, W)
        v_stock = rng_r.uniform(-1, 1, W) / np.sqrt(W)
    elif case == "rand_scale":
        # the deployment init: slopes in the right gamma regime (log-uniform
        # within [gamma/2, 2*gamma]), centers random over the halo'd extent
        w0 = gamma * np.exp(rng.uniform(-np.log(2.0), np.log(2.0), W))
        c_rand = rng.uniform(centers.min(), centers.max(), W)
        b0 = -w0 * c_rand
    else:
        raise ValueError(case)

    x_all = np.linspace(-edge, edge, N_TRAIN)
    if case == "datagap":
        inside = np.abs(x_all) < GAP_RADIUS
        keep = ~inside
        idx_in = np.nonzero(inside)[0]
        n_keep = max(1, int(round(GAP_KEEP * idx_in.size)))
        keep[rng.choice(idx_in, n_keep, replace=False)] = True
        x_tr = x_all[keep]
    else:
        x_tr = x_all
    y_tr = f_vec(x_tr)
    x_ev = np.linspace(-edge, edge, N_EVAL)
    y_ev = f_vec(x_ev)
    ev_in = np.abs(x_ev) < GAP_RADIUS

    # ---- the start-state readout ----
    if case == "rand":
        v0, c00 = v_stock, 0.0
    else:
        # expD15's half-solved protocol: damped solve (mu = (1e-6 s1)^2 --
        # truncation explodes on the clustered geometry), take half plus
        # noise, rescale so the start explains half the target's norm.
        z = np.tanh(x_tr[:, None] * w0[None, :] + b0[None, :])
        A_aug = np.hstack([z, np.ones((x_tr.size, 1))])
        U, s, Vt = np.linalg.svd(A_aug, full_matrices=False)
        mu_ = (1e-6 * s[0]) ** 2
        d = Vt.T @ ((s / (s ** 2 + mu_)) * (U.T @ y_tr))
        v0 = 0.5 * d[:W] + 0.05 * np.abs(d[:W]).mean() * rng.standard_normal(W)
        c00 = 0.5 * d[W]
        pred = z @ v0 + c00
        npred = np.linalg.norm(pred)
        if npred > 0:
            k = 0.5 * np.linalg.norm(y_tr) / npred
            v0, c00 = v0 * k, c00 * k

    theta0 = torch.tensor(np.concatenate([w0, b0, v0, [c00]]))

    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(y_tr)
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(y_ev)
    ye_norm = float(torch.linalg.norm(ye))
    ye_in = ye[torch.tensor(ev_in)]
    ye_out = ye[torch.tensor(~ev_in)]
    n_in = float(torch.linalg.norm(ye_in))
    n_out = float(torch.linalg.norm(ye_out))

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

    def f_pred(th):
        return model(xt, th)

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - yt).detach()

    def f_jvp(th, dvec):
        _, jd = torch.func.jvp(f_pred, (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed_v):
        thr = th.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(f_pred(thr), thr, grad_outputs=seed_v)
        return g.detach()

    def f_residgrad(th):
        thr = th.detach().requires_grad_(True)
        pred = model(xt, thr)
        r = (pred - yt).detach()
        (g,) = torch.autograd.grad(pred, thr,
                                   grad_outputs=(2.0 / x_tr.size) * r)
        return r, g.detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(model(xe, th) - ye)) / ye_norm

    def eval_rel_split(th):
        """(global, inside |x|<GAP_RADIUS, outside). A global average hides
        the datagap effect entirely (ORIENTATION section 5, metric 1)."""
        with torch.no_grad():
            e = model(xe, th) - ye
            g = float(torch.linalg.norm(e)) / ye_norm
            ei = float(torch.linalg.norm(e[torch.tensor(ev_in)])) / n_in
            eo = float(torch.linalg.norm(e[torch.tensor(~ev_in)])) / n_out
        return g, ei, eo

    def _floor_split(th):
        """Terminal-exact-solve proxy from the CURRENT geometry: truncated-SVD
        readout fit on the (kept) training data, scored on the full eval grid,
        globally and split by region."""
        t = th.detach().numpy()
        z = np.tanh(x_tr[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        A = np.hstack([z, np.ones((x_tr.size, 1))])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        keep_s = s > RCOND * s[0]
        sol = Vt.T @ (np.where(keep_s, 1.0 / np.where(keep_s, s, 1.0), 0.0)
                      * (U.T @ y_tr))
        ze = np.tanh(x_ev[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        err = np.hstack([ze, np.ones((x_ev.size, 1))]) @ sol - y_ev
        fl = float(np.linalg.norm(err) / np.linalg.norm(y_ev))
        fi = float(np.linalg.norm(err[ev_in]) / np.linalg.norm(y_ev[ev_in]))
        fo = float(np.linalg.norm(err[~ev_in]) / np.linalg.norm(y_ev[~ev_in]))
        return fl, fi, fo

    def f_jac_full(th):
        """Full analytic J (n x m), one pass. Reference-scale only."""
        with torch.no_grad():
            zz = xt * th[:W] + th[W:2 * W]
            t = torch.tanh(zz)
            sech2 = 1.0 - t * t
            v = th[2 * W:3 * W]
            return torch.cat([sech2 * v * xt, sech2 * v, t,
                              torch.ones(xt.shape[0], 1, dtype=th.dtype)], 1)

    def f_jac_cols(th, cols):
        return f_jac_full(th)[:, cols]

    groups = {"w": np.arange(0, W), "b": np.arange(W, 2 * W),
              "v": np.arange(2 * W, 3 * W), "c0": np.array([3 * W])}
    idx_readout = torch.arange(2 * W, 3 * W + 1)

    fl0 = _floor_split(theta0)
    return dict(theta0=theta0, W=W, m=m, centers=cen, gamma=gamma,
                fn=fn_name, N=N, case=case, seed=seed,
                floor0=fl0[0], floor0_in=fl0[1], floor0_out=fl0[2],
                geometry_floor=lambda th: _floor_split(th)[0],
                geometry_floor_split=_floor_split,
                f_pred=f_pred, f_resid=f_resid, f_jvp=f_jvp, f_vjp=f_vjp,
                f_residgrad=f_residgrad, eval_rel=eval_rel,
                eval_rel_split=eval_rel_split,
                f_jac_full=f_jac_full, f_jac_cols=f_jac_cols,
                groups=groups, idx_readout=idx_readout,
                y_norm=float(torch.linalg.norm(yt)),
                n_train=int(x_tr.size), x_tr=x_tr, y_tr=y_tr)


# ================================================= Method C discovery =======

def discover_pertensor(env, theta, seed=0):
    """expD15 Method C: per-tensor sampling with verification. Visit tensors
    in descending total value; sample 2, probe against the current L; if the
    samples are clean, tentatively admit the whole tensor and verify once.
    Returns (bool mask over m, J-evaluation count)."""
    rr = np.random.default_rng(seed)
    th = theta.detach().numpy()
    m = th.size
    J0 = env["f_jac_full"](theta).numpy()
    n0 = np.maximum(np.linalg.norm(J0, axis=0), 1e-300)
    r = env["f_resid"](theta).numpy()
    val = (J0.T @ r) ** 2 / np.maximum((J0 * J0).sum(0), 1e-300)
    groups = env["groups"]
    names = sorted(groups, key=lambda nm: -float(val[groups[nm]].sum()))
    tval = {nm: float(val[groups[nm]].sum()) for nm in names}
    inL = np.zeros(m, bool)
    captured, evals = 0.0, 1

    def probe(idx):
        L = np.sort(np.asarray(idx))
        z = rr.integers(0, 2, len(L)).astype(float) * 2 - 1
        s = np.zeros(m)
        s[L] = PROBE_DELTA * np.maximum(np.abs(th[L]), 1e-2) * z
        J1 = env["f_jac_full"](theta + torch.tensor(s)).numpy()
        return L, np.linalg.norm(J1[:, L] - J0[:, L], axis=0) / n0[L]

    for pos_i, nm in enumerate(names):
        rem = sum(tval[q] for q in names[pos_i:])
        if captured > 0 and rem < 1e-3 * captured:
            break
        idx = groups[nm]
        S = rr.choice(idx, min(2, len(idx)), replace=False)
        L, mv = probe(np.concatenate([np.nonzero(inL)[0], S]))
        evals += 1
        pos = {p: j for j, p in enumerate(L)}
        if not all(mv[pos[c]] <= PROBE_TOL for c in S):
            continue
        L2, mv2 = probe(np.concatenate([np.nonzero(inL)[0], idx]))
        evals += 1
        if mv2.max() <= PROBE_TOL:
            inL[idx] = True
            captured += float(val[idx].sum())
    mask = torch.zeros(m, dtype=torch.bool)
    mask[np.nonzero(inL)[0]] = True
    return mask, evals


# ============================================== the two new diagnostics =====

def coherent_travel(sum_vec, sum_len):
    """D(W) = ||sum dtheta|| / sum ||dtheta||: 1 = coherent, 1/sqrt(W) = walk."""
    return float(torch.linalg.norm(sum_vec)) / sum_len if sum_len > 0 else 0.0


def snr_corrected(mh, vh, mask, b1):
    """Noise-floor-corrected Adam SNR on the masked block (ORIENTATION sec 4).
    Pure noise gives E[m^2]/E[v] = q = (1-b1)/(1+b1), not zero."""
    q = (1.0 - b1) / (1.0 + b1)
    ratio = (mh[mask] ** 2 / (vh[mask] + 1e-300)).mean()
    return float(np.clip((float(ratio) - q) / (1.0 - q), 0.0, 1.0))


# ================================================================ trainer ===

def train_lobo1(env, iters, *, lr=1e-3, betas=(0.9, 0.999), eps_adam=1e-8,
                lset="pertensor", refresh=200, tau=3, lsolver="direct",
                alpha_fixed=1e-15, rho_mode="fixed", rho_fixed=0.0,
                clip=False, astep="adam", record=5, floor_every=50,
                travel_window=100, seed=0, verbose=False):
    """iteration_0's optimizer with: Method C discovery, the floor trajectory,
    coherent travel and corrected SNR on the (w,b) block, region-split eval.
    lset="none" reduces it to stock Adam (verified bit-for-bit in t0)."""
    b1, b2 = betas
    theta = env["theta0"].clone()
    m, W = env["m"], env["W"]
    f_jvp, f_vjp = env["f_jvp"], env["f_vjp"]
    n_tr = env["n_train"]
    geo = torch.zeros(m, dtype=torch.bool)
    geo[:2 * W] = True                      # the architecture's (w,b) block

    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    t_adam = 0

    passes = 0
    if lset == "none":
        members = torch.zeros(m, dtype=torch.bool)
    elif lset == "readout":
        members = torch.zeros(m, dtype=torch.bool)
        members[env["idx_readout"]] = True
    elif lset == "pertensor":
        members, ev = discover_pertensor(env, theta, seed=seed)
        passes += ev
    else:
        raise ValueError(lset)
    idx = torch.nonzero(members, as_tuple=False).flatten()
    n_member_changes, n_discoveries = 0, 1 if lset == "pertensor" else 0

    pw = torch.randn(int(idx.numel()), dtype=theta.dtype) if idx.numel() else None
    sigma1 = 1.0
    trav_vec = torch.zeros(m, dtype=theta.dtype)
    trav_len = 0.0

    hist = {k: [] for k in ("it", "loss", "rel", "rel_in", "rel_out", "rho",
                            "B", "dnorm", "lsqr_it", "sigma1", "vnorm",
                            "lenA", "nA", "snr", "passes")}
    fhist = {k: [] for k in ("it", "floor", "floor_in", "floor_out")}
    thist = {k: [] for k in ("it", "travel")}

    def masked_ops(th):
        def av(z):
            dv = torch.zeros(m, dtype=th.dtype)
            dv[idx] = z
            return f_jvp(th, dv)

        def atu(u):
            return f_vjp(th, u)[idx]
        return av, atu

    def log_floor(it):
        fl, fi, fo = env["geometry_floor_split"](theta)
        fhist["it"].append(it)
        fhist["floor"].append(fl)
        fhist["floor_in"].append(fi)
        fhist["floor_out"].append(fo)

    log_floor(0)
    for it in range(1, iters + 1):
        r, g = env["f_residgrad"](theta)
        passes += 2

        t_adam += 1
        ma = b1 * ma + (1.0 - b1) * g
        va = b2 * va + (1.0 - b2) * g * g
        mh = ma / (1.0 - b1 ** t_adam)
        vh = va / (1.0 - b2 ** t_adam)
        adam_full = -lr * mh / (vh.sqrt() + eps_adam)

        # ---- A: Adam on the non-selected coordinates ----
        stepA = adam_full.clone()
        if idx.numel():
            stepA[idx] = 0.0
        nA = float(torch.linalg.norm(stepA))
        pA = stepA / nA if nA > 0 else stepA

        # ---- L: the solve ----
        dmu = None
        lsqr_it = 0
        if idx.numel():
            av, atu = masked_ops(theta)
            if it == 1 or (it - 1) % refresh == 0:
                sigma1, pw = power_sigma1(av, atu, pw, iters=3)
                passes += 6
            mu = (float(alpha_fixed) * sigma1) ** 2
            if lsolver == "lsqr":
                dmu, lsqr_it, _, _ = lsqr_damped_op(av, atu, -r, mu,
                                                    int(idx.numel()), tau)
                passes += 2 * max(lsqr_it, 1)
            elif lsolver == "direct":
                J = env["f_jac_cols"](theta, idx)
                cn = torch.linalg.norm(J, dim=0)
                cmax = float(cn.max()) if cn.numel() else 0.0
                live = cn > COL_FLOOR * cmax if cmax > 0 else cn > 0
                dmu = torch.zeros(int(idx.numel()), dtype=theta.dtype)
                if bool(live.any()):
                    Jl = J[:, live]
                    U, s, Vt = torch.linalg.svd(Jl, full_matrices=False)
                    inv = torch.where(s > RCOND * s[0], s / (s ** 2 + mu),
                                      torch.zeros_like(s))
                    dmu[live] = Vt.T @ (inv * (U.T @ (-r)))
                passes += int(idx.numel())
            else:
                raise ValueError(lsolver)

        if idx.numel() and dmu is not None:
            nL = float(torch.linalg.norm(dmu))
            pL = torch.zeros(m, dtype=theta.dtype)
            if nL > 0:
                pL[idx] = dmu / nL
        else:
            nL, pL = 0.0, None

        # ---- rho and the A step length ----
        gainA = gainL = 0.0
        tstarA = float("inf")
        rho = float(rho_fixed) if rho_mode == "fixed" else 0.0
        if nA > 0 and (rho_mode == "gain" or astep == "linesearch"):
            jA = f_jvp(theta, pA)
            passes += 1
            den = float(jA.dot(jA))
            gainA = (float(r.dot(jA)) ** 2 / den) if den > 0 else 0.0
            tstarA = (-float(r.dot(jA)) / den) if den > 0 else 0.0
        if rho_mode == "gain":
            if pL is not None and nL > 0:
                jLv = f_jvp(theta, pL)
                passes += 1
                den = float(jLv.dot(jLv))
                gainL = (float(r.dot(jLv)) ** 2 / den) if den > 0 else 0.0
            tot = gainA + gainL
            rho = (gainL / tot) if tot > 0 else 0.0
        elif rho_mode != "fixed":
            raise ValueError(rho_mode)

        # ---- the step ----
        E = float(torch.linalg.norm(adam_full))
        dth = torch.zeros(m, dtype=theta.dtype)
        lenA = 0.0
        if nA > 0:
            lenA = nA if astep == "adam" else min(nA, max(tstarA, 0.0))
            dth = dth + (1.0 - rho) * lenA * pA
        if pL is not None and nL > 0:
            take = min(nL, rho * E) if clip else nL
            dth = dth + take * pL

        th_new = theta + dth
        if not np.isfinite(float(torch.linalg.norm(th_new))):
            ma.zero_()
            va.zero_()
            continue
        theta = th_new

        # ---- coherent travel on the (w,b) block ----
        trav_vec += dth * geo
        trav_len += float(torch.linalg.norm(dth[geo]))
        if it % travel_window == 0:
            thist["it"].append(it)
            thist["travel"].append(coherent_travel(trav_vec[geo], trav_len))
            trav_vec.zero_()
            trav_len = 0.0

        # ---- rediscover on a cadence ----
        if lset == "pertensor" and it % refresh == 0:
            new, ev = discover_pertensor(env, theta, seed=seed + it)
            passes += ev
            n_discoveries += 1
            if not torch.equal(new, members):
                n_member_changes += 1
                members = new
                idx = torch.nonzero(members, as_tuple=False).flatten()
                pw = torch.randn(int(idx.numel()), dtype=theta.dtype) \
                    if idx.numel() else None

        if it % floor_every == 0 or it == iters:
            log_floor(it)
        if it % record == 0 or it == iters:
            rel, rin, rout = env["eval_rel_split"](theta)
            with torch.no_grad():
                rr_ = env["f_resid"](theta)
            hist["it"].append(it)
            hist["loss"].append(float(rr_.dot(rr_)) / n_tr)
            hist["rel"].append(rel)
            hist["rel_in"].append(rin)
            hist["rel_out"].append(rout)
            hist["rho"].append(float(rho))
            hist["B"].append(int(idx.numel()))
            hist["dnorm"].append(float(nL))
            hist["lsqr_it"].append(int(lsqr_it))
            hist["sigma1"].append(float(sigma1))
            hist["vnorm"].append(float(torch.linalg.norm(theta[env["idx_readout"]])))
            hist["lenA"].append(float(lenA))
            hist["nA"].append(float(nA))
            hist["snr"].append(snr_corrected(mh, vh, geo, b1))
            hist["passes"].append(int(passes))
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:5d}  rel {hist['rel'][-1]:.3e}  "
                  f"floor {fhist['floor'][-1]:.3e}  B {int(idx.numel())}")

    fl, fi, fo = env["geometry_floor_split"](theta)
    return dict(hist=hist, fhist=fhist, thist=thist, theta=theta,
                best_rel=float(np.min(hist["rel"])), final_rel=hist["rel"][-1],
                floor_final=fl, floor_final_in=fi, floor_final_out=fo,
                passes=passes, B_final=int(idx.numel()),
                n_member_changes=n_member_changes,
                n_discoveries=n_discoveries)


# ============================================================== the arms ====

ARMS = {
    # baseline: no solve
    "adam":  dict(lset="none"),
    # exact solve, Adam's own step on A, no energy split: fully unthrottled
    "full":  dict(lset="pertensor", lsolver="direct", alpha_fixed=1e-15,
                  rho_mode="fixed", rho_fixed=0.0, clip=False, astep="adam"),
    # one throttle: the exact-line-search cap on the A step
    "ls":    dict(lset="pertensor", lsolver="direct", alpha_fixed=1e-15,
                  rho_mode="fixed", rho_fixed=0.0, clip=False,
                  astep="linesearch"),
    # both throttles: iteration_0's headline (nodamp) configuration
    "lobo0": dict(lset="pertensor", lsolver="direct", alpha_fixed=1e-15,
                  rho_mode="gain", clip=False, astep="linesearch"),
    # the cheap deployable under-solver, unthrottled
    "tau1":  dict(lset="pertensor", lsolver="lsqr", tau=1, alpha_fixed=1e-8,
                  rho_mode="fixed", rho_fixed=0.0, clip=False, astep="adam"),
}
