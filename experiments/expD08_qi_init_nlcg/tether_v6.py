"""Tether v6: CONTINUOUS bounded auto-tether -- no trips, no ratchet, no
unbounded T (Sam directive, optimizer_audit.md).

The controller. At every refresh the drift gauge reads the nonlinearity
accumulated over the window, delta = ||r_carried - r_fresh||. Instead of
tripping a monotone ratchet, T is a STATE updated by a fixed monotone
function of the current reading:

    target = max(BETA * ||r_fresh||, 10 * eps * ||y||)      (noise-guarded)
    T <- clip(T * (delta / target)^KAPPA, 1, T_MAX)

Model: geometry damage per window is second order in step size, so
delta ~ D0 / T^2; in log space the update  logT += KAPPA*(log delta -
log target)  has fixed point delta = target and is critically damped at
KAPPA = 1/2 (d logT_next / d logT = 1 - 2*KAPPA = 0). Two-sided by
construction: when the geometry stops bending, delta falls below target
and T decays -- the tether relaxes on its own. The per-refresh change is
rate-limited to a factor RHO (metric jolts stay small).

Blowup / readout-strangling are structurally impossible:
  * q_hat = q / max(q), and q_hat below QREL_FLOOR = 1e-10 is set to 0 --
    the fp64 second-difference probe cannot resolve ratios below ~1e-10,
    so smaller readings are "measured zero" (the gauge-noise lesson
    applied to the probe itself). Linear coordinates get S == 1 exactly.
  * T_MAX = 1e12 caps everything else.
  S_i = 1 + T * q_hat_i as in v4/iter6 (anisotropy is load-bearing, v5).

Metric-vs-conjugacy variants (the measured variable, one at a time):
  A  metric form, basis KEPT across the (rate-limited) metric change,
     direction restarted on any T change. Bet: small jolts leave the
     solved span usable; double projection + exhaustion mop up leakage.
  B  damping form: NO metric anywhere. CG runs on raw coordinates and the
     step is alpha = -g.d / (c + d.M d), M_i = lhat * T * q_hat_i
     (per-coordinate Levenberg damping -- anisotropic, unlike v5's fatal
     scalar). Conjugacy untouched by T changes.
  C  metric form + full basis reset on T change (control = old behavior
     with the new controller).

Benches:
  --select   12-cell slice (TARGETS3 x N in {64,256,512}) x variants
  --grid     full 30-run grid for one variant (default the select winner)
  --depth2   2-hidden-layer tanh MLP from RANDOM init (both layers must
             learn; the lock is NOT "geometry vs readout") on sine+runge,
             v6 vs adam vs slim-untethered vs iter7; per-tensor probe.
  --render   figures (selection bars, grid compare, tether magnitude)

Outputs under results/.../expD08_qi_init_nlcg/tether/v6/.
Run:  uv run python experiments/expD08_qi_init_nlcg/tether_v6.py --select
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

expd06 = run.expd06
OUT = run.OUT_DIR.parent / "tether" / "v6"

# COMMITTED constants (selection slices 2026-07-25; see tether_v6.md).
# THE open issue is that no single (BETA, RHO, DEADBAND) spans all
# widths: N=512 engagement wants (1e-8, 100, 2.0) -- exp 3.3e-15 below
# floor there -- but that same point rams T to 1e10 while small-N
# geometry is still legitimately learning (sine@32: 2.9e-2, five orders
# lost). Shipped point below = broadest measured support (strong at
# N=256, sane everywhere); the width bracket is documented in
# tether_v6.md as the annealed-target ("uncertainty dial") lead.
BETA = 1e-6          # target = max(BETA*min||r||, noise floor)
KAPPA = 0.5          # controller exponent (critically damped, see header)
T_MAX = 1e12         # hard cap
QREL_FLOOR = 1e-10   # probe resolution: smaller q ratios are measured zero
RHO = 10.0           # max T change per window (ramp speed cap)
DOWN_FRAC = 0.1      # relax 10x slower than lock (asymmetric rates: the
                     # tether "breathes" and loses orders if it unlocks as
                     # eagerly as it locks -- measured; Adam's slow
                     # second-moment decay is the same asymmetry)
REPROBE_DECADES = 1.0
DEADBAND = 0.05      # decades; hold T (and the metric) for smaller moves


def controller_step(T, drift, rnorm, y_norm, frac=1.0):
    """One controller update. `frac` = (steps since last read)/refresh:
    the exponent and the rate limit both scale with it, so the aggregate
    update over any 200 steps is cadence-invariant (decoupled-gauge form;
    frac=1 recovers the refresh-coupled update)."""
    noise_floor = 10.0 * 2.22e-16 * y_norm
    # rnorm passed in is the RUNNING MIN of ||r||: if the run gets worse,
    # the target does not relax with it (a diverging r must read as MORE
    # nonlinearity, never as license to loosen -- measured death spiral
    # with the naive current-r target under fast read cadences).
    target = max(BETA * rnorm, noise_floor)
    step = KAPPA * frac * np.log10(max(drift, 1e-300) / target)
    lim = frac * np.log10(RHO)
    step = min(max(step, -lim), lim)
    if step < 0:
        step *= DOWN_FRAC
    return min(max(T * 10.0 ** step, 1.0), T_MAX)


def qhat_from(q_acc):
    qm = float(q_acc.max())
    if qm <= 0.0:
        return torch.zeros_like(q_acc)
    qh = q_acc / qm
    qh[qh < QREL_FLOOR] = 0.0
    return qh


def train_v6(theta0, f_resid, f_jvp, f_vjp, f_pred, n_train, iters,
             frames_at, get_wbv, eval_rel, y_norm, refresh=200,
             variant="A", probe_fn=None, snap_S=False, read_every=None):
    """Continuous-tether trainer. probe_fn(theta)->q (defaults to run.py's
    full per-coordinate probe). Returns (losses, rels, snaps, theta, stats).
    stats: T_hist [(it,T)], probes, exhaust, full_reset, revert, refresh.

    Decoupled gauge (Sam directive): the drift gauge is READ every
    `read_every` steps at the cost of one extra forward pass -- the fresh
    residual is compared to the carried one and DISCARDED (never written
    to r_hat, so no noise re-roll). Residual REPLACEMENT stays on the
    `refresh` timer. The controller state T evolves continuously at read
    cadence (cadence-invariant aggregate, see controller_step); the
    METRIC is re-applied (with a basis reset, variant C) only when T has
    moved more than DEADBAND decades from the applied value.
    read_every=None couples reads to the refresh timer (old behavior)."""
    if probe_fn is None:
        def probe_fn(th):
            return run.probe_self_curvature(f_pred, th)
    m = theta0.numel()
    metric = variant in ("A", "C")
    S = torch.ones(m, dtype=theta0.dtype)
    qh = torch.zeros(m, dtype=theta0.dtype)
    phi = theta0.clone()
    q_acc = None
    T = 1.0
    logT_at_probe = 0.0

    def raw(p):
        return p / S if metric else p

    rhat = f_resid(raw(phi))
    g = f_vjp(raw(phi), (2.0 / n_train) * rhat)
    if metric:
        g = g / S
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0, "probes": 0, "T_hist": [],
             "T_final": 1.0, "trips": []}

    def project(vec):
        if not basis:
            return vec.clone()
        B = torch.stack(basis)
        out = vec - B.t().mv(B.mv(vec))
        return out - B.t().mv(B.mv(out))

    def append_to_basis(vec):
        v = project(vec)
        n2 = float(v.dot(v))
        if n2 > 1e-300 and len(basis) < m:
            basis.append(v / float(np.sqrt(n2)))

    def snap_at(it):
        snaps[it] = get_wbv(raw(phi))
        if snap_S:
            snaps[it]["S"] = (1.0 + T * qh).numpy().copy()

    if 0 in frames_at:
        snap_at(0)
    gt = g.clone()
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    exhaust_streak = 0

    T_ctrl = 1.0            # continuous controller state (read cadence)
    it_last_read = 0
    it_last_replace = 0
    rnorm_min = float("inf")
    k_read = read_every if read_every is not None else refresh

    for it in range(1, iters + 1):
        is_replace = it % refresh == 0
        is_read = is_replace or (it % k_read == 0)
        if is_read:
            r_fresh = f_resid(raw(phi))
            drift_now = float(torch.linalg.norm(rhat - r_fresh))
            rnorm_min = min(rnorm_min,
                            float(torch.linalg.norm(r_fresh)))
            steps = max(it - it_last_read, 1)
            it_last_read = it
            # running drift since replacement, projected to one window;
            # at the replacement read this IS the refresh-coupled reading
            elapsed = max(it - it_last_replace, 1)
            reading = drift_now * (refresh / elapsed)
            T_ctrl = controller_step(T_ctrl, reading, rnorm_min, y_norm,
                                     frac=steps / refresh)
            stats["T_hist"].append((it, T_ctrl))
            if is_replace:                      # replacement: noise
                rhat = r_fresh                  # re-roll, rate-limited
                stats["refresh"] += 1
                it_last_replace = it
            # read-only case: fresh residual DISCARDED (no noise re-roll)
            changed = abs(np.log10(max(T_ctrl, 1e-300))
                          - np.log10(max(T, 1e-300))) > DEADBAND
            if changed and (q_acc is None or (np.log10(max(T_ctrl, 1.0))
                            - logT_at_probe) >= REPROBE_DECADES):
                q = probe_fn(raw(phi))
                q_acc = q if q_acc is None else torch.maximum(q_acc, q)
                stats["probes"] += 1
                logT_at_probe = np.log10(max(T_ctrl, 1.0))
            if changed:
                T = T_ctrl
                qh = qhat_from(q_acc) if q_acc is not None else qh
            if (changed and metric and q_acc is not None
                    and float(qh.max()) > 0):
                S_new = 1.0 + T * qh
                ratio = S_new / S
                phi = phi * ratio
                S = S_new
                if variant == "C":
                    basis = []
                    stats["full_reset"] += 1
                elif variant == "D" and basis:
                    # transport: rescale each vector, re-orthonormalize
                    nb: list[torch.Tensor] = []
                    for b_ in basis:
                        v = b_ * ratio
                        for u in nb:
                            v = v - u * float(u.dot(v))
                        n2 = float(v.dot(v))
                        if n2 > 1e-300:
                            v = v / float(np.sqrt(n2))
                            for u in nb:
                                v = v - u * float(u.dot(v))
                            n2 = float(v.dot(v))
                            if n2 > 1e-300:
                                nb.append(v / float(np.sqrt(n2)))
                    basis = nb
                lhat = None
            if is_replace or changed:
                g = f_vjp(raw(phi), (2.0 / n_train) * rhat)
                if metric:
                    g = g / S
                gt = project(g)
                ggt = float(gt.dot(gt))
                if changed:
                    d = -gt
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break
        jd = f_jvp(raw(phi), (d / S) if metric else d)
        c = (2.0 / n_train) * float(jd.dot(jd))
        if variant == "B":
            damp = float((d * d).dot(qh)) * (lhat if lhat else 0.0) * T
            denom = c + damp
        else:
            denom = c
        if denom > 0 and np.isfinite(denom):
            alpha = -float(g.dot(d)) / denom
            if c > 0:
                lam = c / dn2
                lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        phi_new = phi + alpha * d
        rhat_new = rhat + alpha * jd
        g_new = f_vjp(raw(phi_new), (2.0 / n_train) * rhat_new)
        if metric:
            g_new = g_new / S
        if not np.isfinite(float(torch.linalg.norm(g_new))):
            stats["revert"] += 1
            d = -gt
            if it in frames_at:
                snap_at(it)
            continue
        append_to_basis(gt)
        gt_new = project(g_new)
        if float(gt_new.dot(gt_new)) < 1e-24 * float(g_new.dot(g_new)):
            stats["exhaust"] += 1
            exhaust_streak += 1
            gt_new = g_new.clone()
            if exhaust_streak >= 20:
                basis = []
                stats["full_reset"] += 1
                exhaust_streak = 0
            beta = 0.0
        else:
            exhaust_streak = 0
            beta = float(gt_new.dot(gt_new - gt)) / ggt if ggt > 0 else 0.0
        d = -gt_new + beta * d
        phi, rhat, g, gt = phi_new, rhat_new, g_new, gt_new
        ggt = float(gt.dot(gt))
        losses.append(float(rhat.dot(rhat)) / n_train)
        rels.append(eval_rel(raw(phi)))
        if it in frames_at:
            snap_at(it)
    stats["T_final"] = T
    final = get_wbv(raw(phi))
    if snap_S:
        final["S"] = (1.0 + T * qh).numpy().copy()
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final
    return losses, rels, snaps, raw(phi), stats


# ---------------- toy (single-hidden QI) closures + runner ----------------

def toy_closures(fn_name, N):
    """Same problem setup as run.run_case (copied: run_case is not
    factored for reuse and is the sibling agent's lane)."""
    fn_text = run.TARGETS[fn_name]
    f_vec = run.make_fn(fn_text)
    centers, gamma, _ = run.geometry(N, run.LAM)
    W = centers.size
    theta0 = torch.tensor(np.concatenate([
        np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))
    x_tr = np.linspace(-1, 1, run.N_TRAIN)
    x_ev = np.linspace(-1, 1, run.N_EVAL)
    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(f_vec(x_tr))
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))
    ye_norm = float(torch.linalg.norm(ye))

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

    def f_vjp(th, seed):
        thr = th.detach().requires_grad_(True)
        pred = f_pred(thr)
        (gg,) = torch.autograd.grad(pred, thr, grad_outputs=seed)
        return gg.detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(model(xe, th) - ye)) / ye_norm

    def get_wbv(th):
        t = th.detach().numpy()
        return {"w": t[:W].copy(), "b": t[W:2 * W].copy(),
                "v": t[2 * W:3 * W].copy()}

    A = np.hstack([run.build_phi(x_tr, gamma, centers, "tanh"),
                   np.ones((x_tr.size, 1))])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > run.RCOND * s[0]
    sol = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                  * (U.T @ f_vec(x_tr)))
    A_ev = np.hstack([run.build_phi(x_ev, gamma, centers, "tanh"),
                      np.ones((x_ev.size, 1))])
    floor = float(np.linalg.norm(A_ev @ sol - f_vec(x_ev))
                  / np.linalg.norm(f_vec(x_ev)))
    return dict(theta0=theta0, f_pred=f_pred, f_resid=f_resid, f_jvp=f_jvp,
                f_vjp=f_vjp, eval_rel=eval_rel, get_wbv=get_wbv,
                floor=floor, y_norm=float(torch.linalg.norm(yt)))


def run_case_v6(job):
    global BETA, DEADBAND
    torch.set_num_threads(job.get("threads", 1))
    torch.set_default_dtype(torch.float64)
    if job.get("beta") is not None:
        BETA = job["beta"]
    if job.get("deadband") is not None:
        DEADBAND = job["deadband"]
    global RHO
    if job.get("rho") is not None:
        RHO = job["rho"]
    C = toy_closures(job["fn"], job["N"])
    iters = job.get("iters", run.ITERS)
    frames_at = (set(expd06.frame_steps(iters)) if job.get("save_snaps")
                 else {0})
    t0 = time.time()
    losses, rels, snaps, _, stats = train_v6(
        C["theta0"], C["f_resid"], C["f_jvp"], C["f_vjp"], C["f_pred"],
        run.N_TRAIN, iters, frames_at, C["get_wbv"], C["eval_rel"],
        C["y_norm"], variant=job.get("variant", "A"),
        read_every=job.get("read_every"),
        snap_S=job.get("snap_S", False))
    out = {"variant": job.get("variant", "A"), "beta": BETA,
           "read": job.get("read_every"), "deadband": DEADBAND,
           "rho": RHO, "fn": job["fn"],
           "N": job["N"], "floor": C["floor"], "rels": rels,
           "best_rel": float(np.min(rels)) if rels else float("nan"),
           "T_hist": stats["T_hist"], "T_final": stats["T_final"],
           "probes": stats["probes"], "refresh": stats["refresh"],
           "exhaust": stats["exhaust"], "wall_s": round(time.time() - t0, 1)}
    if job.get("snap_S") and job.get("save_snaps"):
        sd = OUT / "snaps"
        sd.mkdir(parents=True, exist_ok=True)
        ss = sorted(snaps)
        np.savez_compressed(
            sd / f"{job['fn']}--N{job['N']}--{out['variant']}.npz",
            steps=np.array(ss, dtype=np.int64),
            w=np.stack([snaps[s]["w"] for s in ss]),
            b=np.stack([snaps[s]["b"] for s in ss]),
            v=np.stack([snaps[s]["v"] for s in ss]),
            S=np.stack([snaps[s]["S"] for s in ss]),
            rels=np.array(rels), floor=C["floor"],
            stats=json.dumps({k: stats[k] for k in
                              ("T_hist", "T_final", "probes", "refresh")}))
    return out


# --------------------------- depth-2 bench ---------------------------

D2_H = (64, 64)
D2_FNS = ["sine", "runge"]
D2_ITERS = 3000
D2_NTR, D2_NEV = 2003, 4001


def mlp2_closures(fn_name, seed=0, h=D2_H):
    """2-hidden-layer tanh MLP, RANDOM init (both layers must learn)."""
    f_vec = run.make_fn(run.TARGETS[fn_name])
    h1, h2 = h
    gen = torch.Generator().manual_seed(seed)
    shapes = [("W1", (h1, 1)), ("b1", (h1,)), ("W2", (h2, h1)),
              ("b2", (h2,)), ("v", (h2,)), ("c0", (1,))]
    fan_in = {"W1": 1, "b1": 1, "W2": h1, "b2": h1, "v": h2, "c0": h2}
    parts, slices, off = [], [], 0
    for name, sh in shapes:
        n = int(np.prod(sh))
        bound = 1.0 / np.sqrt(fan_in[name])
        parts.append((torch.rand(n, generator=gen, dtype=torch.float64)
                      * 2 - 1) * bound)
        slices.append((name, off, n, sh))
        off += n
    theta0 = torch.cat(parts)
    m = off
    x_tr = np.linspace(-1, 1, D2_NTR)
    x_ev = np.linspace(-1, 1, D2_NEV)
    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(f_vec(x_tr))
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))
    ye_norm = float(torch.linalg.norm(ye))

    def model(x, th):
        i = dict((nm, (o, n, sh)) for nm, o, n, sh in slices)
        W1o, W1n, W1s = i["W1"]
        b1o, b1n, _ = i["b1"]
        W2o, W2n, W2s = i["W2"]
        b2o, b2n, _ = i["b2"]
        vo, vn, _ = i["v"]
        c0o, _, _ = i["c0"]
        z1 = torch.tanh(x @ th[W1o:W1o + W1n].reshape(W1s).t()
                        + th[b1o:b1o + b1n])
        z2 = torch.tanh(z1 @ th[W2o:W2o + W2n].reshape(W2s).t()
                        + th[b2o:b2o + b2n])
        return z2 @ th[vo:vo + vn] + th[c0o]

    def f_pred(th):
        return model(xt, th)

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - yt).detach()

    def f_jvp(th, dvec):
        _, jd = torch.func.jvp(f_pred, (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed_):
        thr = th.detach().requires_grad_(True)
        (gg,) = torch.autograd.grad(f_pred(thr), thr, grad_outputs=seed_)
        return gg.detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(model(xe, th) - ye)) / ye_norm

    def get_wbv(th):
        t = th.detach().numpy()
        return {nm: t[o:o + n].copy() for nm, o, n, _ in slices}

    def probe_per_tensor(th, n_samp=6, eps_rel=1e-3):
        rng = np.random.default_rng(0)
        scores = torch.zeros(m, dtype=th.dtype)
        with torch.no_grad():
            for base in (th, th + 0.1 * torch.clamp(th.abs(), min=1.0)
                         * torch.from_numpy(
                             rng.integers(0, 2, m) * 2.0 - 1.0)):
                p0 = f_pred(base)
                for _, o, n, _ in slices:
                    idx = rng.choice(n, size=min(n_samp, n), replace=False)
                    qt = 0.0
                    for ii in idx:
                        ii = int(o + ii)
                        s = max(abs(float(base[ii])), 1.0)
                        eps = eps_rel * s
                        tp = base.clone()
                        tp[ii] += eps
                        tm = base.clone()
                        tm[ii] -= eps
                        d2 = f_pred(tp) - 2.0 * p0 + f_pred(tm)
                        qt = max(qt, float(torch.linalg.norm(d2))
                                 / eps ** 2 * s ** 2)
                    scores[o:o + n] = torch.maximum(
                        scores[o:o + n],
                        torch.full((n,), qt, dtype=th.dtype))
        return scores

    def final_lstsq_rel(th):
        """Reference: exact readout solve at the found features."""
        with torch.no_grad():
            t = th
            i = dict((nm, (o, n, sh)) for nm, o, n, sh in slices)
            W1o, W1n, W1s = i["W1"]
            b1o, b1n, _ = i["b1"]
            W2o, W2n, W2s = i["W2"]
            b2o, b2n, _ = i["b2"]
            z1 = torch.tanh(xt @ t[W1o:W1o + W1n].reshape(W1s).t()
                            + t[b1o:b1o + b1n])
            z2 = torch.tanh(z1 @ t[W2o:W2o + W2n].reshape(W2s).t()
                            + t[b2o:b2o + b2n])
            A = np.hstack([z2.numpy(), np.ones((D2_NTR, 1))])
            sol, *_ = np.linalg.lstsq(A, yt.numpy(), rcond=1e-15)
            z1e = torch.tanh(xe @ t[W1o:W1o + W1n].reshape(W1s).t()
                             + t[b1o:b1o + b1n])
            z2e = torch.tanh(z1e @ t[W2o:W2o + W2n].reshape(W2s).t()
                             + t[b2o:b2o + b2n])
            Ae = np.hstack([z2e.numpy(), np.ones((D2_NEV, 1))])
            return float(np.linalg.norm(Ae @ sol - ye.numpy()) / ye_norm)

    return dict(theta0=theta0, m=m, f_pred=f_pred, f_resid=f_resid,
                f_jvp=f_jvp, f_vjp=f_vjp, eval_rel=eval_rel,
                get_wbv=get_wbv, probe=probe_per_tensor, slices=slices,
                y_norm=float(torch.linalg.norm(yt)), yt=yt,
                final_lstsq_rel=final_lstsq_rel)


def run_depth2(job):
    torch.set_num_threads(job.get("threads", 1))
    torch.set_default_dtype(torch.float64)
    C = mlp2_closures(job["fn"], seed=job.get("seed", 0))
    opt = job["opt"]
    iters = job.get("iters", D2_ITERS)
    t0 = time.time()
    T_hist, S_hist = [], []
    if opt == "adam":
        th = C["theta0"].clone().requires_grad_(True)
        optim = torch.optim.Adam([th], lr=1e-3)
        rels = []
        for it in range(iters):
            optim.zero_grad()
            r = C["f_pred"](th) - C["yt"]
            loss = (r * r).mean()
            loss.backward()
            optim.step()
            rels.append(C["eval_rel"](th.detach()))
        stats = {}
        theta_fin = th.detach()
    elif opt in ("v6A", "v6B", "v6C", "v6D", "iter7", "slim"):
        if opt.startswith("v6"):
            losses, rels, snaps, theta_fin, stats = train_v6(
                C["theta0"], C["f_resid"], C["f_jvp"], C["f_vjp"],
                C["f_pred"], D2_NTR, iters, {0}, C["get_wbv"],
                C["eval_rel"], C["y_norm"], variant=opt[-1],
                probe_fn=C["probe"], read_every=50)
            T_hist = stats["T_hist"]
        elif opt == "iter7":
            losses, rels, snaps, theta_fin, stats = run.train_iter6(
                C["theta0"], C["f_resid"], C["f_jvp"], C["f_vjp"],
                C["f_pred"], D2_NTR, iters, {0}, C["get_wbv"],
                C["eval_rel"], C["y_norm"], trip_mode="noise",
                ratchet=run.RATCHET)
            T_hist = [(t, run.RATCHET ** (i + 1))
                      for i, t in enumerate(stats["trips"])]
        else:  # slim untethered = v6 machinery with probe forced zero
            losses, rels, snaps, theta_fin, stats = train_v6(
                C["theta0"], C["f_resid"], C["f_jvp"], C["f_vjp"],
                C["f_pred"], D2_NTR, iters, {0}, C["get_wbv"],
                C["eval_rel"], C["y_norm"], variant="A",
                probe_fn=lambda th: torch.zeros_like(th))
    else:
        raise ValueError(opt)
    out = {"opt": opt, "fn": job["fn"], "seed": job.get("seed", 0),
           "rels": [float(x) for x in rels],
           "best_rel": float(np.min(rels)),
           "final_lstsq_rel": C["final_lstsq_rel"](theta_fin),
           "T_hist": T_hist, "wall_s": round(time.time() - t0, 1)}
    return out


# ------------------------------ mains ------------------------------

SEL_FNS = run.TARGETS3
SEL_NS = [64, 256, 512]
VARIANTS = ["C", "D"]   # A = kept stale basis: catastrophic divergence
                        # (sine@32 blew to 1e6 after first metric change);
                        # B = damping-only: T decays to 1, stalls at 2e-2.
                        # Both measured 2026-07-25, single-cell probes.


def _merge_rows(path, rows, key):
    old = json.loads(path.read_text()) if path.exists() else []
    for r in old:
        r.setdefault("beta", 0.01)
        r.setdefault("read", None)
        r.setdefault("deadband", 0.05)
        r.setdefault("rho", 10.0)
    keep = {key(r): r for r in old}
    for r in rows:
        keep[key(r)] = r
    path.write_text(json.dumps(list(keep.values())))


def main_select(workers, only_variant=None, only_n=None, beta=None,
                read_every=None, deadband=None, rho=None):
    threads = max(1, (os.cpu_count() or 4) // workers)
    global BETA
    if beta is not None:
        BETA = beta
    variants = [only_variant] if only_variant else VARIANTS
    ns = [only_n] if only_n else SEL_NS
    jobs = [{"variant": v, "fn": f, "N": n, "threads": threads,
             "beta": BETA, "read_every": read_every,
             "deadband": deadband, "rho": rho}
            for v in variants for f in SEL_FNS for n in ns]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_case_v6, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['variant']} b={r['beta']:.0e} "
                  f"rd={r['read']} "
                  f"{r['fn']:13s} N={r['N']:<4d} best={r['best_rel']:.2e} "
                  f"floor={r['floor']:.1e} T={r['T_final']:.1e} "
                  f"probes={r['probes']} ({r['wall_s']}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    _merge_rows(OUT / "select_rows.json", rows,
                lambda r: (r["variant"], r["beta"], r.get("read"),
                           r.get("deadband", 0.05), r.get("rho", 10.0),
                           r["fn"], r["N"]))
    print("merged into", OUT / "select_rows.json")


def main_grid(workers, variant, beta=None, read_every=None,
              only_n=None):
    threads = max(1, (os.cpu_count() or 4) // workers)
    ns = [only_n] if only_n else run.WIDTHS
    jobs = [{"variant": variant, "fn": f, "N": n, "threads": threads,
             "beta": beta, "read_every": read_every,
             "snap_S": n == 256 and f in run.TARGETS3,
             "save_snaps": n == 256 and f in run.TARGETS3}
            for f in run.TARGETS for n in ns]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_case_v6, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['fn']:13s} N={r['N']:<4d} "
                  f"best={r['best_rel']:.2e} floor={r['floor']:.1e} "
                  f"T={r['T_final']:.1e} ({r['wall_s']}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    _merge_rows(OUT / f"grid_rows_{variant}.json", rows,
                lambda r: (r["fn"], r["N"]))
    print("merged", OUT / f"grid_rows_{variant}.json")


def main_depth2(workers, variant):
    threads = max(1, (os.cpu_count() or 4) // workers)
    opts = [f"v6{variant}", "adam", "slim", "iter7"]
    jobs = [{"opt": o, "fn": f, "threads": threads}
            for o in opts for f in D2_FNS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_depth2, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['opt']:6s} {r['fn']:8s} "
                  f"best={r['best_rel']:.2e} "
                  f"lstsq@final={r['final_lstsq_rel']:.2e} "
                  f"({r['wall_s']}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "depth2_rows.json").write_text(json.dumps(rows))
    print("saved", OUT / "depth2_rows.json")


# ------------------------------ figures ------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_ablations():
    """Three-panel ablation figure from select_rows.json:
    (1) C vs D (transport) at beta=1e-2, refresh-coupled;
    (2) beta sweep at N=512 (rd=50, db=0.05, rho=10) + the rho100/db2
        point -- the width-engagement bracket;
    (3) read-cadence at N=256, beta=1e-6."""
    plt = _mpl()
    rows = json.loads((OUT / "select_rows.json").read_text())
    for r in rows:
        r.setdefault("beta", 0.01)
        r.setdefault("read", None)
        r.setdefault("deadband", 0.05)
        r.setdefault("rho", 10.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fcol = dict(zip(SEL_FNS, ["#1f77b4", "#d62728", "#2ca02c",
                              "#9467bd"]))

    ax = axes[0]
    for fn in SEL_FNS:
        for var, ls in [("C", "-"), ("D", ":")]:
            pts = sorted([(r["N"], r["best_rel"]) for r in rows
                          if r["variant"] == var and r["fn"] == fn
                          and r["beta"] == 0.01 and r["read"] is None])
            if pts:
                ax.plot(*zip(*pts), ls, marker="o", ms=4, lw=1.5,
                        color=fcol[fn],
                        label=f"{fn} ({'reset' if var == 'C' else 'transport'})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e0)
    ax.set_xlabel("width N")
    ax.set_ylabel("best eval rel L2")
    ax.set_title("basis on metric change: reset (C, solid)\n"
                 "vs transport (D, dotted); beta=1e-2", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2,
              fontsize=6, borderaxespad=0)

    ax = axes[1]
    betas = [1e-4, 1e-6, 1e-8]
    for fn in SEL_FNS:
        ys = []
        for b in betas:
            m = [r for r in rows if r["variant"] == "C" and r["fn"] == fn
                 and r["N"] == 512 and r["beta"] == b and r["read"] == 50
                 and r["deadband"] == 0.05 and r["rho"] == 10.0]
            ys.append(m[0]["best_rel"] if m else np.nan)
        ax.plot(betas, ys, "o-", lw=1.5, ms=5, color=fcol[fn], label=fn)
        m2 = [r for r in rows if r["variant"] == "C" and r["fn"] == fn
              and r["N"] == 512 and r["beta"] == 1e-8
              and r["deadband"] == 2.0 and r["rho"] == 100.0]
        if m2:
            ax.plot([1e-8], [m2[0]["best_rel"]], "*", ms=13,
                    color=fcol[fn])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e0)
    ax.invert_xaxis()
    ax.set_xlabel("target scale beta (stars: rho=100, deadband=2.0)")
    ax.set_title("N=512 engagement vs beta (rd=50)\n"
                 "stars = fast-ramp point (breaks N=32)", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=4,
              fontsize=7, borderaxespad=0)

    ax = axes[2]
    cadences = [1, 10, 50, None]
    xs = [1, 10, 50, 200]
    for fn in SEL_FNS:
        ys = []
        for c in cadences:
            m = [r for r in rows if r["variant"] == "C" and r["fn"] == fn
                 and r["N"] == 256 and r["beta"] == 1e-6
                 and r["read"] == c and r["deadband"] == 0.05]
            ys.append(m[0]["best_rel"] if m else np.nan)
        ax.plot(xs, ys, "o-", lw=1.5, ms=5, color=fcol[fn], label=fn)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e0)
    ax.set_xlabel("gauge read cadence (steps; 200 = refresh-only)")
    ax.set_title("read cadence at N=256, beta=1e-6\n"
                 "(fast reads churn the basis deep in the run)",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=4,
              fontsize=7, borderaxespad=0)

    fig.tight_layout()
    p = OUT / "ablations.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("saved", p)


def render_grid(variant):
    plt = _mpl()
    rows = json.loads((OUT / f"grid_rows_{variant}.json").read_text())
    by = {(r["fn"], r["N"]): r for r in rows}
    ref = {}
    for it in ("iteration_5", "iteration_7"):
        p = run.OUT_DIR.parent / it / "expD08_rows.json"
        for r in json.loads(p.read_text()):
            ref[(it, r["fn"], r["N"])] = r
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True,
                             sharey=True)
    for k, fn in enumerate(run.TARGETS):
        ax = axes.flat[k]
        floors = [by[(fn, n)]["floor"] for n in run.WIDTHS]
        ax.plot(run.WIDTHS, floors, "k--", lw=1.2, label="lstsq floor")
        for it, lab, col in [("iteration_5", "fixed T=1e6", "#999999"),
                             ("iteration_7", "ratchet (iter7)", "#d62728")]:
            ys = [ref[(it, fn, n)]["best_rel"] for n in run.WIDTHS]
            ax.plot(run.WIDTHS, ys, "o-", color=col, lw=1.4, ms=4,
                    label=lab)
        ys = [by[(fn, n)]["best_rel"] for n in run.WIDTHS]
        ax.plot(run.WIDTHS, ys, "o-", color="#1f77b4", lw=1.8, ms=5,
                label=f"v6 continuous ({variant})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e0)
        ax.set_title(fn)
        ax.grid(alpha=0.3, which="both")
        if k >= 3:
            ax.set_xlabel("width N")
        if k % 3 == 0:
            ax.set_ylabel("best eval rel L2")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=9)
    fig.suptitle(f"tether v6 ({variant}) full grid vs iter5 / iter7",
                 y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = OUT / f"grid_compare_{variant}.png"
    fig.savefig(p, dpi=140)
    print("saved", p)


def render_magnitude(variant):
    """Standing tether-magnitude diagnostic for v6: S max/median per
    block + T trajectory (log axis) + eval rel L2 (twin axis)."""
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    blocks = [("w", "#d62728"), ("b", "#ff7f0e"), ("v", "#1f77b4")]
    for k, fn in enumerate(run.TARGETS3):
        ax = axes.flat[k]
        z = np.load(OUT / "snaps" / f"{fn}--N256--{variant}.npz")
        steps = z["steps"].astype(int)
        S = z["S"]
        W = z["w"].shape[1]
        st = json.loads(str(z["stats"]))
        rels = z["rels"]
        for j, (name, col) in enumerate(blocks):
            Sb = S[:, j * W:(j + 1) * W]
            ax.step(steps, Sb.max(axis=1), where="post", color=col,
                    lw=1.6, label=f"max S ({name})")
            ax.step(steps, np.median(Sb, axis=1), where="post", color=col,
                    lw=1.0, ls="--", alpha=0.7, label=f"median S ({name})")
        th = np.array(st["T_hist"])
        if len(th):
            ax.step(th[:, 0], th[:, 1], where="post", color="k", lw=1.2,
                    label="T (controller)")
        ax.axhline(1.0, color="#2ca02c", lw=1.0, alpha=0.8)
        ax.axhline(1.0 + T_MAX, color="#2ca02c", lw=1.0, ls=":",
                   alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.5, T_MAX * 100)
        ax.set_title(f"{fn}   T_final={st['T_final']:.0e}   "
                     f"probes={st['probes']}", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        if k % 2 == 0:
            ax.set_ylabel("tether magnitude $S_i$ / T")
        if k >= 2:
            ax.set_xlabel("iteration")
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(rels) + 1), rels, color="0.45", lw=1.0)
        ax2.axhline(float(z["floor"]), color="k", ls=":", lw=0.9)
        ax2.set_yscale("log")
        ax2.set_ylim(1e-16, 1e0)
        if k % 2 == 1:
            ax2.set_ylabel("eval rel L2 (gray; dotted = floor)",
                           color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=7, fontsize=8)
    fig.suptitle(f"v6 ({variant}) tether magnitude vs learning   N=256   "
                 "(solid green: S=1; dotted green: hard cap 1+T_MAX)",
                 y=0.925, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = OUT / f"tether_magnitude_v6{variant}_N256.png"
    fig.savefig(p, dpi=140)
    print("saved", p)


def render_depth2():
    plt = _mpl()
    rows = json.loads((OUT / "depth2_rows.json").read_text())
    opts = sorted({r["opt"] for r in rows})
    colors = {"adam": "tab:green", "slim": "black", "iter7": "#d62728",
              "v6C": "#1f77b4", "v6D": "#2ca02c"}
    fig, axes = plt.subplots(2, len(D2_FNS), figsize=(12, 8), sharex=True)
    for j, fn in enumerate(D2_FNS):
        ax = axes[0, j]
        for r in rows:
            if r["fn"] != fn:
                continue
            rel = np.minimum.accumulate(np.array(r["rels"]))
            ax.semilogy(np.arange(1, len(rel) + 1), rel,
                        color=colors.get(r["opt"], "gray"),
                        label=f"{r['opt']} (lstsq@final "
                              f"{r['final_lstsq_rel']:.0e})")
            ax.set_ylim(1e-16, 1e1)
        ax.set_title(f"depth-2 MLP, {fn} (random init)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if j == 0:
            ax.set_ylabel("best-so-far eval rel L2")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2,
                  fontsize=7, borderaxespad=0)
        axT = axes[1, j]
        for r in rows:
            if r["fn"] != fn or not r["T_hist"]:
                continue
            th = np.array(r["T_hist"], dtype=float)
            axT.step(th[:, 0], th[:, 1], where="post",
                     color=colors.get(r["opt"], "gray"), label=r["opt"])
        axT.set_yscale("log")
        axT.set_ylim(0.5, T_MAX * 100)
        axT.axhline(T_MAX, color="#2ca02c", ls=":", lw=1.0)
        axT.grid(alpha=0.3, which="both")
        axT.set_xlabel("iteration")
        if j == 0:
            axT.set_ylabel("tether T (dotted green: cap)")
        axT.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                   fontsize=7, borderaxespad=0)
    fig.suptitle("depth-2 bench: both hidden layers must learn "
                 "(the lock is NOT geometry-vs-readout)", y=0.99,
                 fontsize=10)
    fig.tight_layout()
    p = OUT / "depth2_bench.png"
    fig.savefig(p, dpi=140)
    print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--depth2", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--variant", default="C")
    ap.add_argument("--only-variant", default=None)
    ap.add_argument("--only-n", type=int, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--read", type=int, default=None)
    ap.add_argument("--deadband", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.select:
        main_select(a.workers, a.only_variant, a.only_n, a.beta, a.read,
                    a.deadband, a.rho)
    if a.grid:
        main_grid(a.workers, a.variant, a.beta, a.read, a.only_n)
    if a.depth2:
        main_depth2(a.workers, a.variant)
    if a.render:
        if (OUT / "select_rows.json").exists():
            render_ablations()
        if (OUT / f"grid_rows_{a.variant}.json").exists():
            render_grid(a.variant)
        if (OUT / "snaps" / f"sine--N256--{a.variant}.npz").exists():
            render_magnitude(a.variant)
        if (OUT / "depth2_rows.json").exists():
            render_depth2()
