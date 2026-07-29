"""expD08 auto-tether: discover and apply the geometry lock with ZERO
architecture knowledge.

The fixed tether (T=1e6 on the hidden layer) is worth ~7 orders but names a
parameter block, which requirement 1 forbids. This replaces it with two
measured, architecture-blind mechanisms:

WHO to slow -- nonlinearity, not curvature. (A curvature-magnitude rule such
as Jacobi/Adam/LM is provably backwards here: at the solution the readout
TAIL directions are flatter than the geometry block, so slowing
low-curvature coordinates kills exactly the directions that must be
polished.) A coordinate is safe to polish forever iff the prediction map is
linear in it. Probed exactly, per coordinate, by a second difference:

    qraw_i = || p(theta + eps_i e_i) - 2 p(theta) + p(theta - eps_i e_i) ||
             / eps_i^2                  (~ ||d^2 p / d theta_i^2||)
    q_i    = qraw_i * max(|theta_i|, 1)^2   (bend per unit RELATIVE motion)

For a linearly-entering coordinate q_i = 0 to rounding at ANY theta. The
probe runs at the FIRST REFRESH (iteration 200), not at init: at the seed9
init the readout is zero, which makes every self-curvature vanish; after
one refresh period the readout has magnitude and the nonlinear block is
visible. v4: the tether is CONTINUOUS, S_i = 1 + T * q_i / max(q) -- every
coordinate slowed in proportion to its measured nonlinearity, no mask and
no threshold (v1-v3's binary mask failed on weakly-bending coordinates
whose q is continuously small; archived in tether/v3). Probe cost: 2(2m+1)
forward passes per probe event. Scale note (requirement 5): at large m the
per-coordinate probe is O(m) forwards; the scalable form samples a few
coordinates per tensor (parameters in one tensor share statistics), making
it O(#tensors) -- structure every framework exposes, not architecture
knowledge.

WHEN / HOW MUCH -- the carried residual is a free basin gauge. At every
refresh the optimizer holds both the linearly-propagated residual r_hat and
the fresh one; their gap delta is exactly the nonlinearity the trajectory
accumulated over the last 200 steps. Ratchet: if

    ||r_hat - r_fresh|| > 0.1 * ||r_fresh||

then T <- 100 * T, S is rebuilt from the accumulated q (elementwise max
over all probes so far -- locks never shrink), and the basis/curvature
estimate are reset (the metric changed; old conjugacy is void). T starts
at 1 (early geometry wander is allowed -- Adam moved w by ~0.5 and the
geometry still refit to the floor) and only ever increases: lock on the
way into the basin, never unlock. No loss comparisons anywhere.

Grid: 6 targets x N in {32,64,128,256,512}, 3000 iters, slim core.
Success = within ~1 order of iteration_5 (fixed T=1e6) without block names.

Run:  uv run python experiments/expD08_qi_init_nlcg/tether.py  (--smoke)
Outputs under results/checkpoint_D_optimizers/expD08_qi_init_nlcg/tether/
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
VERSION = "v5"                 # v5: PER-STEP linearization control replaces
                               # the entire probe/ratchet apparatus. v4's
                               # diagnosis: the drift gauge lags by one
                               # refresh period, so the early wander damages
                               # the geometry before the gauge can see it,
                               # and locking freezes the damage in. The fix
                               # must act per step: Levenberg damping
                               # alpha = -g.d/(c + mu ||d||^2) with mu
                               # adapted by a GRADIENT-SPACE gain ratio (no
                               # loss values). mu -> 0 reduces exactly to
                               # the slim core. No probes, no metric, no
                               # trips: T = 1 everywhere by construction.
                               # (v4 = continuous tether S = 1 + T q/max q,
                               # in tether/v4; v1-v3 binary mask, tether/v3.)
TETHER_DIR = run.OUT_DIR.parent / "tether" / VERSION
ITERS = 3000
REFRESH = run.REFRESH          # 200: probe/ratchet cadence = refresh cadence
TRIP_FRAC = 0.01               # drift > 1% of the residual -> ratchet
RATCHET = 100.0                # T multiplier per trip
PROBE_EPS = 1e-3               # relative probe amplitude


def _self_curvature_at(f_pred, theta):
    with torch.no_grad():
        p0 = f_pred(theta)
        q = torch.zeros_like(theta)
        for i in range(theta.numel()):
            s = max(abs(float(theta[i])), 1.0)
            eps = PROBE_EPS * s
            tp = theta.clone()
            tp[i] += eps
            tm = theta.clone()
            tm[i] -= eps
            d2 = f_pred(tp) - 2.0 * p0 + f_pred(tm)
            q[i] = float(torch.linalg.norm(d2)) / eps ** 2 * s ** 2
    return q


def probe_self_curvature(f_pred, theta):
    """Per-coordinate q (see module docstring), evaluated at theta AND at
    one randomly perturbed theta~, taking the max. The invariant that
    separates the blocks is identically-zero (a linear coordinate has zero
    self-curvature at EVERY theta) vs generically-nonzero (a degenerate
    geometry coordinate -- e.g. a neuron whose readout weight is ~0 --
    probes as linear at the special current point but wakes up under a
    random perturbation). 2 * (2m+1) forward passes."""
    g = torch.Generator().manual_seed(0)
    scale = torch.clamp(theta.abs(), min=1.0)
    signs = (torch.randint(0, 2, theta.shape, generator=g,
                           dtype=torch.int64) * 2 - 1).to(theta.dtype)
    theta_p = theta + 0.1 * scale * signs
    return torch.maximum(_self_curvature_at(f_pred, theta),
                         _self_curvature_at(f_pred, theta_p))


def train_slim_autotether(theta0, f_resid, f_jvp, f_vjp, f_pred, n_train,
                          iters, frames_at, get_wbv, eval_rel, y_norm,
                          refresh=REFRESH):
    """The slim core in an ADAPTIVE metric phi = S * theta. S starts at the
    identity; at the first refresh the nonlinearity mask is probed, and at
    any refresh where the carried-residual drift trips, S is ratcheted up
    by RATCHET on masked coordinates (basis/direction/curvature reset).
    All closures (f_resid/f_jvp/f_vjp/f_pred) take RAW theta.

    Trip condition: drift must exceed BOTH 0.1*||r_fresh|| and the gauge's
    own noise floor ~10*eps*||y||. Near the precision floor, r_hat and
    r_fresh differ by frozen-vs-fresh ROUNDING noise (~1e-16 absolute), so
    the relative reading alone saturates at O(1) forever with the geometry
    innocent; a gauge must not act on readings below its own noise."""
    m = theta0.numel()
    S = torch.ones(m, dtype=theta0.dtype)
    phi = theta0.clone()                       # phi = S * theta, S = 1 now
    q_acc = None                               # elementwise-max of probes
    T = 1.0

    def raw(p):
        return p / S

    rhat = f_resid(raw(phi))
    g = f_vjp(raw(phi), (2.0 / n_train) * rhat) / S
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0, "trips": [], "T_final": 1.0,
             "mask_frac": None}

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

    if 0 in frames_at:
        snaps[0] = get_wbv(raw(phi))
    gt = g.clone()
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    exhaust_streak = 0

    for it in range(1, iters + 1):
        if it % refresh == 0:
            r_fresh = f_resid(raw(phi))
            stats["refresh"] += 1
            drift = float(torch.linalg.norm(rhat - r_fresh))
            rnorm = float(torch.linalg.norm(r_fresh))
            stats.setdefault("gauge", []).append(
                [it, drift, rnorm])       # diagnostic: what the gauge saw
            rhat = r_fresh
            noise_floor = 10.0 * 2.22e-16 * y_norm
            tripped = drift > max(TRIP_FRAC * rnorm, noise_floor)
            if q_acc is None or tripped:
                # probe at the first refresh, and RE-probe on every trip:
                # a coordinate whose readout is still ~0 probes as linear
                # (its second derivative truly is zero there) and only
                # turns nonlinear later. q accumulates by elementwise max,
                # so per-coordinate locks, like T, never shrink.
                q = probe_self_curvature(f_pred, raw(phi))
                q_acc = q if q_acc is None else torch.maximum(q_acc, q)
                if "q" not in stats:
                    stats["q"] = q.numpy()
            if tripped:
                # CONTINUOUS tether: every coordinate slowed in proportion
                # to its measured nonlinearity. No mask, no threshold: the
                # readout (q/max q ~ 1e-10, measured) is untouched until
                # T ~ 1e10, while weak benders get proportional locks.
                T *= RATCHET
                stats["trips"].append(it)
                S_new = 1.0 + T * (q_acc / float(q_acc.max()))
                phi = phi * (S_new / S)        # same raw theta, new metric
                S = S_new
                stats["mask_frac"] = float((S > 10.0).double().mean())
                basis = []                     # metric changed: old
                lhat = None                    # conjugacy/curvature void
            g = f_vjp(raw(phi), (2.0 / n_train) * rhat) / S
            gt = project(g)
            ggt = float(gt.dot(gt))
            if stats["trips"] and stats["trips"][-1] == it:
                d = -gt
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break
        jd = f_jvp(raw(phi), d / S)
        dhd = (2.0 / n_train) * float(jd.dot(jd))
        if dhd > 0 and np.isfinite(dhd):
            alpha = -float(g.dot(d)) / dhd
            lam = dhd / dn2
            lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        phi_new = phi + alpha * d
        rhat_new = rhat + alpha * jd
        g_new = f_vjp(raw(phi_new), (2.0 / n_train) * rhat_new) / S
        if not np.isfinite(float(torch.linalg.norm(g_new))):
            stats["revert"] += 1               # sole safeguard: NaN reject
            d = -gt
            if it in frames_at:
                snaps[it] = get_wbv(raw(phi))
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
            snaps[it] = get_wbv(raw(phi))
    stats["T_final"] = T
    final_wbv = get_wbv(raw(phi))
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, raw(phi), stats


RHO_BAD, RHO_GOOD = 0.5, 0.1   # gain-ratio thresholds (LM-standard shape)
MU_FACTOR = 10.0               # mu multiplier / divisor per adaptation
MU_SEED = 1e-6                 # first nonzero mu = MU_SEED * lhat


def train_slim_damped(theta0, f_resid, f_jvp, f_vjp, n_train, iters,
                      frames_at, get_wbv, eval_rel, refresh=REFRESH):
    """v5 core: the slim optimizer + per-step Levenberg damping, untethered
    (T = 1 on every coordinate, no probes, no ratchet).

    Step: alpha = -g.d / (c + mu ||d||^2), c = (2/n)||Jd||^2. mu is adapted
    by a gradient-space gain ratio -- never a loss value (loss comparisons
    go blind at rel ~1e-12):

        predicted  g_hat = g + alpha (2/n) J^T (J d)     [one extra VJP]
        rho = ||g_actual - g_hat|| / ||g_hat - g||
        rho > RHO_BAD  -> mu = max(MU_FACTOR mu, MU_SEED lhat)
        rho < RHO_GOOD -> mu = mu / MU_FACTOR (0 below 1e-30 lhat)

    mu = 0 reduces exactly to train_residual_cg_slim. Geometry protection
    comes from steps never leaving the linearization's validity region,
    acting per step instead of per refresh (the v4 lag problem)."""
    theta = theta0.clone()
    rhat = f_resid(theta)

    def grad_of(th, r):
        return f_vjp(th, (2.0 / n_train) * r)

    g = grad_of(theta, rhat)
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0, "damped_iters": 0, "mu_max": 0.0,
             "mu_final": 0.0}

    def project(vec):
        if not basis:
            return vec.clone()
        B = torch.stack(basis)
        out = vec - B.t().mv(B.mv(vec))
        return out - B.t().mv(B.mv(out))

    def append_to_basis(vec):
        v = project(vec)
        n2 = float(v.dot(v))
        if n2 > 1e-300 and len(basis) < theta.numel():
            basis.append(v / float(np.sqrt(n2)))

    if 0 in frames_at:
        snaps[0] = get_wbv(theta)
    gt = g.clone()
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    mu = 0.0
    exhaust_streak = 0

    for it in range(1, iters + 1):
        if it % refresh == 0:
            rhat = f_resid(theta)
            stats["refresh"] += 1
            g = grad_of(theta, rhat)
            gt = project(g)
            ggt = float(gt.dot(gt))
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break
        u = f_jvp(theta, d)
        c = (2.0 / n_train) * float(u.dot(u))
        denom = c + mu * dn2
        if denom > 0 and np.isfinite(denom):
            alpha = -float(g.dot(d)) / denom
            if c > 0:
                lam = c / dn2
                lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        # predicted new gradient under the Gauss-Newton linear model
        g_hat = g + alpha * f_vjp(theta, (2.0 / n_train) * u)
        theta_new = theta + alpha * d
        rhat_new = rhat + alpha * u
        g_new = grad_of(theta_new, rhat_new)
        if not np.isfinite(float(torch.linalg.norm(g_new))):
            stats["revert"] += 1               # NaN reject
            d = -gt
            mu = max(mu * MU_FACTOR, MU_SEED * (lhat or 1.0))
            if it in frames_at:
                snaps[it] = get_wbv(theta)
            continue
        pred_change = float(torch.linalg.norm(g_hat - g))
        gn_scale = max(float(torch.linalg.norm(g)),
                       float(torch.linalg.norm(g_new)))
        if pred_change < 1e-14 * gn_scale:
            # gauge noise guard: a predicted change below the rounding
            # noise of a gradient evaluation is unmeasurable -- rho would
            # read noise/tiny = "bad" forever and ratchet mu to infinity
            # (the same failure class as the v4 drift gauge). Decay
            # optimistically; if steps are genuinely nonlinear, rho reads
            # bad again as soon as it is measurable.
            if mu > 0:
                mu = mu / MU_FACTOR
                if lhat is not None and mu < 1e-30 * lhat:
                    mu = 0.0
        else:
            rho = float(torch.linalg.norm(g_new - g_hat)) / pred_change
            if rho > RHO_BAD:
                mu = max(mu * MU_FACTOR, MU_SEED * (lhat or 1.0))
            elif rho < RHO_GOOD and mu > 0:
                mu = mu / MU_FACTOR
                if lhat is not None and mu < 1e-30 * lhat:
                    mu = 0.0
        if mu > 0:
            stats["damped_iters"] += 1
            stats["mu_max"] = max(stats["mu_max"], mu)
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
        theta, rhat, g, gt = theta_new, rhat_new, g_new, gt_new
        ggt = float(gt.dot(gt))
        losses.append(float(rhat.dot(rhat)) / n_train)
        rels.append(eval_rel(theta))
        if it in frames_at:
            snaps[it] = get_wbv(theta)
    stats["mu_final"] = mu
    final_wbv = get_wbv(theta)
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, theta, stats


# --------------------------------- one run ---------------------------------

def tether_case(job):
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    fn_name, N = job["fn"], job["N"]
    f_vec = run.make_fn(run.TARGETS[fn_name])
    centers, gamma, _halo = run.geometry(N, run.LAM)
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
        (gg,) = torch.autograd.grad(f_pred(thr), thr, grad_outputs=seed)
        return gg.detach()

    def eval_rel(th):
        with torch.no_grad():
            r = model(xe, th) - ye
            return float(torch.linalg.norm(r)) / ye_norm

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

    t0 = time.time()
    losses, rels, snaps, theta_fin, stats = train_slim_damped(
        theta0, f_resid, f_jvp, f_vjp, run.N_TRAIN, job["iters"],
        set(), get_wbv, eval_rel)

    q = stats.pop("q", None)
    out = {"fn": fn_name, "N": N, "floor": floor, "rels": rels,
           "best_rel": float(np.nanmin(rels)) if rels else float("nan"),
           "stats": stats, "wall_s": round(time.time() - t0, 1)}
    if q is not None and job.get("save_q"):
        qdir = TETHER_DIR / "probes"
        qdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(qdir / f"{fn_name}--N{N}.npz", q=q, W=W)
    return out


# --------------------------------- figures ---------------------------------

def render_probe_hist(path):
    """Receipt for the detector: per-block distributions of the probe score
    q at the first refresh. The (v, c0) block must sit orders below (w, b)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted((TETHER_DIR / "probes").glob("*.npz"))
    if not files:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, path_ in zip(axes.flat, files):
        z = np.load(path_)
        q, W = z["q"], int(z["W"])
        qpos = np.maximum(q, 1e-300)
        blocks = [("w", qpos[:W]), ("b", qpos[W:2 * W]),
                  ("v", qpos[2 * W:3 * W]), ("c0", qpos[3 * W:])]
        lo = max(float(qpos.max()) * 1e-24, 1e-300)
        bins = np.logspace(np.log10(lo), np.log10(float(qpos.max()) * 10), 48)
        for lb, arr in blocks:
            ax.hist(np.clip(arr, bins[0], None), bins=bins, alpha=0.55,
                    label=lb)
        ax.set_xscale("log")
        ax.set_title(path_.stem.replace("--", " "), fontsize=10)
        ax.set_xlabel("self-curvature score q")
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=False)
    fig.suptitle("auto-tether nonlinearity probe at first refresh; "
                 "blocks labeled for READING only -- the optimizer never "
                 "sees them (v4: continuous S = 1 + T q/max q)", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_scaling_vs_fixed(rows, path):
    """2x3 per-target: best eval rel L2 vs N -- auto-tether vs iteration_5
    fixed T=1e6, floors dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    it5 = {}
    it5_path = run.OUT_DIR / "expD08_rows.json"
    if it5_path.exists():
        for r in json.loads(it5_path.read_text()):
            it5[(r["fn"], r["N"])] = r["best_rel"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, fn in zip(axes.flat, run.TARGETS):
        rs = sorted([r for r in rows if r["fn"] == fn], key=lambda r: r["N"])
        ns = [r["N"] for r in rs]
        ax.loglog(ns, [r["best_rel"] for r in rs], "-o", color="tab:blue",
                  ms=5, label=f"{VERSION}: per-step damped slim, untethered")
        f5 = [it5.get((fn, n)) for n in ns]
        if all(v is not None for v in f5):
            ax.loglog(ns, f5, "-s", color="tab:gray", ms=4, lw=1.0,
                      alpha=0.85, label="fixed tether 1e6 (iteration_5)")
        ax.loglog(ns, [r["floor"] for r in rs], "--", color="k", lw=1.5,
                  label="lstsq floor (frozen init geometry)")
        ax.set_ylim(1e-16, 1.0)
        ax.set_title(fn, fontsize=11)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"best eval rel $L_2$")
        ax.set_xticks(run.WIDTHS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle("expD08 auto-tether vs fixed tether -- slim core, "
                 "3000 iters, drift-ratcheted metric on probed-nonlinear "
                 "coordinates", y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    widths = run.WIDTHS if not args.smoke else [32, 128]
    fns = list(run.TARGETS) if not args.smoke else ["sine", "runge"]
    iters = ITERS if not args.smoke else 450
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    jobs = [{"fn": f, "N": n, "iters": iters, "threads": threads,
             "save_q": n == 256}
            for f in fns for n in widths]
    print(f"expD08 tether {VERSION} | {len(jobs)} runs | iters={iters} | "
          f"per-step gradient-space damping (rho {RHO_GOOD:g}/{RHO_BAD:g}, "
          f"x{MU_FACTOR:g}), untethered")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(tether_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            s = r["stats"]
            print(f"  [{i}/{len(jobs)}] {r['fn']:13s} N={r['N']:<4d} "
                  f"best_rel={r['best_rel']:.3e} floor={r['floor']:.1e} "
                  f"damped={s.get('damped_iters', 0)} "
                  f"mu_max={s.get('mu_max', 0):.1e} "
                  f"mu_fin={s.get('mu_final', 0):.1e} "
                  f"exh={s['exhaust']} rst={s['full_reset']} "
                  f"rev={s['revert']} ({r['wall_s']}s)", flush=True)
    print(f"training done in {time.time()-t0:.1f}s")

    TETHER_DIR.mkdir(parents=True, exist_ok=True)
    (TETHER_DIR / "tether_rows.json").write_text(json.dumps(
        [{k: r[k] for k in ("fn", "N", "floor", "best_rel", "stats")}
         for r in rows]))
    render_scaling_vs_fixed(rows, TETHER_DIR / "autotether_vs_fixed_2x3.png")
    render_probe_hist(TETHER_DIR / "probe_histograms.png")
    print(f"done; outputs under {TETHER_DIR}")


if __name__ == "__main__":
    main()
