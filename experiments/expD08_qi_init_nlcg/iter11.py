"""Iteration 11 -- the two-regime certify-and-solve optimizer.

Design (spec: results/.../iteration_11/fable_thoughts.md, approved by Sam
2026-07-27): two regimes coexist, split per-parameter by a measured
certificate, not by a schedule.

  base regime     STOCK Adam on every parameter, every step (fresh
                  residual + gradient, fused into one eval). Membership
                  decides where the periodic exact solve applies -- NOT
                  what Adam is allowed to touch (masking the exploit set
                  out of Adam froze the readout at a stale optimum for a
                  whole refresh window: measured 4x cost on dl_test).

  exploit regime  Parameters the nonlinearity probe certifies as
                  effectively linear enter the exploit set E (size B,
                  capped at B <= sqrt(C_MEM * m): a BxB-class bank is
                  then O(m) total -- Sam's sqrt(k) rule). At each refresh
                  their subproblem min_d || J_E d + r || is solved.
                  Two ablation arms, plotted over each other everywhere:

                  varpro (blue)  direct: J_E by B JVP columns, columns
                       below 1e-8 of the max column norm dropped (data-
                       blind parameters must not be solved), truncated-
                       SVD min-norm correction at rcond 1e-15.
                  qn (red)       iterative: dense inverse-BFGS bank
                       (BxB) on the exploit subproblem, exact Gauss-
                       Newton step lengths, uncontaminated secant pairs
                       y = A s from one extra VJP. Runs the CG-lineage
                       carried-residual loop.

Certificate: per-parameter relative nonlinearity qhat from the second-
difference probe (at theta AND one perturbed point, so a zero readout
cannot fake linear geometry), with the iteration-8 ABSOLUTE noise guard.
Hysteresis: enter below Q_ENTER, exit above Q_EXIT; demotion returns the
parameter to Adam-only (reactivation is built in, nothing is ever
frozen). Re-probes fire at refresh boundaries when the drift gauge reads
motion above its own noise floor, with exponential backoff while
membership is stable. On an exactly linear problem everything certifies
and the optimizer IS a least-squares solver; with nothing linear it
degrades to stock Adam. No tether, no metric, no trips, no ratchet, no
control decision that reads a loss value, NaN reject only.

STANDARD OUTPUT SUITE (the structure every iteration follows from here):

  1D_test/   scaling_2x3.png            6 targets, min eval rel L2 vs N
             loss_curves_2x2_N256.png   4 targets, train+eval vs iteration
             gifs_varpro/, gifs_qn/     params_N{128,256,512}.gif (4x3
                                        parameter movies) + dial_N256.gif
                                        (the certificate that selects E)
  batching/  softwin_batchsize_vs_error.png, strongwin_interpolation_bars.png
  depth2/    depth2.png                 light bench, 2 targets x 2 arms

Run:  uv run python experiments/expD08_qi_init_nlcg/iter11.py --grid
      ... --batching | --depth2 | --render
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
OUT = run.OUT_DIR.parent / "iteration_11"
OUT_1D = OUT / "1D_test"
OUT_BATCH = OUT / "batching"
OUT_D2 = OUT / "depth2"

# ------------------------------ constants ------------------------------

C_MEM = 1024         # sqrt(k) budget: B_cap = sqrt(C_MEM * m), bank = B^2
                     # <= C_MEM * m, i.e. O(m) with an explicit constant.
                     #
                     # HONEST NOTE, because this constant is doing real
                     # work: the STRICT form of Sam's rule is B ~ sqrt(k)
                     # with a k-sized bank, i.e. C_MEM = 1. This toy
                     # family cannot satisfy that at any width -- a
                     # single-hidden-layer net's exactly-linear block IS
                     # the readout, which is Theta(k) (m/3 here), so at
                     # N=512 a strict budget would allow B = sqrt(2764) =
                     # 52 against a 922-coordinate readout. Realistic
                     # architectures are the opposite case (a 1e9-param
                     # model's linear head is 1e3-1e4 << sqrt(k) = 3e4),
                     # so the rule binds only on the degenerate toy.
                     # C_MEM = 1024 keeps the readout intact through
                     # N=512 (b_cap = 1682 vs 922). What happens when the
                     # cap DOES truncate a coupled block is measured
                     # separately (1D_test/rows_cap256_binding.json:
                     # b_cap 841 < readout 922 costs 12 orders on 3 of 6
                     # targets) -- truncation is only safe among
                     # INDEPENDENT candidates.
Q_ENTER = 1e-8       # certify: qhat below this (probe separation measured
                     # at 9-12 orders; readout reads ~1.6e-10 of max)
Q_EXIT = 1e-7        # demote: qhat above this (hysteresis deadband)
RCOND = 1e-15        # truncation for the varpro solve (repo standard)
COL_FLOOR = 1e-8     # observability: drop J_E columns below this x max
ARMS = ["varpro", "qn"]
ARM_COLOR = {"varpro": "tab:blue", "qn": "tab:red", "adam": "0.55"}
ARM_LABEL = {"varpro": "varpro -- direct solve on the exploit set",
             "qn": "qn -- dense BFGS bank on the exploit set"}


# ---------------------------- the certificate ----------------------------

def update_membership(qhat, members, b_cap, info=None):
    """Hysteresis + budget, pure function (unit-tested).

    Enter when qhat < Q_ENTER; stay while qhat < Q_EXIT; leave otherwise.
    If more than b_cap parameters qualify, degrade by SELECTION (never by
    chunk rotation -- block-coordinate descent across coupled chunks is
    first-order across blocks and stalls by the measured Phase-0 law).

    What to select on is the subtle part, and getting it wrong was a
    measured failure: ranking by linearity (smallest qhat) is exactly
    BACKWARDS, because a parameter reads as perfectly linear precisely
    when the data cannot see it. At N=512 the budget binds and that rule
    handed 628 of 841 slots to saturated (data-blind) halo geometry while
    starving the readout to 213 of 922 coordinates -- the solve then
    diverged to ~1e10 on all six targets. `info` is the data energy each
    coordinate carries (Adam's second moment, free from state we already
    keep; the gradient magnitude at the first call). Certification says a
    parameter is LINEAR; info says it is INFORMED; the budget must go to
    parameters that are both. Falls back to the linearity ranking only
    when no info is supplied."""
    # ADMISSION needs both halves of the certificate: linear AND informed.
    # Certifying on linearity alone admits parameters the data cannot see
    # at all -- and the separation is not a judgement call: at N=512 all
    # 642 certified halo-geometry coordinates carry gradient EXACTLY 0.0
    # (saturated tanh' underflows in fp64) against a readout median of
    # 1.7e-2, so "informed" is simply info > 0, with no threshold to tune.
    # Admitting the blind ones cost ~1 order on the sine family (they
    # contribute near-null columns that the solve then has to regularize
    # around). Demotion uses the same test, so a coordinate that
    # saturates later leaves on its own.
    informed = (info > 0) if info is not None else torch.ones_like(members)
    stay = members & (qhat < Q_EXIT) & informed
    enter = (~members) & (qhat < Q_ENTER) & informed
    new = stay | enter
    n_new = int(new.sum())
    if n_new > b_cap:
        idx = torch.nonzero(new, as_tuple=False).flatten()
        score = -qhat[idx] if info is None else info[idx]
        keep = idx[torch.argsort(score, descending=True)[:b_cap]]
        new = torch.zeros_like(new)
        new[keep] = True
    return new


def auto_chunk(n_out, width_hint, budget=4e7):
    """Kept as the call-site knob for probe cost; see probe_dense."""
    return int(max(8, min(256, budget / max(n_out * width_hint, 1))))


def probe_dense(f_pred, theta, chunk=None):
    """Dense per-parameter nonlinearity probe: run.probe_self_curvature
    (second differences at theta AND one perturbed point, so a zero
    readout cannot fake linear geometry). Per-parameter resolution is a
    diagnostic -- it is what the dial gif shows; the deployable form is
    the per-tensor sampled probe.

    A vmap-vectorized twin was built and measured: it reproduces this to
    2e-17 but runs 0.8x the speed on CPU (vmap overhead dominates at
    these shapes, and the (chunk x n x W) activation forces chunk ~ 18 at
    N=512 anyway), so the reference loop stands."""
    return run.probe_self_curvature(f_pred, theta)


# --------------------------- the exploit solvers ---------------------------

def varpro_delta(theta, idx, rhat, f_jvp, rcond=RCOND):
    """Direct solve of  min_d || J_E d + rhat ||  by truncated SVD.

    Toy (full-batch) form: J_E materialized column-by-column via JVPs.
    Deployable form: stream row-blocks through an incremental QR (R
    factor BxB) using batch-restricted JVPs -- identical arithmetic,
    O(B^2) persistent state. Returns the full-size correction (min-norm
    in the subproblem, zero off E)."""
    m = theta.numel()
    cols = []
    for i in idx.tolist():
        e = torch.zeros(m, dtype=theta.dtype)
        e[i] = 1.0
        cols.append(f_jvp(theta, e).numpy())
    J = np.stack(cols, axis=1)
    # Observability guard (measured failure, bring-up 2): data-blind
    # members (e.g. saturated halo neurons -- zero curvature AND ~zero
    # Jacobian column) poison the min-norm solve: eps-level V-components
    # of numerically-kept tail modes are amplified by 1/sigma into O(10)
    # corrections along directions the linearization cannot certify
    # (delta_geometry ~ 29 mangled runge@64 to 2e+02). The solver must
    # not solve directions it cannot observe: drop columns below
    # COL_FLOOR x the largest column norm (measured gap: live columns
    # O(1), blind columns <= 1e-14). Dropped members simply receive no
    # solve -- neither regime can learn a data-blind parameter.
    col_norms = np.linalg.norm(J, axis=0)
    cmax = float(col_norms.max()) if col_norms.size else 0.0
    if cmax <= 0.0:
        return torch.zeros(m, dtype=theta.dtype)
    live = col_norms > COL_FLOOR * cmax
    J = J[:, live]
    idx_live = idx[torch.from_numpy(np.nonzero(live)[0])]
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return torch.zeros(m, dtype=theta.dtype)
    keep = s > rcond * s[0]
    coef = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0) \
        * (U.T @ (-rhat.numpy()))
    d_sub = Vt.T @ coef
    delta = torch.zeros(m, dtype=theta.dtype)
    delta[idx_live] = torch.from_numpy(d_sub)
    return delta


def bfgs_update(H, s, y):
    """Standard inverse-BFGS update, skipped when the pair is degenerate.
    Pairs here are exact Gauss-Newton products y = A s (never gradient
    differences), so sy > 0 up to rounding on any least-squares problem."""
    sy = float(s.dot(y))
    if not np.isfinite(sy) or sy <= 1e-300:
        return H
    rho = 1.0 / sy
    Hy = H @ y
    yHy = float(y.dot(Hy))
    ssT = torch.outer(s, s)
    H -= rho * (torch.outer(Hy, s) + torch.outer(s, Hy))
    H += rho * rho * yHy * ssT + rho * ssT
    return H


# ------------------------------- the core -------------------------------

def train_iter11(theta0, f_resid, f_jvp, f_vjp, f_pred, n_train, iters,
                 frames_at, get_wbv, eval_rel, y_norm, refresh=run.REFRESH,
                 arm="varpro", lr=1e-3, betas=(0.9, 0.999), eps_adam=1e-8,
                 snap_S=False, probe_fn=None, f_residgrad=None,
                 batch_iter=None, c_mem=C_MEM):
    """The iteration-11 loop. Returns (losses, rels, snaps, theta, stats).

    varpro arm: the base regime is STOCK Adam -- fresh residual and
    gradient every step (pass f_residgrad to fuse them into one charged
    eval), on ALL parameters. No carried residual: that is CG-lineage
    machinery, and feeding Adam a linearized 200-step-stale gradient was
    measured to cost 3-7x on dl_test's nonconvex transient. The direct
    solve is noise-safe on a fresh residual (it is lstsq with a 1e-16-
    noisy rhs -- Lesson 2 governs iterative recurrences, not one-shot
    solves). The drift gauge is the function-space motion of the
    residual over a refresh window.

    qn arm: the CG-lineage carried-residual loop with the dense bank.

    batch_iter (optional): an iterator of row-index tensors. When given,
    every closure is called with an idx argument and the Adam step, the
    gauge and the solve all see batch rows only -- the per-batch regime.
    The probe keeps its own fixed gauge rows (f_pred is not indexed).

    State: theta, r_ref (varpro) / rhat (qn), Adam (m, v), membership
    mask + qhat, and the qn bank H (BxB)."""
    b1, b2 = betas
    m = theta0.numel()
    theta = theta0.clone()
    noise_floor = 10.0 * 2.22e-16 * y_norm
    b_cap = max(1, int(np.sqrt(c_mem * m)))
    fresh_mode = arm == "varpro"
    batched = batch_iter is not None
    if batched and not fresh_mode:
        raise ValueError(
            "the qn arm carries a residual bound to specific rows and is "
            "undefined under resampled minibatches (see main_batching)")

    members = torch.zeros(m, dtype=torch.bool)
    qhat = torch.ones(m, dtype=theta.dtype)
    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    t_adam = 0
    H = None
    idx_E = None
    backoff = 1
    windows_since_probe = 0

    losses, rels, snaps = [], [], {}
    stats = {"probes": 0, "solves": 0, "revert": 0, "refresh": 0,
             "member_changes": [], "B_hist": [], "B_final": 0,
             "b_cap": b_cap, "arm": arm}

    if probe_fn is None:
        def probe_fn(th):
            return probe_dense(f_pred, th)

    # index plumbing: one code path, idx passed only in batched mode
    def R(th, rows):
        return f_resid(th, rows) if batched else f_resid(th)

    def JV(th, d, rows):
        return f_jvp(th, d, rows) if batched else f_jvp(th, d)

    def RG(th, rows, n_eff):
        if f_residgrad is not None:
            return f_residgrad(th, rows) if batched else f_residgrad(th)
        r = R(th, rows)
        gg = (f_vjp(th, (2.0 / n_eff) * r, rows) if batched
              else f_vjp(th, (2.0 / n_eff) * r))
        return r, gg

    def certify(it):
        """Probe + membership update; returns True if membership changed."""
        nonlocal qhat, members, H, idx_E
        q = probe_fn(theta)
        stats["probes"] += 1
        # Absolute noise guard (permanent lesson, iteration 8): the
        # probe's own rounding is q_noise ~ 4 eps ||p|| / PROBE_EPS^2;
        # relative normalization alone promotes pure noise to qhat = 1
        # on an exactly-linear problem. Entries below the guard are
        # measured zero (= measured linear).
        with torch.no_grad():
            p_scale = max(float(torch.linalg.norm(f_pred(theta))), y_norm)
        q_noise = 4.0 * 2.22e-16 / run.PROBE_EPS ** 2 * p_scale
        q = torch.where(q < 100.0 * q_noise, torch.zeros_like(q), q)
        qm = float(q.max())
        qhat = q / qm if qm > 0 else torch.zeros_like(q)
        new = update_membership(qhat, members, b_cap, info=info_vec)
        changed = not torch.equal(new, members)
        if changed:
            stats["member_changes"].append(it)
            H = None                      # bank not transportable
            members = new
            idx_E = torch.nonzero(members, as_tuple=False).flatten()
        stats["B_hist"].append((it, int(members.sum())))
        return changed

    def snap(it):
        s_ = get_wbv(theta)
        if snap_S:
            s_["Q"] = qhat.numpy().copy()
            s_["E"] = members.numpy().astype(np.float64)
        snaps[it] = s_

    def gauge(it, drift):
        """Refresh-boundary re-probe decision (backoff on stability)."""
        nonlocal backoff, windows_since_probe
        stats["refresh"] += 1
        windows_since_probe += 1
        if drift > noise_floor and windows_since_probe >= backoff:
            changed = certify(it)
            windows_since_probe = 0
            backoff = 1 if changed else min(backoff * 2, 8)

    rows0 = None
    n_eff = n_train
    rhat = R(theta, rows0)
    r_ref = rhat.clone()
    # data energy per coordinate, for the budget selection. One VJP at
    # startup (Adam's second moment is still zero here); from the first
    # step onward the running second moment supersedes it.
    info_vec = (f_vjp(theta, (2.0 / n_train) * rhat, rows0) if batched
                else f_vjp(theta, (2.0 / n_train) * rhat)).abs()
    certify(0)
    if 0 in frames_at:
        snap(0)

    for it in range(1, iters + 1):
        rows = next(batch_iter) if batched else None
        if batched:
            n_eff = int(rows.numel())

        if fresh_mode:
            # STOCK Adam base: fresh residual + gradient every step.
            rhat, g = RG(theta, rows, n_eff)
            if it % refresh == 0 and not batched:
                gauge(it, float(torch.linalg.norm(rhat - r_ref)))
                r_ref = rhat.clone()
            elif it % refresh == 0:
                # batched: the gauge reads the fixed probe rows, so that
                # sampling noise is never read as geometry motion
                r_g = f_resid(theta)
                gauge(it, float(torch.linalg.norm(r_g - r_ref)))
                r_ref = r_g
        else:
            if it % refresh == 0:
                r_fresh = R(theta, rows)
                drift = float(torch.linalg.norm(rhat - r_fresh))
                rhat = r_fresh
                gauge(it, drift)
            g = (f_vjp(theta, (2.0 / n_eff) * rhat, rows) if batched
                 else f_vjp(theta, (2.0 / n_eff) * rhat))

        # Adam moments and step on EVERY parameter (see module docstring).
        t_adam += 1
        ma = b1 * ma + (1.0 - b1) * g
        va = b2 * va + (1.0 - b2) * g * g
        mh = ma / (1.0 - b1 ** t_adam)
        vh = va / (1.0 - b2 ** t_adam)
        info_vec = vh                      # data energy for the next probe
        step_c = -lr * mh / (vh.sqrt() + eps_adam)
        if not fresh_mode:
            step_c = step_c * (~members)   # qn arm owns its members' step

        alpha, d_full, jdE = 0.0, None, None
        if arm == "qn" and bool(members.any()):
            gE = g[idx_E]
            B = int(idx_E.numel())
            if H is None:
                H = torch.eye(B, dtype=theta.dtype)
            dE = -(H @ gE)
            d_full = torch.zeros(m, dtype=theta.dtype)
            d_full[idx_E] = dE
            jdE = JV(theta, d_full, rows)
            c = (2.0 / n_eff) * float(jdE.dot(jdE))
            if c > 0 and np.isfinite(c):
                alpha = -float(g.dot(d_full)) / c
                # exact GN secant pair: y = A s from one VJP, s = alpha*d
                y_full = (f_vjp(theta, (2.0 / n_eff) * jdE, rows) if batched
                          else f_vjp(theta, (2.0 / n_eff) * jdE)) * alpha
                H = bfgs_update(H, alpha * dE, y_full[idx_E])
            else:
                alpha = 0.0

        if fresh_mode:
            th_new = theta + step_c
            rh_new = rhat                  # refreshed next iteration
        else:
            jd_c = JV(theta, step_c, rows)
            dtheta = step_c if alpha == 0.0 else step_c + alpha * d_full
            rh_new = (rhat + jd_c if alpha == 0.0
                      else rhat + jd_c + alpha * jdE)
            th_new = theta + dtheta
        if not (np.isfinite(float(torch.linalg.norm(th_new)))
                and np.isfinite(float(torch.linalg.norm(rh_new)))):
            stats["revert"] += 1          # sole safeguard: NaN reject
            ma = ma * 0.0
            va = va * 0.0                 # poisoned moments go too
            if it in frames_at:
                snap(it)
            continue
        theta, rhat = th_new, rh_new

        # The exploit solve runs AFTER the base step and BEFORE the
        # record, so (a) it addresses the state the base regime just
        # produced and (b) the recorded trajectory contains the solved
        # point (recording pre-solve hid the floor behind one step of
        # base-regime wander -- measured in bring-up 1). The residual is
        # recomputed fresh at the post-step point.
        if arm == "varpro" and bool(members.any()) \
                and (it == 1 or it % refresh == 0):
            r_s = R(theta, rows)
            delta = varpro_delta(theta, idx_E, r_s,
                                 (lambda th, d: JV(th, d, rows)))
            jd = JV(theta, delta, rows)
            th_try, rh_try = theta + delta, r_s + jd
            if np.isfinite(float(torch.linalg.norm(th_try))) and \
                    np.isfinite(float(torch.linalg.norm(rh_try))):
                theta, rhat = th_try, rh_try
                stats["solves"] += 1
            else:
                stats["revert"] += 1

        losses.append(float(rhat.dot(rhat)) / max(n_eff, 1))
        rels.append(eval_rel(theta))
        if it in frames_at:
            snap(it)

    stats["B_final"] = int(members.sum())
    final = get_wbv(theta)
    if snap_S:
        final["Q"] = qhat.numpy().copy()
        final["E"] = members.numpy().astype(np.float64)
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final
    return losses, rels, snaps, theta, stats


# ------------------------------ case plumbing ------------------------------

def build_case(fn_name, N):
    """Standard seed9 protocol: QI geometry, zero readout, all trainable.
    Mirrors run.run_case's closures without the tether machinery."""
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
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] \
            + th[-1]

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
        (g,) = torch.autograd.grad(pred, thr, grad_outputs=seed)
        return g.detach()

    def f_residgrad(th):
        """Fused fresh residual + MSE gradient (one autograd graph)."""
        thr = th.detach().requires_grad_(True)
        pred = model(xt, thr)
        r = (pred - yt).detach()
        (g,) = torch.autograd.grad(
            pred, thr, grad_outputs=(2.0 / run.N_TRAIN) * r)
        return r, g.detach()

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
    keep = s > RCOND * s[0]
    sol = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                  * (U.T @ f_vec(x_tr)))
    A_ev = np.hstack([run.build_phi(x_ev, gamma, centers, "tanh"),
                      np.ones((x_ev.size, 1))])
    floor = float(np.linalg.norm(A_ev @ sol - f_vec(x_ev))
                  / np.linalg.norm(f_vec(x_ev)))

    chunk = auto_chunk(run.N_TRAIN, W)

    def probe_fn(th):
        return probe_dense(f_pred, th, chunk)

    return {"theta0": theta0, "W": W, "centers": centers, "floor": floor,
            "f_pred": f_pred, "f_resid": f_resid, "f_jvp": f_jvp,
            "f_vjp": f_vjp, "f_residgrad": f_residgrad,
            "eval_rel": eval_rel, "get_wbv": get_wbv, "probe_fn": probe_fn,
            "y_norm": float(torch.linalg.norm(yt))}


def run_case11(job):
    torch.set_num_threads(job.get("threads", 1))
    torch.set_default_dtype(torch.float64)
    env = build_case(job["fn"], job["N"])
    frames_at = (set(expd06.frame_steps(job["iters"]))
                 if job.get("save_snaps", True) else {0})
    t0 = time.time()
    losses, rels, snaps, _, stats = train_iter11(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], run.N_TRAIN, job["iters"], frames_at,
        env["get_wbv"], env["eval_rel"], env["y_norm"],
        refresh=job.get("refresh", run.REFRESH), arm=job["arm"],
        lr=job.get("lr", 1e-3), snap_S=job.get("snap_S", True),
        f_residgrad=env["f_residgrad"], probe_fn=env["probe_fn"],
        c_mem=job.get("c_mem", C_MEM))
    out = {"fn": job["fn"], "N": job["N"], "arm": job["arm"],
           "seed": run.SEED_NAME, "floor": env["floor"],
           "losses": losses, "rels": rels,
           "best_rel": float(np.min(rels)) if rels else float("nan"),
           "stats": stats, "wall_s": round(time.time() - t0, 1)}
    if job.get("save_snaps", True):
        snap_dir = OUT_1D / "snaps"
        snap_dir.mkdir(parents=True, exist_ok=True)
        ssteps = sorted(snaps)
        extra = {}
        if all("Q" in snaps[s] for s in ssteps):
            extra["Q"] = np.stack([snaps[s]["Q"] for s in ssteps])
            extra["E"] = np.stack([snaps[s]["E"] for s in ssteps])
        np.savez_compressed(
            snap_dir / f"{job['fn']}--N{job['N']}--{job['arm']}.npz",
            steps=np.array(ssteps, dtype=np.int64),
            w=np.stack([snaps[s]["w"] for s in ssteps]),
            b=np.stack([snaps[s]["b"] for s in ssteps]),
            v=np.stack([snaps[s]["v"] for s in ssteps]),
            losses=np.array(losses), rels=np.array(rels),
            floor=env["floor"], centers=env["centers"],
            stats=json.dumps(stats), **extra)
    return out


def load_rows_1d():
    """Rebuild rows (incl. per-frame snaps) from the persisted npz."""
    rows = []
    for p in sorted((OUT_1D / "snaps").glob("*.npz")):
        fn, rest = p.stem.split("--N")
        n_s, arm = rest.split("--")
        z = np.load(p)
        r = {"fn": fn, "N": int(n_s), "arm": arm, "seed": run.SEED_NAME,
             "floor": float(z["floor"]), "losses": z["losses"].tolist(),
             "rels": z["rels"].tolist(),
             "best_rel": float(np.min(z["rels"])),
             "stats": json.loads(str(z["stats"])),
             "snaps": {str(int(s)): {"w": z["w"][i].tolist(),
                                     "b": z["b"][i].tolist(),
                                     "v": z["v"][i].tolist()}
                       for i, s in enumerate(z["steps"])}}
        if "Q" in z:
            r["steps"] = z["steps"]
            r["Q"], r["E"] = z["Q"], z["E"]
        rows.append(r)
    return rows


# ------------------------------- 1D figures -------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_scaling(rows, path, iters=None):
    """2x3, one panel per target: best eval rel L2 vs width, both arms
    (blue = varpro, red = qn) against the frozen-init lstsq floor."""
    plt = _mpl()
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    iters = run.ITERS if iters is None else iters
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, fn in zip(axes.flat, run.TARGETS):
        for arm in ARMS:
            rs = sorted([r for r in rows if r["fn"] == fn
                         and r["arm"] == arm], key=lambda r: r["N"])
            if not rs:
                continue
            ax.loglog([r["N"] for r in rs], [r["best_rel"] for r in rs],
                      "-o", color=ARM_COLOR[arm], ms=5, lw=1.6,
                      label=ARM_LABEL[arm])
        rs = sorted([r for r in rows if r["fn"] == fn],
                    key=lambda r: r["N"])
        seen, ns, fl = set(), [], []
        for r in rs:
            if r["N"] not in seen:
                seen.add(r["N"])
                ns.append(r["N"])
                fl.append(r["floor"])
        if ns:
            ax.loglog(ns, fl, "--", color="k", lw=1.5,
                      label="lstsq floor (frozen init geometry)")
        ax.set_xscale("log")               # explicit: an empty panel must
        ax.set_yscale("log")               # still carry the fixed axes
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
    fig.suptitle("iteration 11 scaling -- best eval relative $L_2$ vs width "
                 f"(QI geometry + zero readout, all params free, {iters} "
                 "iterations)", y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def render_loss_2x2(rows, N, path, iters=None):
    """2x2 over the 4 gif targets: train rel L2 (solid) and eval rel L2
    (dashed) vs iteration, both arms, floor dotted. ONE unit everywhere:
    train MSE is converted by sqrt(MSE / mean y^2)."""
    plt = _mpl()
    from matplotlib.lines import Line2D
    iters = run.ITERS if iters is None else iters
    TOP = 2.0
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    for ax, fn in zip(axes.flat, run.TARGETS3):
        y2 = run._y2mean(fn)
        for zi, arm in enumerate(["qn", "varpro"]):   # blue drawn on top
            rs = [r for r in rows if r["fn"] == fn and r["N"] == N
                  and r["arm"] == arm]
            if not rs:
                continue
            r = rs[0]
            tr = np.sqrt(np.array(r["losses"]) / y2)
            ev = np.array(r["rels"])
            for y, ls in ((tr, "-"), (ev, "--")):
                # peg divergence at the axis top instead of letting the
                # curve silently leave the frame (an unexplained gap
                # reads as "the run stopped")
                yc = np.minimum(y, TOP * 0.98)
                ax.semilogy(np.arange(1, len(y) + 1), yc,
                            color=ARM_COLOR[arm], lw=1.1 if ls == "-" else 1.0,
                            ls=ls, alpha=1.0 if ls == "-" else 0.75,
                            zorder=2 + zi)
                out = np.nonzero(y > TOP)[0]
                if out.size:
                    ax.scatter(out + 1, np.full(out.size, TOP * 0.98),
                               s=3, marker="^", color=ARM_COLOR[arm],
                               zorder=4 + zi)
            ax.axhline(r["floor"], color="k", ls=":", lw=1.2, zorder=1)
        ax.set_ylim(1e-16, 2.0)
        ax.set_title(fn, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel(r"relative $L_2$")
    axes[1, 0].set_ylabel(r"relative $L_2$")
    handles = [Line2D([0], [0], color=ARM_COLOR[a], lw=1.4,
                      label=ARM_LABEL[a]) for a in ARMS]
    handles += [Line2D([0], [0], color="0.35", lw=1.1, label="train rel $L_2$"),
                Line2D([0], [0], color="0.35", lw=1.0, ls="--",
                       label="eval rel $L_2$"),
                Line2D([0], [0], color="k", ls=":", lw=1.2,
                       label="lstsq floor"),
                Line2D([0], [0], marker="^", ls="", color="0.35", ms=4,
                       label="pegged at axis top (diverged)")]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.suptitle(f"iteration 11 -- N={N}, {iters} iterations "
                 "(all curves relative $L_2$; the solve re-floors at every "
                 "refresh, Adam wanders between)", y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_dial_gif(rows, arm, N, path, fps=8):
    """THE DIAL (4th gif): the certificate that selects the exploit set.

    Same 4x3 layout as the parameter movies -- rows = the 4 gif targets,
    columns = the (w | b | v) blocks -- but the y axis is the certificate
    score qhat_i (relative measured nonlinearity, log scale) plotted
    against each neuron's CURRENT center. Red = in the exploit set (this
    parameter is being solved), gray = Adam only. The two horizontal
    lines are the hysteresis thresholds: below Q_ENTER a parameter is
    admitted, above Q_EXIT it is demoted back to Adam. The shaded band at
    the bottom is "measured zero" -- qhat below the probe's own rounding
    noise, i.e. indistinguishable from exactly linear."""
    plt = _mpl()
    from matplotlib.lines import Line2D
    from PIL import Image
    import io

    cases = {r["fn"]: r for r in rows
             if r["N"] == N and r["arm"] == arm and "Q" in r}
    fns = [f for f in run.TARGETS3 if f in cases]
    if not fns:
        print(f"  no dial data for {arm} N={N}")
        return
    steps = list(cases[fns[0]]["steps"])
    total = steps[-1]
    QFLOOR = 1e-17
    XW = 2.0
    images = []
    for si, s in enumerate(steps):
        fig, axes = plt.subplots(len(fns), 3, figsize=(12.5, 2.5 * len(fns)),
                                 squeeze=False)
        for i, fn in enumerate(fns):
            r = cases[fn]
            W = len(r["snaps"][str(s)]["w"])
            w_ = np.array(r["snaps"][str(s)]["w"])
            b_ = np.array(r["snaps"][str(s)]["b"])
            c = -b_ / np.where(np.abs(w_) > 1e-12, w_, 1e-12)
            Q, E = r["Q"][si], r["E"][si] > 0.5
            blocks = [("first-layer w", slice(0, W), c),
                      ("bias b", slice(W, 2 * W), c),
                      ("readout v", slice(2 * W, 3 * W), c)]
            rel_now = (r["rels"][min(max(int(s) - 1, 0), len(r["rels"]) - 1)]
                       if r["rels"] else np.nan)
            for j, (ttl, sl, xs) in enumerate(blocks):
                ax = axes[i, j]
                q = np.maximum(Q[sl], QFLOOR)
                e = E[sl]
                vis = np.abs(xs) <= XW
                ax.axvspan(-1, 1, color="#1f4e8c", alpha=0.05)
                ax.axhspan(QFLOOR, 1e-16, color="0.85", alpha=0.7)
                ax.axhline(Q_ENTER, color="tab:red", ls="--", lw=0.9)
                ax.axhline(Q_EXIT, color="tab:orange", ls=":", lw=0.9)
                ax.scatter(xs[vis & ~e], q[vis & ~e], s=7, color="0.55",
                           edgecolors="none")
                ax.scatter(xs[vis & e], q[vis & e], s=9, color="tab:red",
                           edgecolors="none")
                ax.set_yscale("log")
                ax.set_ylim(QFLOOR, 3.0)
                ax.set_xlim(-XW, XW)
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(ttl, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{fn}\n" + r"$\hat q$", fontsize=7)
            axes[i, 2].text(0.98, 0.92, f"B={int(E.sum())}   "
                            f"rel={rel_now:.1e}", transform=axes[i, 2].transAxes,
                            ha="right", fontsize=6, color="#333")
        handles = [Line2D([0], [0], marker="o", ls="", color="tab:red",
                          label="in exploit set (solved exactly)"),
                   Line2D([0], [0], marker="o", ls="", color="0.55",
                          label="Adam only"),
                   Line2D([0], [0], color="tab:red", ls="--",
                          label=r"admit below $Q_{\rm enter}=10^{-8}$"),
                   Line2D([0], [0], color="tab:orange", ls=":",
                          label=r"demote above $Q_{\rm exit}=10^{-7}$"),
                   Line2D([0], [0], color="0.85", lw=6,
                          label="measured zero (probe noise floor)")]
        H = 2.5 * len(fns)                 # reserve a fixed 1.0in header
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0 - 0.06 / H), ncol=5,
                   frameon=False, fontsize=8)
        fig.suptitle(f"iteration 11 dial -- {arm}   N={N}   step {s}/{total}"
                     r"   (x = current centers $-b_k/w_k$,  y = certificate "
                     r"$\hat q$)", fontsize=10, y=1.0 - 0.62 / H)
        fig.tight_layout(rect=[0, 0, 1, 1.0 - 1.0 / H])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  saved {path} ({len(images)} frames, "
          f"{path.stat().st_size/1e6:.1f} MB)")


def render_1d(rows, iters=None, gifs=True):
    render_scaling(rows, OUT_1D / "scaling_2x3.png", iters)
    have = {r["N"] for r in rows}
    nloss = 256 if 256 in have else max(have)
    render_loss_2x2(rows, nloss, OUT_1D / f"loss_curves_2x2_N{nloss}.png",
                    iters)
    if not gifs:
        return
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        gdir = OUT_1D / f"gifs_{arm}"
        for N in run.GIF_WIDTHS:
            if not any(r["N"] == N for r in arm_rows):
                continue
            expd06.render_gif3(arm_rows, run.SEED_NAME, N,
                               gdir / f"params_N{N}.gif", scatter=True)
        dial_n = 256 if 256 in have else max(have)
        render_dial_gif(arm_rows, arm, dial_n, gdir / f"dial_N{dial_n}.gif")


def render_coupling_law(path):
    """DIAGNOSTIC (not part of the standard suite): why the solved state
    does not survive the next base step.

    After the exploit solve, the certified block sits at its exact
    optimum, but the base regime then moves the UNCERTIFIED coordinates
    by ~eta per coordinate (Adam's step is scale-free: m/sqrt(v) ~ sign,
    so the step length does not shrink as the residual does). Those
    coordinates are the features the solved readout multiplies, so the
    prediction error re-injected by one base step is ~ ||v_solved||*eta.
    Sweeping eta at two very different ||v|| tests that law directly:
    the QI toy solves to ||v|| ~ 0.5 (expA05: the readout norm is O(1)
    and decays with width), while random depth-2 features force
    ||v|| ~ 1e5, and the damage tracks the product across five orders."""
    plt = _mpl()
    from matplotlib.lines import Line2D
    import tether_v6 as tv6
    etas = np.logspace(-8, -2, 13)
    cases = []

    env = build_case("sine", 128)
    W = env["W"]
    th = env["theta0"].clone()
    m = th.numel()
    E = torch.zeros(m, dtype=torch.bool)
    E[2 * W:] = True
    th_s = th + varpro_delta(th, torch.nonzero(E).flatten(),
                             env["f_resid"](th), env["f_jvp"])
    cases.append(("QI toy, N=128 (sine)", float(th_s[2 * W:].norm()),
                  env["eval_rel"], th_s, E, "tab:blue"))

    C = tv6.mlp2_closures("sine", seed=0, h=(64, 64))
    off = {nm: (o, n) for nm, o, n, _ in C["slices"]}
    th2 = C["theta0"].clone()
    m2 = C["m"]
    E2 = torch.zeros(m2, dtype=torch.bool)
    vo, vn = off["v"]
    E2[vo:vo + vn] = True
    E2[off["c0"][0]] = True
    th2_s = th2 + varpro_delta(th2, torch.nonzero(E2).flatten(),
                               C["f_resid"](th2), C["f_jvp"])
    cases.append(("depth-2, random features (sine)",
                  float(th2_s[vo:vo + vn].norm()), C["eval_rel"], th2_s,
                  E2, "tab:red"))

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, vnorm, ev, th_solved, mask, col in cases:
        base = ev(th_solved)
        ys = []
        for eta in etas:
            step = torch.zeros(th_solved.numel(), dtype=th_solved.dtype)
            step[~mask] = eta
            ys.append(ev(th_solved + step))
        ax.loglog(etas, ys, "-o", color=col, ms=4, lw=1.6,
                  label=f"{label}   " + r"$\|v\|=$" + f"{vnorm:.1e}")
        ax.loglog(etas, vnorm * etas, ls="--", color=col, lw=1.0, alpha=0.7)
        ax.axhline(base, color=col, ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"base-regime step size per coordinate  $\eta$")
    ax.set_ylabel(r"eval rel $L_2$ after ONE base step")
    ax.set_ylim(1e-16, 1e4)
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="0.4", ls="--",
                       label=r"prediction  $\|v\|\,\eta$"),
                Line2D([0], [0], color="0.4", ls=":",
                       label="eval rel at the solved point")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=1, frameon=False, fontsize=9)
    fig.suptitle("coupling law: one base step re-injects "
                 r"$\sim\|v_{\rm solved}\|\,\eta$ of error", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ------------------------------- batching -------------------------------

POOL_SOFT = 8012
FRACS = [("1", 1), ("3/4", None), ("1/2", 2), ("1/4", 4), ("1/16", 16)]
POOL_STRONG = 8000
B_STRONG = 2003


def batch_env(fn_name, N, n_tr):
    """Full-batch-capable environment on an n_tr-point pool with
    index-restricted closures (the batched code path)."""
    f_vec = run.make_fn(run.TARGETS[fn_name])
    centers, gamma, _halo = run.geometry(N, run.LAM)
    W = centers.size
    theta0 = torch.tensor(np.concatenate([
        np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))
    x_tr = np.linspace(-1, 1, n_tr)
    x_ev = np.linspace(-1, 1, run.N_EVAL)
    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(f_vec(x_tr))
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))
    ye_norm = float(torch.linalg.norm(ye))
    # fixed probe/gauge rows: the certificate must never read sampling
    # noise as geometry motion (<=2003 rows, as in iteration_9's batched
    # form)
    g_idx = torch.from_numpy(np.sort(np.random.default_rng(1).choice(
        n_tr, size=min(n_tr, 2003), replace=False)))

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] \
            + th[-1]

    def f_pred(th):
        return model(xt[g_idx], th)

    def f_resid(th, rows=None):
        with torch.no_grad():
            x_ = xt if rows is None else xt[rows]
            y_ = yt if rows is None else yt[rows]
            return (model(x_, th) - y_).detach()

    def f_jvp(th, dvec, rows=None):
        x_ = xt if rows is None else xt[rows]
        _, jd = torch.func.jvp(lambda t: model(x_, t), (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed, rows=None):
        x_ = xt if rows is None else xt[rows]
        thr = th.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(model(x_, thr), thr, grad_outputs=seed)
        return g.detach()

    def f_residgrad(th, rows=None):
        x_ = xt if rows is None else xt[rows]
        y_ = yt if rows is None else yt[rows]
        thr = th.detach().requires_grad_(True)
        pred = model(x_, thr)
        r = (pred - y_).detach()
        (g,) = torch.autograd.grad(pred, thr,
                                   grad_outputs=(2.0 / len(r)) * r)
        return r, g.detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(model(xe, th) - ye)) / ye_norm

    def get_wbv(th):
        t = th.detach().numpy()
        return {"w": t[:W].copy(), "b": t[W:2 * W].copy(),
                "v": t[2 * W:3 * W].copy()}

    A = np.hstack([run.build_phi(x_tr, gamma, centers, "tanh"),
                   np.ones((n_tr, 1))])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > RCOND * s[0]
    sol = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                  * (U.T @ f_vec(x_tr)))
    A_ev = np.hstack([run.build_phi(x_ev, gamma, centers, "tanh"),
                      np.ones((run.N_EVAL, 1))])
    floor = float(np.linalg.norm(A_ev @ sol - f_vec(x_ev))
                  / np.linalg.norm(f_vec(x_ev)))
    return dict(theta0=theta0, f_pred=f_pred, f_resid=f_resid,
                f_jvp=f_jvp, f_vjp=f_vjp, f_residgrad=f_residgrad,
                eval_rel=eval_rel, get_wbv=get_wbv, floor=floor,
                y_norm=float(torch.linalg.norm(yt)))


def batch_case11(job):
    torch.set_num_threads(job.get("threads", 1))
    torch.set_default_dtype(torch.float64)
    import batching
    n_tr = job["n_train"]
    env = batch_env(job["fn"], job["N"], n_tr)
    bsize = job["bsize"]
    bit = (None if bsize >= n_tr
           else batching.make_batch_iter(n_tr, bsize, seed=0))
    t0 = time.time()
    losses, rels, _, _, stats = train_iter11(
        env["theta0"], env["f_resid"], env["f_jvp"], env["f_vjp"],
        env["f_pred"], n_tr, job["iters"], {0}, env["get_wbv"],
        env["eval_rel"], env["y_norm"], arm=job["arm"], snap_S=False,
        f_residgrad=env["f_residgrad"], batch_iter=bit)
    return {"fn": job["fn"], "arm": job["arm"],
            "blabel": job.get("blabel"), "armlabel": job.get("armlabel"),
            "bsize": bsize, "best_rel": float(np.min(rels)),
            "floor": env["floor"], "wall_s": round(time.time() - t0, 1)}


def render_soft(rows, path):
    """Soft win: does error degrade smoothly as the batch shrinks at
    constant information volume, or cliff?"""
    plt = _mpl()
    present = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    xs = np.arange(len(FRACS))
    for k, fn in enumerate(run.TARGETS3):
        ax = axes.flat[k]
        for arm in present:
            rs = {r["blabel"]: r for r in rows
                  if r["fn"] == fn and r["arm"] == arm}
            ys = [rs[lb]["best_rel"] if lb in rs else np.nan
                  for lb, _ in FRACS]
            ax.semilogy(xs, ys, "o-", color=ARM_COLOR[arm], lw=1.8, ms=5,
                        label=ARM_LABEL[arm])
        fl = [r["floor"] for r in rows if r["fn"] == fn]
        if fl:
            ax.axhline(fl[0], color="k", ls=":", lw=1.2,
                       label="pool lstsq floor")
        ax.set_xticks(xs)
        ax.set_xticklabels([lb for lb, _ in FRACS])
        ax.set_ylim(1e-16, 1e0)
        ax.set_title(fn, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if k >= 2:
            ax.set_xlabel(f"batch fraction of the {POOL_SOFT}-point pool")
        if k % 2 == 0:
            ax.set_ylabel(r"best eval rel $L_2$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    fig.suptitle("soft win: batch size vs error at constant information "
                 f"volume (N=256, random batches, no accumulation)",
                 y=0.92, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def render_strong(rows, path):
    """Strong win: does one random 2003-batch per step approach full
    batch / exact accumulation on the same pool?"""
    plt = _mpl()
    arms3 = ["full2003", "pool_random", "pool_accum"]
    labels = ["full batch\nn=2003", "8000 pool\nrandom 2003/step",
              "8000 pool\nfull-pool step"]
    present = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    width = 0.38 if len(present) > 1 else 0.6
    for k, fn in enumerate(run.TARGETS3):
        ax = axes[k]
        for ai, arm in enumerate(present):
            rs = {r["armlabel"]: r for r in rows
                  if r["fn"] == fn and r["arm"] == arm}
            xs = np.arange(3) + (ai - 0.5 * (len(present) - 1)) * width
            vals = [max(rs[a]["best_rel"], 1e-16) if a in rs else np.nan
                    for a in arms3]
            ax.bar(xs, vals, width=width, color=ARM_COLOR[arm],
                   label=ARM_LABEL[arm])
            for x, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(x, v * 1.6, f"{v:.0e}", ha="center", fontsize=6)
        fl = [r["floor"] for r in rows
              if r["fn"] == fn and r["armlabel"] == "full2003"]
        if fl:
            ax.axhline(fl[0], color="k", ls=":", lw=1.2,
                       label="n=2003 lstsq floor")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(fn, fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel(r"best eval rel $L_2$")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=9)
    fig.suptitle("strong win: interpolation-regime bars (N=256; win = the "
                 "middle bar approaching its neighbours)", y=0.98,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main_batching(workers, iters):
    """Batching bench, varpro only.

    The `qn` arm is excluded by construction, not by preference: it runs
    the CG-lineage carried residual, and a carried residual is bound to
    SPECIFIC ROWS. Under resampled minibatches the carried vector and the
    current batch's forward pass have different lengths (measured: an
    8012-long seed against a 6009-row output), and even where the sizes
    happen to match the state refers to rows that are no longer being
    evaluated. The varpro arm has no such state -- it recomputes the
    residual every step -- which is exactly why it survives batching.
    `qn` is retired on the full-batch grid anyway (3-9 orders short)."""
    threads = max(1, (os.cpu_count() or 4) // workers)
    OUT_BATCH.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fn in run.TARGETS3:
        for arm in ["varpro"]:
            for lb, dv in FRACS:
                bs = (int(round(POOL_SOFT * 0.75)) if dv is None
                      else -(-POOL_SOFT // dv))
                jobs.append({"fn": fn, "N": 256, "arm": arm, "blabel": lb,
                             "bsize": bs, "n_train": POOL_SOFT,
                             "iters": iters, "threads": threads,
                             "kind": "soft"})
            for al, n_tr, bs in (("full2003", 2003, 2003),
                                 ("pool_random", POOL_STRONG, B_STRONG),
                                 ("pool_accum", POOL_STRONG, POOL_STRONG)):
                jobs.append({"fn": fn, "N": 256, "arm": arm,
                             "armlabel": al, "bsize": bs, "n_train": n_tr,
                             "iters": iters, "threads": threads,
                             "kind": "strong"})
    soft, strong, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(batch_case11, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            (soft if futs[fut]["kind"] == "soft" else strong).append(r)
            tag = r["blabel"] or r["armlabel"]
            print(f"  [{i}/{len(jobs)}] {r['arm']:6s} {r['fn']:9s} "
                  f"{tag:12s} b={r['bsize']:<5d} best={r['best_rel']:.2e} "
                  f"floor={r['floor']:.1e} ({r['wall_s']}s)", flush=True)
    print(f"batching done in {time.time() - t0:.0f}s")
    (OUT_BATCH / "soft_rows.json").write_text(json.dumps(soft))
    (OUT_BATCH / "strong_rows.json").write_text(json.dumps(strong))
    render_soft(soft, OUT_BATCH / "softwin_batchsize_vs_error.png")
    render_strong(strong, OUT_BATCH / "strongwin_interpolation_bars.png")


# ------------------------------- depth-2 -------------------------------

D2_WIDTHS = [32, 64]        # hidden width per layer (h1 = h2 = W)


def run_depth2_11(job):
    """Light gate: iteration 11 on the 2-hidden-layer bench (RANDOM init,
    both hidden layers must learn -- the architecture-independence guard,
    where the correct lock is NOT 'freeze all but the last layer').
    Reuses tether_v6's environment (closures, final-lstsq reference); the
    probe is the dense per-parameter form so the dial gif resolves
    individual parameters."""
    torch.set_num_threads(job.get("threads", 1))
    torch.set_default_dtype(torch.float64)
    import tether_v6 as tv6
    W = job["width"]
    C = tv6.mlp2_closures(job["fn"], seed=job.get("seed", 0), h=(W, W))
    chunk = auto_chunk(tv6.D2_NTR, W)

    def probe_fn(th):
        return probe_dense(C["f_pred"], th, chunk)

    iters = job.get("iters", tv6.D2_ITERS)
    frames_at = set(expd06.frame_steps(iters))
    t0 = time.time()

    if job["arm"] == "adam":              # grayed-out baseline per cell
        th = C["theta0"].clone().requires_grad_(True)
        optim = torch.optim.Adam([th], lr=1e-3)
        losses, rels = [], []
        for _ in range(iters):
            optim.zero_grad(set_to_none=True)
            r = C["f_pred"](th) - C["yt"]
            loss = (r * r).mean()
            loss.backward()
            optim.step()
            losses.append(float(loss))
            rels.append(C["eval_rel"](th.detach()))
        return {"arm": "adam", "fn": job["fn"], "width": W,
                "best_rel": float(np.min(rels)), "rels": rels,
                "losses": losses,
                "final_lstsq_rel": C["final_lstsq_rel"](th.detach()),
                "B_hist": [], "m": C["m"],
                "slices": [(nm, o, n) for nm, o, n, _ in C["slices"]],
                "stats": {"probes": 0, "solves": 0, "revert": 0,
                          "B_final": 0},
                "wall_s": round(time.time() - t0, 1)}

    losses, rels, snaps, theta_fin, stats = train_iter11(
        C["theta0"], C["f_resid"], C["f_jvp"], C["f_vjp"], C["f_pred"],
        tv6.D2_NTR, iters, frames_at, C["get_wbv"], C["eval_rel"],
        C["y_norm"], arm=job["arm"], probe_fn=probe_fn, snap_S=True)
    slices = [(nm, o, n) for nm, o, n, _ in C["slices"]]
    ssteps = sorted(snaps)
    snap_dir = OUT_D2 / "snaps"
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snap_dir / f"{job['fn']}--h{W}--{job['arm']}.npz",
        steps=np.array(ssteps, dtype=np.int64),
        Q=np.stack([snaps[s]["Q"] for s in ssteps]),
        E=np.stack([snaps[s]["E"] for s in ssteps]),
        losses=np.array(losses), rels=np.array(rels),
        slices=json.dumps(slices))
    return {"arm": job["arm"], "fn": job["fn"], "width": W,
            "best_rel": float(np.min(rels)), "rels": rels,
            "losses": losses,
            "final_lstsq_rel": C["final_lstsq_rel"](theta_fin),
            "B_hist": stats["B_hist"], "m": C["m"], "slices": slices,
            "stats": {k: stats[k] for k in
                      ("probes", "solves", "revert", "B_final")},
            "wall_s": round(time.time() - t0, 1)}


def render_depth2_loss(rows, path, iters=None):
    """2x2: rows = the two targets, columns = the two hidden widths.
    Per arm: train rel L2 (solid) and eval rel L2 (dashed) from RANDOM
    init. Gray = the prior bench's Adam run at the standard width, as the
    reference every optimizer here has to beat."""
    plt = _mpl()
    from matplotlib.lines import Line2D
    fns = sorted({r["fn"] for r in rows})
    widths = sorted({r["width"] for r in rows})
    fig, axes = plt.subplots(len(fns), len(widths),
                             figsize=(6.0 * len(widths), 3.9 * len(fns)),
                             squeeze=False)
    for i, fn in enumerate(fns):
        y2 = run._y2mean(fn)
        for j, W in enumerate(widths):
            ax = axes[i][j]
            zord = {"adam": 1, "qn": 2, "varpro": 3}   # blue never hidden
            TOP = 2.0
            for r in sorted([r for r in rows if r["fn"] == fn
                             and r["width"] == W],
                            key=lambda r: zord.get(r["arm"], 0)):
                col = ARM_COLOR.get(r["arm"], "0.55")
                z = zord.get(r["arm"], 0)
                tr = np.sqrt(np.array(r["losses"]) / y2)
                ev = np.array(r["rels"])
                for y, ls in ((tr, "-"), (ev, "--")):
                    lw = (1.0 if r["arm"] == "adam" else 1.2) \
                        if ls == "-" else 1.0
                    ax.semilogy(np.arange(1, len(y) + 1),
                                np.minimum(y, TOP * 0.98), color=col,
                                lw=lw, ls=ls,
                                alpha=1.0 if ls == "-" else 0.75, zorder=z)
                    out = np.nonzero(y > TOP)[0]
                    if out.size:
                        ax.scatter(out + 1, np.full(out.size, TOP * 0.98),
                                   s=3, marker="^", color=col, zorder=z + 4)
            ax.set_ylim(1e-16, 2.0)
            ax.set_title(f"{fn}   hidden {W}x{W}", fontsize=10)
            ax.grid(True, which="both", alpha=0.25)
            if i == len(fns) - 1:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel(r"relative $L_2$")
    handles = [Line2D([0], [0], color=ARM_COLOR[a], lw=1.4,
                      label=ARM_LABEL[a]) for a in ARMS]
    handles += [Line2D([0], [0], color="0.55", lw=1.0,
                       label="Adam alone (baseline, same init)"),
                Line2D([0], [0], color="0.35", lw=1.2,
                       label="train rel $L_2$"),
                Line2D([0], [0], color="0.35", lw=1.0, ls="--",
                       label="eval rel $L_2$"),
                Line2D([0], [0], marker="^", ls="", color="0.35", ms=4,
                       label="pegged at axis top (diverged)")]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("iteration 11 depth-2 -- 2 hidden layers, RANDOM init "
                 "(no QI geometry, both hidden layers must learn)",
                 y=1.06, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_depth2_dial_gif(arm, path, fps=8):
    """Depth-2 dial (4x3): rows = the four (target, hidden width) cells,
    columns = the three parameter groups (hidden layer 1 | hidden layer 2
    | readout). Same certificate axis as the 1D dial -- y = qhat, red =
    in the exploit set, dashed/dotted = the admit/demote thresholds, gray
    band = measured zero. Read it as: which layers does the certificate
    admit on an architecture where 'last layer' is not the right answer
    a priori?"""
    plt = _mpl()
    from matplotlib.lines import Line2D
    from PIL import Image
    import io

    cells = []
    for p in sorted((OUT_D2 / "snaps").glob(f"*--{arm}.npz")):
        fn, rest = p.stem.split("--h")
        w_s, _ = rest.split("--")
        z = np.load(p)
        cells.append({"fn": fn, "width": int(w_s), "Q": z["Q"], "E": z["E"],
                      "steps": z["steps"], "rels": z["rels"],
                      "slices": json.loads(str(z["slices"]))})
    if not cells:
        print(f"  no depth-2 dial data for {arm}")
        return
    cells.sort(key=lambda c: (c["fn"], c["width"]))
    steps = list(cells[0]["steps"])
    total = steps[-1]
    QFLOOR = 1e-17
    groups = [("W1  (layer 1 weights)", ("W1",)),
              ("b1  (layer 1 bias)", ("b1",)),
              ("W2  (layer 2 weights)", ("W2",)),
              ("b2  (layer 2 bias)", ("b2",)),
              ("v, c0  (readout)", ("v", "c0"))]
    images = []
    for si, s in enumerate(steps):
        fig, axes = plt.subplots(len(cells), len(groups),
                                 figsize=(16.5, 2.5 * len(cells)),
                                 squeeze=False)
        for i, c in enumerate(cells):
            off = {nm: (o, n) for nm, o, n in c["slices"]}
            Q, E = c["Q"][si], c["E"][si] > 0.5
            rel_now = c["rels"][min(max(int(s) - 1, 0), len(c["rels"]) - 1)]
            for j, (ttl, names) in enumerate(groups):
                ax = axes[i][j]
                q_parts, e_parts = [], []
                for nm in names:
                    if nm in off:
                        o, n = off[nm]
                        q_parts.append(Q[o:o + n])
                        e_parts.append(E[o:o + n])
                q = np.maximum(np.concatenate(q_parts), QFLOOR)
                e = np.concatenate(e_parts)
                xs = np.arange(q.size)
                ax.axhspan(QFLOOR, 1e-16, color="0.85", alpha=0.7)
                ax.axhline(Q_ENTER, color="tab:red", ls="--", lw=0.9)
                ax.axhline(Q_EXIT, color="tab:orange", ls=":", lw=0.9)
                ax.scatter(xs[~e], q[~e], s=5, color="0.55",
                           edgecolors="none")
                ax.scatter(xs[e], q[e], s=7, color="tab:red",
                           edgecolors="none")
                ax.set_yscale("log")
                ax.set_ylim(QFLOOR, 3.0)
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(ttl, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{c['fn']}  {c['width']}x{c['width']}\n"
                                  + r"$\hat q$", fontsize=7)
                if i == len(cells) - 1:
                    ax.set_xlabel("parameter index in group", fontsize=7)
            axes[i][-1].text(0.98, 0.90, f"B={int(E.sum())}   "
                             f"rel={rel_now:.1e}",
                             transform=axes[i][-1].transAxes, ha="right",
                             fontsize=6, color="#333")
        handles = [Line2D([0], [0], marker="o", ls="", color="tab:red",
                          label="in exploit set (solved exactly)"),
                   Line2D([0], [0], marker="o", ls="", color="0.55",
                          label="Adam only"),
                   Line2D([0], [0], color="tab:red", ls="--",
                          label=r"admit below $Q_{\rm enter}=10^{-8}$"),
                   Line2D([0], [0], color="tab:orange", ls=":",
                          label=r"demote above $Q_{\rm exit}=10^{-7}$"),
                   Line2D([0], [0], color="0.85", lw=6,
                          label="measured zero (probe noise floor)")]
        H = 2.5 * len(cells)               # reserve a fixed 1.0in header
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0 - 0.06 / H), ncol=5,
                   frameon=False, fontsize=8)
        fig.suptitle(f"iteration 11 depth-2 dial -- {arm}   step {s}/{total}"
                     "   (random init, both hidden layers free;  y = "
                     r"certificate $\hat q$)", fontsize=10,
                     y=1.0 - 0.62 / H)
        fig.tight_layout(rect=[0, 0, 1, 1.0 - 1.0 / H])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  saved {path} ({len(images)} frames, "
          f"{path.stat().st_size/1e6:.1f} MB)")


def render_depth2_all(rows, iters=None):
    render_depth2_loss(rows, OUT_D2 / "depth2_loss_2x2.png", iters)
    for arm in ARMS:
        render_depth2_dial_gif(arm, OUT_D2 / f"gifs_{arm}" / "dial.gif")


def main_depth2(workers, iters):
    import tether_v6 as tv6
    threads = max(1, (os.cpu_count() or 4) // workers)
    jobs = [{"fn": f, "arm": a, "width": W, "threads": threads,
             "iters": iters}
            for f in tv6.D2_FNS for W in D2_WIDTHS
            for a in ARMS + ["adam"]]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_depth2_11, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['arm']:6s} {r['fn']:8s} "
                  f"h={r['width']:<3d} best={r['best_rel']:.2e} "
                  f"lstsq@final={r['final_lstsq_rel']:.2e} "
                  f"B={r['stats']['B_final']}/{r['m']} ({r['wall_s']}s)",
                  flush=True)
    OUT_D2.mkdir(parents=True, exist_ok=True)
    (OUT_D2 / "depth2_rows.json").write_text(json.dumps(rows))
    render_depth2_all(rows, iters)


# --------------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="the 1D_test suite")
    ap.add_argument("--batching", action="store_true")
    ap.add_argument("--depth2", action="store_true")
    ap.add_argument("--render", action="store_true",
                    help="re-render 1D figures/gifs from saved snaps")
    ap.add_argument("--coupling", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--iters", type=int, default=run.ITERS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--widths", type=int, nargs="*", default=None,
                    help="rerun only these widths (merges into rows.json)")
    ap.add_argument("--cmem", type=float, default=C_MEM,
                    help="sqrt(k) budget constant: b_cap = sqrt(cmem * m)")
    args = ap.parse_args()

    if args.coupling:
        render_coupling_law(OUT / "coupling_law.png")
        return
    if args.render:
        render_1d(load_rows_1d(), args.iters, gifs=not args.no_gifs)
        return
    if args.batching:
        main_batching(args.workers, min(args.iters, 1500))
        return
    if args.depth2:
        main_depth2(args.workers, args.iters)
        return

    widths = (args.widths if args.widths else
              ([32, 128] if args.smoke else run.WIDTHS))
    fns = ["sine", "runge"] if args.smoke else list(run.TARGETS)
    iters = 200 if args.smoke else args.iters
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    jobs = [{"fn": f, "N": n, "arm": a, "iters": iters, "threads": threads,
             "snap_S": True, "c_mem": args.cmem}
            for f in fns for n in widths for a in ARMS]
    print(f"iter11 1D_test | {len(jobs)} runs | {len(fns)} targets x "
          f"{widths} x {ARMS} | iters={iters}")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_case11, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            s = r["stats"]
            print(f"  [{i}/{len(jobs)}] {r['arm']:6s} N={r['N']:<4d} "
                  f"{r['fn']:13s} best={r['best_rel']:.3e} "
                  f"floor={r['floor']:.1e} B={s['B_final']} "
                  f"solves={s['solves']} probes={s['probes']} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"training done in {time.time() - t0:.0f}s")
    OUT_1D.mkdir(parents=True, exist_ok=True)
    # rebuild rows.json from ALL persisted snaps, so a partial-width
    # rerun merges into the suite instead of truncating it
    all_rows = load_rows_1d()
    (OUT_1D / "rows.json").write_text(json.dumps(
        [{k: r[k] for k in ("fn", "N", "arm", "floor", "best_rel",
                            "losses", "rels", "stats")} for r in all_rows]))
    render_1d(all_rows, iters, gifs=not args.no_gifs)


if __name__ == "__main__":
    main()
