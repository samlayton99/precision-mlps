"""expD08 batching study: slim optimizer vs Adam under minibatching.

Grid (PER optimizer): batch in {B, B/4, B/16, B/64, B/256} (B = all 10000
train points) x 4 targets (TARGETS3) x N in {64, 128, 256, 512} = 80 runs,
3000 optimizer steps each. Standard DL data order: the train set is
RESHUFFLED at every epoch and consumed in batches. Same seed9 init, same
data, same batch schedule for both optimizers; outputs in sibling
subfolders batching/slim/ and batching/adam/.

slim design rule (the ONLY adaptation of the iteration_5 core): the
carried-residual recursion r_hat += alpha J d is valid only when every
sample is present at every step, i.e. full batch. With minibatches the
batch residual is recomputed FRESH at each visit -- carrying across visits
would apply stale increments. Everything else is unchanged: projection
memory, exhaustion escape, exact Gauss-Newton steps on the batch quadratic,
the NaN reject. Known structural fragility (measured, not a bug): with b
samples the batch quadratic has rank <= b, so most directions have near-
zero batch curvature and the exact step alpha = -g.d/c can divide by ~0.

adam control: torch.optim.Adam lr=1e-3 on raw coordinates, same batches.

Outputs per optimizer subfolder:
  gifs/params_{fn}_N256.gif    4 rows = B, B/4, B/16, B/64; cols = w | b | v
                               (theory-locked y-limits; per-panel sigma and
                               max|.| of the plotted coefficients)
  batch_scaling_2x2.png        per-fn: best eval rel L2 vs batch size
  batching_rows.json, snaps/
Plus batching/comparison_2x2.png overlaying slim (solid) vs adam (dashed).

Run:  uv run python experiments/expD08_qi_init_nlcg/batching.py
      [--opt slim|adam|both] [--smoke] [--render-only] [--workers K]
"""
from __future__ import annotations

import argparse
import io
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
import run  # noqa: E402  (expD08 run.py: slim core + geometry/targets)

expd06 = run.expd06
BATCH_ROOT = run.OUT_DIR.parent / "batching"
BLEVELS = [("B", 1), ("B/4", 4), ("B/16", 16), ("B/64", 64), ("B/256", 256)]
GIF_BLEVELS = ["B", "B/4", "B/16", "B/64"]
FNS = expd06.TARGETS3                     # sine, sine_8pi, runge, exp
WIDTHS = [64, 128, 256, 512]
GIF_N = 256
ITERS = 3000
N_TRAIN_B = 10000                         # batching study train-set size


def out_dir(opt):
    return BATCH_ROOT / opt


def _tag(blabel):
    return blabel.replace("/", "d")


def make_batch_iter(n, bsize, seed=0):
    """Standard DL order: reshuffle every epoch, consume in batches."""
    rng = np.random.default_rng(seed)

    def gen():
        while True:
            perm = rng.permutation(n)
            for i in range(0, n, bsize):
                yield torch.from_numpy(perm[i:i + bsize].copy())
    return gen()


# ---------------- slim core over reshuffled fresh minibatches ----------------

def train_slim_fresh_batches(theta0, f_resid_S, f_jvp_S, f_vjp_S, batch_iter,
                             iters, frames_at, get_wbv, eval_rel,
                             guards=(), n_total=None, refresh=200,
                             f_resid_full=None, f_vjp_full=None):
    """The slim core with the per-iteration gradient taken on the next
    minibatch from `batch_iter`, residual fresh at each visit. Same memory /
    exhaustion / exact-step / NaN-reject machinery as
    train_residual_cg_slim.

    `guards` (hardened-minibatch ablation, one at a time):
      floor  sampling-noise curvature floor: alpha = -g.d/(c + mu ||d||^2),
             mu = lhat * (1/b - 1/n). Vanishes at full batch by
             construction, so it cannot alter the validated method.
      trust  gradient-norm spike rejection (the iteration_4 safeguard,
             minibatch path only): reject if ||g|| > 100x the last-10
             accepted, with the 10-streak escape.
      gate   grow the memory basis only when batch-to-batch gradient
             variability ||g_k - g_{k-1}|| / ||g_k|| < 0.3 (memory built
             from noise measured worse than none).
      svrg   variance-reduced gradient with the refresh full pass as
             anchor (requires f_resid_full / f_vjp_full):
               g = g_anchor + (2/b) [J_S(th)^T r_S(th)
                                     - J_S(th_a)^T r_a[S]]
             E_S[g] is the exact full gradient; the noise scales with
             ||th - th_a||, not 1/b, so it vanishes as the run converges.
             Anchor refreshed every `refresh` iterations (one full
             forward + one full VJP, amortized).
    """
    guards = frozenset(guards)
    theta = theta0.clone()
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0}

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
    gt = None
    ggt = 0.0
    d = None
    lhat = None
    exhaust_streak = 0
    theta_prev = None
    g_prev_raw = None
    gn_window: list[float] = []
    revert_streak = 0
    use_svrg = "svrg" in guards
    theta_a = None
    r_a = None
    g_a = None

    for it in range(1, iters + 1):
        if use_svrg and (theta_a is None or it % refresh == 0):
            theta_a = theta.clone()            # anchor: full pass + full VJP
            r_a = f_resid_full(theta_a)
            g_a = f_vjp_full(theta_a, (2.0 / n_total) * r_a)
            stats["refresh"] += 1
        S = next(batch_iter)
        b = S.numel()
        r_S = f_resid_S(theta, S)
        g = f_vjp_S(theta, (2.0 / b) * r_S, S)
        if use_svrg:
            g = g_a + g - f_vjp_S(theta_a, (2.0 / b) * r_a[S], S)
        gn = float(torch.linalg.norm(g))
        if not np.isfinite(gn):
            stats["revert"] += 1               # NaN reject (always on)
            if theta_prev is not None:
                theta = theta_prev
            if gt is not None:
                d = -gt
            if it in frames_at:
                snaps[it] = get_wbv(theta)
            continue
        if ("trust" in guards and gn_window
                and gn > 100.0 * max(gn_window) and revert_streak < 10):
            stats["revert"] += 1
            revert_streak += 1
            if theta_prev is not None:
                theta = theta_prev
            if gt is not None:
                d = -gt
            if it in frames_at:
                snaps[it] = get_wbv(theta)
            continue
        revert_streak = 0
        gvar = (float(torch.linalg.norm(g - g_prev_raw)) / max(gn, 1e-300)
                if g_prev_raw is not None else float("inf"))
        g_prev_raw = g.clone()
        if gt is not None and ("gate" not in guards or gvar < 0.3):
            append_to_basis(gt)
        gt_new = project(g)
        if float(gt_new.dot(gt_new)) < 1e-24 * float(g.dot(g)):
            stats["exhaust"] += 1
            exhaust_streak += 1
            gt_new = g.clone()
            if exhaust_streak >= 20:
                basis = []
                stats["full_reset"] += 1
                exhaust_streak = 0
            beta = 0.0
        else:
            exhaust_streak = 0
            beta = (float(gt_new.dot(gt_new - gt)) / ggt
                    if (gt is not None and ggt > 0) else 0.0)
        d = -gt_new + beta * d if d is not None else -gt_new
        jd = f_jvp_S(theta, d, S)
        dn2 = float(d.dot(d))
        dhd = (2.0 / b) * float(jd.dot(jd))    # batch Gauss-Newton curvature
        mu = 0.0
        if "floor" in guards and lhat is not None and n_total:
            mu = lhat * (1.0 / b - 1.0 / n_total)
        denom = dhd + mu * dn2
        if denom > 0 and np.isfinite(denom):
            alpha = -float(g.dot(d)) / denom
            if dhd > 0:
                lam = dhd / dn2
                lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        theta_prev = theta
        theta = theta + alpha * d
        gn_window = (gn_window + [gn])[-10:]
        gt, ggt = gt_new, float(gt_new.dot(gt_new))
        losses.append(float(r_S.dot(r_S)) / b)  # batch MSE (diagnostic only)
        rels.append(eval_rel(theta))
        if it in frames_at:
            snaps[it] = get_wbv(theta)
    final_wbv = get_wbv(theta)
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, theta, stats


def train_slim_accum(theta0, f_resid_S, f_jvp_S, f_vjp_S, batch_iter,
                     n_batches_per_epoch, budget_batches, get_wbv,
                     eval_rel, n_total, fixed_k=None):
    """Stage 3: SNR-adaptive gradient ACCUMULATION.

    Rationale (lesson 7): below the 1/sqrt(b) batch-agreement scale the
    deep spectrum exists only in the exact operator, so only exact
    aggregation across batches makes progress -- no batch-local method
    (guards, SVRG, Adam) can cross that wall. This trainer aggregates the
    gradient over k consecutive batches per optimizer step, and the
    curvature along the new direction over the SAME batches (a second,
    JVP-only sweep), so both refer to the union quadratic. One monotone
    schedule rule: when the two half-accumulations disagree by more than
    their mean (SNR < 1), k doubles -- capped at one epoch, where the
    aggregate is the EXACT full-batch gradient (disjoint batches sum
    deterministically; no sqrt-averaging).

    Floor and gate retained (the stage-1 stable base): floor with the
    union size, mu = lhat*(1/|U| - 1/n), still zero when |U| = n.
    Budget is counted in batch-gradient evaluations (gradient sweep k,
    curvature sweep k/2), comparable with every other run.
    fixed_k: ablation control -- constant k (e.g. one full epoch from the
    first step), isolating aggregation from the schedule."""
    theta = theta0.clone()
    basis: list[torch.Tensor] = []
    losses, rels = [], []
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "k_final": 0, "k_history": [], "steps": 0}

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

    kmax = n_batches_per_epoch
    k = min(fixed_k, kmax) if fixed_k else min(2, kmax)
    spent = 0.0
    gt = None
    ggt = 0.0
    d = None
    lhat = None
    exhaust_streak = 0
    g_prev_raw = None
    theta_prev = None

    while spent < budget_batches:
        # ---- gradient sweep: aggregate J^T r over k batches ----
        acc = torch.zeros_like(theta)
        accA = torch.zeros_like(theta)
        batches_used = []
        nU = nA = 0
        half = (k + 1) // 2
        for j in range(k):
            S = next(batch_iter)
            b = S.numel()
            r_S = f_resid_S(theta, S)
            gS = f_vjp_S(theta, r_S, S)        # unscaled J_S^T r_S
            acc += gS
            nU += b
            if j < half:
                accA += gS
                nA += b
            batches_used.append(S)
        spent += k
        g = (2.0 / nU) * acc
        gn = float(torch.linalg.norm(g))
        if not np.isfinite(gn):
            stats["revert"] += 1               # NaN reject
            if theta_prev is not None:
                theta = theta_prev
            if gt is not None:
                d = -gt
            continue
        # ---- SNR schedule: halves must agree, else double k ----
        if fixed_k is None and k >= 2 and k < kmax:
            gA = (2.0 / nA) * accA
            gB = (2.0 / (nU - nA)) * (acc - accA)
            dis = float(torch.linalg.norm(gA - gB)) / max(gn, 1e-300)
            if dis > 1.0:
                k = min(2 * k, kmax)
        # ---- memory (gate on aggregated-gradient variability) ----
        gvar = (float(torch.linalg.norm(g - g_prev_raw)) / max(gn, 1e-300)
                if g_prev_raw is not None else float("inf"))
        g_prev_raw = g.clone()
        if gt is not None and gvar < 0.3:
            append_to_basis(gt)
        gt_new = project(g)
        if float(gt_new.dot(gt_new)) < 1e-24 * gn * gn:
            stats["exhaust"] += 1
            exhaust_streak += 1
            gt_new = g.clone()
            if exhaust_streak >= 20:
                basis = []
                stats["full_reset"] += 1
                exhaust_streak = 0
            beta = 0.0
        else:
            exhaust_streak = 0
            beta = (float(gt_new.dot(gt_new - gt)) / ggt
                    if (gt is not None and ggt > 0) else 0.0)
        d = -gt_new + beta * d if d is not None else -gt_new
        gt, ggt = gt_new, float(gt_new.dot(gt_new))
        # ---- curvature sweep along the NEW d, same batches ----
        c_acc = 0.0
        for S in batches_used:
            jd = f_jvp_S(theta, d, S)
            c_acc += float(jd.dot(jd))
        spent += 0.5 * k                       # JVP ~ half a gradient eval
        c = (2.0 / nU) * c_acc
        dn2 = float(d.dot(d))
        mu = (lhat or 0.0) * (1.0 / nU - 1.0 / n_total)
        denom = c + mu * dn2
        if denom > 0 and np.isfinite(denom):
            alpha = -float(g.dot(d)) / denom
            if c > 0:
                lam = c / dn2
                lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        theta_prev = theta
        theta = theta + alpha * d
        stats["steps"] += 1
        stats["k_history"].append(k)
        losses.append(float(g.dot(g)))         # diagnostic only
        rels.append(eval_rel(theta))
    stats["k_final"] = k
    stats["k_history"] = stats["k_history"][:8] + ["..."] \
        if len(stats["k_history"]) > 8 else stats["k_history"]
    return losses, rels, {}, theta, stats


def train_slim_carried_accum(theta0, f_resid_S, f_jvp_S, f_vjp_S,
                             epoch_batches, iters, get_wbv, eval_rel,
                             n_total, refresh=200):
    """Stage 4 ("carried"): FULL-BATCH SLIM COMPUTED IN BATCH-SIZED CHUNKS.

    At exact-epoch aggregation with disjoint batches, the union is the
    whole dataset, so the carried-residual recursion applies verbatim:
    per outer step, one JVP sweep over the epoch's batches computes
    u_S = J_S d and the exact curvature c = (2/n) sum ||u_S||^2; after the
    step the stored u_S advance the carried residual r_hat[S] += alpha u_S
    exactly; one VJP sweep at the new theta computes the exact gradient
    from r_hat. Mathematically identical to train_residual_cg_slim (up to
    floating-point summation order): batch size becomes a memory layout,
    not an information loss. The floor mu = lhat(1/n - 1/n) and the
    resample gate are identically zero/inert and omitted. Budget `iters`
    counts OUTER steps (each costs one epoch of JVP + one of VJP, the same
    FLOPs as a full-batch iteration).

    `epoch_batches`: a fixed disjoint partition of the data (list of index
    tensors). Returns (losses, rels, snaps, theta, stats)."""
    theta = theta0.clone()
    n = n_total
    rhat = torch.zeros(n, dtype=theta0.dtype)
    for S in epoch_batches:
        rhat[S] = f_resid_S(theta, S)

    def grad_at(th):
        acc = torch.zeros_like(theta)
        for S in epoch_batches:
            acc += f_vjp_S(th, (2.0 / n) * rhat[S], S)
        return acc

    g = grad_at(theta)
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0}

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

    gt = g.clone()
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    exhaust_streak = 0

    for it in range(1, iters + 1):
        if it % refresh == 0:
            for S in epoch_batches:            # frozen new noise draw
                rhat[S] = f_resid_S(theta, S)
            stats["refresh"] += 1
            g = grad_at(theta)
            gt = project(g)
            ggt = float(gt.dot(gt))
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break
        us = []                                # JVP sweep at theta
        c_acc = 0.0
        for S in epoch_batches:
            u_S = f_jvp_S(theta, d, S)
            us.append(u_S)
            c_acc += float(u_S.dot(u_S))
        dhd = (2.0 / n) * c_acc
        if dhd > 0 and np.isfinite(dhd):
            alpha = -float(g.dot(d)) / dhd
            lam = dhd / dn2
            lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        theta_new = theta + alpha * d
        rhat_new = rhat.clone()
        for S, u_S in zip(epoch_batches, us):
            rhat_new[S] = rhat[S] + alpha * u_S
        g_new = torch.zeros_like(theta)        # VJP sweep at theta_new
        for S in epoch_batches:
            g_new += f_vjp_S(theta_new, (2.0 / n) * rhat_new[S], S)
        if not np.isfinite(float(torch.linalg.norm(g_new))):
            stats["revert"] += 1               # sole safeguard: NaN reject
            d = -gt
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
        theta, rhat, g, gt = theta_new, rhat_new, g_new, gt_new
        ggt = float(gt.dot(gt))
        losses.append(float(rhat.dot(rhat)) / n)
        rels.append(eval_rel(theta))
    return losses, rels, snaps, theta, stats


# --------------------------------- one run ---------------------------------

def batch_case(job):
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    fn_name, N = job["fn"], job["N"]
    blabel, div, opt = job["blabel"], job["div"], job["opt"]
    f_vec = run.make_fn(run.TARGETS[fn_name])
    centers, gamma, _halo = run.geometry(N, run.LAM)
    W = centers.size
    theta0 = torch.tensor(np.concatenate([
        np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))

    # optional overrides (halo study): data domain + train count. Defaults
    # reproduce the original study exactly. Eval stays on [-1,1] always.
    lo, hi = job.get("domain", (-1.0, 1.0))
    n_tr = job.get("n_train", N_TRAIN_B)
    x_tr = np.linspace(lo, hi, n_tr)
    x_ev = np.linspace(-1, 1, run.N_EVAL)
    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(f_vec(x_tr))
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))
    ye_norm = float(torch.linalg.norm(ye))

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

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

    def eval_raw(th):
        with torch.no_grad():
            r = model(xe, th) - ye
            return float(torch.linalg.norm(r)) / ye_norm

    def wbv_raw(th):
        t = th.detach().numpy()
        return {"w": t[:W].copy(), "b": t[W:2 * W].copy(),
                "v": t[2 * W:3 * W].copy()}

    save_snaps = job.get("save_snaps", False)
    frames_at = (set(expd06.frame_steps(job["iters"])) if save_snaps
                 else {0})
    bsize = int(job.get("bsize", -(-n_tr // div)))     # ceil default
    batch_iter = make_batch_iter(n_tr, bsize, seed=0)
    t0 = time.time()

    if opt == "adam":                          # control: Adam, raw coords
        theta = theta0.clone().requires_grad_(True)
        optim = torch.optim.Adam([theta], lr=1e-3)
        rels, losses = [], []
        snaps = {}
        if 0 in frames_at:
            snaps[0] = wbv_raw(theta)
        stats = {}
        for it in range(1, job["iters"] + 1):
            S = next(batch_iter)
            optim.zero_grad(set_to_none=True)
            loss = ((model(xt[S], theta) - yt[S]) ** 2).mean()
            loss.backward()
            optim.step()
            losses.append(float(loss))
            rels.append(eval_raw(theta))
            if it in frames_at:
                snaps[it] = wbv_raw(theta)
    elif opt == "iter9":                       # iteration_9: no-B core,
        # ceiling-fixed tether + staged recycling; full batch = train_iter9,
        # batched = train_iter9b (all slow machinery on a fixed gauge subset)
        def f_resid9(th, idx=None):
            with torch.no_grad():
                if idx is None:
                    return (model(xt, th) - yt).detach()
                return (model(xt[idx], th) - yt[idx]).detach()

        def f_jvp9(th, dvec, idx=None):
            x_ = xt if idx is None else xt[idx]
            _, jd = torch.func.jvp(lambda t: model(x_, t), (th,), (dvec,))
            return jd.detach()

        def f_vjp9(th, seed, idx=None):
            x_ = xt if idx is None else xt[idx]
            thr = th.detach().requires_grad_(True)
            (gg,) = torch.autograd.grad(model(x_, thr), thr,
                                        grad_outputs=seed)
            return gg.detach()

        rng_p = np.random.default_rng(1)
        g_idx = torch.from_numpy(
            np.sort(rng_p.choice(n_tr, size=min(n_tr, 2003),
                                 replace=False)))

        def f_pred9(th):
            return model(xt[g_idx], th)

        recycler = (None if job.get("no_recycle") else
                    run.make_recycler(W, xt, w_cap=n_tr / 8.0))
        if div == 1:
            losses, rels, snaps, _, stats = run.train_iter9(
                theta0, f_resid9, f_jvp9, f_vjp9,
                lambda th: model(xt, th), n_tr, job["iters"], frames_at,
                wbv_raw, eval_raw, float(torch.linalg.norm(yt)),
                refresh=run.REFRESH, ratchet=job.get("ratchet", run.RATCHET),
                snap_S=job.get("snap_S9", False), recycle_fn=recycler)
        else:
            y_g = float(torch.linalg.norm(yt[g_idx]))
            losses, rels, snaps, _, stats = run.train_iter9b(
                theta0, f_resid9, f_jvp9, f_vjp9, f_pred9, n_tr,
                job["iters"], frames_at, wbv_raw, eval_raw, y_g,
                batch_iter, g_idx, refresh=run.REFRESH,
                ratchet=job.get("ratchet", run.RATCHET),
                snap_S=job.get("snap_S9", False), recycle_fn=recycler)
    elif opt == "iter8":                       # iteration_8: auto-tether,
        # raw coordinates, one code path for any batch size (full batch =
        # rows persist; the optimizer never branches on batch character)
        def f_resid8(th, idx=None):
            with torch.no_grad():
                if idx is None:
                    return (model(xt, th) - yt).detach()
                return (model(xt[idx], th) - yt[idx]).detach()

        def f_jvp8(th, dvec, idx=None):
            x_ = xt if idx is None else xt[idx]
            _, jd = torch.func.jvp(lambda t: model(x_, t), (th,), (dvec,))
            return jd.detach()

        def f_vjp8(th, seed, idx=None):
            x_ = xt if idx is None else xt[idx]
            thr = th.detach().requires_grad_(True)
            (gg,) = torch.autograd.grad(model(x_, thr), thr,
                                        grad_outputs=seed)
            return gg.detach()

        # probe on a fixed <=2003-point subsample: nonlinearity detection
        # does not need the full pool, and the dense probe would otherwise
        # dominate batched-run cost
        rng_p = np.random.default_rng(1)
        p_idx = torch.from_numpy(
            np.sort(rng_p.choice(n_tr, size=min(n_tr, 2003),
                                 replace=False)))

        def f_pred8(th):
            return model(xt[p_idx], th)

        losses, rels, snaps, _, stats = run.train_iter8(
            theta0, f_resid8, f_jvp8, f_vjp8, f_pred8, n_tr,
            job["iters"], frames_at, wbv_raw, eval_raw,
            float(torch.linalg.norm(yt)),
            batch_iter=(None if div == 1 else batch_iter),
            refresh=run.REFRESH, ema=job.get("ema", 0.9),
            snap_S=job.get("snap_S8", False),
            ablate=frozenset(job.get("ablate8", ())))
    else:                                      # slim
        scale = torch.ones(3 * W + 1, dtype=torch.float64)
        scale[:2 * W] = run.TETHER

        def p_eval(phi):
            return eval_raw(phi / scale)

        def p_wbv(phi):
            return wbv_raw(phi / scale)

        if div == 1:                           # full batch: carried residual
            def p_resid(phi):
                with torch.no_grad():
                    return (model(xt, phi / scale) - yt).detach()

            def p_jvp(phi, dvec):
                _, jd = torch.func.jvp(lambda th: model(xt, th),
                                       (phi / scale,), (dvec / scale,))
                return jd.detach()

            def p_vjp(phi, seed):
                thr = (phi / scale).detach().requires_grad_(True)
                (gg,) = torch.autograd.grad(model(xt, thr), thr,
                                            grad_outputs=seed)
                return gg.detach() / scale

            losses, rels, snaps, _, stats = run.train_residual_cg_slim(
                theta0 * scale, p_resid, p_jvp, p_vjp, n_tr,
                job["iters"], frames_at, p_wbv, p_eval, refresh=run.REFRESH)
        else:                                  # minibatch: fresh per visit
            def p_resid_S(phi, S):
                with torch.no_grad():
                    return (model(xt[S], phi / scale) - yt[S]).detach()

            def p_jvp_S(phi, dvec, S):
                _, jd = torch.func.jvp(lambda th: model(xt[S], th),
                                       (phi / scale,), (dvec / scale,))
                return jd.detach()

            def p_vjp_S(phi, seed, S):
                thr = (phi / scale).detach().requires_grad_(True)
                (gg,) = torch.autograd.grad(model(xt[S], thr), thr,
                                            grad_outputs=seed)
                return gg.detach() / scale

            def p_resid_full(phi):
                with torch.no_grad():
                    return (model(xt, phi / scale) - yt).detach()

            def p_vjp_full(phi, seed):
                thr = (phi / scale).detach().requires_grad_(True)
                (gg,) = torch.autograd.grad(model(xt, thr), thr,
                                            grad_outputs=seed)
                return gg.detach() / scale

            guards_ = frozenset(job.get("guards", ()))
            if "carried" in guards_:
                rng2 = np.random.default_rng(0)
                perm2 = rng2.permutation(n_tr)
                epoch_batches = [
                    torch.from_numpy(perm2[i:i + bsize].copy())
                    for i in range(0, n_tr, bsize)]
                losses, rels, snaps, _, stats = train_slim_carried_accum(
                    theta0 * scale, p_resid_S, p_jvp_S, p_vjp_S,
                    epoch_batches, job["iters"], p_wbv, p_eval, n_tr,
                    refresh=run.REFRESH)
            elif "accum" in guards_ or "accum_fixed" in guards_:
                n_bpe = -(-n_tr // bsize)
                losses, rels, snaps, _, stats = train_slim_accum(
                    theta0 * scale, p_resid_S, p_jvp_S, p_vjp_S,
                    batch_iter, n_bpe, job["iters"], p_wbv, p_eval,
                    n_tr,
                    fixed_k=(n_bpe if "accum_fixed" in guards_ else None))
            else:
                losses, rels, snaps, _, stats = train_slim_fresh_batches(
                    theta0 * scale, p_resid_S, p_jvp_S, p_vjp_S, batch_iter,
                    job["iters"], frames_at, p_wbv, p_eval,
                    guards=job.get("guards", ()), n_total=n_tr,
                    refresh=run.REFRESH, f_resid_full=p_resid_full,
                    f_vjp_full=p_vjp_full)

    out = {"fn": fn_name, "N": N, "blabel": blabel, "bsize": bsize,
           "opt": opt, "guards": sorted(job.get("guards", ())),
           "floor": floor, "rels": rels,
           "best_rel": float(np.nanmin(rels)) if rels else float("nan"),
           "stats": stats, "wall_s": round(time.time() - t0, 1),
           "domain": [lo, hi], "n_train": n_tr}
    if save_snaps:
        snap_dir = Path(job.get("snap_dir", out_dir(opt) / "snaps"))
        snap_dir.mkdir(parents=True, exist_ok=True)
        ssteps = sorted(snaps)
        np.savez_compressed(
            snap_dir / f"{fn_name}--N{N}--{_tag(blabel)}.npz",
            steps=np.array(ssteps, dtype=np.int64),
            w=np.stack([snaps[s]["w"] for s in ssteps]),
            b=np.stack([snaps[s]["b"] for s in ssteps]),
            v=np.stack([snaps[s]["v"] for s in ssteps]),
            rels=np.array(rels), floor=floor, stats=json.dumps(stats))
        out["snaps"] = {str(k): {kk: vv.tolist() for kk, vv in s_.items()}
                        for k, s_ in snaps.items()}
    return out


def load_rows(opt):
    rows = json.loads((out_dir(opt) / "batching_rows.json").read_text())
    for path in sorted((out_dir(opt) / "snaps").glob("*.npz")):
        fn_name, n_part, tag = path.stem.split("--")
        blabel = tag.replace("d", "/", 1) if tag != "B" else "B"
        z = np.load(path)
        for r in rows:
            if (r["fn"] == fn_name and r["N"] == int(n_part[1:])
                    and r["blabel"] == blabel):
                r["snaps"] = {str(int(s)): {"w": z["w"][i].tolist(),
                                            "b": z["b"][i].tolist(),
                                            "v": z["v"][i].tolist()}
                              for i, s in enumerate(z["steps"])}
    return rows


# --------------------------------- figures ---------------------------------

def render_gif_batch(rows, opt, fn_name, path, fps=8):
    """4 rows = batch sizes (B, B/4, B/16, B/64), cols = (w | b | v) vs the
    CURRENT effective centers -b_k/w_k; one target fn at N=256. Y-limits
    are LOCKED to the theory regime (identical across rows, so rows compare
    directly); each panel is annotated with the standard deviation and the
    largest absolute value of its coefficients over ALL neurons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    XW = 2.0
    cases = {r["blabel"]: r for r in rows
             if r["fn"] == fn_name and r["N"] == GIF_N
             and r["blabel"] in GIF_BLEVELS and "snaps" in r}
    labels = [b for b in GIF_BLEVELS if b in cases]
    if not labels:
        print(f"  no snaps for {opt}/{fn_name}; skipping gif")
        return
    steps = sorted(int(s) for s in cases[labels[0]]["snaps"])
    total = steps[-1]
    h = 2.0 / GIF_N
    gamma = run.LAM / h
    cgrid = np.linspace(-1, 1, 401)
    overlay_v = (h / 2.0) * expd06.fprime(run.TARGETS[fn_name])(cgrid)

    rbcmap = LinearSegmentedColormap.from_list("redblue", ["red", "blue"])
    ncolors = {}
    for lb in labels:
        sn0 = cases[lb]["snaps"][str(steps[0])]
        w0, b0 = np.array(sn0["w"]), np.array(sn0["b"])
        c0 = -b0 / np.where(np.abs(w0) > 1e-12, w0, 1e-12)
        ranks = np.argsort(np.argsort(c0))
        ncolors[lb] = rbcmap(ranks / max(len(c0) - 1, 1))
    A = float(np.abs(overlay_v).max()) or 1e-3
    col_lims = [(0.0, 2.0 * gamma),
                (-1.6 * gamma, 1.6 * gamma),
                (float(overlay_v.min()) - 3.0 * A,
                 float(overlay_v.max()) + 3.0 * A)]

    images = []
    for s in steps:
        fig, axes = plt.subplots(len(labels), 3,
                                 figsize=(12.5, 2.5 * len(labels)),
                                 squeeze=False)
        for i, lb in enumerate(labels):
            sn = cases[lb]["snaps"][str(s)]
            w_, b_, v_ = (np.array(sn[k]) for k in ("w", "b", "v"))
            c = -b_ / np.where(np.abs(w_) > 1e-12, w_, 1e-12)
            cols = ncolors[lb]
            vis = np.abs(c) <= XW
            for j, (y, ttl) in enumerate([(w_, "first-layer w"),
                                          (b_, "bias b"),
                                          (v_, "readout v")]):
                ax = axes[i, j]
                ax.axvspan(-1, 1, color="#1f4e8c", alpha=0.05)
                if j == 0:
                    ax.axhline(gamma, color="#2ca02c", lw=1.0, ls="--",
                               alpha=0.8)
                elif j == 1:
                    ax.plot(cgrid, -gamma * cgrid, color="#2ca02c", lw=1.0,
                            ls="--", alpha=0.8)
                else:
                    ax.plot(cgrid, overlay_v, color="#2ca02c", lw=1.0,
                            ls="--", alpha=0.8)
                ax.scatter(c[vis], y[vis], c=cols[vis], s=9,
                           edgecolors="none", zorder=3)
                ax.set_xlim(-XW, XW)
                ax.set_ylim(*col_lims[j])
                try:
                    ax.ticklabel_format(useOffset=False, style="plain",
                                        axis="y")
                except (AttributeError, ValueError):
                    pass
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(ttl + "  (green = theory)", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"batch {lb}", fontsize=8)
                ax.text(0.02, 0.96,
                        f"$\\sigma$={np.std(y):.2g}  "
                        f"max$|{'wbv'[j]}|$={np.abs(y).max():.2g}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6, color="#333",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
        fig.suptitle(f"{opt}   {fn_name}   N={GIF_N}   batching (epoch "
                     f"reshuffle, n={N_TRAIN_B})   step {s}/{total}"
                     r"   (x = current centers $-b_k/w_k$)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
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


def render_batch_scaling(rows, opt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {64: "tab:blue", 128: "tab:orange", 256: "tab:green",
              512: "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, fn in zip(axes.flat, FNS):
        for N in WIDTHS:
            rs = {r["blabel"]: r for r in rows
                  if r["fn"] == fn and r["N"] == N}
            bs = [rs[lb]["bsize"] for lb, _ in BLEVELS if lb in rs]
            ys = [rs[lb]["best_rel"] for lb, _ in BLEVELS if lb in rs]
            if not bs:
                continue
            ax.loglog(bs, ys, "-o", color=colors[N], ms=4, label=f"N={N}")
            ax.axhline(list(rs.values())[0]["floor"], color=colors[N],
                       ls=":", lw=0.9, alpha=0.6)
        ax.set_ylim(1e-16, 2.0)
        ax.set_title(fn, fontsize=11)
        ax.set_xlabel("batch size")
        ax.set_ylabel("best eval rel $L_2$")
        ax.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=4, frameon=False)
    fig.suptitle(f"expD08 batching -- {opt}, {ITERS} steps, epoch "
                 f"reshuffle, n={N_TRAIN_B} (dotted = per-N lstsq floor)",
                 y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_comparison(path):
    """Overlay: slim (solid) vs adam (dashed), per fn, lines per N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    try:
        rows = {o: json.loads((out_dir(o) / "batching_rows.json").read_text())
                for o in ("slim", "adam")}
    except FileNotFoundError:
        return
    colors = {64: "tab:blue", 128: "tab:orange", 256: "tab:green",
              512: "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, fn in zip(axes.flat, FNS):
        for o, ls in (("slim", "-"), ("adam", "--")):
            for N in WIDTHS:
                rs = {r["blabel"]: r for r in rows[o]
                      if r["fn"] == fn and r["N"] == N}
                bs = [rs[lb]["bsize"] for lb, _ in BLEVELS if lb in rs]
                ys = [rs[lb]["best_rel"] for lb, _ in BLEVELS if lb in rs]
                if bs:
                    ax.loglog(bs, ys, ls, marker="o" if o == "slim" else "x",
                              color=colors[N], ms=4, lw=1.2)
        ax.set_ylim(1e-16, 2.0)
        ax.set_title(fn, fontsize=11)
        ax.set_xlabel("batch size")
        ax.set_ylabel("best eval rel $L_2$")
        ax.grid(True, which="both", alpha=0.25)
    handles = ([Line2D([0], [0], color=colors[N], lw=1.5, label=f"N={N}")
                for N in WIDTHS]
               + [Line2D([0], [0], color="k", ls="-", marker="o", ms=4,
                         label="slim"),
                  Line2D([0], [0], color="k", ls="--", marker="x", ms=4,
                         label="adam")])
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=6, frameon=False)
    fig.suptitle(f"expD08 batching -- slim vs adam, {ITERS} steps, epoch "
                 f"reshuffle, n={N_TRAIN_B}", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="both", choices=["slim", "adam", "both"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--ablate", action="store_true",
                    help="hardened-minibatch ablation: one guard at a time "
                         "then combos, on 4 fns x N in {64,256} x "
                         "{B/4,B/16,B/64}; baseline = existing slim rows")
    ap.add_argument("--svrg", action="store_true",
                    help="stage 2: SVRG variants {svrg, svrg+gate, "
                         "svrg+floor+gate} on the ablation grid")
    ap.add_argument("--accum", action="store_true",
                    help="stage 3: SNR-adaptive accumulation {accum, "
                         "accum_fixed} on the ablation grid")
    ap.add_argument("--carried", action="store_true",
                    help="stage 4: full-batch slim computed in batch-sized "
                         "chunks (carried residual at exact-epoch "
                         "aggregation); budget = 3000 OUTER steps")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    opts = ["slim", "adam"] if args.opt == "both" else [args.opt]

    if args.ablate or args.svrg or args.accum or args.carried:
        if args.carried:
            variants = [("carried",)]
        elif args.accum:
            variants = [("accum",), ("accum_fixed",)]
        elif args.svrg:
            variants = [("svrg",), ("svrg", "gate"), ("svrg", "floor", "gate")]
        else:
            variants = [("floor",), ("trust",), ("gate",),
                        ("floor", "trust"), ("floor", "trust", "gate")]
        threads = max(1, (os.cpu_count() or 4) // args.workers)
        jobs = [{"fn": f, "N": n, "blabel": lb, "div": dv,
                 "iters": ITERS, "threads": threads, "opt": "slim",
                 "guards": list(v)}
                for v in variants
                for lb, dv in BLEVELS if lb in ("B/4", "B/16", "B/64")
                for f in FNS for n in (64, 256)]
        print(f"expD08 batching ablation | {len(jobs)} runs | "
              f"variants={['+'.join(v) for v in variants]}")
        results, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(batch_case, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                s = r["stats"]
                print(f"  [{i}/{len(jobs)}] {'+'.join(r['guards']):16s} "
                      f"{r['fn']:9s} N={r['N']:<4d} {r['blabel']:5s} "
                      f"best_rel={r['best_rel']:.3e} "
                      f"rev={s.get('revert', 0)} fb={s.get('alpha_fallback', 0)} "
                      f"({r['wall_s']}s)", flush=True)
        print(f"ablation done in {time.time()-t0:.1f}s")
        out_name = ("carried/carried_rows.json" if args.carried
                    else "accum_rows.json" if args.accum
                    else "svrg_rows.json" if args.svrg
                    else "ablations_rows.json")
        (BATCH_ROOT / out_name).parent.mkdir(parents=True, exist_ok=True)
        (BATCH_ROOT / out_name).write_text(json.dumps(
            [{k: r[k] for k in ("fn", "N", "blabel", "bsize", "guards",
                                "floor", "best_rel", "rels")}
             for r in results]))
        # summary vs the unguarded slim baseline and the adam bar
        base = {(r["fn"], r["N"], r["blabel"]): r["best_rel"]
                for r in json.loads(
                    (out_dir("slim") / "batching_rows.json").read_text())}
        adam = {(r["fn"], r["N"], r["blabel"]): r["best_rel"]
                for r in json.loads(
                    (out_dir("adam") / "batching_rows.json").read_text())}
        print(f"\n{'variant':<18s} {'med vs slim-base':>17s} "
              f"{'med vs adam':>12s} {'best cell':>26s}")
        for v in variants:
            rs = [r for r in results if r["guards"] == sorted(v)]
            dv_ = [np.log10(r["best_rel"]
                            / base[(r["fn"], r["N"], r["blabel"])])
                   for r in rs]
            da = [np.log10(r["best_rel"]
                           / adam[(r["fn"], r["N"], r["blabel"])])
                  for r in rs]
            bi = int(np.argmin([r["best_rel"] for r in rs]))
            print(f"{'+'.join(v):<18s} {float(np.median(dv_)):>+17.2f} "
                  f"{float(np.median(da)):>+12.2f} "
                  f"{rs[bi]['fn']}@N{rs[bi]['N']} {rs[bi]['blabel']}: "
                  f"{rs[bi]['best_rel']:.1e}")
        return

    if args.render_only:
        for o in opts:
            rows = load_rows(o)
            render_batch_scaling(rows, o, out_dir(o) / "batch_scaling_2x2.png")
            if not args.no_gifs:
                for fn in FNS:
                    render_gif_batch(rows, o, fn,
                                     out_dir(o) / "gifs"
                                     / f"params_{fn}_N256.gif")
        render_comparison(BATCH_ROOT / "comparison_2x2.png")
        return

    levels = BLEVELS if not args.smoke else BLEVELS[:2]
    fns = FNS if not args.smoke else ["sine"]
    widths = WIDTHS if not args.smoke else [64]
    iters = ITERS if not args.smoke else 120
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    jobs = [{"fn": f, "N": n, "blabel": lb, "div": dv, "iters": iters,
             "threads": threads, "opt": o, "save_snaps": (n == GIF_N)}
            for o in opts for lb, dv in levels for f in fns for n in widths]
    print(f"expD08 batching | {len(jobs)} runs | opts={opts} | iters={iters}"
          f" | n={N_TRAIN_B} | levels={[lb for lb, _ in levels]}")

    rows_by_opt = {o: [] for o in opts}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(batch_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows_by_opt[r["opt"]].append(r)
            s = r["stats"]
            print(f"  [{i}/{len(jobs)}] {r['opt']:5s} {r['fn']:13s} "
                  f"N={r['N']:<4d} {r['blabel']:6s} (b={r['bsize']:5d}) "
                  f"best_rel={r['best_rel']:.3e} floor={r['floor']:.1e} "
                  f"exh={s.get('exhaust', 0)} rst={s.get('full_reset', 0)} "
                  f"fb={s.get('alpha_fallback', 0)} rev={s.get('revert', 0)} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"training done in {time.time()-t0:.1f}s")

    for o in opts:
        od = out_dir(o)
        od.mkdir(parents=True, exist_ok=True)
        slim_rows = [{k: r[k] for k in ("fn", "N", "blabel", "bsize", "opt",
                                        "floor", "best_rel", "rels")}
                     for r in rows_by_opt[o]]
        (od / "batching_rows.json").write_text(json.dumps(slim_rows))
        render_batch_scaling(rows_by_opt[o], o, od / "batch_scaling_2x2.png")
        if not args.no_gifs:
            for fn in (FNS if not args.smoke else fns):
                render_gif_batch(rows_by_opt[o], o, fn,
                                 od / "gifs" / f"params_{fn}_N256.gif")
    render_comparison(BATCH_ROOT / "comparison_2x2.png")
    print(f"done; outputs under {BATCH_ROOT}")


if __name__ == "__main__":
    main()
