"""expD08 -- QI geometry + zero readout, trained by the expD07 winner.

The convergence experiment the expD07 suite was built to spec:
  init      = expD06/seed9_zero_readout: QI geometry in affine coordinates
              (w = gamma, b = -gamma*c, uniform centers + halo, lambda*=0.25),
              readout v = 0, bias c0 = 0 -- and EVERYTHING trainable.
  optimizer = expD07's winner: nonlinear CG (PR+, Armijo line search with
              quadratic-interpolation trial step) with FULL orthogonalized
              gradient memory (the budget-scaling study: window >= rank
              reaches the SVD floor; smaller windows fail). Basis resets on
              exhaustion or line-search failure (nonquadratic drift).
  grid      = N in {32, 64, 128, 256, 512} x the 6-function family = 30 runs,
              1000 NLCG iterations each, fp64 full batch.

Outputs (format of expD06/seed10):
  gifs/params_N{128,256,512}.gif   3-column scatter parameter movies
                                   (w | b | v vs current centers, theory green)
  loss_curves_2x2_N256.png         train MSE + eval rel L2, 4 gif targets
  scaling_2x3.png                  min eval rel L2 vs N, all 6 functions,
                                   lstsq floor dashed (fixed 1e-16..1 scale)

Run:  uv run python experiments/expD08_qi_init_nlcg/run.py   (--smoke)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expA06_readout_structure"))
from app import build_phi, geometry, make_fn  # noqa: E402

# reuse expD06's rendering + frame schedule verbatim (Sam: keep that format)
_spec = importlib.util.spec_from_file_location(
    "expd06_run", REPO_ROOT / "experiments" / "expD06_derivative_readout_init" / "run.py")
expd06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(expd06)

# iteration_1: TETHER=1 (plain metric), ITERS=1000 -- train==eval stalled
#              ~1e-7 rel L2 (no generalization gap; pure optimizer stall).
# iteration_2: geometry TETHER T=1e3 + ITERS=3000 -- wall dropped to
#              5e-9..7e-10 rel, flat from ~iter 150 (a wall, not a crawl).
# iteration_3: ORTHOGONALIZED SECANT-CG -- every control signal comes from
#              GRADIENTS, none from loss values (loss-value comparisons go
#              blind at rel ~1e-12; gradients are linear in the residual and
#              stay informative to rel ~1e-16 = the lstsq floor):
#                * secant curvature step alpha = -g.d / (d.(g(x+td)-g(x))/t)
#                  with the probe at the previous accepted step (2 grad
#                  evals/iteration, exact on quadratics);
#                * gradient-norm trust window instead of Armijo;
#                * basis KEPT on exhaustion (raw-gradient step addresses
#                  reopened directions; full reset only after a 20-streak);
#                * instrumented: exhaust/reset/fallback/revert counters.
#              TETHER=1e6 (wall level tracked T from 1 -> 1e3; push harder).
# iteration_4: RESIDUAL-CARRYING CG. The T=1e9 probe showed the tether is
#              exhausted; the discriminating test showed the remaining 3-4
#              orders come from recomputing r = model(x)-y FRESH each
#              iteration (1e-16-abs cancellation noise in every gradient:
#              fresh 2.2e-10 vs recursive 5.6e-14 on the same frozen
#              system). Fix = the CGLS trick in autograd form:
#                * carry r_hat across iterations: r_hat += alpha * J d,
#                  with J d from one forward-mode JVP;
#                * gradient = VJP with seed (2/n) r_hat (one backward);
#                * curvature = Gauss-Newton (2/n)||J d||^2 from the SAME
#                  JVP -- exact for the linear-in-readout structure, no
#                  probe needed;
#                * refresh r_hat from a true forward pass every 200 its
#                  (freezes one noise draw instead of re-sampling noise
#                  every iteration; basis/direction NOT touched).
#              Still: no loss comparisons, ~2 autograd passes/iteration.
# iteration_5: SLIM. The hardening ablation (hardening.py; hardening/*.json)
#              measured three iteration_4 safeguards as free or harmful and
#              they are deleted: the gradient-norm trust window (removing it
#              IMPROVED the median by 0.36 orders), the uphill-direction
#              reset, and the PR+ beta clamp. Deletions verified in
#              combination ("slim" combo: matched/beat iteration_4 on 7 of 8
#              cells). Remaining: carried residual + JVP exact steps + full
#              orthogonal memory (double projection) + exhaustion escape +
#              refresh + a one-line non-finite reject. Same tether (T=1 with
#              the final optimizer loses ~7 orders median -- measured, and
#              not a safeguard-lockout artifact; replacing the tether with a
#              data-derived metric is the next hardening step).
# iteration_6: CONSOLIDATION. The slim core + the v4 AUTO-TETHER built in
#              (architecture-blind: per-coordinate nonlinearity probe +
#              drift-ratcheted continuous metric S_i = 1 + T q_i/max q,
#              monotone, gauge guarded by its own noise floor) + the
#              batched-mode guards in their inert full-batch forms (the
#              curvature floor mu = lhat*(1/b - 1/n) is identically zero at
#              b = n; the memory gate is defined on batch-resample noise,
#              zero in deterministic full batch). Design identity (Sam):
#              far from the basin it behaves like an ordinary first-order
#              method (T = 1, geometry free); as it enters the basin it
#              locks the nonlinear coordinates and becomes the conjugate-
#              gradient machine. Rejected on the way here (measured):
#              binary-mask tethers (v1-v3, threshold boundary), isotropic
#              per-step damping (v5, strangles the readout with the
#              geometry). See results/.../tether/tether.md.
# iteration_8: CONTINUOUS EVERYTHING (spec: iteration_8/fable_thoughts.md
#              rev 4). All discrete control is gone -- no trips, no
#              ratchet, no basis resets, no deadbands, no read cadences,
#              no batch-mode branches. Two per-step EMA signals drive
#              every adaptation continuously:
#                Dbar = EMA(||r_pred - r_fresh|| / ||r||)  (bend; one
#                       read-only forward per step, same rows -- the
#                       carried residual is REPLACED only on the 200
#                       timer, so the noise draw stays frozen)
#                rhobar = EMA(gradient-space surprise: |g_new.d - (g.d
#                       + alpha c)| / (||g_new|| ||d||))  (trust; free.
#                       Zero on any quadratic even under damping; large
#                       under batch switching or nonlinearity)
#              Consumers (all continuous, all self-inert when trusted):
#                tether  S_i = 1 + T qhat_i / max qhat with the
#                        MEMORYLESS map T = clip(Dbar/10eps, 1, 1e12);
#                        closed loop: T slows geometry by S^2, bend
#                        falls, Dbar falls, T relaxes (EMA w=0.9 makes
#                        the loop contraction ~0.5). qhat EMA-blended
#                        (0.5) across dense probes at it=200,1200,2200,
#                        resolution-floored so the readout can never be
#                        tethered at any T. Implemented as a diagonal
#                        PRECONDITIONER p = S^-2 in raw theta space --
#                        no phi coordinates, no basis transport ever.
#                damping mu = lhat (1/b - 1/n + rhobar): instantaneous
#                        structural floor (batch rank <= b) + measured
#                        LM-style caution; both terms -> 0 smoothly on a
#                        trusted deterministic full batch.
#                beta    (1 - rhobar) beta_PR + rhobar 0.9: one formula
#                        from CG (trusted) to heavy ball (chaos).
#                B       kept; admission PACED by trust (a += 1-rhobar,
#                        admit at a>=1): full trust = admit every step
#                        (iter7 behavior recovered), chaos = never. The
#                        fp-zero escape stays (numerical existence); the
#                        20-streak reset (an event) is deleted.
#              One code path for any batch size: full batch is the
#              length-1 case of the batch iterator. Residual state is
#              row-bound (cache validity, not a mode).
# iteration_7: STAIRCASE FIX. Two constants changed after the staircase
#              study (iteration_6/staircase/): trip threshold = the
#              gauge's own noise floor (trips continue until geometry
#              damage reaches rounding, removing the drift-equilibrium
#              cap that stranded runs ~2 orders above the floor), and
#              ratchet = 1e4 (locking T reached in ~2 trips: fewer basis
#              resets, shorter staircase). Measured, 12-cell slice:
#              median -0.04 vs iteration_6 with floor-level wins at
#              N=512. Rejected on the way (measured): carrying the basis
#              across metric changes (+1.07 median -- the solved space
#              is NOT transportable across a metric change), and
#              event-driven recalibration (+0.18 -- each refresh re-rolls
#              the carried residual's frozen noise draw, and the
#              exhaustion signal over-fires; the 200-step timer is a
#              principled RATE LIMITER on noise re-rolls, not a hack).
# iteration_9: B REMOVED (<10-vector state). Phase-0 bake-off verdict
#              (iteration_9/b_replacement/, 23 arms): B's floor value is
#              init-relative; no O(m) mechanism substitutes from cold
#              states (precision/function-space/replay all fail lawfully;
#              exchange rate = spectral density). Tether = iteration_7's
#              trip/ratchet with a T ceiling (1e12) and a probe-resolution
#              floor (readout untetherable at any T); trips are cheap (no
#              basis to void — direction reset only). Cold-start depth via
#              STAGED RECYCLING (iteration_9/staged_gate/): at a measured
#              stall, absorb lowest-utilization units into the bias,
#              re-seed them sharper, train only that group (boosting via
#              readout linearity, no target arithmetic). recycle_fn is an
#              architecture adapter closure (same status as get_wbv).
ITERATION = 9
TRIP_MODE = "noise"            # legacy (iteration_6/7 core only)
TETHER = 1e6                    # legacy fixed tether (core="slim"/"v4" only)
ITERS = 3000
REFRESH = 200
OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / f"iteration_{ITERATION}")
LAM = 0.25
WIDTHS = [32, 64, 128, 256, 512]
GIF_WIDTHS = [128, 256, 512]
N_TRAIN, N_EVAL = 2003, 4001
RCOND = 1e-15
TARGETS = expd06.TARGETS          # canonical 6
TARGETS3 = expd06.TARGETS3        # the 4 gif/loss-curve targets
SEED_NAME = "qi_zero_readout_nlcg_ortho"


# ------------------------- optimizer (expD07 winner) -------------------------

def train_secant_cg_ortho(theta0, f_grad, iters, frames_at, get_wbv,
                          eval_rel, f_hvp=None):
    """Orthogonalized secant-CG: CG with full orthogonalized gradient
    memory, where EVERY control signal comes from gradients -- no loss
    values are ever compared (loss comparisons go blind at rel ~1e-12;
    gradients are linear in the residual and stay informative to ~1e-16).

      step:      secant curvature over the trial segment,
                 dHd ~ d.(g(x + a_tr d) - g(x)) / a_tr with a_tr = the
                 previous accepted step (max SNR; exact on quadratics),
                 alpha = -g.d / dHd. 2 grad evals per iteration.
      safeguard: gradient-norm trust window (10x over last-10 -> revert
                 step, shrink trial, raise curvature estimate). No Armijo.
      memory:    new gradient projected against the orthonormal basis of
                 past projected gradients (cgls_reortho mechanism). On
                 exhaustion (projection eats the gradient: solved-span
                 error reopened, or done) the basis is KEPT -- one raw-
                 gradient step addresses the reopened directions -- and
                 only a 20-streak of exhaustions triggers a full reset.

    Returns (losses, rels, snaps, theta, stats).
    """
    theta = theta0.clone()
    loss, g = f_grad(theta)
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0}

    def project(vec):
        if not basis:
            return vec.clone()
        B = torch.stack(basis)
        out = vec - B.t().mv(B.mv(vec))
        return out - B.t().mv(B.mv(out))

    def append_to_basis(vec):
        """Orthogonalize against the basis before appending, so the basis
        stays orthonormal even when raw (unprojected) gradients are used."""
        v = project(vec)
        n2 = float(v.dot(v))
        if n2 > 1e-300 and len(basis) < theta.numel():
            basis.append(v / float(np.sqrt(n2)))

    if 0 in frames_at:
        snaps[0] = get_wbv(theta)
    gt = project(g)
    if float(gt.dot(gt)) < 1e-6 * float(g.dot(g)):
        gt = g.clone()
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    alpha_prev = None
    exhaust_streak = 0
    revert_streak = 0
    gn_window = [float(torch.linalg.norm(g))]

    for it in range(1, iters + 1):
        dphi0 = float(g.dot(d))
        if dphi0 >= 0.0:
            d, dphi0 = -gt, -ggt
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break                              # gradient fully exhausted
        # --- secant curvature over the trial segment ---
        if alpha_prev is not None:
            a_tr = alpha_prev
        elif lhat is not None:
            a_tr = 1.0 / lhat
        else:
            a_tr = 1.0 / (1.0 + np.sqrt(ggt))
        a_tr = float(np.clip(a_tr, 1e-30, 1e12))
        if f_hvp is not None:
            # exact curvature via Pearlmutter Hvp: one extra backprop,
            # linear-in-residual precision, no loss values involved
            dhd = float(d.dot(f_hvp(theta, d)))
        else:
            _, g_probe = f_grad(theta + a_tr * d)
            dhd = float(d.dot(g_probe - g)) / a_tr
            if not (dhd > 0 and np.isfinite(dhd)):
                # probe too small to resolve tail curvature -- retry larger
                a_big = a_tr * 1e4
                _, g_probe = f_grad(theta + a_big * d)
                dhd = float(d.dot(g_probe - g)) / a_big
        if dhd > 0 and np.isfinite(dhd):
            alpha = -dphi0 / dhd
            lam = dhd / dn2
            lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = a_tr                       # take the trial step itself
        theta_new = theta + alpha * d
        loss_new, g_new = f_grad(theta_new)
        gn = float(torch.linalg.norm(g_new))
        # --- gradient-norm trust window (NO loss comparisons). Wide (100x)
        # because CG transients legitimately raise ||g||; an escape hatch
        # accepts after 10 consecutive rejections so the shrink spiral can
        # never lock the run out (observed on sine_mixture N<=64). ---
        if (not np.isfinite(gn)) or (gn > 100.0 * max(gn_window)
                                     and revert_streak < 10):
            stats["revert"] += 1
            revert_streak += 1
            alpha_prev = 0.25 * alpha
            lhat = None if lhat is None else lhat * 4.0
            d = -gt                            # retry steepest next iter
            if it in frames_at:
                snaps[it] = get_wbv(theta)     # keep the frame schedule
            continue
        revert_streak = 0
        # --- memory update ---
        append_to_basis(gt)
        gt_new = project(g_new)
        if float(gt_new.dot(gt_new)) < 1e-6 * float(g_new.dot(g_new)):
            stats["exhaust"] += 1
            exhaust_streak += 1
            gt_new = g_new.clone()             # raw step; basis KEPT
            if exhaust_streak >= 20:
                basis = []
                stats["full_reset"] += 1
                exhaust_streak = 0
            beta = 0.0
        else:
            exhaust_streak = 0
            beta = (max(0.0, float(gt_new.dot(gt_new - gt)) / ggt)
                    if ggt > 0 else 0.0)
        d = -gt_new + beta * d
        theta, loss, g, gt = theta_new, loss_new, g_new, gt_new
        ggt = float(gt.dot(gt))
        alpha_prev = alpha
        gn_window = (gn_window + [gn])[-10:]
        losses.append(float(loss))
        rels.append(eval_rel(theta))
        if it in frames_at:
            snaps[it] = get_wbv(theta)
    # pad any frames not reached (early break / reverted iterations) with
    # the final state so gif rendering always has the full schedule
    final_wbv = get_wbv(theta)
    for s in frames_at:
        if s not in snaps:
            snaps[s] = final_wbv
    return losses, rels, snaps, theta, stats


def train_residual_cg_ortho(theta0, f_resid, f_jvp, f_vjp, n_train, iters,
                            frames_at, get_wbv, eval_rel, refresh=200,
                            ablate=()):
    """Residual-carrying orthogonalized CG (iteration_4 core).

    The CGLS residual-recursion trick in autograd form: the per-sample
    residual vector r_hat is CARRIED across iterations (r_hat += alpha J d,
    J d from one forward-mode JVP) instead of recomputed, so gradient
    directions never re-sample the 1e-16 cancellation noise of a fresh
    model(x) - y. Gradient = VJP seeded with (2/n) r_hat; curvature =
    Gauss-Newton (2/n)||J d||^2 from the same JVP (exact wherever the model
    is linear in the parameters being solved). r_hat is refreshed from a
    true forward pass every `refresh` iterations (one frozen noise draw,
    second-order drift quenched; the basis is NOT touched). No loss values
    are ever compared. ~2 autograd passes per iteration.

    `ablate` (hardening study): a collection of mechanism names to switch
    OFF, one at a time, to measure what each one buys. Recognized:
      no_trust, no_exhaust, no_descent, no_beta_clamp, single_proj,
      no_memory, fresh_resid. Default () = the full iteration_4 optimizer.

    Returns (losses, rels, snaps, theta, stats).
    """
    ablate = frozenset(ablate)
    theta = theta0.clone()
    rhat = f_resid(theta)

    def grad_of(th, r):
        return f_vjp(th, (2.0 / n_train) * r)

    g = grad_of(theta, rhat)
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0}

    def project(vec):
        if not basis:
            return vec.clone()
        B = torch.stack(basis)
        out = vec - B.t().mv(B.mv(vec))
        if "single_proj" in ablate:
            return out
        return out - B.t().mv(B.mv(out))

    def append_to_basis(vec):
        v = project(vec)
        n2 = float(v.dot(v))
        if n2 > 1e-300 and len(basis) < theta.numel():
            basis.append(v / float(np.sqrt(n2)))

    if 0 in frames_at:
        snaps[0] = get_wbv(theta)
    gt = project(g)
    if float(gt.dot(gt)) < 1e-6 * float(g.dot(g)):
        gt = g.clone()
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    exhaust_streak = 0
    revert_streak = 0
    gn_window = [float(torch.linalg.norm(g))]

    for it in range(1, iters + 1):
        if it % refresh == 0:
            rhat = f_resid(theta)              # frozen new noise draw
            stats["refresh"] += 1
            g = grad_of(theta, rhat)
            gt = project(g)
            ggt = float(gt.dot(gt))
            # keep d: direction continuity preserves conjugacy across the
            # refresh (resetting = the cgls_refine mistake)
        dphi0 = float(g.dot(d))
        if dphi0 >= 0.0 and "no_descent" not in ablate:
            d, dphi0 = -gt, -ggt
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break
        jd = f_jvp(theta, d)
        dhd = (2.0 / n_train) * float(jd.dot(jd))   # Gauss-Newton, exact
        if dhd > 0 and np.isfinite(dhd):
            alpha = -dphi0 / dhd
            lam = dhd / dn2
            lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        theta_new = theta + alpha * d
        rhat_new = (f_resid(theta_new) if "fresh_resid" in ablate
                    else rhat + alpha * jd)
        g_new = grad_of(theta_new, rhat_new)
        gn = float(torch.linalg.norm(g_new))
        if "no_trust" not in ablate and (
                (not np.isfinite(gn)) or (gn > 100.0 * max(gn_window)
                                          and revert_streak < 10)):
            stats["revert"] += 1
            revert_streak += 1
            lhat = None if lhat is None else lhat * 4.0
            d = -gt
            if it in frames_at:
                snaps[it] = get_wbv(theta)
            continue                            # theta, rhat NOT committed
        revert_streak = 0
        if "no_memory" not in ablate:
            append_to_basis(gt)
        gt_new = project(g_new)
        # trust the projected gradient down to 1e-12 of the raw energy --
        # deep-CG gradients are DOMINATED by solved-span noise and that is
        # normal (cgls semantics); bail to raw only below that
        if ("no_exhaust" not in ablate
                and float(gt_new.dot(gt_new)) < 1e-24 * float(g_new.dot(g_new))):
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
            raw_beta = (float(gt_new.dot(gt_new - gt)) / ggt
                        if ggt > 0 else 0.0)
            beta = (raw_beta if "no_beta_clamp" in ablate
                    else max(0.0, raw_beta))
        d = -gt_new + beta * d
        theta, rhat, g, gt = theta_new, rhat_new, g_new, gt_new
        ggt = float(gt.dot(gt))
        gn_window = (gn_window + [gn])[-10:]
        losses.append(float(rhat.dot(rhat)) / n_train)
        rels.append(eval_rel(theta))
        if it in frames_at:
            snaps[it] = get_wbv(theta)
    return losses, rels, snaps, theta, stats


def train_residual_cg_slim(theta0, f_resid, f_jvp, f_vjp, n_train, iters,
                           frames_at, get_wbv, eval_rel, refresh=200,
                           ablate=()):
    """Residual-carrying orthogonalized CG, slim (iteration_5 core).

    iteration_4 minus the three safeguards the hardening ablation measured
    as free or harmful (hardening/{net_ablate,combo}.json): the gradient-
    norm trust window (its removal IMPROVED the median by 0.36 orders),
    the uphill-direction reset, and the PR+ beta clamp. What remains is
    the measured-essential set: carried residual + JVP exact steps + full
    orthogonal memory with double projection + exhaustion escape +
    periodic refresh. The only safeguard is a one-line non-finite reject.

    `ablate`: mechanism names to switch OFF for ablation studies:
      no_exhaust, single_proj, fresh_resid, no_memory.
    Returns (losses, rels, snaps, theta, stats).
    """
    ablate = frozenset(ablate)
    theta = theta0.clone()
    rhat = f_resid(theta)

    def grad_of(th, r):
        return f_vjp(th, (2.0 / n_train) * r)

    g = grad_of(theta, rhat)
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "full_reset": 0, "alpha_fallback": 0,
             "revert": 0, "refresh": 0}

    def project(vec):
        if not basis:
            return vec.clone()
        B = torch.stack(basis)
        out = vec - B.t().mv(B.mv(vec))
        if "single_proj" in ablate:
            return out
        return out - B.t().mv(B.mv(out))

    def append_to_basis(vec):
        v = project(vec)
        n2 = float(v.dot(v))
        if n2 > 1e-300 and len(basis) < theta.numel():
            basis.append(v / float(np.sqrt(n2)))

    if 0 in frames_at:
        snaps[0] = get_wbv(theta)
    gt = g.clone()                             # basis empty at start
    d = -gt
    ggt = float(gt.dot(gt))
    lhat = None
    exhaust_streak = 0

    for it in range(1, iters + 1):
        if it % refresh == 0:
            rhat = f_resid(theta)              # frozen new noise draw
            stats["refresh"] += 1
            g = grad_of(theta, rhat)
            gt = project(g)
            ggt = float(gt.dot(gt))
            # d and basis kept: direction continuity across the refresh
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or ggt <= 0.0:
            break
        jd = f_jvp(theta, d)
        dhd = (2.0 / n_train) * float(jd.dot(jd))   # Gauss-Newton, exact
        if dhd > 0 and np.isfinite(dhd):
            alpha = -float(g.dot(d)) / dhd
            lam = dhd / dn2
            lhat = lam if lhat is None else max(0.999 * lhat, lam)
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
        theta_new = theta + alpha * d
        rhat_new = (f_resid(theta_new) if "fresh_resid" in ablate
                    else rhat + alpha * jd)
        g_new = grad_of(theta_new, rhat_new)
        if not np.isfinite(float(torch.linalg.norm(g_new))):
            stats["revert"] += 1               # sole safeguard: NaN reject
            d = -gt
            if it in frames_at:
                snaps[it] = get_wbv(theta)
            continue                           # theta, rhat NOT committed
        if "no_memory" not in ablate:
            append_to_basis(gt)
        gt_new = project(g_new)
        if ("no_exhaust" not in ablate
                and float(gt_new.dot(gt_new)) < 1e-24 * float(g_new.dot(g_new))):
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
    final_wbv = get_wbv(theta)
    for s_ in frames_at:                       # pad frames missed by breaks
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, theta, stats


PROBE_EPS = 1e-3
TRIP_FRAC = 0.01               # used only by trip_mode="rel" (iteration_6)
RATCHET = 1e4                  # iteration_7 (was 100 in iteration_6)


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
    """Per-coordinate nonlinearity score q_i ~ ||d^2 p / d theta_i^2||,
    scaled per unit RELATIVE motion, evaluated at theta AND at one randomly
    perturbed point (a coordinate whose readout is ~0 probes as linear at
    the special current point but wakes under perturbation; a truly linear
    coordinate is zero at EVERY theta). 2(2m+1) forward passes."""
    g = torch.Generator().manual_seed(0)
    scale = torch.clamp(theta.abs(), min=1.0)
    signs = (torch.randint(0, 2, theta.shape, generator=g,
                           dtype=torch.int64) * 2 - 1).to(theta.dtype)
    theta_p = theta + 0.1 * scale * signs
    return torch.maximum(_self_curvature_at(f_pred, theta),
                         _self_curvature_at(f_pred, theta_p))


def train_iter6(theta0, f_resid, f_jvp, f_vjp, f_pred, n_train, iters,
                frames_at, get_wbv, eval_rel, y_norm, refresh=200,
                trip_mode="rel", ratchet=RATCHET, keep_basis=False,
                snap_S=False, refresh_mode="timer"):
    """iteration_6 core: the slim optimizer + the built-in auto-tether
    (tether v4), architecture-blind and untethered at start.

    The optimizer runs in an adaptive metric phi = S * theta, S = 1
    initially. At every refresh the carried-vs-fresh residual gap (the
    accumulated nonlinearity of the last `refresh` steps) is compared to
    max(TRIP_FRAC*||r||, 10*eps*||y||) -- the second term because a gauge
    must never act on readings below its own rounding noise. On a trip the
    per-coordinate nonlinearity q is (re-)probed, accumulated by
    elementwise max, and the metric rebuilt: S_i = 1 + T q_i / max q with
    T multiplied by RATCHET. Locks are monotone: T and q only grow. The
    memory basis and curvature estimate reset on a metric change (old
    conjugacy is void). On an exactly-linear problem q = 0 identically
    and S stays 1: the auto-tether self-removes.

    Full-batch form: the batched-mode guards (curvature floor, memory
    gate) are identically inert here -- floor mu = lhat*(1/b - 1/n) = 0 at
    b = n; the gate acts on batch-resample noise, zero in deterministic
    full batch (see batching.py train_slim_fresh_batches for the batched
    forms).

    Staircase-fix flags (each a single-variable experiment; see
    iteration_6/staircase_diagnosis.png -- the step function in the loss
    is the trip cycle, and trips stopping at drift < TRIP_FRAC*||r|| caps
    the final precision ~2 orders above the drift):
      trip_mode  "rel" (default): drift > max(TRIP_FRAC*||r||, noise);
                 "noise": drift > noise floor only -- ratchets continue
                 until geometry damage reaches rounding.
      ratchet    T multiplier per trip (default 100).
      keep_basis on a trip, carry the basis into the new metric
                 (rescale by S_new/S_old + re-orthonormalize) instead of
                 discarding it, removing the rebuild cost per stair.
      snap_S     record the tether magnitudes S alongside each frame.
      refresh_mode  "timer" (default): recalibrate every `refresh` steps.
                 "exhaust": EVENT-DRIVEN -- recalibrate when the solver
                 emits its own completion signal (the projected gradient
                 exhausts: the current right-hand side is solved), which
                 is exactly when a fresh rhs, a drift reading, and a
                 tether update are due. No timer; cadence becomes a
                 measured property of progress (fast early, when the
                 tether must tighten quickly; slow deep, when systems
                 take rank-many iterations). `refresh` then only serves
                 as a loose safety backstop (5x the timer value).
    Returns (losses, rels, snaps, theta_raw, stats)."""
    m = theta0.numel()
    S = torch.ones(m, dtype=theta0.dtype)
    phi = theta0.clone()
    q_acc = None
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
    recal_due = False
    last_recal = 0

    for it in range(1, iters + 1):
        if refresh_mode == "exhaust":
            do_recal = recal_due or (it - last_recal) >= 5 * refresh
        else:
            do_recal = it % refresh == 0
        if do_recal:
            recal_due = False
            last_recal = it
            r_fresh = f_resid(raw(phi))
            stats["refresh"] += 1
            drift = float(torch.linalg.norm(rhat - r_fresh))
            rnorm = float(torch.linalg.norm(r_fresh))
            rhat = r_fresh
            noise_floor = 10.0 * 2.22e-16 * y_norm
            tripped = drift > (noise_floor if trip_mode == "noise"
                               else max(TRIP_FRAC * rnorm, noise_floor))
            if q_acc is None or tripped:
                q = probe_self_curvature(f_pred, raw(phi))
                q_acc = q if q_acc is None else torch.maximum(q_acc, q)
            if tripped and float(q_acc.max()) > 0:
                T *= ratchet
                stats["trips"].append(it)
                S_new = 1.0 + T * (q_acc / float(q_acc.max()))
                ratio = S_new / S
                phi = phi * ratio
                S = S_new
                stats["mask_frac"] = float((S > 10.0).double().mean())
                if keep_basis and basis:
                    # carry the solved span into the new metric: rescale
                    # each basis vector, then re-orthonormalize the set
                    new_basis: list[torch.Tensor] = []
                    for b_ in basis:
                        v = b_ * ratio
                        for nb in new_basis:
                            v = v - nb * float(nb.dot(v))
                        n2 = float(v.dot(v))
                        if n2 > 1e-300:
                            v = v / float(np.sqrt(n2))
                            for nb in new_basis:   # second pass, hygiene
                                v = v - nb * float(nb.dot(v))
                            n2 = float(v.dot(v))
                            if n2 > 1e-300:
                                new_basis.append(v / float(np.sqrt(n2)))
                    basis = new_basis
                else:
                    basis = []
                lhat = None
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
            if refresh_mode == "exhaust":
                recal_due = True               # solver finished its rhs
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
            if snap_S:
                snaps[it]["S"] = S.clone().numpy()
    stats["T_final"] = T
    final_wbv = get_wbv(raw(phi))
    if snap_S:
        final_wbv["S"] = S.clone().numpy()
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    if snap_S and 0 in snaps and "S" not in snaps[0]:
        snaps[0]["S"] = np.ones(m)
    return losses, rels, snaps, raw(phi), stats


T_CEIL = 1e12                  # iteration_9 tether ceiling (audit fix:
                               # iter7's unbounded ratchet reached 1e32+ and
                               # tethered the readout through the back door)
Q_REL_FLOOR9 = 1e-8            # probe scores below this fraction of max are
                               # zeroed: structurally untetherable at ANY T.
                               # Measured (test_iter9): the readout block
                               # probes at ~1.6e-10 of max — 1e-10 left it
                               # tetherable at the T ceiling; 1e-8 clears it
                               # by 2 orders while staying 5+ below geometry.


def train_iter9(theta0, f_resid, f_jvp, f_vjp, f_pred, n_train, iters,
                frames_at, get_wbv, eval_rel, y_norm, refresh=200,
                ratchet=RATCHET, snap_S=False, recycle_fn=None,
                max_recycles=3, stall_gain=0.15, mask0=None):
    """iteration_9: the <10-vector optimizer. train_iter6's design with the
    memory basis B REMOVED and the two measured tether defects fixed.

    Phase-0 verdict behind the B removal (iteration_9/b_replacement/): B's
    floor value is INIT-RELATIVE — the fp64 free depth is ~7 decades from
    wherever the state starts, provided the leftover is not adversarially
    tail-concentrated. From construction-class states the memoryless core
    floors on its own (iteration_9/staged_gate/); from cold states NO O(m)
    mechanism can substitute for rank-sized memory (23-arm bake-off:
    compensated precision, function-space deflation, replayed projectors all
    fail lawfully), so cold-start depth is bought by STAGED RECYCLING of
    low-utilization capacity instead of by memory.

    Tether = iteration_7's (dense probe, drift gauge, noise-floor trip,
    monotone ratchet) with:
      * T ceiling at T_CEIL — trips stop at the cap, T bounded by design;
      * probe-resolution floor Q_REL_FLOOR9 — the readout block is
        untetherable at ANY T;
      * cheap trips — no basis to void; a trip only resets the direction.

    Staged recycling (the cold-start depth mechanism): at a measured stall
    (carried-residual improvement < stall_gain per refresh window, reading
    above the gauge noise floor, tether quiet), recycle_fn — an architecture
    ADAPTER closure with the same status as get_wbv — absorbs the
    lowest-utilization units into the bias, re-seeds them as a sharper-scale
    group, and returns (new_theta, active_mask, info); the stage then trains
    only that group. By readout linearity, training a group against the
    ordinary full loss IS boosting on the frozen groups' residual — no
    target arithmetic, no rescaling (the core is scale-free). recycle_fn may
    return None (nothing safely recyclable) and the schedule no-ops — on
    problems with no redundant capacity this optimizer degrades exactly to
    the memoryless slim core.

    State: phi, S, q_acc, rhat, g, d, mask — under 10 vectors, ~5F/step.
    No control decision reads a loss value. Returns
    (losses, rels, snaps, theta_raw, stats)."""
    m = theta0.numel()
    S = torch.ones(m, dtype=theta0.dtype)
    phi = theta0.clone()
    q_acc = None
    T = 1.0
    mask = (torch.ones(m, dtype=theta0.dtype) if mask0 is None
            else mask0.to(theta0.dtype))

    def raw(p):
        return p / S

    rhat = f_resid(raw(phi))
    g = (f_vjp(raw(phi), (2.0 / n_train) * rhat) / S) * mask
    losses, rels, snaps = [], [], {}
    stats = {"alpha_fallback": 0, "revert": 0, "refresh": 0, "trips": [],
             "T_final": 1.0, "recycles": [], "recycle_info": [],
             "mask_frac": None}

    if 0 in frames_at:
        snaps[0] = get_wbv(raw(phi))
        if snap_S:
            snaps[0]["S"] = S.clone().numpy()
    d = -g
    lhat = None
    noise_floor = 10.0 * 2.22e-16 * y_norm
    prev_rnorm = None
    recycles = 0
    last_event = 0                 # last trip or recycle

    for it in range(1, iters + 1):
        if it % refresh == 0:
            r_fresh = f_resid(raw(phi))
            stats["refresh"] += 1
            drift = float(torch.linalg.norm(rhat - r_fresh))
            rnorm = float(torch.linalg.norm(r_fresh))
            rhat = r_fresh
            tripped = drift > noise_floor and T < T_CEIL
            if q_acc is None or tripped:
                q = probe_self_curvature(f_pred, raw(phi))
                q_acc = q if q_acc is None else torch.maximum(q_acc, q)
                qm = float(q_acc.max())
                if qm > 0:         # resolution floor: untetherable block
                    q_acc[q_acc < Q_REL_FLOOR9 * qm] = 0.0
            if tripped and float(q_acc.max()) > 0:
                T = min(T * ratchet, T_CEIL)
                stats["trips"].append(it)
                S_new = 1.0 + T * (q_acc / float(q_acc.max()))
                phi = phi * (S_new / S)
                S = S_new
                stats["mask_frac"] = float((S > 10.0).double().mean())
                lhat = None
                last_event = it
            stalled = (prev_rnorm is not None and not tripped
                       and rnorm > 100.0 * noise_floor
                       and prev_rnorm - rnorm < stall_gain * prev_rnorm
                       and it - last_event >= 2 * refresh)
            if (stalled and recycle_fn is not None
                    and recycles < max_recycles):
                res = recycle_fn(raw(phi), rhat)
                if res is not None:
                    theta_new, new_mask, info = res
                    recycles += 1
                    last_event = it
                    stats["recycles"].append(it)
                    stats["recycle_info"].append(info)
                    mask = new_mask.to(theta0.dtype)
                    phi = theta_new * S
                    rhat = f_resid(raw(phi))   # theta jumped: honest reread
                    lhat = None
            prev_rnorm = rnorm
            g = (f_vjp(raw(phi), (2.0 / n_train) * rhat) / S) * mask
            if last_event == it:
                d = -g                          # cheap trip/recycle: d only
        gg = float(g.dot(g))
        dn2 = float(d.dot(d))
        if dn2 <= 0.0 or gg <= 0.0:
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
        g_new = (f_vjp(raw(phi_new), (2.0 / n_train) * rhat_new) / S) * mask
        if not np.isfinite(float(torch.linalg.norm(g_new))):
            stats["revert"] += 1               # sole safeguard: NaN reject
            d = -g
            if it in frames_at:
                snaps[it] = get_wbv(raw(phi))
                if snap_S:
                    snaps[it]["S"] = S.clone().numpy()
            continue
        beta = max(0.0, float(g_new.dot(g_new - g)) / gg)
        d = -g_new + beta * d
        phi, rhat, g = phi_new, rhat_new, g_new
        losses.append(float(rhat.dot(rhat)) / n_train)
        rels.append(eval_rel(raw(phi)))
        if it in frames_at:
            snaps[it] = get_wbv(raw(phi))
            if snap_S:
                snaps[it]["S"] = S.clone().numpy()
    stats["T_final"] = T
    final_wbv = get_wbv(raw(phi))
    if snap_S:
        final_wbv["S"] = S.clone().numpy()
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, raw(phi), stats


def make_recycler(W, xt, w_cap, mu_ladder=(4.0, 16.0, 64.0)):
    """Architecture adapter for the QIMlp layout theta = [w, b, v, c0]
    (same status as get_wbv: the CORE stays blind; this closure owns the
    unit structure). Utilization of neuron k: u_k = |v_k| * std_data(tanh_k)
    — the rms damage of removing the neuron after absorbing its mean into
    the bias (a saturated halo neuron has std ~ 0: near-lossless removal).
    Recycles the lowest-u third (bounded so total absorb damage stays below
    half the current residual), re-seeded as an equispaced sharper group:
    w_new = mu * median|w_kept| capped at w_cap (sampling guard), v = 0.
    Returns (theta_new, active_mask over [recycled v, c0], info) or None."""
    state = {"stage": 0, "done": np.zeros(W, dtype=bool)}

    def recycle(theta, rhat):
        th = theta.detach().clone()
        w, b, v = th[:W], th[W:2 * W], th[2 * W:3 * W]
        with torch.no_grad():
            acts = torch.tanh(xt * w + b)          # (n, W)
            mean_a = acts.mean(0)
            std_a = acts.std(0)
        u = (v.abs() * std_a).numpy().copy()
        u[state["done"]] = np.inf
        n = acts.shape[0]
        rms_r = float(torch.linalg.norm(rhat)) / float(np.sqrt(n))
        order = np.argsort(u)
        chosen, acc2 = [], 0.0
        for k in order:
            if not np.isfinite(u[k]) or len(chosen) >= W // 3:
                break
            if u[k] > 0.1 * rms_r and acc2 > 0.0:
                break
            if np.sqrt(acc2 + u[k] ** 2) > 0.5 * rms_r:
                break
            chosen.append(int(k))
            acc2 += float(u[k]) ** 2
        if len(chosen) < max(4, W // 16):
            return None                            # nothing safely recyclable
        idx = np.array(chosen)
        th[-1] += float((v[idx] * mean_a[idx]).sum())   # absorb into bias
        kept = np.setdiff1d(np.arange(W), np.where(state["done"])[0])
        kept = np.setdiff1d(kept, idx)
        w_ref = float(torch.median(w[kept].abs())) if kept.size else 1.0
        mu = mu_ladder[min(state["stage"], len(mu_ladder) - 1)]
        w_new = min(mu * w_ref, w_cap)
        centers = np.linspace(-1.0, 1.0, len(idx) + 2)[1:-1]
        th[idx] = w_new
        th[W + idx] = torch.tensor(-w_new * centers, dtype=th.dtype)
        th[2 * W + idx] = 0.0
        m = th.numel()
        new_mask = torch.zeros(m)
        new_mask[2 * W + idx] = 1.0                # recycled readout
        new_mask[-1] = 1.0                         # bias stays trainable
        state["done"][idx] = True
        state["stage"] += 1
        info = {"n": len(idx), "w_new": w_new,
                "damage_rel": float(np.sqrt(acc2) / max(rms_r, 1e-300))}
        return th, new_mask, info

    return recycle


def train_iter9b(theta0, f_resid_idx, f_jvp_idx, f_vjp_idx, f_pred_gauge,
                 n_total, iters, frames_at, get_wbv, eval_rel, y_gauge_norm,
                 batch_iter, gauge_idx, refresh=200, ratchet=RATCHET,
                 snap_S=False, recycle_fn=None, max_recycles=3,
                 stall_gain=0.15):
    """iteration_9, batched form: per-batch stepping with the structural
    curvature floor mu = lhat (1/b - 1/n) (zero constants, inert at b = n;
    the measured zero-blowup guard), while ALL slow machinery — carried
    residual, drift gauge, probe, tether, recycler — lives on a FIXED gauge
    subset of rows (<= 2003): the gauge residual is carried with one extra
    JVP per step, so the same noise-floor trip rule, T ceiling, probe floor,
    and staged recycling operate identically at any batch size. One code
    path; the optimizer never asks what kind of batch it holds."""
    m = theta0.numel()
    S = torch.ones(m, dtype=theta0.dtype)
    phi = theta0.clone()
    q_acc = None
    T = 1.0
    mask = torch.ones(m, dtype=theta0.dtype)
    n_g = len(gauge_idx)

    def raw(p):
        return p / S

    rg = f_resid_idx(raw(phi), gauge_idx)          # carried gauge residual
    losses, rels, snaps = [], [], {}
    stats = {"alpha_fallback": 0, "revert": 0, "refresh": 0, "trips": [],
             "T_final": 1.0, "recycles": [], "recycle_info": [],
             "mask_frac": None}
    if 0 in frames_at:
        snaps[0] = get_wbv(raw(phi))
        if snap_S:
            snaps[0]["S"] = S.clone().numpy()
    d = None
    g_prev = None
    lhat = None
    noise_floor = 10.0 * 2.22e-16 * y_gauge_norm
    prev_rnorm = None
    recycles = 0
    last_event = 0

    for it in range(1, iters + 1):
        if it % refresh == 0:
            r_fresh = f_resid_idx(raw(phi), gauge_idx)
            stats["refresh"] += 1
            drift = float(torch.linalg.norm(rg - r_fresh))
            rnorm = float(torch.linalg.norm(r_fresh))
            rg = r_fresh
            tripped = drift > noise_floor and T < T_CEIL
            if q_acc is None or tripped:
                q = probe_self_curvature(f_pred_gauge, raw(phi))
                q_acc = q if q_acc is None else torch.maximum(q_acc, q)
                qm = float(q_acc.max())
                if qm > 0:
                    q_acc[q_acc < Q_REL_FLOOR9 * qm] = 0.0
            if tripped and float(q_acc.max()) > 0:
                T = min(T * ratchet, T_CEIL)
                stats["trips"].append(it)
                S_new = 1.0 + T * (q_acc / float(q_acc.max()))
                phi = phi * (S_new / S)
                S = S_new
                stats["mask_frac"] = float((S > 10.0).double().mean())
                lhat = None
                last_event = it
            stalled = (prev_rnorm is not None and not tripped
                       and rnorm > 100.0 * noise_floor
                       and prev_rnorm - rnorm < stall_gain * prev_rnorm
                       and it - last_event >= 2 * refresh)
            if (stalled and recycle_fn is not None
                    and recycles < max_recycles):
                res = recycle_fn(raw(phi), rg)
                if res is not None:
                    theta_new, new_mask, info = res
                    recycles += 1
                    last_event = it
                    stats["recycles"].append(it)
                    stats["recycle_info"].append(info)
                    mask = new_mask.to(theta0.dtype)
                    phi = theta_new * S
                    rg = f_resid_idx(raw(phi), gauge_idx)
                    lhat = None
            prev_rnorm = rnorm
            if last_event == it:
                d = None                        # cheap trip: momentum gone
                g_prev = None
        rows = next(batch_iter)
        b = len(rows)
        r_b = f_resid_idx(raw(phi), rows)
        g = (f_vjp_idx(raw(phi), (2.0 / b) * r_b, rows) / S) * mask
        gn = float(torch.linalg.norm(g))
        if not np.isfinite(gn) or gn == 0.0:
            stats["revert"] += 1
            d = None
            continue
        if d is None or g_prev is None:
            d = -g
        else:
            beta = max(0.0, float(g.dot(g - g_prev))
                       / max(float(g_prev.dot(g_prev)), 1e-300))
            d = -g + beta * d
        jd = f_jvp_idx(raw(phi), d / S, rows)
        c = (2.0 / b) * float(jd.dot(jd))
        dn2 = float(d.dot(d))
        if not (c > 0 and np.isfinite(c)) or dn2 <= 0:
            stats["alpha_fallback"] += 1
            d = None
            continue
        lam = c / dn2
        lhat = lam if lhat is None else max(0.999 * lhat, lam)
        mu = lhat * max(1.0 / b - 1.0 / n_total, 0.0)
        alpha = -float(g.dot(d)) / (c + mu * dn2)
        if not np.isfinite(alpha):
            stats["alpha_fallback"] += 1
            d = None
            continue
        ug = f_jvp_idx(raw(phi), d / S, gauge_idx)
        phi = phi + alpha * d
        rg = rg + alpha * ug
        g_prev = g
        losses.append(float(r_b.dot(r_b)) / b)
        rels.append(eval_rel(raw(phi)))
        if it in frames_at:
            snaps[it] = get_wbv(raw(phi))
            if snap_S:
                snaps[it]["S"] = S.clone().numpy()
    stats["T_final"] = T
    final_wbv = get_wbv(raw(phi))
    if snap_S:
        final_wbv["S"] = S.clone().numpy()
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, raw(phi), stats


T_CAP = 1e12                   # tether hard cap (probe separation scale)
ETA_T = 0.1                    # tether controller gain (per-step, log)
Q_FLOOR = 1e-10                # relative probe-resolution floor: below it
                               # a coordinate is untetherable at ANY T


def train_iter8(theta0, f_resid, f_jvp, f_vjp, f_pred, n_total, iters,
                frames_at, get_wbv, eval_rel, y_norm, batch_iter=None,
                refresh=200, ema=0.9, probe_every=400, snap_S=False,
                ablate=frozenset()):
    """iteration_8: continuous everything (spec: iteration_8/
    fable_thoughts.md rev 4). One code path for any batch size --
    batch_iter yields index tensors, None means full batch every step.

    Closures take an optional index argument: f_resid(theta, idx),
    f_jvp(theta, d, idx), f_vjp(theta, seed, idx); f_pred(theta) is the
    full-data prediction map used only by the periodic probe.

    Two per-step EMA signals drive every adaptation (details in the
    module header): Dbar (bend, from a read-only fresh forward on the
    CURRENT rows -- the carried residual is only REPLACED on the timer)
    and rhobar (trust, gradient-space surprise vs the linear model --
    exactly zero on any quadratic even under damping). The tether is a
    memoryless map of Dbar applied as a diagonal preconditioner
    p = S^-2 in raw theta coordinates; damping, beta, and B-admission
    are continuous functions of rhobar. Numerical-existence guards
    (NaN reject, fp-zero projection fallback) are the only discrete
    acts. Returns (losses, rels, snaps, theta, stats)."""
    no_b = "no_B" in ablate
    w_ema = 0.99 if "ema99" in ablate else ema
    m = theta0.numel()
    eps10 = 10.0 * 2.22e-16
    theta = theta0.clone()
    basis: list[torch.Tensor] = []
    losses, rels, snaps = [], [], {}
    stats = {"exhaust": 0, "revert": 0, "alpha_fallback": 0, "refresh": 0,
             "probes": 0, "T_final": 1.0, "basis_len": 0,
             "T_traj": [], "D_traj": [], "rho_traj": []}

    def project(vec):
        if no_b or not basis:
            return vec.clone()
        B = torch.stack(basis)
        out = vec - B.t().mv(B.mv(vec))
        return out - B.t().mv(B.mv(out))

    def append_to_basis(vec):
        if no_b:
            return
        v = project(vec)
        n2 = float(v.dot(v))
        if n2 > 1e-300 and len(basis) < m:
            basis.append(v / float(np.sqrt(n2)))

    # trust seed: start TRUSTING (rho=0). The structural floor term
    # lhat(1/b - 1/n) guards the catastrophic rank-deficiency case
    # instantly and independently of rho; seeding distrust instead was
    # measured to poison B via half-solved lines (test_iter8 history).
    ln_T = 0.0                        # log tether magnitude (controller)
    w_tether = 0.98                   # log-EMA window for the bend level
    ln_D = np.log(1e-17)              # geometric bend level (log-space
                                      # EMA: bend is spiky-multiplicative;
                                      # a linear EMA over-reacts 10x per
                                      # spike, sawtooths T, and shreds B
                                      # through the metric-decay channel)
    rho_bar = 1.0 if "seed_rho1" in ablate else 0.0
    d_bar = 0.0
    q_hat = None
    qn = None                         # normalized, floored q_hat
    lhat = None
    a_pace = 0.0
    r_acc = 0.0                       # bend-paced retirement accumulator
    e_last = 0.0                      # exactness of the last taken step
    d = None
    u = None                          # raw-space momentum/conjugate carry
    g_last = None                     # previous projected RAW gradient
    stash = None                      # (d_stepped, expected_gd, dnorm)
    idx = None
    rhat = None
    r_prev = None                     # last fresh residual (gauge read)
    S_prev = None                     # metric of the previous step
    T_prev = 1.0
    if 0 in frames_at:
        snaps[0] = get_wbv(theta)
        if snap_S:
            snaps[0]["S"] = np.ones(m)

    for it in range(1, iters + 1):
        if batch_iter is not None:
            idx = next(batch_iter)
        b = n_total if idx is None else int(idx.numel())
        if idx is not None or rhat is None:
            rhat = f_resid(theta, idx)
        g = f_vjp(theta, (2.0 / b) * rhat, idx)
        gnorm = float(torch.linalg.norm(g))
        if not np.isfinite(gnorm):
            stats["revert"] += 1
            stash = None
            continue
        # trust: surprise of the fresh gradient vs the linear model's
        # prediction along the last stepped direction (zero on any
        # quadratic, damped or not; large on batch switch / bend)
        if stash is not None:
            d_st, exp_gd, dnorm_st = stash
            rho_inst = min(1.0, abs(float(g.dot(d_st)) - exp_gd)
                           / (gnorm * dnorm_st + 1e-300))
            rho_bar = w_ema * rho_bar + (1.0 - w_ema) * rho_inst
        # periodic dense probe. The WHO-map ranks are max-accumulated
        # (a coordinate that ever bent keeps its rank -- staleness in the
        # other direction was measured to let late-bending coordinates
        # wander free while T rode the cap); gate MAGNITUDES still relax
        # fully through the memoryless T.
        if it == refresh or (it > refresh
                             and (it - refresh) % probe_every == 0):
            q_new = probe_self_curvature(f_pred, theta)
            q_hat = q_new if q_hat is None else torch.maximum(q_hat, q_new)
            stats["probes"] += 1
            qmax = float(q_hat.max())
            # self-removal guard (Lesson 6): the probe's own rounding
            # noise is q_noise ~ 4 eps ||p|| / PROBE_EPS^2; a relative
            # normalization alone would promote pure noise to qn = 1 on
            # an exactly-linear problem and tether it.
            with torch.no_grad():
                q_noise = (4.0 * 2.22e-16 / PROBE_EPS ** 2
                           * float(torch.linalg.norm(f_pred(theta))))
            if qmax > 100.0 * q_noise:
                qn = q_hat / qmax
                qn = torch.where(qn < Q_FLOOR, torch.zeros_like(qn), qn)
            else:
                qn = None
        # tether magnitude: log-proportional controller with a MEASURED
        # setpoint -- drive the per-step geometry bend to the gauge's own
        # rounding floor, then hold. At the setpoint dlnT = 0 exactly:
        # the metric goes stationary, which is what lets B grow to rank
        # and the deep CG grind happen (measured: every level-tracking
        # form of T keeps moving during descent and shreds the memory).
        # Above the floor T climbs at a rate proportional to the log
        # error (signal-locked, no open-loop slew); below, it relaxes
        # the same way. Bounded [1, T_CAP]. No events, no cadence.
        T = float(np.exp(ln_T))
        if qn is not None:
            S_vec = 1.0 + T * qn
        else:
            S_vec = None
        # metric motion: transport the O(m) carried vectors exactly
        # (contravariant u scales by ratio, covariant gradients by
        # 1/ratio). B is NOT transported (transport measured harmful at
        # jumps; per-step motion is ~1e-6 in the deep phase where B
        # matters, and the retirement laws clean the rest).
        if S_vec is not None:
            ratio = S_vec if S_prev is None else S_vec / S_prev
        elif S_prev is not None:
            ratio = 1.0 / S_prev
        else:
            ratio = None
        if ratio is not None:
            if u is not None:
                # gate IN-FLIGHT momentum: tightening a coordinate must
                # slow motion already stored in u (contravariant u*ratio
                # measured: it exactly cancels the tether at a metric
                # shock -- the frozen geometry kept moving on stored
                # momentum and exact GN steps exploded, rel 2.8e-5 ->
                # 1e3). Relaxing frees only NEW content.
                u = u / torch.clamp(ratio, min=1.0)
            if g_last is not None:
                g_last = g_last / ratio
        S_prev = S_vec
        # paced admission by measured STEP EXACTNESS: a direction may be
        # marked solved only to the degree its line was actually solved.
        # e = c/(c + mu ||d||^2) is the fraction of the exact step taken
        # (1 undamped, ->0 fully damped); a damped search must not hide
        # its own leftover error behind the projection. e embeds rhobar
        # through mu, so this subsumes trust pacing.
        if g_last is not None and not no_b:
            a_pace = min(a_pace + e_last, 2.0)
            if a_pace >= 1.0:
                append_to_basis(g_last)
                a_pace -= 1.0
        ghat = g if S_vec is None else g / S_vec
        gt = project(ghat)
        exhausted = float(gt.dot(gt)) < 1e-24 * float(ghat.dot(ghat))
        if exhausted and not no_b:
            stats["exhaust"] += 1
            gt = ghat.clone()
            # paced retirement, the mirror of paced admission: an
            # exhausted step with a live gradient means marked-solved
            # lines have reopened (fp leak / geometry moved under the
            # metric). Retire the two STALEST marks per exhausted step:
            # retirement must OUTPACE admission (which continues at ~1
            # per exact step) so the basis DRAINS -- a sliding-window
            # restart, incremental and self-limiting, never a reset.
            # (pop-1 and pop-2 measured: perpetual saturation churn,
            # steepest-descent stall at 1e-2 -- the drain must OUTRUN
            # admission decisively. Proportional decay does: ~12% of the
            # basis per exhausted step, oldest first. A decay process,
            # not a reset: it reproduces restarted-CG's fresh-run
            # dynamic, which is where slim's floor actually came from.)
            for _ in range(min(max(2, len(basis) // 8), len(basis))):
                basis.pop(0)
        beta_pr = 0.0
        if g_last is not None and not exhausted:
            gl2 = float(g_last.dot(g_last))
            if gl2 > 0:
                beta_pr = float(gt.dot(gt - g_last)) / gl2
        if "raw_beta" in ablate:
            beta = beta_pr
        else:
            beta = (1.0 - rho_bar) * beta_pr + rho_bar * 0.9
        # preconditioned CG done RIGHT: all direction algebra (project,
        # beta, momentum u) lives in the CURRENT metric (ghat = g/S);
        # the theta-space step is d = u/S. A tightening metric both
        # suppresses new geometry content AND (via the exact transport
        # above) re-suppresses everything the momentum accumulated.
        # (Raw-space beta with use-time p measured: conjugacy computed
        # in the wrong metric, readout polish never converges.)
        u = -gt if u is None else -gt + beta * u
        d = u if S_vec is None else u / S_vec
        dn2 = float(d.dot(d))
        if dn2 <= 0.0:
            u = -gt
            d = u if S_vec is None else u / S_vec
            dn2 = float(d.dot(d))
            if dn2 <= 0.0:
                break
        jd = f_jvp(theta, d, idx)
        c = (2.0 / b) * float(jd.dot(jd))
        if c > 0 and np.isfinite(c):
            lam = c / dn2
            lhat = lam if lhat is None else max(0.999 * lhat, lam)
        # step damping: the STRUCTURAL floor only (the measured anti-
        # blowup device; zero at full batch). rho_bar was measured to
        # strangle the deep spectrum through mu even at 1e-9 -- the v5
        # isotropic-damping death, twice confirmed ("rho_mu" re-enables
        # it for the record).
        mu = 0.0
        if lhat is not None:
            rho_term = rho_bar if "rho_mu" in ablate else 0.0
            mu = lhat * (max(0.0, 1.0 / b - 1.0 / n_total) + rho_term)
        denom = c + mu * dn2
        gd = float(g.dot(d))
        if denom > 0 and np.isfinite(denom):
            alpha = -gd / denom
            e_step = c / denom
        else:
            stats["alpha_fallback"] += 1
            alpha = 1.0 / lhat if lhat is not None else 1e-6
            e_step = 0.0
        theta_new = theta + alpha * d
        rhat_pred = rhat + alpha * jd
        r_fresh = f_resid(theta_new, idx)
        rn = float(torch.linalg.norm(r_fresh))
        # per-step bend: predict from LAST step's fresh residual (r(theta)
        # exactly), not from the drifted carried state -- the carried
        # prediction conflates this step's bend with all drift since the
        # last adoption (measured: inflates Dbar ~200x)
        # (under batching rhat was computed fresh THIS step, so rhat_pred
        # already is the per-step prediction; r_prev applies full-batch)
        bend_pred = (r_prev + alpha * jd) \
            if (idx is None and r_prev is not None) else rhat_pred
        if not (np.isfinite(rn)
                and np.isfinite(float(torch.linalg.norm(theta_new)))):
            stats["revert"] += 1
            u = -gt
            d = u if S_vec is None else u / S_vec
            stash = None
            g_last = gt
            e_last = 0.0
            continue
        delta = float(torch.linalg.norm(bend_pred - r_fresh))
        # LESSON 6: the gauge never acts on readings below its own
        # rounding floor. Raw delta/r fed to T makes T anneal as 1/r on
        # pure noise forever (measured: perpetual metric drift, B capped
        # ~100 << rank, no floor). Noise-subtracted bend stabilizes T
        # exactly when geometry motion reaches rounding.
        # anti-windup: integrate only while the actuator exists (qn from
        # the first probe). Winding up during the free era measured: T at
        # cap the moment the probe lands, wander-damaged geometry frozen
        # forever, zero post-lock progress.
        err = np.log(max(delta, 1e-300) / (eps10 * y_norm)) \
            if qn is not None else 0.0
        # asymmetric rates (v6's measured rule, re-learned here the hard
        # way): locking tracks the signal at full gain; relaxing runs
        # 10x slower. Symmetric relax measured catastrophic -- T unwinds
        # 4 orders in ~50 steps, the freed geometry explodes (rel 73)
        # and the run never recovers.
        step_ln = ETA_T * np.clip(err, -2.0, 2.0)
        if step_ln < 0:
            step_ln *= 0.1
        ln_T = float(np.clip(ln_T + step_ln, 0.0, np.log(T_CAP)))
        ln_D = w_tether * ln_D + (1.0 - w_tether) * np.log(
            max(delta / max(rn, 1e-300), 1e-30))
        d_bar = float(np.exp(ln_D))
        r_prev = r_fresh
        # bend-paced retirement: B's marks are valid only for a
        # stationary quadratic, so they decay at the rate the world is
        # MEASURED to move. Wander era (Dbar ~ 1): rolling window, no
        # stale accumulation. Frozen era (Dbar ~ 1e-14): marks live
        # forever -- the deep CG-with-memory phase. (Exactness pacing
        # alone measured insufficient: exact steps on a MOVING quadratic
        # admitted 444/445 stale marks and hid a 1e-2 error.)
        if not no_b:
            # three continuous decay channels for B's marks:
            #   bend      (quadratic moved)      pace min(1, Dbar)
            #   metric    (coordinates moved)    HALF-LIFE of one decade
            #             of T motion: rate = len(B) |dlnT| / ln(100)
            #             (measured: T relaxing 6 decades left 94 stale
            #             vectors projecting out real signal, exh = 0)
            #   fp leak   handled at exhaust (proportional pop)
            dlnT = abs(np.log(max(T, 1e-300))
                       - np.log(max(T_prev, 1e-300)))
            r_acc = min(r_acc + min(1.0, d_bar)
                        + len(basis) * dlnT / np.log(100.0),
                        float(max(1, len(basis))))
            while r_acc >= 1.0 and basis:
                basis.pop(0)
                r_acc -= 1.0
        T_prev = T
        if idx is not None or it % refresh == 0:
            rhat = r_fresh
            stats["refresh"] += int(idx is None)
        else:
            rhat = rhat_pred
        theta = theta_new
        stash = (d, gd + alpha * c, float(np.sqrt(dn2)))
        g_last = gt
        e_last = e_step
        losses.append(rn * rn / b)
        rels.append(eval_rel(theta))
        stats["T_traj"].append(T)
        stats["D_traj"].append(d_bar)
        stats["rho_traj"].append(rho_bar)
        if it in frames_at:
            snaps[it] = get_wbv(theta)
            if snap_S:
                snaps[it]["S"] = (np.ones(m) if S_vec is None
                                  else S_vec.numpy().copy())
    stats["T_final"] = float(np.clip(d_bar / eps10, 1.0, T_CAP))
    stats["basis_len"] = len(basis)
    final_wbv = get_wbv(theta)
    if snap_S:
        final_wbv["S"] = (np.ones(m) if qn is None
                          else (1.0 + stats["T_final"] * qn).numpy().copy())
    for s_ in frames_at:
        if s_ not in snaps:
            snaps[s_] = final_wbv
    return losses, rels, snaps, theta, stats


# ------------------------------- one run -------------------------------

def run_case(job):
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    fn_name, N = job["fn"], job["N"]
    fn_text = TARGETS[fn_name]
    f_vec = make_fn(fn_text)
    if job.get("init") == "qi_readout":
        # companion arm (NOT the standard sweep protocol): full fp64 QI
        # construction as init — geometry AND readout. Shows what the
        # memoryless core does from a construction-class state.
        sys.path.insert(0, str(REPO_ROOT))
        from src.construction.qi_mpmath import construct_qi
        from src.data.targets import get_target
        t = get_target(fn_name)
        res = construct_qi(t.fn_numpy, t.deriv_numpy, N, precision="fp64",
                           lambda_star=0.30)
        centers, gamma = res.centers, res.gamma
        W = centers.size
        theta0 = torch.tensor(np.concatenate([
            np.full(W, gamma), -gamma * centers, res.a_coeffs, [res.c0]]))
    else:
        centers, gamma, _halo = geometry(N, LAM)
        W = centers.size
        # expD06/seed9 init: QI geometry, zero readout, all trainable
        theta0 = torch.tensor(np.concatenate([
            np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))

    x_tr = np.linspace(-1, 1, N_TRAIN)
    x_ev = np.linspace(-1, 1, N_EVAL)
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
        (g,) = torch.autograd.grad(pred, thr, grad_outputs=seed)
        return g.detach()

    def eval_rel(th):
        with torch.no_grad():
            r = model(xe, th) - ye
            return float(torch.linalg.norm(r)) / ye_norm

    def get_wbv(th):
        t = th.detach().numpy()
        return {"w": t[:W].copy(), "b": t[W:2 * W].copy(),
                "v": t[2 * W:3 * W].copy()}

    # lstsq floor on the FROZEN init geometry (the reference this init buys)
    A = np.hstack([build_phi(x_tr, gamma, centers, "tanh"),
                   np.ones((x_tr.size, 1))])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > RCOND * s[0]
    sol = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                  * (U.T @ f_vec(x_tr)))
    A_ev = np.hstack([build_phi(x_ev, gamma, centers, "tanh"),
                      np.ones((x_ev.size, 1))])
    floor = float(np.linalg.norm(A_ev @ sol - f_vec(x_ev))
                  / np.linalg.norm(f_vec(x_ev)))

    # geometry tether: optimize phi = scale * theta with scale = T on the
    # (w, b) block, 1 on (v, c0). A pure preconditioner -- same minimizers,
    # every parameter still trainable -- but the effective curvature of the
    # geometry block drops by T^2, so CG solves the readout as if the
    # geometry were frozen and moves (w, b) on a T^2-slower timescale.
    scale = torch.ones(3 * W + 1, dtype=torch.float64)
    scale[:2 * W] = job.get("tether", TETHER)

    def p_resid(phi):
        return f_resid(phi / scale)

    def p_jvp(phi, dvec):
        return f_jvp(phi / scale, dvec / scale)

    def p_vjp(phi, seed):
        return f_vjp(phi / scale, seed) / scale

    def p_wbv(phi):
        return get_wbv(phi / scale)

    def p_eval(phi):
        return eval_rel(phi / scale)

    save_snaps = job.get("save_snaps", True)
    frames_at = (set(expd06.frame_steps(job["iters"])) if save_snaps
                 else {0})
    t0 = time.time()
    core_name = job.get("core", "iter9")
    if core_name == "iter9":
        recycler = (None if job.get("no_recycle") else
                    make_recycler(W, xt, w_cap=N_TRAIN / 8.0))
        losses, rels, snaps, _, stats = train_iter9(
            theta0, f_resid, f_jvp, f_vjp, f_pred, N_TRAIN, job["iters"],
            frames_at, get_wbv, eval_rel, float(torch.linalg.norm(yt)),
            refresh=job.get("refresh", REFRESH),
            ratchet=job.get("ratchet", RATCHET),
            snap_S=job.get("snap_S", False),
            recycle_fn=recycler,
            max_recycles=job.get("max_recycles", 3),
            stall_gain=job.get("stall_gain", 0.15))
    elif core_name == "iter8":
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
            (g,) = torch.autograd.grad(model(x_, thr), thr,
                                       grad_outputs=seed)
            return g.detach()

        losses, rels, snaps, _, stats = train_iter8(
            theta0, f_resid8, f_jvp8, f_vjp8, f_pred, N_TRAIN,
            job["iters"], frames_at, get_wbv, eval_rel,
            float(torch.linalg.norm(yt)),
            batch_iter=None,
            refresh=job.get("refresh", REFRESH),
            ema=job.get("ema", 0.9),
            probe_every=job.get("probe_every", 400),
            snap_S=job.get("snap_S", False),
            ablate=frozenset(job.get("ablate", ())))
    elif core_name == "iter6":  # noqa: E501  (legacy iteration_6/7 core)
        # auto-tether manages its own metric: raw closures, no fixed scale
        losses, rels, snaps, _, stats = train_iter6(
            theta0, f_resid, f_jvp, f_vjp, f_pred, N_TRAIN, job["iters"],
            frames_at, get_wbv, eval_rel,
            float(torch.linalg.norm(yt)),
            refresh=job.get("refresh", REFRESH),
            trip_mode=job.get("trip_mode", TRIP_MODE),
            ratchet=job.get("ratchet", RATCHET),
            keep_basis=job.get("keep_basis", False),
            snap_S=job.get("snap_S", False),
            refresh_mode=job.get("refresh_mode", "timer"))
    else:
        core = (train_residual_cg_ortho if core_name == "v4"
                else train_residual_cg_slim)
        losses, rels, snaps, _, stats = core(
            theta0 * scale, p_resid, p_jvp, p_vjp, N_TRAIN, job["iters"],
            frames_at, p_wbv, p_eval, refresh=job.get("refresh", REFRESH),
            ablate=job.get("ablate", ()))

    out = {"fn": fn_name, "N": N, "seed": SEED_NAME, "a": 0.0,
           "floor": floor, "losses": losses, "rels": rels,
           "best_rel": float(np.min(rels)) if rels else float("nan"),
           "stats": stats,
           "wall_s": round(time.time() - t0, 1),
           "centers": centers.tolist()}
    if save_snaps:
        # persist snapshots so gifs can be re-rendered without retraining
        snap_dir = Path(job.get("out_dir", OUT_DIR)) / "snaps"
        snap_dir.mkdir(parents=True, exist_ok=True)
        ssteps = sorted(snaps)
        extra = {}
        if all("S" in snaps[s] for s in ssteps):
            extra["S"] = np.stack([snaps[s]["S"] for s in ssteps])
        np.savez_compressed(
            snap_dir / f"{fn_name}--N{N}.npz",
            steps=np.array(ssteps, dtype=np.int64),
            w=np.stack([snaps[s]["w"] for s in ssteps]),
            b=np.stack([snaps[s]["b"] for s in ssteps]),
            v=np.stack([snaps[s]["v"] for s in ssteps]),
            losses=np.array(losses), rels=np.array(rels), floor=floor,
            stats=json.dumps(stats), **extra)
        out["snaps"] = {str(k): {kk: vv.tolist() for kk, vv in s_.items()}
                        for k, s_ in snaps.items()}
    return out


def load_rows_from_snaps(out_dir=None):
    """Rebuild rows from the persisted npz snapshots (--render-only)."""
    out_dir = OUT_DIR if out_dir is None else out_dir
    rows = []
    for path in sorted((out_dir / "snaps").glob("*.npz")):
        fn_name, n_part = path.stem.split("--N")
        z = np.load(path)
        rows.append({
            "fn": fn_name, "N": int(n_part), "seed": SEED_NAME, "a": 0.0,
            "floor": float(z["floor"]),
            "losses": z["losses"].tolist(), "rels": z["rels"].tolist(),
            "best_rel": float(np.min(z["rels"])),
            "snaps": {str(int(s)): {"w": z["w"][i].tolist(),
                                    "b": z["b"][i].tolist(),
                                    "v": z["v"][i].tolist()}
                      for i, s in enumerate(z["steps"])}})
    return rows


# ------------------------------- figures -------------------------------

def _y2mean(fn_name):
    """mean(y^2) on the train grid -- the MSE <-> rel-L2 conversion factor:
    rel L2 = sqrt(MSE / mean(y^2))."""
    f = make_fn(TARGETS[fn_name])
    return float(np.mean(f(np.linspace(-1, 1, N_TRAIN)) ** 2))


def render_loss_2x2(rows, N, path, tether=None, iters=None):
    """ONE unit everywhere: relative L2. Train MSE is converted via
    sqrt(MSE / mean y^2) so train, eval, and the floor share the axis.
    (An earlier version plotted raw MSE against a rel-L2 floor -- squared
    vs unsquared units on one axis -- which misread as 'train reaches the
    floor'. It never did.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tstr = "auto" if tether is None else f"{tether:g}x"
    iters = ITERS if iters is None else iters
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for ax, fn in zip(axes.flat, TARGETS3):
        rs = [r for r in rows if r["fn"] == fn and r["N"] == N]
        if not rs:
            continue
        r = rs[0]
        steps = np.arange(1, len(r["losses"]) + 1)
        train_rel = np.sqrt(np.array(r["losses"]) / _y2mean(fn))
        ax.semilogy(steps, train_rel, color="tab:blue", lw=1.2)
        ax.semilogy(steps, r["rels"], color="tab:blue", lw=1.0, ls="--",
                    alpha=0.75)
        ax.axhline(r["floor"], color="k", ls=":", lw=1.2)
        ax.set_ylim(1e-16, 2.0)
        ax.set_title(fn, fontsize=10)
        ax.set_xlabel("NLCG iteration")
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel("relative $L_2$")
    handles = [Line2D([0], [0], color="tab:blue", lw=1.2,
                      label=r"train rel $L_2$ = $\sqrt{\mathrm{MSE}/\overline{y^2}}$"),
               Line2D([0], [0], color="tab:blue", lw=1.0, ls="--",
                      label="eval rel $L_2$"),
               Line2D([0], [0], color="k", ls=":", lw=1.2,
                      label="lstsq floor (init geometry, rel $L_2$)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.suptitle(f"expD08 -- QI init + zero readout, NLCG-ortho, N={N} "
                 f"(tether {tstr}, {iters} it) -- ALL curves rel $L_2$",
                 y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_scaling(rows, path, tether=None, iters=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    tstr = "auto" if tether is None else f"{tether:g}x"
    iters = ITERS if iters is None else iters
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, fn in zip(axes.flat, TARGETS):
        rs = sorted([r for r in rows if r["fn"] == fn], key=lambda r: r["N"])
        ns = [r["N"] for r in rs]
        ax.loglog(ns, [r["best_rel"] for r in rs], "-o", color="tab:blue",
                  ms=5, label="min eval rel $L_2$ (trained, all params free)")
        # min train rel L2 = sqrt(min MSE / mean y^2): shows the optimizer
        # stalls in TRAIN too (train ~= eval -- no generalization gap)
        ax.loglog(ns, [float(np.sqrt(np.min(r["losses"]) / _y2mean(fn)))
                       for r in rs], "-s", color="tab:cyan", ms=4, lw=1.0,
                  alpha=0.8, label="min train rel $L_2$")
        ax.loglog(ns, [r["floor"] for r in rs], "--", color="k", lw=1.5,
                  label="lstsq floor (frozen init geometry)")
        ax.set_ylim(1e-16, 1.0)
        ax.set_title(fn, fontsize=11)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"min eval rel $L_2$")
        ax.set_xticks(WIDTHS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle("expD08 scaling -- min relative $L_2$ vs width "
                 f"(QI init, zero readout, {iters} NLCG-ortho iterations, "
                 f"geometry tether {tstr})",
                 y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild all figures/gifs from persisted snapshots")
    ap.add_argument("--iteration", type=int, default=ITERATION,
                    help="which iteration_<k> results dir to render")
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.render_only:
        iter_conf = {1: (1.0, 1000), 2: (1e3, 3000), 3: (1e6, 3000),
                     4: (1e6, 3000), 5: (1e6, 3000), 6: (None, ITERS),
                     7: (None, ITERS), 8: (None, ITERS), 9: (None, ITERS)}
        tether_i, iters_i = iter_conf.get(args.iteration, (None, ITERS))
        out_dir = OUT_DIR.parent / f"iteration_{args.iteration}"
        rows = load_rows_from_snaps(out_dir)
        if not rows:
            print(f"No snapshots under {out_dir}/snaps/. Run training first.")
            sys.exit(1)
        render_loss_2x2(rows, 256, out_dir / "loss_curves_2x2_N256.png",
                        tether=tether_i, iters=iters_i)
        render_scaling(rows, out_dir / "scaling_2x3.png",
                       tether=tether_i, iters=iters_i)
        if not args.no_gifs:
            for N in GIF_WIDTHS:
                expd06.render_gif3(rows, SEED_NAME, N,
                                   out_dir / "gifs" / f"params_N{N}.gif",
                                   scatter=True)
        return

    widths = [32, 128] if args.smoke else WIDTHS
    fns = ["sine", "runge"] if args.smoke else list(TARGETS)
    iters = 200 if args.smoke else ITERS
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    jobs = [{"fn": f, "N": n, "iters": iters, "threads": threads,
             "snap_S": True}
            for f in fns for n in widths]
    print(f"expD08 | {len(jobs)} runs | widths={widths} | iters={iters} | "
          f"init=seed9 (QI geometry, zero readout) | opt=NLCG-ortho")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            s = r.get("stats", {})
            print(f"  [{i}/{len(jobs)}] N={r['N']:<4d} {r['fn']:13s} "
                  f"best_rel={r['best_rel']:.3e} floor={r['floor']:.1e} "
                  f"exh={s.get('exhaust', 0)} rst={s.get('full_reset', 0)} "
                  f"fb={s.get('alpha_fallback', 0)} rev={s.get('revert', 0)} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"training done in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = [{k: r[k] for k in ("fn", "N", "seed", "floor", "best_rel",
                               "losses", "rels")} for r in rows]
    (OUT_DIR / "expD08_rows.json").write_text(json.dumps(slim))

    render_loss_2x2(rows, 256 if not args.smoke else widths[-1],
                    OUT_DIR / "loss_curves_2x2_N256.png")
    render_scaling(rows, OUT_DIR / "scaling_2x3.png")
    for N in (GIF_WIDTHS if not args.smoke else widths):
        expd06.render_gif3(rows, SEED_NAME, N,
                           OUT_DIR / "gifs" / f"params_N{N}.gif",
                           scatter=True)
    print(f"done; outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
