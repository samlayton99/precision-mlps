"""dl_test -- do the suite winners survive contact with real deep learning?

Trains a small tanh MLP (d_in -> 64 -> 64 -> 1, fp64, FULL batch) on three
real regression tasks from the expF04 cache (airfoil, parkinsons,
bike_sharing) with the deployable suite winners in their genuinely
NONLINEAR forms:

  adam     torch Adam lr=1e-3 (the DL baseline)
  nlcg     nonlinear CG, PR+ with descent restarts, Armijo backtracking
           line search (trial step carried over), direction refresh every
           100 iterations -- the DL translation of the suite's nlcg_pr
  gd_bb    Barzilai-Borwein step, nonmonotone, safeguarded (clamped alpha,
           reset on non-finite) -- the DL translation of gd_bb

Budget is counted in function/gradient EVALS (each loss-only eval and each
loss+grad eval costs 1), so the line search is charged honestly. Records
train/test MSE at log-spaced checkpoints; writes rows JSON + one figure.

Usage: uv run python experiments/expD07_lstsq_optim_suite/dl_test/run.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CACHE = REPO_ROOT / "data" / "cache_all20"
OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD07_lstsq_optim_suite" / "dl_test")

TASKS = ["airfoil", "parkinsons", "bike_sharing"]
WIDTH, SEED, EVALS_BUDGET = 64, 0, 3000
# The DEPLOYABLE expD07 roster, each run STANDALONE from the same init
# (suite ids kept). SGD-family lrs use 1/L_hat with L_hat probed at init
# (finite-difference curvature along g) -- the DL analogue of the suite's
# 1/L rule. Two-stage designs are entries like any other.
OPTS = ["adam", "amsgrad", "sgd_mom09", "sgd_mom099", "nesterov",
        "gd_bb", "nlcg_pr", "adam_nlcgD", "padam_hb", "slim_rcg", "iter6",
        "tether_v6", "iter9", "iter11v", "iter11q"]
COLORS = {"adam": "tab:green", "amsgrad": "tab:purple",
          "sgd_mom09": "tab:orange", "sgd_mom099": "tab:cyan",
          "nesterov": "tab:brown", "gd_bb": "tab:pink",
          "nlcg_pr": "tab:olive", "adam_nlcgD": "tab:red",
          "padam_hb": "tab:blue", "slim_rcg": "black",
          "iter6": "magenta", "tether_v6": "teal",
          "iter9": "rebeccapurple", "iter11v": "navy",
          "iter11q": "crimson"}
ADAM_FRAC = 0.3          # hybrid stage 1 fraction of budget

torch.set_default_dtype(torch.float64)


class _BudgetDone(Exception):
    """Raised by the iter11 eval hook when the eval budget is exhausted."""


class MLP(nn.Module):
    def __init__(self, d_in, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def checkpoints(budget, n=50):
    pts = np.unique(np.round(np.logspace(0, np.log10(budget), n))).astype(int)
    return [0] + [int(p) for p in pts]


def make_funcs(model, xtr, ytr):
    params = list(model.parameters())

    def set_vec(vec):
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(vec, params)

    def f(vec):
        set_vec(vec)
        with torch.no_grad():
            return float(F.mse_loss(model(xtr), ytr))

    def fg(vec):
        set_vec(vec)
        for p in params:
            p.grad = None
        loss = F.mse_loss(model(xtr), ytr)
        loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in params]).detach().clone()
        return float(loss), g

    return set_vec, f, fg


def run_opt(opt_name, task, budget=EVALS_BUDGET):
    d = torch.load(CACHE / f"{task}.pt", weights_only=True)
    xtr = d["xtr"].double()
    ytr = d["ytr"].double()
    xte = d["xte"].double()
    yte = d["yte"].double()
    torch.manual_seed(SEED)
    model = MLP(d["d_in"], WIDTH)
    set_vec, f, fg = make_funcs(model, xtr, ytr)
    v = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    ckpts = checkpoints(budget)
    ptr, traj = [0], {"evals": [], "train_mse": [], "test_mse": []}

    def snap(evals, vec, force=False):
        due = ptr[0] < len(ckpts) and evals >= ckpts[ptr[0]]
        if not (due or force):
            return
        while ptr[0] < len(ckpts) and ckpts[ptr[0]] <= evals:
            ptr[0] += 1
        set_vec(vec)
        with torch.no_grad():
            traj["evals"].append(int(evals))
            traj["train_mse"].append(float(F.mse_loss(model(xtr), ytr)))
            traj["test_mse"].append(float(F.mse_loss(model(xte), yte)))

    t0 = time.time()
    evals = [0]

    def F_(vec):
        evals[0] += 1
        return f(vec)

    def FG(vec):
        evals[0] += 1
        return fg(vec)

    def stage_adam(v0, budget_end):
        """Returns (params_vector, frozen_diag D = sqrt(v_hat)+1e-8)."""
        set_vec(v0)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        step_n = 0
        while evals[0] < budget_end:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xtr), ytr)
            loss.backward()
            opt.step()
            evals[0] += 1
            step_n += 1
            if evals[0] >= ckpts[min(ptr[0], len(ckpts) - 1)]:
                vec = torch.nn.utils.parameters_to_vector(
                    params).detach().clone()
                snap(evals[0], vec)
        vhat = torch.cat([
            opt.state[p]["exp_avg_sq"].reshape(-1) for p in params
        ]) / (1.0 - 0.999 ** max(step_n, 1))     # bias-corrected
        dvec = torch.sqrt(vhat) + 1e-8
        return (torch.nn.utils.parameters_to_vector(params).detach().clone(),
                dvec)

    def stage_nlcg(v0, budget_end, dvec=None):
        """PR+ NLCG, Armijo backtracking with quadratic-interpolation trial
        step; line-search failure resets the direction (not the run).
        With dvec: runs in the frozen-diagonal metric (u = sqrt(D) * theta),
        i.e. NLCG on the Adam-preconditioned landscape."""
        if dvec is not None:
            dsq = torch.sqrt(dvec)

            def to_theta(u):
                return u / dsq

            def F_m(u):
                return F_(to_theta(u))

            def FG_m(u):
                loss_u, g_u = FG(to_theta(u))
                return loss_u, g_u / dsq
        else:
            def to_theta(u):
                return u

            F_m, FG_m = F_, FG
        vv = v0 if dvec is None else v0 * torch.sqrt(dvec)
        loss, g = FG_m(vv)
        d_dir = -g
        gg = float(g.dot(g))
        prev_drop = None
        it = 0
        while evals[0] < budget_end:
            it += 1
            dphi0 = float(g.dot(d_dir))
            if dphi0 >= 0.0:
                d_dir, dphi0 = -g, -gg
            # trial step: quadratic estimate from the previous decrease
            if prev_drop is not None and dphi0 < 0:
                a = min(2.0 * prev_drop / (-dphi0), 1e3)
                a = max(a, 1e-14)
            else:
                a = 1.0 / (1.0 + np.sqrt(gg))
            accepted = False
            for _ in range(30):
                if evals[0] >= budget_end:
                    break
                if F_m(vv + a * d_dir) <= loss + 1e-4 * a * dphi0:
                    accepted = True
                    break
                a *= 0.5
            if not accepted:
                if float(d_dir.dot(-g)) / max(np.sqrt(float(
                        d_dir.dot(d_dir)) * gg), 1e-300) > 0.99:
                    break                      # steepest descent also failed
                d_dir = -g                     # reset direction, retry
                prev_drop = None
                continue
            v_new = vv + a * d_dir
            loss_new, g_new = FG_m(v_new)
            prev_drop = max(loss - loss_new, 1e-300)
            beta = max(0.0, float(g_new.dot(g_new - g)) / gg)
            if it % 100 == 0:
                beta = 0.0
            d_dir = -g_new + beta * d_dir
            vv, loss, g, gg = v_new, loss_new, g_new, float(g_new.dot(g_new))
            snap(evals[0], to_theta(vv))
            if gg == 0.0:
                break
        return to_theta(vv)

    def stage_phb(v0, budget_end, dvec, beta=0.95):
        """Frozen-diagonal preconditioned heavy ball, NO line search:
        lr = (1+beta)/L_hat with L_hat a running (decayed-max) BB/Rayleigh
        curvature estimate in the D-metric, L_hat = max(.99 L_hat, s.y/s.Ds).
        Overshoot self-corrects: oscillation aligns s with the top mode,
        raising L_hat. Pure first-order, Adam-class memory."""
        vv = v0
        loss, g = FG(vv)
        v_prev = vv.clone()
        lhat = None
        lr = 1e-3                              # cautious first step
        best_v, best_loss = vv.clone(), loss
        window = [loss]
        while evals[0] < budget_end:
            v_new = vv - lr * (g / dvec) + beta * (vv - v_prev)
            loss_new, g_new = FG(v_new)
            # trust-window rejection: heavy ball is nonmonotone but bounded;
            # a 10x jump over the recent window means lr broke stability --
            # reject, shrink lr, kill momentum, raise the curvature estimate
            if not np.isfinite(loss_new) or loss_new > 10.0 * max(window):
                lr *= 0.25
                lhat = None if lhat is None else lhat * 4.0
                v_prev = vv.clone()
                if lr < 1e-18:
                    break
                continue
            s, yk = v_new - vv, g_new - g
            sds = float(s.dot(dvec * s))
            sy = float(s.dot(yk))
            if sy > 0 and sds > 0:
                ray = sy / sds
                lhat = ray if lhat is None else max(0.99 * lhat, ray)
                lr = (1.0 + beta) / lhat * 0.5
            v_prev, vv, g, loss = vv, v_new, g_new, loss_new
            window = (window + [loss])[-10:]
            if loss < best_loss:
                best_loss, best_v = loss, vv.clone()
            snap(evals[0], vv)
        return best_v

    def stage_bb(v0, budget_end):
        """Raydan's global BB: BB1 step + NONMONOTONE Armijo safeguard
        (reference window M=10)."""
        vv = v0
        loss, g = FG(vv)
        window = [loss]
        lr = 1.0 / (1.0 + np.sqrt(float(g.dot(g))))
        while evals[0] < budget_end:
            fmax = max(window[-10:])
            gg = float(g.dot(g))
            a = lr
            accepted = False
            for _ in range(15):
                if evals[0] >= budget_end:
                    break
                if F_(vv - a * g) <= fmax - 1e-4 * a * gg:
                    accepted = True
                    break
                a *= 0.5
            if not accepted:
                break
            v_new = vv - a * g
            loss_new, g_new = FG(v_new)
            s, yk = v_new - vv, g_new - g
            sy = float(s.dot(yk))
            lr = float(np.clip(float(s.dot(s)) / sy, 1e-12, 1e3)) \
                if sy > 0 else a
            vv, g, loss = v_new, g_new, loss_new
            window.append(loss)
            snap(evals[0], vv)
        return vv

    def probe_lipschitz(v0):
        """Local curvature along g by finite difference: L_hat =
        ||g(v - t g) - g(v)|| / (t ||g||). Costs 2 evals."""
        _, g0 = FG(v0)
        gn = float(torch.linalg.norm(g0))
        t = 1e-6 * (1.0 + float(torch.linalg.norm(v0))) / (1.0 + gn)
        _, g1 = FG(v0 - t * g0)
        lhat = float(torch.linalg.norm(g1 - g0)) / (t * gn + 1e-300)
        return max(lhat, 1e-12), g0

    def stage_sgdmom(v0, budget_end, beta):
        lhat, g = probe_lipschitz(v0)
        lr = 1.0 / lhat
        vv, v_prev = v0.clone(), v0.clone()
        loss = f(v0)
        window = [loss]
        while evals[0] < budget_end:
            v_new = vv - lr * g + beta * (vv - v_prev)
            loss_new, g_new = FG(v_new)
            if not np.isfinite(loss_new) or loss_new > 10.0 * max(window):
                lr *= 0.25                    # reject: stay at vv, kill mom
                v_prev = vv.clone()
                if lr < 1e-18:
                    break
                continue
            v_prev, vv, g = vv, v_new, g_new
            window = (window + [loss_new])[-10:]
            snap(evals[0], vv)
        return vv

    def stage_nesterov(v0, budget_end):
        lhat, g0 = probe_lipschitz(v0)
        lr = 1.0 / lhat
        x, x_prev = v0.clone(), v0.clone()
        x_good = v0.clone()
        window = [f(v0)]
        t_it = 0
        while evals[0] < budget_end:
            t_it += 1
            z = x + ((t_it - 1.0) / (t_it + 2.0)) * (x - x_prev)
            loss_z, g_z = FG(z)
            if not np.isfinite(loss_z) or loss_z > 10.0 * max(window):
                # revert to the last good iterate, restart the schedule
                x = x_good.clone()
                x_prev = x_good.clone()
                lr *= 0.25
                t_it = 0
                if lr < 1e-18:
                    break
                continue
            x_good = x.clone()   # current x validated by its z-eval
            x_prev = x
            x = z - lr * g_z
            window = (window + [loss_z])[-10:]
            snap(evals[0], x)
        return x

    def stage_amsgrad(v0, budget_end):
        set_vec(v0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
        while evals[0] < budget_end:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xtr), ytr)
            loss.backward()
            opt.step()
            evals[0] += 1
            vec = torch.nn.utils.parameters_to_vector(
                model.parameters()).detach().clone()
            snap(evals[0], vec)
        return torch.nn.utils.parameters_to_vector(
            model.parameters()).detach().clone()

    def stage_slim(v0, budget_end, refresh=200):
        """expD08 iteration_5 slim core, verbatim mechanics: carried
        residual (r_hat += alpha J d via one JVP), gradient = VJP seeded
        with (2/n) r_hat, Gauss-Newton exact steps, full orthogonalized
        gradient memory (double projection), exhaustion escape, NaN reject.
        No loss values are ever compared. No tether: standard DL init, so
        there is no trusted block to slow down (T=1 coordinates).
        Charged 1 eval per JVP, 1 per VJP, 1 per refresh forward pass."""
        from torch.func import functional_call
        from torch.func import jvp as t_jvp

        names, shapes, sizes = [], [], []
        for nm, p in model.named_parameters():
            names.append(nm)
            shapes.append(p.shape)
            sizes.append(p.numel())

        def unflat(vec):
            out, i = {}, 0
            for nm, sh, sz in zip(names, shapes, sizes):
                out[nm] = vec[i:i + sz].reshape(sh)
                i += sz
            return out

        def pred(vec):
            return functional_call(model, unflat(vec), (xtr,))

        n = xtr.shape[0]

        def f_resid(vec):
            evals[0] += 1
            with torch.no_grad():
                return (pred(vec) - ytr).detach()

        def f_jvp(vec, dvec):
            evals[0] += 1
            _, jd = t_jvp(pred, (vec,), (dvec,))
            return jd.detach()

        def f_vjp(vec, seed):
            evals[0] += 1
            vr = vec.detach().requires_grad_(True)
            (gg,) = torch.autograd.grad(pred(vr), vr, grad_outputs=seed)
            return gg.detach()

        theta = v0.clone()
        rhat = f_resid(theta)
        g = f_vjp(theta, (2.0 / n) * rhat)
        basis = []

        def project(vec):
            if not basis:
                return vec.clone()
            B = torch.stack(basis)
            out = vec - B.t().mv(B.mv(vec))
            return out - B.t().mv(B.mv(out))

        def append_to_basis(vec):
            vv = project(vec)
            n2 = float(vv.dot(vv))
            if n2 > 1e-300 and len(basis) < theta.numel():
                basis.append(vv / float(np.sqrt(n2)))

        gt = g.clone()
        d = -gt
        ggt = float(gt.dot(gt))
        lhat = None
        exhaust_streak = 0
        it = 0
        while evals[0] < budget_end:
            it += 1
            if it % refresh == 0:
                rhat = f_resid(theta)
                g = f_vjp(theta, (2.0 / n) * rhat)
                gt = project(g)
                ggt = float(gt.dot(gt))
            dn2 = float(d.dot(d))
            if dn2 <= 0.0 or ggt <= 0.0:
                break
            jd = f_jvp(theta, d)
            dhd = (2.0 / n) * float(jd.dot(jd))
            if dhd > 0 and np.isfinite(dhd):
                alpha = -float(g.dot(d)) / dhd
                lam = dhd / dn2
                lhat = lam if lhat is None else max(0.999 * lhat, lam)
            else:
                alpha = 1.0 / lhat if lhat is not None else 1e-6
            theta_new = theta + alpha * d
            rhat_new = rhat + alpha * jd
            g_new = f_vjp(theta_new, (2.0 / n) * rhat_new)
            if not np.isfinite(float(torch.linalg.norm(g_new))):
                d = -gt                        # sole safeguard: NaN reject
                continue
            append_to_basis(gt)
            gt_new = project(g_new)
            if float(gt_new.dot(gt_new)) < 1e-24 * float(g_new.dot(g_new)):
                exhaust_streak += 1
                gt_new = g_new.clone()
                if exhaust_streak >= 20:
                    basis = []
                    exhaust_streak = 0
                beta = 0.0
            else:
                exhaust_streak = 0
                beta = float(gt_new.dot(gt_new - gt)) / ggt if ggt > 0 else 0.0
            d = -gt_new + beta * d
            theta, rhat, g, gt = theta_new, rhat_new, g_new, gt_new
            ggt = float(gt.dot(gt))
            snap(evals[0], theta)
        return theta

    def stage_iter6(v0, budget_end, refresh=200, mode="ratchet"):
        """expD08 iteration_6: the slim core + the built-in auto-tether,
        with the SCALABLE per-tensor probe (sample a few coordinates per
        named tensor; parameters in one tensor share nonlinearity
        statistics). Probe forwards are charged against the eval budget.
        Honest expectation stated up front: the drift gauge was designed
        for trusted inits; on an untrusted DL init it may over-lock.

        mode="v6": tether_v6 continuous controller instead of the trip
        ratchet (final expD08 tether_v6 design). The gauge is READ every
        50 steps (one extra forward, charged; the fresh residual is
        DISCARDED, never written to r_hat); the reading is the running
        drift since the last replacement projected to one 200-step
        window. T updates in log space toward drift == target with
        target = max(BETA*min_so_far||r||, noise floor), BETA = 1e-6,
        exponent 0.5, rate limit 10x/window, ASYMMETRIC (relaxes 10x
        slower than it locks -- prevents breathing), clipped to
        [1, 1e12]. q ratios below the probe resolution (1e-10) are
        zeroed so the readout can never be tethered; the metric is
        re-applied (basis reset) only when T moved > 0.05 decades.
        T readings recorded in traj["v6_T"]."""
        from torch.func import functional_call
        from torch.func import jvp as t_jvp

        names, shapes, sizes, offs = [], [], [], []
        off = 0
        for nm, p in model.named_parameters():
            names.append(nm)
            shapes.append(p.shape)
            sizes.append(p.numel())
            offs.append(off)
            off += p.numel()
        m = off

        def unflat(vec):
            out = {}
            for nm, sh, sz, o in zip(names, shapes, sizes, offs):
                out[nm] = vec[o:o + sz].reshape(sh)
            return out

        def pred(vec):
            return functional_call(model, unflat(vec), (xtr,))

        n = xtr.shape[0]
        y_norm = float(torch.linalg.norm(ytr))

        def probe_per_tensor(theta_raw, n_samp=6, eps_rel=1e-3):
            """Per-tensor nonlinearity score: max sampled per-coordinate
            second difference, at theta and one perturbed point."""
            rng = np.random.default_rng(0)
            scores = torch.zeros(m, dtype=theta_raw.dtype)
            for base in (theta_raw,
                         theta_raw + 0.1 * torch.clamp(
                             theta_raw.abs(), min=1.0)
                         * torch.from_numpy(
                             rng.integers(0, 2, m) * 2.0 - 1.0)):
                with torch.no_grad():
                    p0 = pred(base)
                    evals[0] += 1
                    for sz, o in zip(sizes, offs):
                        idx = rng.choice(sz, size=min(n_samp, sz),
                                         replace=False)
                        qt = 0.0
                        for i in idx:
                            i = int(o + i)
                            s = max(abs(float(base[i])), 1.0)
                            eps = eps_rel * s
                            tp = base.clone()
                            tp[i] += eps
                            tm = base.clone()
                            tm[i] -= eps
                            d2 = pred(tp) - 2.0 * p0 + pred(tm)
                            evals[0] += 2
                            qt = max(qt, float(torch.linalg.norm(d2))
                                     / eps ** 2 * s ** 2)
                        scores[o:o + sz] = torch.maximum(
                            scores[o:o + sz],
                            torch.full((sz,), qt,
                                       dtype=theta_raw.dtype))
            return scores

        S = torch.ones(m, dtype=v0.dtype)
        phi = v0.clone()
        q_acc = None
        T = 1.0
        v6_logT_probe = 0.0

        def raw(p_):
            return p_ / S

        def f_resid(vec):
            evals[0] += 1
            with torch.no_grad():
                return (pred(vec) - ytr).detach()

        def f_jvp(vec, dvec):
            evals[0] += 1
            _, jd = t_jvp(pred, (vec,), (dvec,))
            return jd.detach()

        def f_vjp(vec, seed):
            evals[0] += 1
            vr = vec.detach().requires_grad_(True)
            (gg,) = torch.autograd.grad(pred(vr), vr, grad_outputs=seed)
            return gg.detach()

        rhat = f_resid(raw(phi))
        g = f_vjp(raw(phi), (2.0 / n) * rhat) / S
        basis = []

        def project(vec):
            if not basis:
                return vec.clone()
            B = torch.stack(basis)
            out = vec - B.t().mv(B.mv(vec))
            return out - B.t().mv(B.mv(out))

        def append_to_basis(vec):
            if mode == "iter9":      # iteration_9: no memory basis, ever
                return
            vv = project(vec)
            n2 = float(vv.dot(vv))
            if n2 > 1e-300 and len(basis) < m:
                basis.append(vv / float(np.sqrt(n2)))

        gt = g.clone()
        d = -gt
        ggt = float(gt.dot(gt))
        lhat = None
        exhaust_streak = 0
        it = 0
        v6_read, v6_beta, v6_down = 50, 1e-6, 0.1
        v6_db, v6_lrho = 0.05, 1.0    # deadband decades, log10(RHO)
        T_ctrl, it_lr, it_lrep = 1.0, 0, 0
        rnorm_min = float("inf")
        while evals[0] < budget_end:
            it += 1
            is_replace = it % refresh == 0
            if mode == "v6" and (is_replace or it % v6_read == 0):
                r_fresh = f_resid(raw(phi))
                drift = float(torch.linalg.norm(rhat - r_fresh))
                rnorm_min = min(rnorm_min,
                                float(torch.linalg.norm(r_fresh)))
                noise_floor = 10.0 * 2.22e-16 * y_norm
                steps = max(it - it_lr, 1)
                it_lr = it
                elapsed = max(it - it_lrep, 1)
                reading = drift * (refresh / elapsed)
                target = max(v6_beta * rnorm_min, noise_floor)
                stp = 0.5 * (steps / refresh) * np.log10(
                    max(reading, 1e-300) / target)
                lim = (steps / refresh) * v6_lrho
                stp = min(max(stp, -lim), lim)
                if stp < 0:
                    stp *= v6_down
                T_ctrl = min(max(T_ctrl * 10.0 ** stp, 1.0), 1e12)
                if is_replace:
                    rhat = r_fresh
                    it_lrep = it
                changed = abs(np.log10(max(T_ctrl, 1e-300))
                              - np.log10(max(T, 1e-300))) > v6_db
                if changed and (q_acc is None or np.log10(
                        max(T_ctrl, 1.0)) - v6_logT_probe >= 1.0):
                    q = probe_per_tensor(raw(phi))
                    q_acc = (q if q_acc is None
                             else torch.maximum(q_acc, q))
                    v6_logT_probe = np.log10(max(T_ctrl, 1.0))
                if changed and q_acc is not None:
                    T = T_ctrl
                    qm = float(q_acc.max())
                    qh = (q_acc / qm if qm > 0
                          else torch.zeros_like(q_acc))
                    qh[qh < 1e-10] = 0.0
                    if float(qh.max()) > 0:
                        S_new = 1.0 + T * qh
                        phi = phi * (S_new / S)
                        S = S_new
                        basis = []
                        lhat = None
                traj.setdefault("v6_T", []).append((evals[0], T_ctrl))
                if is_replace or changed:
                    g = f_vjp(raw(phi), (2.0 / n) * rhat) / S
                    gt = project(g)
                    ggt = float(gt.dot(gt))
                    if changed:
                        d = -gt
            elif mode != "v6" and is_replace:
                r_fresh = f_resid(raw(phi))
                drift = float(torch.linalg.norm(rhat - r_fresh))
                rnorm = float(torch.linalg.norm(r_fresh))
                rhat = r_fresh
                noise_floor = 10.0 * 2.22e-16 * y_norm
                if mode == "iter9":  # noise-floor trip, hard T ceiling
                    tripped = drift > noise_floor and T < 1e12
                else:
                    tripped = drift > max(0.01 * rnorm, noise_floor)
                if q_acc is None or tripped:
                    q = probe_per_tensor(raw(phi))
                    q_acc = (q if q_acc is None
                             else torch.maximum(q_acc, q))
                    if mode == "iter9" and float(q_acc.max()) > 0:
                        q_acc[q_acc < 1e-8 * float(q_acc.max())] = 0.0
                if tripped and float(q_acc.max()) > 0:
                    T = (min(T * 1e4, 1e12) if mode == "iter9"
                         else T * 100.0)
                    S_new = 1.0 + T * (q_acc / float(q_acc.max()))
                    phi = phi * (S_new / S)
                    S = S_new
                    basis = []
                    lhat = None
                g = f_vjp(raw(phi), (2.0 / n) * rhat) / S
                gt = project(g)
                ggt = float(gt.dot(gt))
                d = -gt
            dn2 = float(d.dot(d))
            if dn2 <= 0.0 or ggt <= 0.0:
                break
            u = f_jvp(raw(phi), d / S)
            c = (2.0 / n) * float(u.dot(u))
            if c > 0 and np.isfinite(c):
                alpha = -float(g.dot(d)) / c
                lam = c / dn2
                lhat = lam if lhat is None else max(0.999 * lhat, lam)
            else:
                alpha = 1.0 / lhat if lhat is not None else 1e-6
            phi_new = phi + alpha * d
            rhat_new = rhat + alpha * u
            g_new = f_vjp(raw(phi_new), (2.0 / n) * rhat_new) / S
            if not np.isfinite(float(torch.linalg.norm(g_new))):
                d = -gt
                continue
            append_to_basis(gt)
            gt_new = project(g_new)
            if float(gt_new.dot(gt_new)) < 1e-24 * float(g_new.dot(g_new)):
                exhaust_streak += 1
                gt_new = g_new.clone()
                if exhaust_streak >= 20:
                    basis = []
                    exhaust_streak = 0
                beta = 0.0
            else:
                exhaust_streak = 0
                beta = (float(gt_new.dot(gt_new - gt)) / ggt
                        if ggt > 0 else 0.0)
            d = -gt_new + beta * d
            phi, rhat, g, gt = phi_new, rhat_new, g_new, gt_new
            ggt = float(gt.dot(gt))
            snap(evals[0], raw(phi))
        return raw(phi)

    def stage_iter11(v0, budget_end, arm):
        """expD08 iteration_11: two-regime certify-and-solve (stock Adam
        base + discovered-linear exploit set, arm = varpro | qn). All
        forwards/JVPs/VJPs -- including the probe and the varpro solve's
        B Jacobian columns -- are charged against the eval budget; the
        run stops via the eval hook when the budget is exhausted."""
        import importlib.util as ilu
        d08_dir = REPO_ROOT / "experiments" / "expD08_qi_init_nlcg"
        sys.path.insert(0, str(d08_dir))
        spec = ilu.spec_from_file_location("expd08_iter11",
                                           d08_dir / "iter11.py")
        it11 = ilu.module_from_spec(spec)
        spec.loader.exec_module(it11)
        from torch.func import functional_call
        from torch.func import jvp as t_jvp

        names, shapes, sizes, offs = [], [], [], []
        off = 0
        for nm, p in model.named_parameters():
            names.append(nm)
            shapes.append(p.shape)
            sizes.append(p.numel())
            offs.append(off)
            off += p.numel()
        m = off

        def unflat(vec):
            return {nm: vec[o:o + sz].reshape(sh) for nm, sh, sz, o
                    in zip(names, shapes, sizes, offs)}

        def pred(vec):
            return functional_call(model, unflat(vec), (xtr,))

        n = xtr.shape[0]
        y_norm = float(torch.linalg.norm(ytr))

        def f_pred(vec):
            evals[0] += 1
            return pred(vec)

        def f_resid(vec):
            evals[0] += 1
            with torch.no_grad():
                return (pred(vec) - ytr).detach()

        def f_jvp(vec, dvec):
            evals[0] += 1
            _, jd = t_jvp(pred, (vec,), (dvec,))
            return jd.detach()

        def f_vjp(vec, seed_):
            evals[0] += 1
            vr = vec.detach().requires_grad_(True)
            (gg,) = torch.autograd.grad(pred(vr), vr, grad_outputs=seed_)
            return gg.detach()

        def f_residgrad(vec):
            """Fused fresh residual + gradient: one charged eval, same
            as the other stages' loss.backward()."""
            evals[0] += 1
            vr = vec.detach().requires_grad_(True)
            pr = pred(vr)
            r = (pr - ytr).detach()
            (gg,) = torch.autograd.grad(pr, vr,
                                        grad_outputs=(2.0 / n) * r)
            return r, gg.detach()

        def probe_per_tensor(theta_raw, n_samp=6, eps_rel=1e-3):
            rng = np.random.default_rng(0)
            scores = torch.zeros(m, dtype=theta_raw.dtype)
            for base in (theta_raw,
                         theta_raw + 0.1 * torch.clamp(
                             theta_raw.abs(), min=1.0)
                         * torch.from_numpy(
                             rng.integers(0, 2, m) * 2.0 - 1.0)):
                with torch.no_grad():
                    p0 = pred(base)
                    evals[0] += 1
                    for sz, o in zip(sizes, offs):
                        idx = rng.choice(sz, size=min(n_samp, sz),
                                         replace=False)
                        qt = 0.0
                        for i in idx:
                            i = int(o + i)
                            s = max(abs(float(base[i])), 1.0)
                            eps = eps_rel * s
                            tp = base.clone()
                            tp[i] += eps
                            tm = base.clone()
                            tm[i] -= eps
                            d2 = pred(tp) - 2.0 * p0 + pred(tm)
                            evals[0] += 2
                            qt = max(qt, float(torch.linalg.norm(d2))
                                     / eps ** 2 * s ** 2)
                        scores[o:o + sz] = torch.maximum(
                            scores[o:o + sz],
                            torch.full((sz,), qt, dtype=theta_raw.dtype))
            return scores

        last = {"v": v0.detach().clone()}

        def eval_hook(vec):
            last["v"] = vec.detach().clone()
            snap(evals[0], vec)
            if evals[0] >= budget_end:
                raise _BudgetDone
            return 0.0

        def get_wbv_dummy(vec):
            return {"w": np.zeros(1), "b": np.zeros(1), "v": np.zeros(1)}

        try:
            it11.train_iter11(
                v0.clone(), f_resid, f_jvp, f_vjp, f_pred, n,
                iters=2 * budget_end, frames_at={0},
                get_wbv=get_wbv_dummy, eval_rel=eval_hook, y_norm=y_norm,
                refresh=200, arm=arm, probe_fn=probe_per_tensor,
                f_residgrad=f_residgrad)
        except _BudgetDone:
            pass
        return last["v"]

    snap(0, v)
    if opt_name == "adam":
        v, _ = stage_adam(v, budget)
        snap(evals[0], v, force=True)
    elif opt_name == "amsgrad":
        v = stage_amsgrad(v, budget)
        snap(evals[0], v, force=True)
    elif opt_name == "sgd_mom09":
        v = stage_sgdmom(v, budget, 0.9)
    elif opt_name == "sgd_mom099":
        v = stage_sgdmom(v, budget, 0.99)
    elif opt_name == "nesterov":
        v = stage_nesterov(v, budget)
    elif opt_name == "nlcg_pr":
        v = stage_nlcg(v, budget)
    elif opt_name == "gd_bb":
        v = stage_bb(v, budget)
    elif opt_name == "adam_nlcgD":
        v, dvec = stage_adam(v, int(ADAM_FRAC * budget))
        snap(evals[0], v, force=True)
        v = stage_nlcg(v, budget, dvec=dvec)
        snap(evals[0], v, force=True)
    elif opt_name == "padam_hb":
        v, dvec = stage_adam(v, int(ADAM_FRAC * budget))
        snap(evals[0], v, force=True)
        v = stage_phb(v, budget, dvec)
        snap(evals[0], v, force=True)
    elif opt_name == "slim_rcg":
        v = stage_slim(v, budget)
        snap(evals[0], v, force=True)
    elif opt_name == "iter6":
        v = stage_iter6(v, budget)
        snap(evals[0], v, force=True)
    elif opt_name == "tether_v6":
        v = stage_iter6(v, budget, mode="v6")
        snap(evals[0], v, force=True)
    elif opt_name == "iter9":
        v = stage_iter6(v, budget, mode="iter9")
        snap(evals[0], v, force=True)
    elif opt_name in ("iter11v", "iter11q"):
        v = stage_iter11(v, budget,
                         arm=("varpro" if opt_name == "iter11v" else "qn"))
        snap(evals[0], v, force=True)
    else:
        raise ValueError(opt_name)

    return {"task": task, "opt": opt_name, "traj": traj,
            "best_train": float(np.min(traj["train_mse"])),
            "best_test": float(np.min(traj["test_mse"])),
            "wall_s": round(time.time() - t0, 1)}


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(TASKS), figsize=(4.8 * len(TASKS), 7.4))
    for j, task in enumerate(TASKS):
        for i, key in enumerate(["train_mse", "test_mse"]):
            ax = axes[i, j]
            for r in rows:
                if r["task"] != task:
                    continue
                t = r["traj"]
                steps = np.maximum(np.array(t["evals"], dtype=float), 1.0)
                vals = np.minimum.accumulate(np.array(t[key]))
                ax.loglog(steps, vals, color=COLORS[r["opt"]], label=r["opt"])
            ax.set_title(f"{task} -- {'train' if i == 0 else 'test'} MSE",
                         fontsize=10)
            ax.set_xlabel("function/grad evals")
            ax.set_ylabel("best-so-far MSE")
            ax.grid(True, alpha=0.3, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle("expD07/dl_test -- fp64 full-batch MLP (d->64->64->1 tanh) "
                 "on real regression tasks", y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.945), ncol=len(labels), frameon=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "dl_curves.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these optimizers; merge into rows.json")
    args = ap.parse_args()
    opts = args.only if args.only else OPTS

    rows = []
    if args.only and (OUT_DIR / "rows.json").exists():
        rows = [r for r in json.loads((OUT_DIR / "rows.json").read_text())
                if r["opt"] not in opts]
    for task in TASKS:
        for opt in opts:
            r = run_opt(opt, task)
            rows.append(r)
            print(f"{task:14s} {opt:6s} best_train={r['best_train']:.4e} "
                  f"best_test={r['best_test']:.4e}  ({r['wall_s']}s)",
                  flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rows.json").write_text(json.dumps(rows))
    make_figure(rows)


if __name__ == "__main__":
    main()
