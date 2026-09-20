"""expD07 harness -- run every registered optimizer over the problem registry.

Order is problem-major: all pending optimizers on one problem (in parallel
across a process pool), then the next problem. Every finished run is appended
to runs.jsonl -- the cache: (problem_id, opt_id) pairs already present are
skipped, so appending optimizers to optimizers.py later only runs the new
pairs and the figures re-aggregate everything. Figures are regenerated as runs
finish (throttled to one redraw per few seconds).

Note: the cache key does NOT include --steps; rerunning with a different
budget mixes budgets. Delete runs.jsonl (or the affected lines) to redo.

Usage:
    uv run python experiments/expD07_lstsq_optim_suite/build_problems.py  # once
    uv run python experiments/expD07_lstsq_optim_suite/run.py [--steps 20000]
        [--workers 6] [--plot-only]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    FIG_DIR, MANIFEST_PATH, append_run, digits, load_manifest, load_problem,
    load_runs, rel_excess,
)
from optimizers import OPTIMIZERS, make_optimizer  # noqa: E402
import plotting  # noqa: E402

STEPS_DEFAULT = 20000
N_CHECKPOINTS = 60
PLOT_EVERY_S = 5.0


def checkpoint_steps(total: int, n: int = N_CHECKPOINTS) -> list[int]:
    pts = np.unique(np.round(np.logspace(0, np.log10(total), n))).astype(int)
    return [0] + [int(p) for p in pts if p <= total] + (
        [total] if total not in pts else [])


def optimize(A, y, entry, spec, steps, A_ev=None, y_ev=None) -> dict:
    """One full-batch run from v0 = 0. A, y are fp64 torch tensors.

    traj["step"] counts GRAD EVALS: 1 per gradient step, 1 per CGLS
    iteration (2 matvecs, same cost), 1 per L-BFGS closure call.
    """
    R, cols = A.shape
    lipschitz = 2.0 * entry["smax"] ** 2 / R
    ckpts = checkpoint_steps(steps)
    l_star, l0 = entry["L_star"], entry["L0"]
    is_phi = entry["kind"] == "phi"
    if is_phi:
        y_ev_norm = float(torch.linalg.norm(y_ev))
    traj = {"step": [], "train_mse": [], "rel_l2": [], "rel_excess": []}
    if is_phi:
        traj["eval_rel_l2"] = []
        traj["eval_linf"] = []
    ptr = [0]

    def snap(evals, x, force=False):
        due = ptr[0] < len(ckpts) and evals >= ckpts[ptr[0]]
        if not (due or force):
            return
        while ptr[0] < len(ckpts) and ckpts[ptr[0]] <= evals:
            ptr[0] += 1
        with torch.no_grad():
            r = A.mv(x) - y
            mse = float(r.dot(r)) / R
            traj["step"].append(int(evals))
            traj["train_mse"].append(mse)
            # train relative L2: ||Av - y|| / ||y|| = sqrt(mse / L0)
            traj["rel_l2"].append(float(np.sqrt(mse / l0)))
            traj["rel_excess"].append(rel_excess(mse, l_star, l0))
            if is_phi:
                e = A_ev.mv(x) - y_ev
                traj["eval_rel_l2"].append(
                    float(torch.linalg.norm(e)) / y_ev_norm)
                traj["eval_linf"].append(float(e.abs().max()))

    def grad(x):
        return (2.0 / R) * A.t().mv(A.mv(x) - y)

    t0 = time.time()
    family = spec["family"]

    if family in ("sgd", "adam"):
        v = torch.zeros(cols, dtype=torch.float64, requires_grad=True)
        opt = make_optimizer(spec, [v], lipschitz)
        snap(0, v)
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            r = A.mv(v) - y
            loss = r.dot(r) / R
            loss.backward()
            opt.step()
            snap(step, v)

    elif family == "nesterov":
        # accelerated gradient, convex schedule beta_t = (t-1)/(t+2)
        lr = spec.get("lr_rel", 1.0) / lipschitz
        x = torch.zeros(cols, dtype=torch.float64)
        x_prev = x.clone()
        snap(0, x)
        for t in range(1, steps + 1):
            z = x + ((t - 1.0) / (t + 2.0)) * (x - x_prev)
            x_prev = x
            x = z - lr * grad(z)
            snap(t, x)

    elif family == "bb":
        # gradient descent with the Barzilai-Borwein (BB1) step,
        # safeguarded back to 1/L when s.y <= 0 or non-finite
        x = torch.zeros(cols, dtype=torch.float64)
        g = grad(x)
        lr = 1.0 / lipschitz
        snap(0, x)
        for t in range(1, steps + 1):
            x_new = x - lr * g
            g_new = grad(x_new)
            s, yk = x_new - x, g_new - g
            sy = float(s.dot(yk))
            lr = float(s.dot(s)) / sy if (sy > 0 and np.isfinite(sy)) \
                else 1.0 / lipschitz
            x, g = x_new, g_new
            snap(t, x)

    elif family == "cgls":
        # CG on the normal equations (the Krylov-optimal quadratic solver)
        x = torch.zeros(cols, dtype=torch.float64)
        r = y.clone()
        s = A.t().mv(r)
        p = s.clone()
        gamma = float(s.dot(s))
        snap(0, x)
        for t in range(1, steps + 1):
            q = A.mv(p)
            qq = float(q.dot(q))
            if qq <= 0.0 or gamma <= 0.0 or not np.isfinite(qq):
                snap(t, x, force=True)
                break
            alpha = gamma / qq
            x = x + alpha * p
            r = r - alpha * q
            s = A.t().mv(r)
            gamma_new = float(s.dot(s))
            p = s + (gamma_new / gamma) * p
            gamma = gamma_new
            snap(t, x)

    elif family == "nlcg":
        # nonlinear CG, Polak-Ribiere+ with restarts; exact line search on
        # the quadratic (in DL this would be a secant search costing ~1
        # extra eval, so each iteration is charged 2 grad evals).
        # spec["restart"] = K > 0: every K iterations reset the direction
        # AND recompute the residual from scratch (kills recurrence drift --
        # the fp64 orthogonality-loss fix, iterative-refinement style).
        restart_k = spec.get("restart", 0)
        x = torch.zeros(cols, dtype=torch.float64)
        r = -y.clone()                        # r = A x - y
        g = (2.0 / R) * A.t().mv(r)
        d = -g
        gg = float(g.dot(g))
        snap(0, x)
        it = 0
        while 2 * (it + 1) <= steps:
            it += 1
            Ad = A.mv(d)
            denom = float(Ad.dot(Ad))
            if denom <= 0.0 or not np.isfinite(denom):
                snap(2 * it, x, force=True)
                break
            alpha = -float(r.dot(Ad)) / denom
            x = x + alpha * d
            r = r + alpha * Ad
            if restart_k and it % restart_k == 0:
                r = A.mv(x) - y               # fresh residual
                g_new = (2.0 / R) * A.t().mv(r)
                d = -g_new
            else:
                g_new = (2.0 / R) * A.t().mv(r)
                beta = max(0.0, float(g_new.dot(g_new - g)) / gg)  # PR+
                d = -g_new + beta * d
                if float(g_new.dot(d)) >= 0.0:  # not descent -> restart
                    d = -g_new
            g, gg = g_new, float(g_new.dot(g_new))
            snap(2 * it, x)
            if gg == 0.0:
                break

    elif family == "cgls_refine":
        # CGLS with iterative-refinement restarts: every `cycle` iterations
        # fold d into x, recompute the residual EXACTLY, and restart CGLS on
        # the residual system. Resets fp64 recurrence drift each cycle.
        cycle = spec.get("cycle", 300)
        x = torch.zeros(cols, dtype=torch.float64)
        snap(0, x)
        t = 0
        done = False
        while t < steps and not done:
            r0 = y - A.mv(x)                  # fresh residual, exact
            d = torch.zeros(cols, dtype=torch.float64)
            r = r0.clone()
            s = A.t().mv(r)
            p = s.clone()
            gamma = float(s.dot(s))
            for _ in range(min(cycle, steps - t)):
                t += 1
                q = A.mv(p)
                qq = float(q.dot(q))
                if qq <= 0.0 or gamma <= 0.0 or not np.isfinite(qq):
                    done = True
                    snap(t, x + d, force=True)
                    break
                alpha = gamma / qq
                d = d + alpha * p
                r = r - alpha * q
                s = A.t().mv(r)
                gamma_new = float(s.dot(s))
                p = s + (gamma_new / gamma) * p
                gamma = gamma_new
                snap(t, x + d)
            x = x + d

    elif family == "cgls_reortho":
        # CGLS with FULL reorthogonalization of the A^T-residuals against
        # the stored Krylov basis (the memory-heavy ceiling reference:
        # proves the plain-CGLS stall is orthogonality loss). Extra cost
        # ~1 matvec-equivalent per iteration -> charged 2 evals/iter.
        x = torch.zeros(cols, dtype=torch.float64)
        r = y.clone()
        s = A.t().mv(r)
        basis = [s / torch.linalg.norm(s)]
        p = s.clone()
        gamma = float(s.dot(s))
        snap(0, x)
        it = 0
        while 2 * (it + 1) <= steps:
            it += 1
            q = A.mv(p)
            qq = float(q.dot(q))
            if qq <= 0.0 or gamma <= 1e-300 or not np.isfinite(qq):
                snap(2 * it, x, force=True)
                break
            alpha = gamma / qq
            x = x + alpha * p
            r = r - alpha * q
            s = A.t().mv(r)
            B = torch.stack(basis)            # [k, cols]
            s = s - B.t().mv(B.mv(s))         # two-pass MGS-lite
            s = s - B.t().mv(B.mv(s))
            gamma_new = float(s.dot(s))
            if gamma_new <= 1e-300:
                snap(2 * it, x, force=True)
                break
            if len(basis) < cols:
                basis.append(s / float(np.sqrt(gamma_new)))
            p = s + (gamma_new / gamma) * p
            gamma = gamma_new
            snap(2 * it, x)

    elif family == "bb_rms":
        # frozen RMS diagonal + BB: warmup accumulates an AMSGrad-style
        # max of g^2 along a short GD trajectory, then freezes the diagonal
        # D = sqrt(vmax) (Adam-like memory: one accumulator) and runs BB in
        # the D-preconditioned metric: x+ = x - alpha * g/D with
        # alpha = (s^T D s)/(s^T y). Frozen P > 0 -> linear convergence.
        warmup = min(spec.get("warmup", 100), steps // 2)
        x = torch.zeros(cols, dtype=torch.float64)
        vmax = torch.zeros(cols, dtype=torch.float64)
        snap(0, x)
        g = grad(x)
        for t in range(1, warmup + 1):
            vmax = torch.maximum(vmax, g * g)
            x = x - (1.0 / lipschitz) * g
            g = grad(x)
            snap(t, x)
        dpre = torch.sqrt(vmax)
        dpre = torch.clamp(dpre, min=1e-12 * float(dpre.max()) + 1e-300)
        lr = float(dpre.min()) / lipschitz    # safe start in the D-metric
        for t in range(warmup + 1, steps + 1):
            x_new = x - lr * (g / dpre)
            g_new = grad(x_new)
            s, yk = x_new - x, g_new - g
            sy = float(s.dot(yk))
            lr = float(s.dot(dpre * s)) / sy if (sy > 0 and np.isfinite(sy)) \
                else float(dpre.min()) / lipschitz
            x, g = x_new, g_new
            snap(t, x)

    elif family == "lsqr":
        # LSQR (Paige-Saunders): Golub-Kahan bidiagonalization, solves the
        # least-squares problem without forming normal equations -- the
        # numerically stable Krylov reference. 2 matvecs = 1 eval per iter.
        x = torch.zeros(cols, dtype=torch.float64)
        beta = float(torch.linalg.norm(y))
        u = y / beta
        v = A.t().mv(u)
        alpha = float(torch.linalg.norm(v))
        v = v / alpha
        w = v.clone()
        phibar, rhobar = beta, alpha
        snap(0, x)
        for t in range(1, steps + 1):
            u = A.mv(v) - alpha * u
            beta = float(torch.linalg.norm(u))
            if beta > 0:
                u = u / beta
            vv = A.t().mv(u) - beta * v
            alpha = float(torch.linalg.norm(vv))
            if alpha > 0:
                v = vv / alpha
            rho = float(np.hypot(rhobar, beta))
            c, s_rot = rhobar / rho, beta / rho
            theta = s_rot * alpha
            rhobar = -c * alpha
            phi = c * phibar
            phibar = s_rot * phibar
            x = x + (phi / rho) * w
            w = v - (theta / rho) * w
            snap(t, x)
            if beta == 0.0 or alpha == 0.0:
                snap(t, x, force=True)
                break

    elif family == "lbfgs":
        v = torch.zeros(cols, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS(
            [v], lr=1.0, max_iter=spec.get("max_iter", 50),
            history_size=spec.get("history", 100),
            line_search_fn="strong_wolfe",
            tolerance_grad=0.0, tolerance_change=0.0)
        n_evals = [0]

        def closure():
            n_evals[0] += 1
            opt.zero_grad(set_to_none=True)
            r = A.mv(v) - y
            loss = r.dot(r) / R
            loss.backward()
            return loss

        snap(0, v)
        prev_loss, stalled = None, 0
        while n_evals[0] < steps:
            loss = float(opt.step(closure))
            snap(min(n_evals[0], steps), v, force=True)
            if not np.isfinite(loss):
                break
            stalled = stalled + 1 if (prev_loss is not None
                                      and loss >= prev_loss) else 0
            if stalled >= 2:   # fully converged / line search dead
                break
            prev_loss = loss

    else:
        raise ValueError(f"Unknown optimizer family: {family}")

    rec = {
        "problem_id": entry["id"], "opt_id": spec["id"], "kind": entry["kind"],
        "cls": entry["cls"], "N": entry["N"], "R": entry["R"], "steps": steps,
        "final_rel_l2": traj["rel_l2"][-1],
        "floor_rel_l2": float(np.sqrt(l_star / l0)),
        "final_rel_excess": traj["rel_excess"][-1],
        "digits": digits(traj["rel_excess"][-1]),
        "wall_s": round(time.time() - t0, 2), "traj": traj,
    }
    if is_phi:
        rec["final_eval_rel_l2"] = traj["eval_rel_l2"][-1]
        rec["final_eval_linf"] = traj["eval_linf"][-1]
        rec["floor_eval_rel_l2"] = entry["floor_eval_rel_l2"]
        rec["floor_eval_linf"] = entry.get("floor_eval_linf")
        rec["ratio_to_floor"] = (
            rec["final_eval_rel_l2"] / entry["floor_eval_rel_l2"])
    return rec


def run_single(entry: dict, spec: dict, steps: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    prob = load_problem(entry)
    tens = {k: torch.from_numpy(np.ascontiguousarray(val))
            for k, val in prob.items()}
    return optimize(tens["A"], tens["y"], entry, spec, steps,
                    A_ev=tens.get("A_ev"), y_ev=tens.get("y_ev"))


def _summary(runs):
    print("\nSUMMARY (geo-mean BEST-over-trajectory relative L2 -- "
          "synthetic train | phi eval):")
    ids = []
    for r in runs:
        if r["opt_id"] not in ids:
            ids.append(r["opt_id"])

    def _geo(vals):
        return float(np.exp(np.mean(np.log(np.maximum(vals, 1e-16)))))

    def _best(r, key):
        return min(r["traj"][key])

    for oid in ids:
        syn = [_best(r, "rel_l2") for r in runs
               if r["opt_id"] == oid and r["kind"] == "synthetic"]
        phi = [_best(r, "eval_rel_l2") for r in runs
               if r["opt_id"] == oid and r["kind"] == "phi"]
        s = f"{_geo(syn):.3g}" if syn else "--"
        p = f"{_geo(phi):.3g}" if phi else "--"
        print(f"  {oid:16s} syn_best_rel_l2={s}  phi_best_eval_rel_l2={p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print("No manifest. Run build_problems.py first.")
        sys.exit(1)
    manifest = load_manifest()
    runs = load_runs()
    if args.plot_only:
        plotting.make_all(runs, FIG_DIR, manifest)
        _summary(runs)
        return

    done = {(r["problem_id"], r["opt_id"]) for r in runs}
    todo = [(entry, spec) for entry in manifest for spec in OPTIMIZERS
            if (entry["id"], spec["id"]) not in done]
    print(f"{len(manifest)} problems x {len(OPTIMIZERS)} optimizers | "
          f"{len(done)} cached, {len(todo)} to run | steps={args.steps}")
    if not todo:
        plotting.make_all(runs, FIG_DIR, manifest)
        _summary(runs)
        return

    threads = max(1, (os.cpu_count() or 4) // args.workers)
    n_done, t0, last_plot = 0, time.time(), 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # problem-major: finish all optimizers on one problem, then move on
        for entry in manifest:
            pending = [s for s in OPTIMIZERS
                       if (entry["id"], s["id"]) not in done]
            if not pending:
                continue
            futures = {pool.submit(run_single, entry, spec, args.steps,
                                   threads): spec for spec in pending}
            for fut in as_completed(futures):
                rec = fut.result()
                append_run(rec)
                runs.append(rec)
                n_done += 1
                tail = (f" eval={rec['final_eval_rel_l2']:.2e} "
                        f"(floor {rec['floor_eval_rel_l2']:.1e})"
                        if rec["kind"] == "phi" else
                        f" digits={rec['digits']:.2f}")
                print(f"  [{n_done}/{len(todo)} {time.time()-t0:6.0f}s] "
                      f"{rec['problem_id']:34s} {rec['opt_id']:14s}{tail}",
                      flush=True)
                if time.time() - last_plot > PLOT_EVERY_S:
                    plotting.make_all(runs, FIG_DIR, manifest)
                    last_plot = time.time()

    plotting.make_all(runs, FIG_DIR, manifest)
    print(f"\ndone in {time.time()-t0:.0f}s; figures in {FIG_DIR}")
    _summary(runs)


if __name__ == "__main__":
    main()
