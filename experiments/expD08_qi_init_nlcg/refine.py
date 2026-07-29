"""iteration_10 Phase 0 -- does outer refinement on an EXTENDED-PRECISION
residual buy the floor with O(1) memory?

The campaign's Phase-0 law (iteration_9) reads: attainable rel L2 ~ 5e-7 x
10^-(decades protected by stored own-process parameter vectors), i.e. the floor
costs a rank-sized basis B. Every arm behind that law shares one property: all
digits had to come out of ONE Krylov process whose residual was known only to
fp64 absolute accuracy. Two controls in the record are consistent with the law
being an artifact of that, not an information limit:

  * expD07 `cgls_refine` / `nlcg_restart` restarted on an fp64-RECOMPUTED
    residual (absolute error u|y|), and went backwards.
  * expD08 `b_replacement` compensated the CG recurrence but re-quenched the
    carried residual with a plain fp64 forward every 200 iterations
    (`rhat = A @ vv - y`, `re[:] = 0`) -- deleting the compensation 50x per run.

This script tests the untested cell: a memoryless inner solve, restarted on a
residual recomputed in DOUBLE-DOUBLE, with the iterate accumulated in
double-double. That is ~2 extra vectors of state, not rank.

Ladder (attributes the error channel):
  B            full memory reference (target)
  noB          memoryless, fp64 requench @200 (the 5e-7 baseline)
  noB_nq       memoryless, no requench          -> is the requench load-bearing?
  ir_fp64      outer refinement, fp64 residual  -> reproduces cgls_refine
  ir_dd        outer refinement, dd residual    -> THE CANDIDATE
  ir_dd_ddg    + dd gradient matvec             -> is A^T r cancellation the wall?
  ir_dd_dds    + dd alpha/beta scalars          -> is scalar rounding the wall?

Run:  python experiments/expD08_qi_init_nlcg/refine.py [--quick] [--arms a,b]
Out:  results/checkpoint_D_optimizers/expD08_qi_init_nlcg/iteration_10/phase0/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expD07_lstsq_optim_suite"))
import common  # noqa: E402

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / "iteration_10" / "phase0")

BUDGET = 20_000          # fp64 matvec-pairs, same unit as the expD07 suite
REQUENCH = 200           # the historical carried-residual refresh cadence
TRUST = 1e-24            # B-arm exhaustion escape (iteration_7 constant)
STREAK_MAX = 20

PROBLEMS = [
    "phi_sine_N256_r4", "phi_sine_8pi_N256_r4", "phi_runge_N256_r4",
    "phi_sine_mixture_N256_r4", "phi_exp_N256_r4", "phi_abs_cubed_N256_r4",
    "syn_illcond14_iso_N256_R1024", "syn_phispec_iso_N256_R1024",
]
QUICK = ["phi_sine_N256_r4", "phi_runge_N256_r4", "syn_phispec_iso_N256_R1024"]

ALL_ARMS = ["B", "noB", "noB_nq", "ir_fp64", "ir_dd", "ir_dd_ddg", "ir_dd_dds",
            "tik_fixed", "tik_cont", "tik_cont_dd"]


# ------------------------------------------------------------------ dd helpers
# Error-free transformations. No FMA in numpy, so two_prod uses Dekker splitting.

def two_sum(a, b):
    """s + e == a + b exactly, s == fl(a + b)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = 134217729.0 * a  # 2^27 + 1
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """p + e == a * b exactly."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_renorm(s, e):
    hi = s + e
    return hi, e - (hi - s)


def dd_add(x, xe, y, ye):
    """(x,xe) + (y,ye) as a dd pair."""
    s, e = two_sum(x, y)
    return dd_renorm(s, e + xe + ye)


def dd_axpy(x, xe, a, ae, d, de):
    """(x,xe) <- (x,xe) + (a,ae) * (d,de); a scalar (fp64 or dd pair)."""
    p, pe = two_prod(a, d)
    s, se = two_sum(x, p)
    err = xe + se + pe + a * de
    if ae != 0.0:
        err = err + ae * d
    return dd_renorm(s, err)


def dd_sum(p, pe=None):
    """Compensated pairwise reduction of p (+ exact companion pe) -> (hi, lo)."""
    err = 0.0 if pe is None else float(pe.sum())
    p = np.asarray(p, dtype=float)
    while p.size > 1:
        if p.size % 2:
            p = np.append(p, 0.0)
        s, e = two_sum(p[0::2], p[1::2])
        err += float(e.sum())
        p = s
    return dd_renorm(float(p[0]), err)


def dd_dot(a, b, ae=None, be=None):
    """Dot product to ~u^2 relative accuracy. Returns a dd pair."""
    p, pe = two_prod(a, b)
    hi, lo = dd_sum(p, pe)
    if ae is not None:
        hi, lo = dd_add(hi, lo, *dd_sum(ae * b))
    if be is not None:
        hi, lo = dd_add(hi, lo, *dd_sum(a * be))
    return hi, lo


def dd_gemv(A, x, xe=None):
    """A @ (x + xe) to ~u^2, per row. Returns (hi, lo) arrays of length A.shape[0]."""
    P, Pe = two_prod(A, x[None, :])
    err = Pe.sum(axis=1)
    if xe is not None:
        err = err + A @ xe
    while P.shape[1] > 1:
        if P.shape[1] % 2:
            P = np.concatenate([P, np.zeros((P.shape[0], 1))], axis=1)
        P, E = two_sum(P[:, 0::2], P[:, 1::2])
        err = err + E.sum(axis=1)
    hi = P[:, 0] + err
    return hi, err - (hi - P[:, 0])


DD_GEMV_COST = 8         # fp64-matvec-equivalents charged for one dd_gemv


def dd_selftest(verbose=True):
    """Validate the primitives against mpmath ground truth on adversarial input.

    The pitfall this guards (Sam's note, pitfall 1): a silently-broken dd path
    looks exactly like 'precision buys zero orders'.
    """
    from mpmath import mp, mpf
    mp.dps = 60
    rng = np.random.default_rng(0)
    ok = True

    # 1. dot product with catastrophic cancellation: exact answer ~1e-9 of terms
    a = rng.normal(size=2001)
    b = rng.normal(size=2001)
    b = b - a * (float(a @ b) / float(a @ a))       # force near-orthogonality
    exact = sum((mpf(float(x)) * mpf(float(y)) for x, y in zip(a, b)), mpf(0))
    plain = float(a @ b)
    hi, lo = dd_dot(a, b)
    # normalize by the TERM scale: after cancellation the exact answer is itself
    # O(u) of the terms, so relative-to-exact is not the meaningful yardstick.
    sc = mpf(float(np.abs(a * b).sum()))
    e_plain = abs(mpf(plain) - exact) / sc
    e_dd = abs(mpf(hi) + mpf(lo) - exact) / sc
    if verbose:
        print(f"  dd_dot: plain {float(e_plain):.2e}  dd {float(e_dd):.2e} "
              f"(of term scale)")
    ok &= float(e_dd) < 1e-30 and float(e_dd) < float(e_plain) * 1e-10

    # 2. gemv with cancellation: A x where the row sums nearly cancel
    A = rng.normal(size=(64, 301))
    x = rng.normal(size=301)
    A = A - np.outer(A @ x / (x @ x), x)            # force A x ~ 0
    hi, lo = dd_gemv(A, x)
    plain = A @ x
    ex = np.array([float(sum((mpf(float(A[i, j])) * mpf(float(x[j]))
                              for j in range(A.shape[1])), mpf(0)))
                   for i in range(A.shape[0])])
    sc = float(np.abs(A).max() * np.abs(x).max() * A.shape[1])
    e_plain = float(np.abs(plain - ex).max()) / sc
    e_dd = float(np.abs((hi + lo) - ex).max()) / sc
    if verbose:
        print(f"  dd_gemv: plain rel {e_plain:.2e}  dd rel {e_dd:.2e}")
    ok &= e_dd < 1e-30 and e_dd < e_plain * 1e-10

    # 3. dd accumulation retains increments far below the running total
    x, xe = 1.0, 0.0
    for _ in range(10_000):
        x, xe = dd_axpy(np.array([x]), np.array([xe]), 1e-22, 0.0,
                        np.array([1.0]), np.array([0.0]))
        x, xe = float(x[0]), float(xe[0])
    got = mpf(x) + mpf(xe)
    want = mpf(1) + mpf(10_000) * mpf("1e-22")
    ok &= abs(got - want) / abs(want) < 1e-30
    if verbose:
        print(f"  dd_axpy drift: {float(abs(got - want) / abs(want)):.2e}")
        print("  dd selftest:", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------- the arms

def _metric_fn(prob, entry):
    """Universal unit: relative L2. eval-grid for phis, train for synthetic."""
    if entry["kind"] == "phi":
        A_ev, y_ev = prob["A_ev"], prob["y_ev"]
        ny = float(np.linalg.norm(y_ev))
        return lambda v: float(np.linalg.norm(A_ev @ v - y_ev) / ny)
    A, y = prob["A"], prob["y"]
    ny = float(np.linalg.norm(y))
    return lambda v: float(np.linalg.norm(A @ v - y) / ny)


def _floor(entry, prob):
    if entry["kind"] == "phi":
        return float(entry["floor_eval_rel_l2"])
    return float(np.sqrt(max(entry["L_star"], 0.0) * prob["A"].shape[0])
                 / np.linalg.norm(prob["y"]))


def _inner_cg(A, r0, sc, iters, cost0, budget, ddg=False, dds=False,
              on_check=None, checks=()):
    """Memoryless CG (carried inner residual, exact GN step, PR+) on
    min_delta ||A delta + r0||^2 * sc.  Returns (delta, cost_used).

    ddg: compute the gradient by a dd matvec (attribution arm).
    dds: keep alpha/beta as dd scalars (attribution arm).
    """
    R, m = A.shape
    delta = np.zeros(m)
    s = r0.copy()                        # inner carried residual, s = A delta + r0
    se = np.zeros(R) if ddg else None
    cost = cost0

    def grad(s, se):
        if ddg:
            hi, lo = dd_gemv(A.T, s, se)
            return sc * hi, sc * lo
        return sc * (A.T @ s), None

    h, hl = grad(s, se)
    if ddg:
        cost += DD_GEMV_COST
    p = -h
    for it in range(1, iters + 1):
        if cost >= budget:
            break
        u = A @ p
        if dds:
            c = sc * dd_dot(u, u)[0]
            num = -dd_dot(h, p, ae=hl)[0]
        else:
            c = sc * float(u @ u)
            num = -float(h @ p)
        cost += 1
        if not math.isfinite(c) or c <= 0.0:
            break
        alpha = num / c
        if not math.isfinite(alpha):
            break
        delta = delta + alpha * p
        if ddg:
            s, se = dd_axpy(s, se, alpha, 0.0, u, np.zeros(R))
        else:
            s = s + alpha * u
        h_new, hl_new = grad(s, se)
        cost += DD_GEMV_COST if ddg else 1
        if not np.isfinite(h_new).all():
            break
        if dds:
            den = dd_dot(h, h, ae=hl, be=hl)[0]
            nb = (dd_dot(h_new, h_new, ae=hl_new, be=hl_new)[0]
                  - dd_dot(h_new, h, ae=hl_new, be=hl)[0])
        else:
            den = float(h @ h)
            nb = float(h_new @ h_new) - float(h_new @ h)
        beta = max(0.0, nb / den) if den > 0 else 0.0
        p = -h_new + beta * p
        h, hl = h_new, hl_new
        if on_check is not None and cost in checks:
            on_check(delta, cost)
    return delta, cost


def _sigma_max(A, iters=40):
    """Matrix-free power iteration on A^T A. O(1) memory, ~40 matvec pairs."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=A.shape[1])
    x /= np.linalg.norm(x)
    s = 0.0
    for _ in range(iters):
        z = A.T @ (A @ x)
        s = float(np.linalg.norm(z))
        if s == 0.0:
            break
        x = z / s
    return math.sqrt(s)


def _damped_stage(A, y, v, lam, sc, max_it, cost, budget, patience,
                  record=None, checks=(), dd=False, ve=None):
    """Memoryless CG on the DAMPED objective
        sc * ( ||A v - y||^2 + lam^2 ||v||^2 ).
    Carried residual, exact GN step (one JVP), PR+. O(1) state.
    Stage ends on a gradient-norm stall (never a loss comparison -- Lesson 1).
    """
    R, m = A.shape
    l2 = lam * lam
    if dd:
        fh, fl = dd_gemv(A, v, ve)
        r, rl = dd_add(fh, fl, -y, np.zeros(R))
        cost += DD_GEMV_COST
    else:
        r, rl = A @ v - y, None
        cost += 1
    g = sc * (A.T @ r + l2 * v)
    d = -g
    gmin, since = float(np.linalg.norm(g)), 0
    for _ in range(max_it):
        if cost >= budget:
            break
        u = A @ d
        c = sc * (float(u @ u) + l2 * float(d @ d))
        cost += 1
        if not math.isfinite(c) or c <= 0.0:
            break
        alpha = -float(g @ d) / c
        if not math.isfinite(alpha):
            break
        if dd:
            v, ve = dd_axpy(v, ve, alpha, 0.0, d, np.zeros(m))
            r, rl = dd_axpy(r, rl, alpha, 0.0, u, np.zeros(R))
        else:
            v = v + alpha * d
            r = r + alpha * u
        g_new = sc * (A.T @ (r if not dd else r + rl)
                      + l2 * (v if not dd else v + ve))
        cost += 1
        if not np.isfinite(g_new).all():
            break
        den = float(g @ g)
        beta = max(0.0, (float(g_new @ g_new) - float(g_new @ g)) / den) \
            if den > 0 else 0.0
        d = -g_new + beta * d
        g = g_new
        gn = float(np.linalg.norm(g))
        if gn < gmin:
            gmin, since = gn, 0
        else:
            since += 1
            if since >= patience:
                break
        if record is not None and cost in checks:
            record(v if not dd else v + ve, cost)
    return v, ve, cost


def run_arm(job):
    entry, arm = job["entry"], job["arm"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    sc = 2.0 / R
    metric = _metric_fn(prob, entry)
    floor = _floor(entry, prob)
    inner_iters = job.get("inner", m)

    checks = sorted(set(np.unique(np.logspace(0, math.log10(BUDGET), 70)
                                  .astype(int)).tolist()) | {BUDGET})
    checkset = set(checks)
    rels, costs = [], []
    best = np.inf
    t0 = time.time()

    def record(v, cost):
        nonlocal best
        rel = metric(v)
        if math.isfinite(rel):
            rels.append(rel)
            costs.append(int(cost))
            best = min(best, rel)

    if arm.startswith("tik_"):
        # ---- Tikhonov damping: cap kappa_eff at sigma1/lambda, then anneal ---
        dd = arm.endswith("_dd")
        s1 = _sigma_max(A)
        cost = 40
        lam_rel0 = job.get("lam_rel0", 1e-2)
        rho = job.get("rho", 10 ** -0.5)
        patience = job.get("patience", 30)
        max_stage = job.get("max_stage", 400)
        v, ve = np.zeros(m), np.zeros(m)
        lam = s1 * lam_rel0
        stages = []
        while cost < BUDGET:
            v, ve, cost = _damped_stage(
                A, y, v, lam, sc, max_stage, cost, BUDGET, patience,
                record=record, checks=checkset, dd=dd, ve=ve)
            record(v + ve if dd else v, cost)
            stages.append({"lam_rel": lam / s1, "cost": int(cost),
                           "metric": rels[-1] if rels else None})
            if arm == "tik_fixed":
                if cost >= BUDGET:
                    break
                continue                     # keep grinding at the same lambda
            lam *= rho
            if lam / s1 < 1e-17:
                break
        extra = {"sigma1": s1, "stages": stages, "lam_rel0": lam_rel0,
                 "rho": rho}
    elif arm.startswith("ir_"):
        # ---- outer iterative refinement -------------------------------------
        use_dd = "dd" in arm
        ddg, dds = arm == "ir_dd_ddg", arm == "ir_dd_dds"
        v, ve = np.zeros(m), np.zeros(m)
        cost, n_outer = 0, 0
        while cost < BUDGET:
            # STEP 1: residual at the current iterate. The one expensive place.
            if n_outer == 0:
                r, rl = -y.copy(), np.zeros(R)      # exact at v = 0
            elif use_dd:
                fh, fl = dd_gemv(A, v, ve)
                r, rl = dd_add(fh, fl, -y, np.zeros(R))
                cost += DD_GEMV_COST
            else:
                r, rl = (A @ (v + ve)) - y, np.zeros(R)
                cost += 1
            # STEP 2: correction, working precision, only ~6 digits needed
            delta, cost = _inner_cg(
                A, r, sc, inner_iters, cost, BUDGET, ddg=ddg, dds=dds,
                on_check=lambda d, c: record(v + ve + d, c), checks=checkset)
            # STEP 3: update, extended precision
            v, ve = dd_axpy(v, ve, 1.0, 0.0, delta, np.zeros(m))
            n_outer += 1
            record(v + ve, cost)
            if n_outer > 400:
                break
        extra = {"n_outer": n_outer, "inner": inner_iters}
    else:
        # ---- single memoryless / full-memory process (the historical arms) ---
        use_B = arm == "B"
        requench = arm in ("B", "noB")
        v = np.zeros(m)
        rhat = -y.copy()
        B = np.zeros((m, m)) if use_B else None
        k, streak = 0, 0

        def project(x):
            if k == 0:
                return x.copy()
            for _ in range(2):
                x = x - B[:k].T @ (B[:k] @ x)
            return x

        g = sc * (A.T @ rhat)
        gt = project(g) if use_B else g
        d = -gt
        cost = 1
        while cost < BUDGET:
            if requench and cost % REQUENCH < 2:
                rhat = A @ v - y
                cost += 1
            u = A @ d
            c = sc * float(u @ u)
            cost += 1
            if not math.isfinite(c) or c <= 0.0:
                d = -(project(g) if use_B else g)
                continue
            alpha = -float(g @ d) / c
            if not math.isfinite(alpha):
                break
            v = v + alpha * d
            rhat = rhat + alpha * u
            g_new = sc * (A.T @ rhat)
            cost += 1
            if not np.isfinite(g_new).all():
                break
            if use_B:
                if k < m:
                    b_new = project(gt)
                    nb = float(np.linalg.norm(b_new))
                    if nb > 0.0:
                        B[k] = b_new / nb
                        k += 1
                gt_new = project(g_new)
                if float(gt_new @ gt_new) < TRUST * float(g_new @ g_new):
                    streak += 1
                    if streak >= STREAK_MAX:
                        k, streak = 0, 0
                    gt_new, beta = g_new.copy(), 0.0
                else:
                    streak = 0
                    den = float(gt @ gt)
                    beta = max(0.0, (float(gt_new @ gt_new)
                                     - float(gt_new @ gt)) / den)
            else:
                gt_new = g_new
                den = float(gt @ gt)
                beta = max(0.0, (float(gt_new @ gt_new)
                                 - float(gt_new @ gt)) / den) if den > 0 else 0.0
            d = -gt_new + beta * d
            g, gt = g_new, gt_new
            if cost in checkset:
                record(v, cost)
        extra = {"B_k": int(k)}

    return {"id": entry["id"], "arm": arm, "best_rel": best, "floor": floor,
            "rels": rels, "costs": costs, "kind": entry["kind"],
            "wall_s": round(time.time() - t0, 1), **extra}


# ---------------------------------------------------------------------- driver

def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [p for p in PROBLEMS if any(r["id"] == p for r in rows)]
    arms = [a for a in ALL_ARMS if any(r["arm"] == a for r in rows)]
    cmap = {"B": "#111111", "noB": "#8c8c8c", "noB_nq": "#c0a000",
            "ir_fp64": "#1f77b4", "ir_dd": "#d62728",
            "ir_dd_ddg": "#9467bd", "ir_dd_dds": "#2ca02c",
            "tik_fixed": "#17becf", "tik_cont": "#e377c2",
            "tik_cont_dd": "#7f4f24"}
    ncol = 4
    nrow = int(math.ceil(len(ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.5 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, pid in enumerate(ids):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        fl = None
        for arm in arms:
            r = next((x for x in rows if x["id"] == pid and x["arm"] == arm), None)
            if r is None or not r["rels"]:
                continue
            fl = r["floor"]
            run = np.minimum.accumulate(np.array(r["rels"]))
            ax.plot(r["costs"], run, color=cmap.get(arm), lw=1.6,
                    label=arm, zorder=3 if arm.startswith("ir_dd") else 2)
        if fl:
            ax.axhline(fl, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 2.0)
        ax.set_xlim(1, BUDGET)
        ax.set_title(pid.replace("phi_", "").replace("syn_", ""), fontsize=9)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        if i % ncol == 0:
            ax.set_ylabel("rel $L_2$ (running min)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("fp64 matvec-pairs")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color="k", ls=":")],
               labels + ["lstsq floor"], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(labels) + 1, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if nrow > 1 else 0.86))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--inner", type=int, default=0, help="0 = #cols")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam0", type=float, default=1e-2)
    ap.add_argument("--rho", type=float, default=10 ** -0.5)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--maxstage", type=int, default=400)
    args = ap.parse_args()

    print("dd primitive validation (mpmath ground truth):")
    if not dd_selftest():
        raise SystemExit("dd primitives FAILED validation -- fix before measuring")

    man = {e["id"]: e for e in common.load_manifest()}
    pids = QUICK if args.quick else PROBLEMS
    arms = [a for a in args.arms.split(",") if a]
    jobs = [{"entry": man[p], "arm": a, "lam_rel0": args.lam0, "rho": args.rho,
             "patience": args.patience, "max_stage": args.maxstage,
             **({"inner": args.inner} if args.inner else {})}
            for p in pids for a in arms]
    print(f"{len(jobs)} runs ({len(pids)} problems x {len(arms)} arms)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_arm, jobs):
            rows.append(r)
            print(f"  {r['id']:<30s} {r['arm']:<12s} best {r['best_rel']:.3e} "
                  f"floor {r['floor']:.1e}  "
                  f"x{r['best_rel'] / r['floor']:.1e}  {r['wall_s']}s")
    tag = args.tag or ("quick" if args.quick else "full")
    (OUT_DIR / f"rows_{tag}.json").write_text(json.dumps(rows, indent=1))
    figure(rows, OUT_DIR / f"trajectories_{tag}.png")

    print(f"\ngeo-mean best rel L2 by arm ({time.time() - t0:.0f}s total):")
    for a in arms:
        sel = [r for r in rows if r["arm"] == a and math.isfinite(r["best_rel"])]
        if not sel:
            continue
        gm = math.exp(sum(math.log(max(r["best_rel"], 1e-17)) for r in sel) / len(sel))
        gr = math.exp(sum(math.log(max(r["best_rel"], 1e-17) / r["floor"])
                          for r in sel) / len(sel))
        print(f"  {a:<12s} {gm:.3e}   ({gr:.1e} x floor)")
    print(f"\nwrote {OUT_DIR}/rows_{tag}.json, trajectories_{tag}.png")


if __name__ == "__main__":
    main()
