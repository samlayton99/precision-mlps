"""Phase 0 of iteration_9: can compensated (double-double) recurrence replace
the memory basis B at O(m) state?

Testbed: frozen expD07 least-squares systems (deterministic quadratics with
known SVD floors). All arms share the slim core: carried residual, exact
GN steps from one matvec pair, PR+ conjugate direction, refresh timer 200,
NaN reject, no loss comparisons. Arms differ ONLY in the anti-leak mechanism:

  B          full orthogonal basis, double projection, exhaustion escape
             (iteration_7's engine; the reference — should reach the floor)
  noB        nothing (prices B for real; never measured before — the
             historical ladder added memory BEFORE the carried residual)
  noB_dd1    compensated dot products only (alpha, beta, c) — no extra state
  noB_dd2    dd1 + compensated state vectors v, r_hat (persistent error
             companions, folded into the fresh residual at each refresh;
             deployable cost identical to slim)
  noB_dd3    dd2 + full error feed-through each iteration: g sees A^T re,
             u sees A de, direction d carries its own companion
             (deployable cost ~2 extra passes/step)

Gate (binding, from fable_thoughts.md): a compensated arm reaches <=1e-13
(within ~1 order of the B arm) on the phi problems, else STOP and report.

Budget 10k iterations (= 20k grad-eval equivalents, matching the expD07
suite). Grading best-over-trajectory at ~60 log-spaced checkpoints.

Run: uv run python experiments/expD08_qi_init_nlcg/b_replacement.py
Out: results/.../expD08_qi_init_nlcg/iteration_9/b_replacement/
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "expD07_lstsq_optim_suite"))
import common  # noqa: E402  (expD07 problem I/O)

OUT_DIR = (REPO / "results" / "checkpoint_D_optimizers" / "expD08_qi_init_nlcg"
           / "iteration_9" / "b_replacement")

PROBLEM_IDS = [
    "phi_sine_N256_r4", "phi_sine_8pi_N256_r4", "phi_runge_N256_r4",
    "phi_sine_mixture_N256_r4", "phi_exp_N256_r4", "phi_abs_cubed_N256_r4",
    "syn_illcond14_iso_N256_R1024", "syn_phispec_iso_N256_R1024",
]
ARMS = ["B", "noB", "noB_dd1", "noB_dd2", "noB_dd3"]
# Round 2 (after the dd gate failed): head-k deflation — keep the FIRST k
# basis vectors (they span the solved head, the leak's source), frozen, and
# project them out forever. Lesson 4's "partial memory worse than none" was
# measured on a LAST-k window; first-k has never been tested. Plus the
# full-dd diagnostic (dd matvecs too): does exact-arithmetic CG floor?
HEADK_ARMS = ["headk64", "headk128", "headk256", "headk384"]
DDMV_IDS = ["phi_sine_N256_r4", "syn_phispec_iso_N256_R1024"]
# Round 3: frozen-head deflation in FUNCTION space — basis of normalized
# step images u = A d (n-dim, so memory k x n is INDEPENDENT of parameter
# count m). Deflation applied to the gradient seed: g_defl = A^T P(rhat),
# with alpha still from the full residual (true line minimum). This is the
# working-notes candidate #2 (function-space memory), resample-pinned per
# Lesson 7.
FHEAD_ARMS = ["fhead128", "fhead256", "fhead384"]
# Round 5: hybrid — small param-space head (kills the sigma_1-amplified
# v-drift that walls fhead) + function-space basis for the remaining depth
# (m-independent memory). Predicted: wall drops by the protected decades.
HYB_ARMS = ["hyb32f512", "hyb64f512"]
ITERS = 10_000
REFRESH = 200
TRUST = 1e-24     # exhaustion threshold on projected-gradient energy (B arm)
STREAK_MAX = 20   # consecutive exhaustions before B empties (B arm)

# ---------------------------------------------------------------- dd helpers


def two_sum(a, b):
    """Exact sum: returns (s, e) with s + e == a + b exactly, s = fl(a+b)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = 134217729.0 * a  # 2^27 + 1 (Dekker)
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """Exact product: (p, e) with p + e == a * b exactly (no FMA needed)."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_renorm(s, e):
    hi = s + e
    return hi, e - (hi - s)


def dd_axpy(x, xe, alpha, d, de=None):
    """(x, xe) <- (x + xe) + alpha * (d + de), compensated."""
    p, pe = two_prod(alpha, d)
    s, se = two_sum(x, p)
    err = xe + (se + pe)
    if de is not None:
        err = err + alpha * de
    return dd_renorm(s, err)


def tree_sum(p):
    """Compensated pairwise reduction: (hi, lo) with hi + lo ~= exact sum."""
    err = 0.0
    while p.size > 1:
        if p.size % 2:
            p = np.append(p, 0.0)
        s, e = two_sum(p[0::2], p[1::2])
        err += float(e.sum())
        p = s
    return float(p[0]), err


def dd_dot(a, b):
    p, pe = two_prod(a, b)
    hi, lo = tree_sum(p)
    return hi + (lo + float(pe.sum()))


def selftest():
    rng = np.random.default_rng(0)
    # adversarial cancellation: dot that plain fp64 gets wrong
    a = rng.normal(size=4001) * 1e8
    b = rng.normal(size=4001)
    import math
    exact = math.fsum([float(x) * float(y) for x, y in zip(a, b)])
    plain = float(a @ b)
    comp = dd_dot(a, b)
    assert abs(comp - exact) <= 1e-8 * abs(exact) + 1e-12, (comp, exact)
    assert abs(comp - exact) < abs(plain - exact) or plain == exact
    # dd_axpy keeps low bits a plain axpy drops
    x = np.ones(8)
    xe = np.zeros(8)
    for _ in range(1000):
        x, xe = dd_axpy(x, xe, 1e-20, np.ones(8))
    assert abs((x[0] + xe[0]) - (1 + 1000 * 1e-20)) < 1e-32
    print("dd selftest passed")


# ---------------------------------------------------------------- the engine


def run_arm(job):
    os.environ.setdefault("OMP_NUM_THREADS", str(job.get("threads", 1)))
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    is_phi = entry["kind"] == "phi"
    arm = job["arm"]
    headk = arm.startswith("headk")
    use_B = arm == "B" or headk
    k_max = min(int(arm[5:]), m) if headk else m
    ddots = arm in ("noB_dd1", "noB_dd2", "noB_dd3")
    dstate = arm in ("noB_dd2", "noB_dd3")
    dfull = arm == "noB_dd3"

    dot = dd_dot if ddots else lambda a, b: float(a @ b)

    v = np.zeros(m)
    ve = np.zeros(m) if dstate else None
    rhat = -y.copy()               # residual at v = 0
    re = np.zeros(R) if dstate else None
    de = np.zeros(m) if dfull else None

    def grad(rh, rerr):
        g = (2.0 / R) * (A.T @ rh)
        if dfull and rerr is not None:
            g = g + (2.0 / R) * (A.T @ rerr)
        return g

    B = np.zeros((min(k_max, m), m)) if use_B else None
    k = 0
    streak = 0

    def project(x):
        if k == 0:
            return x.copy()
        for _ in range(2):  # double projection
            x = x - B[:k].T @ (B[:k] @ x)
        return x

    g = grad(rhat, re)
    gt = project(g) if use_B else g
    d = -gt
    if dfull:
        de = np.zeros(m)

    checks = sorted(set(np.unique(np.logspace(0, np.log10(ITERS), 60)
                                  .astype(int)).tolist()) | {ITERS})
    rels, its = [], []
    best = np.inf

    def metric():
        vv = v + ve if dstate else v
        if is_phi:
            r_ev = prob["A_ev"] @ vv - prob["y_ev"]
            return float(np.linalg.norm(r_ev) / np.linalg.norm(prob["y_ev"]))
        r_tr = A @ vv - y
        return float(np.linalg.norm(r_tr) / np.linalg.norm(y))

    t0 = time.time()
    for it in range(1, ITERS + 1):
        if it % REFRESH == 0:  # requench carried residual (honest forward)
            vv = v + ve if dstate else v
            rhat = A @ vv - y
            if dstate:
                re[:] = 0.0
        u = A @ d
        if dfull:
            u = u + A @ de
        c2 = dot(u, u)
        c = (2.0 / R) * c2
        if not np.isfinite(c) or c <= 0.0:
            d = -(project(g) if use_B else g)
            if dfull:
                de[:] = 0.0
            continue
        alpha = -dot(g, d) / c
        if not np.isfinite(alpha):
            break
        if dstate:
            v, ve = dd_axpy(v, ve, alpha, d, de)
            rhat, re = dd_axpy(rhat, re, alpha, u)
        else:
            v = v + alpha * d
            rhat = rhat + alpha * u
        g_new = grad(rhat, re)
        if not np.isfinite(g_new).all():
            break
        if use_B:
            if k < k_max:  # append the just-used direction's gradient rep
                b_new = project(gt)
                nb = np.linalg.norm(b_new)
                if nb > 0.0:
                    B[k] = b_new / nb
                    k += 1
            gt_new = project(g_new)
            if dot(gt_new, gt_new) < TRUST * dot(g_new, g_new):
                streak += 1
                if streak >= STREAK_MAX and not headk:
                    k = 0       # full-B arm empties; frozen head never does
                    streak = 0
                gt_new = g_new.copy()
                beta = 0.0
            else:
                streak = 0
                denom = dot(gt, gt)
                beta = (dot(gt_new, gt_new) - dot(gt_new, gt)) / denom
                beta = max(0.0, beta)
        else:
            gt_new = g_new
            denom = dot(gt, gt)
            beta = max(0.0, (dot(gt_new, gt_new) - dot(gt_new, gt)) / denom)
        if dfull:
            p, pe = two_prod(beta, d)
            s, se = two_sum(-gt_new, p)
            d, de = dd_renorm(s, se + pe + beta * de)
        else:
            d = -gt_new + beta * d
        g, gt = g_new, gt_new
        if it in checks or it == checks[-1]:
            rel = metric()
            rels.append(rel)
            its.append(it)
            best = min(best, rel)

    if is_phi:
        floor = entry["floor_eval_rel_l2"]
    else:
        floor = float(np.sqrt(max(entry["L_star"], 0.0) * R)
                      / np.linalg.norm(y))
    return {"id": entry["id"], "arm": arm, "best_rel": best, "floor": floor,
            "rels": rels, "its": its, "B_k": int(k) if use_B else 0,
            "wall_s": round(time.time() - t0, 1)}


def run_fspace(job):
    """Frozen-head deflation in function space. State: U (k_max x n) of
    orthonormalized step images, carried residual, direction. The deflated
    gradient g_defl = (2/R) A^T Pi_U^2(rhat) drives the direction; the step
    length uses the FULL residual (exact line minimum, as in every arm).

    hybrid variant (arm 'hyb<kp>f<kf>'): adds a SMALL frozen param-space head
    (k_p vectors) on top. Theory: fhead's wall is sigma_1-amplified v-drift
    in the top directions (steps re-enter them at the projection's own eps
    residue and the function-space gauge hides it); protecting the top k_p
    decades in PARAM space kills the loud drift, and the remaining drift is
    amplified only by sigma_{k_p+1} — predicted wall drop ~ the protected
    decades. Memory: k_p x m + k_f x n (the k_f x n half is m-independent)."""
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    is_phi = entry["kind"] == "phi"
    sc = 2.0 / R
    if job["arm"].startswith("hyb"):
        kp_s, kf_s = job["arm"][3:].split("f")
        k_p, k_max = min(int(kp_s), m), min(int(kf_s), R)
    else:
        k_p, k_max = 0, min(int(job["arm"][5:]), R)

    Bp = np.zeros((k_p, m)) if k_p else None
    kb = 0

    def project_p(x):
        if kb == 0:
            return x.copy()
        for _ in range(2):
            x = x - Bp[:kb].T @ (Bp[:kb] @ x)
        return x

    U = np.zeros((k_max, R))
    k = 0

    def project_r(x):
        if k == 0:
            return x.copy()
        for _ in range(2):
            x = x - U[:k].T @ (U[:k] @ x)
        return x

    v = np.zeros(m)
    rhat = -y.copy()
    g_raw = sc * (A.T @ rhat)
    gt = project_p(sc * (A.T @ project_r(rhat)))
    d = -gt
    checks = sorted(set(np.unique(np.logspace(0, np.log10(ITERS), 60)
                                  .astype(int)).tolist()) | {ITERS})
    rels, its = [], []
    best = np.inf
    t0 = time.time()
    for it in range(1, ITERS + 1):
        if it % REFRESH == 0:
            rhat = A @ v - y
        u = A @ d
        c2 = float(u @ u)
        c = sc * c2
        if not np.isfinite(c) or c <= 0.0:
            d = -gt
            continue
        alpha = -sc * float(rhat @ u) / c   # full-residual line minimum
        if not np.isfinite(alpha):
            break
        v = v + alpha * d
        rhat = rhat + alpha * u
        if k < k_max:   # append the searched image direction, frozen head
            b_new = project_r(u)
            nb = np.linalg.norm(b_new)
            if nb > 0.0:
                U[k] = b_new / nb
                k += 1
        if kb < k_p:    # hybrid: frozen param-space head from used gradients
            bp_new = project_p(gt)
            nbp = np.linalg.norm(bp_new)
            if nbp > 0.0:
                Bp[kb] = bp_new / nbp
                kb += 1
        g_raw = sc * (A.T @ rhat)
        gt_new = project_p(sc * (A.T @ project_r(rhat)))
        if not np.isfinite(gt_new).all():
            break
        if float(gt_new @ gt_new) < TRUST * float(g_raw @ g_raw):
            gt_new = g_raw.copy()   # exhaustion: fall back to raw gradient
            beta = 0.0
        else:
            beta = max(0.0, float(gt_new @ (gt_new - gt))
                       / float(gt @ gt))
        d = -gt_new + beta * d
        gt = gt_new
        if it in checks:
            if is_phi:
                rel = float(np.linalg.norm(prob["A_ev"] @ v - prob["y_ev"])
                            / np.linalg.norm(prob["y_ev"]))
            else:
                rel = float(np.linalg.norm(A @ v - y) / np.linalg.norm(y))
            rels.append(rel)
            its.append(it)
            best = min(best, rel)
    if is_phi:
        floor = entry["floor_eval_rel_l2"]
    else:
        floor = float(np.sqrt(max(entry["L_star"], 0.0) * R)
                      / np.linalg.norm(y))
    return {"id": entry["id"], "arm": job["arm"], "best_rel": best,
            "floor": floor, "rels": rels, "its": its, "B_k": int(k),
            "wall_s": round(time.time() - t0, 1)}


def run_replay(job):
    """Round 4: the replay hypothesis. fp64 is bitwise deterministic, so the
    direction history is a deterministic function of (operator, seed) and can
    be REGENERATED instead of STORED: CG's directions are Lanczos vectors of
    M = A^T A, produced by a 3-term recurrence carrying 2 live vectors +
    O(T) scalars. If projecting against the replayed stream deflates as well
    as projecting against stored B, the rank cost moves from memory to flops.

    Offline v0 (this function) holds the stream in RAM purely to MEASURE
    projection quality — the science gates, not the deployable form:
      mode 'qr'    project against the sequentially-orthonormalized stream
                   prefix (what a stored-B replay would give). Gate 1: is the
                   raw stream's span sufficient despite fp64 ghost-lag?
      mode 'gram'  project via Gram pseudo-inverse of the RAW stream prefix
                   (the streamable form — no orthonormal basis ever stored).
                   Gate 2: does projection quality survive G's conditioning?
    s_rate: stream indices consumed per optimizer step (ghosts make the
    stream lag; >1 lets it keep up)."""
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    is_phi = entry["kind"] == "phi"
    sc = 2.0 / R
    mode, s_rate = job["mode"], job["s_rate"]
    T_h = min(job.get("T_h", 2500), ITERS)

    # --- the deterministic Lanczos stream on M = A^T A (raw, no reortho) ---
    h_prev = np.zeros(m)
    h = A.T @ y
    h /= np.linalg.norm(h)
    H = np.zeros((T_h, m))
    beta_prev = 0.0
    for i in range(T_h):
        H[i] = h
        w = A.T @ (A @ h)
        a = float(h @ w)
        w = w - a * h - beta_prev * h_prev
        b = float(np.linalg.norm(w))
        if b == 0.0:
            H = H[: i + 1]
            break
        h_prev, h = h, w / b
        beta_prev = b
    T_h = len(H)

    # --- prefix projectors ---
    if mode == "qr":
        # sequential double-orthonormalization of the stream; keep index map
        Q = np.zeros((T_h, m))
        nq = 0
        accepted_upto = np.zeros(T_h, dtype=int)  # stream idx -> #accepted
        for i in range(T_h):
            w = H[i].copy()
            for _ in range(2):
                if nq:
                    w = w - Q[:nq].T @ (Q[:nq] @ w)
            nw = np.linalg.norm(w)
            if nw > 1e-8 * np.linalg.norm(H[i]):  # skip ghost-redundant
                Q[nq] = w / nw
                nq += 1
            accepted_upto[i] = nq

        def projector(s):
            kq = accepted_upto[min(s, T_h) - 1] if s > 0 else 0

            def proj(z):
                for _ in range(2):
                    z = z - Q[:kq].T @ (Q[:kq] @ z)
                return z
            return proj
    else:  # gram
        G = H @ H.T
        cache = {}

        def projector(s):
            s = min(s, T_h)
            if s == 0:
                return lambda z: z
            if s not in cache:
                lam, V = np.linalg.eigh(G[:s, :s])
                keep = lam > max(1e-13 * lam.max(), 0.0)
                cache.clear()   # only current prefix needed
                cache[s] = (V[:, keep] / np.sqrt(lam[keep]), s)

            W, _ = cache[s]

            def proj(z):
                for _ in range(2):
                    z = z - H[:s].T @ (W @ (W.T @ (H[:s] @ z)))
                return z
            return proj

    # --- slim-noB with the replay projector ---
    v = np.zeros(m)
    rhat = -y.copy()
    g = sc * (A.T @ rhat)
    proj = projector(0)
    gt = g.copy()
    d = -gt
    checks = sorted(set(np.unique(np.logspace(0, np.log10(ITERS), 60)
                                  .astype(int)).tolist()) | {ITERS})
    rels, its = [], []
    best = np.inf
    t0 = time.time()
    P_STRIDE = 100 if mode == "gram" else 25  # gram: eigh cost per refresh
    for it in range(1, ITERS + 1):
        if it % REFRESH == 0:
            rhat = A @ v - y
        if it % P_STRIDE == 1 or it == 1:
            proj = projector(int(s_rate * it))
        u = A @ d
        c = sc * float(u @ u)
        if not np.isfinite(c) or c <= 0.0:
            d = -gt
            continue
        alpha = -float(g @ d) / c
        if not np.isfinite(alpha):
            break
        v = v + alpha * d
        rhat = rhat + alpha * u
        g_new = sc * (A.T @ rhat)
        if not np.isfinite(g_new).all():
            break
        gt_new = proj(g_new)
        if float(gt_new @ gt_new) < TRUST * float(g_new @ g_new):
            gt_new = g_new.copy()   # stream exhausted/over-deflating
            beta = 0.0
        else:
            beta = max(0.0, float(gt_new @ (gt_new - gt)) / float(gt @ gt))
        d = -gt_new + beta * d
        g, gt = g_new, gt_new
        if it in checks:
            if is_phi:
                rel = float(np.linalg.norm(prob["A_ev"] @ v - prob["y_ev"])
                            / np.linalg.norm(prob["y_ev"]))
            else:
                rel = float(np.linalg.norm(A @ v - y) / np.linalg.norm(y))
            rels.append(rel)
            its.append(it)
            best = min(best, rel)
    if is_phi:
        floor = entry["floor_eval_rel_l2"]
    else:
        floor = float(np.sqrt(max(entry["L_star"], 0.0) * R)
                      / np.linalg.norm(y))
    n_acc = int(accepted_upto[-1]) if mode == "qr" else T_h
    return {"id": entry["id"], "arm": f"replay_{mode}_x{s_rate:g}",
            "best_rel": best, "floor": floor, "rels": rels, "its": its,
            "B_k": n_acc, "wall_s": round(time.time() - t0, 1)}


def slim_noB_solve(A, y, iters=ITERS):
    """The memoryless slim core (carried residual, exact GN steps, PR+,
    refresh 200, NaN reject) on a frozen system. Returns the best-by-train
    iterate. This is the per-stage engine of the cascade."""
    R, m = A.shape
    sc = 2.0 / R
    v = np.zeros(m)
    rhat = -y.copy()
    g = sc * (A.T @ rhat)
    d = -g
    best_v, best_tr = v.copy(), np.inf
    for it in range(1, iters + 1):
        if it % REFRESH == 0:
            rhat = A @ v - y
            tr = float(np.linalg.norm(rhat) / np.linalg.norm(y))
            if tr < best_tr:
                best_tr, best_v = tr, v.copy()
        u = A @ d
        c = sc * float(u @ u)
        if not np.isfinite(c) or c <= 0.0:
            d = -g
            continue
        alpha = -float(g @ d) / c
        if not np.isfinite(alpha):
            break
        v = v + alpha * d
        rhat = rhat + alpha * u
        g_new = sc * (A.T @ rhat)
        if not np.isfinite(g_new).all():
            break
        beta = max(0.0, float(g_new @ (g_new - g)) / float(g @ g))
        d = -g_new + beta * d
        g = g_new
    tr = float(np.linalg.norm(A @ v - y) / np.linalg.norm(y))
    if tr < best_tr:
        best_tr, best_v = tr, v.copy()
    return best_v, best_tr


def make_stage_basis(N, x_tr, x_ev, shift_frac=0.0, gamma_mult=1.0):
    """Repo-standard phi basis at width N: uniform centers + default_halo,
    gamma = gamma_mult * lambda*/h, augmented [Phi, 1]. shift_frac shifts
    centers by that fraction of a cell. gamma_mult > 1 sharpens the bumps —
    the booster-stage knob: it aligns the basis's easy band with the
    residue's frequency band AND flattens the spectrum (expA04)."""
    from src.construction.qi_mpmath import default_halo
    h = 2.0 / N
    halo = default_halo(N, lambda_star=common.LAMBDA_STAR)
    centers = (-1.0 + np.arange(-halo, N + halo + 1, dtype=np.float64) * h
               + shift_frac * h)
    gamma = gamma_mult * common.LAMBDA_STAR / h
    return (common.build_phi_aug(x_tr, gamma, centers),
            common.build_phi_aug(x_ev, gamma, centers))


def run_cascade(job):
    """Round 6: the cascade gate. Each stage hands its residue, rescaled to
    O(1), to a FRESH basis (the leftover is tail-concentrated for the OLD
    operator — that is why plain refinement dies — but generic for a new
    one, provided the new basis has resolution margin). Composite:
    f = f1 + s2 f2 + s2 s3 f3. Free depth ~7 decades per stage compounds.
    Literature anchor: Wang & Lai 2024 (multi-stage NNs, machine precision);
    ours differs by using the slim core per stage (7 decades/stage, not ~4)."""
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    x_tr = np.linspace(-1.0, 1.0, A.shape[0])
    x_ev = np.linspace(-1.0, 1.0, prob["A_ev"].shape[0])
    N1 = entry["N"]
    if job["arm"] == "casc_same":
        stages = [(N1, 0.5), (N1, 1.0 / 3.0)]
    else:  # casc_x2
        stages = [(2 * N1, 0.0), (4 * N1, 0.5)]

    t0 = time.time()
    v1, tr1 = slim_noB_solve(A, y, job.get("iters", ITERS))
    resid_tr = y - A @ v1                      # honest leftover, fresh
    resid_ev = prob["y_ev"] - prob["A_ev"] @ v1
    y_norm_ev = np.linalg.norm(prob["y_ev"])
    rels = [float(np.linalg.norm(resid_ev) / y_norm_ev)]
    stage_tr = [tr1]
    scale = 1.0
    for (N_s, shift) in stages:
        s = float(np.sqrt(np.mean(resid_tr ** 2)))
        if s == 0.0:
            break
        y_s = resid_tr / s
        A_s, A_s_ev = make_stage_basis(N_s, x_tr, x_ev)
        if shift:
            A_s, A_s_ev = make_stage_basis(N_s, x_tr, x_ev, shift_frac=shift)
        v_s, tr_s = slim_noB_solve(A_s, y_s, job.get("iters", ITERS))
        stage_tr.append(tr_s)
        resid_tr = (y_s - A_s @ v_s) * s       # leftover in original units
        resid_ev = resid_ev - s * (A_s_ev @ v_s)
        scale *= s
        rels.append(float(np.linalg.norm(resid_ev) / y_norm_ev))
    return {"id": entry["id"], "arm": job["arm"], "best_rel": min(rels),
            "floor": entry["floor_eval_rel_l2"], "rels": rels,
            "its": list(range(len(rels))), "B_k": 0,
            "stage_train_rels": stage_tr,
            "wall_s": round(time.time() - t0, 1)}


def dd_gemv(A, x, xe=None):
    """(hi, lo) per row of A @ (x + xe), products and reduction compensated."""
    P, Pe = two_prod(A, x[None, :])
    err = Pe.sum(axis=1)
    if xe is not None:
        err = err + A @ xe
    while P.shape[1] > 1:
        if P.shape[1] % 2:
            P = np.concatenate([P, np.zeros((P.shape[0], 1))], axis=1)
        P, E = two_sum(P[:, 0::2], P[:, 1::2])
        err = err + E.sum(axis=1)
    return P[:, 0], err


def run_ddmv(job):
    """Diagnostic: EVERYTHING double-double (matvecs, state, dots) — the
    exact-arithmetic-CG check. Not deployable (a network JVP/VJP cannot run
    at dd precision); exists purely to attribute the leak."""
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    is_phi = entry["kind"] == "phi"
    sc = 2.0 / R

    def ddot(a, ae, b, be):
        return dd_dot(a, b) + float(a @ be) + float(ae @ b)

    v, ve = np.zeros(m), np.zeros(m)
    rhat, re = -y.copy(), np.zeros(R)
    gh, gl = dd_gemv(A.T, rhat, re)
    g, ge = dd_renorm(sc * gh, sc * gl)
    d, de = -g, -ge
    checks = sorted(set(np.unique(np.logspace(0, np.log10(ITERS), 60)
                                  .astype(int)).tolist()) | {ITERS})
    rels, its = [], []
    best = np.inf
    t0 = time.time()
    for it in range(1, ITERS + 1):
        if it % REFRESH == 0:
            fh, fl = dd_gemv(A, v, ve)
            rhat, re = dd_renorm(fh - y, fl)
        uh, ul = dd_gemv(A, d, de)
        c = sc * ddot(uh, ul, uh, ul)
        if not np.isfinite(c) or c <= 0.0:
            break
        alpha = -ddot(g, ge, d, de) / c
        if not np.isfinite(alpha):
            break
        v, ve = dd_axpy(v, ve, alpha, d, de)
        rhat, re = dd_axpy(rhat, re, alpha, uh, ul)
        gh, gl = dd_gemv(A.T, rhat, re)
        gn, gne = dd_renorm(sc * gh, sc * gl)
        if not np.isfinite(gn).all():
            break
        denom = ddot(g, ge, g, ge)
        num = ddot(gn, gne, gn, gne) - ddot(gn, gne, g, ge)
        beta = max(0.0, num / denom)
        p, pe = two_prod(beta, d)
        s, se = two_sum(-gn, p)
        d, de = dd_renorm(s, se + pe + beta * de - gne)
        g, ge = gn, gne
        if it in checks:
            vv = v + ve
            if is_phi:
                rel = float(np.linalg.norm(prob["A_ev"] @ vv - prob["y_ev"])
                            / np.linalg.norm(prob["y_ev"]))
            else:
                rel = float(np.linalg.norm(A @ vv - y) / np.linalg.norm(y))
            rels.append(rel)
            its.append(it)
            best = min(best, rel)
    if is_phi:
        floor = entry["floor_eval_rel_l2"]
    else:
        floor = float(np.sqrt(max(entry["L_star"], 0.0) * R)
                      / np.linalg.norm(y))
    return {"id": entry["id"], "arm": "noB_ddmv", "best_rel": best,
            "floor": floor, "rels": rels, "its": its, "B_k": 0,
            "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- driver


ARM_STYLE = {
    "B": ("full basis B (iter7 ref)", "#1f77b4", "-"),
    "noB": ("no B, plain", "#999999", "-"),
    "noB_dd1": ("no B + dd dots", "#2ca02c", "--"),
    "noB_dd2": ("no B + dd dots+state", "#ff7f0e", "-"),
    "noB_dd3": ("no B + dd full feed-through", "#d62728", "-"),
    "noB_ddmv": ("ALL dd incl. matvecs (diagnostic)", "#000000", "--"),
    "headk64": ("frozen head k=64", "#c7b3e5", "-"),
    "headk128": ("frozen head k=128", "#a07ad1", "-"),
    "headk256": ("frozen head k=256", "#7a3fc1", "-"),
    "headk384": ("frozen head k=384", "#4b0f96", "-"),
    "fhead128": ("fn-space head k=128", "#f2c94c", "--"),
    "fhead256": ("fn-space head k=256", "#e69b00", "--"),
    "fhead384": ("fn-space head k=384", "#8c5a00", "--"),
    "replay_qr_x1": ("replay qr x1", "#17becf", "-"),
    "replay_qr_x1.5": ("replay qr x1.5", "#0e7a85", "-"),
    "replay_qr_x2": ("replay qr x2", "#063f45", "-"),
    "replay_gram_x1": ("replay gram x1", "#e377c2", "--"),
    "replay_gram_x1.5": ("replay gram x1.5", "#a3357f", "--"),
    "replay_gram_x2": ("replay gram x2", "#5c1245", "--"),
    "replay_qr_x0.5": ("replay qr x0.5", "#7fdbe6", ":"),
    "replay_qr_x0.25": ("replay qr x0.25", "#b3ecf2", ":"),
    "hyb32f512": ("hybrid p32 + fn512", "#8c564b", "-"),
    "hyb64f512": ("hybrid p64 + fn512", "#5a2d22", "-"),
}
REPLAY_IDS = ["phi_sine_N256_r4", "phi_exp_N256_r4"]
REPLAY_JOBS = [{"mode": mo, "s_rate": sr,
                "T_h": 2500 if mo == "qr" else 1200}
               for mo in ("qr", "gram") for sr in (1.0, 1.5, 2.0)]
# under-deflation controls: does the mismatched stream at least act like a
# lagging headk (partial protection), or is it poisonous at any dose?
REPLAY_JOBS += [{"mode": "qr", "s_rate": sr, "T_h": 2500}
                for sr in (0.5, 0.25)]


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["id"], r["arm"]): r for r in rows}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharey=True)
    for ax, pid in zip(axes.flat, PROBLEM_IDS):
        for arm in ARM_STYLE:
            r = by.get((pid, arm))
            if r is None:
                continue
            lab, col, ls = ARM_STYLE[arm]
            run_min = np.minimum.accumulate(np.array(r["rels"]))
            ax.plot(r["its"], run_min, color=col, ls=ls, lw=1.5, label=lab)
        ax.axhline(by[(pid, "B")]["floor"], color="k", ls=":", lw=1.0,
                   label="SVD floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 1e0)
        ax.set_title(pid.replace("_N256", "").replace("syn_", "syn:"),
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("iteration")
    axes[0, 0].set_ylabel("rel L2 (running min)")
    axes[1, 0].set_ylabel("rel L2 (running min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=6)
    fig.suptitle("Phase 0: B replacement bake-off on frozen expD07 systems "
                 "(phi = eval rel L2, syn = train rel L2)", y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = OUT_DIR / "bakeoff_trajectories.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def main():
    selftest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {e["id"]: e for e in common.load_manifest()}
    rows_path = OUT_DIR / "bakeoff_rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    have = {(r["id"], r["arm"]) for r in rows}
    jobs = [{"entry": manifest[pid], "arm": arm, "threads": 1}
            for pid in PROBLEM_IDS for arm in ARMS + HEADK_ARMS
            if (pid, arm) not in have]
    fjobs = [{"entry": manifest[pid], "arm": arm, "threads": 1}
             for pid in PROBLEM_IDS for arm in FHEAD_ARMS + HYB_ARMS
             if (pid, arm) not in have]
    ddmv_jobs = [{"entry": manifest[pid]} for pid in DDMV_IDS
                 if (pid, "noB_ddmv") not in have]
    rjobs = [{"entry": manifest[pid], **rj}
             for pid in REPLAY_IDS for rj in REPLAY_JOBS
             if (pid, f"replay_{rj['mode']}_x{rj['s_rate']:g}") not in have]
    print(f"bake-off | {len(jobs)} fast + {len(fjobs)} fspace + "
          f"{len(ddmv_jobs)} ddmv + {len(rjobs)} replay runs "
          f"| {ITERS} iters each")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_arm, j): j for j in jobs}
        futs.update({ex.submit(run_fspace, j): j for j in fjobs})
        futs.update({ex.submit(run_ddmv, j): j for j in ddmv_jobs})
        futs.update({ex.submit(run_replay, j): j for j in rjobs})
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(futs)}] {r['arm']:12s} {r['id']:28s} "
                  f"best={r['best_rel']:.3e} floor={r['floor']:.1e} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    rows_path.write_text(json.dumps(rows))
    figure(rows)

    by = {(r["id"], r["arm"]): r for r in rows}
    all_arms = [a for a in ARM_STYLE
                if any((p, a) in by for p in PROBLEM_IDS)]
    print(f"\n{'problem':<28s}" + "".join(f"{a:>13s}" for a in all_arms)
          + f"{'floor':>10s}")
    for pid in PROBLEM_IDS:
        cells = "".join(f"{by[(pid, a)]['best_rel']:>13.2e}"
                        if (pid, a) in by else f"{'-':>13s}"
                        for a in all_arms)
        print(f"{pid:<28s}{cells}{by[(pid, 'B')]['floor']:>10.1e}")
    phi_ids = [p for p in PROBLEM_IDS if p.startswith("phi")]
    for arm in all_arms:
        vals = [by[(p, arm)]["best_rel"] for p in phi_ids if (p, arm) in by]
        if vals:
            g = np.exp(np.mean(np.log(vals)))
            print(f"geo-mean phi eval rel L2  {arm:12s} {g:.2e}  "
                  f"({len(vals)} cells)")


if __name__ == "__main__":
    main()
