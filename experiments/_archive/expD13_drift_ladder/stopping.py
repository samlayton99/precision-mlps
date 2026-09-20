"""Find the stopping rule for one damping level.

The ladder currently stops each level with an ORACLE (eval error <= 1.3x the
exact damped optimum). That is not available in training. This finds a rule
that is: simple, uses only quantities LSQR already computes, and never costs
much against the oracle.

WHAT IS AVAILABLE FOR FREE, per iteration, from the Golub-Kahan recurrence
    rnorm = ||r||                  (phibar)
    atr   = ||A^T r - mu x||       (phibar * alpha * |c|)   normal-eq residual
    anorm = ||A||_F estimate       (running sum of alpha^2 + beta^2)
    xnorm = ||x||
The standard LSQR/LSMR test is
    test2 = atr / (anorm * rnorm)          -- relative normal-equation residual
and the classical criterion is test2 <= atol.

CANDIDATE RULES
    R1..R3  test2 <= c * alpha        for c = 0.1, 1, 10
            (motivated by the measured law "a level converges to accuracy
             ~alpha", so the natural tolerance scales with alpha)
    R4      fixed budget T = C * alpha^-p, the measured iteration law
    R5      held-out residual plateau: recompute ||r|| on a FRESH batch every
            probe, stop after `patience` probes without improvement
    R6      R1 AND the cap of R4 (belt and braces)

SCORED AGAINST THE ORACLE on two axes, both of which matter:
    accuracy ratio  = eval(rule) / eval(oracle)      (>1 means worse accuracy)
    work ratio      = iters(rule) / iters(oracle)    (>1 means wasted work)
Run both on a static Phi and on the drifting ladder, because expD13 showed the
observable can be wildly optimistic while the geometry moves.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core13 as c

OUT = c.RESULTS / "stopping.jsonl"
T0 = time.time()
CAP = 4096


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def lsqr_record(A, rhs, mu, d, maxiter, probe, probe_at, holdout=None,
                brk=1e-14):
    """LSQR with damping, recording every quantity a stopping rule could use."""
    n = A.shape[0]
    sq = float(np.sqrt(mu)) if mu > 0 else 0.0
    damp = sq > 0
    av = (lambda z: np.concatenate([A @ z, sq * z])) if damp else (lambda z: A @ z)
    atu = ((lambda u: A.T @ u[:n] + sq * u[n:]) if damp else (lambda u: A.T @ u))
    u = np.concatenate([rhs, np.zeros(d)]) if damp else rhs.copy()
    beta = float(np.linalg.norm(u))
    if beta == 0:
        return []
    u /= beta
    b0 = beta
    v = atu(u)
    alpha = float(np.linalg.norm(v))
    if alpha == 0:
        return []
    v /= alpha
    w = v.copy()
    x = np.zeros(d)
    phibar, rhobar = beta, alpha
    anorm2 = alpha ** 2
    at = set(probe_at)
    rec = []
    for it in range(1, maxiter + 1):
        un = av(v) - alpha * u
        beta = float(np.linalg.norm(un))
        if beta <= brk * np.sqrt(anorm2):
            break
        u = un / beta
        vn = atu(u) - beta * v
        alpha = float(np.linalg.norm(vn))
        anorm2 += alpha ** 2 + beta ** 2
        if alpha <= brk * np.sqrt(anorm2):
            break
        v = vn / alpha
        rho = float(np.hypot(rhobar, beta))
        cs, sn = rhobar / rho, beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        x += (phi / rho) * w
        w = v - (theta / rho) * w
        if it in at:
            anorm = float(np.sqrt(anorm2))
            atr = float(abs(phibar * alpha * cs))
            rec.append(dict(it=it, ev=float(probe(x)), rnorm=float(phibar),
                            atr=atr, anorm=anorm, xnorm=float(np.linalg.norm(x)),
                            test1=float(phibar / b0),
                            test2=float(atr / max(anorm * phibar, 1e-300)),
                            hold=(float(holdout(x)) if holdout else 0.0)))
    return rec


# ------------------------------------------------------------- the rules ---
def apply_rules(rec, alpha, oracle_target):
    """Return {rule_name: (iters, eval)} plus the oracle."""
    out = {}
    orc = next((q for q in rec if q["ev"] <= oracle_target), rec[-1])
    out["oracle"] = (orc["it"], orc["ev"])
    for nm, cc in (("test2<=0.1a", 0.1), ("test2<=a", 1.0), ("test2<=10a", 10.0)):
        q = next((z for z in rec if z["test2"] <= cc * alpha), rec[-1])
        out[nm] = (q["it"], q["ev"])
    # R4: fixed budget from the measured iteration law
    Tb = int(min(CAP, max(2, 3.0 * alpha ** -0.7)))
    q = next((z for z in rec if z["it"] >= Tb), rec[-1])
    out["budget 3a^-0.7"] = (q["it"], q["ev"])
    # R5: held-out plateau
    best, bit, bev, pat, stop = np.inf, rec[-1]["it"], rec[-1]["ev"], 0, None
    for z in rec:
        if z["hold"] < best * 0.98:
            best, bit, bev, pat = z["hold"], z["it"], z["ev"], 0
        else:
            pat += 1
            if pat >= 4:
                stop = (z["it"], z["ev"])
                break
    out["held-out plateau"] = stop or (bit, bev)
    # R6: test2<=a AND the budget cap
    q = next((z for z in rec if z["test2"] <= alpha or z["it"] >= Tb), rec[-1])
    out["test2<=a + cap"] = (q["it"], q["ev"])
    return out


ALPHAS = [10.0 ** (-e / 2) for e in range(2, 21)]
PROBS = [("qi1d_256", lambda: c.QI1D(256)),
         ("qi2d_2048", lambda: c.QI2D(2048)),
         ("twin_2048", lambda: c.Twin(2048, label="random gapless  d=2048")),
         ("twin_spec2d", None)]


def spec2d():
    q = c.QI2D(2048)
    p = q.at(0.0)
    s = np.linalg.svd(p["A"], compute_uv=False)
    r = int((s > c.RCOND * s[0]).sum())
    d = p["d"]
    del p, q
    return c.Twin(d, spectrum=s[:r].copy(), row_mult=6,
                  label="random, 2-D spectrum  d=2060")


def run(key, dp, nmult):
    p0 = dp.at(0.0)
    fl, r_true, s1 = c.pool_floor(p0)
    d, n = p0["d"], p0["n"]
    rng = np.random.default_rng(7)
    hold = rng.choice(n, min(n // 4, 4096), replace=False)
    w = np.zeros(d)
    log(f"--- {dp.label} nmult={nmult}: d={d} r={r_true} floor={fl:.2e}")
    for lv, a in enumerate(ALPHAS):
        eta = nmult * a
        p = p0 if eta == 0 else dp.at(eta, seed=7 * lv)
        U, s, Vt, _ = c.spectrum(p)
        mu = float((a * s[0]) ** 2)
        dop = c.eval_err(p, c.damped_exact(U, s, Vt, p["y"], mu))
        Ah, yh = p["A"][hold], p["y"][hold]
        rec = lsqr_record(p["A"], p["y"] - p["A"] @ w, mu, d, CAP,
                          probe=lambda z: c.eval_err(p, w + z),
                          probe_at=c.probe_sched(CAP),
                          holdout=lambda z: float(np.linalg.norm(
                              yh - Ah @ (w + z))))
        if not rec:
            continue
        res = apply_rules(rec, a, 1.3 * dop)
        c.append(OUT, dict(key=key, label=dp.label, nmult=nmult, alpha=a,
                           damped_opt=dop, d=d, rank=r_true, floor=fl,
                           rules={k: [int(v[0]), float(v[1])] for k, v in res.items()}))
        w = w + np.zeros(d)  # advance using the ORACLE stop, so levels stay comparable
        # re-solve to the oracle point for the warm start
        x, _ = c.lsqr_damped(p["A"], p["y"] - p["A"] @ w, mu, d,
                             res["oracle"][0], 0)
        w = w + x
        del p, U, s, Vt
        if res["oracle"][1] > 1.3 * dop:
            log(f"   handoff at alpha={a:.1e}")
            break
    del p0


if __name__ == "__main__":
    if OUT.exists():
        OUT.unlink()
    for key, mk in PROBS:
        dp = (spec2d() if mk is None else mk())
        for nmult in (0.0, 1.0):
            run(key, dp, nmult)
        del dp
    log("done")
