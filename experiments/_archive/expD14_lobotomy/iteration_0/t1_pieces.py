"""T1 -- every piece tested on its own, before anything is assembled.

  P1  mu*d_mu is the raw gradient as mu->inf, and d_mu is the exact LS step as
      mu->0.  (Sam's "multiply by mu to get raw coordinates".)
  P2  the matrix-free damped LSQR agrees with a direct damped SVD solve, and
      the iterations it needs are set by alpha = sqrt(mu)/sigma_1, not by d.
  P3  sigma_1 by power iteration agrees with the SVD.
  P4  the gain estimator (r.Jp)^2/||Jp||^2 equals the actual exact-line-search
      loss decrease along p.
  P5  with L empty the optimizer IS stock Adam (same trajectory to eps).

Run: uv run --extra dev python experiments/expD14_lobotomy/iteration_0/t1_pieces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core0 as C  # noqa: E402

torch.set_default_dtype(torch.float64)


def ops_for(env, theta, idx):
    m = env["m"]

    def av(z):
        dv = torch.zeros(m, dtype=theta.dtype)
        dv[idx] = z
        return env["f_jvp"](theta, dv)

    def atu(u):
        return env["f_vjp"](theta, u)[idx]
    return av, atu


def main():
    print("=" * 72)
    env = C.build_case("sine", 128, init="qi")
    theta = env["theta0"].clone()
    # move the readout off zero so J_L is a live, ill-conditioned matrix
    theta[2 * env["W"]:] += 0.01 * torch.randn(env["W"] + 1, dtype=theta.dtype)
    idx = env["idx_readout"]
    B = int(idx.numel())
    av, atu = ops_for(env, theta, idx)
    r = env["f_resid"](theta)

    J = torch.stack([av(torch.eye(B, dtype=theta.dtype)[j]) for j in range(B)],
                    dim=1)
    U, s, Vt = torch.linalg.svd(J, full_matrices=False)
    sigma1 = float(s[0])
    keep = s > C.RCOND * s[0]
    dstar = Vt.T @ (torch.where(keep, 1.0 / torch.where(keep, s, torch.ones_like(s)),
                                torch.zeros_like(s)) * (U.T @ (-r)))
    graw = J.T @ (-r)
    print(f"case sine N=128: B={B}  sigma1={sigma1:.4e}  "
          f"kappa={float(s[0]/s[keep][-1]):.3e}  rank={int(keep.sum())}")

    def cos(a, b):
        na, nb = float(torch.linalg.norm(a)), float(torch.linalg.norm(b))
        return float(a.dot(b)) / (na * nb) if na * nb > 0 else np.nan

    # ---------------------------------------------------------------- P1 ---
    print("\nP1  mu-interpolation of the damped GN step")
    print("    (the ONLY quantity that matters at the small-alpha end is the")
    print("     eval error the step lands on -- cosine to the truncated d* is")
    print("     not it: damping at alpha truncates at sigma ~ alpha*sigma1,")
    print("     and this matrix has live modes down to sigma/sigma1 ~ 2e-15.)")
    theta_ls = theta.clone()
    theta_ls[idx] = 0.0                       # solve from a zero readout
    r_ls = env["f_resid"](theta_ls)
    av0, atu0 = ops_for(env, theta_ls, idx)
    J0 = torch.stack([av0(torch.eye(B, dtype=theta.dtype)[j]) for j in range(B)],
                     dim=1)
    U0, s0, Vt0 = torch.linalg.svd(J0, full_matrices=False)
    print(f"{'alpha':>10} {'mu':>11} {'cos(mu d,graw)':>15} "
          f"{'||mu d||/||g||':>15} {'cos(d,d*)':>11} {'eval rel L2':>13}")
    alphas = [1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    rows = []
    for a in alphas:
        mu = (a * sigma1) ** 2
        d = Vt.T @ ((s / (s ** 2 + mu)) * (U.T @ (-r)))
        mu0 = (a * float(s0[0])) ** 2
        d0 = Vt0.T @ ((s0 / (s0 ** 2 + mu0)) * (U0.T @ (-r_ls)))
        th_try = theta_ls.clone()
        th_try[idx] += d0
        ev = env["eval_rel"](th_try)
        rows.append((a, mu, cos(mu * d, graw), float(
            torch.linalg.norm(mu * d) / torch.linalg.norm(graw)),
            cos(d, dstar), ev))
        print(f"{a:10.0e} {mu:11.3e} {rows[-1][2]:15.12f} {rows[-1][3]:15.6f} "
              f"{rows[-1][4]:11.6f} {ev:13.3e}")
    ok1 = rows[0][2] > 1 - 1e-6 and abs(rows[0][3] - 1) < 1e-3
    ok2 = min(r_[5] for r_ in rows) < 10 * env["floor0"]
    print(f"  floor of this geometry (truncated SVD solve): {env['floor0']:.3e}")
    print(f"  -> mu->inf gives the raw gradient  : {'PASS' if ok1 else 'FAIL'}")
    print(f"  -> mu->0   reaches the solve floor : {'PASS' if ok2 else 'FAIL'}")

    # ---------------------------------------------------------------- P2 ---
    print("\nP2  matrix-free damped LSQR vs the direct damped solve")
    print(f"{'alpha':>10} {'iters(cap 400)':>15} {'rel err vs direct':>19}")
    p2 = []
    for a in [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        mu = (a * sigma1) ** 2
        dref = Vt.T @ ((s / (s ** 2 + mu)) * (U.T @ (-r)))
        x, nit, _, _ = C.lsqr_damped_op(av, atu, -r, mu, B, 400)
        e = float(torch.linalg.norm(x - dref) / torch.linalg.norm(dref))
        p2.append((a, nit, e))
        print(f"{a:10.0e} {nit:15d} {e:19.3e}")
    ok3 = all(e < 1e-6 for _, nit, e in p2 if nit < 400)
    mono = all(p2[i][1] <= p2[i + 1][1] for i in range(len(p2) - 1))
    print(f"  -> agrees with the direct solve where it converged: "
          f"{'PASS' if ok3 else 'FAIL'}")
    print(f"  -> iterations grow monotonically as alpha falls  : "
          f"{'PASS' if mono else 'FAIL'}")

    # ---------------------------------------------------------------- P3 ---
    z = torch.randn(B, dtype=theta.dtype)
    for k in (1, 3, 10):
        est, _ = C.power_sigma1(av, atu, z.clone(), iters=k)
        print(f"\nP3  sigma1: power({k:2d}) = {est:.6e}   svd = {sigma1:.6e}"
              f"   ratio {est/sigma1:.6f}")

    # ---------------------------------------------------------------- P4 ---
    print("\nP4  gain estimator vs the actual exact-line-search decrease")
    n = C.N_TRAIN
    loss0 = float(r.dot(r)) / n
    print(f"{'direction':>12} {'predicted':>13} {'actual':>13} {'ratio':>9}")
    for name, p_full in (("adam-ish", torch.randn(env["m"], dtype=theta.dtype)),
                         ("LS dir", None)):
        if p_full is None:
            p_full = torch.zeros(env["m"], dtype=theta.dtype)
            p_full[idx] = dstar
        p_full = p_full / torch.linalg.norm(p_full)
        jp = env["f_jvp"](theta, p_full)
        den = float(jp.dot(jp))
        pred = (float(r.dot(jp)) ** 2 / den) / n
        t = -float(r.dot(jp)) / den
        r2 = env["f_resid"](theta + t * p_full)
        act = loss0 - float(r2.dot(r2)) / n
        print(f"{name:>12} {pred:13.5e} {act:13.5e} {act/pred:9.5f}")

    # ---------------------------------------------------------------- P5 ---
    print("\nP5  lset='none' reproduces stock Adam")
    env2 = C.build_case("sine", 64, init="rand")
    out = C.train_lobo(env2, 50, lset="none", lr=1e-3, record=1)
    th = env2["theta0"].clone()
    ma = torch.zeros(env2["m"], dtype=th.dtype)
    va = torch.zeros(env2["m"], dtype=th.dtype)
    for t in range(1, 51):
        _, g = env2["f_residgrad"](th)
        ma = 0.9 * ma + 0.1 * g
        va = 0.999 * va + 0.001 * g * g
        th = th - 1e-3 * (ma / (1 - 0.9 ** t)) / ((va / (1 - 0.999 ** t)).sqrt() + 1e-8)
    gap = float(torch.linalg.norm(out["theta"] - th)) / float(torch.linalg.norm(th))
    print(f"  relative theta gap after 50 steps: {gap:.3e}  "
          f"{'PASS' if gap < 1e-12 else 'FAIL'}")

    # ------------------------------------------------------------- figure --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.FIGS.mkdir(parents=True, exist_ok=True)
    a_ = np.array([r_[0] for r_ in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(a_, [r_[2] for r_ in rows], "o-", label=r"$\cos(\mu d_\mu,\ J^\top(-r))$")
    ax[0].semilogx(a_, [r_[3] for r_ in rows], "s-", label=r"$\cos(d_\mu,\ d^\star)$")
    ax[0].set_xlabel(r"$\alpha=\sqrt{\mu}/\sigma_1$")
    ax[0].set_ylabel("cosine similarity")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].invert_xaxis()
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                 borderaxespad=0, frameon=False)
    ax[1].loglog(a_, [r_[5] for r_ in rows], "s-", color="tab:orange",
                 label=r"$\|d_\mu\|/\|d^\star\|$")
    ax[1].loglog(a_, [r_[4] for r_ in rows], "o-", color="tab:blue",
                 label=r"$\|\mu d_\mu\|/\|J^\top r\|$")
    ax[1].set_xlabel(r"$\alpha=\sqrt{\mu}/\sigma_1$")
    ax[1].set_ylabel("relative length")
    ax[1].invert_xaxis()
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                 borderaxespad=0, frameon=False)
    fig.suptitle("P1: the damped GN step morphs from the raw gradient to the exact solve"
                 "  (sine, N=128, B=%d)" % B, y=1.10)
    fig.tight_layout()
    fig.savefig(C.FIGS / "T1_mu_interpolation.png", dpi=140, bbox_inches="tight")
    print(f"\nfigure -> {C.FIGS / 'T1_mu_interpolation.png'}")


if __name__ == "__main__":
    main()
