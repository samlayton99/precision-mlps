"""iteration_10 Phase 0 diagnostic -- WHAT is the memoryless leftover, and what
does a correction step actually see?

The refinement ladder came back null (ir_dd == ir_fp64 == noB), so the wall is
not the precision of the residual. This instruments the stall directly:

  1. spectral profile of the leftover residual vs the spectrum and vs y
  2. coefficient error in the right-singular basis (where is v wrong?)
  3. what an inner correction solve achieves per outer step (the contraction
     factor the refinement argument needs, measured)
  4. the same, with the correction problem's gradient supplied EXACTLY
     (oracle A^T r) -- separates "cannot see the signal" from
     "cannot act on the signal"

Run: python experiments/expD08_qi_init_nlcg/refine_diag.py [--pid ...]
Out: results/.../iteration_10/phase0/diag_<pid>.{png,json}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expD07_lstsq_optim_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from refine import dd_gemv, dd_add, dd_axpy  # noqa: E402

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / "iteration_10" / "phase0")


def memoryless_cg(A, y, iters, v0=None, r0=None, oracle=None, track=None):
    """Memoryless CG (carried residual, exact GN step, PR+). Returns (v, r).

    oracle: optional callable v -> exact gradient (2/R) A^T (A v - y), computed
            in extended precision. Isolates the gradient channel.
    """
    R, m = A.shape
    sc = 2.0 / R
    v = np.zeros(m) if v0 is None else v0.copy()
    rhat = (-y.copy() if r0 is None else r0.copy())
    g = oracle(v) if oracle else sc * (A.T @ rhat)
    d = -g
    for it in range(iters):
        u = A @ d
        c = sc * float(u @ u)
        if not math.isfinite(c) or c <= 0.0:
            break
        alpha = -float(g @ d) / c
        if not math.isfinite(alpha):
            break
        v = v + alpha * d
        rhat = rhat + alpha * u
        g_new = oracle(v) if oracle else sc * (A.T @ rhat)
        if not np.isfinite(g_new).all():
            break
        den = float(g @ g)
        beta = max(0.0, (float(g_new @ g_new) - float(g_new @ g)) / den) if den > 0 else 0.0
        d = -g_new + beta * d
        g = g_new
        if track is not None:
            track.append(float(np.linalg.norm(rhat)))
    return v, rhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="phi_sine_N256_r4")
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()

    man = {e["id"]: e for e in common.load_manifest()}
    entry = man[args.pid]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    sc = 2.0 / R
    is_phi = entry["kind"] == "phi"
    if is_phi:
        A_ev, y_ev = prob["A_ev"], prob["y_ev"]
        ny_ev = float(np.linalg.norm(y_ev))
        met = lambda v: float(np.linalg.norm(A_ev @ v - y_ev) / ny_ev)
    else:
        met = lambda v: float(np.linalg.norm(A @ v - y) / np.linalg.norm(y))
    ny = float(np.linalg.norm(y))

    print(f"{args.pid}: A is {R}x{m}")
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  sigma: {S[0]:.3e} .. {S[-1]:.3e}   kappa {S[0]/S[-1]:.2e}")
    eff = int((S / S[0] > 1e-15).sum())
    print(f"  #sigma/sigma1 > 1e-15: {eff}  (>1e-10: {(S/S[0] > 1e-10).sum()})")

    v_ref = Vt.T @ ((U.T @ y) / np.where(S / S[0] > 1e-15, S, np.inf))
    print(f"  truncated-SVD reference: metric {met(v_ref):.3e}  "
          f"|v|={np.linalg.norm(v_ref):.3e}")

    # ---------------------------------------------------------- 1. the stall
    v_st, r_st = memoryless_cg(A, y, args.iters)
    print(f"\nmemoryless stall after {args.iters} iters: metric {met(v_st):.3e}  "
          f"|r|/|y| {np.linalg.norm(r_st)/ny:.3e}  |v|={np.linalg.norm(v_st):.3e}")

    r_true = A @ v_st - y
    print(f"  carried vs true residual gap: "
          f"{np.linalg.norm(r_st - r_true)/np.linalg.norm(r_true):.3e} relative")

    # spectral profiles
    ur = np.abs(U.T @ r_true)
    uy = np.abs(U.T @ y)
    verr = np.abs(Vt @ (v_st - v_ref))
    vref = np.abs(Vt @ v_ref)

    # ------------------------------------------ 2. what a correction step sees
    # exact gradient of the correction problem, in dd
    gh, gl = dd_gemv(A.T, r_true)
    g_exact = sc * (gh + gl)
    g_fp64 = sc * (A.T @ r_st)
    gn = float(np.linalg.norm(g_exact))
    print(f"\ncorrection-problem gradient: |g_exact| {gn:.3e}   "
          f"fp64 gradient error {np.linalg.norm(g_fp64-g_exact)/gn:.3e} relative")
    # spectral content of the gradient signal vs the fp64 matvec noise
    vg = np.abs(Vt @ g_exact)
    vgn = np.abs(Vt @ (g_fp64 - g_exact))

    # the exact correction, and how big it is
    delta_star = v_ref - v_st
    print(f"  exact correction |delta*|/|v| = "
          f"{np.linalg.norm(delta_star)/np.linalg.norm(v_st):.3e}")
    print(f"  its spectral spread: "
          f"top |v_i.delta*| at i={int(np.argmax(np.abs(Vt@delta_star)))}, "
          f"energy above i={eff}: "
          f"{np.linalg.norm((Vt@delta_star)[eff:])/np.linalg.norm(delta_star):.2e}")

    # -------------------------------------- 3. measured outer contraction rate
    print("\nouter refinement, measured per-step contraction:")
    outer = []
    v, ve = v_st.copy(), np.zeros(m)
    for k in range(6):
        fh, fl = dd_gemv(A, v, ve)
        r, rl = dd_add(fh, fl, -y, np.zeros(R))
        n0 = float(np.linalg.norm(r + rl))
        dv, s = memoryless_cg(A, np.zeros(R), args.iters // 3, r0=r.copy())
        n1 = float(np.linalg.norm(A @ dv + (r + rl)))
        v, ve = dd_axpy(v, ve, 1.0, 0.0, dv, np.zeros(m))
        mt = met(v + ve)
        outer.append({"k": k, "r0": n0 / ny, "inner_reduction": n1 / max(n0, 1e-300),
                      "metric": mt, "dnorm": float(np.linalg.norm(dv))})
        print(f"  outer {k}: |r|/|y| {n0/ny:.3e} -> inner cut it to "
              f"{n1/max(n0,1e-300):.3e} of itself   metric {mt:.3e}  "
              f"|delta| {np.linalg.norm(dv):.2e}")

    # ------------------------------------------ 4. oracle-gradient correction
    print("\nsame, but the inner solve gets the EXACT (dd) gradient:")
    def oracle_factory(r_base):
        def orc(dv):
            hh, hl = dd_gemv(A, dv)
            s, sl = dd_add(hh, hl, r_base, np.zeros(R))
            th, tl = dd_gemv(A.T, s, sl)
            return sc * (th + tl)
        return orc

    oracle_rows = []
    v2, ve2 = v_st.copy(), np.zeros(m)
    for k in range(3):
        fh, fl = dd_gemv(A, v2, ve2)
        r, rl = dd_add(fh, fl, -y, np.zeros(R))
        n0 = float(np.linalg.norm(r + rl))
        dv, _ = memoryless_cg(A, np.zeros(R), 400, r0=r.copy(),
                              oracle=oracle_factory(r + rl))
        n1 = float(np.linalg.norm(A @ dv + (r + rl)))
        v2, ve2 = dd_axpy(v2, ve2, 1.0, 0.0, dv, np.zeros(m))
        mt = met(v2 + ve2)
        oracle_rows.append({"k": k, "r0": n0 / ny,
                            "inner_reduction": n1 / max(n0, 1e-300), "metric": mt})
        print(f"  outer {k}: |r|/|y| {n0/ny:.3e} -> {n1/max(n0,1e-300):.3e} "
              f"of itself   metric {mt:.3e}")

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    idx = np.arange(len(S))
    s_n = S / S[0]

    a = ax[0]
    a.semilogy(idx, s_n, color="k", lw=1.4, label=r"$\sigma_i/\sigma_1$")
    a.semilogy(idx, uy / uy.max(), color="#1f77b4", lw=1.0, alpha=.8,
               label=r"$|u_i^Ty|$ (norm.)")
    a.semilogy(idx, ur / uy.max(), color="#d62728", lw=1.0,
               label=r"$|u_i^Tr_{\rm stall}|$ (same norm.)")
    a.set_title("spectrum, target, and the leftover")
    a.set_xlabel("singular index $i$")

    a = ax[1]
    a.semilogy(idx, vref / vref.max(), color="#1f77b4", lw=1.0,
               label=r"$|v_i^T v_{\rm lstsq}|$")
    a.semilogy(idx, verr / vref.max(), color="#d62728", lw=1.0,
               label=r"$|v_i^T(v_{\rm stall}-v_{\rm lstsq})|$")
    a.set_title("where the coefficient vector is wrong")
    a.set_xlabel("singular index $i$")

    a = ax[2]
    a.semilogy(idx, vg / vg.max(), color="#111111", lw=1.1,
               label=r"$|v_i^Tg|$ exact (dd)")
    a.semilogy(idx, vgn / vg.max(), color="#ff7f0e", lw=1.0,
               label=r"$|v_i^T(g_{\rm fp64}-g)|$ noise")
    a.set_title("correction gradient: signal vs fp64 matvec noise")
    a.set_xlabel("singular index $i$")

    for a in ax:
        a.set_ylim(1e-20, 5.0)
        a.grid(alpha=.25, which="both", lw=.4)
        a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2,
                 frameon=False, fontsize=8, borderaxespad=0)
    fig.suptitle(f"{args.pid} -- memoryless stall {met(v_st):.2e}, "
                 f"floor {(_f := entry.get('floor_eval_rel_l2', 0)) and f'{_f:.2e}'}",
                 y=1.0, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"diag_{args.pid}.png", dpi=140)
    plt.close(fig)

    (OUT_DIR / f"diag_{args.pid}.json").write_text(json.dumps({
        "pid": args.pid, "R": R, "m": m, "kappa": float(S[0] / S[-1]),
        "eff_rank_1e15": eff, "stall_metric": met(v_st),
        "ref_metric": met(v_ref), "outer": outer, "oracle": oracle_rows,
        "delta_star_rel": float(np.linalg.norm(delta_star)
                                / np.linalg.norm(v_st)),
    }, indent=1))
    print(f"\nwrote {OUT_DIR}/diag_{args.pid}.png")


if __name__ == "__main__":
    main()
