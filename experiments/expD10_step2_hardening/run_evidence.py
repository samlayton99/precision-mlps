"""Regenerate the expD10 evidence pack: evidence.json + all figures.

Covers: E4 stopping (stress trajectory + stopped-vs-oracle), E1 ordering
(true/permuted/seriated through the whitening), E2 memory (implicit-dd
parity + fp64-implicit control), E3 dd+bJac ordering, precision-agnostic
fp32 arm. Noise/batching/warm data come from run_floors.py / warm.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import lsmr, LinearOperator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core10 as c                      # noqa: E402

OUT = c.RESULTS
FIG = c.FIGS
FIG.mkdir(parents=True, exist_ok=True)
EV = {}
RNG = np.random.default_rng(7)


def lsmr_traj(B, b, iters, cb, stride=100):
    beta = np.linalg.norm(b); u = b / beta
    v = B.T @ u; alpha = np.linalg.norm(v); v = v / alpha
    alphabar = alpha; zetabar = alpha * beta
    rho = rhobar = cbar = 1.0; sbar = 0.0
    h = v.copy(); hbar = np.zeros(B.shape[1]); x = np.zeros(B.shape[1])
    for j in range(1, iters + 1):
        u = B @ v - alpha * u; beta = np.linalg.norm(u); u = u / beta
        v = B.T @ u - beta * v; alpha = np.linalg.norm(v); v = v / alpha
        rhoold = rho; rho = np.hypot(alphabar, beta)
        cs = alphabar / rho; sn = beta / rho
        theta = sn * alpha; alphabar = cs * alpha
        rhobarold = rhobar; thetabar = sbar * rho
        rhobar = np.hypot(cbar * rho, theta)
        cbar = cbar * rho / rhobar; sbar = theta / rhobar
        zeta = cbar * zetabar; zetabar = -sbar * zetabar
        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (theta / rho) * h
        if j % stride == 0:
            cb(j, x, abs(zetabar))
    return x


def fs_solve(A, y, k=128, order=None, maxiter=8000):
    pieces, B = c.whiten(A, k, order=order)
    res = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=maxiter)
    return c.unwhiten(pieces, res[0], A.shape[1]), int(res[2]), pieces, B


def fp64_implicit(A, pieces):
    n, d = A.shape

    def mv(v):
        out = np.zeros(n)
        for C, piv, R, nk in pieces:
            if nk == 0:
                continue
            t = sla.solve_triangular(R[:, :nk], v[C[:nk]], lower=False)
            out += A[:, C[piv[:nk]]] @ t
        return out

    def rmv(u):
        out = np.zeros(d)
        for C, piv, R, nk in pieces:
            if nk == 0:
                continue
            s = A[:, C[piv[:nk]]].T @ u
            out[C[:nk]] = sla.solve_triangular(R[:, :nk].T, s, lower=True)
        return out

    return LinearOperator((n, d), matvec=mv, rmatvec=rmv, dtype=float)


# ---------------------------------------------------------------- E4 -------
def e4_stopping():
    print("E4 ...")
    rows = []
    for fn in c.cv.TARGETS:
        for N in c.cv.WIDTHS:
            prob = c.cv.make_problem(fn, N)
            A, y = prob["A"], prob["y"]
            d = A.shape[1]
            w, it, pieces, B = fs_solve(A, y)
            e_stop = c.eval_err(prob, w)
            e_or = min(
                c.eval_err(prob, c.unwhiten(
                    pieces, lsmr(B, y, atol=0, btol=0, conlim=0,
                                 maxiter=m)[0], d))
                for m in (25, 50, 100, 200, 342, 500, 800))
            rows.append({"fn": fn, "N": N, "stopped": e_stop, "oracle": e_or,
                         "floor": prob["floor_svd"], "iters": it})
    EV["stopping"] = rows

    prob = c.cv.make_problem("abs_cubed", 512)
    A, y = prob["A"], prob["y"]
    d = A.shape[1]
    pieces, B = c.whiten(A, 64)
    traj = []

    def cb(j, z, zb):
        traj.append({"it": j, "zetabar": zb, "znorm": float(np.linalg.norm(z)),
                     "eval": c.eval_err(prob, c.unwhiten(pieces, z, d))})
    lsmr_traj(B, y, 20000, cb, stride=200)
    res = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=60000)
    EV["stress"] = {"traj": traj, "fs_it": int(res[2]),
                    "fs_eval": c.eval_err(prob, c.unwhiten(pieces, res[0], d)),
                    "floor": prob["floor_svd"]}


# ---------------------------------------------------------------- E1 -------
def e1_ordering():
    print("E1 ...")
    rows = []
    for fn in c.cv.TARGETS:
        for N in c.cv.WIDTHS:
            prob = c.cv.make_problem(fn, N)
            A, y = prob["A"], prob["y"]
            d = A.shape[1]
            perm = RNG.permutation(d)
            Ap = A[:, perm]
            e_true, it_t, _, _ = fs_solve(A, y)
            e_true = c.eval_err(prob, e_true)

            def back(wp):
                w = np.zeros(d); w[perm] = wp
                return w
            wp, it_p, _, _ = fs_solve(Ap, y)
            e_perm = c.eval_err(prob, back(wp))
            sr = RNG.choice(A.shape[0], 256, replace=False)
            order = c.seriate(Ap[sr])
            ws, it_s, _, _ = fs_solve(Ap, y, order=order)
            e_ser = c.eval_err(prob, back(ws))
            rows.append({"fn": fn, "N": N, "true": e_true, "perm": e_perm,
                         "ser": e_ser, "it_true": it_t, "it_ser": it_s})
    EV["ordering"] = rows


# ---------------------------------------------------------------- E2 -------
def e2_memory():
    print("E2 ...")
    rows = []
    for fn, N in (("sine", 128), ("runge", 256), ("sine_8pi", 256)):
        prob = c.cv.make_problem(fn, N)
        A, y = prob["A"], prob["y"]
        d = A.shape[1]
        pieces, B = c.whiten(A, 128)
        r1 = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=3000)
        e_mat = c.eval_err(prob, c.unwhiten(pieces, r1[0], d))
        r2 = lsmr(fp64_implicit(A, pieces), y, atol=1e-15, btol=1e-15,
                  conlim=0, maxiter=1500)
        e_64 = c.eval_err(prob, c.unwhiten(pieces, r2[0], d))
        opdd = c.implicit_dd_operator(A, pieces)
        # oracle-grid for dd (FS variance separated out in the writeup)
        e_dd = min(c.eval_err(prob, c.unwhiten(
            pieces, lsmr(opdd, y, atol=0, btol=0, conlim=0, maxiter=m)[0], d))
            for m in (150, 350, 500))
        r3 = lsmr(opdd, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=1500)
        e_dd_stop = c.eval_err(prob, c.unwhiten(pieces, r3[0], d))
        rows.append({"fn": fn, "N": N, "materialized": e_mat,
                     "fp64_implicit": e_64, "dd_implicit": e_dd,
                     "dd_implicit_stopped": e_dd_stop})
    EV["memory"] = rows


# ---------------------------------------------------------------- E3 -------
def e3_ddbjac():
    print("E3 ...")
    c.patch_dd_fast()
    import ddpc
    rows = []
    for fn, N in (("sine", 128), ("runge", 128), ("sine_8pi", 256), ("runge", 256)):
        prob = c.cv.make_problem(fn, N)
        A, y = prob["A"], prob["y"]
        d = A.shape[1]
        rec = {"fn": fn, "N": N}
        for kind in ("contig", "random", "seriated"):
            if kind == "contig":
                perm = np.arange(d)
            elif kind == "random":
                perm = RNG.permutation(d)
            else:
                pr = RNG.permutation(d)
                sr = RNG.choice(A.shape[0], 256, replace=False)
                perm = pr[c.seriate(A[:, pr][sr])]
            Ap = A[:, perm]
            fac = ddpc.block_factors(Ap, 128, rcond=1e-15)
            _, p = ddpc.plsqr_dd(Ap, y, fac, 400, prob["A_ev"][:, perm],
                                 prob["y_ev"], prob["ye_norm"],
                                 probe=(100, 200, 400))
            rec[kind] = {str(i): p[i] for i in (100, 200, 400)}
        rows.append(rec)
    EV["ddbjac"] = rows


# ------------------------------------------------------------- fp32 --------
def e_fp32():
    print("fp32 ...")
    rows = []
    for fn in c.cv.TARGETS:
        for N in c.cv.WIDTHS:
            prob = c.cv.make_problem(fn, N)
            A64, y64 = prob["A"], prob["y"]
            d = A64.shape[1]
            A = A64.astype(np.float32); y = y64.astype(np.float32)
            pieces = []; B = np.zeros(A.shape, dtype=np.float32)
            for t in range(0, d, 128):
                C = np.arange(t, min(t + 128, d))
                Q, Rf, piv = sla.qr(A[:, C], mode="economic", pivoting=True)
                dg = np.abs(np.diag(Rf))
                nk = int((dg > 1e-6 * max(dg[0], 1e-30)).sum())
                B[:, C[:nk]] = Q[:, :nk]
                pieces.append((C, piv, Rf[:nk, :], nk))
            res = lsmr(B, y, atol=1e-7, btol=1e-7, conlim=0, maxiter=5000)
            z = res[0].astype(np.float32)
            w = np.zeros(d, dtype=np.float32)
            for C, piv, Rf, nk in pieces:
                if nk == 0:
                    continue
                cc = sla.solve_triangular(Rf[:, :nk], z[C[:nk]], lower=False)
                out = np.zeros(len(C), dtype=np.float32)
                out[piv[:nk]] = cc
                w[C] = out
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            keep = s > 1e-6 * s[0]
            a32 = (Vt.T * np.where(keep, 1 / np.where(keep, s, 1), 0)) @ (U.T @ y)
            e64r = [r for r in EV["stopping"] if r["fn"] == fn and r["N"] == N][0]
            rows.append({"fn": fn, "N": N,
                         "fp32": c.eval_err(prob, w.astype(np.float64)),
                         "fp32_floor": c.eval_err(prob, a32.astype(np.float64)),
                         "fp64": e64r["stopped"], "fp64_floor": e64r["floor"],
                         "iters": int(res[2])})
    EV["fp32"] = rows


# ------------------------------------------------------------ figures ------
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [f"{r['fn']}\nN={r['N']}" for r in EV["stopping"]]
    xx = np.arange(len(cells))

    # F1 stopping
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    t = EV["stress"]["traj"]
    it = [q["it"] for q in t]
    ax.semilogy(it, [q["eval"] for q in t], color="C0", label="eval rel $L_2$ (oracle view)")
    ax.semilogy(it, [max(q["zetabar"], 1e-26) for q in t], color="C1",
                label=r"$\|B^\top r\|$ (observable)")
    ax.semilogy(it, [q["znorm"] for q in t], color="C2", label=r"$\|z\|$ (observable)")
    ax.axvline(EV["stress"]["fs_it"], color="k", ls="--", lw=1.2,
               label=f"Fong-Saunders stop (it {EV['stress']['fs_it']})")
    ax.set_xlabel("LSMR iteration"); ax.set_ylim(1e-26, 1e8)
    ax.set_title("stress cell abs_cubed d=922 k=64: signatures precede damage", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    ax = axs[1]
    ax.semilogy(xx, [r["oracle"] for r in EV["stopping"]], "o", color="C0", label="oracle best")
    ax.semilogy(xx, [r["stopped"] for r in EV["stopping"]], "s", color="C3", label="stopped (observable)")
    ax.semilogy(xx, [r["floor"] for r in EV["stopping"]], "_", ms=16, color="k", label="SVD floor")
    ax.set_xticks(xx); ax.set_xticklabels(cells, fontsize=7)
    ax.set_ylim(1e-16, 1e-8); ax.set_ylabel("eval rel $L_2$")
    ax.set_title("9 cells: stopped vs oracle vs floor", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "F1_stopping.png", dpi=140); plt.close(fig)

    # F2 ordering
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    for key, col, lab in (("true", "C0", "true order"), ("perm", "C3", "permuted (random blocks)"),
                          ("ser", "C2", "seriated (recovered)")):
        ax.semilogy(xx, [r[key] for r in EV["ordering"]], "o-", color=col, label=lab, ms=4)
    ax.set_xticks(xx); ax.set_xticklabels(cells, fontsize=7)
    ax.set_ylim(1e-16, 1e-4); ax.set_ylabel("eval rel $L_2$")
    ax.set_title("fp64 whitening: seriation recovers a destroyed order", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    ax = axs[1]
    dcells = [f"{r['fn']} N={r['N']}" for r in EV["ddbjac"]]
    dx = np.arange(len(dcells))
    for key, col, lab in (("contig", "C0", "contiguous"), ("random", "C3", "random"),
                          ("seriated", "C2", "seriated")):
        ax.semilogy(dx, [r[key]["400"] for r in EV["ddbjac"]], "o-", color=col, label=lab, ms=4)
    ax.set_xticks(dx); ax.set_xticklabels(dcells, fontsize=7)
    ax.set_ylim(1e-17, 1e-3); ax.set_ylabel("eval rel $L_2$ at 400 iters")
    ax.set_title("dd + block-Jacobi (k=128): ordering matters here too", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "F2_ordering.png", dpi=140); plt.close(fig)

    # F3 memory + precision
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    mc = [f"{r['fn']} N={r['N']}" for r in EV["memory"]]
    mx = np.arange(len(mc)); wdt = 0.27
    ax.bar(mx - wdt, [r["materialized"] for r in EV["memory"]], wdt, color="C0", label="materialized $B$")
    ax.bar(mx, [r["fp64_implicit"] for r in EV["memory"]], wdt, color="C3", label="implicit, fp64 (control)")
    ax.bar(mx + wdt, [r["dd_implicit"] for r in EV["memory"]], wdt, color="C2", label="implicit, compensated")
    ax.set_yscale("log"); ax.set_ylim(1e-16, 1e-4)
    ax.set_xticks(mx); ax.set_xticklabels(mc, fontsize=8); ax.set_ylabel("eval rel $L_2$")
    ax.set_title("killing the $n\\times d$ copy: implicit whitening", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3, axis="y")
    ax = axs[1]
    ax.semilogy(xx, [r["fp64"] for r in EV["fp32"]], "o", color="C0", label="pipeline @ fp64")
    ax.semilogy(xx, [r["fp64_floor"] for r in EV["fp32"]], "_", ms=14, color="C0", label="fp64 SVD floor")
    ax.semilogy(xx, [r["fp32"] for r in EV["fp32"]], "s", color="C1", label="pipeline @ fp32")
    ax.semilogy(xx, [r["fp32_floor"] for r in EV["fp32"]], "_", ms=14, color="C1", label="fp32 SVD floor")
    ax.set_xticks(xx); ax.set_xticklabels(cells, fontsize=7)
    ax.set_ylim(1e-16, 1e-3); ax.set_ylabel("eval rel $L_2$")
    ax.set_title("precision-agnostic: same pipeline lands on each dtype's floor", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=4,
              borderaxespad=0, fontsize=7, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "F3_memory_precision.png", dpi=140); plt.close(fig)

    # F4 floors (noise + batching + anytime)
    floors = json.loads((OUT / "floors.json").read_text())
    warm = json.loads((OUT / "warm.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    sigs = [1e-8, 1e-6, 1e-4, 1e-2]
    for fn, col in (("sine", "C0"), ("sine_8pi", "C1"), ("runge", "C2")):
        for N, mk in ((64, "o"), (128, "s"), (256, "^")):
            pts = [r for r in floors if r["arm"] == "noise" and r["fn"] == fn
                   and r["N"] == N and r["noise"] > 0]
            ax.loglog([r["noise"] for r in pts], [r["eval"] for r in pts],
                      mk, color=col, ms=4)
    ss = np.array(sigs)
    ax.loglog(ss, 0.272 * ss, "k--", lw=1.2, label=r"statistical floor $0.272\,\sigma$")
    ax.set_xlabel(r"noise $\sigma_{rel}$"); ax.set_ylabel("eval rel $L_2$")
    ax.set_title("noise: all 9 cells on the floor, observable stopping", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=1,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    ax = axs[1]
    for fn, col in (("sine", "C0"), ("sine_8pi", "C1"), ("runge", "C2")):
        for N, mk in ((64, "o"), (128, "s"), (256, "^")):
            pts = sorted([r for r in floors if r["arm"] == "batch" and r["fn"] == fn
                          and r["N"] == N], key=lambda r: r["bmult"])
            ax.semilogy([r["bmult"] for r in pts], [r["eval"] for r in pts],
                        mk + "-", color=col, ms=4, lw=0.8,
                        label=f"{fn} N={N}" if N == 128 else None)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["d", "2d", "4d", "8d"])
    ax.set_ylim(1e-16, 1e-4); ax.set_xlabel("batch rows b"); ax.set_ylabel("eval rel $L_2$")
    ax.set_title("batching: whiten+solve on the batch; b>=4d at floor", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3,
              borderaxespad=0, fontsize=7, frameon=False)
    ax.grid(alpha=.3)
    ax = axs[2]
    curve = warm["anytime"]
    ax.loglog([q[0] for q in curve], [q[1] for q in curve], "o-", color="C0",
              label="warm from $10^{-3}$ (sine N=256)")
    ax.axhline(1.6e-14, color="k", ls=":", lw=1, label="cell floor")
    ax.set_xlabel("pass budget (LSMR iterations)"); ax.set_ylabel("eval rel $L_2$")
    ax.set_ylim(1e-15, 1e-2)
    ax.set_title("anytime: smooth progress at any budget", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2,
              borderaxespad=0, fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "F4_floors.png", dpi=140); plt.close(fig)
    print("figures saved to", FIG)


def main():
    t0 = time.time()
    e4_stopping()
    e1_ordering()
    e2_memory()
    e_fp32()
    e3_ddbjac()
    (OUT / "evidence.json").write_text(json.dumps(EV, indent=1))
    figures()
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
