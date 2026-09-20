"""Compute budget for the Adam + damped-GN + terminal-solve optimizer.

B1  streaming QR absorbs any batch size; Gram accumulation does not
B2  where the compute actually goes, as a function of network DEPTH

Cost model (standard 6ND accounting):
    W      = 6*P*b        one forward+backward over a batch of b, P parameters
    J      = free         the linear block's Jacobian IS the cached activations
    QR     = 2(d+b)d^2    folding a batch into the triangular factor R
    inner  = 2*T*d^2      T LSQR iterations against R (d x d), not the raw rows
    term   = 2*d^3        one terminal solve: ~r matvecs + one-sided reorth
For a transformer, one layer holds ~12h^2 parameters and P ~ 12*L*h^2, so with
d = h the ratio d^2/P is 1/(12L): the overhead falls as the model gets DEEPER
and is independent of width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core12 as k


def data_b1():
    rows = []
    for name, mk in (("QI-1D N=256", lambda: k.prob_qi1d(256)),
                     ("QI-2D N=256", lambda: k.prob_qi2d(256)),
                     ("random gapless d=512", lambda: k.prob_twin(512))):
        p = mk()
        A, y, d, n = p["A"], p["y"], p["d"], p["n"]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        kp = s > 1e-15 * s[0]
        ref = (Vt.T * np.where(kp, 1 / np.where(kp, s, 1), 0)) @ (U.T @ y)
        base = k.eval_err(p, ref)
        for b in (d, d // 4, d // 16, d // 64, 8):
            b = max(int(b), 1)
            rng = np.random.default_rng(0)
            order = rng.permutation(n)
            M = np.zeros((0, d + 1))
            G = np.zeros((d, d))
            c = np.zeros(d)
            for i in range(0, n, b):
                S = order[i:i + b]
                M = np.vstack([M, np.hstack([A[S], y[S][:, None]])])
                # Only factor once the buffer is TALL. Calling qr(mode='r') on a
                # wide matrix segfaults in this LAPACK build, and it buys
                # nothing: a buffer with fewer rows than columns is already as
                # compressed as it can be.
                if M.shape[0] >= d + 1:
                    M = np.linalg.qr(M, mode="r")[:d + 1]
                G += A[S].T @ A[S]
                c += A[S].T @ y[S]
            if M.shape[0] > d + 1:
                M = np.linalg.qr(M, mode="r")[:d + 1]
            Ra, rz = M[:, :d], M[:, d]
            Uq, sq, Vq = np.linalg.svd(Ra, full_matrices=False)
            kq = sq > 1e-15 * sq[0]
            xq = (Vq.T * np.where(kq, 1 / np.where(kq, sq, 1), 0)) @ (Uq.T @ rz)
            xg = np.linalg.pinv(G, rcond=1e-15) @ c
            rows.append(dict(label=name, d=d, n=n, b=b, ref=base,
                             qr=k.eval_err(p, xq), gram=k.eval_err(p, xg),
                             state_qr=d * (d + 1) // 2, state_rows=n * d))
            print(f"  {name:<22} b={b:<5d} QR {rows[-1]['qr']:.2e}  "
                  f"Gram {rows[-1]['gram']:.2e}  (ref {base:.2e})", flush=True)
        del p
    return rows


def figure(rows):
    fig, axs = plt.subplots(1, 2, figsize=(14.6, 5.6))
    fig.subplots_adjust(left=0.07, right=0.955, bottom=0.13, top=0.66, wspace=0.28)

    ax = axs[0]
    names = sorted({q["label"] for q in rows})
    for i, nm in enumerate(names):
        sub = sorted([q for q in rows if q["label"] == nm], key=lambda q: -q["b"])
        x = [q["b"] / q["d"] for q in sub]
        ax.loglog(x, [q["qr"] for q in sub], "o-", color=f"C{i}", ms=6, lw=2)
        ax.loglog(x, [q["gram"] for q in sub], "s--", color=f"C{i}", ms=5, lw=1.5,
                  alpha=.75)
        ax.axhline(sub[0]["ref"], color=f"C{i}", ls=":", lw=1.2, alpha=.8)
    ax.plot([], [], "ko-", label="streaming QR  (state $d^2/2$)")
    ax.plot([], [], "ks--", alpha=.75, label="Gram accumulation  (same state)")
    ax.plot([], [], "k:", label="full-rows reference (state $nd$)")
    ax.set_xlabel("batch fraction  $b/d$")
    ax.set_ylabel("eval rel $L_2$")
    ax.set_ylim(1e-16, 1e-2)
    ax.invert_xaxis()
    ax.grid(alpha=.3, which="both")
    ax.set_title("B1  batch size stops mattering", fontsize=10.5, pad=54)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8, frameon=False)

    ax = axs[1]
    L = np.logspace(0, 2.3, 200)
    h, b, T, S = 8192.0, 2048.0, 100.0, 1e5
    P = 12 * L * h ** 2
    W = 6 * P * b
    qr = 2 * (h + b) * h ** 2
    inner = 2 * T * h ** 2
    term = 2 * h ** 3
    ax.loglog(L, (qr + inner) / W, "-", color="C1", lw=2.4,
              label="inner loop (QR update + LSQR), per step")
    ax.loglog(L, term / (W * S), "-", color="C2", lw=2.4,
              label=f"terminal solve, amortised over $S=10^5$ steps")
    ax.loglog(L, 1.1 * h ** 2 / (2 * P), "-", color="C3", lw=2.4,
              label="LS memory as a fraction of Adam's $2P$ state")
    ax.axhline(1.0, color="k", lw=1.2)
    ax.text(1.15, 1.35, "= as expensive as Adam itself", fontsize=8)
    for Lv, nm in ((1, "1 layer\n(our QI net)"), (32, "32 layers"), (96, "96 layers")):
        ax.axvline(Lv, color="grey", ls=":", lw=1)
        ax.text(Lv * 1.06, 3e-7, nm, fontsize=7.5, rotation=90, va="bottom")
    ax.set_xlabel("network depth $L$  (layers)")
    ax.set_ylabel("fraction of the Adam forward-backward cost")
    ax.set_ylim(1e-8, 1e2)
    ax.grid(alpha=.3, which="both")
    ax.set_title(r"B2  overhead $\propto 1/L$, independent of width"
                 f"   ($h$={int(h)}, $b$={int(b)}, $T$={int(T)})",
                 fontsize=10.5, pad=54)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8, frameon=False)

    fig.text(0.5, 0.985, "B   Is this feasible at scale, and what breaks it?",
             ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, 0.945,
             "Left: three ways to carry information across batches at the same $O(d^2)$ state. QR is backward stable; the Gram squares $\\kappa$ ($10^{28}$ at $\\kappa=10^{14}$) and dies.\n"
             "Right: the cost model. The linear block's Jacobian is free (it is the cached activations), so the only real overhead is the QR update -- and it shrinks as the model gets deeper.",
             ha="center", va="top", fontsize=8.8, linespacing=1.5)
    fig.savefig(k.FIGS / "B_budget.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    out = k.RESULTS / "budget.jsonl"
    if out.exists():
        out.unlink()
    rows = data_b1()
    for q in rows:
        k.append(out, q)
    figure(rows)
    print("figure ->", k.FIGS / "B_budget.png")
