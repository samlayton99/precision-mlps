"""Full three-stage accounting: Adam step | per-step GN update | terminal solve.

COST MODEL (flops per training step unless marked ONE-TIME)

  stage 1  Adam            F_train           measured/published training flops
                           memory 3*P*4 B    (fp32 weights + 2 Adam moments)

  stage 2  GN update on one block, per step
           residual        2*m*h*o
           QR fold-in      2*m*h^2           m rows folded into the triangular
                                             factor R this step (m is a knob;
                                             the accumulated window only has to
                                             reach ~4h rows, not m per step)
           cheap solve     2*T*h^2           T LSQR its against R (caps ~1e-6)
           memory          8*(h^2/2 + h*o) B

  stage 3  terminal solve, ONE TIME
           full-reorth LSQR against R   ~2*h^3     (measured 10x fewer flops
                                                    than SVD(R)'s ~20*h^3, both
                                                    reach the floor)
           memory  8*(h^2/2 + r*h) B

  distributed  TSQR tree-reduce of R: h^2/2 floats per rank, vs the gradient
               all-reduce's P floats.

WHAT h IS -- this is the whole ballgame:
  last linear layer        h = input width.  J = I_o (x) Phi separates, so one
                           R serves all o outputs.            FEASIBLE
  hidden layer, exact GN   J rows are G_i (x) phi_i^T (Khatri-Rao). Does NOT
                           separate, so h = o_l * h_l = the block's full
                           parameter count.                   INFEASIBLE
  whole network            h = P.                             INFEASIBLE
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

GB = 1e9
# name, P, F_train (flops/step), h (cols of the design matrix), o, rows/step,
# m (rows folded per step), loss, kind
SCEN = [
    ("QI MLP N=1024 (ours)",      3.7e3, 2.4e8,   1844,      1,  11064,  1844, "MSE", "last"),
    ("small regression MLP",      1.05e6, 6.5e9,   512,      1,   1024,   512, "MSE", "last"),
    ("ResNet-50 regression head", 2.56e7, 3.1e12, 2048,      1,    256,   256, "MSE", "last"),
    ("SD1.5 UNet conv_out",       8.6e8, 3.3e13,  2880,      4, 131072,  2880, "MSE", "last"),
    ("wide UNet conv_out C=1280", 2.6e9, 9.6e13, 11520,      4, 131072, 11520, "MSE", "last"),
    ("DiT-XL/2 final linear",     6.75e8, 2.3e13, 1152,     32,  16384,  1152, "MSE", "last"),
    ("FLUX-scale DiT final",      1.2e10, 1.4e14, 3072,     64,  32768,  3072, "MSE", "last"),
    ("7B transformer LM head",    7e9, 8.6e13,    4096,  32000,   2048,  4096, "CE",  "last"),
    ("7B hidden layer, exact GN", 7e9, 8.6e13, 4096*4096,     1,   2048,  4096, "MSE", "hidden"),
    ("7B whole net, no blocking", 7e9, 8.6e13,     7e9,      1,   2048,  4096, "MSE", "all"),
]


def rows():
    out = []
    for nm, P, F, h, o, nrow, m, loss, kind in SCEN:
        h = float(h)
        gn = 2 * m * h * h + 2 * m * h * o          # QR fold-in + residual
        cheap = 2 * 100 * h * h                      # T=100 LSQR its against R
        term = 2 * h ** 3
        mem_adam = 3 * P * 4
        mem_gn = 8 * (h * h / 2 + h * o)
        mem_term = 8 * (h * h / 2 + 0.6 * h * h)
        out.append(dict(name=nm, P=P, F=F, h=h, o=o, m=m, loss=loss, kind=kind,
                        gn=gn, cheap=cheap, term=term,
                        gn_frac=gn / F, cheap_frac=cheap / F,
                        term_frac_step=term / F, term_frac_run=term / (F * 1e5),
                        mem_adam=mem_adam, mem_gn=mem_gn, mem_term=mem_term,
                        comm_ratio=(h * h / 2) / P))
    return out


def table(rs):
    print(f"{'scenario':<28}{'h':>10}{'GN/step':>10}{'terminal':>11}"
          f"{'GN mem':>11}{'Adam mem':>11}{'comm':>9}  verdict")
    print(f"{'':<28}{'':>10}{'/Adam':>10}{'/1 step':>11}{'(GB)':>11}{'(GB)':>11}"
          f"{'/grad':>9}")
    for q in rs:
        if q["kind"] != "last":
            v = "IMPOSSIBLE"
        elif q["loss"] == "CE":
            v = "N/A (GGN, no separation)"
        elif q["gn_frac"] > 1.0:
            v = "GN dominates"
        elif q["gn_frac"] > 0.1:
            v = "heavy but usable"
        else:
            v = "FREE"
        print(f"{q['name']:<28}{q['h']:>10.3g}{q['gn_frac']:>10.1e}"
              f"{q['term_frac_step']:>11.1e}{q['mem_gn']/GB:>11.4f}"
              f"{q['mem_adam']/GB:>11.3f}{q['comm_ratio']:>9.1e}  {v}")


def figure(rs):
    ok = [q for q in rs if q["kind"] == "last" and q["loss"] == "MSE"]
    bad = [q for q in rs if q["kind"] != "last"]
    fig, axs = plt.subplots(1, 2, figsize=(15.4, 6.0))
    fig.subplots_adjust(left=0.30, right=0.965, bottom=0.12, top=0.70, wspace=0.55)

    ax = axs[0]
    y = np.arange(len(ok))
    ax.barh(y + 0.22, [q["gn_frac"] for q in ok], 0.36, color="C0",
            label="GN update, per step (QR fold-in)")
    ax.barh(y - 0.22, [q["term_frac_step"] for q in ok], 0.36, color="C2",
            label="terminal solve (ONE TIME, vs one step)")
    ax.axvline(1.0, color="k", lw=1.6)
    ax.text(1.25, len(ok) - 0.4, "= as costly\nas the Adam step", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([q["name"] for q in ok], fontsize=8.5)
    ax.set_xlabel("fraction of one Adam forward-backward step")
    ax.set_xlim(1e-7, 1e4)
    ax.grid(alpha=.3, axis="x", which="both")
    ax.set_title("A  compute, MSE + last-layer cases (the feasible set)",
                 fontsize=10.5, pad=38)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8, frameon=False)

    ax = axs[1]
    allq = ok + bad
    y = np.arange(len(allq))
    cols = ["C2"] * len(ok) + ["C3"] * len(bad)
    ax.barh(y, [q["mem_gn"] / GB for q in allq], 0.5, color=cols)
    ax.axvline(80, color="k", ls="--", lw=1.6)
    ax.text(110, 0.2, "one H100\n(80 GB)", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([q["name"] for q in allq], fontsize=8.5)
    ax.set_xlabel("least-squares state (GB)")
    ax.set_xlim(1e-6, 1e12)
    ax.grid(alpha=.3, axis="x", which="both")
    ax.set_title("B  memory: red = no separability, so $h$ = the whole block",
                 fontsize=10.5, pad=38)

    fig.text(0.5, 0.985,
             "A   Three-stage accounting: is the Gauss-Newton update the same order as Adam?",
             ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, 0.945,
             "Feasibility is decided by ONE thing: whether the block's Jacobian separates. For the last layer $J=I_o\\otimes\\Phi$, so $h$ is the layer's INPUT width and the state is $h^2/2$.\n"
             "For a hidden layer the Jacobian rows are $G_i\\otimes\\phi_i^\\top$ (Khatri-Rao) and do not separate, so $h$ becomes the block's full parameter count -- and everything explodes.",
             ha="center", va="top", fontsize=8.8, linespacing=1.5)
    fig.savefig(k.FIGS / "A_accounting.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rs = rows()
    table(rs)
    figure(rs)
    print("\nfigure ->", k.FIGS / "A_accounting.png")
