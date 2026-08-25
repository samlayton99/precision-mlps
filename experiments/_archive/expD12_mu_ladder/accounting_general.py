"""Three-stage accounting for an ARBITRARY locally-linear subset L, |L| = d.

NO architectural assumption. L is whatever step 3 identifies as locally linear;
it may sit anywhere in the network, need not be contiguous, and may change
between steps. Step 3 is also assumed to supply the block-structure oracle that
handles the multi-output case.

THE COST UNIT (verified to 1e-16 against autograd, see jac_free test)
  One forward+backward already caches, per sample, the layer inputs phi and the
  backprop deltas.  Together they give the Jacobian action for ANY vector:
        (Jv)_i     = sum_l  delta_l^{(i)T} V_l phi_{l-1}^{(i)}      2*b*d flops
        (J^T u)_l  = sum_i  u_i delta_l^{(i)} phi_{l-1}^{(i)T}      2*b*d flops
  No network traversal.  One LSMR iteration = 4*b*d flops.
  The only new memory is RETAINING the per-sample deltas instead of summing
  them straight into the parameter gradient.

STAGE 1  Adam            6*P*b flops/step,   memory 3*P*4 B
STAGE 2  damped GN on L   4*T*b*d flops/step  ->  ratio 2*T*d/(3*P)
                          memory 5*d*4 B (Krylov vectors) + retained deltas.
                          NO factor and NO preconditioner: both would be O(d^2)
                          and both are invalidated whenever L changes.
STAGE 3  terminal solve   full reorthogonalisation is required to reach the
                          floor (plain LSQR caps at ~1e-6, measured).
                          compute 2.4*b*d^2 + 1.44*d^3 flops
                          memory 4.8*d^2 B  (one-sided reorth, fp64)
         Blocking L into pieces of size D reduces memory to 4.8*D^2 and compute
         to (d/D)*1.44*D^3 -- i.e. LINEAR in d instead of cubic. This is the
         single most valuable thing step 3 can provide.

Damping law (measured, expD12, 7 matrices):
  T ~ alpha^-0.4 .. alpha^-0.87, accuracy reached ~ alpha, alpha = 1/kappa_mu.
  alpha=1e-1 -> T=2..8 ; 1e-2 -> T=8..64 ; 1e-3 -> T=17..206.
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
T_STEP = 5

#  name, P, b, |L|, S, blocked-piece size (None = unblocked)
SCEN = [
    ("QI MLP (ours)",            3.7e3, 1.1e4, 1.8e3, 1e4, None),
    ("small regression MLP",     1.0e6, 1024,  1.0e4, 1e5, None),
    ("ResNet-50, L=1%",          2.6e7, 256,   2.6e5, 1e5, None),
    ("SD1.5 UNet, L=1%",         8.6e8, 1.3e5, 8.6e6, 1e5, None),
    ("SD1.5 UNet, L=1% blocked", 8.6e8, 1.3e5, 8.6e6, 1e5, 1e4),
    ("7B transformer, L=0.1%",   7.0e9, 2048,  7.0e6, 1e5, None),
    ("7B transformer, L=1%",     7.0e9, 2048,  7.0e7, 1e5, None),
    ("7B transformer, L=1% blk", 7.0e9, 2048,  7.0e7, 1e5, 1e4),
    ("7B transformer, L=P",      7.0e9, 2048,  7.0e9, 1e5, None),
    ("7B, L=P, blocked",         7.0e9, 2048,  7.0e9, 1e5, 1e4),
]


def rows():
    out = []
    for nm, P, b, d, S, D in SCEN:
        run = S * 6 * P * b
        s2 = 4 * T_STEP * b * d
        D_eff = d if D is None else D
        nblk = 1 if D is None else d / D
        s3_flops = nblk * (2.4 * b * D_eff ** 2 + 1.44 * D_eff ** 3)
        s3_mem = 4.8 * D_eff ** 2
        out.append(dict(name=nm, P=P, b=b, d=d, S=S, D=D_eff, nblk=nblk,
                        s2_ratio=s2 / (6 * P * b), s2_mem=5 * d * 4,
                        adam_mem=3 * P * 4,
                        s3_flops=s3_flops, s3_frac=s3_flops / run,
                        s3_mem=s3_mem))
    return out


def verdict(q):
    bad_m = q["s3_mem"] / GB > 80
    bad_c = q["s3_frac"] > 0.25
    if q["s2_ratio"] > 1.0 and (bad_m or bad_c):
        return "DEAD"
    if bad_m and bad_c:
        return "stage 3 DEAD"
    if bad_m:
        return f"stage 3 OOM ({q['s3_mem']/GB:.0f} GB) -> BLOCK IT"
    if bad_c:
        return "stage 3 too slow -> BLOCK IT"
    if q["s2_ratio"] > 1.0:
        return "stage 2 heavy (small model)"
    return "FEASIBLE"


def table(rs):
    print(f"{'scenario':<28}{'|L|':>9}{'|L|/P':>8}{'S2 /Adam':>10}"
          f"{'S2 mem':>9}{'S3 mem':>11}{'S3 % run':>11}   verdict")
    for q in rs:
        print(f"{q['name']:<28}{q['d']:>9.1e}{q['d']/q['P']:>8.3f}"
              f"{q['s2_ratio']:>10.3f}{q['s2_mem']/GB:>9.3f}"
              f"{q['s3_mem']/GB:>11.1f}{q['s3_frac']*100:>10.3f}%   {verdict(q)}")


def figure(rs):
    fig, axs = plt.subplots(1, 3, figsize=(19.2, 6.2))
    fig.subplots_adjust(left=0.075, right=0.955, bottom=0.14, top=0.65, wspace=0.30)

    ax = axs[0]
    d = np.logspace(3, 9, 400)
    for D, c, ls, lab in ((None, "C3", "-", "unblocked"),
                          (1e5, "C1", "--", "blocked at $10^5$"),
                          (1e4, "C2", "-.", "blocked at $10^4$")):
        De = d if D is None else np.full_like(d, D)
        ax.loglog(d, 4.8 * De ** 2 / GB, ls, color=c, lw=2.5, label=lab)
    ax.axhline(80, color="k", ls=":", lw=1.8)
    ax.text(1.3e3, 105, "one H100 (80 GB)", fontsize=8.5)
    ax.axhline(80 * 64, color="grey", ls=":", lw=1.4)
    ax.text(1.3e3, 6600, "64-GPU pod", fontsize=8, color="grey")
    ax.set_xlabel("$|L|$ : size of the locally-linear subset")
    ax.set_ylabel("stage-3 state (GB)")
    ax.grid(alpha=.3, which="both")
    ax.set_title("A  stage 3 memory is $O(|L|^2)$ unblocked, $O(D^2)$ blocked\n"
                 "-- but see panel C for whether blocking is legal",
                 fontsize=10.5, pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=8.5, frameon=False)

    ax = axs[1]
    frac = np.logspace(-4, 0, 300)
    for T, c in ((1, "C0"), (5, "C2"), (20, "C1"), (100, "C3")):
        ax.loglog(frac, 2 * T * frac / 3, "-", color=c, lw=2.3,
                  label=f"$T$={T} LSMR iterations/step")
    ax.axhline(1.0, color="k", lw=1.6)
    ax.text(1.3e-4, 1.35, "= doubles the cost of training", fontsize=8.5)
    ax.axhline(0.1, color="grey", ls=":", lw=1.5)
    ax.text(1.3e-4, 0.12, "10% overhead", fontsize=8, color="grey")
    for q in rs:
        if q["D"] == q["d"]:
            ax.axvline(q["d"] / q["P"], color="grey", lw=0.6, alpha=.5)
    ax.set_xlabel("$|L| / P$ : fraction of parameters treated as locally linear")
    ax.set_ylabel("stage-2 cost, as a multiple of the Adam step")
    ax.set_ylim(1e-5, 1e2)
    ax.grid(alpha=.3, which="both")
    ax.set_title("B  stage 2 is cheap because $J_L$ is free\n"
                 r"cost $= \frac{2Td}{3P}$, independent of batch size",
                 fontsize=10.5, pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=8.5, frameon=False)

    # ---- panel C: measured block-coordinate-descent stall
    ax = axs[2]
    BCD = {"QI-1D N=256": {64: [1.42e-1, 9.88e-3, 3.85e-3], 128: [1.46e-1, 4.85e-3, 1.81e-3],
                           0: [1.27e-4, 2.77e-15, 2.77e-15]},
           "random d=512": {64: [1.39e-3, 5.90e-4, 3.05e-4], 128: [4.97e-6, 1.29e-6, 7.13e-7],
                            0: [1.50e-4, 1.11e-15, 1.13e-15]}}
    sw = [1, 10, 60]
    for i, (nm, dd) in enumerate(BCD.items()):
        for D, c, ls in ((64, "C3", "-"), (128, "C1", "--"), (0, "C2", "-.")):
            lab = None
            if i == 0:
                lab = "full solve (1 block)" if D == 0 else f"blocks of {D}"
            ax.loglog(sw, dd[D], ls, color=c, lw=2.2, marker="o" if i == 0 else "s",
                      ms=5, alpha=1.0 if i == 0 else .55, label=lab)
    ax.axhspan(1e-16, 1e-14, color="C2", alpha=.15)
    ax.text(1.2, 3e-15, "fp64 floor", fontsize=8.5, color="C2")
    ax.set_xlabel("block-coordinate sweeps (exact solve per block)")
    ax.set_ylabel("eval rel $L_2$")
    ax.set_ylim(1e-16, 1e0)
    ax.grid(alpha=.3, which="both")
    ax.set_title("C  MEASURED: arbitrary blocking does NOT\nreach the floor -- it stalls at $10^{-3}..10^{-6}$",
                 fontsize=10.5, pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=8.5, frameon=False)

    fig.text(0.5, 0.985,
             "G   Accounting for a general locally-linear subset $L$ — no architectural assumption",
             ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, 0.945,
             "One cached forward+backward yields $Jv$ and $J^\\top u$ for ANY $v,u$ at $2bd$ flops each (verified to $10^{-16}$), so the per-step solve costs $2Td/(3P)$ of an Adam step.\n"
             "The design then hinges on stage 3, which needs stored orthogonality: $O(|L|^2)$ memory. Panel A shows blocking would fix that -- but panel C shows blocking only works if the blocks are GENUINELY decoupled.",
             ha="center", va="top", fontsize=8.8, linespacing=1.5)
    fig.savefig(k.FIGS / "G_general_accounting.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rs = rows()
    table(rs)
    figure(rs)
    print("\nfigure ->", k.FIGS / "G_general_accounting.png")
