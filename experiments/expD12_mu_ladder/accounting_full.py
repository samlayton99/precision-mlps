"""STEP 2 -- the full accounting, true to the setup.

SETUP (step 3's jobs are assumed done and are NOT costed here)
  P            all parameters
  L subset P   the parameters step 3 has identified as LOCALLY linear,
               f(x) ~ h(x)v for v = theta_L.  d = |L|.  L may sit anywhere,
               need not be contiguous, and may CHANGE between steps.
  blocks       step 3 also certifies which coordinates of L are EXACTLY
               decoupled -- off-block-diagonal identically zero, so each block
               is its own subproblem and solving them independently is exact,
               one pass, no sweeps.  The canonical certified case is a
               multi-output layer, J = I_o (x) Phi: o blocks that all share the
               SAME design matrix Phi (b x h), so ONE factorisation serves all o.
  mu           damping = confidence in the local linearity.  mu -> inf gives a
               gradient step, mu -> 0 gives the exact solve.  Schedule is step 3.

PER STEP
  one fwd+bwd -> Adam updates P\\L ; damped GN updates L
  the Jacobian action is FREE for arbitrary v,u from the cached per-sample
  layer inputs phi and backprop deltas (verified to 1e-16):
      (Jv)_i    = sum_l delta_l^{(i)T} V_l phi^{(i)}_{l-1}      2*b*d flops
      (J^T u)_l = sum_i u_i delta_l^{(i)} phi^{(i)T}_{l-1}      2*b*d flops

AT THE END
  one terminal solve per certified block, to machine eps or the noise floor.

COST SUMMARY (derived below)
  stage 1  Adam        6*P*b flops/step        mem 12*P B (fp32 w + 2 moments)
  stage 2  GN step     4*T*b*d flops/step      mem 20*d B (Krylov) + 4*b*O_L B
                       => ratio 2*T*d/(3*P) of one Adam step, batch-independent
  stage 3  terminal    per certified block:
             shared-design (I_o (x) Phi):  2*n*h^2 + 20*h^3 + 2*h^2*o flops,
                                           mem 8*(h^2/2) + 4*h*o B
             independent block beta:       2*n*d_b^2 + 20*d_b^3 flops,
                                           mem 4*d_b^2 B
           governed by the LARGEST certified block, never by |L|.
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

GB, MB = 1e9, 1e6
T = 5           # LSMR iterations per GN step
S = 1e5         # training steps

# name, P, F_train(flops/step), n(rows/step), h(cols of one block),
# o(blocks sharing that design matrix), O_L(total output width in L), shared?
SCEN = [
    ("QI MLP (ours)",            3.7e3, 2.4e8,  1.1e4,  1844,      1,  1844, True),
    ("small regression MLP",     1.0e6, 6.5e9,  1024,   1.0e4,     1,  3000, False),
    ("SD1.5 UNet final conv",    8.6e8, 3.3e13, 1.31e5, 2880,      4,  2880, True),
    ("wide UNet C=1280",         2.6e9, 9.6e13, 1.31e5, 11520,     4, 11520, True),
    ("DiT-XL/2 final linear",    6.8e8, 2.3e13, 1.64e4, 1152,     32,  1152, True),
    ("FLUX-scale DiT final",     1.2e10, 1.4e14, 3.3e4, 3072,     64,  3072, True),
    ("pixel decoder (huge o)",   2.0e8, 5.0e12, 4096,   1024, 196608,  1024, True),
    ("7B, L=1% CERTIFIED 1e4",   7.0e9, 8.6e13, 2048,   1.0e4,  7000, 4.7e4, False),
    ("7B, L=1% NOT certified",   7.0e9, 8.6e13, 2048,   7.0e7,     1, 4.7e4, False),
]


def rows():
    out = []
    for nm, P, F, n, h, o, O_L, shared in SCEN:
        d = h * o
        run = S * F
        # ---- stage 2
        s2_flops = 4 * T * n * d
        s2_ratio = s2_flops / F
        s2_krylov = 20 * d                       # 5 vectors fp32
        # Retained deltas are EXTRA memory only for layers whose deltas would
        # otherwise be freed. For a final layer the design matrix IS the cached
        # input activation and delta is just the residual -- nothing extra.
        s2_delta = 0.0 if shared else 4 * n * O_L
        # ---- stage 3
        if shared:                               # ONE factorisation serves all o
            s3_flops = 2 * n * h ** 2 + 20 * h ** 3 + 2 * h ** 2 * o
            s3_mem = 8 * (h ** 2 / 2) + 4 * h * o
        else:                                    # o independent blocks of size h
            s3_flops = o * (2 * n * h ** 2 + 20 * h ** 3)
            s3_mem = 4 * h ** 2                  # sequential, largest block
        out.append(dict(name=nm, P=P, F=F, n=n, h=h, o=o, d=d, shared=shared,
                        s2_ratio=s2_ratio, s2_krylov=s2_krylov, s2_delta=s2_delta,
                        s3_flops=s3_flops, s3_frac=s3_flops / run, s3_mem=s3_mem,
                        adam_mem=12 * P))
    return out


def verdict(q):
    if q["s3_mem"] / GB > 80:
        return "S3 OOM"
    if q["s3_frac"] > 0.25:
        return "S3 too slow"
    if q["s2_ratio"] > 1.0:
        return "S2 heavy (tiny model)"
    return "FEASIBLE"


def table(rs):
    print("STAGE 2 -- per step\n")
    print(f"{'scenario':<28}{'|L|':>10}{'S2/Adam':>10}{'Krylov':>10}"
          f"{'deltas':>10}{'Adam mem':>11}")
    for q in rs:
        print(f"{q['name']:<28}{q['d']:>10.2e}{q['s2_ratio']:>10.1e}"
              f"{q['s2_krylov']/MB:>9.1f}M{q['s2_delta']/MB:>9.1f}M"
              f"{q['adam_mem']/GB:>10.2f}G")
    print("\nSTAGE 3 -- terminal, once\n")
    print(f"{'scenario':<28}{'blk h':>9}{'#blk':>9}{'shared?':>9}"
          f"{'flops':>11}{'% of run':>11}{'memory':>11}   verdict")
    for q in rs:
        print(f"{q['name']:<28}{q['h']:>9.2e}{q['o']:>9.2e}"
              f"{('yes' if q['shared'] else 'no'):>9}{q['s3_flops']:>11.1e}"
              f"{q['s3_frac']*100:>10.4f}%{q['s3_mem']/GB:>10.3f}G   {verdict(q)}")


def figure(rs):
    fig, axs = plt.subplots(1, 3, figsize=(19.4, 6.4))
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.30, top=0.66, wspace=0.34)

    names = [q["name"] for q in rs]
    y = np.arange(len(rs))

    ax = axs[0]
    ax.barh(y + 0.20, [q["s2_ratio"] for q in rs], 0.36, color="C0",
            label="stage 2, per step")
    ax.barh(y - 0.20, [max(q["s3_frac"], 1e-12) for q in rs], 0.36, color="C2",
            label="stage 3, once, as % of whole run")
    ax.axvline(1.0, color="k", lw=1.6)
    ax.text(1.6, len(rs) - 0.7, "= cost of\nthe Adam step", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(1e-9, 1e3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("fraction (stage 2: of one Adam step; stage 3: of the whole run)")
    ax.grid(alpha=.3, axis="x", which="both")
    ax.set_title("A  compute", fontsize=11, pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8.5, frameon=False)

    ax = axs[1]
    ax.barh(y + 0.20, [(q["s2_krylov"] + q["s2_delta"]) / GB for q in rs], 0.36,
            color="C0", label="stage 2 state (Krylov + retained deltas)")
    ax.barh(y - 0.20, [q["s3_mem"] / GB for q in rs], 0.36, color="C2",
            label="stage 3 state (largest certified block)")
    ax.plot([q["adam_mem"] / GB for q in rs], y, "kd", ms=7, ls="none",
            label="Adam's own state, $12P$ B")
    ax.axvline(80, color="k", ls="--", lw=1.6)
    ax.text(110, 0.1, "one H100", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(1e-6, 1e5)
    ax.set_yticks(y); ax.set_yticklabels([], fontsize=8.5)
    ax.set_xlabel("memory (GB)")
    ax.grid(alpha=.3, axis="x", which="both")
    ax.set_title("B  memory", fontsize=11, pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8.5, frameon=False)

    ax = axs[2]
    hh = np.logspace(2.5, 6, 300)
    ax.loglog(hh, 4 * hh ** 2 / GB, "-", color="C2", lw=2.6,
              label="stage 3 state vs LARGEST certified block $h$")
    ax.axhline(80, color="k", ls="--", lw=1.6)
    ax.text(4e2, 110, "one H100 (80 GB)", fontsize=8.5)
    ax.axhline(80 * 64, color="grey", ls=":", lw=1.4)
    ax.text(4e2, 7000, "64-GPU pod", fontsize=8, color="grey")
    ax.axvline(1.4e5, color="C3", ls=":", lw=2)
    ax.text(1.55e5, 1e-4, "$h\\approx1.4\\times10^5$\nsingle-GPU ceiling",
            fontsize=8.5, color="C3")
    for q in rs:
        ax.plot([q["h"]], [q["s3_mem"] / GB], "o", ms=7,
                color="C2" if q["s3_mem"] / GB < 80 else "C3")
    ax.set_xlabel("$h$ : size of the largest CERTIFIED-decoupled block")
    ax.set_ylabel("stage-3 state (GB)")
    ax.grid(alpha=.3, which="both")
    ax.set_title("C  the only thing that caps the method\n"
                 "$|L|$ itself is irrelevant -- only the biggest block matters",
                 fontsize=11, pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8.5, frameon=False)

    fig.text(0.5, 0.985,
             "F   Step 2, full accounting: Adam  |  per-step damped Gauss-Newton on $L$  |  one terminal solve per certified block",
             ha="center", va="top", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.94,
             "Stage 2 costs $2Td/(3P)$ of an Adam step and is batch-independent, because the Jacobian action is free from the cached forward/backward intermediates.\n"
             "Stage 3 is governed by the LARGEST exactly-decoupled block that step 3 certifies -- for a multi-output layer $J=I_o\\otimes\\Phi$, that is the layer's input width $h$, regardless of how many outputs there are.",
             ha="center", va="top", fontsize=8.9, linespacing=1.5)
    fig.savefig(k.FIGS / "F_full_accounting.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rs = rows()
    table(rs)
    figure(rs)
    print("\nfigure ->", k.FIGS / "F_full_accounting.png")
