"""Audit: what does the linear-block solve ACTUALLY cost on real architectures?

Two things get checked, because both decide feasibility and I asserted both
without proof last time.

S1  SEPARABILITY.  A final layer W (o x h) has o*h parameters.  Does the
    least-squares problem cost (o*h)^2 state, or h^2?
    Claim: the Jacobian is I_o (x) Phi, so the normal equations block-diagonalise
    into o INDEPENDENT problems that share one design matrix Phi (b x h).
    State is h^2/2 (one triangular factor) + h*o (the transformed right-hand
    sides), NOT (o*h)^2/2.  Verified against a brute-force Kronecker solve.

S2  DIRECT vs ITERATIVE.  If a triangular factor R is being maintained anyway,
    an SVD of R (O(h^3), once) makes EVERY damping level closed-form at O(h^2).
    Compare that against the iterative mu-ladder for accuracy and work.

S3  Real architectures, small to large: what is h, what is o, what does the
    state actually come to in bytes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core12 as k


# ------------------------------------------------------------------- S1 ---
def s1_separability(h=60, o=7, n=400, seed=0):
    """Solve a multi-output linear layer two ways and compare."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, h)))[0]
    V = np.linalg.qr(rng.standard_normal((h, h)))[0]
    sig = np.logspace(0, -13, h)                      # ill-conditioned, like Phi
    Phi = (U * sig) @ V.T
    Wtrue = rng.standard_normal((o, h))
    Y = Phi @ Wtrue.T                                  # n x o

    # (a) shared factor, o right-hand sides
    t = time.time()
    M = np.linalg.qr(np.hstack([Phi, Y]), mode="r")[:h + o]
    R, Z = M[:h, :h], M[:h, h:]
    Ur, sr, Vr = np.linalg.svd(R, full_matrices=False)
    kp = sr > 1e-15 * sr[0]
    Wa = (Vr.T * np.where(kp, 1 / np.where(kp, sr, 1), 0)) @ (Ur.T @ Z)   # h x o
    ta = time.time() - t
    state_a = h * (h + 1) // 2 + h * o

    # (b) brute force: one big (n*o) x (o*h) Kronecker system
    t = time.time()
    Jbig = np.kron(np.eye(o), Phi)                     # (n*o) x (o*h)
    ybig = Y.T.reshape(-1)
    Wb = np.linalg.lstsq(Jbig, ybig, rcond=1e-15)[0].reshape(o, h)
    tb = time.time() - t
    state_b = (o * h) * (o * h + 1) // 2

    pa = np.linalg.norm(Phi @ Wa - Y) / np.linalg.norm(Y)
    pb = np.linalg.norm(Phi @ Wb.T - Y) / np.linalg.norm(Y)
    print(f"S1  separability check   h={h}, o={o}, layer params o*h={o*h}")
    print(f"    shared R + {o} RHS : resid {pa:.3e}   state {state_a:>9d} floats  {ta*1e3:6.1f} ms")
    print(f"    brute Kronecker    : resid {pb:.3e}   state {state_b:>9d} floats  {tb*1e3:6.1f} ms")
    print(f"    agreement |Wa-Wb|/|W| = "
          f"{np.linalg.norm(Wa.T - Wb)/np.linalg.norm(Wb):.2e}")
    print(f"    state ratio (brute/shared) = {state_b/state_a:.1f}x   "
          f"-> scales as o^2, which is the trap\n")


# ------------------------------------------------------------------- S2 ---
def s2_direct_vs_iterative():
    print("S2  direct (SVD of the maintained R) vs the iterative mu-ladder")
    print(f"    {'problem':<24}{'route':<26}{'work':>16}{'eval rel L2':>14}{'secs':>8}")
    for name, mk in (("QI-1D N=256", lambda: k.prob_qi1d(256)),
                     ("QI-2D N=2048", lambda: k.prob_qi2d(2048)),
                     ("random gapless d=2048", lambda: k.prob_twin(2048))):
        p = mk()
        A, y, d, n = p["A"], p["y"], p["d"], p["n"]
        # maintain R by streaming QR (batches of d/8), then SVD it
        t = time.time()
        rng = np.random.default_rng(0)
        order = rng.permutation(n)
        b = max(d // 8, 1)
        M = np.zeros((0, d + 1))
        for i in range(0, n, b):
            S = order[i:i + b]
            M = np.vstack([M, np.hstack([A[S], y[S][:, None]])])
            if M.shape[0] >= d + 1:
                M = np.linalg.qr(M, mode="r")[:d + 1]
        if M.shape[0] > d + 1:
            M = np.linalg.qr(M, mode="r")[:d + 1]
        t_qr = time.time() - t
        t = time.time()
        Ur, sr, Vr = np.linalg.svd(M[:, :d], full_matrices=False)
        t_svd = time.time() - t
        z = Ur.T @ M[:, d]
        kp = sr > 1e-15 * sr[0]
        x = (Vr.T * np.where(kp, 1 / np.where(kp, sr, 1), 0)) @ z
        print(f"    {name:<24}{'streaming QR + SVD(R)':<26}"
              f"{'O(nd^2)+O(d^3)':>16}{k.eval_err(p, x):>14.2e}{t_qr+t_svd:>8.2f}")
        # every damping level is then closed form
        mus = [(a * p["sigma1"]) ** 2 for a in (1e-2, 1e-5, 1e-8)]
        t = time.time()
        for mu in mus:
            xm = (Vr.T * (sr / (sr ** 2 + mu))) @ z
        t_lv = (time.time() - t) / len(mus)
        print(f"    {'':<24}{'  + each mu level':<26}{'O(d^2)':>16}"
              f"{'':>14}{t_lv:>8.4f}")
        del p


# ------------------------------------------------------------------- S3 ---
ARCH = [
    # name, h_in (input width of the final linear/conv), o (output dim), loss
    ("QI MLP N=1024 (ours)",            1844,        1, "MSE"),
    ("ResNet-50 fc",                    2048,     1000, "CE"),
    ("ViT-L/16 head",                   1024,     1000, "CE"),
    ("SD1.5 / SDXL UNet conv_out",   320 * 9,        4, "MSE"),
    ("wide UNet conv_out (C=1280)", 1280 * 9,        4, "MSE"),
    ("DiT-XL/2 final linear",           1152,       32, "MSE"),
    ("FLUX-scale DiT final linear",     3072,       64, "MSE"),
    ("GPT-3 175B LM head",             12288,    50257, "CE"),
    ("Llama-3 70B LM head",             8192,   128256, "CE"),
    ("hypothetical pixel decoder",      1024,   196608, "MSE"),
]


def s3_table():
    print("\nS3  real architectures. State is h^2/2 (triangular factor) + h*o "
          "(transformed RHS).")
    print(f"    {'model':<32}{'h_in':>8}{'o':>9}{'layer par':>12}"
          f"{'R state':>11}{'RHS state':>11}{'total GB':>10}{'SVD TF':>9}  loss")
    for nm, h, o, loss in ARCH:
        par = h * o
        sr = h * (h + 1) / 2
        srhs = h * o
        gb = (sr + srhs) * 4 / 1e9                      # fp32 bytes
        svd = 20 * h ** 3 / 1e12
        print(f"    {nm:<32}{h:>8}{o:>9}{par/1e6:>11.1f}M{sr/1e6:>10.1f}M"
              f"{srhs/1e6:>10.1f}M{gb:>10.3f}{svd:>9.2f}  {loss}")
    print("    (SVD TF = teraflops for one dense SVD of R; fp32 bytes assumed)")


if __name__ == "__main__":
    s1_separability()
    s2_direct_vs_iterative()
    s3_table()


# ---------------------------------------------------------------- figure ---
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.8))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.30, top=0.72, wspace=0.26)

    ax = axs[0]
    mse = [a for a in ARCH if a[3] == "MSE"]
    ce = [a for a in ARCH if a[3] == "CE"]
    for grp, mk, lab in ((mse, "o", "MSE loss (this method applies)"),
                         (ce, "x", "cross-entropy (it does NOT -- see text)")):
        h = np.array([a[1] for a in grp], float)
        o = np.array([a[2] for a in grp], float)
        ax.loglog(h, (h * (h + 1) / 2 + h * o) * 4 / 1e9, mk, ms=11, mew=2.2,
                  color="C2" if mk == "o" else "C3", label=lab, ls="none")
        ax.loglog(h, ((o * h) * (o * h + 1) / 2) * 4 / 1e9, mk, ms=9, mew=2,
                  color="C1" if mk == "o" else "C0", ls="none", alpha=.85,
                  label=("naive: treat the layer as ONE problem" if mk == "o" else None))
        for a in grp:
            ax.annotate(a[0].split("(")[0].strip()[:22], (a[1], (a[1]*(a[1]+1)/2 + a[1]*a[2])*4/1e9),
                        fontsize=6.5, rotation=32, xytext=(3, 4),
                        textcoords="offset points")
    ax.axhline(80, color="k", ls="--", lw=1.4)
    ax.text(900, 110, "one H100 (80 GB)", fontsize=8)
    ax.set_xlabel(r"$h_{in}$ : input width of the final layer")
    ax.set_ylabel("least-squares state (GB, fp32)")
    ax.grid(alpha=.3, which="both")
    ax.set_title("S1  separability is what makes it feasible", fontsize=10.5, pad=42)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8, frameon=False)

    ax = axs[1]
    names = ["QI-1D\nN=256", "QI-2D\nN=2048", "random\nd=2048"]
    direct = [0.08, 5.65, 4.15]
    iterat = [110.0, 190.0, 120.0]
    x = np.arange(3)
    ax.bar(x - 0.2, iterat, 0.4, color="C1", label=r"iterative $\mu$-ladder (~$10^4$ LSQR its)")
    ax.bar(x + 0.2, direct, 0.4, color="C2", label="streaming QR + one SVD of $R$")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("wall-clock to reach the fp64 floor (s)")
    for i, (a, b) in enumerate(zip(iterat, direct)):
        ax.text(i + 0.2, b * 1.35, f"{a/b:.0f}x\nfaster", ha="center", fontsize=7.5)
    ax.grid(alpha=.3, axis="y", which="both")
    ax.set_title("S2  both reach the floor; direct is far cheaper\n"
                 "and then every $\\mu$ is closed-form at $O(h^2)$",
                 fontsize=10.5, pad=42)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=8, frameon=False)

    fig.text(0.5, 0.985, "S   Does this actually scale, and should the solve be iterative at all?",
             ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, 0.945,
             "Left: a final layer $W$ is $o\\times h$, so it has $o\\cdot h$ parameters -- but the least-squares problem SEPARATES into $o$ problems sharing one design matrix $\\Phi$ ($b\\times h$).\n"
             "State is $h^2/2 + ho$, not $(oh)^2/2$. Orange shows what the naive framing would cost. Right: measured, on the problems from expD12.",
             ha="center", va="top", fontsize=8.8, linespacing=1.5)
    fig.savefig(k.FIGS / "S_scale_audit.png", dpi=140)
    plt.close(fig)
