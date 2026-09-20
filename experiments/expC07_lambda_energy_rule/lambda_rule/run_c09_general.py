"""(ex-expC09, consolidated into expC07/lambda_rule on 2026-09-08; see hardened_rule.md.) The right-wall prediction survives
the skeptic review; the left-wall overlay (green line, truncation rule) is retained as an observation only.
Can the fiber-theorem aliasing floor predict the lambda curve for non-sinusoidal targets?

expC08 showed that for pure tones the measured right wall of the lambda U-curve lies on the exact
least-squares floor sqrt(Lambda_r(theta0; lambda)) with no fitting. Here the targets are rational
functions, polynomials, a chirp, a near-step, a C^2 function, etc. The prediction is the general
fiber projection (docs/lambda_rule_theory.md, Theorem 1): extend the target past the interval, project
onto the shift-invariant space of the kernel translates fiber by fiber in Fourier space, invert, and
score the error on [-1, 1] only. Two extensions (analytic continuation for `reach` cells, then a C-inf
taper) give an extension-sensitivity band.

Measured side: expC08's solver (frozen uniform grid, SVD lstsq readout), 3 activations x 10 targets x
4 widths x 60 lambdas, at halo 32 (the expC08 rule) and at a large fixed halo 256 so the small-lambda
side is not halo-limited. The LEFT wall is overlaid from the backward-stability bound with c = 1,
E_left = eps * ||A||_2 * ||x||_2 / ||b||_2, with ||x|| the fiber-theory coefficient norm (coefficients
f_hat(theta)/w_r(theta) per fiber, fibers below the SVD cutoff eps*sigma_max dropped) over the cells the
solve uses, and ||A||_2 the largest singular value the solver returns. Figures: per activation two 5x4
grids (targets x widths), a ratio plot measured/predicted on both walls, and a summary table.

Run:  uv run --extra dev python experiments/expC07_lambda_energy_rule/lambda_rule/run_c09_general.py   (--smoke, --fig-only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUT_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule"
DATA = OUT_DIR / "data"
LAM_GRID = [float(f"{l:.5g}") for l in np.geomspace(0.03, 1.5, 60)]
WIDTHS = [64, 128, 256, 512]
HALO = 32
HALO_BIG = 256             # the un-confounded measured curve for the left wall
EPS = 2.0 ** -52
ACTS = ["tanh", "gelu", "swish"]
KERNEL_ORDER = {"tanh": 1, "gelu": 2, "swish": 2}
# Extension for the prediction. The target is continued analytically for `reach` cells past each end of
# the interval, then tapered to zero with a C-inf window over TAPER cells. The taper creates near-Nyquist
# content whose projection error leaks back toward the interior at about lambda/2 per cell (measured), so
# the taper must sit hundreds of cells out. For targets that grow, the continuation is capped at X_MAX
# units so the box amplitude stays within fp64 dynamic range; the alternate (halved) reach exposes the
# cells where the prediction depends on the extension.
REACH_CELLS = 800          # main reach in cells (alternate = half)
X_MAX = {"exp": 4.0, "poly4": 3.0, "abs3": 6.0,   # cap on |x| for the analytic continuation of growing targets
         "chirp": 1.6}                            # chirp: its continuation crosses Nyquist a few units out
TAPER_CELLS = {"chirp": 60}                       # per-target taper width (default TAPER)
TAPER = 200                # taper width in cells
MARGIN = 20                # zero margin after the taper
N_SUB = 16                 # sub-samples per cell for the prediction FFT
REACHES = ["main", "half"]
PRED_LAM = list(LAM_GRID)      # the lambda grid the stored predictions live on (smoke mode shrinks it)

TARGETS = {
    "runge25":   ("$1/(1+25x^2)$",                 lambda x: 1.0 / (1.0 + 25.0 * x ** 2)),
    "runge100":  ("$1/(1+100x^2)$",                lambda x: 1.0 / (1.0 + 100.0 * x ** 2)),
    "rat_off":   ("$1/((x-0.5)^2+0.04)$",          lambda x: 1.0 / ((x - 0.5) ** 2 + 0.04)),
    "tanh10":    (r"$\tanh(10x)$",                 lambda x: np.tanh(10.0 * x)),
    "gauss20":   ("$e^{-20x^2}$",                  lambda x: np.exp(-20.0 * x ** 2)),
    "poly4":     ("$x^4-x^2$",                     lambda x: x ** 4 - x ** 2),
    "exp":       ("$e^{x}$",                       lambda x: np.exp(x)),
    "expsin":    (r"$e^{\sin 3\pi x}$",            lambda x: np.exp(np.sin(3 * np.pi * x))),
    "chirp":     (r"$\sin(10\pi x^2)$",            lambda x: np.sin(10 * np.pi * x ** 2)),
    "abs3":      ("$|x|^3$",                       lambda x: np.abs(x) ** 3),
}
GROUPS = {"A": ["runge25", "runge100", "rat_off", "tanh10", "gauss20"],
          "B": ["poly4", "exp", "expsin", "chirp", "abs3"]}


# ----------------------------------------------------------------------------- measured side
def apply_act(z, name):
    from scipy.special import erf, expit
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish":
        return z * expit(z)
    return np.tanh(z)


def geometry(N, lam, halo=HALO):
    halo = int(halo)
    h = 2.0 / N
    centers = -1.0 + np.arange(-halo, N + halo + 1, dtype=np.float64) * h
    return centers, lam / h


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_name, lam, N = job["act"], job["fn"], job["lam"], job["N"]
    halo = job.get("halo", HALO)
    f = TARGETS[fn_name][1]
    centers, gamma = geometry(N, lam, halo)
    x = np.linspace(-1, 1, max(2003, 2 * centers.size + 3))
    A = np.hstack([apply_act(gamma * (x[:, None] - centers[None, :]), act), np.ones((x.size, 1))])
    b = f(x)
    sol, _, rank, sv = sp_lstsq(A, b, lapack_driver="gelsd")
    xe = np.linspace(-1, 1, 4001)
    fe = f(xe)
    resid = apply_act(gamma * (xe[:, None] - centers[None, :]), act) @ sol[:-1] + sol[-1] - fe
    return {"act": act, "fn": fn_name, "lam": lam, "N": N, "halo": int(halo),
            "rel_l2": float(np.linalg.norm(resid) / np.linalg.norm(fe)), "linf": float(np.max(np.abs(resid))),
            "s_max": float(sv[0]), "rank": int(rank), "coef_norm": float(np.linalg.norm(sol[:-1])),
            "b_norm": float(np.linalg.norm(b))}


# ----------------------------------------------------------------------------- predicted side
def log_khat(act, xi):
    """log |Khat(xi)/Khat(0)| for xi > 0 (float64, overflow-safe)."""
    xi = np.asarray(xi, dtype=float)
    if act == "tanh":
        a = np.pi * xi / 2
        return np.log(a) - (a + np.log1p(-np.exp(-2 * a)) - np.log(2))
    if act == "gelu":
        return np.log1p(xi ** 2) - xi ** 2 / 2
    a = np.pi * xi
    return 2 * np.log(np.pi * xi) + (a + np.log1p(np.exp(-2 * a)) - np.log(2)) \
        - 2 * (a + np.log1p(-np.exp(-2 * a)) - np.log(2))


def taper_window(u, u0, width):
    """1 for |u| <= u0, C-inf decay to 0 over [u0, u0 + width], 0 beyond (symmetric in u)."""
    t = (np.abs(u) - u0) / width
    w = np.ones_like(u)
    mid = (t > 0) & (t < 1)
    g = lambda s: np.exp(-1.0 / np.clip(s, 1e-300, None)) * (s > 0)   # noqa: E731
    w[mid] = g(1 - t[mid]) / (g(1 - t[mid]) + g(t[mid]))
    w[t >= 1] = 0.0
    return w


def reach_cells(fn_name, N, which="main"):
    h = 2.0 / N
    rc = REACH_CELLS
    if fn_name in X_MAX:
        rc = min(rc, int((X_MAX[fn_name] - 1.0) / h))
    return rc if which == "main" else rc // 2


def fiber_prediction(act, fn_name, N, lam_grid, reach="main", halo=HALO):
    """Whole-line fiber projection of the extended target. Returns a dict with
    'err': relative L2 error on [-1,1] (the right wall / resolution floor) and
    'coef': 2-norm of the network coefficients a_k over the cells |u_k| <= N/2 + halo (the ones an interval
    solve with that halo uses), with fibers whose Gram singular value is below EPS x the largest dropped
    (the truncated-SVD rule). E_left = EPS * ||A|| * coef / ||b|| is the backward-stability line.

    Grid units u = x/h. Periodic box of L = N + 2*(reach + TAPER + MARGIN) cells, N_SUB samples per
    cell. Fibers are residues k mod L of the DFT index. For each fiber the optimal coefficient is
    c* = <F, W>/|W|^2 with W_m = w_r(theta + 2 pi m), w_r = lam^(r-1) Khat(theta/lam)/(i theta)^r; the
    m = 0 residual is computed without cancellation. Fiber theta = 0 (residue 0): the constant is
    exact (bias column), the harmonics 2 pi m are left as error.
    """
    r = KERNEL_ORDER[act]
    h = 2.0 / N
    f = TARGETS[fn_name][1]
    reach = reach_cells(fn_name, N, reach)
    taper = TAPER_CELLS.get(fn_name, TAPER)
    side = reach + taper + MARGIN
    L = N + 2 * side
    M = L * N_SUB
    u = -side + np.arange(M) / N_SUB               # cells, interval is u in [0, N]... shift below
    # put the interval at u in [-N/2, N/2] for symmetry of the window
    u = u - N / 2 + 0.0
    x = u * h
    w = taper_window(u, N / 2 + reach, taper)
    ft = f(x) * w
    F = np.fft.fft(ft) / M
    k = np.fft.fftfreq(M) * M                      # integer frequencies of the box
    theta = 2 * np.pi * k / L                      # grid-unit frequency
    res = np.mod(np.rint(k).astype(int), L)        # fiber residue
    m0 = (np.abs(theta) < 1e-12)
    interior = (u >= -N / 2) & (u <= N / 2)
    fnorm = np.sqrt(np.mean(np.abs(ft[interior]) ** 2))

    kcell = np.arange(L) - side - N / 2                 # lattice point k sits at u = kcell[k]
    keep_cells = np.abs(kcell) <= N / 2 + halo
    out, out_coef = [], []
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for lam in lam_grid:
            logw = (r - 1) * np.log(lam) + log_khat(act, np.abs(theta) / lam) - r * np.log(np.abs(theta))
            logw[m0] = -np.inf
            # normalize each fiber by its largest |w| (the projection is scale-free fiber by fiber); this
            # keeps every fiber's dominant term at 1 and avoids denormal sums at small lambda
            fibmax = np.full(L, -np.inf)
            np.maximum.at(fibmax, res, np.where(np.isfinite(logw), logw, -np.inf))
            fibmax = np.where(np.isfinite(fibmax), fibmax, 0.0)
            wmag = np.exp(logw - fibmax[res])
            # w_r(theta) = |w| / (i theta)^r phase; both signs of theta handled by the phase
            phase = (-1j * np.sign(theta)) ** r
            W = wmag * phase
            W[m0] = 0.0
            S = np.bincount(res, weights=np.abs(W) ** 2, minlength=L)
            fw = F * np.conj(W)
            G = np.bincount(res, weights=fw.real, minlength=L) + 1j * np.bincount(res, weights=fw.imag, minlength=L)
            c = np.where(S > 0, G / np.where(S > 0, S, 1), 0.0)
            # truncated-SVD rule: per-fiber Gram singular value sqrt(S_true) relative to the largest fiber
            logS_true = 2 * fibmax + np.log(np.where(S > 0, S, 1e-300))
            keep_fiber = (logS_true - logS_true.max()) >= 2 * np.log(EPS)
            c = np.where(keep_fiber, c, 0.0)
            E = F - c[res] * W
            # m = 0 term of each fiber (|theta| < pi): recompute without cancellation
            base = np.abs(theta) < np.pi
            S0 = np.abs(W) ** 2
            Sne = S[res] - S0
            Gne = G[res] - fw
            E_base = (F * Sne - W * Gne) / np.where(S[res] > 0, S[res], 1)
            E = np.where(base & (S[res] > 0) & keep_fiber[res], E_base, E)
            E[m0] = 0.0                                   # constant absorbed by the bias
            e = np.fft.ifft(E * M)
            out.append(float(np.sqrt(np.mean(np.abs(e[interior]) ** 2)) / fnorm))
            # coefficients: a_hat(res) = c(res) * L / wmax(res); a_k = IDFT_L(a_hat); constant fiber -> bias
            a_hat = np.where(keep_fiber, c * L * np.exp(-np.where(keep_fiber, fibmax, 0.0)), 0.0)
            a_hat[0] = 0.0
            a_k = np.fft.ifft(a_hat)
            out_coef.append(float(np.sqrt(np.sum(np.abs(a_k[keep_cells]) ** 2))))
    return {"err": np.array(out), "coef": np.array(out_coef)}


# ----------------------------------------------------------------------------- analysis
def cell_rows(rows, act, fn, N):
    return sorted([r for r in rows if r["act"] == act and r["fn"] == fn and r["N"] == N], key=lambda r: r["lam"])


def curve(rows, act, fn, N):
    pts = cell_rows(rows, act, fn, N)
    return np.array([r["lam"] for r in pts]), np.array([r["rel_l2"] for r in pts])


def left_line(rows, preds, act, fn, N, reach="main"):
    """E_left(lambda) = EPS * s_max(lambda) * coef_fiber(lambda) / ||b||  (c = 1), on the measured lambda grid."""
    pts = cell_rows(rows, act, fn, N)
    lam = np.array([r["lam"] for r in pts])
    smax = np.array([r["s_max"] for r in pts]); bnorm = np.array([r["b_norm"] for r in pts])
    coef = np.interp(np.log(lam), np.log(PRED_LAM), preds[f"{act}|{fn}|{N}|{reach}|coef"])
    return lam, EPS * smax * coef / bnorm


def on_wall(lam, rel, pred=None, lo_factor=30.0, hi=1e-3, side="right"):
    """Mask of lambda points on a wall: measured above lo_factor x the measured floor, below hi, and to the
    right (aliasing wall) or left (conditioning wall) of the argmin."""
    if not rel.size:
        return np.zeros(0, dtype=bool)
    floor = np.exp(np.median(np.log(rel[rel <= np.percentile(rel, 30)])))
    k = int(np.argmin(rel))
    where = (lam > lam[k]) if side == "right" else (lam < lam[k])
    return (rel > lo_factor * floor) & (rel < hi) & where


def make_figures(rows, preds, rows32=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for act in ACTS:
        for gname, fns in GROUPS.items():
            fig, axes = plt.subplots(5, 4, figsize=(15, 15), sharex=True, sharey=True)
            for i, fn in enumerate(fns):
                for j, N in enumerate(WIDTHS):
                    ax = axes[i, j]
                    lam, rel = curve(rows, act, fn, N)
                    p_main = preds[f"{act}|{fn}|{N}|{REACHES[0]}|err"]
                    p_alt = preds[f"{act}|{fn}|{N}|{REACHES[1]}|err"]
                    if rows32 is not None:
                        l32, r32 = curve(rows32, act, fn, N)
                        if l32.size:
                            ax.loglog(l32, r32, "-", color="0.75", lw=1.0)
                    ax.loglog(PRED_LAM, p_alt, ":", color="#f2a65a", lw=1.2)
                    ax.loglog(PRED_LAM, p_main, "--", color="#d1352b", lw=1.4)
                    if lam.size:
                        ll, el = left_line(rows, preds, act, fn, N)
                        ax.loglog(ll, el, "-.", color="#2a9d5c", lw=1.3)
                        ax.loglog(lam, rel, "-", color="tab:blue", lw=1.5)
                    ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1)
                    ax.grid(True, which="both", alpha=0.25)
                    if i == 0:
                        ax.set_title(f"N = {N}", fontsize=11)
                    if j == 0:
                        ax.set_ylabel(TARGETS[fn][0] + "\n" + r"eval rel $L_2$", fontsize=10)
                    if i == 4:
                        ax.set_xlabel(r"$\lambda=\gamma h$")
            handles = [Line2D([0], [0], color="tab:blue", lw=1.6, label=f"measured (halo {HALO_BIG}, lstsq)"),
                       Line2D([0], [0], color="0.75", lw=1.0, label=f"measured (halo {HALO})"),
                       Line2D([0], [0], color="#d1352b", lw=1.4, ls="--", label="right wall: fiber projection (reach R)"),
                       Line2D([0], [0], color="#f2a65a", lw=1.2, ls=":", label="same, reach R/2"),
                       Line2D([0], [0], color="#2a9d5c", lw=1.3, ls="-.", label=r"left wall: $\varepsilon\,\|A\|_2\|\hat x\|_2/\|b\|_2$, fiber coefficients, c = 1")]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=9)
            fig.suptitle(f"expC09 -- {act}, targets {gname}: measured $\\lambda$ curve vs the fiber-theorem prediction (no fitting)",
                         y=0.972, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            p = OUT_DIR / "figures" / f"c09_general_grid_{act}_{gname}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"  saved {p}")

    # ratio plot: measured / predicted on the wall, all cells, per activation
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    cmap = plt.get_cmap("tab10")
    stats = {}
    for row_i, side in enumerate(("right", "left")):
        for ax, act in zip(axes[row_i], ACTS):
            for ti, fn in enumerate(TARGETS):
                for N in WIDTHS:
                    lam, rel = curve(rows, act, fn, N)
                    if not lam.size:
                        continue
                    if side == "right":
                        p = np.interp(np.log(lam), np.log(PRED_LAM), preds[f"{act}|{fn}|{N}|{REACHES[0]}|err"])
                    else:
                        p = left_line(rows, preds, act, fn, N)[1]
                    mask = on_wall(lam, rel, side=side)
                    if mask.sum() == 0:
                        continue
                    ratio = rel[mask] / p[mask]
                    stats.setdefault(f"{act}|{side}", []).append({"fn": fn, "N": N, "n_pts": int(mask.sum()),
                                                                  "ratio_median": float(np.median(ratio)),
                                                                  "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())})
                    ax.semilogx(lam[mask], ratio, "o", color=cmap(ti % 10), ms=4 + 1.2 * WIDTHS.index(N), alpha=0.7, mec="none")
            ax.axhline(1.0, color="k", lw=1.0)
            ax.set_yscale("log"); ax.set_ylim(0.03, 30); ax.set_xlim(0.03, 1.5)
            ax.set_title(f"{act}: {side} wall", fontsize=12); ax.grid(alpha=0.3, which="both")
            if row_i == 1:
                ax.set_xlabel(r"$\lambda$")
    axes[0, 0].set_ylabel("measured / predicted, aliasing wall")
    axes[1, 0].set_ylabel("measured / predicted, conditioning wall (c = 1)")
    handles = [Line2D([0], [0], marker="o", color=cmap(ti % 10), lw=0, ms=6, label=TARGETS[fn][0]) for ti, fn in enumerate(TARGETS)]
    handles += [Line2D([0], [0], marker="o", color="gray", lw=0, ms=4 + 1.2 * k, label=f"N={N}") for k, N in enumerate(WIDTHS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False, fontsize=9)
    fig.suptitle(f"expC09 -- measured / predicted on both walls (points > 30x the cell floor and < 1e-3), measured at halo {HALO_BIG}, no fitting",
                 y=0.92, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = OUT_DIR / "figures" / "c09_wall_ratio.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")
    (DATA / "c09_wall_ratios.json").write_text(json.dumps(stats, indent=1))
    for key in stats:
        print(f"\n{key} wall: measured/predicted per cell (median [min, max] over wall points)")
        for s in stats[key]:
            print(f"  {s['fn']:9s} N={s['N']:3d} n={s['n_pts']:2d}  {s['ratio_median']:.2f} [{s['ratio_min']:.2f}, {s['ratio_max']:.2f}]")
    # diagnostic: fiber coefficient norm vs the solver's coefficient norm on the left wall
    print("\ncoefficient norm, solver / fiber formula, median over left-wall points per activation:")
    for act in ACTS:
        rat = []
        for fn in TARGETS:
            for N in WIDTHS:
                pts = cell_rows(rows, act, fn, N)
                if not pts:
                    continue
                lam = np.array([r["lam"] for r in pts]); rel = np.array([r["rel_l2"] for r in pts])
                cn = np.array([r["coef_norm"] for r in pts])
                cf = np.interp(np.log(lam), np.log(PRED_LAM), preds[f"{act}|{fn}|{N}|main|coef"])
                m = on_wall(lam, rel, side="left")
                rat += list(cn[m] / cf[m])
        if rat:
            print(f"  {act}: median {np.median(rat):.2f}, 10-90%: {np.percentile(rat, 10):.2f}..{np.percentile(rat, 90):.2f}, n={len(rat)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fig-only", action="store_true")
    ap.add_argument("--repredict", action="store_true", help="with --fig-only: recompute the predictions")
    ap.add_argument("--halo", type=int, default=HALO_BIG, help="halo for the measured sweep (rows file is per halo)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    lam_grid, widths, fns, acts = LAM_GRID, WIDTHS, list(TARGETS), ACTS
    if args.smoke:
        lam_grid, widths, fns, acts = [0.15, 0.25, 0.4], [64], ["runge25", "poly4", "exp"], ["tanh"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = DATA / f"c09_rows_halo{args.halo}.json"
    rows32_path = DATA / f"c09_rows_halo{HALO}.json"
    preds_path = DATA / "c09_preds.json"

    def predict_all():
        global PRED_LAM
        PRED_LAM = list(lam_grid)
        out = {}
        for a in acts:
            for f in fns:
                for n in widths:
                    for rc in REACHES:
                        d = fiber_prediction(a, f, n, lam_grid, reach=rc, halo=args.halo)
                        out[f"{a}|{f}|{n}|{rc}|err"] = d["err"]
                        out[f"{a}|{f}|{n}|{rc}|coef"] = d["coef"]
        preds_path.write_text(json.dumps({k: v.tolist() for k, v in out.items()}))
        return out

    if args.fig_only:
        rows = json.loads(rows_path.read_text())
        preds = predict_all() if args.repredict else {k: np.array(v) for k, v in json.loads(preds_path.read_text()).items()}
    else:
        jobs = sorted([{"act": a, "fn": f, "lam": l, "N": n, "halo": args.halo}
                       for a in acts for f in fns for l in lam_grid for n in widths], key=lambda j: -j["N"])
        print(f"expC09 | {len(jobs)} solves | acts={acts} | targets={fns} | N={widths} | halo {args.halo}")
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_cell, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 500 == 0 or i == len(jobs):
                    rows_path.write_text(json.dumps(rows))
                    print(f"  [{i}/{len(jobs)}] ({time.time() - t0:.0f}s)", flush=True)
        print("predictions ...")
        preds = predict_all()
        if args.smoke:
            for f in fns:
                lam, rel = curve(rows, "tanh", f, 64)
                print(f, "measured", rel, "pred", preds[f"tanh|{f}|64|main|err"], "left", left_line(rows, preds, "tanh", f, 64)[1])
            return
    rows32 = json.loads(rows32_path.read_text()) if (rows32_path.exists() and args.halo != HALO) else None
    make_figures(rows, preds, rows32)


if __name__ == "__main__":
    main()
