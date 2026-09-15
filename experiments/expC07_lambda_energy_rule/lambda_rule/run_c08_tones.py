"""(ex-expC08, consolidated into expC07/lambda_rule on 2026-09-08; see hardened_rule.md for what survives.)
Does the frequency-resolved aliasing rule predict the lambda wall per (target, width)?

Riff on expC07. expC07's rule is width- and frequency-independent: |Khat(2 pi/lambda)|/|Khat(0)| = eps*
(tanh 0.25, gelu 0.707, swish 0.455). The NEW rule (Sam, 2026-09-08) budgets fp64 eps = 2^-52 against
the two first grid ghosts of the target's own frequency theta0 = h*Omega (h = 2/(N-1)):

    2 [ Khat((2pi - theta0)/lambda) + Khat((2pi + theta0)/lambda) ] / Khat(theta0/lambda) = eps / B,

so the certified lambda shrinks as the target frequency approaches Nyquist (q = theta0/pi -> 1) and tends
to tanh 0.2358 / gelu 0.6866 / swish 0.4313 as q -> 0. `predictions.py` reproduces Sam's tables from it.

Sweep: 3 activations x 6 targets x 6 widths (16..512) x 60 lambdas in geomspace(0.03, 1.5) x a halo
ladder {0, 4, 8, 16, 32, 64, default = max(70, 0.4N)} centers per side; frozen uniform grid (h = 2/N),
SVD lstsq readout on a dense grid, eval rel L2 and L_inf on 4001 points. The halo ladder is there because
expD19/expF17 found the default rule oversized (halo 8 beats halo 59 for tanh at lambda 0.25) but not
uniformly sufficient; each lambda curve is reported as the best over the ladder, and the per-halo curves
are plotted separately. Figures: one 6x6 grid per activation (rows = targets, cols = widths), measured
rel L2 vs lambda (best over halo; default-halo curve in light gray), red dashed = new rule, black dotted
= expC07 rule, dot = measured minimum; one 6x6 halo-sensitivity grid per activation; one wall-vs-q summary.

Run:  uv run --extra dev python experiments/expC07_lambda_energy_rule/lambda_rule/run_c08_tones.py   (--smoke, --fig-only)
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expA06_readout_structure"))
from app import geometry  # noqa: E402  (uniform grid, halo sized at lambda 0.25, h = 2/N)
from predictions_c08 import PREV, TABLE, TARGETS as TARGET_SPEC, WIDTHS  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule"
DATA = OUT_DIR / "data"
LAM_GRID = [float(f"{l:.5g}") for l in np.geomspace(0.03, 1.5, 60)]
HALOS = [0, 4, 8, 16, 32, 64, "default"]
ACTS = ["tanh", "gelu", "swish"]
TARGET_ORDER = list(TARGET_SPEC)
TARGET_TEX = {
    "sin_1pi": r"$\sin(\pi x)$", "cos_4pi": r"$\cos(4\pi x)$",
    "f8": r"$f_8=\sin(3\pi x)\cos(5\pi x)$", "sin_16pi": r"$\sin(16\pi x)$",
    "f32": r"$f_{32}=0.8\sin(2\pi x)+0.2\cos(32\pi x)$", "sin_100pi": r"$\sin(100\pi x)$",
}


def target_fn(name):
    return {
        "sin_1pi": lambda x: np.sin(np.pi * x),
        "cos_4pi": lambda x: np.cos(4 * np.pi * x),
        "f8": lambda x: np.sin(3 * np.pi * x) * np.cos(5 * np.pi * x),
        "sin_16pi": lambda x: np.sin(16 * np.pi * x),
        "f32": lambda x: 0.8 * np.sin(2 * np.pi * x) + 0.2 * np.cos(32 * np.pi * x),
        "sin_100pi": lambda x: np.sin(100 * np.pi * x),
    }[name]


def apply_act(z, name):
    from scipy.special import erf, expit
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish":
        return z * expit(z)
    return np.tanh(z)


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_name, lam, N = job["act"], job["fn"], job["lam"], job["N"]
    halo = job.get("halo", "default")
    f = target_fn(fn_name)
    centers, gamma, halo_used = geometry(N, lam, halo=None if halo == "default" else halo)
    x = np.linspace(-1, 1, max(2003, 2 * centers.size + 3))
    Phi = apply_act(gamma * (x[:, None] - centers[None, :]), act)
    A = np.hstack([Phi, np.ones((x.size, 1))])
    sol, *_ = sp_lstsq(A, f(x), lapack_driver="gelsd")
    xe = np.linspace(-1, 1, 4001)
    Phi_e = apply_act(gamma * (xe[:, None] - centers[None, :]), act)
    fe = f(xe)
    resid = Phi_e @ sol[:-1] + sol[-1] - fe
    return {"act": act, "fn": fn_name, "lam": lam, "N": N, "halo": halo, "halo_n": int(halo_used),
            "rel_l2": float(np.linalg.norm(resid) / np.linalg.norm(fe)),
            "linf": float(np.max(np.abs(resid)))}


# ---- exact fiber-theorem aliasing floor (docs/lambda_rule_theory.md, Theorem 2 / Cor. 2.2) ----
KERNEL_ORDER = {"tanh": 1, "gelu": 2, "swish": 2}
TARGET_MODES = {  # (omega/pi, amplitude) pairs; rel L2 floor^2 = sum a^2 Lambda_r(theta) / sum a^2
    "sin_1pi": [(1, 1.0)], "cos_4pi": [(4, 1.0)], "f8": [(8, 0.5), (2, 0.5)],
    "sin_16pi": [(16, 1.0)], "f32": [(2, 0.8), (32, 0.2)], "sin_100pi": [(100, 1.0)],
}


def log_khat(act, xi):
    """log |Khat(xi)/Khat(0)| in float64 without overflow (xi > 0 array)."""
    xi = np.abs(np.asarray(xi, dtype=float))
    if act == "tanh":
        a = np.pi * xi / 2
        return np.log(a) - (a + np.log1p(-np.exp(-2 * a)) - np.log(2))   # log(a/sinh a)
    if act == "gelu":
        return np.log1p(xi ** 2) - xi ** 2 / 2
    a = np.pi * xi                                                          # swish
    return 2 * np.log(np.pi * xi) + (a + np.log1p(np.exp(-2 * a)) - np.log(2)) \
        - 2 * (a + np.log1p(-np.exp(-2 * a)) - np.log(2))                  # log(pi^2 xi^2 cosh a / sinh^2 a)


def fiber_floor(act, fn, N, lam_grid, m_max=8):
    """sqrt( sum_k a_k^2 Lambda_r(theta_k; lambda) / sum_k a_k^2 ), theta_k = omega_k h, h = 2/N."""
    r = KERNEL_ORDER[act]
    h = 2.0 / N
    out = np.zeros(len(lam_grid))
    wsum = 0.0
    for w_over_pi, amp in TARGET_MODES[fn]:
        th = w_over_pi * np.pi * h
        if th >= np.pi:          # beyond Nyquist: no fiber floor (treat as unresolvable)
            out += amp ** 2 * np.ones(len(lam_grid))
            wsum += amp ** 2
            continue
        ms = np.arange(-m_max, m_max + 1)
        tt = th + 2 * np.pi * ms
        for i, lam in enumerate(lam_grid):
            lt = 2 * log_khat(act, tt / lam) - 2 * r * np.log(np.abs(tt))
            lt -= lt.max()
            v = np.exp(lt)
            out[i] += amp ** 2 * (v[ms != 0].sum() / v.sum())
        wsum += amp ** 2
    return np.sqrt(out / wsum)


def chosen_halo(N):
    """The single halo per width chosen by halo_rule.py (falls back to 'default' if not yet chosen)."""
    p = DATA / "c08_halo_selection.json"
    if not p.exists():
        return "default"
    return json.loads(p.read_text())[str(N)]["halo"]


def curve(rows, act, fn, N, halo="fixed"):
    """rel L2 vs lambda for one cell. halo='fixed' uses the per-width halo from halo_rule.json;
    halo='best' takes the minimum over the ladder at each lambda (diagnostic only, confounds lambda)."""
    if halo == "fixed":
        halo = chosen_halo(N)
    pts = [r for r in rows if r["act"] == act and r["fn"] == fn and r["N"] == N
           and (halo == "best" or r.get("halo", "default") == halo)]
    by_lam = {}
    for r in pts:
        by_lam[r["lam"]] = min(by_lam.get(r["lam"], np.inf), r["rel_l2"])
    lam = np.array(sorted(by_lam))
    return lam, np.array([by_lam[l] for l in lam])


def best_halo(rows, act, fn, N):
    """Per lambda, the halo entry that attains the minimum (ties -> smallest halo)."""
    pts = [r for r in rows if r["act"] == act and r["fn"] == fn and r["N"] == N]
    out = {}
    for r in pts:
        key = (r["rel_l2"], r["halo_n"])
        if r["lam"] not in out or key < out[r["lam"]]:
            out[r["lam"]] = key
    lam = np.array(sorted(out))
    return lam, np.array([out[l][1] for l in lam])


def smooth_log(v, k=5):
    """Running median of log(v) over k lambda points; floor jitter spans two decades, the wall does not."""
    lv = np.log(np.asarray(v, dtype=float))
    r = k // 2
    return np.exp(np.array([np.median(lv[max(0, i - r): i + r + 1]) for i in range(len(lv))]))


def right_shoulder(lam, rel, factor=10.0):
    """The wall: largest lambda at which the SMOOTHED error is within `factor` of the smoothed minimum."""
    if lam.size == 0:
        return np.nan
    sm = smooth_log(rel)
    ok = np.where(sm <= factor * sm.min())[0]
    return float(lam[ok.max()])


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for act in ACTS:
        fig, axes = plt.subplots(6, 6, figsize=(19, 17), sharex=True, sharey=True)
        for i, fn in enumerate(TARGET_ORDER):
            for j, N in enumerate(WIDTHS):
                ax = axes[i, j]
                lam_d, rel_d = curve(rows, act, fn, N, halo="default")
                if lam_d.size:
                    ax.loglog(lam_d, rel_d, "-", color="0.75", lw=1.0)
                ff = fiber_floor(act, fn, N, LAM_GRID)
                ax.loglog(LAM_GRID, ff, "-.", color="0.45", lw=0.9)
                lam, rel = curve(rows, act, fn, N)
                if lam.size:
                    ax.loglog(lam, rel, "-", color="tab:blue", lw=1.5)
                    k = int(np.argmin(rel))
                    ax.plot(lam[k], rel[k], "o", color="tab:blue", ms=6, mec="k", mew=0.5, zorder=5)
                new = TABLE[act][fn][j]
                ax.axvline(PREV[act], color="k", lw=1.3, ls=":")
                if new is not None:
                    ax.axvline(new, color="#d1352b", lw=1.6, ls="--")
                    tag = f"new {new:.3f} | prev {PREV[act]:.3f}"
                else:
                    tag = f"new: none (q>=1) | prev {PREV[act]:.3f}"
                ax.text(0.03, 0.04, tag, transform=ax.transAxes, fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
                ax.set_xlim(0.03, 1.5)
                ax.set_ylim(1e-16, 1e1)
                ax.grid(True, which="both", alpha=0.25)
                if i == 0:
                    hn = [r["halo_n"] for r in rows if r["N"] == N and r["halo"] == chosen_halo(N)][:1]
                    ax.set_title(f"N = {N}   (halo {hn[0] if hn else '?'})", fontsize=11)
                if j == 0:
                    ax.set_ylabel(TARGET_TEX[fn] + "\n" + r"eval rel $L_2$", fontsize=9)
                if i == 5:
                    ax.set_xlabel(r"$\lambda = \gamma h$")
        handles = [Line2D([0], [0], color="tab:blue", lw=1.6, label="measured at the per-width halo rule (frozen grid, lstsq readout)"),
                   Line2D([0], [0], color="0.75", lw=1.0, label="measured, default halo max(70, 0.4N)"),
                   Line2D([0], [0], color="0.45", lw=0.9, ls="-.", label="exact aliasing floor (fiber theorem)"),
                   Line2D([0], [0], marker="o", color="tab:blue", lw=0, ms=6, mec="k", label="measured minimum"),
                   Line2D([0], [0], color="#d1352b", lw=1.6, ls="--", label="new rule (frequency-resolved, fp64 eps)"),
                   Line2D([0], [0], color="k", lw=1.3, ls=":", label=f"previous rule (expC07, {PREV[act]:.3f})")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6, frameon=False, fontsize=10)
        fig.suptitle(f"expC08 -- {act}: rel $L_2$ vs $\\lambda$ per (target, width); new rule in red, expC07 rule in black",
                     y=0.975, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = OUT_DIR / "figures" / f"c08_tones_grid_{act}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  saved {p}")


def make_halo_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("viridis")
    colors = {h: cmap(i / (len(HALOS) - 1)) for i, h in enumerate(HALOS)}
    for act in ACTS:
        fig, axes = plt.subplots(6, 6, figsize=(19, 17), sharex=True, sharey=True)
        for i, fn in enumerate(TARGET_ORDER):
            for j, N in enumerate(WIDTHS):
                ax = axes[i, j]
                for h in HALOS:
                    lam, rel = curve(rows, act, fn, N, halo=h)
                    if lam.size:
                        ax.loglog(lam, rel, "-", color=colors[h], lw=1.2)
                new = TABLE[act][fn][j]
                ax.axvline(PREV[act], color="k", lw=1.0, ls=":")
                if new is not None:
                    ax.axvline(new, color="#d1352b", lw=1.2, ls="--")
                ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1)
                ax.grid(True, which="both", alpha=0.25)
                if i == 0:
                    ax.set_title(f"N = {N}   (default halo {max(70, int(0.4 * N))})", fontsize=10)
                if j == 0:
                    ax.set_ylabel(TARGET_TEX[fn] + "\n" + r"eval rel $L_2$", fontsize=9)
                if i == 5:
                    ax.set_xlabel(r"$\lambda = \gamma h$")
        handles = [Line2D([0], [0], color=colors[h], lw=1.6, label=f"halo {h}") for h in HALOS]
        handles += [Line2D([0], [0], color="#d1352b", lw=1.2, ls="--", label="new rule"),
                    Line2D([0], [0], color="k", lw=1.0, ls=":", label="expC07 rule")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=9, frameon=False, fontsize=10)
        fig.suptitle(f"expC08 -- {act}: halo sensitivity of the $\\lambda$ curve (halo = centers per side beyond $[-1,1]$)",
                     y=0.975, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = OUT_DIR / "figures" / f"c08_halo_ladder_{act}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  saved {p}")


def make_wall_figure(rows):
    """Measured wall and argmin vs q = theta0/pi (frequency per grid cell, repo h = 2/N), per activation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from predictions_c08 import solve_kernel_ratio
    import mpmath as mp
    mp.mp.dps = 30

    markers = {"sin_1pi": "o", "cos_4pi": "s", "f8": "^", "sin_16pi": "D", "f32": "v", "sin_100pi": "P"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    qq = np.geomspace(2e-3, 0.98, 120)
    for ax, act in zip(axes, ACTS):
        # continuous new-rule curve in q (theta0 = q*pi), h-convention-free
        lam_new = []
        for q in qq:
            # solve_kernel_ratio takes (N, omega/pi) with h = 2/(N-1); emulate q directly via a large N trick
            Nfake = 10 ** 6
            lam_new.append(solve_kernel_ratio(act, Nfake, q * (Nfake - 1) / 2, variant="sum4"))
        ax.plot(qq, lam_new, "-", color="#d1352b", lw=1.8)
        ax.axhline(PREV[act], color="k", lw=1.3, ls=":")
        for fn in TARGET_ORDER:
            w = TARGET_SPEC[fn][0]
            for N in WIDTHS:
                q = (2.0 / N) * w
                if q >= 1:
                    continue
                lam, rel = curve(rows, act, fn, N)
                if not lam.size or rel.min() > 1e-6:
                    continue
                ax.plot(q, right_shoulder(lam, rel), markers[fn], color="tab:blue", ms=7, mfc="none", mew=1.4)
                ax.plot(q, lam[int(np.argmin(rel))], markers[fn], color="tab:blue", ms=5, alpha=0.55)
        ax.set_xscale("log")
        ax.set_xlim(2e-3, 1.0)
        ax.set_ylim(0, {"tanh": 0.5, "gelu": 1.1, "swish": 0.9}[act])
        ax.set_xlabel(r"$q=\theta_0/\pi = h\Omega/\pi$   (frequency per grid cell, $h=2/N$)")
        ax.set_ylabel(r"$\lambda$")
        ax.set_title(act, fontsize=12)
        ax.grid(alpha=0.3)
    handles = [Line2D([0], [0], color="#d1352b", lw=1.8, label="new rule $\\lambda(q)$"),
               Line2D([0], [0], color="k", lw=1.3, ls=":", label="expC07 rule"),
               Line2D([0], [0], marker="o", color="tab:blue", lw=0, ms=7, mfc="none", mew=1.4, label="measured wall (10x shoulder)"),
               Line2D([0], [0], marker="o", color="tab:blue", lw=0, ms=5, alpha=0.55, label="measured argmin")]
    handles += [Line2D([0], [0], marker=markers[fn], color="gray", lw=0, ms=6, label=TARGET_TEX[fn]) for fn in TARGET_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9)
    fig.suptitle("expC08 -- where the measured wall sits vs the two rules, as a function of frequency per cell (cells with min rel $L_2 \\leq 10^{-6}$)",
                 y=0.86, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    p = OUT_DIR / "figures" / "c08_wall_vs_q.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


def summarize(rows):
    """Per cell: measured argmin, the 10x right shoulder (wall), and the error at each rule's lambda."""
    table = []
    for act in ACTS:
        print(f"\n{act}: per cell  argmin | wall(10x) | new | prev | err@new | err@prev | min")
        for fn in TARGET_ORDER:
            for j, N in enumerate(WIDTHS):
                lam, rel = curve(rows, act, fn, N)
                if not lam.size:
                    continue
                new, prev = TABLE[act][fn][j], PREV[act]
                k = int(np.argmin(rel))
                wall = right_shoulder(lam, rel)
                e_new = float(np.interp(np.log(new), np.log(lam), np.log(rel))) if new else np.nan
                e_prev = float(np.interp(np.log(prev), np.log(lam), np.log(rel)))
                hl, hb = best_halo(rows, act, fn, N)
                rec = {"act": act, "fn": fn, "N": N, "argmin": float(lam[k]), "min": float(rel[k]),
                       "wall10": wall, "new": new, "prev": prev,
                       "best_halo_at_argmin": int(hb[int(np.argmin(np.abs(hl - lam[k])))]) if hl.size else None,
                       "err_at_new": float(np.exp(e_new)) if new else None,
                       "err_at_prev": float(np.exp(e_prev))}
                table.append(rec)
                ns = f"{new:.3f}" if new else "  -- "
                en = f"{rec['err_at_new']:.1e}" if new else "   --  "
                print(f"  {fn:10s} N={N:4d}  {lam[k]:.3f} | {wall:.3f} | {ns} | {prev:.3f} | {en} | "
                      f"{rec['err_at_prev']:.1e} | {rel[k]:.1e}")
    (DATA / "c08_summary.json").write_text(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fig-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    lam_grid, widths, fns, acts = LAM_GRID, WIDTHS, TARGET_ORDER, ACTS
    if args.smoke:
        lam_grid, widths, fns, acts = [0.1, 0.235, 0.5], [16, 64], ["sin_1pi", "sin_16pi"], ["tanh"]

    rows_path = DATA / "c08_rows_halo_ladder.json"
    if args.fig_only:
        rows = json.loads(rows_path.read_text())
    else:
        halos = HALOS if not args.smoke else [8, "default"]
        jobs = sorted([{"act": a, "fn": f, "lam": l, "N": n, "halo": hh}
                       for a in acts for f in fns for l in lam_grid for n in widths for hh in halos],
                      key=lambda j: -j["N"])
        print(f"expC08 | {len(jobs)} solves | acts={acts} | targets={fns} | "
              f"lambda {lam_grid[0]}..{lam_grid[-1]} ({len(lam_grid)} pts) | N={widths} | workers={args.workers}")
        rows, t0 = [], time.time()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_cell, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 1000 == 0 or i == len(jobs):
                    rows_path.write_text(json.dumps(rows))
                    print(f"  [{i}/{len(jobs)}] ({time.time()-t0:.0f}s)", flush=True)
        if args.smoke:
            for r in sorted(rows, key=lambda r: (r["fn"], r["N"], r["lam"])):
                print(r)
            return

    summarize(rows)
    make_figures(rows)
    make_halo_figures(rows)
    make_wall_figure(rows)


if __name__ == "__main__":
    main()
