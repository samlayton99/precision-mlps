"""(ex-expC08.) SUPERSEDED SELECTOR, kept for the record: the skeptic review (SKEPTIC_REVIEW_2026-09-08.md, P5) showed the
tolerance 1.6 was set 5% above halo 32's worst p90; the defensible statement is halo >= 16 cells per side independent of N.
Choose ONE halo per width from the expC08 halo ladder (Sam, 2026-09-08: a single rule per width, big
enough never to worry about, not so big that it is unnecessary noise in the solve; never tied to lambda,
which would confound the lambda curves).

Criterion. Floor jitter spans two decades cell to cell, so raw per-lambda ratios are noise; each cell's
log-error curve is first smoothed by a running median over SMOOTH lambda points. For width N and ladder
halo h the penalty is
    rho_h(cell, lambda) = smoothed rel_l2(h, lambda) / smoothed rel_l2(default, lambda),
relative to the default halo max(70, 0.4N) (the reference that is certainly big enough), over resolved
cells (act, target with cell min rel L2 <= 1e-6) and lambda in [LAM_LO, 1.5] (basin and both walls; below
LAM_LO the kernel reach exceeds every ladder halo and nothing is being tested). Per (N, h) we report the
geometric mean (typical penalty; < 1 means the smaller halo is typically better, i.e. the big halo was
adding noise), the 90th percentile, and the max with the cell responsible. The rule is the smallest halo
whose 90th-percentile penalty is <= TOL at EVERY width -- one number for all widths, because the halo
requirement is a kernel reach in grid cells, not a fraction of N. Writes halo_rule.json and
figures/expC08_halo_adequacy.png.

Run:  uv run --extra dev python experiments/expC07_lambda_energy_rule/lambda_rule/halo_ladder_c08.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_c08_tones import ACTS, DATA, HALOS, LAM_GRID, OUT_DIR, TARGET_ORDER, WIDTHS  # noqa: E402

import os
TOL = float(os.environ.get("HALO_TOL", 1.6))
LAM_LO = float(os.environ.get("HALO_LAM_LO", 0.15))
RESOLVED = 1e-6
SMOOTH = 5


def smooth_log(v, k=SMOOTH):
    lv = np.log(np.asarray(v, dtype=float))
    out = np.empty_like(lv)
    r = k // 2
    for i in range(len(lv)):
        out[i] = np.median(lv[max(0, i - r): i + r + 1])
    return np.exp(out)


def load_rows():
    return json.loads((DATA / "c08_rows_halo_ladder.json").read_text())


def adequacy(rows):
    """ratios[N][halo] = array over LAM_GRID of rho_h(lambda); also the per-N default halo size."""
    idx = {}
    for r in rows:
        idx[(r["act"], r["fn"], r["N"], r["halo"], r["lam"])] = r["rel_l2"]
    halo_n = {}
    for r in rows:
        halo_n[(r["N"], r["halo"])] = r["halo_n"]
    ratios, stats = {}, {}
    lam = np.array(LAM_GRID)
    band = lam >= LAM_LO
    for N in WIDTHS:
        ratios[N], stats[N] = {}, {}
        cells = [(a, f) for a in ACTS for f in TARGET_ORDER
                 if min(idx.get((a, f, N, h, l), np.inf) for h in HALOS for l in LAM_GRID) <= RESOLVED]
        sm = {(a, f, h): smooth_log([idx[(a, f, N, h, l)] for l in LAM_GRID]) for a, f in cells for h in HALOS}
        for h in HALOS:
            stack = np.array([sm[(a, f, h)] / sm[(a, f, "default")] for a, f in cells])   # cells x lambda
            ratios[N][h] = {"p90": np.percentile(stack, 90, axis=0), "gmean": np.exp(np.log(stack).mean(axis=0))}
            R = stack[:, band]
            i, j = np.unravel_index(R.argmax(), R.shape)
            stats[N][h] = {"gmean": float(np.exp(np.log(R).mean())), "p90": float(np.percentile(R, 90)),
                           "max": float(R.max()), "max_cell": list(cells[i]), "max_lam": float(lam[band][j])}
    return ratios, stats, halo_n


def choose(stats, halo_n):
    """One halo for all widths: the smallest ladder entry whose p90 penalty is <= TOL at every width."""
    fixed = [h for h in HALOS if h != "default"]
    ok = [h for h in fixed if all(stats[N][h]["p90"] <= TOL for N in WIDTHS)]
    h = min(ok) if ok else "default"
    rule = {}
    for N in WIDTHS:
        rule[N] = {"halo": h, "halo_n": halo_n[(N, h)], "stats": {str(hh): stats[N][hh] for hh in HALOS}}
    return rule


def figure(ratios, halo_n, rule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("viridis")
    colors = {h: cmap(i / (len(HALOS) - 1)) for i, h in enumerate(HALOS)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, N in zip(axes.flat, WIDTHS):
        for h in HALOS:
            ax.loglog(LAM_GRID, ratios[N][h]["p90"], "-", color=colors[h], lw=1.8 if h == rule[N]["halo"] else 1.0)
            ax.loglog(LAM_GRID, ratios[N][h]["gmean"], "--", color=colors[h], lw=0.8, alpha=0.7)
        ax.axhline(TOL, color="k", lw=0.9, ls="--")
        ax.axvline(LAM_LO, color="k", lw=0.9, ls=":")
        ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-1, 1e8)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"N = {N}:  chosen halo = {rule[N]['halo_n']}  (default {halo_n[(N, 'default')]})", fontsize=10)
        ax.set_xlabel(r"$\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel("smoothed rel $L_2$(halo) / rel $L_2$(default halo)\nsolid = 90th pct over cells, dashed = geometric mean")
    handles = [Line2D([0], [0], color=colors[h], lw=1.6, label=f"halo {h}") for h in HALOS]
    handles += [Line2D([0], [0], color="k", lw=0.9, ls="--", label=f"tolerance {TOL:g}x on the 90th pct"),
                Line2D([0], [0], color="k", lw=0.9, ls=":", label=f"band start $\\lambda={LAM_LO}$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=9, frameon=False, fontsize=9)
    fig.suptitle("expC08 -- halo adequacy per width: error penalty of each halo relative to the default max(70, 0.4N), over resolved cells",
                 y=0.93, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = OUT_DIR / "archive" / "figures_full" / "expC08_halo_adequacy.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def main():
    rows = load_rows()
    ratios, stats, halo_n = adequacy(rows)
    rule = choose(stats, halo_n)
    print(f"band lambda >= {LAM_LO}, penalty vs default halo: gmean / p90 / max (cell, lambda)")
    for N in WIDTHS:
        print(f"N={N:4d}: chosen halo {rule[N]['halo_n']:3d}")
        for h in HALOS:
            st = stats[N][h]
            print(f"        halo {str(h):>7s} ({halo_n[(N, h)]:3d}): {st['gmean']:8.2f} {st['p90']:8.2f} {st['max']:10.1f}"
                  f"  ({st['max_cell'][0]} {st['max_cell'][1]} @ {st['max_lam']:.3f})")
    out = {str(N): rule[N] for N in WIDTHS}
    out["criterion"] = {"tol": TOL, "lam_lo": LAM_LO, "resolved": RESOLVED, "smooth": SMOOTH}
    (DATA / "c08_halo_selection.json").write_text(json.dumps(out, indent=1))
    figure(ratios, halo_n, rule)


if __name__ == "__main__":
    main()
