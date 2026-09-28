"""expC11 figures: the three-panel figure with the fully p-bit panel (c), and a diagnostic comparison.

Run from the repository root after run.py (--sweep, --anchors, --standard):
    uv run --extra dev python experiments/expC11_true_precision_law/plot.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/checkpoint_C_geometry/expC11_true_precision_law"
DATA = OUT / "data"
FORWARD_ONLY = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/precision_law_W1024_strict/data/summary.json"


def intercept(rows, key="rule_error"):
    """log2 C of E = C 2^-p, slope fixed at -1, fitted on p = 16..40 (the panel's rule)."""
    return float(np.mean([np.log2(r[key]) + r["p"] for r in rows if 16 <= r["p"] <= 40]))


def diagnostic():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted((json.loads(s) for s in (DATA / "measurements.jsonl").read_text().splitlines()), key=lambda r: r["p"])
    forward = json.loads(FORWARD_ONLY.read_text())
    anchors = json.loads((DATA / "anchors.json").read_text())
    standard = json.loads((DATA / "standard.json").read_text())
    ps = np.array([r["p"] for r in rows])
    true_err = np.array([r["relative_l2"] for r in rows])
    fwd_p = np.array([r["p"] for r in forward])
    fwd_err = np.array([r["rule_error"] for r in forward])
    c_true = intercept([{"p": r["p"], "rule_error": r["relative_l2"]} for r in rows])
    c_fwd = intercept(forward)
    green, grey = plt.get_cmap("viridis")(.45), "#9a9a9a"

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    ax.plot(ps, true_err, "o-", color=green, lw=2, ms=3.5, label="Everything at $p$ bits (expC11)")
    ax.plot(fwd_p, fwd_err, "o-", color=grey, lw=1.6, ms=3,
            label="Forward pass at $p$ bits, solve in FP64 (expC09 strict)")
    pp = np.array([8, 53])
    ax.plot(pp, np.exp2(c_true - pp), "--", color=green, lw=1.2, label=f"$C\\,2^{{-p}}$, $C={2 ** c_true:.0f}$")
    ax.plot(pp, np.exp2(c_fwd - pp), "--", color=grey, lw=1.2, label=f"$C\\,2^{{-p}}$, $C={2 ** c_fwd:.1f}$")
    markers = {"gelsd": "s", "gelss": "^", "gelsy": "D"}
    for name, p in (("fp32", 24), ("fp64", 53)):
        for drv, err in standard[name].items():
            ax.scatter([p], [err], marker=markers[drv], s=46, facecolor="none", edgecolor="#c73435", zorder=5,
                       label=f"numpy tanh + scipy {drv}, native FP32/FP64" if name == "fp32" else None)
    ax.scatter([24], [anchors["fp32"]["error_ieee_range"]], marker="*", s=120, color="#1f4e99", zorder=6,
               label="Exact IEEE FP32 (= reference SGELSS, bit for bit)")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 10)
    ax.set_yticks([10.0 ** k for k in range(-16, 1, 4)])
    ax.set_xlim(8, 53)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 53])
    ax.set_xlabel("Working precision $p$ (bits)")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.grid(alpha=.2)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=8, borderaxespad=0)

    ax = axes[1]
    ax.plot(ps, true_err * 2.0 ** ps, "o-", color=green, lw=2, ms=3.5, label=r"everything at $p$ bits, $\mathrm{RCOND}=2\cdot2^{-p}$")
    ax.plot(fwd_p, fwd_err * 2.0 ** fwd_p, "o-", color=grey, lw=1.6, ms=3, label="forward only (expC09 strict)")
    shades = plt.get_cmap("viridis")(np.linspace(.15, .85, 3))
    factors = rows[0]["cutoff_factors"]
    for j, kappa in enumerate(factors[1:], start=1):
        e = np.array([r["relative_l2_by_cutoff"][j] for r in rows])
        ax.plot(ps, e * 2.0 ** ps, "-", color=shades[j - 1], lw=1, alpha=.8, label=rf"$\mathrm{{RCOND}}={kappa}\cdot2^{{-p}}$")
    ax.axvspan(16, 40, color="#000000", alpha=.04, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    ax.set_xlim(8, 53)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 53])
    ax.set_xlabel("Working precision $p$ (bits)")
    ax.set_ylabel(r"Error $\times\,2^{p}$  (the constant $C$ in $E=C\,2^{-p}$)")
    ax.grid(alpha=.2)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=8, borderaxespad=0)
    fig.suptitle(r"Chirp $\sin(8\pi(x+1)^2)$, $W=1024$, refined-rule $\lambda(p)$; shaded: the $p$ range of the intercept fit",
                 fontsize=10, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    path = OUT / "figures" / "diagnostic_true_vs_forward_only.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    from experiments.expC09_bandwidth_figures.combined import main as render_three_panel
    render_three_panel("rule", 1024, true_precision_data=DATA, paper_style=True)
    diagnostic()
