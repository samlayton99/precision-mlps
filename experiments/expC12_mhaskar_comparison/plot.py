"""PNG figures for the Mhaskar/QUILL parameter-quantization comparison."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/checkpoint_C_geometry/expC12_mhaskar_comparison"


def main(output=OUT):
    data, figures = output / "data", output / "figures"
    strict = output.name == "strict"
    figures.mkdir(exist_ok=True)
    rows = json.loads((data / "summary.json").read_text())
    search = np.load(data / "mhaskar_search.npz")
    diagnostic = np.load(data / "diagnostics.npz")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 14, "axes.titlesize": 16,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    colors = plt.get_cmap("viridis")
    p = np.array([r["p"] for r in rows])
    q = np.array([r["quill_error"] for r in rows])
    m = np.array([r["mhaskar_error"] for r in rows])
    intercept = np.mean(np.log2(q[(p >= 16) & (p <= 40)])+p[(p >= 16) & (p <= 40)])
    fig, ax = plt.subplots(figsize=(8.1, 5.7))
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.88)
    ax.semilogy(p, q, "o-", ms=3.5, lw=2.3, color=colors(.45), label=r"QUILL (predicted $\lambda$)")
    ax.semilogy(p, m, "s-", ms=3.5, lw=2.3, color=colors(.08), label="Mhaskar (selected degree/step)")
    ax.semilogy(p, np.exp2(intercept-p), "--", color=".35", lw=1.8, label=r"$O(\log(1/\varepsilon))$")
    ax.set(xlim=(8, 53), ylim=(1e-16, 10), xlabel=r"Working precision $p$ (bits)",
           ylabel=r"Relative $L^2$ error", title="Precision law")
    ax.set_xticks([8, 16, 24, 32, 40, 48, 53])
    ax.set_yticks(10.**np.arange(-16, 1, 4))
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", bbox_to_anchor=(1., .86), framealpha=.95, fontsize=11)
    fig.savefig(figures / "precision_law_comparison.png", dpi=240)
    plt.close(fig)

    # Polynomial baseline vs neural conversion, and explicit step-size tradeoff.
    fig, axes = plt.subplots(1, 2, figsize=(13., 5.7))
    fig.subplots_adjust(left=.075, right=.98, bottom=.16, top=.76, wspace=.29)
    ax = axes[0]
    ax.semilogy(search["degrees"], search["polynomial_error"], "o-", color=colors(.45), lw=2., ms=3,
                label="Chebyshev polynomial (FP64)")
    errors_for_degree = search["screen_error"] if strict else search["validation_error"]
    ax.semilogy(search["degrees"], np.min(errors_for_degree[:, :, -1], axis=0),
                "s-", color=colors(.08), lw=2., ms=3,
                label="Tanh network (best screening-grid step)" if strict else "Tanh network (best validation step)")
    ax.set(xlim=(0, 160), ylim=(1e-16, 1e12), xlabel="Polynomial degree",
           ylabel=r"Relative $L^2$ error")
    ax.set_title("Polynomial approximation and conversion", y=1.26, fontsize=15)
    ax.set_yticks(10.**np.arange(-16, 13, 4))
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=10)
    ax = axes[1]
    for degree, errors, color in zip(diagnostic["degrees"], diagnostic["network_error"],
                                     colors(np.linspace(.08, .86, len(diagnostic["degrees"])))):
        ax.loglog(diagnostic["steps"], errors, lw=2., color=color, label=f"Degree {degree}")
    ax.set(xlim=(diagnostic["steps"][0], diagnostic["steps"][-1]), ylim=(1e-4, 1e24),
           xlabel=r"Difference step $h$", ylabel=r"Relative $L^2$ error")
    ax.set_title("53-bit step-size sensitivity" if strict else "FP64 step-size sensitivity", y=1.26, fontsize=15)
    ax.set_yticks(10.**np.arange(-4, 25, 4))
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=10, ncol=3)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(figures / "mhaskar_diagnostics.png", dpi=240)
    plt.close(fig)
    from experiments.expC09_bandwidth_figures.combined import main as combined
    combined(comparison_data=data)
    print(figures)


if __name__ == "__main__":
    main()
