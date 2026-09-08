#!/usr/bin/env python3
"""Plot selected lambda, implied gamma, and held-out error against width."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
rows = json.loads((RESULTS / "zo_whitening_width_sweep.json").read_text())["selected"]
rows = [row for row in rows if row["width"] <= 128]
styles = {"full": ("#0072B2", "o", "Global whitening"),
          "partial": ("#D55E00", "s", "Block whitening (32 features)")}

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
for method, (color, marker, label) in styles.items():
    data = sorted((r for r in rows if r["method"] == method), key=lambda r: r["width"])
    widths = [r["width"] for r in data]
    axes[0].plot(widths, [r["final_lambda"] for r in data], marker=marker, color=color, label=label)
    axes[1].plot(widths, [r["final_gamma"] for r in data], marker=marker, color=color, label=label)
    axes[2].plot(widths, [r["test_relative_l2"] for r in data], marker=marker, color=color, label=label)

for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Hidden width")
    ax.grid(True, which="both", alpha=0.22)
axes[0].set_ylabel(r"Final $\lambda=\gamma h$")
axes[0].set_title("Selected normalized bandwidth")
axes[1].set_yscale("log", base=2)
axes[1].set_ylabel(r"$\gamma=\lambda/h$")
axes[1].set_title("Selected feature scale")
axes[2].set_yscale("log")
axes[2].set_ylabel(r"Test relative $L_2$")
axes[2].set_title("Held-out interpolation error")
axes[0].legend(frameon=True, fontsize=8, loc="best")
fig.suptitle("ZO w/ Whitening", y=1.02, fontsize=13)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(RESULTS / f"lambda_gamma_error_vs_width.{suffix}", dpi=220,
                bbox_inches="tight")
