"""Consolidated 3x4 lambda tradeoff plot.

Merges all data from exp01 and exp0A into a single all_data.json,
then produces a 3-row x 4-column figure:
  Row 1: fp64 QI + fp64 lstsq
  Row 2: mpmath QI + fp64 lstsq
  Row 3: mpmath QI + mpmath lstsq

Columns: sine, runge, exp, sine_mixture

Usage:
    python3 experiments/exp01_lambda_tradeoff/plot_consolidated.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP01_DIR = REPO_ROOT / "results" / "exp01_lambda_tradeoff"
EXP0A_DIR = REPO_ROOT / "results" / "exp0A_QI_vs_learn"
ALL_DATA_PATH = EXP01_DIR / "all_data.json"
PLOT_PATH = EXP01_DIR / "consolidated_linf.png"

TARGETS = ["sine", "runge", "exp", "sine_mixture"]
EXCLUDE_N = {256, 512}


def build_all_data():
    """Load all 5 data files and merge into a unified list."""
    all_data = {}  # key: (target, N, lambda, qi_prec, ls_prec) -> entry

    def key(target, N, lam, qi_prec, ls_prec):
        return (target, N, round(lam, 4), qi_prec, ls_prec)

    def add(k, qi_linf, qi_rel_l2, ls_linf, ls_rel_l2, source):
        if k[1] in EXCLUDE_N:
            return
        if k not in all_data:
            all_data[k] = {
                "target": k[0], "N": k[1], "lambda_star": k[2],
                "qi_precision": k[3], "lstsq_precision": k[4],
                "qi_linf": qi_linf, "qi_rel_l2": qi_rel_l2,
                "lstsq_linf": ls_linf, "lstsq_rel_l2": ls_rel_l2,
                "source": source,
            }

    # --- exp01/data.json: fp64/fp64 ---
    with open(EXP01_DIR / "data.json") as f:
        for r in json.load(f):
            add(key(r["target"], r["N"], r["lambda_star"], "fp64", "fp64"),
                r["qi_linf"], r["qi_rel_l2"], r["lstsq_linf"], r["lstsq_rel_l2"],
                "exp01/data.json")

    # --- exp01/data_fine_fp64.json: fp64/fp64 (fine, overwrites coarse) ---
    with open(EXP01_DIR / "data_fine_fp64.json") as f:
        for r in json.load(f):
            k = key(r["target"], r["N"], r["lambda_star"], "fp64", "fp64")
            if k[1] not in EXCLUDE_N:
                all_data[k] = {
                    "target": k[0], "N": k[1], "lambda_star": k[2],
                    "qi_precision": "fp64", "lstsq_precision": "fp64",
                    "qi_linf": r["qi_linf"], "qi_rel_l2": r["qi_rel_l2"],
                    "lstsq_linf": r["lstsq_linf"], "lstsq_rel_l2": r["lstsq_rel_l2"],
                    "source": "exp01/data_fine_fp64.json",
                }

    # --- exp01/data_mpmath.json: mpmath/fp64 ---
    with open(EXP01_DIR / "data_mpmath.json") as f:
        for r in json.load(f):
            add(key(r["target"], r["N"], r["lambda_star"], "mpmath", "fp64"),
                r["qi_linf"], r["qi_rel_l2"], r["lstsq_linf"], r["lstsq_rel_l2"],
                "exp01/data_mpmath.json")

    # --- exp01/data_fine.json: mpmath/fp64 (fine, overwrites coarse) ---
    with open(EXP01_DIR / "data_fine.json") as f:
        for r in json.load(f):
            k = key(r["target"], r["N"], r["lambda_star"], "mpmath", "fp64")
            if k[1] not in EXCLUDE_N:
                all_data[k] = {
                    "target": k[0], "N": k[1], "lambda_star": k[2],
                    "qi_precision": "mpmath", "lstsq_precision": "fp64",
                    "qi_linf": r["qi_linf"], "qi_rel_l2": r["qi_rel_l2"],
                    "lstsq_linf": r["lstsq_linf"], "lstsq_rel_l2": r["lstsq_rel_l2"],
                    "source": "exp01/data_fine.json",
                }

    # --- exp0A/data.json: 4-way, extract 3 precision combos ---
    with open(EXP0A_DIR / "data.json") as f:
        for r in json.load(f):
            t, N, lam = r["target"], r["N"], r["lambda_star"]
            if N in EXCLUDE_N:
                continue
            # fp64/fp64
            add(key(t, N, lam, "fp64", "fp64"),
                r["qi_f64_linf"], r["qi_f64_rel_l2"],
                r["ls_f64_linf"], r["ls_f64_rel_l2"],
                "exp0A/data.json")
            # mpmath/fp64
            add(key(t, N, lam, "mpmath", "fp64"),
                r["qi_mp_linf"], r["qi_mp_rel_l2"],
                r["ls_f64_linf"], r["ls_f64_rel_l2"],
                "exp0A/data.json")
            # mpmath/mpmath
            add(key(t, N, lam, "mpmath", "mpmath"),
                r["qi_mp_linf"], r["qi_mp_rel_l2"],
                r["ls_mp_linf"], r["ls_mp_rel_l2"],
                "exp0A/data.json")

    result = sorted(all_data.values(),
                    key=lambda r: (r["qi_precision"], r["lstsq_precision"],
                                   r["target"], r["N"], r["lambda_star"]))
    return result


def plot_consolidated(all_data):
    """Create the 3x4 consolidated plot."""
    rows = [
        ("fp64", "fp64", "fp64 / fp64"),
        ("mpmath", "fp64", "mpmath / fp64"),
        ("mpmath", "mpmath", "mpmath / mpmath"),
    ]

    palette = {
        16: "#1f77b4",
        32: "#ff7f0e",
        64: "#2ca02c",
        96: "#9467bd",
        128: "#d62728",
    }

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(r"Lambda Tradeoff: $L_\infty$ Error (dashed = QI, solid = lstsq)",
                 fontsize=14, y=0.99)

    for row_idx, (qi_prec, ls_prec, row_label) in enumerate(rows):
        row_data = [r for r in all_data
                    if r["qi_precision"] == qi_prec and r["lstsq_precision"] == ls_prec]

        all_N = sorted(set(r["N"] for r in row_data))

        for col_idx, target in enumerate(TARGETS):
            ax = axes[row_idx][col_idx]
            target_data = [r for r in row_data if r["target"] == target]

            for N in all_N:
                if N in EXCLUDE_N:
                    continue
                color = palette.get(N, "#333333")
                width_data = sorted(
                    [r for r in target_data if r["N"] == N],
                    key=lambda r: r["lambda_star"],
                )
                if not width_data:
                    continue

                lams = [r["lambda_star"] for r in width_data]
                qi_vals = [r["qi_linf"] for r in width_data]
                ls_vals = [r["lstsq_linf"] for r in width_data]

                # Filter NaNs
                lam_qi = [l for l, v in zip(lams, qi_vals) if v == v]
                val_qi = [v for v in qi_vals if v == v]
                lam_ls = [l for l, v in zip(lams, ls_vals) if v == v]
                val_ls = [v for v in ls_vals if v == v]

                if val_qi:
                    ax.semilogy(lam_qi, val_qi, "--", color=color, linewidth=1.2)
                if val_ls:
                    ax.semilogy(lam_ls, val_ls, "-", color=color, linewidth=1.2)

            ax.set_xlim(0.1, 0.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if row_idx == 0:
                ax.set_title(target, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(r"$L_\infty$ error", fontsize=9)
            if row_idx == 2:
                ax.set_xlabel(r"$\lambda$", fontsize=9)

        # Row label on the right side
        axes[row_idx][3].annotate(
            row_label, xy=(1.02, 0.5), xycoords="axes fraction",
            fontsize=10, ha="left", va="center", rotation=-90,
            fontweight="bold",
        )

    # Compact legend: one line per N (color swatch), plus method explanation
    legend_elements = []
    for N in sorted(palette.keys()):
        color = palette[N]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linestyle="-", linewidth=2.5,
                       label=f"N={N}"))
    # Method markers
    legend_elements.append(
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   linewidth=1.2, label="QI (dashed)"))
    legend_elements.append(
        plt.Line2D([0], [0], color="gray", linestyle="-",
                   linewidth=1.2, label="lstsq (solid)"))

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 0.97, 0.97])
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    all_data = build_all_data()

    # Save
    EXP01_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALL_DATA_PATH, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved {ALL_DATA_PATH} ({len(all_data)} entries)")

    # Count per precision combo
    from collections import Counter
    combo_counts = Counter((r["qi_precision"], r["lstsq_precision"]) for r in all_data)
    for (qp, lp), cnt in sorted(combo_counts.items()):
        print(f"  {qp}/{lp}: {cnt} entries")

    plot_consolidated(all_data)
