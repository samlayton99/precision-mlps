"""Plot fp64 QI and least-squares readout coefficients on shared geometry.

Rows are targets and columns are QI interior resolutions. The x-axis includes
halo centers, but each panel's y-range is capped at three times the largest
coefficient attached to a center in [-1, 1] so extreme halo values cannot make
the interior comparison unreadable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import construct_qi, default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target


RESULTS_DIR = (
    REPO_ROOT
    / "results"
    / "checkpoint_A_numerics"
    / "expA02_qi_vs_lstsq"
)
OUTPUT_PATH = RESULTS_DIR / "qi_vs_lstsq_coefficients_fp64.png"

TARGETS = [
    ("runge", "Runge"),
    ("sine_mixture", "Sine mixture"),
    ("exp", r"$e^x$"),
    ("sine_8pi", r"$\sin(8\pi x)$"),
]
WIDTHS = [64, 128, 256]
LAMBDA_STAR = 0.25
KC = 160


def panel_limit(
    centers: np.ndarray, *coefficient_sets: np.ndarray
) -> float:
    """Return a symmetric y-limit that may clip extreme halo coefficients."""
    centers = np.asarray(centers, dtype=np.float64)
    coefficients = [
        np.asarray(values, dtype=np.float64) for values in coefficient_sets
    ]
    interior = np.abs(centers) <= 1.0 + 1e-12
    all_max = max(float(np.max(np.abs(values))) for values in coefficients)
    interior_max = max(
        float(np.max(np.abs(values[interior]))) for values in coefficients
    )
    if all_max == 0.0:
        return 1.0
    if interior_max == 0.0:
        return 1.05 * all_max
    return min(1.05 * all_max, 3.0 * interior_max)


def coefficient_pair(target_name: str, N: int):
    """Construct QI and lstsq fp64 coefficients on identical QI geometry."""
    target = get_target(target_name)
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    qi = construct_qi(
        target.fn_numpy,
        target.deriv_numpy,
        N,
        precision="fp64",
        lambda_star=LAMBDA_STAR,
        Kc=KC,
        halo=halo,
        use_cache=True,
    )

    centers = qi.centers
    gamma = np.full(len(centers), qi.gamma, dtype=np.float64)
    n_train = max(512, 2 * len(centers))
    x_train = np.linspace(-1.0, 1.0, n_train, dtype=np.float64)
    y_train = target.fn_numpy(x_train)
    phi = build_phi(x_train, gamma, centers)
    lstsq_coefficients, _, _ = solve_readout_with_bias(
        phi, y_train, method="lstsq"
    )
    return centers, qi.a_coeffs, lstsq_coefficients


def make_figure() -> Path:
    """Build and save the requested 4-by-3 coefficient overlay."""
    fig, axes = plt.subplots(
        len(TARGETS),
        len(WIDTHS),
        figsize=(15, 12),
        sharex=False,
        sharey=False,
    )

    for row, (target_name, target_label) in enumerate(TARGETS):
        for col, N in enumerate(WIDTHS):
            ax = axes[row, col]
            centers, qi_coefficients, lstsq_coefficients = coefficient_pair(
                target_name, N
            )

            ax.axvspan(-1.0, 1.0, color="0.92", zorder=0)
            ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
            ax.plot(
                centers,
                lstsq_coefficients,
                color="red",
                linewidth=1.1,
                label="lstsq fp64",
                zorder=3,
            )
            ax.plot(
                centers,
                qi_coefficients,
                color="blue",
                linewidth=1.0,
                label="QI fp64",
                zorder=2,
            )

            limit = panel_limit(
                centers, qi_coefficients, lstsq_coefficients
            )
            ax.set_ylim(-limit, limit)
            ax.grid(True, alpha=0.22)
            if row == 0:
                ax.set_title(f"$N={N}$")
            if col == 0:
                ax.set_ylabel(f"{target_label}\nreadout coefficient")
            if row == len(TARGETS) - 1:
                ax.set_xlabel("center $c_i$")

            relative_gap = np.linalg.norm(
                lstsq_coefficients - qi_coefficients
            ) / max(np.linalg.norm(qi_coefficients), 1e-300)
            print(
                f"{target_name:13s} N={N:3d} width={len(centers):3d} "
                f"relative coefficient gap={relative_gap:.3e} "
                f"y-limit={limit:.3e}"
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        r"expA02: fp64 readout coefficients on shared QI geometry "
        r"($\lambda=0.25$)",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    make_figure()
