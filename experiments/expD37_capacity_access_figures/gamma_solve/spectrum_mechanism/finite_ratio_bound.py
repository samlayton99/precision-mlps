"""Fixed-reference finite-gamma lower bound versus actual eigenvalue ratios.

No training. New eigenvalues are used only for validation and comparison;
the correction and row-sum bound use the new finite kernel itself.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

from run import Geometry, OUT, design


def main_kernel(geom, gamma):
    x, centers = geom.arrays()
    d = x[:, None] - x[None, :]
    f = np.full_like(d, 1 / gamma)
    np.divide(d, np.tanh(gamma * d), out=f, where=d != 0)
    return (len(centers) + 1) / geom.m - 2 * f / (geom.h * geom.m)


def symmetric(a):
    return (a + a.T) / 2


def calculate():
    geom = Geometry()
    x, centers = geom.arrays()
    reference = 8.0
    ranks = np.array([12, 20, 32])
    gammas = np.unique(np.r_[np.geomspace(reference, 128, 81), [8, 16, 32, 64, 128]])
    b0, _ = design(x, centers, reference)
    u0, s0, _ = la.svd(b0, full_matrices=False, lapack_driver="gesvd")
    initial_eigenvalues = s0 * s0
    initial_ratios = initial_eigenvalues[ranks - 1] / initial_eigenvalues[0]
    k00 = main_kernel(geom, reference)
    rows = []
    worst_reconstruction = 0.0
    worst_epsilon_disagreement = 0.0
    for gamma in gammas:
        b, _ = design(x, centers, gamma)
        k = b @ b.T
        upper_top = np.max(np.sum(np.abs(k), axis=1))
        delta_main = main_kernel(geom, gamma) - k00
        # Only the rectangular SVD is used for the actual small eigenvalues.
        actual_values = la.svdvals(b) ** 2
        assert upper_top >= actual_values[0] * (1 - 1e-12)
        for column, rank in enumerate(ranks):
            u = u0[:, :rank]
            c = symmetric(np.diag(initial_eigenvalues[:rank]) + u.T @ delta_main @ u)
            ub = u.T @ b
            compressed_actual = symmetric(ub @ ub.T)
            remainder = symmetric(compressed_actual - c)
            eigen_c, vectors_c = la.eigh(c)
            assert eigen_c[0] > 0
            normalized_remainder = (vectors_c.T @ remainder @ vectors_c) / np.sqrt(
                eigen_c[:, None] * eigen_c[None, :]
            )
            epsilon = np.max(np.abs(la.eigvalsh(symmetric(normalized_remainder))))
            # An independent generalized eigensolve checks the whitening formula.
            epsilon_generalized = np.max(np.abs(la.eigvalsh(remainder, c)))
            worst_epsilon_disagreement = max(
                worst_epsilon_disagreement, abs(epsilon - epsilon_generalized)
            )
            numerator = max(1 - epsilon, 0) * eigen_c[0]
            lower = numerator / upper_top
            actual_ratio = actual_values[rank - 1] / actual_values[0]
            compressed_min = la.svdvals(ub)[-1] ** 2
            tolerance = 5e-11 * actual_values[0]
            assert numerator <= compressed_min + tolerance
            assert compressed_min <= actual_values[rank - 1] + tolerance
            assert lower <= actual_ratio * (1 + 1e-7)
            reconstruction = la.norm(c + remainder - compressed_actual, 2)
            worst_reconstruction = max(worst_reconstruction, reconstruction)
            rows.append({
                "gamma": float(gamma), "rank": int(rank),
                "actual_ratio": float(actual_ratio), "lower_bound": float(lower),
                "bound_fraction": float(lower / actual_ratio),
                "initial_ratio": float(initial_ratios[column]),
                "epsilon": float(epsilon), "main_compressed_min": float(eigen_c[0]),
                "actual_compressed_min": float(compressed_min),
                "upper_lambda1": float(upper_top), "actual_lambda1": float(actual_values[0]),
                "improvement_bound": float(lower / initial_ratios[column]),
            })
    assert worst_epsilon_disagreement < 1e-6
    return {
        "reference_gamma": reference, "ranks": ranks.tolist(),
        "geometry": {"interior_intervals": geom.N, "tanh_neurons": len(centers),
                     "halo_per_side": geom.halo, "samples": geom.m, "spacing": geom.h,
                     "center_integration_bounds": list(geom.bounds)},
        "precision": "float64; evaluated theorem bound, not interval-certified",
        "correction": "measured from the actual finite kernel; no fitted parameters",
        "validation": {"max_reconstruction_absolute": worst_reconstruction,
                       "max_epsilon_solver_disagreement": worst_epsilon_disagreement,
                       "bound_inequalities_passed": True},
        "rows": rows,
    }


def plot(result, path):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.25, 1]})
    colors = ["#31688e", "#35b779"]
    for col, rank in enumerate(result["ranks"]):
        rows = [row for row in result["rows"] if row["rank"] == rank]
        gamma = np.array([r["gamma"] for r in rows])
        actual = np.array([r["actual_ratio"] for r in rows])
        lower = np.array([r["lower_bound"] for r in rows])
        initial = rows[0]["initial_ratio"]
        ax = axes[0, col]
        ax.plot(gamma, actual, color=colors[0], lw=2.5)
        ax.plot(gamma, np.where(lower > 0, lower, np.nan), color=colors[1], lw=2.5, ls="--")
        ax.axhline(initial, color="0.45", lw=1.3, ls=":")
        ax.set_yscale("log")
        ax.set_title(rf"Eigenvalue rank $i={rank}$", pad=13)
        ax.set_ylabel(rf"$\lambda_{{{rank}}}(\gamma)/\lambda_1(\gamma)$")
        # Each axis is explicitly labelled; separate limits preserve small growth
        # in earlier ranks while showing the five-decade change of rank 32.
        ax.set_ylim(min(initial, min(lower[lower > 0])) / 1.65, max(actual) * 1.7)
        ax.grid(True, which="major", alpha=.23)
        bottom = axes[1, col]
        bottom.plot(gamma, lower / actual, color=colors[1], lw=2.2)
        bottom.axhline(1, color="0.45", lw=1.2, ls=":")
        bottom.set_ylim(0, 1.05)
        bottom.set_yticks([0, .25, .5, .75, 1])
        bottom.set_ylabel("Lower bound / actual")
        bottom.set_xlabel(r"Tanh slope $\gamma$")
        bottom.grid(True, which="major", alpha=.23)
        for a in (ax, bottom):
            a.set_xscale("log", base=2)
            a.set_xlim(8, 128)
            a.set_xticks([8, 16, 32, 64, 128])
            a.xaxis.set_major_formatter(ScalarFormatter())
    fig.suptitle("Does the lower bound capture the increase in eigenvalue ratios?", y=.98, fontsize=17)
    fig.text(.5, .931,
             r"Fixed reference $\gamma_0=8$ throughout  |  153 tanh neurons + bias  |  263 samples",
             ha="center", fontsize=11)
    handles = [Line2D([], [], color=colors[0], lw=2.5, label="Actual finite-kernel ratio"),
               Line2D([], [], color=colors[1], lw=2.5, ls="--", label="Lower bound, including correction"),
               Line2D([], [], color=".45", lw=1.3, ls=":", label=r"Initial ratio at $\gamma_0=8$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .911), ncol=3, frameon=False)
    fig.text(.5, .024,
             "The correction is measured from the finite kernel; new eigenvalues are used only for comparison.\n"
             "Top-row vertical limits differ by rank. FP64 evaluations of the theorem bound; no interval rounding guarantee.",
             ha="center", va="bottom", fontsize=9.5, color=".3")
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.13, hspace=.24, wspace=.32)
    fig.savefig(path, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result = calculate()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data/finite_ratio_bound.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, OUT / "finite_ratio_bound.png")
    selected = [row for row in result["rows"] if row["gamma"] in [8, 16, 64, 128]]
    print(json.dumps({"validation": result["validation"], "selected": selected}, indent=2))
