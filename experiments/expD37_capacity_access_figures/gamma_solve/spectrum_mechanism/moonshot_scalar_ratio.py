"""Evaluate proved scalar ratio bounds against already saved results.

No feature construction, eigensolve, quadrature, or training is performed.
The scalar formulas are exact-arithmetic bounds; plotted numbers and the
imported positive target masses are FP64 evaluations, not certificates.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism"
DEST = BASE / "moonshot_scalar_ratio"


def logcosh(x):
    return np.logaddexp(x, -x) - np.log(2.)


def scalar_bound(gamma, rank, sample_radius=1., center_radius=None):
    if rank <= 3:
        return 1.
    if center_radius is None:
        denominator = 4 * gamma * sample_radius + np.log(16.)
    else:
        denominator = (np.log(16.)
            + 2 * logcosh(gamma * (sample_radius + center_radius))
            - 2 * logcosh(gamma * (sample_radius - center_radius)))
    return float(min(1., 16 * np.exp(-2 * np.pi**2 * (rank - 3) / denominator)))


def necessary_steps(cap, mass, epsilon):
    if mass <= epsilon**2:
        return 0.
    return float(np.log(np.sqrt(mass) / epsilon) / -np.log1p(-cap / 2))


def main():
    source = BASE / "direct_ratio_upper_bound/data.json"
    saved = json.loads(source.read_text())
    rows = []
    for record in saved["records"]:
        gamma = record["gamma"]
        if gamma > 16:
            continue
        entries = []
        for item in record["ranks"]:
            rank, mass = item["rank"], item["positive_tail_mass"]
            finite = scalar_bound(gamma, rank, center_radius=76/64)
            universal = scalar_bound(gamma, rank)
            assert item["actual_ratio"] <= finite * (1 + 1e-10)
            assert finite <= universal * (1 + 1e-10)
            entries.append(dict(item, finite_scalar_upper=finite,
                universal_scalar_upper=universal,
                finite_scalar_steps=necessary_steps(finite, mass, saved["epsilon"]),
                universal_scalar_steps=necessary_steps(universal, mass, saved["epsilon"])))
        rows.append(dict(gamma=gamma, ranks=entries,
            actual_steps=record["actual_steps"],
            integral_best_steps=record["necessary_steps"],
            finite_scalar_best_steps=max(v["finite_scalar_steps"] for v in entries),
            universal_scalar_best_steps=max(v["universal_scalar_steps"] for v in entries)))
    DEST.mkdir(parents=True, exist_ok=True)
    output = dict(source=str(source.relative_to(ROOT)), geometry=saved["geometry"],
        sample_radius=1., center_radius=76/64, epsilon=saved["epsilon"],
        scalar_step_search_ranks=[v["rank"] for v in rows[0]["ranks"]],
        note="Scalar step maxima use only the four saved cutoffs. Integral step maxima use the source's full cutoff search. FP64, not interval-certified.",
        records=rows)
    (DEST / "data.json").write_text(json.dumps(output, indent=2) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.subplots_adjust(top=.63, bottom=.13, wspace=.27)
    gamma = [r["gamma"] for r in rows]
    rank26 = [next(v for v in r["ranks"] if v["rank"] == 26) for r in rows]
    for key, label, color, style in [
        ("actual_ratio", "Actual finite ratio", "black", "-"),
        ("upper_ratio", "Fourier integral upper bound", "#238b8d", "--"),
        ("finite_scalar_upper", "Scalar: finite center interval", "#cb702a", "-."),
        ("universal_scalar_upper", "Scalar: arbitrary centers", "#7852a0", ":"),
    ]:
        ax[0].plot(gamma, [r[key] for r in rank26], label=label, color=color, ls=style, lw=2)
    for key, label, color, style in [
        ("actual_steps", "Finite-kernel spectral forecast", "black", "-"),
        ("integral_best_steps", "Fourier necessary steps", "#238b8d", "--"),
        ("finite_scalar_best_steps", "Scalar necessary: finite interval", "#cb702a", "-."),
        ("universal_scalar_best_steps", "Scalar necessary: arbitrary centers", "#7852a0", ":"),
    ]:
        ax[1].plot(gamma, [r[key] for r in rows], label=label, color=color, ls=style, lw=2)
    for panel in ax:
        panel.set_xscale("log", base=2)
        panel.set_yscale("log")
        panel.set_xticks([4, 8, 16], ["4", "8", "16"])
        panel.set_xlabel(r"Common slope $\gamma$")
        panel.grid(alpha=.2, which="both")
        panel.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    ax[0].set_ylabel(r"$\lambda_{26}/\lambda_1$")
    ax[0].set_title("Eigenvalue ratio and three upper bounds", y=1.35, fontsize=10)
    ax[1].set_ylabel("Updates to 1% relative error")
    ax[1].set_title("Necessary delays from saved positive target masses", y=1.35, fontsize=10)
    fig.suptitle("Absolute ratio bounds; fixed geometry and mixed-sine target\n"
                 "No training or new eigensolves; checked floating-point evaluations", fontsize=11)
    fig.savefig(DEST / "scalar_ratio_comparison.png", dpi=190)
    plt.close(fig)
    for row in rows:
        if row["gamma"] in [4., 8., 16.]:
            entry = next(v for v in row["ranks"] if v["rank"] == 26)
            print(json.dumps(dict(gamma=row["gamma"], rank26=entry,
                scalar_finite_steps=row["finite_scalar_best_steps"],
                scalar_universal_steps=row["universal_scalar_best_steps"])))


if __name__ == "__main__":
    main()
