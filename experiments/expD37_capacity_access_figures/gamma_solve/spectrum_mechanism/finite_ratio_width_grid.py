"""Width/reference-gamma grid for the finite-change ratio lower bound.

Preserves the earlier single-width figure. No training. Numerically unresolved
values remain missing, distinct from structural zeros and zero theorem bounds.
"""
import json
from collections import Counter

import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

from finite_ratio_bound import OUT, design, symmetric
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

WIDTHS = [64, 128, 256, 512]
SAMPLES = [131, 263, 521, 1031]
REFERENCES = [16., 8., 4., 2.]
DIVISORS = [32, 16, 8, 4]
GAMMAS = 2. ** np.linspace(1, 7, 61)
EPS = np.finfo(float).eps


def main_matrix(x, h, width, gamma):
    d = x[:, None] - x[None, :]
    f = np.full_like(d, 1 / gamma)
    np.divide(d, np.tanh(gamma * d), out=f, where=d != 0)
    return (width + 1) / len(x) - 2 * f / (h * len(x))


def calculate():
    rows, geometries = [], []
    for n, m in zip(WIDTHS, SAMPLES):
        ranks = [n // divisor for divisor in DIVISORS]
        h = 2 / n
        halo = int(np.ceil(np.sqrt(n)))
        centers = -1 + np.arange(-halo, n + halo + 1) * h
        x = np.linspace(-1, 1, m)
        width = len(centers)
        geometries.append(dict(N=n, m=m, h=h, halo_per_side=halo,
                               tanh_neurons=width, feature_columns=width + 1))
        references = {}
        for reference in REFERENCES:
            b0, _ = design(x, centers, reference)
            u, s, _ = la.svd(b0, full_matrices=False, lapack_driver="gesvd")
            maxrank = max(i for i in ranks if i <= len(s))
            u = u[:, :maxrank]
            projected = u.T @ b0
            references[reference] = dict(u=u, base=projected @ projected.T,
                                        main=main_matrix(x, h, width, reference))
        for gamma in GAMMAS:
            b, _ = design(x, centers, gamma)
            singular = la.svdvals(b)
            actual_values = singular ** 2
            k = b @ b.T
            upper_top = np.max(np.sum(np.abs(k), axis=1))
            main = main_matrix(x, h, width, gamma)
            assert upper_top >= actual_values[0] * (1 - 1e-12)
            singular_threshold = 10 * EPS * max(b.shape) * singular[0]
            for reference, ref in references.items():
                if gamma < reference * (1 - 1e-14):
                    continue
                u = ref["u"]
                delta = main - ref["main"]
                c_full = symmetric(ref["base"] + u.T @ delta @ u)
                ub = u.T @ b
                t_full = symmetric(ub @ ub.T)
                # Conservative screening scale for cancellation in projected
                # dense matrices, not an interval-arithmetic enclosure.
                arithmetic_scale = la.norm(ref["base"], 2) + la.norm(delta, "fro")
                c_floor = 64 * EPS * arithmetic_scale
                for rank in ranks:
                    row = dict(N=n, m=m, reference_gamma=reference, gamma=float(gamma),
                               rank=rank, actual_ratio=None, lower_bound=None,
                               bound_fraction=None, epsilon=None, upper_lambda1=float(upper_top),
                               actual_lambda1=float(actual_values[0]),
                               compressed_min=None, compressed_resolution_floor=float(c_floor))
                    if rank > len(singular):
                        row.update(actual_ratio=0., lower_bound=0.,
                                   actual_status="structural_zero", bound_status="structural_zero")
                        rows.append(row)
                        continue
                    raw_ratio = actual_values[rank - 1] / actual_values[0]
                    resolved = singular[rank - 1] > singular_threshold
                    row.update(actual_ratio=float(raw_ratio) if resolved else None,
                               actual_ratio_raw=float(raw_ratio),
                               actual_status="resolved" if resolved else "below_resolution")
                    c = c_full[:rank, :rank]
                    t = t_full[:rank, :rank]
                    remainder = symmetric(t - c)
                    eigen_c, vectors_c = la.eigh(c)
                    row["compressed_min"] = float(eigen_c[0])
                    if eigen_c[0] <= c_floor:
                        row["bound_status"] = "compressed_matrix_unresolved"
                        rows.append(row)
                        continue
                    normalized = (vectors_c.T @ remainder @ vectors_c) / np.sqrt(
                        eigen_c[:, None] * eigen_c[None, :])
                    epsilon = float(np.max(np.abs(la.eigvalsh(symmetric(normalized)))))
                    # Independently check the whitening/generalized-eigenvalue
                    # calculation before publishing a positive lower bound.
                    other = float(np.max(np.abs(la.eigvalsh(remainder, c))))
                    row["epsilon"] = epsilon
                    row["epsilon_solver_disagreement"] = abs(epsilon - other)
                    if abs(epsilon - other) > 2e-5 * max(1, epsilon):
                        row["bound_status"] = "correction_calculation_unresolved"
                        rows.append(row)
                        continue
                    lower = max(1 - epsilon, 0) * eigen_c[0] / upper_top
                    compressed_min = la.svdvals(ub[:rank])[-1] ** 2
                    if (lower * upper_top > compressed_min * (1 + 1e-5)
                            or lower > raw_ratio * (1 + 1e-5)):
                        row["bound_status"] = "inequality_numerically_unresolved"
                        rows.append(row)
                        continue
                    row.update(lower_bound=float(lower),
                               bound_status="positive" if lower > 0 else "zero",
                               bound_fraction=float(lower / raw_ratio) if resolved else None)
                    rows.append(row)
        counts = Counter(r["bound_status"] for r in rows if r["N"] == n)
        print(f"N={n}: {dict(counts)}", flush=True)
    return dict(widths=WIDTHS, reference_gammas=REFERENCES, rank_divisors=DIVISORS,
                gammas=GAMMAS.tolist(), geometries=geometries,
                correction="measured from the new finite kernel",
                precision="FP64 with explicit resolution screening; not interval-certified",
                rows=rows)


def plot(result, tightness=False):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, axes = plt.subplots(4, 4, figsize=(17, 13), sharey=True)
    colors = plt.cm.viridis([.06, .34, .64, .9])
    for row_index, reference in enumerate(REFERENCES):
        for col, n in enumerate(WIDTHS):
            ax = axes[row_index, col]
            ranks = [n // divisor for divisor in DIVISORS]
            for index, rank in enumerate(ranks):
                data = [r for r in result["rows"] if r["N"] == n
                        and r["reference_gamma"] == reference and r["rank"] == rank]
                gamma = np.array([r["gamma"] for r in data])
                if all(r["actual_status"] == "structural_zero" for r in data):
                    ax.text(.97, .08, rf"$i={rank}$: structurally zero", transform=ax.transAxes,
                            ha="right", fontsize=8.5, color=".35")
                    continue
                if tightness:
                    fraction = np.array([np.nan if r["bound_fraction"] is None
                                         else r["bound_fraction"] for r in data])
                    ax.plot(gamma, fraction, color=colors[index], lw=1.9)
                else:
                    actual = np.array([np.nan if r["actual_ratio"] is None
                                       else r["actual_ratio"] for r in data])
                    lower = np.array([np.nan if not r["lower_bound"]
                                      else r["lower_bound"] for r in data])
                    ax.plot(gamma, actual, color=colors[index], lw=2)
                    ax.plot(gamma, lower, color=colors[index], ls="--", lw=1.65)
                    # Bottom markers identify zero bounds, not tiny positive values.
                    zeros = np.array([r["bound_status"] == "zero" for r in data])
                    if zeros.any():
                        ax.plot(gamma[zeros], np.full(zeros.sum(), 1.7e-26),
                                color=colors[index], marker="v", ls="none", ms=3.2)
            ax.set_xscale("log", base=2)
            ax.set_xlim(reference, 128)
            ax.set_xticks([g for g in [2, 4, 8, 16, 32, 64, 128] if g >= reference])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            if tightness:
                ax.set_ylim(-.035, 1.04)
                ax.set_yticks([0, .25, .5, .75, 1])
                ax.axhline(1, color=".6", lw=.8, ls=":")
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-26, 1.2)
                ax.set_yticks([1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1])
            ax.grid(True, which="major", alpha=.2)
            if row_index == 0:
                geom = result["geometries"][col]
                rank_text = ", ".join(str(rank) for rank in ranks)
                ax.set_title(rf"$N={n}$" + f"  |  i = {rank_text}\n"
                             f"{geom['tanh_neurons']} tanh neurons; m={geom['m']}", pad=12)
            if col == 0:
                quantity = "Bound / actual" if tightness else r"$\lambda_i(\gamma)/\lambda_1(\gamma)$"
                ax.set_ylabel(rf"Start $\gamma_0={reference:g}$" + "\n" + quantity, labelpad=9)
            if row_index == 3:
                ax.set_xlabel(r"Tanh slope $\gamma$")
    title = "How tight is the lower bound?" if tightness else "Actual eigenvalue ratios and finite-change lower bounds"
    fig.suptitle(title, y=.985, fontsize=18)
    handles = [Line2D([], [], color=color, lw=2.4, label=rf"$i=N/{divisor}$")
               for color, divisor in zip(colors, DIVISORS)]
    if not tightness:
        handles.extend([Line2D([], [], color=".2", lw=2, label="Actual (solid)"),
                        Line2D([], [], color=".2", lw=2, ls="--", label="Lower bound (dashed)")])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .959),
               ncol=len(handles), frameon=False)
    if tightness:
        footer = ("Gaps mean the calculation could not be resolved reliably in FP64. A curve near zero indicates a loose bound.\n"
                  "Colors identify the same fraction of width in every column. Fixed starting subspace per panel.")
    else:
        footer = ("Gaps denote unresolved FP64 calculations. Axis ranges are shared across widths.\n"
                  "Corrections are measured from the finite kernel. Fixed starting subspace per panel; no interval certification.")
    fig.text(.5, .017, footer, ha="center", va="bottom", fontsize=10, color=".3")
    fig.subplots_adjust(left=.085, right=.985, bottom=.092, top=.865, hspace=.20, wspace=.13)
    name = "finite_ratio_width_grid_tightness.png" if tightness else "finite_ratio_width_grid.png"
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result = calculate()
    (OUT / "data/finite_ratio_width_grid.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result)
    plot(result, tightness=True)
    print("Saved both 4 by 4 grids and numerical data.", flush=True)
