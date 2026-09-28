"""Nontrivial gamma transitions: doubling curves and two-gamma heatmaps.

All spectra are computed directly; no training or curve fitting. The new
kernel is used for the measured correction and row-sum denominator bound.
Unresolved initial singular subspaces are explicitly excluded from bounds.
"""
import json
from collections import Counter

import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

from finite_ratio_width_grid import WIDTHS, SAMPLES, DIVISORS, main_matrix
from finite_ratio_bound import OUT, design, symmetric
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter

EPS = np.finfo(float).eps
GAMMAS = 2. ** np.linspace(1, 7, 61)
HEAT_STARTS = [2., 4., 8., 16., 32., 64.]
HEAT_ENDS = [4., 8., 16., 32., 64., 128.]
COLORS = plt.cm.viridis([.06, .34, .64, .9])


def calculate():
    records, geometries = [], []
    for n, m in zip(WIDTHS, SAMPLES):
        ranks = [n // d for d in DIVISORS]
        h = 2 / n
        halo = int(np.ceil(np.sqrt(n)))
        centers = -1 + np.arange(-halo, n + halo + 1) * h
        x = np.linspace(-1, 1, m)
        geometries.append(dict(N=n, m=m, h=h, halo_per_side=halo,
                               tanh_neurons=len(centers), ranks=ranks))
        for start_index in range(51):
            ga = float(GAMMAS[start_index])
            target_indices = {start_index + 10}
            if start_index % 10 == 0:
                target_indices.update(range(start_index + 10, 61, 10))
            ba, _ = design(x, centers, ga)
            ua, sa, _ = la.svd(ba, full_matrices=False, lapack_driver="gesvd")
            ua = ua[:, :max(ranks)]
            uaba = ua.T @ ba
            base = symmetric(uaba @ uaba.T)
            main_a = main_matrix(x, h, len(centers), ga)
            start_threshold = 10 * EPS * max(ba.shape) * sa[0]
            for target_index in sorted(target_indices):
                gb = float(GAMMAS[target_index])
                bb, _ = design(x, centers, gb)
                sb = la.svdvals(bb)
                kb = bb @ bb.T
                upper = float(np.max(np.sum(np.abs(kb), axis=1)))
                assert upper >= sb[0] ** 2 * (1 - 1e-12)
                delta = main_matrix(x, h, len(centers), gb) - main_a
                c_full = symmetric(base + ua.T @ delta @ ua)
                uabb = ua.T @ bb
                t_full = symmetric(uabb @ uabb.T)
                c_floor = 64 * EPS * (la.norm(base, 2) + la.norm(delta, "fro"))
                final_threshold = 10 * EPS * max(bb.shape) * sb[0]
                for rank, divisor in zip(ranks, DIVISORS):
                    start_ratio = float((sa[rank - 1] / sa[0]) ** 2)
                    final_ratio = float((sb[rank - 1] / sb[0]) ** 2)
                    start_gap = float(sa[rank - 1] - sa[rank])
                    start_resolved = (sa[rank - 1] > start_threshold and start_gap > start_threshold)
                    final_resolved = sb[rank - 1] > final_threshold
                    row = dict(N=n, m=m, rank=rank, divisor=divisor,
                               start_gamma=ga, final_gamma=gb,
                               doubling=target_index == start_index + 10,
                               heatmap=start_index % 10 == 0 and target_index % 10 == 0,
                               start_ratio=start_ratio, actual_ratio=final_ratio if final_resolved else None,
                               actual_ratio_raw=final_ratio, start_resolved=bool(start_resolved),
                               start_singular_ratio=float(sa[rank - 1] / sa[0]),
                               start_gap_ratio=start_gap / float(sa[0]),
                               start_resolution_threshold=start_threshold / float(sa[0]),
                               final_resolved=bool(final_resolved), lower_bound=None,
                               bound_fraction=None, improvement_bound=None, epsilon=None,
                               upper_lambda1=upper, actual_lambda1=float(sb[0] ** 2))
                    if not start_resolved:
                        row["status"] = "starting_subspace_unresolved"
                        records.append(row)
                        continue
                    if not final_resolved:
                        row["status"] = "final_spectrum_unresolved"
                        records.append(row)
                        continue
                    c = c_full[:rank, :rank]
                    t = t_full[:rank, :rank]
                    remainder = symmetric(t - c)
                    cv, cq = la.eigh(c)
                    row.update(compressed_min=float(cv[0]), compressed_floor=float(c_floor))
                    if cv[0] <= c_floor:
                        row["status"] = "compressed_matrix_unresolved"
                        records.append(row)
                        continue
                    scaled = (cq.T @ remainder @ cq) / np.sqrt(cv[:, None] * cv[None, :])
                    epsilon = float(max(abs(la.eigvalsh(symmetric(scaled)))))
                    independent = float(max(abs(la.eigvalsh(remainder, c))))
                    row.update(epsilon=epsilon, correction_solver_difference=abs(epsilon - independent))
                    if abs(epsilon - independent) > 2e-5 * max(1, epsilon):
                        row["status"] = "correction_unresolved"
                        records.append(row)
                        continue
                    lower = float(max(1 - epsilon, 0) * cv[0] / upper)
                    compressed_actual_min = float(la.svdvals(uabb[:rank])[-1] ** 2)
                    # Checks assess numerical evaluation, not a rounding enclosure.
                    if (lower * upper > compressed_actual_min * (1 + 1e-5)
                            or lower > final_ratio * (1 + 1e-5)):
                        row["status"] = "inequality_unresolved"
                        records.append(row)
                        continue
                    row.update(lower_bound=lower, bound_fraction=lower / final_ratio,
                               improvement_bound=lower / start_ratio,
                               compressed_actual_min=compressed_actual_min,
                               status="positive" if lower > 0 else "zero")
                    records.append(row)
        print(f"N={n}: {dict(Counter(r['status'] for r in records if r['N'] == n))}", flush=True)
    return dict(widths=WIDTHS, samples=SAMPLES, rank_divisors=DIVISORS,
                heatmap_starts=HEAT_STARTS, heatmap_finals=HEAT_ENDS,
                geometries=geometries, records=records,
                scope="FP64 evaluation of measured-correction theorem bounds; no interval certification",
                resolution="initial singular value and cutoff gap must both exceed 10*eps*max(shape)*s1",
                prediction_inputs="initial subspace, spatial gamma difference, measured new-kernel correction, row-sum bound")


def plot_doubling(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, n in enumerate(WIDTHS):
        for divisor, color in zip(DIVISORS, COLORS):
            rows = [r for r in data["records"] if r["N"] == n and r["divisor"] == divisor and r["doubling"]]
            gamma = np.array([r["start_gamma"] for r in rows])
            actual = np.array([np.nan if r["actual_ratio"] is None else r["actual_ratio"] for r in rows])
            lower = np.array([np.nan if not r["lower_bound"] else r["lower_bound"] for r in rows])
            fraction = np.array([np.nan if r["bound_fraction"] is None else r["bound_fraction"] for r in rows])
            axes[0, col].plot(gamma, actual, color=color, lw=1.8, marker=".", ms=3.3)
            axes[0, col].plot(gamma, lower, color=color, lw=1.6, ls="--", marker=".", ms=2.8)
            axes[1, col].plot(gamma, fraction, color=color, lw=1.8, marker=".", ms=3.3)
        axes[0, col].set_title(rf"$N={n}$" + "\n" + "i = " + ", ".join(str(n // d) for d in DIVISORS), pad=12)
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylim(1e-26, 1.2)
        axes[0, col].set_yticks([1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1])
        axes[1, col].set_ylim(0, 1.04)
        axes[1, col].set_yticks([0, .25, .5, .75, 1])
        axes[1, col].axhline(1, color=".6", ls=":", lw=.8)
        axes[1, col].set_xlabel(r"Starting slope $\gamma_a$ (final slope is $2\gamma_a$)", fontsize=10)
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_xlim(2, 64)
            ax.set_xticks([2, 4, 8, 16, 32, 64])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, alpha=.20)
        if col:
            axes[0, col].tick_params(labelleft=False)
            axes[1, col].tick_params(labelleft=False)
    axes[0, 0].set_ylabel(r"Final ratio $\lambda_i(2\gamma_a)/\lambda_1(2\gamma_a)$")
    axes[1, 0].set_ylabel("Lower bound / actual")
    fig.suptitle("Doubling gamma: actual eigenvalue ratios versus lower bounds", y=.98, fontsize=17)
    handles = [Line2D([], [], color=c, lw=2.4, label=rf"$i=N/{d}$") for c, d in zip(COLORS, DIVISORS)]
    handles += [Line2D([], [], color=".2", lw=2, label="Actual (solid)"),
                Line2D([], [], color=".2", lw=2, ls="--", label="Lower bound (dashed)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .939), ncol=6, frameon=False)
    fig.text(.5, .022, "Dots are direct numerical evaluations; lines only connect them. Missing bounds indicate unresolved calculations.\n"
             "The starting eigenspace is recomputed for each starting gamma. Corrections use the finite kernel; no interval certification.",
             ha="center", fontsize=10, color=".3")
    fig.subplots_adjust(left=.075, right=.985, top=.80, bottom=.145, hspace=.23, wspace=.14)
    fig.savefig(OUT / "finite_ratio_doubling.png", dpi=185)
    plt.close(fig)


def format_gain(x):
    if x == 0:
        return "0"
    if .9 <= x <= 1.1:
        return f"{x:.3f}"
    if .1 <= x < 100:
        return f"{x:.2g}"
    return f"{x:.0e}"


def plot_heatmap(data, gain=False):
    fig, axes = plt.subplots(4, 4, figsize=(17, 14), sharex=True, sharey=True)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("white")
    if gain:
        positives = [r["improvement_bound"] for r in data["records"]
                     if r["heatmap"] and r["improvement_bound"] is not None and r["improvement_bound"] > 0]
        vmax = 10. ** np.ceil(np.log10(max(positives)))
        norm = LogNorm(vmin=.1, vmax=vmax, clip=True)
    else:
        norm = Normalize(vmin=0, vmax=1)
    starts, ends = np.array(HEAT_STARTS), np.array(HEAT_ENDS)
    xedges = np.r_[starts / np.sqrt(2), starts[-1] * np.sqrt(2)]
    yedges = np.r_[ends / np.sqrt(2), ends[-1] * np.sqrt(2)]
    for row_index, divisor in enumerate(DIVISORS):
        for col, n in enumerate(WIDTHS):
            ax = axes[row_index, col]
            values = np.full((len(ends), len(starts)), np.nan)
            records = {}
            for r in data["records"]:
                if r["N"] == n and r["divisor"] == divisor and r["heatmap"]:
                    a = int(np.argmin(abs(starts - r["start_gamma"])))
                    b = int(np.argmin(abs(ends - r["final_gamma"])))
                    records[b, a] = r
                    metric = r["improvement_bound"] if gain else r["bound_fraction"]
                    if metric is not None:
                        values[b, a] = max(metric, norm.vmin) if gain else metric
            mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(values),
                                 cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=.55)
            for (b, a), r in records.items():
                metric = r["improvement_bound"] if gain else r["bound_fraction"]
                if metric is None:
                    ax.add_patch(Rectangle((xedges[a], yedges[b]), xedges[a + 1] - xedges[a],
                                           yedges[b + 1] - yedges[b], facecolor="#d4d4d4", edgecolor="white", lw=.55))
                    continue
                label = format_gain(metric) if gain else (f"{metric:.2f}" if metric >= .01 else f"{metric:.0e}")
                position = norm(max(metric, norm.vmin)) if gain else norm(metric)
                ax.text(starts[a], ends[b], label, ha="center", va="center",
                        color="white" if position < .48 else "black", fontsize=8.4,
                        fontweight="bold" if gain and metric > 1 else "normal")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks(starts)
            ax.set_yticks(ends)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlim(xedges[0], xedges[-1])
            ax.set_ylim(yedges[0], yedges[-1])
            ax.tick_params(labelsize=9)
            ax.text(.97, .07, rf"$i={n // divisor}$", transform=ax.transAxes, ha="right", fontsize=11, color=".25")
            if row_index == 0:
                ax.set_title(rf"$N={n}$", pad=10, fontsize=14)
            if row_index == 3:
                ax.set_xlabel(r"Starting $\gamma_a$", fontsize=11)
            if col == 0:
                ax.set_ylabel(rf"$i=N/{divisor}$" + "\n" + r"Final $\gamma_b$", fontsize=11)
    title = "How much improvement does the lower bound establish?" if gain else "How close is the lower bound to the actual final ratio?"
    fig.suptitle(title, y=.985, fontsize=18)
    subtitle = (r"Cell = lower bound at $\gamma_b$ / actual ratio at $\gamma_a$; values above 1 establish improvement"
                if gain else r"Cell = lower bound at $\gamma_b$ / actual ratio at $\gamma_b$; 1 means exact")
    fig.text(.46, .947, subtitle, ha="center", fontsize=12)
    fig.legend(handles=[Patch(facecolor="#d4d4d4", label="Numerically unresolved"),
                        Patch(facecolor="white", edgecolor=".65", label=r"Not tested: $\gamma_b<2\gamma_a$")],
               loc="upper center", bbox_to_anchor=(.46, .928), ncol=2, frameon=False)
    fig.subplots_adjust(left=.075, right=.90, top=.865, bottom=.08, hspace=.10, wspace=.08)
    cax = fig.add_axes([.923, .18, .014, .58])
    cbar = fig.colorbar(mesh, cax=cax)
    if gain:
        exponents = np.unique(np.r_[-1, 0, np.arange(2, int(np.log10(norm.vmax)) + 1, 2)])
        cbar.set_ticks(10. ** exponents)
        cbar.set_label("Lower bound / starting ratio (logarithmic color scale)", labelpad=12)
    else:
        cbar.set_ticks([0, .25, .5, .75, 1])
        cbar.set_label("Lower bound / actual final ratio", labelpad=12)
    fig.text(.48, .018, "Fixed uniform geometry within each panel; only gamma changes. Corrections are measured from the finite kernel.\n"
             "Unresolved starting eigenspaces are excluded. FP64 evaluations, without interval certification.",
             ha="center", fontsize=10, color=".3")
    name = "finite_ratio_transition_gain.png" if gain else "finite_ratio_transition_tightness.png"
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result = calculate()
    (OUT / "data/finite_ratio_transitions.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_doubling(result)
    plot_heatmap(result)
    plot_heatmap(result, gain=True)
    print("Saved doubling figure, two heatmaps, and raw numerical data.", flush=True)
