"""Checked two-sided ratio intervals and full target-weighted GD error bands.

No optimizer is run. Reuse the saved rank plot, and evaluate the existing
corrected Fourier construction at five slopes for the complete spectral sum.
The FP64 numerical allowance below is checked, not an interval certificate.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                     "axes.titlesize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11.5})

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from direct_ratio_interval import construction, RANKS
from direct_ratio_upper_bound import Geometry, design, OUT
from core import target, weights, hitting_time

DEST = OUT / "note_interval_figures"
GAMMAS = [4., 8., 16., 32., 64.]
RATIO_CUTOFF = 1e-18
ROUNDING_MULTIPLIER = 64.
EPSILON = .01
MAXIMUM = 10**18
COLORS = plt.colormaps["viridis"](np.linspace(.08, .84, len(GAMMAS)))


def guarded_ratios(result, feature_matrix, basis, resolved):
    """Interlacing bounds, widened to guard FP64 corrected-matrix rounding.

    Exact formulas have no nu. Subtracting nu below and adding it above makes
    their numerical evaluation conservative in the observed refinement test.
    The nullspace from the rectangular feature count is aggregated separately.
    """
    beta = result["mu"]
    nu = ROUNDING_MULTIPLIER * np.finfo(float).eps * np.max(np.abs(beta))
    e = np.ones(len(feature_matrix)) / np.sqrt(len(feature_matrix))
    mean = feature_matrix.T @ e
    ell = float(mean @ mean)
    coupling = float(la.norm(basis.T @ feature_matrix @ mean))
    err_down = result["delta"] + result["exterior_tail"]
    err_up = result["delta"] + result["integral_tail"]
    centered_top = beta[0] + err_up + nu
    top = (ell + centered_top + np.hypot(ell-centered_top, 2*coupling))/2
    size = feature_matrix.shape[1]
    # Rank i uses beta_i below and beta_(i-1) above (one-based indices).
    lower = np.r_[1., np.maximum(0., beta[1:size]-err_down-nu)/top]
    upper = np.r_[1., np.minimum(1., np.maximum(0., beta[:size-1]+err_up+nu)/ell)]
    # A numerical singular direction cannot support a positive decay claim.
    lower[~resolved] = 0.
    return lower, upper, dict(numerical_allowance=float(nu),
        top_lower=ell, top_upper=float(top), analytic_lower_error=float(err_down),
        analytic_upper_error=float(err_up), coupling=coupling)


def residual(ratios, p, steps):
    """Stable evaluation even when a rate is much smaller than machine epsilon."""
    return np.sqrt(np.exp(2*np.asarray(steps)[:, None]
                          * np.log1p(-np.asarray(ratios)[None, :]/2)) @ p)


def counts(lower, actual, upper, p, p_lower):
    # Faster allowed rates produce the LOWER residual and NECESSARY count.
    return dict(necessary=hitting_time(upper/2, p_lower, epsilon=EPSILON, maximum=MAXIMUM),
                actual=hitting_time(actual/2, p, epsilon=EPSILON, maximum=MAXIMUM),
                sufficient=hitting_time(lower/2, p, epsilon=EPSILON, maximum=MAXIMUM))


def evaluate(gamma, geometry, steps):
    x, centers = geometry.arrays()
    basis = la.null_space(np.ones((1, len(x))))
    B, _ = design(x, centers, gamma)
    u, s, _ = la.svd(B, full_matrices=False, lapack_driver="gesvd")
    lam = s*s
    rho = lam/lam[0]
    p = weights(u, target(x, "mixed"))
    resolved = rho > RATIO_CUTOFF
    result = construction(x, centers, geometry.h, gamma, basis)
    lower, upper, allowances = guarded_ratios(result, B, basis, resolved)
    assert np.all(lower <= rho)
    assert np.all(rho <= upper)
    assert np.all(lower <= upper)
    assert allowances["top_lower"] <= lam[0] <= allowances["top_upper"]
    # Append the zero eigenvalue of the omitted left-singular-vector subspace.
    actual, low, high = np.r_[rho, 0.], np.r_[lower, 0.], np.r_[upper, 0.]
    p_lower = p.copy()
    p_lower[:-1][~resolved] = 0.
    # The computed null projection can mix with unresolved singular vectors;
    # drop it too instead of claiming a numerically certified lower floor.
    p_lower[-1] = 0.
    errors = dict(lower=residual(high, p_lower, steps),
                  actual=residual(actual, p, steps), upper=residual(low, p, steps))
    assert np.all(errors["lower"] <= errors["actual"] + 2e-15)
    assert np.all(errors["actual"] <= errors["upper"] + 2e-15)
    hits = counts(low, actual, high, p, p_lower)
    assert hits["necessary"] <= hits["actual"] <= hits["sufficient"]

    # Independently change both quadrature resolution and actual-SVD driver.
    refined = construction(x, centers, geometry.h, gamma, basis, order=16, padding=24)
    rl, ru, ra = guarded_ratios(refined, B, basis, resolved)
    assert np.all(rl <= rho) and np.all(rho <= ru)
    rotation = result["integral_basis"].T @ refined["integral_basis"]
    matrix_change = float(la.norm(result["matrix"]
                                  - rotation @ refined["matrix"] @ rotation.T, 2))
    assert matrix_change < allowances["numerical_allowance"]
    u2, s2, _ = la.svd(B, full_matrices=False, lapack_driver="gesdd")
    p2 = weights(u2, target(x, "mixed"))
    rho2 = (s2/s2[0])**2
    assert np.all(lower[resolved] <= rho2[resolved])
    assert np.all(rho2[resolved] <= upper[resolved])
    p2_lower = p2.copy()
    p2_lower[:-1][~resolved] = 0.
    p2_lower[-1] = 0.
    refined_counts = counts(np.r_[rl, 0.], actual, np.r_[ru, 0.], p, p_lower)
    driver_counts = counts(low, np.r_[rho2, 0.], high, p2, p2_lower)
    alternative_errors = dict(lower=residual(np.r_[ru, 0.], p_lower, steps),
                              upper=residual(np.r_[rl, 0.], p, steps))
    validation = dict(quadrature_orders=[10, 16], integration_paddings=[20, 24],
        transported_corrected_matrix_change=matrix_change,
        max_corrected_eigenvalue_change=float(np.max(abs(result["mu"]-refined["mu"]))),
        refined_numerical_allowance=ra["numerical_allowance"],
        resolved_svd_driver_max_relative=float(np.max(abs(rho2[resolved]/rho[resolved]-1))),
        target_weights_driver_l1_difference=float(np.sum(abs(p2-p))),
        computed_null_mass_other_driver=float(p2[-1]),
        aggregate_unresolved_mass_other_driver=float(p2[:-1][~resolved].sum()+p2[-1]),
        refined_counts=refined_counts, independent_svd_counts=driver_counts,
        max_refined_lower_error_difference=float(np.max(abs(errors["lower"]-alternative_errors["lower"]))),
        max_refined_upper_error_difference=float(np.max(abs(errors["upper"]-alternative_errors["upper"]))),
        strict_resolved_ratio_ordering_passed=True,
        strict_all_retained_ratio_ordering_passed=True)
    record = dict(gamma=gamma, counts=hits, allowances=allowances,
        resolved_mode_count=int(resolved.sum()), positive_lower_rate_count=int((lower>0).sum()),
        computed_null_mass=float(p[-1]), numerical_unresolved_mass=float(p[:-1][~resolved].sum()),
        upper_nondecaying_mass=float(p[low==0].sum()),
        upper_residual_floor=float(np.sqrt(p[low==0].sum())),
        lower_initial_residual=float(np.sqrt(p_lower.sum())),
        actual_eigenvalues=lam.tolist(), actual_ratios=actual.tolist(),
        corrected_eigenvalues=result["mu"].tolist(),
        lower_ratios=low.tolist(), upper_ratios=high.tolist(),
        target_weights=p.tolist(), lower_curve_weights=p_lower.tolist(),
        numerically_resolved=np.r_[resolved, False].tolist(),
        relative_error_curves={name: values.tolist() for name, values in errors.items()},
        validation=validation)
    print(json.dumps({key: record[key] for key in ["gamma", "counts", "resolved_mode_count",
          "positive_lower_rate_count", "computed_null_mass", "numerical_unresolved_mass",
          "upper_residual_floor", "validation"]}), flush=True)
    return record


def styles():
    return [Line2D([0], [0], color="black", ls=ls, lw=1.6, label=name)
            for ls, name in [("--", "Lower bound"), ("-", "Actual"), (":", "Upper bound")]]


def plot_ratios(saved):
    records = saved["records"]
    gamma = [r["gamma"] for r in records]
    colors = plt.colormaps["viridis"](np.linspace(.08, .84, len(RANKS)))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for color, rank in zip(colors, RANKS):
        rows = [next(item for item in r["ranks"] if item["rank"] == rank) for r in records]
        low, actual, high = [np.array([item[key] for item in rows])
                             for key in ["lower", "actual", "upper"]]
        axes[0].fill_between(gamma, low, high, color=color, alpha=.10)
        for values, ls in [(low, "--"), (actual, "-"), (high, ":")]:
            axes[0].plot(gamma, values, ls, color=color, lw=1.65)
        axes[1].plot(gamma, low/actual, "--", color=color, lw=1.65)
        axes[1].plot(gamma, high/actual, ":", color=color, lw=1.65)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(GAMMAS, ["4", "8", "16", "32", "64"])
        ax.set_xlim(4, 64)
        ax.set_xlabel(r"Common slope $\gamma$")
        ax.grid(alpha=.18)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-16, .1)
    axes[0].set_ylabel(r"Eigenvalue ratio $\rho_i=\lambda_i/\lambda_1$")
    axes[0].set_title("Two-sided ratio intervals", pad=10)
    axes[1].axhline(1., color="0.5", lw=.8)
    axes[1].set_ylim(0., 1.8)
    axes[1].set_ylabel("Bound / actual ratio")
    axes[1].set_title("Both sides remain informative", pad=10)
    handles = [Line2D([0], [0], color=color, label=f"Rank {rank}")
               for color, rank in zip(colors, RANKS)]
    fig.legend(handles=handles+styles(), loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=7, frameon=False, fontsize=11.5, columnspacing=1.25)
    fig.suptitle("Finite tanh network: corrected Fourier intervals", y=.995, fontsize=14)
    fig.subplots_adjust(left=.085, right=.985, bottom=.15, top=.745, wspace=.29)
    fig.savefig(DEST / "ratio_intervals.png", dpi=220)
    plt.close(fig)


def plot_error_time(records, steps):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    for color, row in zip(COLORS, records):
        errors = row["relative_error_curves"]
        axes[0].fill_between(steps[1:], errors["lower"][1:], errors["upper"][1:],
                             color=color, alpha=.09)
        for name, ls in [("lower", "--"), ("actual", "-"), ("upper", ":")]:
            axes[0].plot(steps[1:], errors[name][1:], ls, color=color, lw=1.6)
    axes[0].axhline(EPSILON, color="0.4", ls="-.", lw=.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(1, 1e15)
    axes[0].set_ylim(1e-5, 1.1)
    axes[0].set_xlabel("GD steps (spectral calculation)")
    axes[0].set_ylabel("Relative training residual")
    axes[0].set_title("Full target-weighted residual; line = 1%", fontsize=13, pad=12)
    g = [row["gamma"] for row in records]
    lo, actual, hi = [np.array([row["counts"][key] for row in records], dtype=float)
                      for key in ["necessary", "actual", "sufficient"]]
    axes[1].fill_between(g, lo, hi, color="0.6", alpha=.16)
    for values, ls in [(lo, "--"), (actual, "-"), (hi, ":")]:
        axes[1].plot(g, values, ls, color="0.18", lw=1.6)
        axes[1].scatter(g, values, c=COLORS, s=27, zorder=3)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(GAMMAS, ["4", "8", "16", "32", "64"])
    axes[1].set_xlim(3.7, 69.)
    axes[1].set_ylim(1e4, 1e13)
    axes[1].set_xlabel(r"Common slope $\gamma$")
    axes[1].set_ylabel("Steps to 1% relative training residual")
    axes[1].set_title("Necessary ≤ actual ≤ sufficient steps", fontsize=13, pad=12)
    for ax in axes:
        ax.grid(alpha=.18)
    handles = [Line2D([0], [0], color=color, label=rf"$\gamma={gamma:g}$")
               for color, gamma in zip(COLORS, GAMMAS)]
    fig.legend(handles=handles+styles(), loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=8, frameon=False, fontsize=11.5, columnspacing=1.25)
    fig.suptitle(r"Both ratio bounds enter the full error sum: $\eta=1/(2\lambda_1)$",
                 y=.995, fontsize=14)
    fig.subplots_adjust(left=.08, right=.985, bottom=.145, top=.745, wspace=.32)
    fig.savefig(DEST / "full_residual_and_steps.png", dpi=220)
    plt.close(fig)


def write_manifest(records):
    lines = ["# Figures for the two-sided gamma ratio note", "",
        "These are spectral calculations, not training runs. Geometry: 263 equally spaced samples on [-1,1], 153 tanh centers including the existing halo, spacing h=1/64, and one bias feature. The target is exactly core.target(x, 'mixed'): sin(2πx) + 0.5 sin(6πx) + 0.25 sin(10πx).", "",
        "- `ratio_intervals.png`: replot of the saved direct_ratio_interval/data.json at ranks 13, 20, 26, and 33, over all 33 saved slopes. Actual is solid, lower dashed, upper dotted. This plot retains the original unpadded FP64 evaluations of the analytic intervals.",
        "- `full_residual_and_steps.png`: complete target-weighted relative residual and necessary/actual/sufficient count to 1%. Uses newly computed full corrected S spectra at gamma 4, 8, 16, 32, 64; it includes a numerical allowance explained below.",
        "- `data.json`: all corrected eigenvalues, finite-feature SVD eigenvalues, ratios, target weights, computed null mass, unresolved mass, residual arrays, count intervals, and validation statistics. Only PNG figures are generated.", "",
        "The finite-feature SVD supplies actual target weights p_i = |u_i^T y|² / ||y||² and actual ratios. The Fourier bounds do not replace these weights. The lower error uses upper ratio bounds, and the upper error uses lower ratio bounds. All curves use exp(2 n log1p(-rho/2)); no iteration is simulated. Integer crossing counts use monotone bisection, with a search limit of 10^18 steps.", "",
        "## Analytic bounds and numerical guard", "",
        "The existing construction gives beta_i, the descending eigenvalues of corrected S. Before the numerical allowance, the ratio lower numerator is [beta_i-delta-tau]_+ and the upper numerator is beta_(i-1)+delta+integral_tail, with the existing scalar normalizations L and ell. rho_1=1 is exact. Interlacing is applied only to valid indices. There are at most 154 nonzero kernel eigenvalues; the remaining 109 structural sample-space zero modes are represented by the appended computed null component. The final eigenvalue lower bound is zero.", "",
        "For the residual calculation, nu=64*eps64*||S||_2 is subtracted in every lower numerator and added in every upper numerator and in the centered top-eigenvalue allowance used for L. This is a conservative numerical guard checked against independent quadrature, not a rigorous interval certificate. It prevents tiny spurious positive eigenvalues of the corrected matrix from being treated as trustworthy lower rates. The analytic tails alone do not bound FP64 rounding.", "",
        "Actual ratios greater than 10^-18 define numerically resolved SVD modes. Unresolved positive-mode mass is never silently treated as exact nullspace. The lower residual drops its nonnegative contribution and also drops the computed null component, whose numerical allocation can mix with unresolved tiny singular directions. The upper residual keeps their full energy at zero assigned lower rate. It also keeps every other mode whose guarded lower ratio is zero. The actual spectral calculation retains the computed null component at zero rate; this mass is reported separately from unresolved positive-mode energy. Because near-zero singular vectors are not individually stable, the aggregate unresolved-plus-null mass is the more meaningful numerical diagnostic; these plots do not certify the true asymptotic null floor.", "",
        "## Count table", "", "| Gamma | Necessary | Actual spectral | Sufficient | Upper-error floor |",
        "|---:|---:|---:|---:|---:|"]
    for row in records:
        c = row["counts"]
        lines.append(f"| {row['gamma']:g} | {c['necessary']:,} | {c['actual']:,} | {c['sufficient']:,} | {row['upper_residual_floor']:.6g} |")
    lines += ["", "All five sufficient curves cross 1%; no unavailable count was replaced by a finite estimate. The lower-error initial value can be slightly below one because unresolved positive mass and computed null mass are dropped conservatively.", "",
        "## Validation", "",
        "At every plotted slope, including gamma 4 and 8, refine the existing center quadrature from order 10/padding 20 to order 16/padding 24. Compare the full corrected matrices after orthogonal change of basis. The observed operator discrepancy is below nu. Recompute finite-feature SVD using both LAPACK gesvd and gesdd, compare resolved ratios and weights, and recompute crossings and bands. The guarded lower <= actual <= upper ordering passes strict floating-point comparisons on every resolved mode, also using the alternate SVD driver. Full retained-mode ordering passes with the primary SVD. The full error curves and count intervals also pass ordering checks. The exact count digits describe the current FP64 calculation; use about three significant figures in explanatory prose.", "",
        "Run from the repository root with `.venv/bin/python experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/note_interval_figures.py`. Existing saved figures and reports are left intact.", ""]
    (DEST / "MANIFEST.md").write_text("\n".join(lines))


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    saved = json.loads((OUT / "direct_ratio_interval/data.json").read_text())
    steps = np.r_[0., np.geomspace(1., 1e15, 451)]
    records = [evaluate(gamma, Geometry(), steps) for gamma in GAMMAS]
    output = dict(geometry=saved["geometry"], target="mixed", epsilon=EPSILON,
        eta="0.5 / lambda_1", ratio_resolution_cutoff=RATIO_CUTOFF,
        rounding_allowance="64 * eps64 * ||S||_2", steps=steps.tolist(),
        records=records, saved_ratio_interval_source="../direct_ratio_interval/data.json",
        scope="Checked FP64 numerical evaluations of analytic inequalities, not interval certificates; no training")
    (DEST / "data.json").write_text(json.dumps(output, separators=(",", ":")) + "\n")
    plot_ratios(saved)
    plot_error_time(records, steps)
    write_manifest(records)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
