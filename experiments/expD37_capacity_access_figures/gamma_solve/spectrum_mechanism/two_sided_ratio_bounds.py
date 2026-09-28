"""Two-sided finite-gamma ratio bounds from the same main/correction split.

The upper numerator uses a full sample-space complement of the old leading
rank-(i-1) trial space. It bounds the main and correction pieces separately;
diagonalizing the new kernel or its compression is only a validation check.
All results are FP64 evaluations of exact inequalities, not interval bounds.
"""
import csv
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
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

from finite_ratio_bound import main_kernel, symmetric
from run import Geometry, OUT, design

DEST = OUT / "two_sided_ratio_bounds"
EPS = np.finfo(float).eps
REFERENCES = [8.0, 64.0]
RANKS = [12, 20, 32]
COLORS = {"actual": "#31688e", "lower": "#258a62", "upper": "#b35b13"}


def largest(matrix):
    return float(la.eigvalsh(matrix, subset_by_index=[len(matrix)-1]*2)[0])


def fourier_check(geom, old_gamma, new_gamma, q):
    """Independent scalar quadrature of the signed all-vector identity."""
    x, _ = geom.arrays()
    main_value = float(q @ (main_kernel(geom, new_gamma)
                           - main_kernel(geom, old_gamma)) @ q)

    def multiplier_squared(gamma, omega):
        z = np.pi * omega / (2 * gamma)
        if z < 1e-3:
            return 1 - z*z/3 + z**4/15 - 2*z**6/189
        return (2*z*np.exp(-z)/(-np.expm1(-2*z)))**2

    def integrand(omega):
        if omega < 1e-3 * min(old_gamma, new_gamma):
            # Difference series avoids subtracting two nearly unit values.
            a, b = np.pi/(2*old_gamma), np.pi/(2*new_gamma)
            density = ((a*a-b*b)/3 + (b**4-a**4)*omega**2/15
                       - 2*(b**6-a**6)*omega**4/189)
        else:
            density = (multiplier_squared(new_gamma, omega)
                       - multiplier_squared(old_gamma, omega)) / omega**2
        amplitude = abs(np.exp(-1j * omega*x) @ q)**2
        return density * amplitude

    cutoff = 30 * max(old_gamma, new_gamma)
    integral, estimate = quad(integrand, 0, cutoff, epsabs=1e-11,
                              epsrel=2e-10, limit=1000)
    fourier_value = 4 / (np.pi*geom.h*geom.m) * integral
    # |sum q_a exp(-i w x_a)|^2 <= m for a unit q. This tail estimate
    # bounds both multipliers separately beyond the finite cutoff.
    tail = sum(16 / (np.pi*geom.h) * (np.pi/(2*g))**2
               * np.exp(-np.pi*cutoff/g) / (np.pi/g)
               / (1-np.exp(-np.pi*cutoff/g))**2
               for g in [old_gamma, new_gamma])
    return dict(reference_gamma=old_gamma, gamma=new_gamma,
                spatial_value=main_value, fourier_value=fourier_value,
                absolute_difference=abs(main_value-fourier_value),
                quadrature_error_estimate=4/(np.pi*geom.h*geom.m)*estimate,
                analytic_tail_upper=tail)


def calculate():
    geom = Geometry()
    x, centers = geom.arrays()
    gammas = np.unique(np.round(np.r_[np.geomspace(4, 64, 65), [4, 8, 16, 32, 64]], 12))
    snapshots = {}
    for gamma in np.unique(np.r_[gammas, 2.0]):
        b, _ = design(x, centers, gamma)
        # Actual small eigenvalues come from rectangular SVD, not a Gram solve.
        s = la.svdvals(b)
        snapshots[float(gamma)] = dict(b=b, eigenvalues=s*s,
                                      main=main_kernel(geom, gamma))

    rows, validations, full_spectra = [], [], []
    for gamma in gammas:
        ev = snapshots[float(gamma)]["eigenvalues"]
        full_spectra.append(np.r_[ev/ev[0], np.zeros(geom.m-len(ev))])
    cases = [(reference, gammas, RANKS) for reference in REFERENCES]
    cases += [(2.0, [4.0], [12, 20, 26, 32]),
              (4.0, [8.0], [12, 20, 26, 32])]
    for reference, new_gammas, ranks in cases:
        old = snapshots[reference]
        # full_matrices=True is essential: the thin SVD omits m-(W+1)
        # sample nullspace vectors that belong in every trailing trial space.
        u_all, old_s, _ = la.svd(old["b"], full_matrices=True,
                                  lapack_driver="gesvd")
        assert u_all.shape == (geom.m, geom.m)
        orthogonality = la.norm(u_all.T @ u_all-np.eye(geom.m), 2)
        for gamma in new_gammas:
            new = snapshots[float(gamma)]
            b, ev = new["b"], new["eigenvalues"]
            k = b @ b.T
            delta = new["main"] - old["main"]
            upper_top = float(np.max(np.sum(np.abs(k), axis=1)))
            # One fixed unit trial vector yields a LOWER bound on lambda_1.
            lower_top = float(la.norm(b.T @ u_all[:, 0])**2)
            assert 0 < lower_top <= ev[0]*(1+5e-12)
            assert upper_top >= ev[0]*(1-5e-12)
            resolution_floor = 64*EPS*(old_s[0]**2 + la.norm(delta, "fro"))

            for rank in ranks:
                leading, trailing = u_all[:, :rank], u_all[:, rank-1:]
                lead_old, lead_new = leading.T @ old["b"], leading.T @ b
                # Projected old baseline keeps the theorem valid for any
                # numerically computed orthonormal trial basis, regardless of
                # whether individual old singular vectors are resolved.
                c_lead = symmetric(lead_old @ lead_old.T
                                   + leading.T @ delta @ leading)
                r_lead = symmetric(lead_new @ lead_new.T-c_lead)
                c_values, c_vectors = la.eigh(c_lead)
                epsilon = None
                epsilon_disagreement = None
                lower = 0.0
                if c_values[0] <= 0:
                    lower_status = "main_compression_not_positive_definite"
                elif c_values[0] <= resolution_floor:
                    lower_status = "main_compression_unresolved_in_fp64"
                else:
                    relative = symmetric((c_vectors.T @ r_lead @ c_vectors)
                                         / np.sqrt(c_values[:, None]*c_values[None, :]))
                    epsilon = float(np.max(np.abs(la.eigvalsh(relative))))
                    check = float(np.max(np.abs(la.eigvalsh(r_lead, c_lead))))
                    epsilon_disagreement = abs(epsilon-check)
                    assert epsilon_disagreement <= 2e-5*max(1, epsilon)
                    lower = max(1-epsilon, 0)*c_values[0]/upper_top
                    lower_status = "positive" if lower > 0 else "relative_correction_at_least_one"

                tail_old, tail_new = trailing.T @ old["b"], trailing.T @ b
                c_tail = symmetric(tail_old @ tail_old.T
                                   + trailing.T @ delta @ trailing)
                r_tail = symmetric(tail_new @ tail_new.T-c_tail)
                main_max, remainder_max = largest(c_tail), largest(r_tail)
                upper_uncapped = (main_max+remainder_max)/lower_top
                assert upper_uncapped > 0
                upper = min(1.0, upper_uncapped)

                # The quantities below are validation diagnostics only. They
                # do not enter either bound and are not predictive claims.
                actual_ratio = float(ev[rank-1]/ev[0])
                lead_actual_min = float(la.svdvals(lead_new)[-1]**2)
                tail_actual_max = float(la.svdvals(tail_new)[0]**2)
                tolerance = 2e-11*ev[0]
                assert lower*upper_top <= lead_actual_min+tolerance
                assert lead_actual_min <= ev[rank-1]+tolerance
                assert ev[rank-1] <= tail_actual_max+tolerance
                assert tail_actual_max <= main_max+remainder_max+tolerance
                assert lower <= actual_ratio*(1+2e-5)
                assert actual_ratio <= upper*(1+2e-5)
                rows.append(dict(
                    reference_gamma=reference, gamma=float(gamma), rank=rank,
                    direction="increase" if gamma > reference else
                              "decrease" if gamma < reference else "reference",
                    actual_ratio=actual_ratio, lower_bound=float(lower), upper_bound=upper,
                    lower_over_actual=float(lower/actual_ratio),
                    upper_over_actual=float(upper/actual_ratio),
                    lower_status=lower_status, leading_main_min=float(c_values[0]),
                    epsilon=epsilon, epsilon_solver_difference=epsilon_disagreement,
                    upper_lambda1=upper_top, lower_lambda1=lower_top,
                    actual_lambda1=float(ev[0]), upper_uncapped=upper_uncapped,
                    trailing_dimension=trailing.shape[1],
                    omitted_thin_svd_nullspace_dimension=geom.m-len(old_s),
                    trailing_main_max=main_max, trailing_correction_max=remainder_max,
                    validation_only_leading_min=lead_actual_min,
                    validation_only_trailing_max=tail_actual_max,
                    validation_only_trailing_ratio=tail_actual_max/lower_top,
                    validation_only_subspace_factor=tail_actual_max/ev[rank-1],
                    validation_only_split_factor=(main_max+remainder_max)/tail_actual_max,
                    denominator_factor=float(ev[0]/lower_top),
                    relative_subspace_orthogonality_error=float(orthogonality),
                    leading_resolution_floor=float(resolution_floor)))
        # Check both signs using nonzero-mean vectors; the identity does not
        # require a zero-sum restriction.
        for gamma in ([4.0, 16.0, 64.0] if reference in REFERENCES else []):
            if gamma != reference:
                record = fourier_check(geom, reference, gamma,
                                       (u_all[:, 0]+u_all[:, 11])/np.sqrt(2))
                assert record["absolute_difference"] < 2e-9
                validations.append(record)

    return dict(
        geometry=dict(intervals=geom.N, tanh_neurons=len(centers), samples=geom.m,
                      halo_per_side=geom.halo, spacing=geom.h,
                      integration_bounds=list(geom.bounds)),
        references=REFERENCES, ranks=RANKS, gammas=gammas.tolist(),
        precision="FP64 evaluation of exact inequalities; no interval certification",
        correction="Measured from the new finite feature kernel; no fitted parameters",
        validation=dict(bound_inequalities_passed=True, fourier_checks=validations,
                        max_orthogonality_error=max(r["relative_subspace_orthogonality_error"] for r in rows),
                        max_relative_norm_solver_difference=max(r["epsilon_solver_difference"] or 0 for r in rows)),
        rows=[r for r in rows if r["reference_gamma"] in REFERENCES],
        forward_pair_rows=[r for r in rows if r["reference_gamma"] not in REFERENCES]), np.asarray(full_spectra)


def forward_driver_check():
    """Sensitivity of small-gamma upper bounds to the reference SVD driver."""
    geom = Geometry()
    x, centers = geom.arrays()
    old, _ = design(x, centers, 2.0)
    new, _ = design(x, centers, 4.0)
    delta = main_kernel(geom, 4.0)-main_kernel(geom, 2.0)
    records = []
    for driver in ["gesvd", "gesdd"]:
        u, s, _ = la.svd(old, full_matrices=True, lapack_driver=driver)
        denominator = float(la.norm(new.T @ u[:, 0])**2)
        for rank in [20, 26, 32]:
            v = u[:, rank-1:]
            a, b = v.T @ old, v.T @ new
            c = symmetric(a @ a.T+v.T @ delta @ v)
            r = symmetric(b @ b.T-c)
            records.append(dict(reference_gamma=2.0, gamma=4.0, rank=rank,
                                driver=driver, upper_bound=(largest(c)+largest(r))/denominator,
                                old_sigma=float(s[rank-1]),
                                old_sigma_over_sigma1=float(s[rank-1]/s[0])))
    return records


def plot(result):
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12,
                         "axes.labelsize": 11})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharex=True, sharey="col")
    for col, rank in enumerate(RANKS):
        all_rank = [r for r in result["rows"] if r["rank"] == rank]
        floor = min(r["actual_ratio"] for r in all_rank)/6
        top = max(r["upper_bound"] for r in all_rank)*2.5
        for row_index, reference in enumerate(REFERENCES):
            ax = axes[row_index, col]
            records = [r for r in all_rank if r["reference_gamma"] == reference]
            gamma = np.array([r["gamma"] for r in records])
            actual = np.array([r["actual_ratio"] for r in records])
            lower = np.array([r["lower_bound"] for r in records])
            upper = np.array([r["upper_bound"] for r in records])
            ax.plot(gamma, actual, color=COLORS["actual"], lw=2.6, zorder=3)
            ax.plot(gamma, np.where(lower > 0, lower, np.nan),
                    color=COLORS["lower"], lw=2.2, ls="--", zorder=4)
            ax.plot(gamma, upper, color=COLORS["upper"], lw=2.7, ls=":", zorder=5)
            # Log axes cannot display zero; explicit markers reveal all gaps.
            ax.scatter(gamma[lower == 0], np.full(np.sum(lower == 0), floor*1.25),
                       marker="v", color=COLORS["lower"], s=14, zorder=6)
            ax.axvline(reference, color=".65", lw=.8, zorder=1)
            ax.set(yscale="log", ylim=(floor, top), xlim=(4, 64),
                   title=rf"Rank $i={rank}$  |  Reference $\gamma_a={reference:g}$")
            ax.set_xscale("log", base=2)
            ax.set_xticks([4, 8, 16, 32, 64])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(which="major", alpha=.2)
            factor = records[0]["upper_over_actual"]
            ax.text(.025, .95, f"At gamma 4: upper / actual = {factor:.3g}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.88))
            if col == 0:
                ax.set_ylabel(r"Eigenvalue ratio $\lambda_i(\gamma)/\lambda_1(\gamma)$")
            if row_index == 1:
                ax.set_xlabel(r"New tanh slope $\gamma$")
    handles = [Line2D([], [], color=COLORS["actual"], lw=2.6, label="Actual (rectangular SVD)"),
               Line2D([], [], color=COLORS["lower"], lw=2.2, ls="--", label="Lower bound"),
               Line2D([], [], color=COLORS["upper"], lw=2.7, ls=":", label="Upper bound"),
               Line2D([], [], color=COLORS["lower"], marker="v", lw=0, markersize=5,
                      label="Lower bound is zero (shown at floor)")]
    fig.suptitle("Both directions of the eigenvalue-ratio bound", fontsize=17, y=.98)
    fig.text(.5, .939, "153 tanh neurons + bias | 263 samples | each row uses one fixed reference", ha="center")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .918), ncol=4, frameon=False)
    fig.text(.5, .025,
             "Upper: full old trailing sample space, separate main/correction bounds, lower bound on the top eigenvalue.\n"
             "Lower: old leading space, relative correction, upper bound on the top eigenvalue. Gray line marks the reference.\n"
             "Corrections use the finite kernel. Actual new eigenvalues are comparison only. FP64 evaluations; no interval guarantee.",
             ha="center", color=".3", fontsize=9)
    fig.subplots_adjust(left=.075, right=.987, top=.82, bottom=.14, hspace=.31, wspace=.22)
    fig.savefig(DEST / "two_sided_ratio_bounds.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="col")
    for col, rank in enumerate(RANKS):
        for row_index, reference in enumerate(REFERENCES):
            ax = axes[row_index, col]
            records = [r for r in result["rows"] if r["rank"] == rank
                       and r["reference_gamma"] == reference]
            gamma = np.array([r["gamma"] for r in records])
            ax.plot(gamma, [r["upper_over_actual"] for r in records],
                    color=COLORS["upper"], lw=2.4, ls=":")
            ax.plot(gamma, [r["validation_only_subspace_factor"]*r["denominator_factor"]
                           for r in records], color=".4", lw=1.6, ls="-.")
            lower = np.array([r["lower_over_actual"] for r in records])
            ax.plot(gamma, np.where(lower > 0, lower, np.nan),
                    color=COLORS["lower"], lw=2, ls="--")
            ax.axhline(1, color=".2", lw=.8)
            ax.set(yscale="log", xlim=(4, 64),
                   title=rf"Rank $i={rank}$  |  Reference $\gamma_a={reference:g}$")
            ax.set_xscale("log", base=2)
            ax.set_xticks([4, 8, 16, 32, 64])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(which="major", alpha=.18)
            if col == 0:
                ax.set_ylabel("Bound / actual ratio")
            if row_index == 1:
                ax.set_xlabel(r"New tanh slope $\gamma$")
    handles = [Line2D([], [], color=COLORS["upper"], lw=2.4, ls=":", label="Upper / actual"),
               Line2D([], [], color=COLORS["lower"], lw=2, ls="--", label="Lower / actual"),
               Line2D([], [], color=".4", lw=1.6, ls="-.", label="Unsplit trailing compression / actual (diagnostic)")]
    fig.suptitle("Where the bounds lose accuracy", fontsize=17, y=.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.text(.5, .025,
             "A ratio of 1 is exact. Zero lower bounds are omitted here and explicitly marked in the main figure.\n"
             "Gray uses the new compressed kernel only to diagnose trial-space looseness; it is not the reported decomposition bound.",
             ha="center", fontsize=9, color=".3")
    fig.subplots_adjust(left=.07, right=.99, top=.83, bottom=.14, hspace=.3, wspace=.22)
    fig.savefig(DEST / "bound_tightness.png", dpi=190)
    plt.close(fig)


def save(result, spectra):
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "two_sided_ratio_bounds.json").write_text(json.dumps(result, indent=2)+"\n")
    with (DEST / "two_sided_ratio_bounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    with (DEST / "forward_pair_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["forward_pair_rows"][0]))
        writer.writeheader()
        writer.writerows(result["forward_pair_rows"])
    np.savez_compressed(DEST / "actual_spectra.npz", gammas=result["gammas"],
                        ranks=np.arange(1, spectra.shape[1]+1), ratios=spectra)
    plot(result)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result, spectra = calculate()
        driver_check = forward_driver_check()
    save(result, spectra)
    (DEST / "forward_pair_svd_driver_check.json").write_text(json.dumps(driver_check, indent=2)+"\n")
    print(json.dumps(dict(validation=result["validation"],
                          selected=[r for r in result["rows"] if r["rank"] == 32
                                    and r["gamma"] in [4, 8, 16, 64]]), indent=2))
