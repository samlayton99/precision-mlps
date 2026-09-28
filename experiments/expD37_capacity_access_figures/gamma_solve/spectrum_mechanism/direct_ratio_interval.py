"""Two-sided, single-gamma Fourier bounds for the finite tanh kernel.

Retain the omitted-center matrix and finitely many explicit Poisson aliases.
The remaining lattice and omitted-center errors have analytic bounds. Actual
finite-kernel eigenvalues are used only for numerical validation and plotting.
Scalar normalization uses the model's constant-direction block.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from direct_ratio_upper_bound import Geometry, design, integral_factor, grid_error, OUT

DEST = OUT / "direct_ratio_interval"
# Include both symmetry classes; all-even ranks would make the upper
# interlacing bound unusually sharp in this symmetric geometry.
RANKS = [13, 20, 26, 33]
ALIAS_TOL = 1e-18


def exterior_factor(x, centers, h, gamma, padding=18):
    """First omitted lattice columns, evaluated without saturated subtraction."""
    count = int(np.ceil(padding / (gamma * h)))
    right = centers[-1] + h * np.arange(1, count + 1)
    left = centers[0] - h * np.arange(1, count + 1)
    zr = np.exp(-2 * gamma * (right[None, :] - x[:, None]))
    zl = np.exp(-2 * gamma * (x[:, None] - left[None, :]))
    # Subtract each saturated constant before projecting. Overall signs of
    # columns are immaterial to their outer products.
    factor = np.column_stack((2*zr/(1+zr), 2*zl/(1+zl))) / np.sqrt(len(x))
    first_right = centers[-1] + (count + 1) * h
    first_left = centers[0] - (count + 1) * h
    assert first_right > max(x) and first_left < min(x)
    tail = 4 * (np.exp(-4*gamma*(first_right-max(x)))
                + np.exp(-4*gamma*(min(x)-first_left))) / (-np.expm1(-4*gamma*h))
    return factor, float(tail), count


def poisson_correction(x, h, gamma, lattice_origin, count):
    """Explicit paired nonzero Fourier coefficients of product-minus-one."""
    d = x[:, None] - x
    midpoint = (x[:, None] + x) / 2
    dcoth = np.full_like(d, 1/gamma)
    np.divide(d, np.tanh(gamma*d), out=dcoth, where=d != 0)
    correction = np.zeros_like(d)
    for k in range(1, count+1):
        nu = 2*np.pi*k/h
        z = np.pi*nu/(2*gamma)
        multiplier = 2*z*np.exp(-z)/(-np.expm1(-2*z))
        correction += (-4*multiplier/(h*len(x)) * dcoth
                       * np.sinc(nu*d/(2*np.pi))
                       * np.cos(nu*(midpoint-lattice_origin)))
    return correction


def construction(x, centers, h, gamma, basis, order=10, padding=20):
    factor, integral_tail = integral_factor(x, h, gamma, basis, order, padding)
    v, s, _ = la.svd(factor, full_matrices=False, lapack_driver="gesvd")
    alpha = s*s
    exterior, exterior_tail, exterior_count = exterior_factor(x, centers, h, gamma)
    delta, theta = grid_error(x, h, gamma)
    exponent = 2*np.pi*theta/(gamma*h)
    alias_count = max(0, int(np.ceil(np.log(delta/ALIAS_TOL)/exponent)))
    delta_retained = delta*np.exp(-exponent*alias_count)
    aliases = poisson_correction(x, h, gamma, centers[0], alias_count)
    tail_in_basis = v.T @ basis.T @ exterior
    alias_in_basis = v.T @ basis.T @ aliases @ basis @ v
    # Keep the diagonal integral spectrum; do not first form a badly
    # conditioned Gram matrix and subtract its leading eigenvalues.
    matrix = np.diag(alpha) + alias_in_basis - tail_in_basis @ tail_in_basis.T
    matrix = (matrix + matrix.T)/2
    mu = la.eigvalsh(matrix, driver="evr")[::-1]
    return dict(mu=mu, alpha=alpha, matrix=matrix, integral_basis=v,
                delta=delta_retained, original_delta=delta,
                integral_tail=integral_tail, exterior_tail=exterior_tail,
                alias_count=alias_count, exterior_count=exterior_count,
                theta=theta)


def evaluate(gamma, geo, refinement=False):
    x, centers = geo.arrays()
    basis = la.null_space(np.ones((1, len(x))))
    result = construction(x, centers, geo.h, gamma, basis)
    mu = result["mu"]
    # Truncating the continuum center integral only underestimates it.
    err_up = result["delta"] + result["integral_tail"]
    err_down = result["delta"] + result["exterior_tail"]
    b, _ = design(x, centers, gamma)
    e = np.ones(len(x))/np.sqrt(len(x))
    readout_mean = b.T @ e
    a = float(readout_mean @ readout_mean)
    coupling = float(la.norm(basis.T @ b @ readout_mean))
    centered_top_upper = float(mu[0] + err_up)
    top_upper = (a + centered_top_upper + np.hypot(a-centered_top_upper, 2*coupling))/2
    # Independent actual finite model, only for checking the theorem.
    actual_u, actual_s, _ = la.svd(b, full_matrices=False, lapack_driver="gesvd")
    actual_values = actual_s**2
    actual_ratio = actual_values/actual_values[0]
    assert a <= actual_values[0]*(1+1e-12) <= top_upper*(1+2e-12)
    rows = []
    for rank in RANKS:
        eigen_lower = max(0., float(mu[rank-1]-err_down))
        eigen_upper = max(0., float(mu[rank-2]+err_up))
        low = eigen_lower/top_upper
        high = min(1., eigen_upper/a)
        actual = float(actual_ratio[rank-1])
        assert low <= actual*(1+1e-5) and actual <= high*(1+1e-5)
        rows.append(dict(rank=rank, eigenvalue_lower=eigen_lower,
                         eigenvalue_actual=float(actual_values[rank-1]),
                         eigenvalue_upper=eigen_upper, lower=low,
                         actual=actual, upper=high,
                         actual_constant_alignment=float(abs(e @ actual_u[:, rank-1])),
                         lower_over_actual=low/actual, upper_over_actual=high/actual,
                         bulk_integral_upper=float(result["alpha"][rank-2]/a)))
    # The unretained corrections should explain any exact-arithmetic
    # mismatch. Floating-point quadrature/SVD errors are checked separately.
    qb = basis.T @ b
    discrepancy = qb@qb.T - result["integral_basis"]@result["matrix"]@result["integral_basis"].T
    matrix_error = float(la.norm(discrepancy, 2))
    assert matrix_error <= max(err_up, err_down) + 1e-10
    record = dict(gamma=float(gamma), top_actual=float(actual_values[0]),
                  top_lower=a, top_upper=float(top_upper), coupling=coupling,
                  alias_count=result["alias_count"], exterior_count=result["exterior_count"],
                  grid_remainder=result["delta"], uncorrected_grid_bound=result["original_delta"],
                  exterior_remainder=result["exterior_tail"],
                  integral_tail=result["integral_tail"],
                  actual_matrix_discrepancy=matrix_error, ranks=rows)
    if refinement:
        refined = construction(x, centers, geo.h, gamma, basis, order=16, padding=24)
        indices = np.array(RANKS)-1
        both = np.r_[indices, indices-1]
        error = float(np.max(abs(refined["mu"][both]/mu[both]-1)))
        assert error < 1e-4
        other = la.svd(b, full_matrices=False, compute_uv=False,
                       lapack_driver="gesdd")**2
        record["validation"] = dict(quadrature_refinement_relative=error,
            actual_svd_driver_relative=float(np.max(abs(other[indices]/actual_values[indices]-1))))
    return record


def plot(records, normalized=True):
    gamma = np.array([r["gamma"] for r in records])
    colors = plt.colormaps["viridis"](np.linspace(.1, .85, len(RANKS)))
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.4))
    for color, rank in zip(colors, RANKS):
        data = [next(v for v in r["ranks"] if v["rank"] == rank) for r in records]
        keys = (("lower", "actual", "upper") if normalized else
                ("eigenvalue_lower", "eigenvalue_actual", "eigenvalue_upper"))
        lo, actual, hi = [np.array([r[key] for r in data]) for key in keys]
        axes[0].fill_between(gamma, lo, hi, color=color, alpha=.12)
        axes[0].plot(gamma, actual, color=color, lw=1.8)
        axes[0].plot(gamma, lo, color=color, ls="--", lw=1.1)
        axes[0].plot(gamma, hi, color=color, ls=":", lw=1.4)
        axes[1].plot(gamma, lo/actual, color=color, ls="--")
        axes[1].plot(gamma, hi/actual, color=color, ls=":")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
        ax.set_xlabel(r"Common slope $\gamma$; no reference gamma")
        ax.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\lambda_i(K_\gamma)/\lambda_1(K_\gamma)$" if normalized
                       else r"$\lambda_i(K_\gamma)$")
    axes[0].set_title("Actual ratios and the proved interval" if normalized
                      else "Actual eigenvalues and the proved interval", pad=12)
    axes[1].axhline(1, color="gray", lw=1)
    axes[1].set_ylim(0, 1.75)
    axes[1].set_ylabel("Bound / actual ratio" if normalized else "Bound / actual eigenvalue")
    axes[1].set_title("Tightness of each side", pad=12)
    rank_handles = [Line2D([0], [0], color=color, label=f"Rank {rank}")
                    for color, rank in zip(colors, RANKS)]
    styles = [Line2D([0], [0], color="black", ls=style, label=label)
              for style, label in [("-", "Actual"), ("--", "Lower bound"), (":", "Upper bound")]]
    fig.legend(handles=rank_handles+styles, loc="upper center", bbox_to_anchor=(.5, .91),
               ncol=7, frameon=False, fontsize=9)
    quantity = "eigenvalue-ratio" if normalized else "eigenvalue"
    fig.suptitle(f"Single-gamma {quantity} intervals: N=128, 153 tanh neurons, 263 samples\n"
                 "Fourier integral + explicit halo and lattice corrections", y=.995, fontsize=12)
    fig.subplots_adjust(left=.08, right=.985, bottom=.14, top=.77, wspace=.27)
    fig.savefig(DEST / ("direct_ratio_interval.png" if normalized else "direct_eigenvalue_interval.png"), dpi=190)
    plt.close(fig)


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    geometry = Geometry()
    gammas = np.unique(np.r_[np.geomspace(4, 64, 33), [4, 8, 16, 32, 64]])
    records = []
    for gamma in gammas:
        record = evaluate(gamma, geometry, refinement=gamma in [4., 8., 16., 32., 64.])
        records.append(record)
        if "validation" in record:
            print(json.dumps(record), flush=True)
    # One deliberately nonuniform sample set checks that alignment of the
    # sample and center grids was not used inadvertently.
    stress = evaluate(8., Geometry(data_jitter=.8), refinement=True)
    output = dict(geometry=dict(N=geometry.N, m=geometry.m, h=geometry.h,
                                W=len(geometry.arrays()[1])),
                  records=records, nonuniform_sample_validation=stress,
                  numerical_scope="FP64 evaluations of analytic inequalities, not interval-certified numbers")
    (DEST / "data.json").write_text(json.dumps(output, indent=2)+"\n")
    plot(records)
    plot(records, normalized=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
