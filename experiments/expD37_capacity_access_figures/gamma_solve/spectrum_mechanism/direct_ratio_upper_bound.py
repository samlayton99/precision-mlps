"""Reference-free Fourier upper bound and necessary GD steps.

The theorem uses an infinite-center integral and an analytic lattice-error
bound, not measured finite-kernel corrections. Numerical integral spectra
are FP64 quadrature evaluations, not interval certificates. No training.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
from scipy import linalg as la
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import Geometry, design, OUT, fourier_modal
from core import target, weights, hitting_time

DEST = OUT / "direct_ratio_upper_bound"
RANKS = [12, 20, 26, 32]
TOL = .01


def integral_factor(x, h, gamma, basis, order=10, padding=20):
    """A square-root factor of the centered whole-center-line Gram matrix.

    Integrate piecewise; there is no subtraction of large Gram eigenvalues.
    The two omitted tails have operator norm <= tail_error.
    """
    end = max(abs(x)) + padding / gamma
    count = int(np.ceil(2 * end * max(gamma, 8)))
    edges = np.linspace(-end, end, count + 1)
    z, w = roots_legendre(order)
    centers = ((edges[1:] + edges[:-1])[:, None] / 2
               + np.diff(edges)[:, None] * z / 2).ravel()
    mass = (np.diff(edges)[:, None] * w / 2).ravel()
    t = np.tanh(gamma * (x[:, None] - centers))
    t -= t.mean(axis=0)
    factor = basis.T @ (t * np.sqrt(mass / (h * len(x))))
    tail_error = 2 / (h * gamma) * np.exp(-4 * padding)
    return factor, tail_error


def grid_error(x, h, gamma):
    # Any theta in (0, pi/2) is valid. This choice nearly minimizes the
    # analytic bound; it is not a fitted parameter or a reference gamma.
    theta = np.arctan(np.pi / (2 * gamma * h))
    spread = np.dot(x - x.mean(), x - x.mean())
    delta = (8 * gamma * spread / (3 * h * len(x))
             / np.cos(theta)**4 / np.expm1(2 * np.pi * theta / (gamma * h)))
    return float(delta), float(theta)


def necessary_steps(cap, mass):
    if mass <= TOL**2:
        return 0.
    return float(np.log(np.sqrt(mass) / TOL) / -np.log1p(-cap / 2))


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    geo = Geometry()
    x, centers = geo.arrays()
    basis = la.null_space(np.ones((1, len(x))))
    y = target(x, "mixed") / np.sqrt(len(x))
    gammas = np.unique(np.r_[np.geomspace(4, 32, 25), [4, 8, 16, 32, 64]])
    records, validation = [], []
    for gamma in gammas:
        b, _ = design(x, centers, gamma)
        u, s, _ = la.svd(b, full_matrices=False, lapack_driver="gesvd")
        lam = s*s
        ratios = lam / lam[0]
        p = weights(u, y)
        ell = float(len(x) * np.sum(b.mean(axis=0)**2))
        factor, tail = integral_factor(x, geo.h, gamma, basis)
        integral_u, integral_s, _ = la.svd(factor, full_matrices=False,
                                         lapack_driver="gesvd")
        alpha = integral_s**2
        delta, theta = grid_error(x, geo.h, gamma)
        caps = np.minimum(1., (alpha[:len(s)-1] + tail + delta) / ell)
        # Compression interlacing loses one rank: alpha[i-2] bounds lambda_i.
        resolved = ratios > 1e-18
        assert np.all(caps[resolved[1:]] >= ratios[1:][resolved[1:]] * (1 - 1e-7))
        assert 1 - 1e-12 <= ell <= lam[0] * (1 + 1e-12)
        # Count only numerically resolved positive modes for target mass.
        entries = []
        for rank in RANKS:
            mass = float(p[:-1][rank-1:][resolved[rank-1:]].sum())
            cap = float(caps[rank-2])
            entries.append(dict(rank=rank, actual_ratio=float(ratios[rank-1]),
                upper_ratio=cap, integral_eigenvalue=float(alpha[rank-2]),
                positive_tail_mass=mass, necessary_steps=necessary_steps(cap, mass)))
        actual_steps = hitting_time(np.r_[ratios / 2, 0.], p, maximum=10**18)
        tail_mass = np.cumsum(np.where(resolved, p[:-1], 0.)[::-1])[::-1]
        step_bounds = np.array([necessary_steps(float(cap), float(mass))
                               for cap, mass in zip(caps, tail_mass[1:])])
        best = int(np.argmax(step_bounds))
        rec = dict(gamma=float(gamma), top=float(lam[0]), top_lower=ell,
                   grid_error=delta, strip_angle=theta, integral_tail_error=tail,
                   actual_steps=actual_steps, ranks=entries,
                   best_cutoff_rank=best+2, necessary_steps=float(step_bounds[best]))
        records.append(rec)
        if gamma in [4., 8., 16., 32., 64.]:
            fine, fine_tail = integral_factor(x, geo.h, gamma, basis,
                                             order=16, padding=24)
            fine_alpha = la.svdvals(fine)**2
            selected = np.array(RANKS)-2
            refine = float(np.max(abs(fine_alpha[selected] / alpha[selected] - 1)))
            assert refine < 1e-6
            # Independent Fourier integral on the continuum trial vectors.
            q = basis @ integral_u[:, selected]
            _, _, energy, _, fourier_tail = fourier_modal(
                geo, gamma, q, 16385, odd=False)
            error = float(np.max(abs(energy / alpha[selected] - 1)))
            assert error < 1e-5
            u2, s2, _ = la.svd(b, full_matrices=False, lapack_driver="gesdd")
            p2 = weights(u2, y)
            mass2 = float(p2[:-1][25:][(s2[25:] / s2[0])**2 > 1e-18].sum())
            main_mass = entries[2]["positive_tail_mass"]
            validation.append(dict(gamma=float(gamma),
                quadrature_refinement_max_relative=refine,
                independent_fourier_max_relative=error,
                fourier_omitted_tail=fourier_tail,
                svd_driver_rank26_mass_relative=abs(mass2/main_mass-1)))
            print(json.dumps(rec), flush=True)
    result = dict(geometry=dict(N=geo.N, m=geo.m, W=len(centers), h=geo.h),
                  epsilon=TOL, target="mixed", records=records,
                  validation=validation,
                  scope="FP64 evaluations of a proved inequality; no interval certificate")
    (DEST / "data.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(records)


def plot(records):
    selected = [r for r in records if r["gamma"] <= 32]
    g = np.array([r["gamma"] for r in selected])
    colors = plt.colormaps["viridis"](np.linspace(.1, .85, len(RANKS)))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    for col, rank in zip(colors, RANKS):
        rows = [next(v for v in r["ranks"] if v["rank"] == rank) for r in selected]
        axes[0].plot(g, [r["actual_ratio"] for r in rows], color=col)
        axes[0].plot(g, [r["upper_ratio"] for r in rows], color=col, ls="--")
    axes[1].plot(g, [r["necessary_steps"] for r in selected], color="#2a788e",
                 ls="--", marker="o", ms=3, label="Necessary steps (best ratio cutoff)")
    axes[1].plot(g, [r["actual_steps"] for r in selected], color="black", lw=2,
                 label="Actual finite-kernel spectral calculation")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        ax.set_xlabel(r"Common slope $\gamma$ (no reference slope)")
        ax.grid(alpha=.2, which="both")
    axes[0].set_ylabel(r"$\lambda_i(K_\gamma)/\lambda_1(K_\gamma)$")
    axes[0].set_title("Ratio: actual (solid), upper bound (dashed)")
    axes[1].set_ylabel("Steps to 1% relative training error")
    axes[1].set_title("Necessary steps from the upper ratio bounds")
    handles = [Line2D([0], [0], color=col, label=f"Rank {rank}")
               for col, rank in zip(colors, RANKS)]
    axes[0].legend(handles=handles, loc="lower right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Direct Fourier bound, fixed finite geometry: N=128, m=263\n"
                 "Whole-line center integral + analytic lattice error; mixed-sine target",
                 fontsize=12)
    fig.savefig(DEST / "direct_upper_ratios_and_delay.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
