"""Present existing target-weight comparisons and actual finite kernels."""
import json

import numpy as np
from threadpoolctl import threadpool_limits
from run import OUT, Geometry, design
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def alignment_figure():
    saved = json.loads((OUT.parent / "data/default.json").read_text())
    result = saved["comparison"]
    gamma = np.asarray(result["gammas"])
    cap = result["max_steps"]
    reference = gamma[result["reference_index"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True, sharey=True)
    for ax, variant, color, title, label in zip(
        axes, ["fixed_p", "fixed_rates"], ["#1768a5", "#ce6518"],
        ["Change eigenvalue ratios; keep target coefficients fixed",
         "Change target coefficients; keep eigenvalue ratios fixed"],
        [r"Fixed $p_i$ at $\gamma_0=8$", r"Fixed $\lambda_i/\lambda_1$ at $\gamma_0=8$"]
    ):
        low = np.array([cap if v is None else v for v in result[f"steps_{variant}_lower"]])
        high = np.array([cap if v is None else v for v in result[f"steps_{variant}_upper"]])
        ax.fill_between(gamma, low, high, color=color, alpha=.13, lw=0)
        for name, cc, ls, legend in [("actual", "#20252c", "-", "Actual kernel: both change"),
                                     (variant, color, "--", label)]:
            values = np.array([np.nan if v is None else v for v in result[f"steps_{name}"]])
            ax.plot(gamma, values, color=cc, ls=ls, lw=2, marker=".", ms=3.5, label=legend)
            ax.scatter(gamma[np.isnan(values)], np.full(np.isnan(values).sum(), cap),
                       color=cc, marker="x", s=22)
        ax.axvline(reference, color=".55", ls=":", lw=1)
        ax.set(xscale="log", yscale="log", xlim=(2, 128), ylim=(1e3, 3e15),
               xlabel=r"Tanh slope $\gamma$", title=title)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(alpha=.18)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.10), frameon=False, fontsize=10)
    axes[0].set_ylabel("Steps to 1% relative L2 error")
    fig.suptitle("Which change accounts for the reduction in GD steps?", y=.99, fontsize=16)
    fig.text(.5, .929, r"Mixed sine | N=128 | 153 tanh neurons + bias | 263 samples | $\eta_\gamma=0.5/\lambda_1(\gamma)$",
             ha="center", fontsize=11)
    fig.text(.5, .032,
             r"$p_i=|u_i^\top y|^2/\|y\|^2$; components paired by descending eigenvalue rank. Crosses: not reached by $10^{15}$ steps." "\n"
             "Shading: uncertainty from unresolved eigendirections. Saved spectral calculations; no training or fitted rate.",
             ha="center", fontsize=10, color=".3")
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.73, wspace=.16)
    fig.savefig(OUT / "eigenvalue_vs_alignment_steps.png", dpi=185)
    plt.close(fig)


def kernel_figure():
    geom = Geometry()
    x, c = geom.arrays()
    gammas = [4., 8., 16., 64.]
    matrices = []
    for gamma in gammas:
        b, _ = design(x, c, gamma)
        matrices.append(b @ b.T)
    vmin = min(k.min() for k in matrices)
    vmax = max(k.max() for k in matrices)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8), sharex=True, sharey=True)
    for ax, gamma, k in zip(axes, gammas, matrices):
        im = ax.imshow(k, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest", extent=(-1, 1, 1, -1), aspect="equal")
        ax.set(title=rf"$\gamma={gamma:g}$", xlabel=r"Column sample $x_b$")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
    axes[0].set_ylabel(r"Row sample $x_a$")
    fig.suptitle(r"Actual finite tanh kernels $K_\gamma=B_\gamma B_\gamma^\top$", y=.98, fontsize=16)
    fig.text(.48, .875, "Same centers and samples at every gamma; bias included; one shared color scale", ha="center", fontsize=11)
    fig.subplots_adjust(left=.055, right=.89, bottom=.18, top=.78, wspace=.16)
    cax = fig.add_axes([.915, .23, .015, .50])
    fig.colorbar(im, cax=cax, label="Kernel entry (training normalization 1/m)")
    fig.text(.48, .035, "N=128 interior intervals | 153 tanh neurons | 263 samples in [−1, 1] | 12 halo centers per side",
             ha="center", fontsize=10, color=".3")
    fig.savefig(OUT / "kernel_gamma_comparison.png", dpi=185)
    plt.close(fig)
    np.savez_compressed(OUT / "data/kernel_gamma_comparison.npz", gammas=gammas, x=x,
                        centers=c, kernels=np.asarray(matrices))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        alignment_figure()
        kernel_figure()
    print("Saved clean coefficient comparison and actual finite-kernel heatmaps.")
