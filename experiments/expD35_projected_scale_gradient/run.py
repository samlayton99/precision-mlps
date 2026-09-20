"""Current-readout projection split, measured offline on ordinary joint GD."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD28_loss_gradient_decomposition import run as baseline

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD35_projected_scale_gradient"


def config():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def split(A, r, J, rcond, driver="gesvd"):
    """Evaluate C^T grad_v L stably as (U^T J)^T(U^T r).

    P is the numerical retained-space projector; no inverse singular values
    are multiplied into the residual. Both J and r are projected for Z^T r_perp.
    """
    U, s, _ = sla.svd(A, full_matrices=False, check_finite=False, lapack_driver=driver)
    rank = int(np.sum(s > rcond*s[0]))
    U = U[:, :rank]
    uj, ur = U.T @ J, U.T @ r
    Z, rp = J-U@uj, r-U@ur
    left, right, total = uj.T@ur, Z.T@rp, J.T@r
    norms = np.array([np.linalg.norm(g) for g in (left, right, total)])
    cosine = np.dot(left, right)/(norms[0]*norms[1]) if norms[0]*norms[1] else np.nan
    # Scale-aware subtraction roundoff estimate, supplemented by SVD/backend audits.
    eps = np.finfo(float).eps
    right_noise = 10*eps*(np.linalg.norm(J)*np.linalg.norm(rp)
                         + np.linalg.norm(Z)*np.linalg.norm(r)
                         + eps*np.linalg.norm(J)*np.linalg.norm(r))
    return dict(left=left, right=right, total=total, norms=norms,
                cosine=np.clip(cosine, -1, 1), rank=rank,
                right_noise=right_noise,
                reconstruction_error=np.linalg.norm(left+right-total))


def measure(state, x, y, rcond, driver="gesvd", backend="torch"):
    a, b, v = (state[k] for k in ("a", "b", "v"))
    if np.any(a == 0):
        raise ValueError("gamma=abs(a) is nondifferentiable at a=0")
    h = baseline.hidden(a, b, x, backend)
    rootn = np.sqrt(len(x))
    A = np.c_[h, np.ones(len(x))]/rootn
    r = (h@v[:-1]+v[-1]-y)/rootn
    J = (1-h*h)*(x[:, None]*v[:-1]*np.sign(a))/rootn
    return split(A, r, J, rcond, driver)


def diagnose(case, cfg):
    x = baseline.previous.midpoint_grid(cfg["n_train"])
    y = baseline.previous.matched.target_values(str(case["target"]), x, cfg)
    rows = []
    for i, step in enumerate(case["steps"]):
        state = {k: case[k][i] for k in ("a", "b", "v")}
        main = measure(state, x, y, cfg["readout_rcond"])
        checks = [measure(state, x, y, cfg["readout_rcond"], driver="gesdd"),
                  measure(state, x, y, cfg["readout_rcond"], backend="numpy")]
        uncertainty = max(main["right_noise"], *(np.linalg.norm(main["right"]-c["right"]) for c in checks))
        left_uncertainty = max(np.linalg.norm(main["left"]-c["left"]) for c in checks)
        resolved = (all(main["rank"] == c["rank"] for c in checks)
                    and main["norms"][1] > 10*uncertainty
                    and main["norms"][0] > 10*left_uncertainty
                    and np.isfinite(main["cosine"]))
        expected = case["training_geometry_gradient"][i, :len(state["a"])]*np.sign(state["a"])
        np.testing.assert_allclose(main["total"], expected, rtol=1e-9, atol=2e-14)
        np.testing.assert_allclose(main["left"]+main["right"], expected, rtol=1e-9, atol=2e-14)
        main.update(right_uncertainty=uncertainty, cosine_resolved=resolved,
                    autodiff_error=np.linalg.norm(main["total"]-expected))
        rows.append(main)
    for key in rows[0]:
        case[key] = np.asarray([r[key] for r in rows])
    return case


def plot(cases, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator

    fig, axes = plt.subplots(4, 4, figsize=(20, 14), dpi=170, sharex=True)
    colors = ["#d07c16", "#12816f", "#202020"]
    labels = [r"Readout-shared: $\|C_\gamma^T\nabla_v L\|_2$",
              r"Out-of-span: $\|Z_\gamma^T(I-P)r\|_2$",
              r"Total: $\|\nabla_\gamma L\|_2$"]
    all_norms = np.concatenate([c["norms"].ravel() for c in cases.values()])
    positive = all_norms[all_norms > 0]
    lower = 10**np.floor(np.log10(positive.min()))
    upper = 10**np.ceil(np.log10(positive.max()))
    for block, arm in enumerate(cfg["arms"]):
        row = 2*block
        for col, target in enumerate(cfg["targets"]):
            case = cases[target, arm]
            steps = case["steps"]
            ax, cx = axes[row, col], axes[row+1, col]
            for j, (color, ls, width) in enumerate(zip(colors, ["-", "-", "--"], [2.5, 1.7, 1.35])):
                vals = case["norms"][:, j].copy()
                vals[vals == 0] = np.nan
                ax.plot(steps, vals, color=color, ls=ls, lw=width, zorder=3+j)
            ax.set_yscale("log")
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
            cx.plot(steps, case["cosine"], color=".65", ls=":", lw=1)
            reliable = np.where(case["cosine_resolved"], case["cosine"], np.nan)
            cx.plot(steps, reliable, color="#63459b", lw=1.6)
            cx.axhline(0, color=".4", lw=.65)
            cx.set_ylim(-1.08, 1.08)
            cx.set_yticks([-1, -.5, 0, .5, 1])
            resolved_count = int(case["cosine_resolved"].sum())
            cx.set_title(f"Resolved direction: {resolved_count}/{len(steps)} snapshots", fontsize=9, color=".35", pad=7)
            if block == 0:
                ax.set_title(baseline.LABELS[target], fontsize=15, pad=13)
            for a in (ax, cx):
                a.set_xscale("symlog", linthresh=10, linscale=.7)
                a.set_xlim(0, cfg["steps"])
                a.set_xticks([0, 10, 100, 1000, 10000], labels=["0", "10", "100", "1,000", "10,000"])
                a.grid(alpha=.18)
                if col:
                    a.tick_params(labelleft=False)
            axes[3, col].set_xlabel("GD step · linear 0–10, logarithmic thereafter", fontsize=10)
        prefix = "Xavier initialization" if arm == "xavier" else "QI initialization · γ₀ = 16"
        axes[row, 0].set_ylabel(prefix+"\nGradient norm", fontsize=12)
        axes[row+1, 0].set_ylabel("Cosine between the two terms\n−1: opposing; +1: aligned", fontsize=11)
    fig.suptitle("Section 2: the current-readout gamma-gradient decomposition", fontsize=21, y=.985)
    handles = [Line2D([], [], color=c, lw=2, ls=ls, label=l)
               for c, ls, l in zip(colors, ["-", "-", "--"], labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.958), ncol=3, frameon=False, fontsize=12)
    fig.text(.5,.905, "Ordinary joint GD throughout · η = 0.002 · 1,024 identical samples on [−1,1] · four function columns", ha="center", fontsize=11)
    fig.subplots_adjust(left=.09, right=.985, top=.865, bottom=.125, hspace=.43, wspace=.17)
    fig.text(.5,.065, "Slopes only: γ = |a| with bias b held fixed; biases and readout still train. N = 128; 177 neurons including 24 halo centers per side; norms include all slopes.\n"
             "P is the SVD-retained projector (relative cutoff 10⁻¹³). Current coefficients are used; no refit or projection enters training.\n"
             "Purple cosine: resolved under SVD/backend checks. Gray dotted cosine: numerically uncertain. Gamma-16 step 0 has zero gradients, so its cosine is undefined.\n"
             "The dashed total can overlap the orange term. Both norm rows share the same log limits; zero norms are omitted, not replaced by a floor.",
             ha="center", va="center", fontsize=10, linespacing=1.6)
    path = RESULTS / "figures/projected_gamma_gradient.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    cases = {}
    with threadpool_limits(limits=cfg["threads"]):
        for arm in cfg["arms"]:
            for target in cfg["targets"]:
                path = RESULTS / "data" / f"{target}__{arm}.npz"
                if path.exists():
                    case = baseline.load(path, cfg)
                else:
                    if args.plot_only:
                        raise FileNotFoundError(path)
                    case = diagnose(baseline.train(target, arm, cfg), cfg)
                    baseline.save(case, cfg, path)
                cases[target, arm] = case
                print(f"Completed {target}/{arm}: {cfg['steps']} steps; "
                      f"max split error {case['reconstruction_error'].max():.3g}; "
                      f"resolved cosine {case['cosine_resolved'].sum()}/{len(case['steps'])}", flush=True)
    plot(cases, cfg)
    summary = {f"{t}/{a}": dict(max_autodiff_error=float(c["autodiff_error"].max()),
               max_reconstruction_error=float(c["reconstruction_error"].max()),
               resolved_cosines=int(c["cosine_resolved"].sum()), snapshots=len(c["steps"]),
               final_norms=c["norms"][-1].tolist()) for (t,a),c in cases.items()}
    (RESULTS/"data/validation.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
