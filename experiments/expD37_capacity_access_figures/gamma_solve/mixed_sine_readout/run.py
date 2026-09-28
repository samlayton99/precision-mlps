"""Repeat the actual-tanh GD / Adam comparison on the note's mixed sine.

Reuse the previous spectral and Adam implementations. Match the note's full
geometry, retaining ordinary raw readout coordinates and zero initialization.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("readout_comparison", HERE.parent / "sqrt_readout/run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def prepare(out, N=512):
    out.mkdir(parents=True, exist_ok=True)
    m, halo = 8193, int(np.ceil(np.sqrt(N)))
    h = 2 / N
    x = np.linspace(-1., 1., m)
    centers = -1 + h * np.arange(-halo, N + halo + 1)
    y = np.sin(2*np.pi*x) + .5*np.sin(6*np.pi*x) + .25*np.sin(10*np.pi*x)
    gammas = np.array([8., 12., 16., 64.])
    hidden = np.tanh(gammas[:, None, None] * (x[None, :, None] - centers[None, None, :]))
    phi = np.concatenate([np.ones((len(gammas), m, 1)), hidden], axis=2)
    cfg = dict(target="sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x)",
               domain=[-1, 1], N=N, m=m, W=len(centers), halo_per_side=halo,
               h=h, gammas=gammas.tolist(), lambdas=(h*gammas).tolist(),
               steps=20000, gd_plot_max_step=10**9, dtype="float64",
               initial_readout="zero", geometry="frozen, uniform centers, common gamma",
               coordinates="raw", samples="uniform including endpoints",
               metric="training relative L2 norm", plot_metric="100 * relative L2 norm",
               loss="0.5*mean((Phi*v-y)^2)", gd_step="0.5/lambda_max(K)",
               adam_lr=.001, adam_betas=[.9, .999], adam_eps=1e-8,
               adam_weight_decay=0., adam_schedule="constant",
               reference="gamma_access_collaborator_note.pdf, section 4; N=512 matches geometry; other widths retain samples, target, gammas and halo=ceil(sqrt(N)) per side")
    np.savez_compressed(out / "problem.npz", x=x, centers=centers, y=y, phi=phi, gammas=gammas)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(cfg), flush=True)


def plot(out):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    cfg, data = base.load(out)
    with np.load(out / "gd.npz") as f: gd = dict(f)
    with np.load(out / "adam.npz") as f: adam = dict(f)
    gsummary = json.loads((out / "gd_summary.json").read_text())
    asummary = json.loads((out / "adam_summary.json").read_text())
    # Independent final-error evaluation in the original sample coordinates.
    final_residual = data["phi"] @ adam["coefficients"] - data["y"][None, :, None]
    final_error = np.linalg.norm(final_residual, axis=(1, 2)) / np.linalg.norm(data["y"])
    discrepancy = float(np.max(abs(final_error - adam["errors"][-1])))
    assert discrepancy < 1e-11
    checks = json.loads((out / "verification.json").read_text())
    checks["adam_final_error_max_difference_original_features"] = discrepancy
    (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")

    colors = ["#2e5fa5", "#cf8e17", "#218f91", "#d44d46"]
    handles = [Line2D([0], [0], color=c, lw=2, label=rf"$\gamma={g:g}$")
               for c, g in zip(colors, data["gammas"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for j, color in enumerate(colors):
        keep = gd["long_steps"] >= 1
        ax.plot(gd["long_steps"][keep], 100*gd["long_errors"][keep, j], color=color, lw=1.8, zorder=3)
        hit = gsummary[j]["first_hit_one_percent"]
        if hit is not None and hit <= cfg["gd_plot_max_step"]:
            ax.plot(hit, 1., "D", color=color, ms=4, zorder=4)
        ax.plot(adam["steps"][1:], 100*adam["errors"][1:, j], color=color,
                lw=.9, ls="--", alpha=.75, zorder=2)
    ax.axhline(1., color=".4", ls=":", lw=1)
    ax.text(1.5, 1.13, "1%", color=".35", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1, cfg["gd_plot_max_step"]), ylim=(.1, 110),
           xlabel="Gradient updates", ylabel="Relative L2 error (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(alpha=.13)
    method_handles = [Line2D([0], [0], color=".2", lw=1.8, label="GD"),
                      Line2D([0], [0], color=".2", lw=1.3, ls="--", label="Adam")]
    fig.suptitle(r"$f(x)=\sin(2\pi x)+\frac{1}{2}\sin(6\pi x)+\frac{1}{4}\sin(10\pi x)$", y=.99, fontsize=12)
    ax.legend(handles=handles + method_handles, loc="lower center", bbox_to_anchor=(.5, 1.015),
              ncol=6, frameon=False, fontsize=9, handlelength=2, columnspacing=1.15,
              borderaxespad=0)
    fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.82)
    fig.savefig(out.parent / "gd_vs_adam.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # A second view uses the screenshot's displayed range, without inventing
    # executed-GD checkpoints: every curve here is an actual-matrix forecast.
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    for j, color in enumerate(colors):
        ax.plot(gd["long_steps"][1:], 100*gd["long_errors"][1:, j], color=color, lw=1.9)
        hit = gsummary[j]["first_hit_one_percent"]
        if hit is not None:
            ax.plot(hit, 1., "D", color=color, ms=6, zorder=4)
            ax.vlines(hit, .1, 1., color=color, ls=":", lw=1)
            ax.text(hit, .13 if j % 2 == 0 else .21, f"{hit:,}", color=color,
                    ha="center", fontsize=10)
    ax.axhline(1., color=".4", ls="--", lw=1)
    ax.text(1100, 1.12, "1% relative error", color=".35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlim=(1e3, 1e9), ylim=(.1, 100), xlabel="GD updates",
           ylabel="Training relative L2 error (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(alpha=.15)
    fig.suptitle("Mixed sine — GD calculated from the actual tanh spectrum", y=.99, fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .93), ncol=4, frameon=False)
    fig.text(.5, .025, "Diamonds mark predicted first 1% crossings. No long GD training run was executed.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.11, right=.97, bottom=.15, top=.78)
    fig.savefig(out.parent / "gd_long_range.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for g, a in zip(gsummary, asummary["rows"]):
        ahit = f"{a['first_hit_one_percent']:,}" if a["first_hit_one_percent"] is not None else "Not within 20,000"
        rows.append(f"| {g['gamma']:g} | {g['first_hit_one_percent']:,} | {ahit} | {100*g['final_error']:.4f}% | {100*a['final_error']:.4f}% |")
    report = """# Mixed sine: actual-kernel GD and 20,000-step Adam

Status: complete; computed results, September 23, 2026.

## TL;DR

The target and geometry match the collaborator note's screenshot. GD is calculated from the actual finite tanh features through one billion steps. Adam executes 20,000 full-batch updates, with the four gamma cases batched independently. Plot values are percentages, so the threshold is 1 on the vertical axis.

## Question

How long do ordinary readout GD and Adam take to reach 1% error on the mixed sine at four frozen gamma values?

## Experiment design

The target is $f(x)=\\sin(2\\pi x)+\\tfrac12\\sin(6\\pi x)+\\tfrac14\\sin(10\\pi x)$ on $[-1,1]$. Use 8,193 uniform endpoint-inclusive samples, center spacing $h=2/512$, 513 interior centers and 23 halo centers per side: 559 tanh features plus bias. The common frozen slopes are $8,12,16,64$. Both optimizers start from zero physical readout coefficients, with no whitening or coefficient scaling. All calculations use FP64.

The objective is $L(v)=\\|\\Phi v-y\\|^2/(2m)$, with $\\Phi=[\\mathbf1,\\tanh(\\gamma(x_i-c_j))]$. The plotted error is $100\\|\\Phi v-y\\|/\\|y\\|$ on the training samples. With $J=\\Phi/\\sqrt m=U\\Sigma V^T$, actual kernel eigenvalues are $\\sigma_i^2$. The GD step is recomputed as $\\eta=0.5/\\sigma_1^2$. GD error uses $E(n)^2=\\sum_i p_i(1-\\eta\\sigma_i^2)^{2n}+p_0$, including the orthogonal target remainder. The original note used saved approximately curvature-normalized steps; our direct recomputation need not reproduce every last integer of its archived counts.

Adam uses learning rate $10^{-3}$, betas $(0.9,0.999)$, epsilon $10^{-8}$, no weight decay and no schedule. Its gradient is evaluated from the original features each step, and ordinary PyTorch Adam updates the raw readouts. All four runs are batched together; every step is recorded. The final optimizer state is saved for continuation. First crossing does not imply that subsequent Adam errors stay below the threshold.

**Code and data.** `experiments/expD37_capacity_access_figures/gamma_solve/mixed_sine_readout/run.py` reuses the verified spectral and Adam routines from `sqrt_readout/run.py`. Configuration, input arrays, trajectories, summaries, verification and resumable Adam state are in `data/` beside this writeup. The earlier square-root experiment is unchanged.

## Results

| Gamma | GD first step ≤1% | Adam first step ≤1% | GD error at 20,000 | Adam error at 20,000 |
|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

### Figures

- [GD versus Adam](gd_vs_adam.png): both methods share one set of axes; solid lines are actual-kernel GD through $10^9$ steps, dashed lines are executed Adam through 20,000. Colors identify gamma. The horizontal threshold is 1%; diamonds mark GD's first crossings. Curves below 0.1% leave the display. No smoothing or best-so-far replacement.
- [Long-range GD, screenshot-style scale](gd_long_range.png): GD displayed from 1,000 to one billion steps; diamonds and vertical guides mark the calculated first 1% crossings. These are predictions, not executed GD measurements or the note's Fourier construction.

## Verification and limits

A short direct-GD trajectory is compared to the spectral residual formula. The batched analytic Adam gradient is compared to ordinary autograd, and final errors are independently recomputed from saved coefficients using the original sample matrix. Discrepancies are recorded in `data/verification.json`. The GD curves describe exact-arithmetic iteration evaluated numerically in FP64; billions of physical updates may behave differently due to accumulated rounding. The error is training error, not held-out error. No Fourier or center-integral kernel approximation is used.

## Conclusions

The plotted curves give the requested direct comparison on the specified mixed sine. Broader claims about other targets or training geometry are outside this run.

## Open questions

No additional ablations were run.
"""
    (out.parent / "mixed_sine_readout_results.md").write_text(report)
    print(json.dumps(dict(gd=gsummary, adam=asummary, checks=checks), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "gd", "verify", "adam", "plot"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.data, args.width)
    elif args.action == "plot": plot(args.data)
    elif args.action == "adam": base.adam(args.data, "cpu")
    else: getattr(base, args.action)(args.data)
