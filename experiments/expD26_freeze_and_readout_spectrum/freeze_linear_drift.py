"""Explain saved early-freezing trajectories using a local linear expansion.

No training or coefficient solving: compare each recorded slope update with
the gradient of f(x) ~= B + S*x, expanding tanh(a*x+b) around b.
"""
from pathlib import Path
import json
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD26_freeze_and_readout_spectrum import freeze as base

RESULTS = base.RESULTS / "freeze_mechanism"
TARGETS = ["sine", "sine_mixture", "runge", "gaussian_envelope"]


def diagnose():
    curves, summary = {}, []
    for target in TARGETS:
        cases, cfg = base.load(target)
        x = base.original.midpoint_grid(cfg["n_train"])
        y = base.original.matched.target_values(target, x, cfg)
        variance = np.mean(x*x)
        assert abs(np.mean(x)) < 1e-15
        target_slope = np.mean(x*y)/variance
        eta = cfg["learning_rate"]
        steps = np.unique(np.r_[2+np.arange(0, 501, 5), 501, 502])
        for name in ("freeze_2", "joint"):
            c = cases[name]
            rows = []
            for step in steps:
                state = {k: c[k][step] for k in base.FIELDS}
                a, b, v = (state[k] for k in base.FIELDS)
                h = np.tanh(x[:, None]*a+b)
                pred = h@v[:-1]+v[-1]
                alpha = v[:-1]*(1-np.tanh(b)**2)
                slope = alpha@a
                gradient_model = alpha*variance*(slope-target_slope)
                gradient_exact = base.numpy_gradients(state, x, y)["a"]
                denominator = np.linalg.norm(gradient_exact)
                expected_next = a-eta*gradient_exact
                if step < 502:
                    np.testing.assert_allclose(expected_next, c["a"][step+1], rtol=3e-14, atol=3e-15)
                    observed = np.mean(abs(c["a"][step+1])-abs(a))/eta
                    exact = np.mean(abs(expected_next)-abs(a))/eta
                    np.testing.assert_allclose(observed, exact, rtol=1e-8, atol=2e-14)
                    model = np.mean(abs(a-eta*gradient_model)-abs(a))/eta
                else:
                    observed = model = np.nan
                rows.append(dict(step=int(step), target_slope=float(target_slope),
                                 fitted_linear_slope=float(np.mean(x*pred)/variance),
                                 expansion_slope=float(slope), mean_gamma=float(np.mean(abs(a))),
                                 observed_drift=float(observed), model_drift=float(model),
                                 sign_alignment=float(np.mean(np.sign(a)*alpha)),
                                 relative_gradient_discrepancy=float(np.linalg.norm(gradient_model-gradient_exact)/denominator)))
            cdata = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
            curves[target, name] = cdata
            summary.append(dict(target=target, branch=name, target_slope=float(target_slope),
                                initial_linear_slope=rows[0]["fitted_linear_slope"],
                                final_linear_slope=rows[-1]["fitted_linear_slope"],
                                initial_mean_gamma=rows[0]["mean_gamma"], final_mean_gamma=rows[-1]["mean_gamma"],
                                max_relative_gradient_discrepancy=float(np.max(cdata["relative_gradient_discrepancy"])),
                                final_sign_alignment=rows[-1]["sign_alignment"]))
    (RESULTS/"data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS/"data"/"linear_drift.npz",
                        **{f"{t}__{n}__{key}": value for (t, n), c in curves.items() for key, value in c.items()})
    (RESULTS/"data"/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return curves, summary


def plot(curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    colors = {"freeze_2": "#482878", "joint": "#21918c"}
    fig, axes = plt.subplots(3, 4, figsize=(18.5, 11.6), dpi=160, sharex=True)
    labels = {"sine": "Sine", "sine_mixture": "Mixed sine", "runge": "Runge", "gaussian_envelope": "Gaussian envelope"}
    for col, target in enumerate(TARGETS):
        for name in ("freeze_2", "joint"):
            c = curves[target, name]
            time = c["step"]-2
            axes[0, col].plot(time, c["fitted_linear_slope"], color=colors[name], lw=2)
            axes[1, col].plot(time, c["mean_gamma"], color=colors[name], lw=2)
            axes[2, col].plot(time, c["observed_drift"], color=colors[name], lw=2.1)
            axes[2, col].plot(time, c["model_drift"], color=colors[name], ls="--", lw=1.6)
        star = curves[target, "joint"]["target_slope"][0]
        axes[0, col].axhline(star, color=".25", ls=":", lw=1.6)
        axes[0, col].set_title(labels[target]+rf" · target slope ${star:.3f}$", fontsize=13, pad=14)
        axes[2, col].axhline(0, color=".7", lw=.8)
        axes[2, col].set_xlabel("Updates since readout freeze at step 2", fontsize=10)
        formatter = ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axes[2, col].yaxis.set_major_formatter(formatter)
        axes[1, col].ticklabel_format(axis="y", style="plain", useOffset=False)
        for row in range(3):
            ax = axes[row, col]
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 500)
            ax.tick_params(labelsize=9)
            ax.margins(y=.15)
    axes[0, 0].set_ylabel("Overall linear trend\n"+r"$s_f=\langle x,f\rangle/\langle x,x\rangle$", fontsize=12)
    axes[1, 0].set_ylabel("Mean scale\n"+r"$\overline{\gamma}=\mathrm{mean}|a_k|$", fontsize=12)
    axes[2, 0].set_ylabel("Mean-scale drift per unit GD time\n"+r"$(\overline{\gamma}_{t+1}-\overline{\gamma}_t)/\eta$", fontsize=12)
    handles = [Line2D([], [], color=colors["freeze_2"], lw=2, label="Frozen readout"),
               Line2D([], [], color=colors["joint"], lw=2, label="Continued joint GD"),
               Line2D([], [], color=".25", ls=":", label="Target's best linear trend (top)"),
               Line2D([], [], color=".25", ls="--", label="Small-slope prediction (bottom)")]
    fig.suptitle("Why early freezing lets mean gamma fall farther\nSaved Xavier runs: correcting an initially wrong linear trend", fontsize=18, y=.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .911), ncol=4, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.095, right=.98, top=.84, bottom=.145, hspace=.27, wspace=.30)
    maximum = max(r["max_relative_gradient_discrepancy"] for r in summary)
    fig.text(.5, .056,
             "Expansion: tanh(aₖx+bₖ) ≈ tanh(bₖ)+aₖx sech²(bₖ), giving f ≈ B+Sx, with S=Σₖ cₖaₖ sech²(bₖ).\n"
             "Predicted slope gradient: cₖ sech²(bₖ) ⟨x²⟩ (S−s*). Bottom row compares its predicted finite mean-gamma step with the recorded step.\n"
             f"Maximum relative error of the full slope-gradient vector across these checked states: {100*maximum:.1f}%. Exact gradients match saved updates.\n"
             "No training repeated; no readout refits. Same seed, η=0.002, 1,024 samples in [−1,1]. Linear axes throughout; initial readout coefficients matter.",
             ha="center", va="center", fontsize=9.5)
    fig.savefig(RESULTS/"linear_drift.png")
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        curves, summary = diagnose()
        plot(curves, summary)
    print(json.dumps(summary, indent=2))
