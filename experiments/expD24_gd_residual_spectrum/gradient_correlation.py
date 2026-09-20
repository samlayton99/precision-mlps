"""Per-neuron spatial magnitude bounds versus centered-scale loss gradients.

Every target trains on the same finite interval, standard tanh model and raw
SGD coordinates. The centered-scale derivative is diagnostic: it holds center,
orientation and readout fixed, rather than identifying it with the raw a step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum import first_steps_refit as matched

RESULTS = matched.rc.RESULTS.parent / "gradient_correlation"
TARGETS = ["sine", "sine_mixture", "runge", "gaussian_envelope"]
ARMS = ["xavier", "scaled_xavier", "qi_zero"]
ARM_LABELS = ["Xavier", "Scaled Xavier · centers preserved", "QI · γ₀ = 16, zero readout"]
CHECK_STEPS = [0, 1, 4, 20, 100, 499]


def spatial_integrals(state, target, config, selected, n):
    """Midpoint quadrature for (1/2) integral on [-1,1], including readout."""
    x = -1+(np.arange(n)+.5)*2/n
    a, b, v = (state[key] for key in ("a", "b", "v"))
    hidden = np.tanh(x[:, None]*a+b)
    residual = hidden @ v[:-1]+v[-1]-matched.target_values(target, x, config)
    center = -b[selected]/a[selected]
    orientation = np.sign(a[selected])
    u = x[:, None]*a[selected]+b[selected]
    decay = np.exp(-2*abs(u))
    sech2 = 4*decay/(1+decay)**2
    phi = (x[:, None]-center)*sech2*(orientation*v[selected])
    return np.mean(abs(residual[:, None])*abs(phi), axis=0), np.mean(residual[:, None]*phi, axis=0)


def linear_fit(x, y):
    """OLS with an intercept, on original linear values; all observations count."""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    dx, dy = x-x.mean(), y-y.mean()
    xx, yy = np.dot(dx, dx), np.dot(dy, dy)
    slope = np.dot(dx, dy)/xx
    intercept = y.mean()-slope*x.mean()
    residual = y-(slope*x+intercept)
    return {"slope": float(slope), "intercept": float(intercept),
            "r_squared": float(1-np.dot(residual, residual)/yy),
            "pearson_r": float(np.dot(dx, dy)/np.sqrt(xx*yy)), "points": len(x)}


def check_centered_autograd(state, problem, expected, selected):
    """Differentiate directly in gamma with fixed z, independently of chain rule."""
    a, b, v = (state[key] for key in ("a", "b", "v"))
    gamma = torch.tensor(abs(a), requires_grad=True)
    center, orientation = torch.tensor(-b/a), torch.tensor(np.sign(a))
    readout = torch.tensor(v)
    preactivation = (problem.tx[:, None]-center)*orientation*gamma
    residual = torch.tanh(preactivation) @ readout[:-1]+readout[-1]-problem.ty
    loss = .5*torch.mean(residual.square())
    result = torch.autograd.grad(loss, gamma)[0].detach().numpy()[selected]
    np.testing.assert_allclose(result, expected, rtol=2e-9, atol=2e-13)
    return float(abs(result-expected).max())


def train_case(target, arm, config, selected):
    problem = matched.MatchedProblem(target, config)
    initial = problem.initial(arm)
    params = {key: torch.nn.Parameter(torch.tensor(value)) for key, value in initial.items()}
    optimizer = torch.optim.SGD(params.values(), lr=config["learning_rate"])
    shape = (config["steps"], len(selected))
    data = {key: np.empty(shape) for key in
            ("integral_bound", "gradient", "training_bound", "training_signed_integral", "dense_signed_integral")}
    losses = np.empty(config["steps"]+1)
    check_states = {key: [] for key in params}
    checks = []
    for step in range(config["steps"]+1):
        optimizer.zero_grad(set_to_none=True)
        features = problem.features(params["a"], params["b"])
        loss = problem.objective(features, params["v"])
        losses[step] = loss.item()
        assert np.isfinite(losses[step])
        if step == config["steps"]:
            break
        loss.backward()
        state = {key: value.detach().numpy().copy() for key, value in params.items()}
        a, b = state["a"], state["b"]
        assert np.all(a != 0)
        center = -b/a
        # a=s*gamma, b=-s*gamma*z => dL/dgamma|z = s*(dL/da-z*dL/db).
        gradient = np.sign(a[selected])*(params["a"].grad.numpy()[selected]
                                         -center[selected]*params["b"].grad.numpy()[selected])
        bound, signed = spatial_integrals(state, target, config, selected, config["n_integral"])
        training_bound, training_signed = spatial_integrals(state, target, config, selected, config["n_train"])
        np.testing.assert_allclose(gradient, training_signed, rtol=2e-8, atol=2e-13)
        assert np.all(abs(gradient) <= training_bound+2e-13)
        for key, value in zip(data, (bound, gradient, training_bound, training_signed, signed)):
            data[key][step] = value
        if step in CHECK_STEPS:
            fine_bound, fine_signed = spatial_integrals(state, target, config, selected, 2*config["n_integral"])
            scale = max(float(fine_bound.max()), 1e-30)
            checks.append({"step": step,
                           "direct_centered_autograd_max_difference": check_centered_autograd(state, problem, gradient, selected),
                           "bound_refinement_difference_over_peak": float(abs(bound-fine_bound).max()/scale),
                           "signed_refinement_difference_over_peak": float(abs(signed-fine_signed).max()/scale),
                           "training_signed_integral_max_difference": float(abs(gradient-training_signed).max())})
            assert checks[-1]["bound_refinement_difference_over_peak"] < 1e-3
            for key in params:
                check_states[key].append(state[key])
        optimizer.step()
    fits = {"signed": linear_fit(data["integral_bound"], data["gradient"]),
            "magnitude": linear_fit(data["integral_bound"], abs(data["gradient"]))}
    report = {"target": target, "arm": arm, "fits": fits, "checks": checks,
              "initial_loss": float(losses[0]), "final_loss": float(losses[-1]),
              "max_signed_identity_difference": float(abs(data["gradient"]-data["training_signed_integral"]).max()),
              "max_continuous_signed_difference_over_peak_bound": float(abs(data["gradient"]-data["dense_signed_integral"]).max()/data["integral_bound"].max())}
    data["loss"] = losses
    for key in params:
        data[f"check_state_{key}"] = np.asarray(check_states[key])
        data[f"final_{key}"] = params[key].detach().numpy().copy()
    return data, report


def run(config):
    # Existing N=128 convention has 129 interior grid slots. Keep that model,
    # and omit its +1 endpoint slot so exactly 128 fixed non-halo slots appear.
    selected = np.arange(config["halo"], config["halo"]+128)
    payload = {"config_json": json.dumps(config), "selected_neurons": selected,
               "steps": np.arange(config["steps"]), "check_steps": np.asarray(CHECK_STEPS)}
    reports = []
    for target in TARGETS:
        for arm in ARMS:
            data, report = train_case(target, arm, config, selected)
            prefix = f"{target}__{arm}__"
            payload.update({prefix+key: value for key, value in data.items()})
            reports.append(report)
            print(json.dumps({key: report[key] for key in ("target", "arm", "fits", "max_signed_identity_difference")}), flush=True)
    payload["reports_json"] = json.dumps(reports)
    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "data/correlation.npz", **payload)
    return payload, reports


def plot(payload, reports, mode, *, log_view=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import ScalarFormatter, MaxNLocator

    lookup = {(r["target"], r["arm"]): r for r in reports}
    exact_pairing = mode == "exact_pairing"
    magnitude = mode != "signed"
    steps = payload["steps"]
    colors = np.repeat(steps, len(payload["selected_neurons"]))
    # Shuffle only drawing order, so the last step does not cover every older dot.
    order = np.random.default_rng(20260912).permutation(len(colors))
    fig, axes = plt.subplots(4, 3, figsize=(16, 16), dpi=180)
    norm = Normalize(0, steps[-1])
    for row, target in enumerate(TARGETS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            prefix = f"{target}__{arm}__"
            x_key = "dense_signed_integral" if exact_pairing else "integral_bound"
            x, gradient = (payload[prefix+key].ravel() for key in (x_key, "gradient"))
            if exact_pairing:
                x = abs(x)
            y = abs(gradient) if magnitude else gradient
            fit = linear_fit(x, y) if exact_pairing else lookup[target, arm]["fits"][mode]
            ax.scatter(x[order], y[order], c=colors[order], cmap="viridis", norm=norm,
                       s=.35, alpha=.55, linewidths=0, rasterized=True,
                       zorder=3 if exact_pairing else 1)
            xmax = 1.04*x.max()
            if log_view:
                nonzero = np.r_[x[x > 0], abs(y[y != 0])]
                threshold = max(1e-16, 10.**np.floor(np.log10(np.quantile(nonzero, .001)/10)))
                line_x = np.r_[0., np.geomspace(threshold*1e-3, xmax, 1024)]
            else:
                line_x = np.array([0., xmax])
            ax.plot(line_x, fit["slope"]*line_x+fit["intercept"], color="black",
                    lw=.9 if exact_pairing else 1.5, zorder=1 if exact_pairing else 2)
            ax.plot(line_x, line_x, color="#777777", ls="--", lw=.9, alpha=.65)
            if mode == "signed":
                ax.plot(line_x, -line_x, color="#777777", ls="--", lw=.9, alpha=.65)
                ax.axhline(0, color="#aaaaaa", lw=.5)
            ax.set_xlim(0, xmax)
            ymax = 1.05*max(abs(y).max(), 1e-30)
            ax.set_ylim((-.04*ymax, ymax) if magnitude else (-ymax, ymax))
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            if log_view:
                from matplotlib.ticker import SymmetricalLogLocator, FuncFormatter
                ax.set_xscale("symlog", linthresh=threshold, linscale=.3)
                ax.set_yscale("symlog", linthresh=threshold, linscale=.3)
                ax.set_ylim(bottom=0 if magnitude else -ymax)
                for axis in (ax.xaxis, ax.yaxis):
                    locator = SymmetricalLogLocator(base=10, linthresh=threshold)
                    locator.set_params(numticks=5)
                    axis.set_major_locator(locator)
                    # Zero and the first log tick otherwise collide at the origin.
                    axis.set_major_formatter(FuncFormatter(
                        lambda value, position, cutoff=threshold: "0" if value == 0 else
                        ("" if abs(value) <= 1.01*cutoff else rf"$10^{{{int(round(np.log10(abs(value))))}}}$")))
                ax.minorticks_off()
            else:
                for axis in (ax.xaxis, ax.yaxis):
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((0, 0))
                    axis.set_major_formatter(formatter)
                    axis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=9)
            prefix_title = ARM_LABELS[col]+"\n" if row == 0 else ""
            fit_text = (f"R² = {fit['r_squared']:.8f}  ·  slope = {fit['slope']:.6f}\n"
                        f"intercept = {fit['intercept']:+.2e}" if exact_pairing else
                        f"R² = {fit['r_squared']:.4f}   ·   y = {fit['slope']:.3g}x {fit['intercept']:+.2e}")
            ax.set_title(prefix_title+fit_text,
                         fontsize=10, pad=12)
            if col == 0:
                label = r"$|\partial L/\partial\gamma_k|$" if magnitude else r"$\partial L/\partial\gamma_k$"
                ax.set_ylabel(matched.rc.LABELS[target]+"\n"+label, fontsize=12)
            if row == 3:
                xlabel = (r"$\left|\frac{c_k}{2}\int_{-1}^{1}e(x)\,\psi_{\gamma_k,z_k}(x)\,dx\right|$" if exact_pairing else
                          r"$B_k=\frac{|c_k|}{2}\int_{-1}^{1}|e(x)|\,|\psi_{\gamma_k,z_k}(x)|\,dx$")
                ax.set_xlabel(xlabel, fontsize=12)
    title = ("Gradient magnitude versus signed integral, absolute value outside" if exact_pairing else
             "Gradient magnitude versus spatial magnitude bound" if magnitude else
             "Signed scale gradient versus spatial magnitude bound")
    if log_view:
        title += " · logarithmic view"
    else:
        title += " · linear axes"
    fig.suptitle(title, fontsize=20, y=.988)
    fig.text(.5, .958, "4 targets × 3 initializations · 64,000 points per panel · ordinary GD on the same 1,024 samples in [−1, 1]", ha="center", fontsize=11)
    color_ax = fig.add_axes([.28, .915, .44, .012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=color_ax, orientation="horizontal")
    cb.set_label("Training state / GD step", fontsize=10)
    cb.set_ticks([0, 100, 200, 300, 400, 499])
    cb.ax.tick_params(labelsize=9)
    bound_label = "y = x (exact agreement)" if exact_pairing else "y = B" if magnitude else "y = ±B"
    axis_note = ("Log spacing with a small linear region at zero; all points retained. OLS and R² use ORIGINAL values, so the black fit can appear curved."
                 if log_view else "Linear axes; limits vary by panel to expose each cloud. Drawing order is shuffled; no points are dropped. R² describes pooled observations, not independent samples.")
    comparison_note = ("Signed pairing from note (2.1), with cancellation retained; absolute value taken only after summation."
                       if exact_pairing else "Spatial absolute-overlap bound, not Fourier bound (2.5).")
    caption = "\n".join([
             "ψγ,z(x) = (x − z) sech²(γ(x − z)), as in note (2.1). Current readout cₖ is included; γₖ = |aₖ|. Center and readout are fixed in this derivative.",
             f"Integral: 4,096-point midpoint quadrature; 1/2 matches the uniform loss density. Black: OLS with intercept on all original values; gray dashed: {bound_label}.",
             comparison_note + " 500 states × 128 slots; all 177 neurons train, rate 0.002.",
             axis_note])
    fig.text(.5, .027, caption,
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.985, bottom=.12, top=.84, hspace=.43, wspace=.30)
    path = RESULTS / f"{mode}{'_log' if log_view else ''}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--mode", choices=["signed", "magnitude", "exact_pairing", "both"], default="both")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.plot_only:
        with np.load(RESULTS / "data/correlation.npz") as source:
            payload = {key: source[key] for key in source.files}
        reports = json.loads(str(payload["reports_json"]))
    else:
        config = yaml.safe_load((matched.rc.HERE / "readout_comparison.yaml").read_text())
        config.update(targets=TARGETS, arms=ARMS, steps=500, n_integral=4096,
                      domain=[-1, 1], matched_training=True,
                      tangent="c_k * sign(a_k) * (x-z_k) * sech^2(a_k*x+b_k), z_k=-b_k/a_k",
                      gradient="sign(a_k) * (grad_a_k - z_k * grad_b_k)",
                      normalization="L = 0.5*mean(e^2); bound = 0.5*integral on [-1,1]",
                      selected_slots="128 slots starting at index 24; both halo blocks and the extra right endpoint omitted",
                      selection_note="For random Xavier geometry these are paired neuron slots; centers are not forced into the domain.")
        payload, reports = run(config)
    for mode in (["signed", "magnitude"] if args.mode == "both" else [args.mode]):
        print(plot(payload, reports, mode), flush=True)
        if mode == "magnitude":
            print(plot(payload, reports, mode, log_view=True), flush=True)


if __name__ == "__main__":
    main()
