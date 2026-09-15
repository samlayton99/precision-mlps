"""Four targets across uniform initial scales: GD, VarPro, and diagnostic refits.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/gamma_comparison.py
--plot-only reuses data/data.npz. Ten early-clustered viridis snapshots per PNG.
Ordinary GD also covers 13 logarithmically spaced initial scales for the motion
plot. All runs start with identical uniform centers, width, and zero readout.
The Gaussian target keeps its whole-line objective and cancelling tanh tails.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
from experiments.expD24_gd_residual_spectrum import whole_line as whole
from experiments.expD24_gd_residual_spectrum.signed_log import (
    FLOOR, signed_log, set_signed_log_axis,
)
from experiments.expD24_gd_residual_spectrum.spectrum import (
    frame_steps, snapshot_steps, snapshot_legend,
)

HERE = Path(__file__).resolve().parent
RESULTS = comparison.RESULTS.parent / "gamma_comparison"


class GammaProblem(comparison.Problem):
    def __init__(self, target, config, gamma, refinement=1):
        super().__init__(target, config, refinement)
        self.gamma = float(gamma)

    def initial(self, arm=None):
        centers, _, _ = whole.uniform_geometry(self.config["resolution"], self.config["halo"])
        return {"a": np.full(len(centers), self.gamma), "b": -self.gamma * centers,
                "v": np.zeros(len(centers) + (not self.whole))}


def gamma_of(case):
    return float(case["parameters"]["a"][0, 0])


def arm_name(gamma):
    return f"gamma_{gamma:.12g}"


def motion_statistics(case):
    """Net displacement, not distance travelled; gamma is the physical |a|."""
    a, b = case["parameters"]["a"], case["parameters"]["b"]
    change = abs(a[-1]) - abs(a[0])
    center_change = -b[-1] / a[-1] + b[0] / a[0]
    return {"target": case["target"], "method": case["method"],
            "initial_gamma": gamma_of(case), "neurons": len(change),
            "mean_signed_gamma_change": float(np.mean(change)),
            "mean_absolute_gamma_change": float(np.mean(abs(change))),
            "max_absolute_gamma_change": float(np.max(abs(change))),
            "mean_absolute_center_change": float(np.mean(abs(center_change))),
            "final_min_gamma": float(np.min(abs(a[-1]))),
            "final_max_gamma": float(np.max(abs(a[-1]))),
            "slopes_changing_sign": int(np.sum(np.sign(a[-1]) != np.sign(a[0])))}


def independent_spatial_energy(a, b, v, config, order=16):
    """Spatial quadrature resolving each transition, independent of Fourier nodes.

    VarPro can create gamma > 1000 in one step. Partition at centers and
    multiples of each transition width instead of relying on a fixed grid.
    """
    from scipy.special import roots_legendre

    gamma, centers = abs(a), -b / a
    extent = max(24., float(abs(centers).max() + 40 / gamma.min()))
    offsets = np.array([-24, -16, -8, -4, -2, -1, -.5, 0, .5, 1, 2, 4, 8, 16, 24])
    edges = np.unique(np.r_[-extent, extent, np.linspace(-4, 4, 129),
                            (centers[:, None] + offsets / gamma[:, None]).ravel()])
    roots, weights = roots_legendre(order)
    mid, half = (edges[1:] + edges[:-1]) / 2, np.diff(edges) / 2
    x = (mid[:, None] + half[:, None] * roots).ravel()
    w = (half[:, None] * weights).ravel()
    energy = 0.0
    for start in range(0, len(x), 4096):
        xx = x[start:start+4096]
        residual = comparison.whole_spatial_features(a, b, xx) @ v - whole.target(xx, config)
        energy += w[start:start+4096] @ residual**2
    signed_v = np.sign(a) * v
    c = signed_v - signed_v.mean()
    distances = np.stack((extent - centers, extent + centers))
    tail_l2 = np.sum(abs(c) / np.sqrt(gamma) * np.sqrt(np.sum(np.exp(-4 * gamma * distances), axis=0)))
    tail_l2 += 2 * abs(c.sum()) * np.exp(-2 * extent)
    assert tail_l2 < 1e-12  # Gaussian target tails beyond this extent are smaller still.
    return float(energy)


def validate_whole_line(cases, config, output):
    checks = []
    target_energy = whole.target_energy(config)
    frequency_check = comparison.Problem("gaussian_envelope", config, refinement=2)
    for case in cases:
        if case["target"] != "gaussian_envelope":
            continue
        for mode, view in case.get("views", {}).items():
            for index in (0, -1):
                a, b = (case["parameters"][key][index] for key in ("a", "b"))
                v = view["refit_v"][index]
                coarse = independent_spatial_energy(a, b, v, config, order=12)
                fine = independent_spatial_energy(a, b, v, config, order=24)
                spectral = target_energy * view["relative_l2"][index]**2
                with torch.no_grad():
                    A = frequency_check.features(torch.tensor(a), torch.tensor(b)).numpy()
                spectral_density = frequency_check.weights * abs(A @ v - frequency_check.y)**2
                displayed_fraction = spectral_density[frequency_check.x <= config["max_mode"]].sum() / spectral_density.sum()
                quadrature_error = abs(fine - coarse) / target_energy
                fourier_error = abs(fine - spectral) / target_energy
                assert max(quadrature_error, fourier_error) < 1e-5, (gamma_of(case), mode, index, quadrature_error, fourier_error)
                checks.append({"gamma": gamma_of(case), "mode": mode,
                               "step": case["snapshot_steps"][index],
                               "spatial_relative_l2": float(np.sqrt(fine / target_energy)),
                               "fourier_relative_l2": float(view["relative_l2"][index]),
                               "displayed_band_energy_fraction": float(displayed_fraction),
                               "spatial_refinement_energy_difference": float(quadrature_error),
                               "spatial_fourier_energy_difference": float(fourier_error)})
    (output / "data/validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Independent spatial/Fourier checks passed for {len(checks)} Gaussian endpoints; "
          f"max normalized energy difference {max(check['spatial_fourier_energy_difference'] for check in checks):.3g}", flush=True)


def subset_case(previous, chosen, gamma):
    indices = [previous["snapshot_steps"].index(step) for step in chosen]
    return {**previous, "arm": arm_name(gamma), "snapshot_steps": chosen,
            "parameters": {key: value[indices].copy() for key, value in previous["parameters"].items()},
            "views": {}}


def whole_gd_case(previous, config):
    """Keep the original, independently validated spatial GD implementation."""
    chosen = snapshot_steps(config["steps"], available=frame_steps(config["steps"]))
    indices = [frame_steps(config["steps"]).index(step) for step in chosen]
    n = config["steps"] + 1
    gamma = previous["initial_gamma"]
    frames = {key: values[indices].copy() for key, values in previous["parameters"].items()}
    norms = np.full(n, np.nan)
    norms[chosen] = np.linalg.norm(frames["v"], axis=1)
    return {"target": "gaussian_envelope", "arm": arm_name(gamma), "method": "gd",
            "snapshot_steps": chosen, "parameters": frames, "loss": previous["loss"].copy(),
            "rank": np.full(n, -1), "stationarity": np.full(n, np.nan),
            "readout_norm": norms, "views": {}}


def run(config, path):
    cases = []
    if path.exists():
        cases, previous_config = comparison.load_cases(path)
        if previous_config != config:
            raise ValueError("Saved configuration differs; choose a separate output folder.")
    old_comparison, _ = comparison.load_cases(comparison.RESULTS / "data.npz")
    old_whole, whole_config = whole.load_data(comparison.RESULTS.parent / "whole_line/data.npz")
    chosen = snapshot_steps(config["steps"], available=frame_steps(config["steps"]))
    sweep = np.geomspace(config["gamma_sweep_min"], config["gamma_sweep_max"], config["gamma_sweep_count"])
    # The four comparison scales must be included exactly, without near-duplicates.
    for gamma in config["initial_gammas"]:
        nearest = np.argmin(abs(sweep - gamma))
        if np.isclose(sweep[nearest], gamma, rtol=1e-12):
            sweep[nearest] = gamma
        else:
            raise ValueError("The drift grid must include every comparison scale.")
    jobs = [(target, float(gamma), "gd") for target in config["targets"] for gamma in sweep]
    jobs += [(target, float(gamma), "varpro") for target in config["targets"] for gamma in config["initial_gammas"]]
    for job_index, (target, gamma, method) in enumerate(jobs):
        previous = next((case for case in cases if case["target"] == target
                         and case["method"] == method and case["arm"] == arm_name(gamma)), None)
        if previous is not None:
            continue
        print(f"[{job_index+1}/{len(jobs)}] {target} / gamma={gamma:g} / {method}", flush=True)
        problem = GammaProblem(target, config, gamma)
        if target == "gaussian_envelope" and method == "gd":
            saved = next((case for case in old_whole if case["initial_gamma"] == gamma), None)
            if saved is None:
                saved = whole.train(whole_config, gamma)
                whole.validate([saved], whole_config)
            case = whole_gd_case(saved, config)
        elif target != "gaussian_envelope" and gamma == 16:
            saved = next(case for case in old_comparison if case["target"] == target
                         and case["arm"] == "qi_zero" and case["method"] == method)
            case = subset_case(saved, chosen, gamma)
        else:
            case = comparison.train(problem, arm_name(gamma), method)
        # Save training before diagnostics so an integration check cannot lose it.
        cases.append(case)
        comparison.save_cases(cases, config, path)
        if gamma in config["initial_gammas"]:
            case["views"] = comparison.diagnose(case, problem)
            comparison.save_cases(cases, config, path)
            for view_name, view in case["views"].items():
                print(f"Evaluated {target} / gamma={gamma:g} / {view_name}: "
                      f"relative L2 {view['relative_l2'][0]:.6g} -> {view['relative_l2'][-1]:.6g}", flush=True)
    # Resume diagnostics too if a previous run stopped after saving training.
    for case in cases:
        gamma = gamma_of(case)
        if gamma in config["initial_gammas"] and not case.get("views"):
            case["views"] = comparison.diagnose(case, GammaProblem(case["target"], config, gamma))
            comparison.save_cases(cases, config, path)
    return cases


def plot_comparisons(cases, config, output, modes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = snapshot_steps(config["steps"], available=frame_steps(config["steps"]))
    k = comparison.display_frequency_grid(config)
    x_values = {target: np.linspace(-2 if target == "gaussian_envelope" else -1,
                                    2 if target == "gaussian_envelope" else 1, 2049)
                for target in config["targets"]}
    lookup = {(case["target"], gamma_of(case), mode): (case, view)
              for case in cases for mode, view in case.get("views", {}).items()}
    validation_path = output / "data/validation.json"
    coverage = ({(check["gamma"], check["mode"]): check["displayed_band_energy_fraction"]
                 for check in json.loads(validation_path.read_text())
                 if check["step"] == config["steps"] and "displayed_band_energy_fraction" in check}
                if validation_path.exists() else {})
    headings = {"gd": "Ordinary gradient descent", "varpro": "VarPro: readout solved before every geometry step",
                "gd_refit": "Ordinary GD: readout refitted only for evaluation"}
    for target in config["targets"]:
        views = [view for (name, _, _), (_, view) in lookup.items() if name == target]
        max_residual = max(float(abs(signed_log(view["residual"])).max()) for view in views)
        max_spectrum = max(float(abs(view["spectrum"]).max()) for view in views) * 1.1
        max_error = max(float(view["relative_l2"].max()) for view in views) * 1.15
        domain = "whole line, cancelling tails" if target == "gaussian_envelope" else "[−1, 1]"
        for mode, heading in headings.items():
            if modes is not None and mode not in modes:
                continue
            destination = output / mode
            destination.mkdir(exist_ok=True)
            fig, axes = plt.subplots(4, 3, figsize=(16, 14.5), dpi=160,
                                     sharex="col", sharey="col")
            colors = snapshot_legend(fig, chosen, y=0.92)
            for row, gamma in enumerate(config["initial_gammas"]):
                case, view = lookup[target, gamma, mode]
                left, middle, right = axes[row]
                for i, color in enumerate(colors):
                    left.plot(x_values[target], signed_log(view["residual"][i]), color=color, lw=1.1)
                    middle.plot(k, np.maximum(abs(view["spectrum"][i]), FLOOR), color=color, lw=1.25)
                set_signed_log_axis(left, max_residual)
                left.set_xlim(x_values[target][[0, -1]])
                left.axhline(0, color="#c3c7cb", lw=0.6, zorder=0)
                left.set_ylabel(f"Initial γ = {gamma:g}\nSigned-log residual", fontsize=12)
                middle.set(xlim=(0, config["max_mode"]), yscale="log", ylim=(FLOOR, max_spectrum),
                           yticks=10.0 ** np.arange(-16, 1, 4))
                middle.set_ylabel("Fourier magnitude", fontsize=11)
                if target == "gaussian_envelope" and (gamma, mode) in coverage:
                    middle.text(.98, .035, f"Shown band: {100 * coverage[gamma, mode]:.1f}% of final error energy",
                                transform=middle.transAxes, ha="right", fontsize=8,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
                right.plot(chosen, np.maximum(view["relative_l2"], FLOOR), color="#939aa1", lw=1)
                right.scatter(chosen, np.maximum(view["relative_l2"], FLOOR), c=colors, s=25, zorder=3)
                right.set(xlim=(0, config["steps"]), yscale="log", ylim=(FLOOR, max_error),
                          yticks=10.0 ** np.arange(-16, 1, 4))
                right.set_ylabel("Evaluated relative L₂", fontsize=11)
                left.text(.5, 1.035, f"Relative L₂: {view['relative_l2'][0]:.3g} → {view['relative_l2'][-1]:.3g}",
                          transform=left.transAxes, ha="center", fontsize=10)
                stats = motion_statistics(case)
                middle.text(.5, 1.035, f"Mean Δγ = {stats['mean_signed_gamma_change']:.3g} · mean |Δγ| = {stats['mean_absolute_gamma_change']:.3g}",
                            transform=middle.transAxes, ha="center", fontsize=10)
                if mode != "gd":
                    right.text(.5, 1.035, f"Readout rank: {view['rank'][0]} → {view['rank'][-1]}",
                               transform=right.transAxes, ha="center", fontsize=10)
                for ax in axes[row]:
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(alpha=.18)
                    ax.tick_params(labelsize=10)
            for ax, title in zip(axes[0], ("Residual in function space", "Residual spectrum", "Error after evaluation")):
                ax.set_title(title, fontsize=14, pad=34)
            axes[-1, 0].set_xlabel("x (Gaussian: central region)", fontsize=11)
            axes[-1, 1].set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=11)
            axes[-1, 2].set_xlabel("GD step", fontsize=11)
            fig.suptitle(f"{comparison.LABELS[target]} · {heading}", fontsize=19, y=.987)
            fig.text(.5, .953, f"Same centers and width at γ₀ = 1, 4, 16, 64 · {domain} · {config['steps']:,} steps · rate {config['learning_rate']:g}",
                     ha="center", fontsize=11)
            fig.text(.5, .020,
                     "Ten actual snapshots, clustered early; viridis indicates the step. Axes match across all gamma values and methods for this function.\n"
                     "Residual: Checkpoint A signed-log display, |e| ≤ 10⁻¹⁶ at the center line. Spectrum/error floor: 10⁻¹⁶. Readout SVD cutoff: 10⁻¹³.\n"
                     f"N = {config['resolution']}; halo {config['halo']} per side; 177 neurons start with zero readouts. Δγ = |a_final| − |a_initial|, averaged over all neurons.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.08, right=.985, top=.817, bottom=.09, hspace=.36, wspace=.28)
            fig.savefig(destination / f"{target}.png")
            plt.close(fig)
    print("Saved static comparison PNGs in gd/, varpro/, and gd_refit/.", flush=True)


def plot_motion(cases, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [motion_statistics(case) for case in cases]
    with (output / "data/gamma_motion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    colors = plt.get_cmap("viridis")(np.linspace(.08, .90, len(config["targets"])))
    signed_values, absolute_values = [], []
    for target, color in zip(config["targets"], colors):
        values = sorted((row for row in rows if row["target"] == target and row["method"] == "gd"),
                        key=lambda row: row["initial_gamma"])
        x = [row["initial_gamma"] for row in values]
        signed = np.array([row["mean_signed_gamma_change"] for row in values])
        absolute = np.array([row["mean_absolute_gamma_change"] for row in values])
        signed_values.extend(signed)
        absolute_values.extend(absolute)
        axes[0].plot(x, signed, "o-", color=color, label=comparison.LABELS[target], ms=5)
        axes[1].plot(x, np.maximum(absolute, FLOOR), "o-", color=color, label=comparison.LABELS[target], ms=5)
    if min(signed_values) > 0:
        # All measured signed means are positive here. A log axis exposes their
        # differences without reserving an empty negative half of the panel.
        axes[0].set_yscale("log")
        for ax in axes:
            ax.set_ylim(.75 * min(signed_values + absolute_values), 1.35 * max(signed_values + absolute_values))
        signed_scale = "log scale; all means positive"
    else:
        nonzero = abs(np.asarray(signed_values))
        nonzero = nonzero[nonzero > 0]
        axes[0].set_yscale("symlog", linthresh=max(FLOOR, min(nonzero, default=FLOOR) / 10))
        axes[0].axhline(0, color="#aeb5bc", lw=.8, zorder=0)
        signed_scale = "signed-log scale"
    axes[0].set(title="Mean signed change", ylabel=f"Mean Δγ ({signed_scale})")
    axes[1].set(title="Mean absolute change", ylabel="Mean |Δγ| (log scale)", yscale="log")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], labels=[1, 2, 4, 8, 16, 32, 64])
        ax.set_xlabel("Initial γ (log scale)")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92), ncol=4, frameon=False)
    fig.suptitle(f"How far do the scales move during {config['steps']:,} steps of ordinary GD?", fontsize=18, y=.987)
    fig.text(.5, .026,
             "Δγⱼ = |aⱼ(final)| − |aⱼ(initial)|. Left: signed changes can cancel; right: average size of each neuron's net change.\n"
             "13 initial scales, same 177 neurons and zero readouts; all neurons, including halo, enter the average. GD rate 0.002.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.08, right=.985, top=.79, bottom=.19, wspace=.25)
    fig.savefig(output / "gamma_motion.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "gamma_comparison.yaml")
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    (args.output / "data").mkdir(parents=True, exist_ok=True)
    path = args.output / "data/data.npz"
    if args.plot_only:
        cases, config = comparison.load_cases(path)
    else:
        config = yaml.safe_load(args.config.read_text())
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        cases = run(config, path)
    torch.set_num_threads(config["threads"])
    validate_whole_line(cases, config, args.output)
    plot_comparisons(cases, config, args.output)
    plot_motion(cases, config, args.output)
    print(f"Saved figures in {args.output}", flush=True)


if __name__ == "__main__":
    main()
