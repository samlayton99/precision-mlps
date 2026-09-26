"""Measured replacements for the three-panel construction schematic.

Run from the repository root with ``python experiments/expC09_bandwidth_figures/run.py``.
The precision model follows expC08: round features, labels, readout coefficients,
products and accumulated sums, but retain FP64 SVD internals. Width always counts
activation neurons, including halo centers, and excludes the output bias.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.additional_targets import target_values, frequency_scale, TITLES, EXTRA_NAMES
SELECTOR = ROOT / "docs/lambda_theorem_compatibility/choosing_optimal_lambda/choose_lambda.py"
spec = importlib.util.spec_from_file_location("runge_choose_lambda", SELECTOR)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def round_bits(x, p):
    """Round binary significands to nearest, ties to even; exponent is unrestricted."""
    x = np.asarray(x, dtype=np.float64)
    if p == 53:
        return x
    if not 2 <= p < 53:
        raise ValueError("Supported significands have 2 through 53 bits")
    m, e = np.frexp(x)
    return np.ldexp(np.rint(m * 2.0**p) / 2.0**p, e)


def geometry(width, halo):
    """Exactly W centers: N+1 interior/endpoints and H centers beyond each end."""
    n = width - 2 * halo - 1
    if n < 1 or halo < 0:
        raise ValueError("Width must accommodate both halos and at least two interior nodes")
    spacing = 2.0 / n
    centers = -1.0 + np.arange(-halo, n + halo + 1) * spacing
    assert len(centers) == width
    return n, spacing, centers


def predictions(width, halo, p, target="runge"):
    n, spacing, _ = geometry(width, halo)
    return {
        "target": target, "width": width, "N": n, "halo": halo, "p": p,
        "omega_scale": frequency_scale(target),
        "basic": selector.choose_lambda("tanh", spacing=spacing, e_tol=2.0**(1-p)),
        "refined": selector.choose_lambda("tanh", spacing=spacing,
                                          e_tol=2.0**(1-p), omega_scale=frequency_scale(target)),
    }


def measure(width, halo, lam, p, train_points, eval_points, target="runge", *, target_fn=None):
    evaluate_target = target_fn if target_fn is not None else lambda x: target_values(x, target)
    n, h, centers = geometry(width, halo)
    gamma = lam / h
    # Match the precision convention of the original bandwidth experiment.
    x = np.linspace(-1, 1, train_points)
    a = np.column_stack((np.tanh(gamma * (x[:, None] - centers)), np.ones(x.size)))
    a = round_bits(a, p)
    y = round_bits(evaluate_target(x), p)
    weights, _, rank, singular = lstsq(a, y, cond=2.0**(1-p), lapack_driver="gelsd")
    weights = round_bits(weights, p)
    xe = np.linspace(-1, 1, eval_points)
    features = round_bits(np.tanh(gamma * (xe[:, None] - centers)), p)
    if p == 53:
        fit = features @ weights[:-1] + weights[-1]
    else:
        fit = np.full(xe.size, weights[-1])
        for k in range(width):
            fit = round_bits(fit + round_bits(features[:, k] * weights[k], p), p)
    truth = evaluate_target(xe)
    residual = fit - truth
    return {
        "target": target, "width": width, "N": n, "halo": halo, "lambda": float(lam), "p": p,
        "gamma": float(gamma), "rank": int(rank),
        "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(truth)),
        "linf": float(np.max(np.abs(residual))),
        "readout_norm": float(np.linalg.norm(weights)), "sigma_max": float(singular[0]),
        "train_points": train_points, "eval_points": eval_points,
    }


def refinement_widths(config):
    return range(config["refinement_width_start"], config["refinement_width_stop"] + 1,
                 config["refinement_width_step"])


def make_jobs(config):
    h = config["halo_per_side"]
    widths = config["bandwidth_widths"]
    pw = config["precision_width"]
    base = np.geomspace(config["lambda_min"], config["lambda_max"], config["lambda_count"])
    pred = [predictions(w, h, 53, target) for target in config["targets"] for w in widths]
    pred += [predictions(pw, h, p, target) for target in config["targets"]
             for p in config["precisions"] if p != 53]
    # Predictions are computed before fitting; include their exact lambda values in each sweep.
    jobs = set()
    for pr in pred:
        for lam in [*base, *(pr[rule]["lambda"] for rule in ("basic", "refined"))]:
            jobs.add((pr["target"], pr["width"], h, float(lam), pr["p"]))
    for target in config["targets"]:
        for w in refinement_widths(config):
            jobs.add((target, w, h, config["refinement_lambda"], 53))
    halo_lams = np.geomspace(config["lambda_min"], config["lambda_max"], config["halo_lambda_count"])
    for pr in pred:
        for lam in [*halo_lams, *(pr[rule]["lambda"] for rule in ("basic", "refined"))]:
            for halo in [h, *config["halo_checks"]]:
                # Increase only the halo, preserving N, h, gamma and interior centers.
                jobs.add((pr["target"], pr["N"] + 2*halo + 1, halo, float(lam), pr["p"]))
    return sorted(jobs), pred


def key(row):
    return row["target"], row["width"], row["halo"], row["lambda"], row["p"]


def draw(config, rows, prediction, target):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    figures = OUT / target / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.65))
    fig.subplots_adjust(left=.066, right=.989, bottom=.25, top=.70, wspace=.27)
    inside_legends = config.get("legend_location") == "upper right"
    if inside_legends:
        fig.subplots_adjust(top=.78)
    halo = config["halo_per_side"]
    basic_color, refined_color = "#238b45", "#c73435"
    width_colors = [plt.get_cmap("viridis")(x) for x in np.linspace(.08, .86, len(config["bandwidth_widths"]))]
    precision_colors = [plt.get_cmap("viridis")(x) for x in np.linspace(.08, .86, len(config["precisions"]))]

    def curve(width, p, halo=halo):
        cells = sorted((r for r in rows if r["width"] == width and r["p"] == p
                        and r["halo"] == halo), key=lambda r: r["lambda"])
        return np.array([r["lambda"] for r in cells]), np.array([r["relative_l2"] for r in cells])

    def pred_for(width, p):
        return next(pr for pr in prediction if pr["width"] == width and pr["p"] == p)

    def prediction_point(ax, pr, rule, color):
        lam = pr[rule]["lambda"]
        row = next(r for r in rows if key(r) == (target, pr["width"], halo, lam, pr["p"]))
        ax.scatter([lam], [row["relative_l2"]], s=35, marker="o" if rule == "basic" else "D",
                   facecolors=color, edgecolors=basic_color if rule == "basic" else refined_color,
                   linewidths=1.4, zorder=8)
        return lam, row["relative_l2"]

    titles = ["(a) Geometric refinement", "(b) Bandwidth prediction", "(c) Working precision"]
    subtitles = [r"Fixed $\lambda=0.25$, $p=53$", r"Width sweep at $p=53$",
                 rf"Fixed $W={config['precision_width']}$"]
    for ax, title, subtitle in zip(axes, titles, subtitles):
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 10 if inside_legends else 1)
        ax.set_yticks([10.0**v for v in range(-16, 1, 4)])
        ax.grid(True, which="major", alpha=.2, lw=.6)
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.set_title(title, loc="left", y=1.19 if inside_legends else 1.43, fontweight="bold")
        ax.text(.5, 1.10 if inside_legends else 1.31, subtitle, ha="center", transform=ax.transAxes)

    ref = [r for r in rows if r["halo"] == halo and r["p"] == 53
           and r["lambda"] == config["refinement_lambda"]
           and r["width"] in refinement_widths(config)]
    ref.sort(key=lambda r: r["width"])
    rw = np.array([r["width"] for r in ref])
    re = np.array([r["relative_l2"] for r in ref])
    axes[0].plot(rw, re, "o-", color="#237bb5", lw=1.4, ms=2.0, label="FP64 measured")
    mask = (re > 1e-12) & (re < .5)
    geometric = None
    if mask.sum() >= 3:
        slope, intercept = np.polyfit(rw[mask], np.log(re[mask]), 1)
        xx = np.linspace(rw.min(), rw.max(), 500)
        axes[0].plot(xx, np.exp(intercept+slope*xx), "--", color="#e18c2b", lw=1.5,
                     label="Geometric trend (fit)")
        geometric = {"slope_per_neuron": float(slope), "intercept": float(intercept),
                     "widths": rw[mask].tolist(), "selection": "1e-12 < error < 0.5",
                     "interpretation": "empirical trend, not a theoretical error bound"}
    axes[0].set_xlim(rw.min(), rw.max())
    axes[0].set_xticks([64, 96, 128, 192, 256])
    axes[0].set_xlabel(r"Network width $W$")
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False,
                   fontsize=9, ncol=1)
    floor = float(np.median(re[rw >= 192]))
    tail = re[rw >= 192]
    if floor < 1e-12 and (not inside_legends or tail.max()/tail.min() < 10):
        floor_point = (rw[-1], re[-1]) if inside_legends else (192, re[rw == 192][0])
        axes[0].annotate("Recovery floor" if re.max() > 1e-12 else "Already near recovery floor",
                         xy=floor_point, xytext=(132, 2e-10), fontsize=9,
                         arrowprops={"arrowstyle": "->", "lw": .7})

    for ax in axes[1:]:
        ax.set_xscale("log")
        ax.set_xlim(config["lambda_min"], config["lambda_max"])
        ticks = [.05, .1, .2, .5, 1.0]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Relative bandwidth $\lambda$")

    for w, color in zip(config["bandwidth_widths"], width_colors):
        lam, err = curve(w, 53)
        axes[1].plot(lam, err, color=color, lw=1.5, label=rf"$W={w}$")
        pr = pred_for(w, 53)
        axes[1].axvline(pr["refined"]["lambda"], color=color, ls=":", alpha=.3, lw=.8)
        for rule in ("basic", "refined"):
            prediction_point(axes[1], pr, rule, color)
    anchors = [pred_for(w, 53)["refined"]["lambda"] for w in config["bandwidth_widths"]]
    axes[1].axvspan(min(anchors), max(anchors), color="#7aa7c6", alpha=.14, zorder=0)
    axes[1].axvline(prediction[0]["basic"]["lambda"], color=basic_color, ls="--", lw=1.1, alpha=.8)
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False,
                   fontsize=9, columnspacing=.8, handlelength=1.5)

    for p, color in zip(config["precisions"], precision_colors):
        lam, err = curve(config["precision_width"], p)
        label = rf"$p={p}$" + (" (FP64)" if p == 53 else "")
        axes[2].plot(lam, err, color=color, lw=1.5, label=label)
        pr = pred_for(config["precision_width"], p)
        for rule in ("basic", "refined"):
            prediction_point(axes[2], pr, rule, color)
    axes[2].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False,
                   fontsize=9, columnspacing=1.4)

    if inside_legends:
        for ax in axes:
            ax.legend(loc="upper right", frameon=True, framealpha=.95, fontsize=8, ncol=1)

    fig.suptitle(TITLES[target], y=.995, fontsize=12 if target == "gaussian_envelope" else 13)
    refined_label = "Refined rule (mean local frequency; heuristic)" if target == "chirp" else "Refined rule"
    handles = [Line2D([], [], marker="o", mfc="white", mec=basic_color, mew=1.4, ls="none",
                      label="Basic bandwidth rule"),
               Line2D([], [], marker="D", mfc="white", mec=refined_color, mew=1.4, ls="none",
                      label=rf"{refined_label}, $\bar\omega={frequency_scale(target):.3g}$")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.525, .068), ncol=2,
               frameon=False, fontsize=9)
    fig.text(.525, .012,
             rf"Measured fits on $[-1,1]$; {halo} halo nodes per side. "
             r"$p<53$: rounded-precision model with FP64 SVD internals.",
             ha="center", fontsize=9, color=".25")
    fig.savefig(figures / f"{target}_bandwidth.png", dpi=220)
    plt.close(fig)

    # A separate control figure keeps the main figure readable.
    control_fig, control_axes = plt.subplots(1, len(config["bandwidth_widths"]), figsize=(14.4, 3.8), sharey=True)
    for ax, w in zip(control_axes, config["bandwidth_widths"]):
        n = geometry(w, halo)[0]
        for hh, style in zip([halo, *config["halo_checks"]], ["-", "--", ":"]):
            cells = sorted((r for r in rows if r["N"] == n and r["halo"] == hh and r["p"] == 53),
                           key=lambda r: r["lambda"])
            ax.loglog([r["lambda"] for r in cells], [r["relative_l2"] for r in cells],
                      style, lw=1.4, label=f"{hh} per side")
        ax.set_title(rf"Main width $W={w}$; fixed $N={n}$")
        ax.set_ylim(1e-16, 10 if inside_legends else 1)
        ax.set_xlim(config["lambda_min"], config["lambda_max"])
        ax.set_xlabel(r"$\lambda$")
        ax.grid(True, which="major", alpha=.2)
    control_axes[0].set_ylabel(r"Relative $L^2$ error")
    control_fig.legend(*control_axes[0].get_legend_handles_labels(), loc="upper center",
                       bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False)
    control_fig.tight_layout(rect=(0, 0, 1, .87))
    control_fig.savefig(figures / "halo_check.png", dpi=180)
    plt.close(control_fig)
    return {"target": target, "geometric_fit": geometric,
            "refinement_floor_median_W_ge_192": floor}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    config = yaml.safe_load(args.config.read_text())
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(SELECTOR.read_bytes()).hexdigest()
    targets_hash = hashlib.sha256((HERE / "targets.py").read_bytes()).hexdigest()
    jobs, pred = make_jobs(config)
    manifest = {"config": config, "config_sha256": config_hash, "selector": str(SELECTOR.relative_to(ROOT)),
                "selector_sha256": source_hash, "targets_sha256": targets_hash, "predictions": pred,
                "precision_model": "expC08 rounded entries/readout/products/sums, FP64 SVD, cutoff 2**(1-p)",
                "width_definition": "W = N + 1 + 2*halo, excluding output bias"}
    if any(target in EXTRA_NAMES for target in config["targets"]):
        manifest["additional_targets_sha256"] = hashlib.sha256((HERE / "additional_targets.py").read_bytes()).hexdigest()
    manifest_path = data_dir / "predictions.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing run has a different configuration/selector; preserve it before rerunning")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data_dir / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = {key(r) for r in rows}
    pending = [j for j in jobs if j not in completed]
    if args.plot_only and pending:
        raise ValueError(f"Missing {len(pending)} measurements")
    start = time.monotonic()
    with threadpool_limits(limits=1), path.open("a") as output:
        for i, (target, w, h, lam, p) in enumerate(pending, 1):
            row = measure(w, h, lam, p, config["train_points"], config["eval_points"], target)
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            if i % 40 == 0 or i == len(pending):
                print(f"Measured {i}/{len(pending)} new cells ({time.monotonic()-start:.1f}s)", flush=True)
    summary = [draw(config, [r for r in rows if r["target"] == target],
                    [pr for pr in pred if pr["target"] == target], target) for target in config["targets"]]
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    for target in config["targets"]:
        print(OUT / target / f"figures/{target}_bandwidth.png")


if __name__ == "__main__":
    main()
