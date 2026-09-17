"""CPU checkpoint analysis and evidence plots; never selects using refit/test error."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from . import core, diagnostics
from .run import Case, write_json

DIAGNOSTICS_REVISION = 2


def analyze_one(task):
    folder_name, step, cutoff, evaluate_test = task
    folder = Path(folder_name)
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    suffix = f"{step:09d}_tau{cutoff:.0e}"
    summary_path = output / f"metrics_{suffix}.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("diagnostics_revision") == DIAGNOSTICS_REVISION and (not evaluate_test or "test" in existing):
            return existing
    start = time.monotonic()
    case = Case(**json.loads((folder / "case.json").read_text()))
    checkpoint = np.load(folder / f"checkpoint_{step:09d}.npz")
    c, gamma = checkpoint["c"], checkpoint["gamma"]
    if not np.isfinite(c).all() or not np.isfinite(gamma).all():
        return {"case": case.key, "step": step, "status": "nonfinite"}
    g = core.geometry(case.n)
    x = np.linspace(-1, 1, case.samples_per_cell * case.n + 1)
    y = core.target(x, case.target, np)
    arrays = diagnostics.checkpoint_arrays(x, y, g.centers, g.h, g.d, c, gamma, g.masks, cutoff,
                                           checkpoint["next_delta_c"], checkpoint["next_delta_lambda"])
    val_x = diagnostics.midpoint_grid(case.validation_points)
    val_y = core.target(val_x, case.target, np)
    pred = diagnostics.prediction(val_x, g.centers, c, gamma)
    refit_pred = diagnostics.prediction(val_x, g.centers, arrays["readout_refit"], gamma)
    arrays.update({"x_validation": val_x, "target_validation": val_y,
                   "prediction_validation": pred, "prediction_refit_validation": refit_pred})
    summary = {"case": case.key, "step": step, "cutoff": cutoff, "diagnostics_revision": DIAGNOSTICS_REVISION,
               "validation": diagnostics.errors(pred, val_y),
               "refit_validation": diagnostics.errors(refit_pred, val_y),
               "rank": int(arrays["retained_rank"]),
               "refit_c_l1": float(np.abs(arrays["readout_refit"]).sum()),
               "refit_a_l2": float(np.linalg.norm(arrays["readout_refit"] / g.d)),
               "gradient_parallel_norm": float(np.linalg.norm(arrays["gradient_parallel"])),
               "gradient_perpendicular_norm": float(np.linalg.norm(arrays["gradient_perpendicular"])),
               "gradient_reconstruction_error": float(np.linalg.norm(arrays["band_gradient_lambda"].sum(axis=0)
                                                                        - arrays["gradient_lambda"])),
               "perpendicular_reconstruction_error": float(np.linalg.norm(arrays["band_gradient_perpendicular"].sum(axis=0)
                                                                             - arrays["gradient_perpendicular"])),
               "band_subtraction_error_norm": float(np.linalg.norm(arrays["band_gradient_subtraction_error"])),
               "cpu_gpu_prediction_max_difference": float(np.max(np.abs(pred - checkpoint["prediction_validation"]))) }
    if evaluate_test:
        test_x = diagnostics.midpoint_grid(65536)
        test_y = core.target(test_x, case.target, np)
        test_pred = diagnostics.prediction(test_x, g.centers, c, gamma)
        test_refit = diagnostics.prediction(test_x, g.centers, arrays["readout_refit"], gamma)
        arrays.update({"x_test": test_x, "target_test": test_y, "prediction_test": test_pred,
                       "prediction_refit_test": test_refit})
        summary["test"] = diagnostics.errors(test_pred, test_y)
        summary["refit_test"] = diagnostics.errors(test_refit, test_y)
    summary["seconds"] = time.monotonic() - start
    np.savez_compressed(output / f"diagnostics_{suffix}.npz", **arrays)
    write_json(summary_path, summary)
    return summary


def all_rows(root, step=None):
    rows = []
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir() or not (folder / "latest.json").exists():
            continue
        case = Case(**json.loads((folder / "case.json").read_text()))
        latest = json.loads((folder / "latest.json").read_text())
        metric = latest["history"][-1]
        status = latest["status"]
        if step is not None and not (status == "nonfinite" and latest["step"] <= step):
            path = folder / f"metrics_{step:09d}.json"
            if not path.exists():
                continue
            metric = json.loads(path.read_text())
            if step != latest["step"]:
                status = "continuing"
        row = {**case.__dict__, "key": case.key, "step": metric["step"], "status": status,
               "validation_rms": metric.get("validation", {}).get("rms", float("nan")),
               "lambda_median": metric.get("lambda_quantiles", [float("nan")] * 5)[2],
               "lambda_max": metric.get("lambda_quantiles", [float("nan")] * 5)[4],
               "folder": str(folder)}
        analyzed = folder / "analysis" / f"metrics_{metric['step']:09d}_tau1e-12.json"
        if analyzed.exists():
            analysis = json.loads(analyzed.read_text())
            row["refit_validation_rms"] = analysis["refit_validation"]["rms"]
            row["retained_rank"] = analysis["rank"]
        rows.append(row)
    return rows


def write_summary(rows, path):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def representative_rows(rows):
    """Show seed 0 at the best two-seed rate pair; never select by refitted error."""
    candidates = defaultdict(list)
    for row in rows:
        if row["seed"] not in (0, 1) or row["step"] < 20000 or not np.isfinite(row["validation_rms"]):
            continue
        key = (row["optimizer"], row["arm"], row["initialization"], row["n"], row["target"],
               row["rate_r"], row["rate_g"])
        candidates[key].append(row)
    selected = {}
    for key, pair in candidates.items():
        if {r["seed"] for r in pair} != {0, 1} or len({r["step"] for r in pair}) != 1:
            continue
        score = np.mean([np.log(max(r["validation_rms"], 1e-300)) for r in pair])
        group = key[:5]
        if group not in selected or score < selected[group][0]:
            selected[group] = score, next(r for r in pair if r["seed"] == 0)
    return [row for _, row in selected.values()]


def plot_evidence(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    chosen = representative_rows(rows)
    if not chosen:
        return
    # These are descriptive pilot trajectories, not a locked final model selection.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row in chosen:
        folder = Path(row["folder"])
        history = json.loads((folder / "latest.json").read_text())["history"]
        history = [r for r in history if r["finite"] and 0 < r["step"] <= row["step"]]
        steps = [r["step"] for r in history]
        label = f"{row['optimizer']} / {row['arm']} / {row['initialization']}"
        line, = axes[0, 0].loglog(steps, [r["validation"]["rms"] for r in history], label=label)
        color = line.get_color()
        axes[0, 1].loglog(steps, [r["lambda_quantiles"][2] for r in history], color=color)
        axes[1, 0].loglog(steps, [r["lambda_travel_rms"] for r in history], color=color)
        analyses = [json.loads(p.read_text()) for p in sorted((folder / "analysis").glob("metrics_*_tau1e-12.json"))]
        analyses = [r for r in analyses if 0 < r["step"] <= row["step"]]
        if analyses:
            axes[1, 1].loglog([r["step"] for r in analyses], [r["refit_validation"]["rms"] for r in analyses],
                              ".-", color=color)
    axes[0, 1].axhline(.25, color="black", linestyle="--", linewidth=1)
    for ax, title, ylabel in zip(axes.ravel(),
                                ["Trained function", "Bandwidth acquisition", "Accumulated bandwidth travel", "Detached readout refit"],
                                ["Validation RMS", "Median |lambda|", "RMS accumulated |delta lambda|", "Refitted validation RMS"]):
        ax.set(title=title, xlabel="Updates", ylabel=ylabel)
        ax.grid(True, alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1), ncol=4, fontsize=8)
    fig.suptitle("Seed 0 at each best two-seed rate pair; selection uses trained validation error; runs may be unfinished",
                 y=.91, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(output / "pilot_trajectories.png", dpi=180)
    plt.close(fig)

    # Keep initializations separate so coincident rate pairs cannot hide each other.
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    finite = [r for r in rows if r["step"] >= 20000 and np.isfinite(r["validation_rms"])]
    log_errors = [np.log10(max(r["validation_rms"], 1e-16)) for r in finite]
    vmin, vmax = (np.floor(min(log_errors)), np.ceil(max(log_errors))) if log_errors else (-8, 0)
    vmax = max(vmax, vmin + 1)
    scatter = None
    for i, optimizer in enumerate(["gd", "adam"]):
        for j, (arm, initialization) in enumerate([(a, b) for a in ["raw", "both"]
                                                   for b in ["xavier", "envelope"]]):
            ax = axes[i, j]
            subset = [r for r in finite if r["optimizer"] == optimizer and r["arm"] == arm
                      and r["initialization"] == initialization and r["seed"] == 0]
            if subset:
                scatter = ax.scatter([r["rate_r"] for r in subset], [r["rate_g"] for r in subset],
                                     c=[np.log10(max(r["validation_rms"], 1e-16)) for r in subset],
                                     s=75, cmap="viridis", vmin=vmin, vmax=vmax,
                                     edgecolors="black", linewidths=.3)
            failed = [r for r in rows if r["optimizer"] == optimizer and r["arm"] == arm
                      and r["initialization"] == initialization and r["seed"] == 0
                      and r["status"] == "nonfinite"]
            ax.scatter([r["rate_r"] for r in failed], [r["rate_g"] for r in failed], marker="x", color="red")
            ax.set(xscale="log", yscale="log", xlabel="Readout learning rate", ylabel="Bandwidth learning rate",
                   title=f"{optimizer.upper()} / {arm} / {initialization}")
    fig.suptitle("Current seed-0 pilot errors; red crosses are nonfinite failures", y=.99)
    fig.tight_layout(rect=(0, 0, .93, .95))
    if scatter is not None:
        fig.colorbar(scatter, cax=fig.add_axes([.95, .2, .012, .6]), label="log10 validation RMS")
    fig.savefig(output / "pilot_rate_map.png", dpi=180)
    plt.close(fig)

    # Keep the envelope representatives for detailed mechanism panels.
    detailed = [r for r in chosen if r["initialization"] == "envelope"][:4]
    if not detailed:
        return
    fig, axes = plt.subplots(len(detailed), 3, figsize=(15, 3.8 * len(detailed)), squeeze=False)
    for panels, row in zip(axes, detailed):
        paths = sorted(p for p in (Path(row["folder"]) / "analysis").glob("diagnostics_*_tau1e-12.npz")
                       if int(p.name.split("_")[1]) <= row["step"])
        if not paths:
            continue
        data = [(int(p.name.split("_")[1]), np.load(p)) for p in paths]
        for step, arrays in data:
            spectrum = np.abs(arrays["residual_fft"][:len(arrays["residual_fft"]) // 2 + 1])**2
            frequency = np.abs(arrays["fft_angular_frequency"][:len(spectrum)])
            panels[0].loglog(frequency[1:], np.maximum(spectrum[1:], 1e-40), label=str(step), alpha=.75)
        steps = [max(1, step) for step, _ in data]
        panels[1].loglog(steps, [np.linalg.norm(a["gradient_parallel"]) for _, a in data], ".-", label="parallel")
        panels[1].loglog(steps, [np.linalg.norm(a["gradient_perpendicular"]) for _, a in data], ".-", label="perpendicular")
        panels[2].plot(steps, [a["force_all"][0] for _, a in data], ".-", label="parallel force")
        panels[2].plot(steps, [a["force_all"][1] for _, a in data], ".-", label="perpendicular force")
        panels[2].set_xscale("log")
        panels[2].set_yscale("symlog", linthresh=1e-12)
        panels[0].set(title=f"{row['optimizer']} / {row['arm']}: residual spectrum", xlabel="Angular frequency", ylabel="DFT coefficient energy")
        panels[1].set(title="Full bandwidth gradient", xlabel="Updates", ylabel="Gradient norm")
        panels[2].set(title="Signed magnitude-increase force", xlabel="Updates", ylabel="Positive favors larger |lambda|")
        for panel in panels:
            panel.legend(loc="lower center", bbox_to_anchor=(.5, 1.1), ncol=3, fontsize=7)
            panel.grid(True, alpha=.2)
        for _, arrays in data:
            arrays.close()
    fig.tight_layout(h_pad=5)
    fig.savefig(output / "pilot_mechanism.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(detailed), 3, figsize=(15, 3.8 * len(detailed)), squeeze=False)
    for panels, row in zip(axes, detailed):
        path = Path(row["folder"]) / "analysis" / f"diagnostics_{row['step']:09d}_tau1e-12.npz"
        if not path.exists():
            for panel in panels:
                panel.text(.5, .5, f"{row['optimizer']} / {row['arm']}: analysis pending",
                           ha="center", va="center", transform=panel.transAxes)
                panel.set_axis_off()
            continue
        with np.load(path) as arrays:
            count = len(arrays["residual_fft"])
            frequency = np.abs(arrays["fft_angular_frequency"][:count // 2 + 1])
            for key, label in [("residual_fft", "live"), ("parallel_fft", "readout-accessible"),
                               ("perpendicular_fft", "perpendicular"), ("refit_fft", "after refit")]:
                energy = np.abs(arrays[key][:len(frequency)])**2
                panels[0].loglog(frequency[1:], np.maximum(energy[1:], 1e-40), label=label)
            centers = arrays["band_bounds"].mean(axis=1)
            centers[0] = 0
            angular_centers = centers * frequency[1]
            for key, label in [("band_gradient_lambda", "full"), ("band_gradient_parallel", "parallel"),
                               ("band_gradient_perpendicular", "perpendicular")]:
                panels[1].plot(angular_centers, np.linalg.norm(arrays[key], axis=1), ".-", label=label)
            panels[1].set_xscale("symlog", linthresh=frequency[1])
            panels[1].set_yscale("log")
            for key, label in [("band_predicted_descent_readout", "readout"),
                               ("band_predicted_descent_geometry", "geometry")]:
                panels[2].plot(angular_centers, arrays[key], ".-", label=label)
            panels[2].set_xscale("symlog", linthresh=frequency[1])
            panels[2].set_yscale("symlog", linthresh=1e-16)
        panels[0].set(title=f"{row['optimizer']} / {row['arm']} at {row['step']:,}",
                      ylabel="DFT coefficient energy", xlabel="Angular frequency")
        panels[1].set(title="Signed gradients summed within each band",
                      ylabel="Norm across slope coordinates", xlabel="Band angular frequency (0 = DC)")
        panels[2].set(title="Next-update linear descent by block",
                      ylabel="Positive predicts loss decrease", xlabel="Band angular frequency (0 = DC)")
        for panel in panels:
            panel.legend(loc="lower center", bbox_to_anchor=(.5, 1.1), ncol=2, fontsize=7)
            panel.grid(True, alpha=.2)
    fig.tight_layout(h_pad=5)
    fig.savefig(output / "pilot_frequency_split.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Case directory, e.g. runs/pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=["latest", "representatives", "all", "plots"], default="representatives")
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--step", type=int, help="Compare a common saved update count, even while workers continue.")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--selection-record", type=Path, help="Required before exposing test results")
    args = parser.parse_args()
    if args.evaluate_test and (args.selection_record is None or not args.selection_record.is_file()):
        parser.error("Test evaluation requires a saved, frozen selection record.")
    rows = all_rows(args.root, args.step)
    tasks = []
    if args.mode != "plots":
        selected = representative_rows(rows) if args.mode == "representatives" else rows
        for row in selected:
            if row["status"] == "nonfinite":
                continue
            folder = Path(row["folder"])
            if args.mode == "latest":
                steps = [row["step"]]
            elif args.mode == "all":
                steps = [int(p.stem.split("_")[1]) for p in folder.glob("checkpoint_*.npz")]
            else:
                available = {int(p.stem.split("_")[1]) for p in folder.glob("checkpoint_*.npz")}
                steps = sorted(available & {0, 1, 10, 100, 1000, 10000, 20000, row["step"]})
            tasks.extend((str(folder), step, args.cutoff, args.evaluate_test) for step in steps)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(analyze_one, tasks):
                print(json.dumps(result), flush=True)
    # Keep the cohort and horizons captured before analysis. New GPU checkpoints
    # must not silently change representative choices after their analysis finishes.
    for row in rows:
        path = Path(row["folder"]) / "analysis" / f"metrics_{row['step']:09d}_tau1e-12.json"
        if path.exists():
            result = json.loads(path.read_text())
            row["refit_validation_rms"] = result["refit_validation"]["rms"]
            row["retained_rank"] = result["rank"]
    evidence = args.root / (f"evidence_{args.step:09d}" if args.step is not None else "evidence_latest")
    evidence.mkdir(exist_ok=True)
    write_summary(rows, evidence / "summary.csv")
    plot_evidence(rows, evidence / "figures")


if __name__ == "__main__":
    main()
