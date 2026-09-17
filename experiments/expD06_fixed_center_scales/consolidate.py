"""Consolidate saved trajectories and detached diagnostics; emit data and figures only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

from . import core
from .analyze import analyze_one, write_summary
from .run import Case, write_json


def trace_windows(folder, ranges):
    values = {bounds: np.full((bounds[1] - bounds[0], 5), np.nan) for bounds in ranges}
    for path in Path(folder).glob("trace_*.npz"):
        _, left, right = path.stem.split("_")
        left, right = int(left), int(right)
        overlaps = [r for r in ranges if max(left, r[0]) < min(right, r[1])]
        if not overlaps:
            continue
        with np.load(path) as saved:
            data = saved["trace"]
            for begin, end in overlaps:
                lo, hi = max(begin, left), min(end, right)
                values[(begin, end)][lo - begin:hi - begin] = data[lo - left:hi - left]
    result = []
    for (begin, end), trace in values.items():
        if not np.isfinite(trace).all():
            raise ValueError(f"Missing or nonfinite trace in {folder}: {begin}:{end}")
        rms = np.sqrt(2 * trace[:, 0])
        result.append({"begin": begin, "end": end,
                       "training_rms_over_time": float(np.sqrt(2 * trace[:, 0].mean())),
                       "training_rms_quantiles": np.quantile(rms, [0, .1, .5, .9, 1]).tolist(),
                       "mean_lambda_update_rms": float(trace[:, 1].mean()),
                       "mean_readout_update_scaled_rms": float(trace[:, 2].mean()),
                       "mean_lambda_gradient_rms": float(trace[:, 3].mean())})
    return result


def summarize_case(folder_name):
    folder = Path(folder_name)
    case = Case(**json.loads((folder / "case.json").read_text()))
    latest = json.loads((folder / "latest.json").read_text())
    row = {**case.__dict__, "key": folder.name, "folder": str(folder),
           "status": latest["status"], "latest_step": latest["step"]}
    if latest["status"] == "nonfinite":
        return row, []
    ends = [s for s in [20000, 80000, 160000, 280000, 300000, 320000, 640000]
            if s <= latest["step"]]
    windows = trace_windows(folder, [(s - 20000, s) for s in ends])
    dense_path = folder / "validation_trace.json"
    dense = json.loads(dense_path.read_text()) if dense_path.exists() else []
    for window in windows:
        sampled = [r["validation_rms"] for r in dense if window["begin"] < r["step"] <= window["end"]]
        if sampled:
            window["sampled_validation_rms_over_time"] = float(np.sqrt(np.mean(np.square(sampled))))
            window["validation_samples"] = len(sampled)
    for end in [20000, 80000, 160000, 320000, 640000]:
        path = folder / f"metrics_{end:09d}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        row[f"validation_{end}"] = m["validation"]["rms"]
        row[f"lambda_median_{end}"] = m["lambda_quantiles"][2]
        row[f"lambda_max_{end}"] = m["lambda_quantiles"][4]
        row[f"lambda_travel_{end}"] = m["lambda_travel_rms"]
    for w in windows:
        row[f"window_rms_{w['end']}"] = w["training_rms_over_time"]
        if "sampled_validation_rms_over_time" in w:
            row[f"window_validation_{w['end']}"] = w["sampled_validation_rms_over_time"]
    if "validation_320000" in row:
        with np.load(folder / "checkpoint_000160000.npz") as a, np.load(folder / "checkpoint_000320000.npz") as b:
            row["lambda_net_rms_160k_320k"] = float(np.sqrt(np.mean((b["lambda"] - a["lambda"])**2)))
            row["lambda_path_rms_160k_320k"] = float(np.sqrt(np.mean((b["lambda_travel"] - a["lambda_travel"])**2)))
    return row, [{"key": folder.name, **w} for w in windows]


def paired_choices(rows, score_key):
    candidates = defaultdict(list)
    for row in rows:
        if score_key not in row:
            continue
        key = (row["optimizer"], row["arm"], row["initialization"], row.get("epsilon_mode", "legacy"),
               row["rate_r"], row["rate_g"])
        candidates[key].append(row)
    best = {}
    for key, pair in candidates.items():
        if len(pair) != 2 or len({r["seed"] for r in pair}) != 2:
            continue
        score = float(np.mean([np.log(max(r[score_key], 1e-300)) for r in pair]))
        if key[:4] not in best or score < best[key[:4]]["mean_log_score"]:
            best[key[:4]] = {"group": key[:4], "score_key": score_key, "mean_log_score": score,
                            "rate_r": key[4], "rate_g": key[5], "cases": [r["key"] for r in pair]}
    return list(best.values())


def mechanism_rows(folder, steps):
    case = Case(**json.loads((folder / "case.json").read_text()))
    g = core.geometry(case.n)
    result = []
    for step in steps:
        path = folder / "analysis" / f"diagnostics_{step:09d}_tau1e-12.npz"
        if not path.exists():
            continue
        with np.load(path) as d, np.load(folder / f"checkpoint_{step:09d}.npz") as p:
            row = {"key": folder.name, "step": step, "optimizer": case.optimizer,
                   "arm": case.arm, "initialization": case.initialization, "seed": case.seed}
            for name in ["residual_train", "residual_parallel", "residual_perpendicular", "residual_refit"]:
                row[f"{name}_rms"] = float(np.sqrt(np.mean(d[name]**2)))
            for region, mask in {"all": np.ones(g.width, bool), **g.masks}.items():
                force = d[f"force_{region}"]
                for name, value in zip(["parallel_force", "perpendicular_force", "tangent_norm",
                                        "projected_tangent_norm", "tangent_ratio", "alignment"], force):
                    row[f"{region}_{name}"] = float(value) if np.isfinite(value) else None
                direction = np.where(mask, np.sign(p["gamma"]), 0.)
                direction /= max(np.linalg.norm(direction), 1.)
                row[f"{region}_actual_outward_update"] = float(direction @ p["next_delta_lambda"])
            if case.optimizer == "adam":
                for block in ["readout", "slope"]:
                    ratio = p[f"adam_{block}_sqrt_v_over_epsilon"]
                    row[f"{block}_epsilon_dominated_fraction"] = float(np.mean(ratio < 1))
                    row[f"{block}_sqrt_v_over_epsilon_median"] = float(np.median(ratio))
            result.append(row)
    return result


def figures(rows, choices, mechanism, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    optimizers = sorted({r["optimizer"] for r in rows})
    groups = sorted({(r["arm"], r["initialization"], r.get("epsilon_mode", "legacy")) for r in rows})
    fig, axes = plt.subplots(len(optimizers), len(groups), squeeze=False,
                             figsize=(4 * len(groups), 3.7 * len(optimizers)))
    scatter = None
    for i, opt in enumerate(optimizers):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            subset = [r for r in rows if r["optimizer"] == opt and
                      (r["arm"], r["initialization"], r.get("epsilon_mode", "legacy")) == group]
            pairs = defaultdict(list)
            for r in subset:
                if "window_rms_320000" in r:
                    pairs[(r["rate_r"], r["rate_g"])].append(r)
            for (rr, rg), pair in pairs.items():
                if len(pair) != 2:
                    continue
                score = np.mean([np.log10(r["window_rms_320000"]) for r in pair])
                scatter = ax.scatter(rr, rg, c=[score], vmin=-6, vmax=0, cmap="viridis", s=80)
            failed = [r for r in subset if r["status"] == "nonfinite"]
            ax.scatter([r["rate_r"] for r in failed], [r["rate_g"] for r in failed], color="red", marker="x")
            ax.set(xscale="log", yscale="log", xlabel="Trained readout LR", ylabel="Trained slope LR",
                   title=f"{opt} / {' / '.join(group)}")
    fig.suptitle("Paired geometric mean of training RMS over updates 300k–320k; red = numerical failure")
    fig.tight_layout(rect=(0, 0, .96, .94))
    if scatter is not None:
        fig.colorbar(scatter, cax=fig.add_axes([.97, .2, .008, .6]), label="log10 window RMS")
    fig.savefig(output / "lr_windows.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for group in sorted({(r["optimizer"], r["initialization"]) for r in rows}):
        subset = [r for r in rows if (r["optimizer"], r["initialization"]) == group and "refit_320000" in r]
        label = " / ".join(group)
        axes[0].scatter([r["window_rms_320000"] for r in subset], [r["validation_320000"] for r in subset], s=12, label=label)
        axes[1].scatter([r["lambda_median_320000"] for r in subset], [r["refit_320000"] for r in subset], s=12, label=label)
        axes[2].scatter([max(r["lambda_path_rms_160k_320k"], 1e-20) for r in subset],
                        [max(r["lambda_net_rms_160k_320k"], 1e-20) for r in subset], s=12, label=label)
    for ax, x, y in zip(axes, ["Late-window training RMS", "Median |lambda|", "Accumulated lambda motion, 160k–320k"],
                         ["Endpoint validation RMS", "Detached refit validation RMS", "Net lambda change, 160k–320k"]):
        ax.set(xscale="log", yscale="log", xlabel=x, ylabel=y)
        ax.grid(alpha=.2)
    axes[1].axvline(.25, color="black", linestyle="--", linewidth=.7)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "fit_geometry_motion.png", dpi=160)
    plt.close(fig)
    chosen = set(k for c in choices if c["score_key"] == "window_rms_320000" for k in c["cases"])
    selected = [r for r in rows if r["key"] in chosen and r["seed"] == min(s["seed"] for s in rows)]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.2 * len(selected)), squeeze=False)
    for panels, r in zip(axes, selected):
        entries = [m for m in mechanism if m["key"] == r["key"] and m["step"] > 0]
        for field, label in [("residual_train_rms", "live"), ("residual_perpendicular_rms", "perpendicular"), ("residual_refit_rms", "refit")]:
            panels[0].loglog([m["step"] for m in entries], [max(m[field], 1e-30) for m in entries], ".-", label=label)
        for field, label in [("all_parallel_force", "parallel"), ("all_perpendicular_force", "perpendicular"), ("all_actual_outward_update", "Adam/GD update")]:
            panels[1].plot([m["step"] for m in entries], [m[field] for m in entries], ".-", label=label)
        panels[1].set_xscale("log")
        panels[1].set_yscale("symlog", linthresh=1e-16)
        folder = Path(r["folder"])
        with np.load(folder / "analysis" / "diagnostics_000320000_tau1e-12.npz") as d, np.load(folder / "checkpoint_000320000.npz") as p:
            frequency = d["fft_angular_frequency"][:len(d["probe_raw_core_cos"])]
            median_gamma = max(float(np.median(np.abs(p["gamma"]))), 1e-30)
            for label in ["raw", "perpendicular"]:
                probe = np.hypot(d[f"probe_{label}_core_cos"], d[f"probe_{label}_core_sin"])
                panels[2].loglog(frequency[1:] / median_gamma, np.maximum(probe[1:], 1e-30), label=label)
        panels[0].set(title=f"{r['optimizer']} / {r['arm']} / {r['initialization']}", xlabel="Updates", ylabel="Residual RMS")
        panels[1].set(xlabel="Updates", ylabel="Signed outward force / actual update")
        panels[2].set(xlabel="Angular frequency / median |gamma|", ylabel="Unit-RMS core probe response")
        for ax in panels:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "mechanism_windows.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.root / "consolidation"
    output.mkdir(exist_ok=True)
    folders = [str(p) for p in sorted(args.root.iterdir()) if (p / "latest.json").exists()]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        summarized = list(pool.map(summarize_case, folders))
    rows = [r for r, _ in summarized]
    windows = [w for _, records in summarized for w in records]
    choices = paired_choices(rows, "validation_320000") + paired_choices(rows, "window_rms_320000")
    if any("window_validation_320000" in r for r in rows):
        choices += paired_choices(rows, "window_validation_320000")
    nulls = [r for r in rows if r["arm"] == "raw" and r["rate_r"] == r["rate_g"]]
    choices += paired_choices(nulls, "window_rms_320000")
    chosen = {k for c in choices for k in c["cases"]}
    write_json(output / "choices.json", choices)
    write_json(output / "windows.json", windows)
    write_json(output / "counts.json", {"cases": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                                       "at_320k": sum("validation_320000" in r for r in rows), "representatives": len(chosen)})
    tasks = set()
    steps = [0, 1, 100, 1000, 20000, 80000, 160000, 320000]
    for row in rows:
        folder = Path(row["folder"])
        if "validation_320000" in row:
            tasks.add((str(folder), 320000, 1e-12, False))
        if row["key"] in chosen:
            for step in steps:
                if (folder / f"checkpoint_{step:09d}.npz").exists():
                    tasks.add((str(folder), step, 1e-12, False))
            for step in [20000, 320000]:
                for cutoff in [1e-10, 1e-14]:
                    tasks.add((str(folder), step, cutoff, False))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for i, result in enumerate(pool.map(analyze_one, sorted(tasks)), 1):
            if i % 25 == 0:
                print(json.dumps({"diagnostics_completed": i, "total": len(tasks)}), flush=True)
    mechanism = []
    for row in rows:
        folder = Path(row["folder"])
        path = folder / "analysis" / "metrics_000320000_tau1e-12.json"
        if path.exists():
            result = json.loads(path.read_text())
            row.update({"refit_320000": result["refit_validation"]["rms"], "rank_320000": result["rank"],
                        "refit_readout_l1_320000": result["refit_c_l1"]})
        if row["key"] in chosen:
            mechanism.extend(mechanism_rows(folder, steps))
    write_summary(rows, output / "summary.csv")
    write_json(output / "mechanism.json", mechanism)
    figures(rows, choices, mechanism, output / "figures")
    print(json.dumps({"complete": True, "output": str(output), "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
