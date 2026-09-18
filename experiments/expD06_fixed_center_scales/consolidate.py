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
from .analyze import DIAGNOSTICS_REVISION, analyze_one, write_summary
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
    # Verify the entire finite trajectory, including gaps between diagnostic windows.
    trace_windows(folder, [(0, latest["step"])])
    row["trace_verified_steps"] = latest["step"]
    ends = [s for s in [20000, 80000, 160000, 280000, 300000, 320000, 640000]
            if s <= latest["step"]]
    windows = trace_windows(folder, [(s - 20000, s) for s in ends])
    dense_path = folder / "validation_trace.json"
    dense = json.loads(dense_path.read_text()) if dense_path.exists() else []
    if case.dense_validation and latest["step"] >= 320000:
        if [r["step"] for r in dense] != list(range(260000, 320001, 1000)):
            raise ValueError(f"Incomplete late validation trace in {folder}")
        if not np.isfinite([r["validation_rms"] for r in dense]).all():
            raise ValueError(f"Nonfinite late validation trace in {folder}")
        row["validation_samples_verified"] = len(dense)
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
            for name in b.files:
                if np.issubdtype(b[name].dtype, np.floating) and b[name].dtype != np.float64:
                    raise ValueError(f"Non-FP64 checkpoint array: {folder}/{name}")
            row["checkpoint_floating_dtype"] = "float64"
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
    panels = [(opt, group) for opt in optimizers for group in groups]
    nrows = (len(panels) + 3) // 4
    fig, axes = plt.subplots(nrows, 4, squeeze=False, figsize=(17, 3.7 * nrows))
    scatter = None
    for i, opt in enumerate(optimizers):
        for j, group in enumerate(groups):
            ax = axes.flat[i * len(groups) + j]
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
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Paired geometric mean of training RMS over updates 300k–320k; red = numerical failure")
    fig.tight_layout(rect=(0, 0, .96, .94))
    if scatter is not None:
        fig.colorbar(scatter, cax=fig.add_axes([.97, .2, .008, .6]), label="log10 window RMS")
    fig.savefig(output / "lr_windows.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def spectral_evidence(rows, choices, output):
    """Export signed band histories and numerical audits of selected trajectories."""
    import matplotlib.pyplot as plt
    from .diagnostics import band_residuals
    cutoff_rows, audit_rows, band_rows = [], [], []
    selected = {k for c in choices for k in c["cases"]}
    score = "window_validation_320000" if any("window_validation_320000" in r for r in rows) else "window_rms_320000"
    displayed = {c["cases"][0] for c in choices if c["score_key"] == score}
    figure_records = []
    for row in rows:
        folder = Path(row["folder"])
        for path in sorted((folder / "analysis").glob("metrics_*_tau*.json")):
            m = json.loads(path.read_text())
            if m.get("diagnostics_revision") != DIAGNOSTICS_REVISION:
                continue
            audit_rows.append({"key": row["key"], "step": m["step"], "cutoff": m["cutoff"], **{
                field: m[field] for field in ["gradient_reconstruction_error", "perpendicular_reconstruction_error",
                                              "band_subtraction_error_norm", "cpu_gpu_prediction_max_difference"]}})
            if row["key"] in selected and m["step"] in [20000, 320000]:
                cutoff_rows.append({"key": row["key"], "step": m["step"], "cutoff": m["cutoff"],
                                    "refit_rms": m["refit_validation"]["rms"], "rank": m["rank"],
                                    "parallel_norm": m["gradient_parallel_norm"], "perpendicular_norm": m["gradient_perpendicular_norm"]})
        if row["key"] not in selected:
            continue
        history = []
        for step in [0, 1, 100, 1000, 20000, 80000, 160000, 320000]:
            path = folder / "analysis" / f"diagnostics_{step:09d}_tau1e-12.npz"
            if not path.exists():
                continue
            with np.load(path) as d, np.load(folder / f"checkpoint_{step:09d}.npz") as p:
                record = {"key": row["key"], "step": step, "bounds": d["band_bounds"].tolist()}
                for name in ["train", "parallel", "perpendicular", "refit"]:
                    _, split, _ = band_residuals(d[f"residual_{name}"] / np.sqrt(len(d["x_train"])))
                    record[f"{name}_band_rms"] = np.linalg.norm(split, axis=1).tolist()
                for region, mask in {"all": np.ones(len(p["gamma"]), bool), **core.geometry(row["n"]).masks}.items():
                    direction = np.where(mask, np.sign(p["gamma"]), 0.)
                    direction /= max(np.linalg.norm(direction), 1.)
                    for component in ["parallel", "perpendicular"]:
                        record[f"{region}_{component}_signed_force"] = (-d[f"band_gradient_{component}"] @ direction).tolist()
                for component in ["geometry", "readout"]:
                    record[f"{component}_predicted_descent"] = d[f"band_predicted_descent_{component}"].tolist()
                record["residual_energy_reconstruction_error"] = float(abs(d["band_energy"].sum() - np.mean(d["residual_train"]**2)))
                record["update_prediction_reconstruction_max_error"] = float(np.max(np.abs(
                    d["prediction_change_measured"] - d["prediction_change_readout"]
                    - d["prediction_change_geometry"] - d["prediction_change_interaction"])))
                history.append(record)
        band_rows.extend(history)
        if row["key"] not in displayed or not history:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
        labels = ["DC" if lo == 0 else str(lo) for lo, hi in history[-1]["bounds"]]
        for ax, component in zip(axes[0], ["train", "perpendicular", "refit"]):
            values = np.array([r[f"{component}_band_rms"] for r in history])
            im = ax.imshow(np.log10(np.maximum(values, 1e-20)), origin="lower", aspect="auto", vmin=-16, vmax=0)
            ax.set(xticks=range(0, len(labels), 2), xticklabels=labels[::2], yticks=range(len(history)),
                   yticklabels=[str(r["step"]) for r in history], xlabel="Band's lowest DFT index", ylabel="Updates",
                   title=f"{component}: log10 band RMS")
            fig.colorbar(im, ax=ax, shrink=.7)
        final = history[-1]
        for component in ["parallel", "perpendicular"]:
            axes[1, 0].plot(final[f"all_{component}_signed_force"], ".-", label=component)
        axes[1, 0].set(title="320k signed outward gradient force", ylabel="Positive = increase |lambda|")
        for component in ["readout", "geometry"]:
            axes[1, 1].plot(final[f"{component}_predicted_descent"], ".-", label=component)
        axes[1, 1].set(title="320k actual-update linearized descent", ylabel="Positive = reduce loss")
        for region in ["core", "halo", "corrected_halo"]:
            axes[1, 2].plot(final[f"{region}_perpendicular_signed_force"], ".-", label=region)
        axes[1, 2].set(title="320k perpendicular force by region", ylabel="Region-normalized outward force")
        for ax in axes[1]:
            ax.set(yscale="symlog", xticks=range(0, len(labels), 2), xticklabels=labels[::2], xlabel="Band's lowest DFT index")
            ax.set_yscale("symlog", linthresh=1e-18)
            ax.axhline(0, color="black", linewidth=.5)
            ax.legend(fontsize=8)
        fig.suptitle(f"{row['optimizer']} / {row['arm']} / {row['initialization']} / {row.get('epsilon_mode', 'legacy')} / seed {row['seed']}\n"
                     f"readout LR {row['rate_r']:.5g}, slope LR {row['rate_g']:.5g}; display chosen by {score}")
        name = f"spectral_{row['key']}.png"
        fig.savefig(output / "figures" / name, dpi=130)
        plt.close(fig)
        figure_records.append({"key": row["key"], "figure": name, "selection": score})
    write_json(output / "numerical_audit.json", audit_rows)
    write_json(output / "cutoff_sensitivity.json", cutoff_rows)
    write_json(output / "spectral_history.json", band_rows)
    write_json(output / "spectral_figures.json", figure_records)


def focused_comparisons(rows, output):
    """Show every crossed condition and paired epsilon contrast, without selecting by refit."""
    from .focused import manifest
    import matplotlib.pyplot as plt
    indexed = {r["key"]: r for r in rows}
    records = [{**r, **indexed[r["key"]]} for r in manifest() if r["key"] in indexed]
    if not records:
        return
    families = ["xavier", "xavier_a_uniform", "xavier_a_reference", "envelope"]
    conditions = [(arm, rr, rg) for arm in ["uniform", "both"] for rr, rg in [(1e-4, 1e-3), (1e-3, 1e-3), (1e-3, 1e-2)]]
    paired = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    fields = ["window_rms_320000", "lambda_median_320000", "refit_320000"]
    titles = ["Complete training-window RMS", "Median |lambda|", "Detached validation refit RMS"]
    for ax, field, title in zip(axes, fields, titles):
        values = np.full((4, 6), np.nan)
        for i, family in enumerate(families):
            for j, (arm, rr, rg) in enumerate(conditions):
                group = [r for r in records if r["kind"] == "primary" and r["initialization"] == family and r["arm"] == arm
                         and r["uniform_readout_lr"] == rr and r["lambda_lr"] == rg and field in r]
                if len(group) == 2:
                    values[i, j] = np.mean([np.log10(max(r[field], 1e-300)) for r in group])
        im = ax.imshow(values, aspect="auto")
        ax.set(yticks=range(4), yticklabels=families, xticks=range(6),
               xticklabels=[f"{a}\n{r:.0e}\n{g:.0e}" for a,r,g in conditions],
               xlabel="Map / uniform-coordinate readout LR / slope LR", title=title)
        ax.tick_params(axis="x", labelsize=7)
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(j, i, f"{10**value:.2g}", ha="center", va="center", fontsize=8,
                        bbox={"facecolor":"white", "alpha":.75, "edgecolor":"none", "pad":1})
        fig.colorbar(im, ax=ax, shrink=.6, label="log10 paired geometric mean")
    fig.suptitle("Fresh seeds 2 and 3; N=512 sine; rates labeled in uniform coordinates")
    fig.savefig(output / "figures" / "focused_conditions.png", dpi=150)
    plt.close(fig)
    for row in records:
        if row["kind"] != "epsilon_control":
            continue
        native = next(r for r in records if r["kind"] == "primary" and all(r[k] == row[k] for k in ["arm", "initialization", "seed", "rate_r", "rate_g"]))
        paired.append({"physical_epsilon_case": row["key"], "native_case": native["key"], **{
            f"{field}_physical_over_native": row[field] / native[field]
            for field in fields + ["window_validation_320000"] if field in row and field in native}})
    write_json(output / "epsilon_contrasts.json", paired)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    groups = defaultdict(list)
    for row in records:
        if row["initialization"] != "xavier_a_uniform" or row["arm"] == "both":
            continue
        label = f"{row['arm']} {row['rate_r']:g}/{row['rate_g']:g}, {row['epsilon_mode']} epsilon"
        groups[label].append(row)
    horizons = [20000, 80000, 160000, 320000]
    for label, pair in groups.items():
        if len(pair) != 2:
            continue
        for ax, field in zip(axes, ["lambda_median", "window_rms"]):
            if not all(f"{field}_{step}" in r for r in pair for step in horizons):
                continue
            values = np.array([[r[f"{field}_{step}"] for step in horizons] for r in pair])
            mean = np.sqrt(values.prod(axis=0))
            line, = ax.loglog(horizons, mean, ".-", label=label)
            ax.fill_between(horizons, values.min(axis=0), values.max(axis=0), color=line.get_color(), alpha=.12)
    axes[0].axhline(.25, color="black", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="Median |lambda|", xlabel="Updates")
    axes[1].set(ylabel="Training RMS over preceding 20k updates", xlabel="Updates")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Xavier on a with w=sqrt(h) a; physical Xavier gamma; geometric means and two-seed ranges")
    fig.savefig(output / "figures" / "focused_xavier_a_trajectories.png", dpi=150)
    plt.close(fig)


def trajectory_figures(rows, choices, mechanism, output):
    import matplotlib.pyplot as plt
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
    trajectory_figures(rows, choices, mechanism, output / "figures")
    spectral_evidence(rows, choices, output)
    focused_comparisons(rows, output)
    print(json.dumps({"complete": True, "output": str(output), "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
