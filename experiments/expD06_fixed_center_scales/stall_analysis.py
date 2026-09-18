"""Focused MSE, parameter, Fourier, and singular-direction evidence; no prose generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

from . import core, diagnostics, run
from .analyze import analyze_one
from .construction_reference import sine_coefficients
from .continue_stall import LABELS, SOURCE_STEP, learning_rate, source_case


GROUPS = [("adam", "both", .001), ("adam", "both", .01), ("adam", "raw", .001),
          ("gd", "both", .01), ("gd", "both", .1), ("gd", "raw", .01)]
STEPS = [0, 100, 1000, 20000, 80000, 160000, 320000]


def group_name(optimizer, arm, rate):
    return f"{optimizer}_{'scaled' if arm == 'both' else 'unscaled'}_{rate:g}"


def group_label(optimizer, arm, rate):
    return f"{'Scaled' if arm == 'both' else 'Unscaled'} training, {'Adam' if optimizer == 'adam' else 'GD'}, eta={rate:g}"


def complete_trace(folder, end):
    result = np.full((end, 5), np.nan)
    for path in sorted(folder.glob("trace_*.npz")):
        _, a, b = path.stem.split("_")
        a, b = int(a), min(int(b), end)
        if a >= b:
            continue
        with np.load(path) as data:
            result[a:b] = data["trace"][:b-a]
    if not np.isfinite(result).all():
        raise ValueError(f"Incomplete or nonfinite trace: {folder}, 0:{end}")
    return result


def save_plot(fig, output, name):
    fig.savefig(output / "figures" / f"{name}.png", dpi=160)
    plt.close(fig)


def positive(value):
    return np.maximum(value, 1e-32)


def historical_adam_audit(root, output):
    """Rank recorded Adam checkpoints separately from complete 20k windows."""
    best_endpoint, best_window = None, None
    for campaign in ["pilot", "focused"]:
        with (root/"runs"/campaign/"consolidation"/"summary.csv").open() as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            if row["optimizer"] != "adam":
                continue
            folder = root/"runs"/campaign/row["key"]
            latest = json.loads((folder/"latest.json").read_text())
            for m in latest["history"]:
                if not m["finite"] or m["step"] == 0:
                    continue
                mse = m["validation"]["rms"]**2
                if best_endpoint is None or mse < best_endpoint["mse"]:
                    best_endpoint = {"mse": mse, "step": m["step"], "campaign": campaign, "case": row}
            for field, value in row.items():
                if field.startswith("window_rms_") and value:
                    mse = float(value)**2
                    if best_window is None or mse < best_window["mse"]:
                        best_window = {"mse": mse, "end": int(field.split("_")[-1]),
                                       "window_steps": 20000, "campaign": campaign, "case": row}
    result = {"best_recorded_validation_checkpoint": best_endpoint,
              "best_recorded_complete_training_window": best_window,
              "scope": "Existing D06 pilot and focused Adam runs; no detached refits; different initializations and rates kept in the source configurations",
              "legacy_report_only": [
                  {"study": "D02", "target": "sine", "reported_relative_l2": 1.4e-5,
                   "note": "Construction initialization, FP32 Adam; raw result files absent in local checkout"},
                  {"study": "D05", "target": "exp", "reported_relative_l2": 7.069e-5,
                   "note": "Best reported non-oracle checkpoint, different target and geometry; raw full-matrix CSV absent locally"}]}
    run.write_json(output/"historical_adam_audit.json", result)


def load_entry(folder, end, steps, label, group, offset=0):
    case = run.Case(**json.loads((folder / "case.json").read_text()))
    paths = sorted(folder.glob("checkpoint_*.npz"))
    history_steps = [int(p.stem.split("_")[1]) for p in paths if int(p.stem.split("_")[1]) <= end]
    if offset:
        history_steps = [s for s in history_steps if s % 20000 == 0 or s == 1000 or s == end]
    fields = ["c", "gamma", "lambda", "lambda_travel", "readout_travel"]
    history = {k: [] for k in fields}
    validation = []
    for step in history_steps:
        with np.load(folder / f"checkpoint_{step:09d}.npz") as data:
            for name in fields:
                history[name].append(data[name])
        m = json.loads((folder / f"metrics_{step:09d}.json").read_text())
        validation.append(m["validation"]["rms"]**2)
    history = {k: np.array(v) for k, v in history.items()}
    detail, metrics = [], []
    for step in steps:
        with np.load(folder / "analysis" / f"diagnostics_{step:09d}_tau1e-12.npz") as d:
            detail.append(dict(d))
        metrics.append(json.loads((folder / "analysis" / f"metrics_{step:09d}_tau1e-12.json").read_text()))
    return {"case": case, "folder": folder, "end": end, "steps": np.array(steps), "offset": offset,
            "history_steps": np.array(history_steps), "history": history, "validation_mse": validation,
            "detail": detail, "metrics": metrics, "trace": complete_trace(folder, end),
            "label": label, "group": group}


def evidence_record(entry):
    c = entry["case"]
    history = entry["history"]
    records = []
    for step, d, metric in zip(entry["steps"], entry["detail"], entry["metrics"]):
        with np.load(entry["folder"] / f"checkpoint_{step:09d}.npz") as state:
            p = dict(state)
        readout = d["prediction_change_readout"]
        bias = d["prediction_change_bias"]
        geometry = d["prediction_change_geometry"]
        weight = readout-bias
        norm = lambda v: float(np.linalg.norm(v))
        rms = lambda v: float(np.sqrt(np.mean(v*v)))
        cg, lg = d["gradient_readout"], d["gradient_lambda"]
        dc, dl = p["next_delta_c"], p["next_delta_lambda"]
        cosine = lambda a, b: float(a@b/(norm(a)*norm(b))) if norm(a)*norm(b) else None
        records.append({"step": int(step), "absolute_step": int(step)+entry["offset"],
            "validation_mse": metric["validation"]["rms"]**2,
            "refit_validation_mse": metric["refit_validation"]["rms"]**2,
            "rank": metric["rank"], "readout_l1": float(np.abs(p["c"]).sum()),
            "refit_l1": metric["refit_c_l1"], "readout_gradient_norm": norm(cg),
            "lambda_gradient_norm": norm(lg), "parallel_gradient_norm": norm(d["gradient_parallel"]),
            "perpendicular_gradient_norm": norm(d["gradient_perpendicular"]),
            "readout_descent_cosine": cosine(-cg, dc), "lambda_descent_cosine": cosine(-lg, dl),
            "readout_update_rms": rms(dc), "lambda_update_rms": rms(dl), "bias_update": float(dc[0]),
            "prediction_update_rms": {"bias": rms(bias), "weights": rms(weight), "geometry": rms(geometry),
                                      "combined": rms(d["prediction_change_measured"])},
            "readout_geometry_cosine": cosine(readout, geometry),
            "readout_linear_loss_change": float(cg@dc),
            "readout_quadratic_cost": float(.5*np.mean(readout**2)),
            "readout_only_exact_loss_change": float(cg@dc+.5*np.mean(readout**2)),
            "joint_measured_loss_change": float(np.mean(d["residual_train"]*d["prediction_change_measured"])
                                                  + .5*np.mean(d["prediction_change_measured"]**2)),
            "prediction_change_closure": norm(d["prediction_change_measured"]-readout-geometry-d["prediction_change_interaction"]),
            "gradient_band_closure": norm(d["band_gradient_lambda"].sum(axis=0)-lg),
            "parseval_error": float(abs(d["band_energy"].sum()-np.mean(d["residual_train"]**2))),
            "regions": {region: {"weight_rms": rms(p["c"][1:][mask]),
                                  "lambda_rms": rms(p["lambda"][mask]),
                                  "weight_update_rms": rms(dc[1:][mask]),
                                  "lambda_update_rms": rms(dl[mask])}
                        for region, mask in core.geometry(c.n).masks.items()},
            "adam_epsilon_fraction": {block: float(np.mean(p[f"adam_{block}_sqrt_v_over_epsilon"] <= 1))
                                      for block in ["readout", "slope"] if f"adam_{block}_sqrt_v_over_epsilon" in p}})
    mse = 2*entry["trace"][:, 0]
    windows = [{"end": end, "mean_mse": float(mse[end-20000:end].mean()),
                "mse_quantiles": np.quantile(mse[end-20000:end], [0, .1, .5, .9, 1]).tolist()}
               for end in range(20000, len(mse)+1, 20000)]
    net = {k: float(np.sqrt(np.mean((history[k][-1]-history[k][0])**2))) for k in ["c", "lambda"]}
    travel = {k: float(np.sqrt(np.mean((history[k+"_travel"][-1]-history[k+"_travel"][0])**2)))
              for k in ["readout", "lambda"]}
    g = core.geometry(c.n)
    cs, gs, _, er, eg = run.case_settings(c, g)
    eta = float(learning_rate(entry["end"], entry["group"].endswith("decay"))) if entry["offset"] else c.rate_r
    power = 2 if c.optimizer == "gd" else 1
    return {"group": entry["group"], "label": entry["label"], "seed": c.seed, "source": str(entry["folder"]),
            "end": entry["end"], "offset": entry["offset"], "case": c.__dict__, "windows": windows,
            "effective_rates_at_end": {"shared_coordinate_rate": eta, "readout_prefactors": (eta*cs**power).tolist(),
                "gamma_prefactor": 0. if entry["group"].startswith("frozen") else eta*gs**power,
                "physical_readout_epsilon": (er/cs).tolist() if c.optimizer == "adam" else None,
                "physical_gamma_epsilon": float(eg/gs) if c.optimizer == "adam" else None},
            "checkpoints": records, "net_displacement_rms": net, "accumulated_travel_rms": travel,
            "frozen_gamma_max_change": float(np.max(np.abs(history["gamma"]-history["gamma"][0]))) }


def error_figure(entries, output, name):
    optimizers = sorted({e["case"].optimizer for e in entries})
    fig, axes = plt.subplots(len(optimizers), 2, figsize=(13, 4*len(optimizers)), squeeze=False, layout="constrained")
    groups = list(dict.fromkeys(e["group"] for e in entries))
    colors = dict(zip(groups, plt.get_cmap("tab10").colors))
    for e in entries:
        ax = axes[optimizers.index(e["case"].optimizer), e["case"].seed]
        mse = 2*e["trace"][:, 0]
        n = len(mse)//1000
        bins = mse[:n*1000].reshape(n, 1000)
        x = (np.arange(n)+1)*1000+e["offset"]
        color = colors[e["group"]]
        ax.fill_between(x, positive(np.quantile(bins, .1, axis=1)), positive(np.quantile(bins, .9, axis=1)), color=color, alpha=.12)
        ax.semilogy(x, positive(bins.mean(axis=1)), color=color, label=e["label"])
        ax.semilogy(e["steps"]+e["offset"], [max(m["refit_validation"]["rms"]**2, 1e-32) for m in e["metrics"]],
                    ":o", color=color, markersize=3)
        ax.scatter(e["history_steps"]+e["offset"], positive(e["validation_mse"]), color=color, s=8, marker="x", alpha=.5)
    for i, row in enumerate(axes):
        for seed, ax in enumerate(row):
            ax.set(title=f"{optimizers[i].upper()}, seed {seed}", xlabel="Total training updates", ylabel="MSE")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Mean training MSE in 1k bins; shading: 10–90% of individual steps\nCrosses: validation checkpoints; dotted: detached validation refit")
    save_plot(fig, output, name)


def dense_figure(pair, output):
    """Show actual adjacent parameter states, not interpolation between checkpoints."""
    g = core.geometry(512)
    indices = [int(np.argmin(abs(g.centers-x))) for x in [-.75, 0, .75]] + [0, g.width-1]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), layout="constrained")
    records = []
    for e in pair:
        end, begin = e["end"], e["end"]-2048
        fragments = []
        for path in sorted(e["folder"].glob("dense_*.npz")):
            _, a, b = path.stem.split("_")
            if int(a) >= end or int(b) <= begin:
                continue
            with np.load(path) as data:
                mask = (data["step"] >= begin) & (data["step"] < end)
                fragments.append({k: data[k][mask] for k in data.files})
        d = {k: np.concatenate([f[k] for f in fragments]) for k in fragments[0]}
        np.testing.assert_array_equal(d["step"], np.arange(begin, end))
        row = axes[e["case"].seed]
        x = d["step"]-begin
        row[0].semilogy(x, positive(2*e["trace"][begin:end, 0]))
        row[0].set(title=f"Seed {e['case'].seed}: individual-step MSE", ylabel="Training MSE")
        for j in indices:
            row[1].plot(x, d["c"][:, j+1]-d["c"][0, j+1], label=f"center={g.centers[j]:.2f}")
            row[2].plot(x, d["gamma"][:, j]-d["gamma"][0, j], label=f"center={g.centers[j]:.2f}")
        row[1].plot(x, d["c"][:, 0]-d["c"][0, 0], "k", lw=.7, label="bias")
        row[1].set(title="Readout movement from window start", ylabel="Change in physical coefficient")
        row[2].set(title="Slope movement from window start", ylabel="Change in gamma")
        cosines = {}
        for block, grad, update in [("readout", "gradient_c", "delta_c"), ("lambda", "gradient_lambda", "delta_lambda")]:
            denominator = np.linalg.norm(d[grad], axis=1)*np.linalg.norm(d[update], axis=1)
            cos = np.divide(-np.sum(d[grad]*d[update], axis=1), denominator,
                            out=np.full(len(x), np.nan), where=denominator != 0)
            row[3].plot(x, cos, label=block, alpha=.7, lw=.6)
            finite = cos[np.isfinite(cos)]
            cosines[block] = {"median": float(np.median(finite)) if len(finite) else None,
                              "uphill_fraction": float(np.mean(finite < 0)) if len(finite) else None}
        row[3].set(title="Step alignment with negative gradient", ylabel="Cosine", ylim=(-1.1, 1.1))
        for ax in row:
            ax.set(xlabel=f"Steps since {begin+e['offset']:,}")
            ax.grid(alpha=.2)
        for ax in row[1:]: ax.legend(fontsize=6)
        records.append({"group": e["group"], "seed": e["case"].seed, "begin": begin, "end": end,
                        "descent_alignment": cosines,
                        "gamma_max_change": float(np.max(abs(d["gamma"]-d["gamma"][0]))),
                        "readout_max_change": float(np.max(abs(d["c"]-d["c"][0])))})
    fig.suptitle(pair[0]["label"]+" — every saved step in the final 2,048-step window")
    save_plot(fig, output, pair[0]["group"]+"_dense")
    return records


def group_figures(pair, output, construction):
    g = core.geometry(512)
    group = pair[0]["group"]
    title = pair[0]["label"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for e in pair:
        row = axes[e["case"].seed]
        h = e["history"]
        for ax, field, label in zip(row[:2], ["c", "gamma"], ["Physical readout w", "Physical slope gamma"]):
            data = h[field][:, 1:] if field == "c" else h[field]
            bound = max(np.abs(data).max(), 1e-6)
            im = ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest", cmap="coolwarm",
                           norm=SymLogNorm(bound/1000, vmin=-bound, vmax=bound), extent=[g.centers[0], g.centers[-1], -.5, len(data)-.5])
            indices = np.unique(np.linspace(0, len(data)-1, min(7, len(data))).astype(int))
            ax.set(yticks=indices, yticklabels=(e["history_steps"][indices]+e["offset"]).astype(str), xlabel="Fixed center", ylabel="Saved update", title=label)
            for edge in [-1, 1]: ax.axvline(edge, color="black", lw=.6)
            fig.colorbar(im, ax=ax, shrink=.7)
        for q, label in [(.1, "10%"), (.5, "median"), (.9, "90%"), (1., "max")]:
            row[2].semilogy(e["history_steps"]+e["offset"], positive(np.quantile(abs(h["lambda"]), q, axis=1)), label=label)
        row[2].axhline(.25, color="black", ls=":", label="construction 0.25")
        row[2].set(title=f"Seed {e['case'].seed}: |lambda|", xlabel="Total updates")
        row[2].legend(fontsize=7)
    fig.suptitle(title+" — checkpoint parameter histories (rows are saved states)")
    save_plot(fig, output, group+"_parameters")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    for e in pair:
        seed = e["case"].seed
        p = e["history"]
        sign = np.sign(p["gamma"][-1])
        live = p["c"][-1, 1:]*sign
        refit = e["detail"][-1]["readout_refit"][1:]*sign
        for ax, mask, label in [(axes[seed, 0], g.core, "Core"), (axes[seed, 1], ~g.core, "Halo")]:
            locations = g.centers[mask] if label == "Core" else np.arange(mask.sum())
            ax.plot(locations, live[mask], ".", ms=3, label="Trained, sign canonicalized")
            ax.plot(locations, refit[mask], ".", ms=3, label="Refit of learned geometry")
            ax.plot(locations, construction["c"][1:][mask], ".", ms=3, label="Construction, lambda=0.25")
            ax.plot(locations, g.alpha[1:][mask], "k_", alpha=.5, label="Reference envelope ±alpha")
            ax.plot(locations, -g.alpha[1:][mask], "k_", alpha=.5)
            ax.set(yscale="symlog", xlabel="Fixed center", ylabel="Physical readout coefficient",
                   title=f"Seed {seed}: {label}")
            if label == "Halo":
                ticks = [0, g.radius-1, g.radius, 2*g.radius-1]
                ax.set(xticks=ticks, xticklabels=[f"{g.centers[mask][i]:.3f}" for i in ticks],
                       xlabel="Halo slots, labeled by fixed center (core omitted)")
                ax.axvline(g.radius-.5, color="black", lw=.6)
            else:
                ax.set_title(f"Seed {seed}: core; bias live/refit/reference = "
                             f"{p['c'][-1,0]:.3g} / {e['detail'][-1]['readout_refit'][0]:.3g} / 0", fontsize=9)
            ax.set_yscale("symlog", linthresh=.01)
            ax.grid(alpha=.2)
        axes[seed, 0].legend(fontsize=7)
    fig.suptitle(title+" — final coefficients; construction and learned slopes differ")
    save_plot(fig, output, group+"_coefficients")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for e in pair:
        row = axes[e["case"].seed]
        bounds = e["detail"][0]["band_bounds"]
        for ax, part in zip(row, ["train", "parallel", "perpendicular"]):
            energy = []
            for d in e["detail"]:
                _, bands, _ = diagnostics.band_residuals(d[f"residual_{part}"]/np.sqrt(len(d["x_train"])))
                energy.append(np.sum(bands*bands, axis=1))
            im = ax.imshow(np.log10(positive(energy)), origin="lower", aspect="auto", interpolation="nearest", vmin=-30, vmax=0)
            ax.set(title=f"Seed {e['case'].seed}: {part}", yticks=range(len(e["steps"])),
                   yticklabels=e["steps"]+e["offset"], xticks=range(0, len(bounds), 2),
                   xticklabels=bounds[::2, 0].astype(int), xlabel="Band's lowest DFT index", ylabel="Updates")
            fig.colorbar(im, ax=ax, label="log10 band MSE", shrink=.75)
    fig.suptitle(title+" — residual spectrum; parallel = readout-accessible at the stated cutoff")
    save_plot(fig, output, group+"_spectra")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for e in pair:
        row = axes[e["case"].seed]
        records = evidence_record(e)["checkpoints"]
        x = e["steps"]+e["offset"]
        for field, label in [("readout_gradient_norm", "readout"), ("lambda_gradient_norm", "lambda"),
                              ("perpendicular_gradient_norm", "lambda, out-of-span")]:
            row[0].semilogy(x, positive([r[field] for r in records]), ".-", label=label)
        row[0].set(title=f"Seed {e['case'].seed}: gradient norms", ylabel="Norm in stated coordinates")
        for field in ["bias", "weights", "geometry", "combined"]:
            row[1].semilogy(x, positive([r["prediction_update_rms"][field] for r in records]), ".-", label=field)
        row[1].set(title="Actual one-step function movement", ylabel="RMS prediction change")
        for field, label in [("readout_descent_cosine", "readout step vs -gradient"),
                             ("lambda_descent_cosine", "lambda step vs -gradient"),
                             ("readout_geometry_cosine", "readout vs geometry function steps")]:
            row[2].plot(x, [r[field] for r in records], ".-", label=label)
        row[2].set(title="Directions and cancellation", ylim=(-1.1, 1.1), ylabel="Cosine")
        for ax in row:
            ax.set(xlabel="Total updates")
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    fig.suptitle(title+" — checkpoint gradients and actual optimizer steps")
    save_plot(fig, output, group+"_gradients")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for e in pair:
        d = e["detail"][-1]
        row = axes[e["case"].seed]
        bins = d["band_bounds"][:, 0]
        row[0].semilogy(bins, positive(np.linalg.norm(d["band_gradient_readout"], axis=1)), ".-", label="readout")
        for part in ["parallel", "perpendicular"]:
            row[0].semilogy(bins, positive(np.linalg.norm(d[f"band_gradient_{part}"], axis=1)), ".-", label="lambda: "+part)
        row[0].set(title=f"Seed {e['case'].seed}: band gradient norms")
        for part in ["readout", "geometry", "bias"]:
            row[1].plot(bins, d[f"band_predicted_descent_{part}"], ".-", label=part)
        row[1].set_yscale("symlog", linthresh=1e-18)
        row[1].set(title="Signed actual-update descent", ylabel="Positive = linearized loss reduction")
        frequencies = np.arange(len(d["probe_raw_all_cos"]))
        for part in ["raw", "perpendicular"]:
            response = np.hypot(d[f"probe_{part}_all_cos"], d[f"probe_{part}_all_sin"])
            row[2].loglog(frequencies[1:], positive(response[1:]), label=part)
        row[2].set(title="Unit-RMS Fourier sensitivity", xlabel="DFT index", ylabel="Tangent response norm")
        for ax in row:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        for ax in row[:2]:
            ax.set_xscale("symlog", linthresh=1)
            ax.set(xlabel="Band's lowest DFT index")
    fig.suptitle(title+" — final Fourier gradients and sensitivity (tiny projected values need cutoff checks)")
    save_plot(fig, output, group+"_fourier_gradients")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for e in pair:
        row = axes[e["case"].seed]
        for step, d in zip(e["steps"], e["detail"]):
            row[0].semilogy(d["singular_values"], label=f"{step+e['offset']} updates")
        d = e["detail"][-1]
        s = d["singular_values"]
        keep = s > 1e-12*s[0]
        row[1].loglog(positive(s[keep]), positive(d["singular_residual_coefficients"][keep]**2), ".", label="Retained")
        row[1].loglog(positive(s[~keep]), positive(d["singular_residual_coefficients"][~keep]**2), "x", label="Discarded at cutoff")
        row[1].axvline(1e-12*s[0], color="black", ls=":")
        row[0].set(title=f"Seed {e['case'].seed}: singular values of A D", xlabel="Singular index", ylabel="Singular value")
        row[1].set(title="Where the final live residual lies", xlabel="Singular value of A D", ylabel="Residual MSE in singular direction")
        for ax in row:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    fig.suptitle(title+" — reference-scaled features, including bias")
    save_plot(fig, output, group+"_singular")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--continuations", action="store_true")
    parser.add_argument("--additional-steps", type=int, default=0, help="Fixed common continuation horizon; zero uses latest complete 20k boundary")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.root / "runs" / ("stall_continuation_analysis" if args.continuations else "stall_analysis")
    (output / "figures").mkdir(parents=True, exist_ok=True)
    run.write_json(output/"analysis_provenance.json", {
        "diagnostics_revision": 3, "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob("*.py")},
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "evaluation_role": "Training objective and diagnostic validation only; no final test selection"})
    specs = []
    if args.continuations:
        end = min(json.loads((args.root/"runs"/"stall"/label/source_case(seed).key/"latest.json").read_text())["step"]
                  for label in LABELS for seed in [0, 1])//20000*20000
        if args.additional_steps:
            if args.additional_steps > end or args.additional_steps % 20000:
                raise ValueError("Requested horizon must be a completed common 20k boundary")
            end = args.additional_steps
        if end < 20000:
            raise ValueError("Wait for every continuation to complete at least 20k updates")
        steps = sorted({0, 20000, end, *[s for s in [80000, 160000, 240000, 400000, 720000] if s <= end]})
        for label in LABELS:
            for seed in [0, 1]:
                folder = args.root/"runs"/"stall"/label/source_case(seed).key
                display = ("Frozen geometry" if label.startswith("frozen") else "Joint training")
                display += ", decaying LR" if label.endswith("decay") else ", constant LR"
                specs.append((folder, end, steps, display, label, SOURCE_STEP))
    else:
        historical_adam_audit(args.root, output)
        with (args.root/"runs"/"pilot"/"consolidation"/"summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        shared = [r for r in rows if r["initialization"] == "envelope" and r["rate_r"] == r["rate_g"]
                  and r["n"] == "512" and r["target"] == "sine" and r["seed"] in ["0", "1"]
                  and r["arm"] in ["raw", "both"] and r["epsilon_mode"] == "legacy" and not r["bias_rate"]]
        run.write_json(output/"shared_rate_grid.json", shared)
        for optimizer, arm, rate in GROUPS:
            for seed in [0, 1]:
                case = run.Case(optimizer=optimizer, arm=arm, rate_r=rate, rate_g=rate, initialization="envelope", seed=seed)
                specs.append((args.root/"runs"/"pilot"/case.key, 320000, STEPS,
                              group_label(optimizer, arm, rate), group_name(optimizer, arm, rate), 0))
    tasks = [(str(folder), step, 1e-12, False) for folder, _, steps, *_ in specs for step in steps]
    tasks += [(str(folder), step, cutoff, False) for folder, end, _, _, _, _ in specs
              for step in [20000, end] for cutoff in [1e-10, 1e-14]]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        metrics = list(pool.map(analyze_one, sorted(set(tasks))))
    run.write_json(output/"cutoff_and_numerical_audit.json", metrics)
    construction = sine_coefficients(dps=80, quadrature_degree=9)
    low = sine_coefficients(dps=50, quadrature_degree=7)
    difference = float(np.max(abs(construction["c"]-low["c"])))
    if difference > 1e-14:
        raise ValueError("Construction coefficients have not stabilized with precision/quadrature refinement")
    x = diagnostics.midpoint_grid(32768)
    y = core.target(x, "sine", np)
    prediction = diagnostics.prediction(x, construction["centers"], construction["c"], construction["gamma"])
    run.save_arrays(output/"construction.npz", **construction)
    run.write_json(output/"construction_verification.json", {"coefficient_refinement_max_difference": difference,
        "validation": diagnostics.errors(prediction, y), "coefficient_l1": float(np.abs(construction["c"]).sum()),
        "source": "theorem_for_sam.pdf A.2, A.26, B.3-B.6, B.16-B.17", "dps_pair": [50, 80], "quadrature_degrees": [7, 9]})
    entries = [load_entry(*spec) for spec in specs]
    evidence = [evidence_record(e) for e in entries]
    run.write_json(output/"evidence.json", evidence)
    for e in entries:
        run.save_arrays(output/f"{e['group']}_s{e['case'].seed}_history.npz", **e["history"],
                        step=e["history_steps"]+e["offset"], validation_mse=e["validation_mse"])
        keep = [k for k in e["detail"][0] if k.startswith(("band_", "force_", "probe_", "singular_", "gradient_", "next_delta_"))
                or k in ["readout_refit", "residual_train", "residual_parallel", "residual_perpendicular", "residual_refit"]]
        run.save_arrays(output/f"{e['group']}_s{e['case'].seed}_diagnostics.npz",
                        step=e["steps"]+e["offset"], **{k: np.stack([d[k] for d in e["detail"]]) for k in keep})
    error_figure(entries, output, "optimization")
    dense_records = []
    for group in dict.fromkeys(e["group"] for e in entries):
        pair = [e for e in entries if e["group"] == group]
        group_figures(pair, output, construction)
        if args.continuations:
            dense_records.extend(dense_figure(pair, output))
    if dense_records:
        run.write_json(output/"dense_evidence.json", dense_records)
    print(json.dumps({"output": str(output), "cases": len(entries), "diagnostics": len(tasks),
                      "end": entries[0]["end"]}), flush=True)


if __name__ == "__main__":
    main()
