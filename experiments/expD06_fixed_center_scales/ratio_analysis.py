"""Detached rate-campaign evidence and figures; never writes scientific prose."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import numpy as np
from scipy.linalg import svd

from . import analyze, core, diagnostics, ratio, readout_solvers, run


def update_budget(x, y, centers, h, c, gamma, dc, dl):
    """Exact finite-update contributions, including the bilinear interaction."""
    a = diagnostics.features(x, centers, gamma)
    changed = diagnostics.features(x, centers, gamma + dl / h)
    r = a @ c - y
    pieces = np.stack((a @ dc, (changed - a) @ c, (changed - a) @ dc))
    measured = changed @ (c + dc) - a @ c
    joint = pieces.sum(axis=0)
    variations = np.vstack((pieces, joint))
    linear = 2 * variations @ r / len(x)
    quadratic = np.mean(variations**2, axis=1)
    return r, pieces, {"mse_change": linear + quadratic, "linear_mse_change": linear,
                      "quadratic_mse_cost": quadratic,
                      "closure_max": float(np.max(np.abs(joint - measured))),
                      "measured_mse_change": float(np.mean((r + measured)**2 - r**2))}


def read_dense(folder, end):
    fragments = []
    for path in sorted(folder.glob("dense_*.npz")):
        _, lo, hi = path.stem.split("_")
        if int(hi) <= end - 2048 or int(lo) >= end:
            continue
        with np.load(path) as data:
            keep = (data["step"] >= end - 2048) & (data["step"] < end)
            if keep.any():
                fragments.append({k: data[k][keep] for k in data.files})
    if not fragments:
        raise ValueError(f"Missing dense evidence: {folder}, {end}")
    result = {k: np.concatenate([d[k] for d in fragments]) for k in fragments[0]}
    # Interrupted saves can overlap; the latest deterministic copy wins.
    order = np.argsort(result["step"], kind="stable")
    result = {k: v[order] for k, v in result.items()}
    _, indices = np.unique(result["step"][::-1], return_index=True)
    indices = np.sort(len(result["step"]) - 1 - indices)
    result = {k: v[indices] for k, v in result.items()}
    np.testing.assert_array_equal(result["step"], np.arange(end - 2048, end))
    return result


def history(folder, end):
    """Join actual ancestor checkpoints, retaining explicit schedule rates."""
    meta = json.loads((folder / "schedule.json").read_text()) if (folder / "schedule.json").exists() else None
    rows = {}
    if meta and meta["source"]:
        parent = history(Path(meta["source"]), meta["source_step"])
        rows = {int(t): {k: parent[k][i] for k in parent if k != "step"} for i, t in enumerate(parent["step"])}
    case = run.Case(**json.loads((folder / "case.json").read_text()))
    knots = meta["knots"] if meta else None
    if (folder / "feedback.json").exists():
        knots = json.loads((folder / "feedback.json").read_text())["knots"]
    for path in sorted(folder.glob("checkpoint_*.npz")):
        t = int(path.stem.split("_")[-1])
        if t > end or (t and t < 1000):
            continue
        with np.load(path) as cp:
            rates = np.asarray(ratio.schedule(t, knots)) if knots else np.array([case.rate_r, case.rate_g])
            rows[t] = {"c": cp["c"], "gamma": cp["gamma"], "eta_a": rates[0], "eta_lambda": rates[1],
                       "validation_mse": float(np.mean((cp["prediction_validation"] -
                           core.target(diagnostics.midpoint_grid(case.validation_points), case.target, np))**2))}
    times = sorted(rows)
    return {"step": np.array(times), **{k: np.stack([rows[t][k] for t in times]) for k in rows[times[0]]}}


def dense_analysis(folder, end, output, case):
    dense = read_dense(folder, end)
    knots = json.loads((folder / "schedule.json").read_text())["knots"]
    if (folder / "feedback.json").exists():
        knots = json.loads((folder / "feedback.json").read_text())["knots"]
    rates = np.asarray(jax.vmap(lambda t: ratio.schedule(t, knots))(dense["step"]))
    run.save_arrays(output / "dense_parameters.npz", **{k: dense[k] for k in ("step", "c", "gamma")},
                    eta_a=rates[:, 0], eta_lambda=rates[:, 1])
    g = core.geometry(case.n)
    x = np.linspace(-1, 1, case.n * case.samples_per_cell + 1)
    y = core.target(x, case.target, np)
    root_m = np.sqrt(len(x))
    basis_a = diagnostics.features(x, g.centers, dense["gamma"][0]) / root_m
    u, singular, vh = svd(basis_a * g.d, full_matrices=False)
    rows = []
    for index in np.arange(0, 2048, 32):
        c, gamma, dc, dl = (dense[k][index] for k in ("c", "gamma", "delta_c", "delta_lambda"))
        r, pieces, budget = update_budget(x, y, g.centers, g.h, c, gamma, dc, dl)
        normalized = r / root_m
        a = diagnostics.features(x, g.centers, gamma) / root_m
        distances = x[:, None] - g.centers
        e = np.exp(-2 * np.abs(distances * gamma))
        j = c[1:] * distances * (4 * e / (1 + e)**2) / (g.h * root_m)
        bounds, rb, _ = diagnostics.band_residuals(normalized)
        coefficients = u.T @ normalized
        retained = singular > 1e-12 * singular[0]
        parallel = u[:, retained] @ coefficients[retained]
        bands_c, bands_l = rb @ a, rb @ j
        norm_w, norm_g = np.linalg.norm(pieces[0]), np.linalg.norm(pieces[1])
        alignment = []
        for delta, gradient in [(dc, dense["gradient_c"][index]), (dl, dense["gradient_lambda"][index])]:
            alignment.append(float(-delta @ gradient / max(np.linalg.norm(delta) * np.linalg.norm(gradient), 1e-300)))
        rows.append({"step": dense["step"][index], **budget, "residual_mse": normalized @ normalized,
                     "singular_residual_coefficients": coefficients,
                     "singular_update_coefficients": (pieces / root_m) @ u,
                     "band_energy": np.sum(rb**2, axis=1),
                     "band_gradient_c": bands_c, "band_gradient_lambda": bands_l,
                     "band_signed_readout_descent": -bands_c @ dc,
                     "band_signed_geometry_descent": -bands_l @ dl,
                     "gradient_lambda_parallel_fixed_basis": j.T @ parallel,
                     "gradient_lambda_perpendicular_fixed_basis": j.T @ (normalized - parallel),
                     "gradient_sum_error": np.array([np.linalg.norm(bands_c.sum(axis=0)-dense["gradient_c"][index]),
                                                     np.linalg.norm(bands_l.sum(axis=0)-dense["gradient_lambda"][index])]),
                     "readout_geometry_cosine": pieces[0] @ pieces[1] / max(norm_w * norm_g, 1e-300),
                     "descent_alignment": alignment})
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    run.save_arrays(output / "dense_mechanism.npz", **arrays, singular_values=singular, band_bounds=bounds,
                    basis_c=dense["c"][0], basis_gamma=dense["gamma"][0], basis_vh=vh,
                    mse_piece_order=np.array(["readout", "geometry", "interaction", "joint"]))
    return {"saved_states": 2048, "analyzed_states": 64, "basis_step": int(dense["step"][0]),
            "maximum_prediction_closure_error": float(np.max(arrays["closure_max"])),
            "maximum_gradient_sum_error": np.max(arrays["gradient_sum_error"], axis=0).tolist(),
            "mean_mse_change": np.mean(arrays["mse_change"], axis=0).tolist(),
            "mean_linear_mse_change": np.mean(arrays["linear_mse_change"], axis=0).tolist(),
            "mean_quadratic_mse_cost": np.mean(arrays["quadratic_mse_cost"], axis=0).tolist(),
            "median_readout_geometry_cosine": float(np.median(arrays["readout_geometry_cosine"])),
            "median_descent_alignment": np.median(arrays["descent_alignment"], axis=0).tolist()}


def analyze_case(task):
    folder, end, output = Path(task[0]), task[1], Path(task[2])
    output.mkdir(parents=True, exist_ok=True)
    case = run.Case(**json.loads((folder / "case.json").read_text()))
    meta = json.loads((folder / "schedule.json").read_text())
    ancestry = history(folder, end)
    run.save_arrays(output / "history.npz", **ancestry, centers=core.geometry(case.n).centers)
    windows = []
    for right in range((meta["source_step"] // 20000 + 1) * 20000, end + 1, 20000):
        losses = run.trace_window_losses(folder, right - 20000, right)
        if losses is None:
            raise ValueError(f"Incomplete scientific trace at {folder}, {right}")
        windows.append({"end": right, "mse_mean": float(2 * np.mean(losses)),
                        "mse_quantiles": np.quantile(2 * losses, [0, .1, .5, .9, 1]).tolist()})
    cutoffs = [analyze.analyze_one((str(folder), end, tau, False)) for tau in (1e-10, 1e-12, 1e-14)]
    with np.load(folder / "analysis" / f"diagnostics_{end:09d}_tau1e-12.npz") as cp:
        keep = [k for k in cp.files if k.startswith(("band_", "force_", "singular_", "gradient_", "probe_"))
                or k in ("readout_refit", "residual_train", "residual_parallel", "residual_perpendicular")]
        run.save_arrays(output / "endpoint.npz", **{k: cp[k] for k in keep})
    detail = dense_analysis(folder, end, output, case)
    g = core.geometry(case.n)
    with np.load(folder / f"checkpoint_{end:09d}.npz") as cp:
        ra, rg = float(cp["eta_a"]), float(cp["eta_lambda"])
        physical = {"eta_a": ra, "eta_lambda": rg, "ordinary_readout_factor": ra * np.sqrt(g.ordinary_alpha),
                    "bias_factor": ra * g.d[0], "readout_factor_range": [float(ra*g.d[1:].min()), float(ra*g.d[1:].max())],
                    "gamma_factor": rg / g.h, "lambda_median": float(np.median(np.abs(cp["lambda"]))),
                    "lambda_quantiles": np.quantile(np.abs(cp["lambda"]), [0, .1, .5, .9, 1]).tolist()}
        physical["adam_moment_over_epsilon_quantiles"] = {
            block: np.quantile(cp[f"adam_{block}_sqrt_v_over_epsilon"], [0, .1, .5, .9, 1]).tolist()
            for block in ("readout", "slope")}
        # Refinement is diagnostic only: never feed the refitted c into a run.
        x_fine = np.linspace(-1, 1, 2 * case.samples_per_cell * case.n + 1)
        y_fine = core.target(x_fine, case.target, np)
        a_fine = diagnostics.features(x_fine, g.centers, cp["gamma"]) / np.sqrt(len(x_fine))
        u_fine, s_fine, vh_fine = svd(a_fine * g.d, full_matrices=False)
        sampling = {"samples_per_cell": 2 * case.samples_per_cell,
                    "live_mse": float(np.linalg.norm(a_fine @ cp["c"] - y_fine / np.sqrt(len(x_fine)))**2),
                    "refits": []}
        val_x = diagnostics.midpoint_grid(case.validation_points)
        val_y = core.target(val_x, case.target, np)
        for tau in (1e-10, 1e-12, 1e-14):
            keep = s_fine > tau * s_fine[0]
            cstar = g.d * (vh_fine[keep].T @ ((u_fine[:, keep].T @ (y_fine / np.sqrt(len(x_fine)))) / s_fine[keep]))
            v = diagnostics.prediction(val_x, g.centers, cstar, cp["gamma"]) - val_y
            sampling["refits"].append({"cutoff": tau, "rank": int(keep.sum()),
                                       "validation_mse": float(np.mean(v**2)), "coefficient_l1": float(np.abs(cstar).sum())})
    record = {"folder": str(folder), "label": meta["label"], "case": case.__dict__, "end": end,
              "source_step": meta["source_step"], "windows": windows, "endpoint": physical,
              "cutoffs": cutoffs, "sampling_refinement": sampling, "dense": detail,
              "complete_minimum": end-meta["source_step"] >= 20000}
    run.write_json(output / "evidence.json", record)
    endpoint_figure(output, ancestry, record)
    return record


def endpoint_figure(output, ancestry, record):
    with np.load(output / "endpoint.npz") as endpoint, np.load(output / "dense_mechanism.npz") as dense:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        singular = dense["singular_values"]
        axes[0, 0].loglog(singular, np.mean(dense["singular_residual_coefficients"]**2, axis=0), ".")
        axes[0, 0].set(xlabel="Singular value of fixed window-start A D", ylabel="Mean residual MSE in mode")
        bins = dense["band_bounds"][:, 0]
        for block, field in [("readout", "band_signed_readout_descent"), ("geometry", "band_signed_geometry_descent")]:
            axes[0, 1].plot(bins, np.mean(dense[field], axis=0), ".-", label=block)
        axes[0, 1].set(xlabel="Lowest DFT index in band", ylabel="Mean signed linearized half-MSE reduction")
        axes[0, 1].set_xscale("symlog", linthresh=1)
        axes[0, 1].set_yscale("symlog", linthresh=1e-20)
        axes[0, 1].legend()
        positions = np.arange(4)
        axes[1, 0].bar(positions-.18, np.mean(dense["linear_mse_change"], axis=0), .36, label="linear MSE change")
        axes[1, 0].bar(positions+.18, np.mean(dense["quadratic_mse_cost"], axis=0), .36, label="quadratic cost")
        axes[1, 0].set_xticks(positions, ["readout", "geometry", "interaction", "joint"])
        axes[1, 0].set_yscale("symlog", linthresh=1e-20)
        axes[1, 0].set_ylabel("Mean finite-step MSE contribution")
        axes[1, 0].legend(fontsize=8)
        centers = core.geometry(record["case"]["n"]).centers
        axes[1, 1].plot(centers, ancestry["c"][-1, 1:], ".", ms=2, label="trained physical w")
        axes[1, 1].plot(centers, endpoint["readout_refit"][1:], ".", ms=2, alpha=.5, label="detached LS, same geometry")
        axes[1, 1].set(xlabel="Fixed physical center", ylabel="Signed physical readout")
        axes[1, 1].set_yscale("symlog", linthresh=.01)
        axes[1, 1].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle(f"{record['label']}: {record['case']['target']}, N={record['case']['n']}, seed {record['case']['seed']}, update {record['end']:,}")
        fig.savefig(output / "mechanism.png", dpi=140)
        plt.close(fig)


def plot_comparisons(records, output):
    for n in (512, 1024):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
        for column, target in enumerate(("sine", "quadratic", "mixed")):
            for seed in (0, 1):
                ax = axes[seed, column]
                for row in records:
                    case = row["case"]
                    if (case["n"], case["target"], case["seed"]) != (n, target, seed) or row["label"].startswith("acquire"):
                        continue
                    ax.semilogy([w["end"] for w in row["windows"]], [w["mse_mean"] for w in row["windows"]],
                                label=row["label"].replace("_", " "))
                ax.set(title=f"{target}, seed {seed}", xlabel="Total updates", ylabel="20k-window training MSE")
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=7)
        fig.suptitle(f"Fixed theory coordinates, Adam: N={n}")
        fig.savefig(output / f"optimization_N{n}.png", dpi=140)
        plt.close(fig)


def analyze_dictionary(task):
    folder, output = map(Path, task)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((folder / "dictionary.json").read_text())
    case = run.Case(**meta["case"])
    g = core.geometry(case.n)
    gamma = np.load(folder / "reference.npz")["gamma"]
    x = np.linspace(-1, 1, case.n * case.samples_per_cell + 1)
    y = core.target(x, case.target, np) / np.sqrt(len(x))
    a = diagnostics.features(x, g.centers, gamma) / np.sqrt(len(x))
    u, singular, vh = svd(a * g.d, full_matrices=False)
    spectra = {"prescribed": singular}
    spectra["differences"] = svd(readout_solvers.dictionary(x, g, gamma, "differences") / np.sqrt(len(x)),
                                  compute_uv=False)
    end = min(json.loads((folder / spec[0] / "latest.json").read_text())["step"] for spec in meta["specifications"])
    end = end // 20000 * 20000
    refits = []
    for tau in (1e-10, 1e-12, 1e-14):
        keep = singular > tau * singular[0]
        cstar = g.d * (vh[keep].T @ ((u[:, keep].T @ y) / singular[keep]))
        val_x = diagnostics.midpoint_grid(case.validation_points)
        val_r = diagnostics.prediction(val_x, g.centers, cstar, gamma) - core.target(val_x, case.target, np)
        refits.append({"tau": tau, "rank": int(keep.sum()), "validation_mse": float(np.mean(val_r**2)),
                       "coefficient_l1": float(np.abs(cstar).sum())})
    records = []
    for label, coord, _, mode in meta["specifications"]:
        path = folder / label
        latest = json.loads((path / "latest.json").read_text())
        if end < 20000:
            records.append({"label": label, "status": "unfinished_below_minimum", "latest": latest})
            continue
        cp = np.load(path / f"checkpoint_{end:09d}.npz")
        r = a @ cp["c"] - y
        coeff = u.T @ r
        modal = {"coefficients": coeff, "singular_values": singular}
        dense = read_dense(path, end)
        # The same prescribed-coordinate SVD basis is used for both solvers.
        indices = np.arange(0, 2048, 32)
        residuals = a @ dense["c"][indices].T - y[:, None]
        changes = a @ dense["delta_c"][indices].T
        modal.update(dense_step=dense["step"][indices], dense_residual_coefficients=(u.T @ residuals).T,
                     dense_update_coefficients=(u.T @ changes).T)
        run.save_arrays(output / f"{label}_modes.npz", **modal)
        metrics = json.loads((path / f"metrics_{end:09d}.json").read_text())
        metrics.update(label=label, coordinates=coord, initialization=mode,
                       residual_mse_outside_reference_span={str(tau): float(np.linalg.norm(r -
                           u[:, singular > tau * singular[0]] @ coeff[singular > tau * singular[0]])**2)
                           for tau in (1e-10, 1e-12, 1e-14)})
        windows = []
        for right in range(20000, end + 1, 20000):
            losses = run.trace_window_losses(path, right-20000, right)
            if losses is None:
                raise ValueError(f"Incomplete solver trace: {path}")
            windows.append({"end": right, "mse_mean": float(2 * np.mean(losses)),
                            "mse_quantiles": np.quantile(2*losses, [0, .1, .5, .9, 1]).tolist()})
        metrics["windows"] = windows
        records.append(metrics)
    run.save_arrays(output / "conditioning.npz", **spectra, gamma=gamma, centers=g.centers)
    result = {"dictionary": folder.name, "case": meta["case"], "end": end, "reference_cutoffs": refits,
              "modal_basis": "same SVD of prescribed A D for all optimizers and coordinates", "solvers": records}
    run.write_json(output / "evidence.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for row in records:
        if "windows" in row:
            axes[0].semilogy([w["end"] for w in row["windows"]], [w["mse_mean"] for w in row["windows"]],
                             label=row["label"].replace("_", " "))
    for coord, spectrum in spectra.items():
        axes[1].semilogy(spectrum, label=coord)
    axes[0].set(xlabel="Readout updates", ylabel="20k-window training MSE")
    axes[1].set(xlabel="Singular index", ylabel="Singular value of scaled dictionary")
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=6)
    fig.suptitle(folder.name)
    fig.savefig(output / "solvers.png", dpi=140)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--solvers-only", action="store_true")
    args = parser.parse_args()
    root = args.root / "runs" / "ratios"
    folders = sorted(p.parent for p in root.glob("*/*/schedule.json") if (p.parent / "latest.json").exists())
    latest = {f: json.loads((f / "latest.json").read_text())["step"] for f in folders}
    primary = [f for f in folders if f.parent.name.startswith("high_") or f.parent.name == "low_shared"]
    common = min(latest[f] for f in primary) // 20000 * 20000 if len(primary) == 60 else None
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for folder in folders:
        end = latest[folder] // 20000 * 20000
        meta = json.loads((folder / "schedule.json").read_text())
        if folder in primary and common:
            end = common
        if args.horizon:
            end = min(end, args.horizon)
        if end - meta["source_step"] < 20000:
            continue
        tasks.append((str(folder), end, str(args.output / folder.parent.name / folder.name)))
    run.write_json(args.output / "provenance.json", {"common_primary_horizon": common, "primary_cases_present": len(primary),
        "tasks": tasks, "test_evaluation": False,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})
    if not args.solvers_only:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(analyze_case, tasks))
        run.write_json(args.output / "evidence.json", records)
        plot_comparisons(records, args.output)
    else:
        records = []
    dictionaries = sorted(p.parent for p in (root / "solvers").glob("*/dictionary.json"))
    solver_tasks = [(str(p), str(args.output / "solvers" / p.name)) for p in dictionaries
                    if all((p / spec[0] / "latest.json").exists()
                           for spec in json.loads((p / "dictionary.json").read_text())["specifications"])]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        solver_records = list(pool.map(analyze_dictionary, solver_tasks))
    run.write_json(args.output / "solver_evidence.json", solver_records)
    print(json.dumps({"analyzed": len(records), "common_primary_horizon": common}), flush=True)


if __name__ == "__main__":
    main()
