"""Offline diagnostics for signal recovery versus population scale acquisition.

The functions use the empirical half-MSE and physical (a,b,c,d) coordinates.
Attribution is an instantaneous gradient-flow diagnostic at a GD state. It is
not an exact finite-step decomposition of logarithmic signal change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PRIMARY = ("sine", "moment3", "moment9")
PACKAGES = ("core20k", "continue100k", "continue600k")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(clean(rows))


def concentration(nonnegative):
    """Top-ceil(10% W) share and participation fraction; undefined at zero."""
    v = np.asarray(nonnegative, dtype=float)
    if np.any(v < 0) or not np.all(np.isfinite(v)):
        raise ValueError("Concentration requires finite nonnegative movements")
    total = float(v.sum())
    if total == 0:
        return dict(top10_share=None, participation_fraction=None)
    top = max(1, int(np.ceil(.1 * v.size)))
    return dict(top10_share=float(np.sort(v)[-top:].sum() / total),
                participation_fraction=float(total**2 / (v.size * (v @ v))))


def acquisition_distance(a, threshold, population):
    """Exact Euclidean distance to at least ceil(population W) acquired slopes."""
    if threshold <= 0 or not 0 < population <= 1:
        raise ValueError("Positive threshold and population in (0,1] required")
    gaps = np.sort(np.maximum(threshold - np.abs(a), 0.))
    return float(np.linalg.norm(gaps[:int(np.ceil(population * len(a)))]))


def population_upper(reference_a, error_radius, threshold, margin):
    """Conditional count bound; the caller must justify the trajectory radius."""
    if error_radius < 0 or not 0 < margin < threshold:
        raise ValueError("Nonnegative radius and margin in (0,threshold) required")
    return float(min(1., np.mean(np.abs(reference_a) >= threshold - margin)
                     + error_radius**2 / (len(reference_a) * margin**2)))


def state_diagnostics(z, d, x, y, eta=.002, n=128):
    """Exact sample-space gradients and signed block actions, without a Hessian."""
    a, b, c = np.asarray(z)
    x, y = np.asarray(x), np.asarray(y)
    m, width = len(x), len(a)
    u = x[:, None] * a + b
    phi = np.tanh(u)
    exp = np.exp(-2 * np.abs(u))
    derivative = 4 * exp / (1 + exp)**2
    second = -2 * phi * derivative
    r = phi @ c + d - y
    r2 = float(r @ r / m)
    tangent_pair = x @ (r[:, None] * derivative) / m
    ga = c * tangent_pair
    gb = c * (r @ derivative) / m
    gc, gd = phi.T @ r / m, float(np.mean(r))
    rc = np.mean(r) + x * (x @ r) / (x @ x)
    ga_coarse = c * (x @ (rc[:, None] * derivative)) / m
    ga_tail = ga - ga_coarse
    fc = phi @ gc
    fq = np.sum(c * derivative * (x[:, None] * ga + gb), axis=1)
    hac_filter = c * (x @ (fc[:, None] * derivative)) / m
    hac_amplitude = gc * tangent_pair
    had = c * gd * (x @ derivative) / m
    haq = c * (x @ (fq[:, None] * derivative)) / m
    haq += c * (x @ (r[:, None] * second * (x[:, None] * ga + gb))) / m
    ga2 = float(ga @ ga)
    resolved = r2 > 0 and ga2 > 1e-28 * max(1., r2)
    actual_delta = np.abs(a - eta * ga) - np.abs(a)
    velocity = -float(np.mean(np.sign(a) * ga))
    scalars = dict(half_mse=r2 / 2, residual_rms=np.sqrt(r2),
                   xi=np.sqrt(ga2 / r2) if r2 > 0 else None,
                   grad_a_norm=np.sqrt(ga2), readout_l2=float(np.linalg.norm(c)),
                   mean_gamma=float(np.mean(np.abs(a))), median_gamma=float(np.median(np.abs(a))),
                   q90_gamma=float(np.quantile(np.abs(a), .9)), max_gamma=float(np.max(np.abs(a))),
                   signed_velocity=velocity,
                   signed_alignment=np.sqrt(width) * velocity / np.sqrt(ga2) if ga2 > 0 else None,
                   coarse_velocity=-float(np.mean(np.sign(a) * ga_coarse)),
                   tail_velocity=-float(np.mean(np.sign(a) * ga_tail)),
                   tail_loss_slope_descent=float(ga_tail @ ga),
                   tail_loss=float(np.mean((r - rc)**2) / 2),
                   coarse_norm=float(np.sqrt(np.mean(rc**2))),
                   positive_step=float(np.maximum(actual_delta, 0).mean()),
                   negative_step=float(np.maximum(-actual_delta, 0).mean()),
                   delta_mean_gamma=float(actual_delta.mean()),
                   crossing_remainder=float(actual_delta.mean() - eta * velocity),
                   growing_fraction=float(np.mean(actual_delta > 0)),
                   fraction_gamma_1=float(np.mean(np.abs(a) >= 1)),
                   fraction_lambda_005=float(np.mean(np.abs(a) >= .05*n/2)),
                   fraction_lambda_025=float(np.mean(np.abs(a) >= .25*n/2)),
                   attribution_resolved=bool(resolved))
    scalars.update({"step_" + k: v for k, v in concentration(np.maximum(actual_delta, 0)).items()})
    for label, h, block2 in (("c_filter", hac_filter, 0.), ("c_amplitude", hac_amplitude, 0.),
                             ("c", hac_filter + hac_amplitude, float(gc @ gc)),
                             ("d", had, gd**2), ("q", haq, float(ga @ ga + gb @ gb))):
        scalars["D_" + label] = float(ga @ h / ga2 - block2 / r2) if resolved else None
    scalars["D_c_normalization"] = -float(gc @ gc) / r2 if r2 > 0 else None
    return scalars, dict(gradient=np.stack((ga, gb, gc)), gradient_d=gd,
                         Hac_filter=hac_filter, Hac_amplitude=hac_amplitude,
                         Had_gd=had, Haq_gq=haq, delta_gamma=actual_delta)


def load_curated(path):
    with np.load(path) as archive:
        data = {k: archive[k] for k in archive.files}
    cfg = json.loads(str(data["configuration"]))
    retained = data.get("plot_trace_columns", np.arange(len(cfg["trace_columns"])))
    columns = {cfg["trace_columns"][int(old)]: i for i, old in enumerate(retained)}
    return data, cfg, json.loads(str(data["cases"])), columns


def curated_analysis(evidence, output):
    """Use archived endpoints and sampled curves; never integrate sparse traces."""
    endpoint_rows, curve_rows, reference_rows, sources = [], [], [], []
    for package in (*PACKAGES, "replicate20k"):
        root = evidence / package
        summary_path = root / "summary.json"
        if not summary_path.exists():
            continue
        sources.append(summary_path)
        summaries = json.loads(summary_path.read_text())
        for row in summaries:
            if row["degree"] == 0:
                endpoint_rows.append(dict(package=package, **row))
        for path in sorted(root.glob("core_*_curves.npz")):
            sources.append(path)
            data, cfg, cases, cols = load_curated(path)
            for i, case in enumerate(cases):
                if case["kappa"] != 1 or case["target"] not in PRIMARY:
                    continue
                for step, tr in zip(data["p0_steps"], data["p0_trace"][i]):
                    v = {name: float(tr[k]) for name, k in cols.items()}
                    R, ga = np.sqrt(2 * v["half_mse"]), v["grad_a_norm"]
                    signed = v["signed_coarse_force"] + v["signed_remainder_force"]
                    curve_rows.append(dict(package=package, target=case["target"], seed=cfg["seed"],
                        width=cfg["width"], step=int(step), geometry_time=step*cfg["eta"],
                        xi=ga/R if R > 0 else None, signed_velocity=signed,
                        signed_alignment=np.sqrt(cfg["width"])*signed/ga if ga > 0 else None,
                        **v))
        reference_path = root / "reference_audit.csv"
        if reference_path.exists():
            sources.append(reference_path)
            with reference_path.open() as stream:
                reference_rows.extend(dict(package=package, **row) for row in csv.DictReader(stream)
                                      if float(row["kappa"]) == 1 and row["target"] in PRIMARY)
    baseline = {(r["n"], r["seed"], r["target"], r["kappa"]): r
                for r in endpoint_rows if r["package"] == "core20k"}
    changes = []
    for row in endpoint_rows:
        key = row["n"], row["seed"], row["target"], row["kappa"]
        if row["package"] not in PACKAGES[1:] or key not in baseline:
            continue
        base = baseline[key]
        changes.append(dict(package=row["package"], n=row["n"], seed=row["seed"], target=row["target"],
            kappa=row["kappa"], xi_ratio_to_20k=row["normalized_slope_gradient"]/base["normalized_slope_gradient"],
            mean_change_after_20k=row["mean_gamma"]-base["mean_gamma"],
            coarse_change_after_20k=row["cumulative_coarse_motion"]-base["cumulative_coarse_motion"],
            tail_change_after_20k=row["cumulative_remainder_motion"]-base["cumulative_remainder_motion"],
            crossing_after_20k=row["cumulative_crossing_remainder"]-base["cumulative_crossing_remainder"],
            heldout_mse=row["heldout_mse"], fraction_lambda_005=row["fraction_lambda_005"],
            fraction_lambda_025=row["fraction_lambda_025"]))
    for name, rows in (("endpoints", endpoint_rows), ("sampled_curves", curve_rows),
                       ("reference_accuracy", reference_rows), ("post20k_changes", changes)):
        write_table(output / f"{name}.csv", rows)
    return sources, curve_rows, changes


def replay_analysis(path, output):
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    cases = json.loads(str(data["cases"]))
    rows, windows, vectors = [], [], {}
    steps = data["steps"]
    # Dense early attribution, logarithmic later states, and exact endpoint states.
    wanted = np.unique(np.r_[np.arange(0, 2001, 20),
        np.rint(np.geomspace(2020, max(2020, int(steps[-1])), 100)),
        0, 20000, 100000, 600000]).astype(int)
    chosen = np.unique([int(np.argmin(abs(steps - s))) for s in wanted if s <= steps[-1]])
    for i, case in enumerate(cases):
        for j in chosen:
            scalar, vector = state_diagnostics(data["z"][i, j], data["d"][i, j],
                                               data["x"], data["y"][i])
            rows.append(dict(**case, step=int(steps[j]), **scalar))
            if steps[j] in (0, 20000, 100000, 600000):
                vectors[f"seed{case['seed']}_{case['target']}_{steps[j]}_gradient"] = vector["gradient"]
        for start, end in ((0, 20000), (20000, 100000), (100000, 600000), (20000, 600000)):
            if start not in steps or end not in steps:
                continue
            j, k = int(np.searchsorted(steps, start)), int(np.searchsorted(steps, end))
            initial, final = data["z"][i, j, 0], data["z"][i, k, 0]
            positive = data["positive"][i, k] - data["positive"][i, j]
            negative = data["negative"][i, k] - data["negative"][i, j]
            net = np.abs(final) - np.abs(initial)
            path_budget = float(data["path"][i, k] - data["path"][i, j])
            item = dict(**case, start=start, end=end, mean_change=float(net.mean()),
                positive_travel=float(positive.mean()), negative_travel=float(negative.mean()),
                fraction_net_growing=float(np.mean(net > 0)),
                accounting_error=float(np.max(abs(positive-negative-net))),
                gradient_path_budget=path_budget, mean_path_budget=path_budget/np.sqrt(len(initial)))
            item.update({"positive_" + name: value for name, value in concentration(positive).items()})
            item.update({"net_positive_" + name: value for name, value in concentration(np.maximum(net, 0)).items()})
            for threshold in (1., 3.2, 16.):
                for population in (.1, .5):
                    distance = acquisition_distance(initial, threshold, population)
                    tag = f"gamma{threshold:g}_population{population:g}"
                    item[f"distance_{tag}"] = distance
                    item[f"path_excludes_{tag}"] = bool(path_budget < distance)
            windows.append(item)
    write_table(output / "state_diagnostics.csv", rows)
    write_table(output / "movement_windows.csv", windows)
    np.savez_compressed(output / "endpoint_gradients.npz", **vectors)
    return rows, windows


def plot_results(output, curves, changes, windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.colormaps["viridis"](np.linspace(.08, .85, 5))
    titles = dict(sine="Sine", moment3="Coarse + degree 3", moment9="Coarse + degree 9")
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
    for col, target in enumerate(PRIMARY):
        for seed in range(5):
            rr = sorted((r for r in curves if r["package"] == "continue600k" and
                         r["target"] == target and r["seed"] == seed), key=lambda r: r["step"])
            x = [r["step"] for r in rr]
            for row, key in enumerate(("xi", "signed_alignment", "readout_l2", "mean_gamma")):
                axes[row, col].plot(x, [r[key] for r in rr], color=colors[seed], lw=1.2,
                                    label=f"Seed {seed}")
                axes[row, col].axvline(20000, color=".5", ls=":", lw=.8)
                axes[row, col].set_xscale("symlog", linthresh=1000)
                axes[row, col].grid(alpha=.2)
            axes[0, col].set_yscale("log")
        axes[0, col].set_title(titles[target])
        axes[1, col].axhline(0, color="black", lw=.7)
        axes[1, col].set_ylim(-1.05, 1.05)
        axes[-1, col].set_xlabel("Equal-rate GD updates")
    for row, label in enumerate((r"Total signal $\Xi$", "Signed alignment", r"Readout norm $\|c\|_2$", r"Mean slope $\overline{\gamma}$")):
        axes[row, 0].set_ylabel(label)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Signal recovery and signed scale learning · W=177 · five paired seeds")
    fig.tight_layout()
    fig.savefig(output / "signal_recovery.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for col, target in enumerate(PRIMARY):
        rr = [r for r in changes if r["target"] == target and r["kappa"] == 1
              and r["package"] == "continue600k"]
        for r in rr:
            axes[col].plot([0, 1, 2], [r["coarse_change_after_20k"], r["tail_change_after_20k"],
                                      r["mean_change_after_20k"]], "o-", color=colors[r["seed"]], alpha=.8)
        axes[col].set_xticks([0, 1, 2], ["Coarse", "Remaining", "Net"])
        axes[col].axhline(0, color="black", lw=.7)
        axes[col].set_title(titles[target])
        axes[col].grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Signed mean-slope change, 20k–600k")
    fig.tight_layout()
    fig.savefig(output / "signed_motion.png", dpi=170)
    plt.close(fig)
    if windows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for col, target in enumerate(PRIMARY):
            rr = [r for r in windows if r["target"] == target and r["start"] == 20000 and r["end"] == 600000]
            for r in rr:
                axes[col].scatter(r["positive_top10_share"], r["positive_participation_fraction"],
                                  color=colors[r["seed"]], label=f"Seed {r['seed']}")
            axes[col].set(xlim=(0, 1), ylim=(0, 1), xlabel="Top 10% share of upward travel", title=titles[target])
            axes[col].grid(alpha=.2)
        axes[0].set_ylabel("Participation fraction of upward travel")
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "movement_concentration.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources, curves, changes = curated_analysis(args.evidence, args.output)
    windows = []
    if args.replay:
        _, windows = replay_analysis(args.replay, args.output)
        sources.append(args.replay)
    plot_results(args.output, curves, changes, windows)
    manifest = dict(role="Retrospective mechanism analysis; no model or checkpoint selection",
        primary_targets=list(PRIMARY), baseline_step=20000,
        missing_neuronwise_data=not bool(windows),
        attribution="Instantaneous flow diagnostic at discrete GD states",
        sparse_trace_policy="No cumulative quantities or continuous maxima inferred from sparse plotting traces",
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
