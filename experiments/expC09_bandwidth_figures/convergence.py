"""Eight-target FP64 convergence comparison; total hidden widths start at 64.

Run ``python experiments/expC09_bandwidth_figures/convergence.py``.
Use --plot-only to redraw, or --validate to also check denser grids and halos.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import geometry
from experiments.expC09_bandwidth_figures.targets import target_values as earlier_target

OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/eight_target_convergence"
LABELS = {
    "sine4": r"$\sin(4\pi x)$",
    "sine24": r"$\sin(24\pi x)$",
    "weak_high_frequency": "Weak HF mixture",
    "chirp": "Chirp",
    "runge25": "Runge (25)",
    "runge100": "Runge (100)",
    "gaussian_envelope": "Gaussian envelope",
    "smooth_bump": "Smooth compact bump",
}


def widths(config):
    values = set()
    start = config["width_min"]
    while start < config["width_max"]:
        for k in range(config["points_per_octave"]):
            w = start + k*start//config["points_per_octave"]
            if w <= config["width_max"]:
                values.add(w)
        start *= 2
    values.add(config["width_max"])
    return sorted(values)


def values(x, config):
    x = np.asarray(x, dtype=np.float64)
    functions = {
        "sine4": lambda: np.sin(4*np.pi*x),
        "sine24": lambda: np.sin(24*np.pi*x),
        "weak_high_frequency": lambda: np.sin(2*np.pi*x) + 1e-3*np.sin(40*np.pi*x),
        "chirp": lambda: np.sin(8*np.pi*(x+1)**2),
        "runge25": lambda: 1/(1+25*x*x),
        "runge100": lambda: 1/(1+100*x*x),
        "gaussian_envelope": lambda: earlier_target(x, "gaussian_envelope"),
    }
    def bump():
        u = x/config["bump_half_width"]
        result = np.zeros_like(u)
        inside = np.abs(u) < 1
        result[inside] = np.exp(-1/(1-u[inside]**2))
        return result
    functions["smooth_bump"] = bump
    return np.column_stack([functions[name]() for name in config["targets"]])


def errors(centers, gamma, coefficients, x, config):
    residual = np.empty((len(x), len(config["targets"])))
    truth = values(x, config)
    for start in range(0, len(x), config["eval_chunk_size"]):
        stop = min(start+config["eval_chunk_size"], len(x))
        features = np.tanh(gamma*(x[start:stop, None]-centers))
        for j in range(len(config["targets"])):
            residual[start:stop, j] = (features @ coefficients[:-1, j] + coefficients[-1, j]
                                       - truth[start:stop, j])
    return (np.linalg.norm(residual, axis=0)/np.linalg.norm(truth, axis=0),
            np.max(np.abs(residual), axis=0))


def measure(width, config, *, halo=None, train_points=None, eval_points=None):
    hcount = config["halo_per_side"] if halo is None else halo
    n, spacing, centers = geometry(width, hcount)
    gamma = config["lambda"]/spacing
    nt = train_points or max(config["minimum_train_points"], config["train_points_per_width"]*width+1)
    ne = eval_points or max(config["minimum_eval_points"], config["eval_points_per_width"]*width+1)
    x = np.linspace(-1, 1, nt)
    a = np.column_stack((np.tanh(gamma*(x[:, None]-centers)), np.ones(nt)))
    y = values(x, config)
    # Preserve the original experiment's single-target solve/evaluation path.
    # Under-resolved fits can have huge coefficients, making batched and vector
    # arithmetic differ visibly through cancellation even with the same cutoff.
    solutions = []
    for j in range(len(config["targets"])):
        coef, _, rank, singular = lstsq(a, y[:, j], cond=2.0**(1-config["precision_bits"]),
                                        lapack_driver="gelsd")
        solutions.append(coef)
    coefficients = np.column_stack(solutions)
    del a
    relative, linf = errors(centers, gamma, coefficients, np.linspace(-1, 1, ne), config)
    row = {"width": width, "N": n, "halo_per_side": hcount, "lambda": config["lambda"],
           "gamma": gamma, "train_points": nt, "eval_points": ne, "rank": int(rank),
           "sigma_max": float(singular[0]), "targets": config["targets"],
           "relative_l2": relative.tolist(), "linf": linf.tolist(),
           "coefficient_norm": np.linalg.norm(coefficients, axis=0).tolist()}
    return row, coefficients


def draw(rows, config):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10.8, 7.5))
    fig.subplots_adjust(left=.105, right=.975, bottom=.14, top=.87)
    w = np.array([r["width"] for r in rows])
    e = np.array([r["relative_l2"] for r in rows])
    # Distinct colors plus marker shapes retain identifiability at shared floors.
    colors = ["#0072B2", "#E69F00", "#8B5FBF", "#D55E00", "#009E73", "#CC79A7", "#6B4C3B", "#444444"]
    markers = ["o", "s", "^", "D", "o", "s", "^", "D"]
    for j, (name, color, marker) in enumerate(zip(config["targets"], colors, markers)):
        ax.plot(w, e[:, j], color=color, marker=marker, ms=3.5, lw=1.65,
                label=LABELS[name], markeredgewidth=.45)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(config["width_min"], config["width_max"])
    ax.set_ylim(1e-16, 2)
    ax.xaxis.set_major_locator(FixedLocator([64, 128, 256, 512, 1024, 2048]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([10.0**k for k in range(-16, 1, 2)])
    ax.grid(True, which="major", ls="--", lw=.7, alpha=.3)
    ax.set_xlabel(r"Total network width $W$", fontsize=14)
    ax.set_ylabel(r"Relative $L^2$ error", fontsize=14)
    ax.legend(loc="upper right", ncol=1, frameon=True, facecolor="white",
              framealpha=.95, fontsize=10, handlelength=2.0)
    fig.suptitle("Fixed-grid tanh MLP convergence", fontsize=20, fontweight="bold", y=.985)
    fig.text(.54, .913, rf"$\lambda={config['lambda']}$, FP64  ·  {config['halo_per_side']} halo nodes per side", ha="center", fontsize=12)
    fig.text(.54, .026, r"Measured least-squares readouts on $[-1,1]$. Width includes halo neurons; output bias excluded.",
             ha="center", fontsize=10, color=".3")
    dest = OUT / "figures"
    dest.mkdir(exist_ok=True, parents=True)
    fig.savefig(dest / "eight_target_convergence.png", dpi=220)
    plt.close(fig)


def validate(rows, config):
    lookup = {r["width"]: r for r in rows}
    checks = []
    for w in config["validation_widths"]:
        row = lookup[w]
        coef = np.load(OUT / f"data/coefficients_W{w}.npz")["coefficients"]
        _, _, centers = geometry(w, config["halo_per_side"])
        # Midpoints of a doubled grid give a shifted, denser evaluation.
        count = 2*row["eval_points"]
        x = -1 + (np.arange(count)+.5)*2/count
        dense, _ = errors(centers, row["gamma"], coef, x, config)
        check = {"width": w, "base_error": row["relative_l2"], "dense_shifted_error": dense.tolist(),
                 "dense_shifted_ratio": (dense/np.array(row["relative_l2"])).tolist()}
        if w in config["refit_validation_widths"]:
            refit, _ = measure(w, config, train_points=2*row["train_points"]-1,
                               eval_points=2*row["eval_points"]-1)
            check["double_train_eval_error"] = refit["relative_l2"]
        # Extend the halo at fixed interior resolution. The control's W increases.
        larger_halo = 2*config["halo_per_side"]
        bigger, _ = measure(row["N"]+2*larger_halo+1, config, halo=larger_halo,
                            train_points=row["train_points"], eval_points=row["eval_points"])
        check["double_halo_error"] = bigger["relative_l2"]
        checks.append(check)
        print(f"Validated W={w}", flush=True)
    result = {"targets": config["targets"], "checks": checks,
              "max_shifted_eval_relative_change": max(abs(v-1) for c in checks for v in c["dense_shifted_ratio"])}
    (OUT / "data/validation.json").write_text(json.dumps(result, indent=2)+"\n")
    # Display the control trajectories as evidence alongside the main figure.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for j, (ax, name) in enumerate(zip(axes.flat, config["targets"])):
        x = [r["width"] for r in checks]
        for field, label, style in [("base_error", "Main", "o-"),
                                    ("dense_shifted_error", "Denser evaluation", "--"),
                                    ("double_halo_error", "Double halo", ":")]:
            ax.loglog(x, [c[field][j] for c in checks], style, label=label, lw=1.4, ms=3)
        chosen = [c for c in checks if "double_train_eval_error" in c]
        ax.loglog([c["width"] for c in chosen], [c["double_train_eval_error"][j] for c in chosen],
                  "s-", label="Denser fit + evaluation", ms=3, lw=1)
        ax.set_title(LABELS[name], fontsize=10)
        ax.set_ylim(1e-16, 2)
        ax.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel(r"Original width $W$")
    for ax in axes[:, 0]: ax.set_ylabel(r"Relative $L^2$ error")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(OUT / "figures/convergence_validation.png", dpi=170)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((HERE / "convergence_config.yaml").read_text())
    if config["precision_bits"] != 53:
        raise ValueError("This comparison uses native FP64 only")
    numerical_source = "\n".join(inspect.getsource(f) for f in (values, measure, errors, geometry, earlier_target))
    manifest = {"config": config, "widths": widths(config),
                "numerics_sha256": hashlib.sha256(numerical_source.encode()).hexdigest(),
                "method": "FP64 tanh features, independent single-target gelsd solves and vector evaluations, cutoff 2**-52"}
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = data / "config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing measurements use another configuration; preserve them before rerunning")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {r["width"] for r in rows}
    pending = [w for w in widths(config) if w not in done]
    if pending and args.plot_only:
        raise ValueError("Missing width measurements")
    start = time.monotonic()
    with threadpool_limits(limits=1), path.open("a") as output:
        for w in pending:
            row, coef = measure(w, config)
            np.savez_compressed(data / f"coefficients_W{w}.npz", coefficients=coef)
            output.write(json.dumps(row)+"\n"); output.flush()
            rows.append(row)
            print(f"W={w}: eight targets measured ({time.monotonic()-start:.1f}s total)", flush=True)
        rows.sort(key=lambda r: r["width"])
        assert len(rows) == len(done | set(pending)) == len(widths(config))
        assert all(r["width"] == r["N"]+2*r["halo_per_side"]+1 for r in rows)
        assert np.all(np.isfinite([r["relative_l2"] for r in rows]))
        draw(rows, config)
        if args.validate:
            result = validate(rows, config)
            print("Largest shifted-grid relative change:", result["max_shifted_eval_relative_change"])
    print(OUT / "figures/eight_target_convergence.png")


if __name__ == "__main__":
    main()
