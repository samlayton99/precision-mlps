"""Measure and draw bandwidth and precision panels for sin(24*pi*x)."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import geometry, measure, selector, SELECTOR

OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/sine24_panels"
CONFIG = {"target": "sin(24*pi*x)", "bandwidth_widths": [96, 128, 192, 256],
          "precision_width": 256, "precisions": [24, 32, 40, 53],
          "halo_per_side": 24, "train_points": 4801, "eval_points": 8001,
          "lambda_min": .05, "lambda_max": 1.5, "lambda_count": 85}


def target(x):
    return np.sin(24*np.pi*x)


def prediction(width, p):
    _, h, _ = geometry(width, CONFIG["halo_per_side"])
    if h*24*np.pi >= np.pi:
        return {"lambda": None, "status": "frequency_outside_resolved_range"}
    return selector.choose_lambda("tanh", spacing=h, e_tol=2.**(1-p), omega_scale=24*np.pi)


def draw(rows, predictions, *, output_path=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 12, "axes.titlesize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))
    fig.subplots_adjust(left=.085, right=.985, bottom=.15, top=.84, wspace=.26)
    colors = [plt.get_cmap("viridis")(v) for v in np.linspace(.08, .86, 4)]
    panels = [[(w, 53, rf"$W={w}$") for w in CONFIG["bandwidth_widths"]],
              [(256, p, rf"$p={p}$"+(" (FP64)" if p == 53 else "")) for p in CONFIG["precisions"]]]
    for ax, curves, title in zip(axes, panels, ["(b) Bandwidth prediction", "(c) Working precision"]):
        for (w, p, label), color in zip(curves, colors):
            cells = sorted([r for r in rows if r["width"] == w and r["p"] == p], key=lambda r: r["lambda"])
            ax.plot([r["lambda"] for r in cells], [r["relative_l2"] for r in cells], color=color, lw=1.5, label=label)
            lam = predictions[f"{w}:{p}"]["lambda"]
            if lam is not None:
                error = next(r["relative_l2"] for r in cells if r["lambda"] == lam)
                ax.scatter([lam], [error], s=34, marker="D", facecolor=color, edgecolor="#c73435", linewidth=1.4, zorder=5)
        ax.set_title(title, loc="center", y=1.12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(.05, 1.5)
        ax.set_ylim(1e-16, 10)
        ax.set_yticks([10.**k for k in range(-16, 1, 4)])
        ax.xaxis.set_major_locator(FixedLocator([.05, .1, .2, .5, 1.]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Relative bandwidth $\lambda$")
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.grid(True, which="major", alpha=.2, lw=.65)
        curves_legend = ax.legend(loc="upper right", ncol=2, framealpha=.95, fontsize=8, columnspacing=.7, handlelength=1.5)
        ax.add_artist(curves_legend)
        ax.legend(handles=[Line2D([], [], marker="D", mfc="white", mec="#c73435", mew=1.4,
                                  ls="none", label=r"Predicted $\lambda$")],
                  loc="lower center", bbox_to_anchor=(.5, 1.015), frameon=False, fontsize=8, borderaxespad=0)
    dest = Path(output_path) if output_path is not None else OUT / "figures/sine24_bandwidth_precision.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=240)
    plt.close(fig)
    print(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    curves = [(w, 53) for w in CONFIG["bandwidth_widths"]]+[(256, p) for p in CONFIG["precisions"] if p != 53]
    predictions = {f"{w}:{p}": prediction(w, p) for w, p in curves}
    jobs = set()
    for w, p in curves:
        grid = list(np.geomspace(.05, 1.5, 85))
        lam = predictions[f"{w}:{p}"]["lambda"]
        if lam is not None:
            grid.append(lam)
        jobs.update((w, p, float(v)) for v in grid)
    manifest = {"config": CONFIG, "predictions": predictions,
                "measure_sha256": hashlib.sha256(inspect.getsource(measure).encode()).hexdigest(),
                "target_sha256": hashlib.sha256(inspect.getsource(target).encode()).hexdigest(),
                "selector_sha256": hashlib.sha256(SELECTOR.read_bytes()).hexdigest(),
                "precision_model": "Rounded features, labels, readouts, products, sums; FP64 SVD internals"}
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = data / "config.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data / "measurements.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {(r["width"], r["p"], r["lambda"]) for r in rows}
    pending = sorted(jobs-done)
    assert not (args.plot_only and pending)
    with threadpool_limits(limits=1), path.open("a") as output:
        for i, (w, p, lam) in enumerate(pending, 1):
            row = measure(w, 24, lam, p, 4801, 8001, "sine24", target_fn=target)
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            if i % 50 == 0 or i == len(pending):
                print(f"Measured {i}/{len(pending)} cells", flush=True)
    assert len(rows) == len(jobs) and all(np.isfinite(r["relative_l2"]) for r in rows)
    draw(rows, predictions)


if __name__ == "__main__":
    main()
