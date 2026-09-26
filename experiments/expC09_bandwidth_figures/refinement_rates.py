"""Dense refinement measurements and inspectable local decay-rate fits."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures import convergence, analytic_packet, smooth_absolute

BASE = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"
OUT = BASE / "refinement_rates"
NAMES = ["sine4", "sine24", "weak_high_frequency", "chirp", "runge25", "runge100", "smooth_absolute", "analytic_packet"]
LABELS = [r"$\sin(4\pi x)$", r"$\sin(24\pi x)$", "Weak high-frequency mix", "Chirp",
          r"Runge: $1/(1+25x^2)$", r"Runge: $1/(1+100x^2)$", "Smooth absolute value", "Oscillatory Gaussian"]
COLORS = ["#0072B2", "#E69F00", "#8B5FBF", "#D55E00", "#009E73", "#CC79A7", "#6B4C3B", "#444444"]


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def collect():
    config = json.loads((convergence.OUT / "data/config.json").read_text())["config"]
    config = dict(config, targets=NAMES[:6])
    original = {r["width"]: r for r in read_rows(convergence.OUT / "data/measurements.jsonl")}
    packet = {r["width"]: r for r in read_rows(analytic_packet.OUT / "data/measurements.jsonl")}
    smooth = {r["width"]: r for r in read_rows(smooth_absolute.OUT / "data/measurements.jsonl")}
    desired = sorted(set(range(50, 385, 2)) | {w for w in original if w <= 1024})
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "targets": NAMES, "widths": desired,
                "dense_interval": [50, 384], "dense_step": 2,
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [convergence.OUT / "data/measurements.jsonl",
                                      analytic_packet.OUT / "data/measurements.jsonl",
                                      smooth_absolute.OUT / "data/measurements.jsonl"]}}
    mp = data / "config.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    mp.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data / "measurements.jsonl"
    rows = read_rows(path) if path.exists() else []
    done = {r["width"] for r in rows}
    with threadpool_limits(limits=1), path.open("a") as output:
        for i, w in enumerate(desired):
            if w in done:
                continue
            if w in original:
                errors = original[w]["relative_l2"][:6]+[smooth[w]["relative_l2"], packet[w]["relative_l2"]]
                source = "original saved measurements"
            else:
                r, _ = convergence.measure(w, config)
                a, _ = smooth_absolute.measure(w, config)
                b, _ = analytic_packet.measure(w, config)
                errors = r["relative_l2"]+[a["relative_l2"], b["relative_l2"]]
                source = "new measurements; same independent solver and chunked evaluation"
            row = {"width": w, "targets": NAMES, "relative_l2": errors, "source": source}
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            if i % 10 == 0:
                print(f"Measured/reused {i+1}/{len(desired)} widths", flush=True)
    return sorted(rows, key=lambda r: r["width"])


def line_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residual = y-(intercept+slope*x)
    sst = np.sum((y-y.mean())**2)
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1-np.sum(residual**2)/sst) if sst else 0.,
            "rmse_decades": float(np.sqrt(np.mean(residual**2))),
            "max_residual_decades": float(np.max(np.abs(residual)))}


def select_fit(widths, errors, mode):
    """Select a contiguous, steep, well-supported local line above the floor.

    Require >=6 points and >=2.5 fitted decades; no points are dropped inside
    a window. Among qualified windows, retain slopes within 15% of the
    steepest qualified slope, then maximize fitted error drop. This favors
    a substantial steep segment instead of the highest-R2 tiny window.
    """
    w = np.asarray(widths, dtype=float)
    y = np.log10(errors)
    x = np.log10(w) if mode == "loglog" else w
    floor = float(np.median(y[w >= 512]))
    safe = y >= floor+1.
    candidates = []
    for i in range(len(w)):
        for j in range(i+6, len(w)+1):
            if not np.all(safe[i:j]):
                break
            fit = line_fit(x[i:j], y[i:j])
            drop = -fit["slope"]*(x[j-1]-x[i])
            if fit["slope"] < 0 and drop >= 2.5 and fit["r2"] >= .995 and fit["rmse_decades"] <= .15 and fit["max_residual_decades"] <= .35:
                candidates.append(dict(fit, start=i, stop=j, fitted_drop_decades=float(drop)))
    if not candidates:
        return {"status": "no_qualified_segment", "floor_log10": floor}
    steepest = max(-c["slope"] for c in candidates)
    near_steepest = [c for c in candidates if -c["slope"] >= .85*steepest]
    best = max(near_steepest, key=lambda c: (c["fitted_drop_decades"], c["stop"]-c["start"]))
    i, j = best["start"], best["stop"]
    sensitivity = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            lo, hi = i+di, j+dj
            if lo >= 0 and hi <= len(w) and hi-lo >= 5 and np.all(safe[lo:hi]):
                sensitivity.append(line_fit(x[lo:hi], y[lo:hi])["slope"])
    other_x = w if mode == "loglog" else np.log10(w)
    other = line_fit(other_x[i:j], y[i:j])
    return dict(best, status="fitted", mode=mode, width_start=int(w[i]), width_stop=int(w[j-1]),
                points=j-i, floor_log10=floor, upper_cutoff_unobserved=bool(i == 0),
                slope_boundary_range=[min(sensitivity), max(sensitivity)],
                alternative_coordinate_fit_same_points=other,
                selected_widths=w[i:j].astype(int).tolist())


def fit_interval(widths, errors, interval):
    """OLS of log10(error) against width, including every point in the interval."""
    w = np.asarray(widths)
    y = np.log10(errors)
    lo, hi = interval
    assert lo in w and hi in w and lo < hi
    selected = (w >= lo) & (w <= hi)
    fit = line_fit(w[selected], y[selected])
    return dict(fit, status="fitted", mode="semilog", selection="user-specified inclusive interval",
                width_start=int(lo), width_stop=int(hi), points=int(selected.sum()),
                selected_widths=w[selected].astype(int).tolist(),
                floor_log10=float(np.median(y[w >= 512])),
                upper_cutoff_unobserved=False)


def draw(rows, all_fits, mode, *, manual=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator

    w = np.array([r["width"] for r in rows])
    errors = np.array([r["relative_l2"] for r in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 4, figsize=(17.6, 9.1), sharey=True)
    fig.subplots_adjust(left=.055, right=.985, bottom=.10, top=.84, wspace=.20, hspace=.38)
    for k, (ax, label, color) in enumerate(zip(axes.flat, LABELS, COLORS)):
        fit = all_fits[mode][k]
        y = errors[:, k]
        ax.plot(w, y, "o-", color=color, lw=1.15, ms=2.2, alpha=.85, zorder=2)
        floor_log = fit["floor_log10"]
        # Show the transition and enough of the floor to inspect the cutoff.
        below = np.log10(y) <= floor_log+1.
        floor_start = next((i for i in range(len(w)-5) if np.all(below[i:i+6]) and np.mean(below[i:]) >= .9), len(w)-1)
        left = int(w.min())
        right = min(1024, int(np.ceil((w[floor_start]+.25*(w[floor_start]-left))/8)*8))
        right = max(right, 96)
        ax.set_xlim(left, right)
        if mode == "loglog":
            ax.set_xscale("log")
            ticks = sorted(set([left, right]+[int(round(t/8)*8) for t in np.geomspace(left, right, 5)[1:-1]]))
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 10)
        ax.set_yticks([10.**v for v in range(-16, 1, 4)])
        ax.set_title(label, fontsize=11, pad=10)
        ax.grid(alpha=.18, lw=.65)
        ax.axhline(10**(floor_log+1.), color=".55", lw=.7, ls=":", zorder=0)
        if fit["status"] == "fitted":
            lo, hi = fit["width_start"], fit["width_stop"]
            ax.axvspan(lo, hi, color="#9bbddd", alpha=.32, zorder=0)
            xx = np.linspace(lo, hi, 200)
            yy = 10**(fit["intercept"]+fit["slope"]*(np.log10(xx) if mode == "loglog" else xx))
            ax.plot(xx, yy, color="#b22222", ls="--", lw=2, zorder=4)
            unit = "" if mode == "loglog" else " / neuron"
            info = f"slope = {fit['slope']:.3g}{unit}\n$R^2$ = {fit['r2']:.4f}\n$W$ = {lo}–{hi}; {fit['points']} points"
            if fit["upper_cutoff_unobserved"]:
                info += f"\nUpper cutoff not observed ($W\\geq{left}$)"
            ax.text(.97, .96, info, transform=ax.transAxes, va="top", ha="right", fontsize=8.7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        else:
            ax.text(.97, .95, "No sufficiently supported linear segment", transform=ax.transAxes,
                    va="top", ha="right", fontsize=8)
        ax.set_xlabel(r"Network width $W$"+(" (log scale)" if mode == "loglog" else ""))
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Relative $L^2$ error")
    equation = r"$\log_{10} E = a + m\log_{10} W$" if mode == "loglog" else r"$\log_{10} E = a + mW$"
    fig.suptitle("Local decay rates: "+equation, y=.982, fontsize=18, fontweight="bold")
    handles = [Line2D([], [], color=".35", marker="o", ms=3, label="Measured error"),
               Line2D([], [], color="#b22222", ls="--", lw=2, label="Linear fit in stated coordinates"),
               Patch(facecolor="#9bbddd", alpha=.4, label="Widths used in fit"),
               Line2D([], [], color=".55", ls=":", label="10 × estimated numerical floor")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=10)
    footer = ("Fixed λ = 0.25, FP64. User-selected intervals; every measured point inside each shaded band is included."
              if manual else "Local fits at fixed λ = 0.25, FP64. Each coordinate system selects its own interval; panels zoom to show the decay and floor.")
    fig.text(.5, .025, footer,
             ha="center", fontsize=10, color=".3")
    dest = OUT / f"figures/refinement_rates_{mode}{'_manual' if manual else ''}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=210)
    plt.close(fig)
    return dest


def manual_run(interval_path):
    requested = json.loads(interval_path.read_text())
    assert set(requested) == set(NAMES)
    base_path = OUT / "data/measurements.jsonl"
    rows = sorted(read_rows(base_path), key=lambda r: r["width"])
    w = np.array([r["width"] for r in rows])
    e = np.array([r["relative_l2"] for r in rows])
    # np.argmin selects the first match: sorted widths break ties downward.
    intervals = {name: [int(w[np.argmin(np.abs(w-bound))]) for bound in bounds]
                 for name, bounds in requested.items()}
    fits = [fit_interval(w, e[:, k], intervals[name]) for k, name in enumerate(NAMES)]
    metadata = {"requested_intervals": requested, "intervals": intervals,
                "snapping": "nearest existing measured width; ties use lower width",
                "coordinate_system": "log10(error) versus width",
                "selection": "all points within inclusive user-specified bounds; no automatic exclusions",
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [base_path]},
                "fits": dict(zip(NAMES, fits))}
    (OUT / "data/manual_fits.json").write_text(json.dumps(metadata, indent=2)+"\n")
    fields = ["target", "width_start", "width_stop", "points", "slope", "r2", "rmse_decades", "max_residual_decades"]
    with (OUT / "data/manual_rates.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for name, f in zip(NAMES, fits):
            row = dict(target=name, **{k: f[k] for k in fields[1:]})
            writer.writerow(row)
            print(row)
    print(draw(rows, {"semilog": fits}, "semilog", manual=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--manual-intervals", type=Path)
    args = parser.parse_args()
    if args.manual_intervals:
        manual_run(args.manual_intervals)
        return
    rows = sorted(read_rows(OUT / "data/measurements.jsonl"), key=lambda r: r["width"]) if args.plot_only else collect()
    w = np.array([r["width"] for r in rows])
    e = np.array([r["relative_l2"] for r in rows])
    fits = {mode: [select_fit(w, e[:, k], mode) for k in range(8)] for mode in ["loglog", "semilog"]}
    (OUT / "data/fits.json").write_text(json.dumps(fits, indent=2)+"\n")
    for mode in fits:
        print(mode)
        for name, fit in zip(NAMES, fits[mode]):
            print(name, {k: fit[k] for k in ["status", "slope", "width_start", "width_stop", "r2", "points"] if k in fit})
        print(draw(rows, fits, mode))


if __name__ == "__main__":
    main()
