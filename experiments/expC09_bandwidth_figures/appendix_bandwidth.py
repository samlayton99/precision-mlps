"""Appendix bandwidth figures: scalar prescriptions and archived FP64 sweeps.

Run with MPLCONFIGDIR=/tmp/precisionmlps-mpl .venv/bin/python this_file.py.
Reuse archived measurements where available; add erf and the new sech target
under the same FP64 least-squares protocol. No smoothing is applied.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import lstsq
from scipy.special import erf, expit
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/lambda_theorem_compatibility/choosing_optimal_lambda"
OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/appendix_bandwidth"
BUNDLE = ROOT / "docs/appendix_notes/bandwidth_figures"
EPS = 2.0**-52
ACTS = ("tanh", "gelu", "swish")
NAMES = {"tanh": "Tanh", "gelu": "GELU", "swish": "SiLU", "erf": "Erf"}
ERROR_ACTS = (*ACTS, "erf")
COLORS = ("#0072B2", "#E69F00", "#8B5FBF", "#D55E00", "#009E73", "#CC79A7", "#6B4C3B", "#444444")
WIDTH_COLORS = tuple(plt.get_cmap("viridis")(v) for v in np.linspace(.07, .87, 6))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = load_module("bandwidth_rules", SOURCE / "reproduce_figures.py")


def log_khat(act, xi):
    return -np.asarray(xi)**2/4 if act == "erf" else rules.log_khat(act, xi)


def log_refined(act, lam, theta):
    if act != "erf":
        return rules.log_refined_ratio(act, lam, theta)
    if not 0 < theta < np.pi:
        raise ValueError("Representative frequency must lie below the grid Nyquist frequency.")
    # erf has r=1; the Gaussian activation itself would have r=0.
    minus, plus = 2*np.pi-theta, 2*np.pi+theta
    return np.logaddexp(np.log(theta/minus)+log_khat(act, minus/lam),
                        np.log(theta/plus)+log_khat(act, plus/lam))-log_khat(act, theta/lam)


def activation(act, z):
    if act == "tanh": return np.tanh(z)
    if act == "gelu": return .5*z*(1+erf(z/np.sqrt(2)))
    if act == "swish": return z*expit(z)
    if act == "erf": return erf(z)
    raise ValueError(act)


def target_value(name, x):
    if name == "mix_2_6_10": return np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(10*np.pi*x)
    if name == "gauss20": return np.exp(-20*x*x)
    if name == "expsin": return np.exp(np.sin(3*np.pi*x))
    if name == "sech5": return 1/np.cosh(5*x)
    raise ValueError(name)


def measure(act, lam, n, targets):
    """Identical to expC08's p=53 solve, with a shared factorization for targets."""
    h = 2/n
    centers = -1+np.arange(-32, n+33)*h
    x = np.linspace(-1, 1, 2003)
    xe = np.linspace(-1, 1, 4001)
    a = np.column_stack((activation(act, (lam/h)*(x[:, None]-centers)), np.ones(len(x))))
    y = np.column_stack([target_value(t, x) for t in targets])
    c = lstsq(a, y, cond=EPS, lapack_driver="gelsd")[0]
    predictions = activation(act, (lam/h)*(xe[:, None]-centers))@c[:-1]+c[-1]
    ye = np.column_stack([target_value(t, xe) for t in targets])
    errors = np.linalg.norm(predictions-ye, axis=0)/np.linalg.norm(ye, axis=0)
    return [{"act": act, "lam": float(lam), "N": int(n), "fn": t, "p": 53, "rel_l2": float(e)} for t, e in zip(targets, errors)]


def threshold(log_score):
    """Locate an actual root; never silently replace it by a bracket endpoint."""
    grid = np.geomspace(.003, 1.5, 401)
    values = np.asarray(log_score(grid))
    assert np.all(np.isfinite(values)) and np.all(np.diff(values) >= -1e-9)
    assert values[0] < np.log(EPS) < values[-1]
    root = brentq(lambda lam: float(log_score(lam) - np.log(EPS)), .003, 1.5,
                  xtol=1e-14, rtol=1e-14)
    assert abs(float(log_score(root) - np.log(EPS))) < 2e-9
    return root


def periodic_frequency(f, count=4096):
    """Fixed-resolution period-two amplitude centroid; DC remains in denominator."""
    x = -1 + 2*np.arange(count)/count
    magnitude = np.abs(np.fft.fft(f(x))/count)
    omega = 2*np.pi*np.fft.fftfreq(count, d=2/count)
    return float(np.dot(magnitude, np.abs(omega))/magnitude.sum())


def target_specs():
    # Known lines/whole-line transforms where available. The final three inputs
    # are stated heuristics, not certified reductions of the full spectrum.
    smooth = periodic_frequency(lambda x: np.log(np.cosh(10*x))/10)
    return [
        ("sine4", r"$\sin(4\pi x)$", 4*np.pi, "exact single frequency"),
        ("sine24", r"$\sin(24\pi x)$", 24*np.pi, "exact single frequency"),
        ("weak_high", r"$\sin(2\pi x)+10^{-3}\sin(40\pi x)$", (2+.001*40)*np.pi/1.001, "exact amplitude-weighted line centroid"),
        ("chirp", r"$\sin(8\pi(x+1)^2)$", 16*np.pi, "mean absolute phase derivative on [-1,1]; same characteristic-frequency proxy as expC09"),
        ("runge25", r"$1/(1+25x^2)$", 5., "whole-line Fourier-amplitude centroid"),
        ("runge100", r"$1/(1+100x^2)$", 10., "whole-line Fourier-amplitude centroid"),
        ("smooth_absolute", r"$\log(\cosh(10x))/10$", smooth, "4096-point raw period-two DFT amplitude centroid, including DC, no window/filter; endpoints match but derivative has a jump"),
        ("packet", r"$e^{-8(x-0.2)^2}\sin(8\pi x)$", 8*np.pi, "carrier frequency proxy"),
    ]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.labelsize": 21, "axes.titlesize": 24, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 2.6,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
        "xtick.major.size": 5, "ytick.major.size": 5, "xtick.major.width": 1.8,
        "ytick.major.width": 1.8, "xtick.minor.size": 3, "ytick.minor.size": 2.5,
        "xtick.major.pad": 7, "ytick.major.pad": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 240})


def finish(fig, name, top_padding=0):
    bounds, padding = "tight", .12
    if top_padding:
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.12)
        bounds.y1 += top_padding
        padding = 0
    for ext in ("png", "pdf"):
        destination = OUT / "figures" / f"{name}.{ext}"
        fig.savefig(destination, bbox_inches=bounds, pad_inches=padding, facecolor="white")
        shutil.copy2(destination, BUNDLE / "figures" / destination.name)
    plt.close(fig)


def width_plot():
    specs = target_specs()
    ns = np.unique(np.rint(np.geomspace(32, 16384, 240)).astype(int))
    ws = ns+2*np.ceil(np.sqrt(ns)).astype(int)+1
    general = {a: threshold(lambda lam: rules.log_khat(a, 2*np.pi/lam)) for a in ACTS}
    table = []
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6), sharey=True)
    fig.subplots_adjust(left=.066, right=.986, top=.90, bottom=.29, wspace=.24)
    ratio_fig, ratio_axes = plt.subplots(1, 3, figsize=(18.5, 6))
    ratio_fig.subplots_adjust(left=.066, right=.986, top=.90, bottom=.29, wspace=.24)
    for col, act in enumerate(ACTS):
        ax = axes[col]
        ratio = ratio_axes[col]
        for current, axis in ((fig, ax), (ratio_fig, ratio)):
            box = axis.get_position()
            current.text((box.x0+box.x1)/2, .98, NAMES[act], ha="center", va="top",
                         fontsize=24, fontweight="bold")
        ax.axhline(general[act], color=".1", lw=2, ls=(0, (5, 3)), zorder=2)
        ratio.axhline(EPS, color=".1", lw=2, ls=(0, (5, 3)), zorder=2)
        for (name, label, omega, convention), color in zip(specs, COLORS):
            keep = 2*omega/ns < np.pi
            nn, ww = ns[keep], ws[keep]
            ls = np.array([threshold(lambda lam: rules.log_refined_ratio(act, lam, 2*omega/n)) for n in nn])
            log_g = rules.log_khat(act, 2*np.pi/ls)
            g = np.exp(log_g)
            ax.plot(nn, ls, color=color, lw=2.7)
            ratio.plot(nn, g, color=color, lw=2.7)
            table.extend({"act": act, "target": name, "N": int(n), "R": int((w-n-1)//2),
                          "W": int(w), "lambda_refined": float(l), "fourier_ratio": float(v),
                          "log10_fourier_ratio": float(logv/np.log(10)),
                          "omega": float(omega)} for n, w, l, v, logv in zip(nn, ww, ls, g, log_g))
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ratio.set_yscale("log")
        ratio.set_yticks([1e-32, EPS, 1e-8])
        ratio.set_yticklabels([r"$10^{-32}$", r"$\varepsilon_{\rm eff}$", r"$10^{-8}$"])
        ratio.set_ylim(1e-33, 1e-7)
        ratio.yaxis.set_minor_locator(FixedLocator([]))
        for axis in (ax, ratio):
            axis.set_xlabel(r"Width parameter $N$", labelpad=9)
            axis.set_xscale("linear")
            axis.set_xlim(0, 8192)
            axis.set_xticks([0, 2048, 4096, 6144, 8192])
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
            axis.grid(axis="y", which="major", alpha=.13, lw=.65)
        if col == 0:
            ax.set_ylabel(r"Predicted bandwidth $\lambda$")
            ratio.set_ylabel(r"$|\widehat K(2\pi/\lambda)|/|\widehat K(0)|$", fontsize=20)
    function_legend = [Line2D([0], [0], color=c, lw=2.7, label=s[1]) for s, c in zip(specs, COLORS)]
    general_legend = Line2D([0], [0], color=".1", ls=(0, (5, 3)), lw=2,
                            label=r"General rule: $G_K(\lambda)=\varepsilon_{\rm eff}$")
    for current in (fig, ratio_fig):
        current.legend(handles=function_legend+[general_legend], loc="lower center",
                       bbox_to_anchor=(.526, .014), ncol=5, borderaxespad=0,
                       frameon=False, fontsize=13.5, columnspacing=1.6, handlelength=1.8, labelspacing=.45)
    finish(fig, "bandwidth_width_rules", top_padding=.12)
    finish(ratio_fig, "bandwidth_fourier_ratios", top_padding=.12)
    (OUT / "data/width_predictions.json").write_text(json.dumps({"epsilon_eff": EPS,
        "general_roots": general, "geometry": "h=2/N, R=ceil(sqrt(N)), W=N+2R+1",
        "frequency_inputs": [{"name": s[0], "omega": s[2], "convention": s[3]} for s in specs],
        "rows": table}, indent=1))
    return general


def error_plot(general):
    source = SOURCE / "source_data/anchor_rule_rows_width.json"
    rows = json.loads(source.read_text())
    archived_keys = {(r['act'], r['lam'], r['N'], r['fn']) for r in rows}
    spec = json.loads((SOURCE / "source_data/anchor_rule_predictions_mean.json").read_text())["spectral"]
    targets = ("mix_2_6_10", "gauss20", "expsin", "sech5")
    titles = (r"$\sin(2\pi x)+\frac{1}{2}\sin(6\pi x)+\frac{1}{4}\sin(10\pi x)$",
              r"$e^{-20x^2}$", r"$e^{\sin(3\pi x)}$", r"$\mathrm{sech}(5x)$")
    # sech(5x) has transform (pi/5)*sech(pi*omega/10).
    centroid = quad(lambda w: w/np.cosh(np.pi*w/10), 0, 200, epsabs=1e-12)[0]/5
    spec["sech5"] = {"omega_M": centroid, "how": "whole-line Fourier-amplitude centroid: 40*Catalan/pi**2"}
    general = {**general, "erf": threshold(lambda lam: log_khat("erf", 2*np.pi/lam))}
    widths = (32, 48, 64, 96, 128, 256)
    additions = OUT / "data/additional_error_sweeps.json"
    added = json.loads(additions.read_text()) if additions.exists() else []
    lambdas = sorted(set(r["lam"] for r in rows))
    existing = {(r['act'], r['lam'], r['N'], r['fn']) for r in rows+added}
    with threadpool_limits(limits=1):
        for act in ERROR_ACTS:
            for n in widths:
                count = 0
                for lam in lambdas:
                    missing_targets = [t for t in targets if (act,lam,n,t) not in existing]
                    if missing_targets:
                        added.extend(measure(act, lam, n, missing_targets))
                        count += len(missing_targets)
                if count:
                    additions.write_text(json.dumps(added, indent=1))
                    print(f"Completed added curves: {act}, N={n}; {count} points", flush=True)
    rows += added
    cache = OUT / "data/refined_marker_fits_4x4.json"
    markers = json.loads(cache.read_text()) if cache.exists() else []
    existing_markers = {(r['act'],r['N'],r['fn']) for r in markers}
    with threadpool_limits(limits=1):
        for act in ERROR_ACTS:
            for target in targets:
                for n in widths:
                    if (act,n,target) not in existing_markers:
                        lam = threshold(lambda lam: log_refined(act, lam, 2*spec[target]["omega_M"]/n))
                        markers.extend(measure(act, lam, n, (target,)))
            cache.write_text(json.dumps(markers, indent=1))
    # Preserve all cached fits, but display only the requested width selection.
    markers = [r for r in markers if r['N'] in widths]
    assert len(markers) == 96 and all(np.isfinite(r["rel_l2"]) and r["rel_l2"] > 0 for r in markers)
    (OUT / 'data/selected_refined_marker_fits.json').write_text(json.dumps(markers, indent=1))
    fig, axes = plt.subplots(4, 4, figsize=(22, 16.2), sharex="col", sharey=True)
    fig.subplots_adjust(left=.066, right=.981, top=.91, bottom=.105, wspace=.16, hspace=.40)
    used = []
    for row, target in enumerate(targets):
        for col, act in enumerate(ERROR_ACTS):
            ax = axes[row, col]
            for n, color in zip(widths, WIDTH_COLORS):
                pts = sorted([r for r in rows if r["act"] == act and r["fn"] == target and r["N"] == n and r["p"] == 53], key=lambda r: r["lam"])
                assert len(pts) == 40
                marker = next(r for r in markers if r["act"] == act and r["fn"] == target and r["N"] == n)
                expected = threshold(lambda lam: log_refined(act, lam, 2*spec[target]["omega_M"]/n))
                assert abs(marker["lam"]-expected) < 1e-12
                used.extend(pts)
                # Include the newly measured marker in the unsmoothed polyline.
                joined = sorted(pts+[marker], key=lambda r: r["lam"])
                ax.plot([r["lam"] for r in joined], [r["rel_l2"] for r in joined], color=color, lw=2.7)
                ax.scatter(marker["lam"], marker["rel_l2"], s=40, marker="D",
                           facecolors="none", edgecolors="#D62728", lw=1.5, zorder=10)
            ax.axvline(general[act], color=".18", ls=(0, (5, 3)), lw=2, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(3e-17, 3)
            ax.set_yticks([1, 1e-4, 1e-8, 1e-12, 1e-16])
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(.2, .4, .6, .8), numticks=100))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(.05, .5 if act == "tanh" else 1)
            ax.set_xticks([.05, .2, .35, .5] if act == "tanh" else [.05, .25, .5, .75, 1])
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(axis="y", which="major", alpha=.13, lw=.65)
            ax.set_title(titles[row], fontsize=18, fontweight="normal", pad=12)
            if row == 3:
                ax.set_xlabel(r"Relative bandwidth $\lambda$", labelpad=7)
            if col == 0:
                ax.set_ylabel(r"Relative $L^2$ error", labelpad=8)
            if row == 0:
                ax.text(.5, 1.31, NAMES[act], transform=ax.transAxes, ha="center", va="bottom", fontsize=26, fontweight="bold")
    handles = [Line2D([0], [0], color=c, lw=2.7, label=f"$N={n}$") for n, c in zip(widths, WIDTH_COLORS)]
    handles += [Line2D([0], [0], color=".2", lw=2, ls=(0, (5, 3)), label="General rule"),
                Line2D([0], [0], color="#D62728", lw=0, marker="D", ms=5.5,
                       markerfacecolor="none", markeredgewidth=1.5, label="Refined rule")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5235, .012), ncol=8,
               frameon=False, fontsize=17, columnspacing=1.4)
    finish(fig, "bandwidth_error_rules")
    (OUT / "data/error_curves.json").write_text(json.dumps(used, indent=1))
    (OUT / "data/provenance.json").write_text(json.dumps({"source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "archived_points": sum((r['act'], r['lam'], r['N'], r['fn']) in archived_keys for r in used),
        "new_curve_points": sum((r['act'], r['lam'], r['N'], r['fn']) not in archived_keys for r in used),
        "new_marker_fits": len(markers), "smoothing": False,
        "geometry": {"halo_each_side": 32, "N": list(widths), "W": [n+65 for n in widths], "h": "2/N", "gamma": "lambda/h"},
        "fit": {"dtype": "float64", "samples": 2003, "evaluation_samples": 4001, "bias": True, "affine_skip": False,
                "solver": "scipy.linalg.lstsq; gelsd; cond=2**-52"},
        "frequency_inputs": {t: spec[t] for t in targets}, "epsilon_eff": EPS, "general_roots": general,
        "display_lambda_limits": {act: [.05, .5 if act == "tanh" else 1] for act in ERROR_ACTS}}, indent=1))


def main():
    for path in (OUT / "data", OUT / "figures", BUNDLE / "figures"):
        path.mkdir(parents=True, exist_ok=True)
    style()
    general = width_plot()
    print("General roots:", general, flush=True)
    error_plot(general)
    print("Wrote both figures and their numerical data:", OUT, flush=True)


if __name__ == "__main__":
    main()
