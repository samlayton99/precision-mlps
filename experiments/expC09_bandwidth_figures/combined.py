"""Render the all-target refinement / chirp bandwidth / chirp precision PNG.

Uses the saved, validated observations; does not perform new fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.analytic_packet import OUT as PACKET_OUT
from experiments.expC09_bandwidth_figures.smooth_absolute import OUT as ABS_OUT

OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"


def main(precision_choice="rule", precision_width=1024, *, comparison_data=None,
         true_precision_data=None, runge_precision_data=None, paper_style=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,
                                  NullFormatter, NullLocator)

    convergence_data = OUT / "eight_target_convergence/data"
    chirp_data = OUT / "additional_targets/data"
    sources = [convergence_data / "config.json", convergence_data / "measurements.jsonl",
               chirp_data / "predictions.json", chirp_data / "measurements.jsonl"]
    cc = json.loads(sources[0].read_text())["config"]
    refinement = sorted([json.loads(s) for s in sources[1].read_text().splitlines()],
                        key=lambda r: r["width"])
    manifest = json.loads(sources[2].read_text())
    config = manifest["config"]
    rows = [json.loads(s) for s in sources[3].read_text().splitlines()]
    rows = [r for r in rows if r["target"] == "chirp" and r["halo"] == config["halo_per_side"]]
    predictions = [p for p in manifest["predictions"] if p["target"] == "chirp"]
    extra_data = OUT / "combined/data/extra_bandwidth"
    sources += [extra_data / "predictions.json", extra_data / "measurements.jsonl"]
    extra = json.loads(sources[-2].read_text())
    for key in ("halo_per_side", "train_points", "eval_points", "lambda_min", "lambda_max", "lambda_count"):
        assert extra["config"][key] == config[key]
    extra_rows = [json.loads(s) for s in sources[-1].read_text().splitlines()]
    for pr in extra["predictions"]:
        expected = set(np.geomspace(config["lambda_min"], config["lambda_max"], config["lambda_count"]))
        expected.update(pr[rule]["lambda"] for rule in ("basic", "refined"))
        cells = [r for r in extra_rows if r["width"] == pr["width"]]
        assert len(cells) == len(expected) and {r["lambda"] for r in cells} == expected
        assert all(r["target"] == "chirp" and r["p"] == 53
                   and r["halo"] == config["halo_per_side"] for r in cells)
    rows += extra_rows
    predictions += extra["predictions"]
    bandwidth_widths = [*config["bandwidth_widths"], *extra["config"]["bandwidth_widths"]]
    assert cc["lambda"] == config["refinement_lambda"] == .25
    assert cc["halo_per_side"] == config["halo_per_side"] == 24
    assert all(r["targets"] == cc["targets"] for r in refinement)
    assert len(refinement) == len({r["width"] for r in refinement}) == 41
    packet_data = PACKET_OUT / "data"
    sources += [packet_data / "config.json", packet_data / "measurements.jsonl"]
    packet_manifest = json.loads(sources[-2].read_text())
    packet = {r["width"]: r for r in map(json.loads, sources[-1].read_text().splitlines())}
    assert set(packet) == {r["width"] for r in refinement}
    for r in refinement:
        p = packet[r["width"]]
        assert all(p[k] == r[k] for k in ("N", "halo_per_side", "lambda", "train_points", "eval_points"))
    sources += [ABS_OUT / "data/config.json", ABS_OUT / "data/measurements.jsonl"]
    abs_manifest = json.loads(sources[-2].read_text())
    smooth_abs = {r["width"]: r for r in map(json.loads, sources[-1].read_text().splitlines())}
    assert set(smooth_abs) == {r["width"] for r in refinement}
    for r in refinement:
        a = smooth_abs[r["width"]]
        assert all(a[k] == r[k] for k in ("N", "halo_per_side", "lambda", "train_points", "eval_points"))

    comparison_rows = None
    if true_precision_data is not None:
        # expC11: the same panel with the model and the least-squares solve both at p bits
        true_precision_data = Path(true_precision_data)
        sources += [true_precision_data / "config.json", true_precision_data / "summary.json"]
        precision_manifest = json.loads(sources[-2].read_text())
        precision_config = precision_manifest["config"]
        precision_rows = json.loads(sources[-1].read_text())
    elif comparison_data is None:
        precision_run = f"precision_law_W{precision_width}" + ("_strict" if precision_width == 1024 else "")
        precision_data = OUT / precision_run / "data"
        sources += [precision_data / "config.json", precision_data / "summary.json",
                    precision_data / "measurements.jsonl"]
        precision_manifest = json.loads((precision_data / "config.json").read_text())
        precision_config = precision_manifest["config"]
        precision_rows = json.loads((precision_data / "summary.json").read_text())
    else:
        comparison_data = Path(comparison_data)
        sources += [comparison_data / "config.json", comparison_data / "summary.json"]
        precision_manifest = json.loads(sources[-2].read_text())
        comparison_rows = json.loads(sources[-1].read_text())
        precision_config = {"width": precision_manifest["config"]["width_budget"],
                            "precisions": list(range(precision_manifest["config"]["precision_min"],
                                                     precision_manifest["config"]["precision_max"]+1))}
        precision_rows = [{"p": r["p"], "width": r["quill_width"], "rule_error": r["quill_error"]}
                          for r in comparison_rows]
    assert [r["p"] for r in precision_rows] == precision_config["precisions"]
    assert all(r["width"] == precision_config["width"] for r in precision_rows)
    assert precision_choice in ("rule", "best")
    if not all(f"{precision_choice}_error" in r for r in precision_rows):
        raise ValueError(f"No {precision_choice} bandwidth measurements saved for W={precision_width}")

    runge_rows = None
    if runge_precision_data is not None:
        assert true_precision_data is not None and precision_choice == "rule"
        runge_precision_data = Path(runge_precision_data)
        sources += [runge_precision_data / "config.json", runge_precision_data / "summary.json",
                    runge_precision_data / "validation.json"]
        runge_manifest = json.loads(sources[-3].read_text())
        runge_rows = json.loads(sources[-2].read_text())
        assert json.loads(sources[-1].read_text())["complete"]
        assert runge_manifest["config"]["target"] == "runge25"
        for key in ("width", "halo_per_side", "train_points", "eval_points", "exponent_range", "solver"):
            assert runge_manifest["config"][key] == precision_config[key]
        assert [r["p"] for r in runge_rows] == precision_config["precisions"]
        assert all(r["width"] == precision_config["width"] for r in runge_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 18 if true_precision_data is not None else 14,
                         "axes.titlesize": 20 if true_precision_data is not None else 16,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.3))
    fig.subplots_adjust(left=.055, right=.99, bottom=.15, top=.87, wspace=.23)
    titles = ["a) Width Scaling",
              "b) Bandwidth Selection",
              "c) Precision Law"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="center", y=1.02, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 10)
        ax.set_yticks([10.**k for k in range(-16, 1, 4)])
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.grid(True, which="major", alpha=.2, lw=.65)

    colors = ["#0072B2", "#E69F00", "#8B5FBF", "#D55E00", "#009E73", "#CC79A7", "#6B4C3B", "#444444"]
    markers = ["o", "s", "^", "D", "o", "s", "^", "D"]
    display_labels = {
        "sine4": r"$\sin(4\pi x)$",
        "sine24": r"$\sin(24\pi x)$",
        "weak_high_frequency": r"$\sin(2\pi x)+10^{-3}\sin(40\pi x)$",
        "chirp": r"$\sin\!\left(8\pi(x+1)^2\right)$",
        "runge25": r"$1/(1+25x^2)$",
        "runge100": r"$1/(1+100x^2)$",
        "gaussian_envelope": r"$\log(\cosh(10x))/10$",
        "smooth_bump": r"$e^{-8(x-0.2)^2}\sin(8\pi x)$",
    }
    widths = [r["width"] for r in refinement]
    for j, (name, color, marker) in enumerate(zip(cc["targets"], colors, markers)):
        replacements = {"smooth_bump": packet, "gaussian_envelope": smooth_abs}
        if name in replacements:
            measurements = replacements[name]
            errors = [measurements[w]["relative_l2"] for w in widths]
        else:
            errors = [r["relative_l2"][j] for r in refinement]
        axes[0].plot(widths, errors, color=color, marker=marker, ms=2.5, lw=1.8,
                     label=display_labels[name])
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlim(cc["width_min"], 1024)
    axes[0].xaxis.set_major_locator(FixedLocator([64, 128, 256, 512, 1024]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_xlabel(r"Total width $W$")
    axes[0].legend(loc="upper right", fontsize=12 if true_precision_data is not None else 10,
                   framealpha=.95, ncol=1)

    refined = "#c73435"
    rule_handles = [Line2D([], [], marker="D", mfc="white", mec=refined,
                           mew=1.4, ls="none",
                           label=r"Predicted $\lambda$")]

    def add_curve(ax, width, p, color, label):
        cells = sorted([r for r in rows if r["width"] == width and r["p"] == p],
                       key=lambda r: r["lambda"])
        assert cells and len(cells) == len({r["lambda"] for r in cells})
        ax.plot([r["lambda"] for r in cells], [r["relative_l2"] for r in cells],
                color=color, lw=2.0, label=label)
        prediction = next(pr for pr in predictions if pr["width"] == width and pr["p"] == p)
        lam = prediction["refined"]["lambda"]
        error = next(r["relative_l2"] for r in cells if r["lambda"] == lam)
        ax.scatter([lam], [error], s=34, marker="D", facecolor=color,
                   edgecolor=refined, linewidth=1.4, zorder=5)

    width_palette = [plt.get_cmap("viridis")(x)
                     for x in np.linspace(.08, .86, len(bandwidth_widths))]
    assert len(width_palette) == len(bandwidth_widths)
    for width, color in zip(bandwidth_widths, width_palette):
        add_curve(axes[1], width, 53, color, rf"$W={width}$")
    ax = axes[1]
    ax.set_xscale("log")
    ax.set_xlim(config["lambda_min"], config["lambda_max"])
    ax.xaxis.set_major_locator(FixedLocator([.05, .1, .2, .5, 1.]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"Relative bandwidth $\lambda$")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, *rule_handles], loc="lower right", ncol=2,
              framealpha=.95, fontsize=9 if true_precision_data is not None else 8,
              columnspacing=.7, handlelength=1.5,
              title=r"$f(x)=\sin\!\left(8\pi(x+1)^2\right)$",
              title_fontsize=10 if true_precision_data is not None else 9)

    ax = axes[2]
    precision_label = r"QUILL (predicted $\lambda$)" if precision_choice == "rule" else r"Best sampled $\lambda$"
    if runge_rows is not None:
        precision_label = r"$\sin\!\left(8\pi(x+1)^2\right)$"
    ax.plot([r["p"] for r in precision_rows], [r[f"{precision_choice}_error"] for r in precision_rows],
            "o-", color=plt.get_cmap("viridis")(.45), lw=2.0, ms=3, label=precision_label)
    if runge_rows is not None:
        ax.plot([r["p"] for r in runge_rows], [r["rule_error"] for r in runge_rows],
                "s-", color=plt.get_cmap("viridis")(.08), lw=2.0, ms=3,
                label=r"$1/(1+25x^2)$")
    if comparison_rows is not None:
        ax.plot([r["p"] for r in comparison_rows], [r["mhaskar_error"] for r in comparison_rows],
                "s-", color=plt.get_cmap("viridis")(.08), lw=2., ms=3,
                label="Mhaskar (selected degree/step)")
    # Fix the one-bit/halved-error slope; fit only the log-space intercept
    # on the descending portion, excluding the high-precision floor.
    fit_rows = [r for r in precision_rows if 16 <= r["p"] <= 40]
    log2_c = float(np.mean([np.log2(r[f"{precision_choice}_error"]) + r["p"]
                           for r in fit_rows]))
    reference_p = np.array([min(precision_config["precisions"]), max(precision_config["precisions"])])
    ax.plot(reference_p, np.exp2(log2_c-reference_p), "--", color="#333333",
            lw=1.8, label=r"$O(\log(1/\varepsilon))$", zorder=4)
    ax.set_xlim(min(precision_config["precisions"]), max(precision_config["precisions"]))
    ax.set_xticks([8, 16, 24, 32, 40, 48, 53])
    ax.set_xlabel(r"Working precision $p$ (bits)")
    ax.legend(loc="upper right", bbox_to_anchor=(1., .88) if comparison_rows is not None else (1., 1.),
              framealpha=.95, fontsize=13 if true_precision_data is not None else (10 if comparison_rows is not None else 11),
              title=(r"QUILL (predicted $\lambda$)" if runge_rows is not None
                     else r"$f(x)=\sin\!\left(8\pi(x+1)^2\right)$"),
              title_fontsize=13 if true_precision_data is not None else 11)

    if true_precision_data is not None:
        fig.subplots_adjust(left=.063, wspace=.25)
        def log_error_label(value, _):
            exponent = int(round(np.log10(value)))
            return rf"$10^{{{exponent}}}$" if exponent % 4 == 0 else ""

        for ax in axes:
            for line in ax.lines:
                line.set_linewidth(line.get_linewidth() + .8)
            for handle in ax.get_legend().legend_handles:
                if isinstance(handle, Line2D) and handle.get_linestyle() != "None":
                    handle.set_linewidth(handle.get_linewidth() + .8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_linewidth(1.5)
            ax.tick_params(which="major", direction="out", length=5, width=1.25,
                           pad=7, top=False, right=False)
            ax.tick_params(which="minor", direction="out", length=2.8, width=.8,
                           top=False, right=False)
            # Long ticks mark every decade; 2,...,9 cluster logarithmically within it.
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.,), numticks=100))
            ax.yaxis.set_major_formatter(FuncFormatter(log_error_label))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=200))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="y", which="minor", length=2.2, width=.65)
            ax.grid(axis="y", which="major", alpha=.13)
            # Keep the lowest y label clear of the x-axis corner.
            ax.set_ylim(3e-17, 10)
            for tick, value in zip(ax.yaxis.get_major_ticks(), ax.yaxis.get_majorticklocs()):
                tick.gridline.set_visible(int(round(np.log10(value))) % 2 == 0)
            ax.set_axisbelow(True)
        axes[0].xaxis.set_minor_locator(FixedLocator(
            [w * m for w in (64, 128, 256, 512) for m in (1.25, 1.5, 1.75)]))
        axes[1].set_xscale("linear")
        axes[1].set_xlim(.05, .65)
        axes[1].xaxis.set_major_locator(FixedLocator([.05, .25, .45, .65]))
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        axes[1].xaxis.set_minor_locator(NullLocator())
        for ax in axes[:2]:
            ax.xaxis.set_minor_formatter(NullFormatter())
        if runge_rows is None:
            # Publication styling for the chirp-only precision figure.
            fig.set_size_inches(18., 5.6)
            fig.subplots_adjust(left=.072, bottom=.18, top=.86, wspace=.20)
            axes[0].xaxis.set_minor_locator(NullLocator())
            for ax in axes[1:]:
                ax.set_ylabel("")
            for ax in axes:
                ax.title.set_fontsize(23)
                ax.xaxis.label.set_fontsize(21)
                ax.yaxis.label.set_fontsize(21)
                ax.tick_params(which="major", labelsize=13, length=6, width=1.6)
                ax.grid(False, axis="x", which="both")
                for side in ("bottom", "left"):
                    ax.spines[side].set_linewidth(2.6)
                for line in ax.lines:
                    line.set_linewidth(3.2 if line.get_linestyle() != "--" else 2.8)
                for handle in ax.get_legend().legend_handles:
                    if isinstance(handle, Line2D) and handle.get_linestyle() != "None":
                        handle.set_linewidth(3.2 if handle.get_linestyle() != "--" else 2.8)
            for ax, size in zip(axes, (13, 10, 14)):
                legend = ax.get_legend()
                for text in legend.get_texts():
                    text.set_fontsize(size)
                legend.get_title().set_fontsize(size)
            for ax in (axes[0], axes[2]):
                for line in [*ax.lines, *ax.get_legend().legend_handles]:
                    if isinstance(line, Line2D) and line.get_marker() not in ("None", "", None):
                        line.set_marker("o")
                        line.set_markersize(5.)
                        line.set_markerfacecolor("white")
                        line.set_markeredgecolor(line.get_color())
                        line.set_markeredgewidth(1.5)
                        if line in ax.lines:
                            line.set_markevery(2)
                            line.set_zorder(3)
            # Keep the reference behind the white marker interiors.
            axes[2].lines[-1].set_zorder(2)
        dest = (runge_precision_data if runge_rows is not None else true_precision_data).parent / "figures"
    else:
        dest = (comparison_data.parent / "figures") if comparison_rows is not None else OUT / "combined/figures"
    dest.mkdir(parents=True, exist_ok=True)
    if true_precision_data is not None:
        path = dest / ("all_targets_chirp_runge25_precision_law_true.png" if runge_rows is not None
                       else "all_targets_chirp_precision_law_true.png")
    else:
        path = dest / ("three_panel_mhaskar_comparison.png" if comparison_rows is not None
                       else f"all_targets_chirp_precision_law_{precision_choice}.png")
    if paper_style:
        assert true_precision_data is not None and runge_rows is None
        from experiments.expC11_true_precision_law import paper_style as publication_style
        publication_style.apply(fig, axes)
        path = path.with_name(path.stem + "_600dpi.png")
    fig.savefig(path, dpi=600 if paper_style else 240)
    if comparison_rows is None and true_precision_data is None:
        fig.savefig(dest / f"all_targets_chirp_precision_law_{precision_choice}_W{precision_width}.png", dpi=240)
        if precision_width == 1024:
            fig.savefig(dest / f"all_targets_chirp_precision_law_{precision_choice}_W1024_strict.png", dpi=240)
        if precision_choice == "rule":
            fig.savefig(dest / "all_targets_chirp_bandwidth.png", dpi=240)
    plt.close(fig)
    provenance = {"sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
                  "panels": {"a": "Eight targets, saved width sweep displayed from 64 to 1024, lambda=0.25, FP64",
                             "b": "Chirp, widths " + "/".join(map(str, bandwidth_widths)) + ", FP64",
                             "c": f"Chirp, W={precision_config['width']}, integer p=8..53, {precision_choice} lambda"},
                  "precision_model": {"a_b": "Native FP64 features and readout evaluation",
                                      "c": precision_manifest.get("precision_model", precision_config.get("solver"))},
                  "halo_per_side": 24,
                  "replacements": [{"removed": "smooth_bump", "added": packet_manifest["formula"]},
                                   {"removed": "gaussian_envelope", "added": abs_manifest["formula"]}],
                  "displayed_rule": "Refined rule only, labeled Predicted lambda",
                  "precision_reference": {"formula": "epsilon=C*2**(-p)",
                                          "display_label": "O(log(1/epsilon))",
                                          "interpretation": "Required significand bits p scale logarithmically with inverse error in the descending regime",
                                          "fixed_log2_slope": -1,
                                          "intercept_fit_precisions": [16, 40],
                                          "log2_C": log2_c, "C": float(np.exp2(log2_c)),
                                          "fit_method": "Least squares in log2 error, intercept only"},
                  "chirp_refined_rule": "Mean local angular frequency 16*pi; heuristic"}
    if true_precision_data is not None:
        if paper_style:
            provenance["figure_style"] = {**publication_style.STYLE_SOURCE,
                "dpi": 600, "figsize_inches": list(fig.get_size_inches()),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "style_sha256": hashlib.sha256(Path(publication_style.__file__).read_bytes()).hexdigest()}
        provenance["panels"]["c"] = ("Chirp, W=1024, integer p=8..53, refined-rule lambda; model and DGELSS solve both "
                                     "in p-bit arithmetic (expC11)")
        if runge_rows is not None:
            provenance["panels"]["c"] = ("Chirp and Runge 25, W=1024, integer p=8..53; model and DGELSS solve "
                                         "both in p-bit arithmetic; each target uses its own refined-rule lambda")
            provenance["precision_reference"]["fitted_target"] = "chirp only; no Runge fit"
            provenance["runge_refined_rule"] = {"omega_scale": 5., "interval": [.03, 3.],
                "at_upper_search_limit": [pr["p"] for pr in runge_manifest["predictions"]
                                          if pr["refined"]["status"] == "upper_search_limit"]}
        provenance_name = "provenance_true_precision_law_600dpi.json" if paper_style else "provenance_true_precision_law.json"
        (dest.parent / provenance_name).write_text(json.dumps(provenance, indent=2)+"\n")
        print(path)
        return path
    if comparison_rows is not None:
        provenance["panels"]["c"] = "Chirp, maximum W=1024, QUILL and Mhaskar; arithmetic protocol specified in precision_model.c"
        (dest.parent / "provenance_three_panel.json").write_text(json.dumps(provenance, indent=2)+"\n")
        print(path)
        return path
    (dest.parent / f"provenance_precision_law_{precision_choice}.json").write_text(json.dumps(provenance, indent=2)+"\n")
    if precision_choice == "rule":
        (dest.parent / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision-law", choices=("rule", "best", "both"), default="rule")
    parser.add_argument("--precision-width", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--true-precision-data", type=Path, default=None,
                        help="expC11 data directory: render panel (c) from the fully p-bit sweep instead")
    args = parser.parse_args()
    if args.true_precision_data is not None:
        main("rule", 1024, true_precision_data=args.true_precision_data)
        raise SystemExit
    if args.precision_width == 1024 and args.precision_law != "rule":
        parser.error("W=1024 has predicted-bandwidth measurements only; use --precision-width 512 for best/both")
    for choice in ("rule", "best") if args.precision_law == "both" else (args.precision_law,):
        main(choice, args.precision_width)
