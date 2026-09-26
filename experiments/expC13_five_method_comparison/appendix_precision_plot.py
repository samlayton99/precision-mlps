"""Replot saved Runge/chirp size comparisons in the expC09 paper-figure style.

No constructions or training are run. The p=24 row uses the saved p-bit
arithmetic sweep (binary32 significand precision, wider exponent range).
At p=53, QUILLS and ChebNet use the native-FP64 tradeoff sweep, exactly as
in tapout_plot.py; the remaining methods use their saved p-bit sweep.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_C_geometry/expC13_five_method_comparison"
DATA = OUT / "data"
STEM = "appendix_tapout_fp32_fp64"
CAPS = [round(32 * 2 ** (k / 4)) for k in range(21)]
METHODS = {
    "quills": ("QUILL", "#0072B2"),
    "chebnet": ("ChebNet", "#E69F00"),
    "mhaskar": ("Mhaskar", "#8B5FBF"),
    "staircase": ("Staircase", "#6B4C3B"),
    "costarelli": ("Costarelli–Spigler", "#009E73"),
}


def saved_curves():
    sources = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    native = read(DATA / "tradeoff_rows.jsonl", lines=True)
    panels = []
    for p in (24, 53):
        for target in ("runge", "chirp"):
            panel = {"target": target, "precision_bits": p, "methods": {}}
            for method in METHODS:
                if p == 53 and method in ("quills", "chebnet"):
                    rows = [r for r in native if r.get("target") == target
                            and r.get("method") == method and r.get("p") == p
                            and r.get("pipeline") == "fp64"]
                    error_key = "val_rel_l2"
                    arithmetic = "native binary64"
                else:
                    filename = "chebnet_neurons" if method == "chebnet" else method
                    rows = read(DATA / "candidates" / f"{target}_{filename}_p{p}.json")
                    error_key = "rel_l2"
                    arithmetic = f"pfloat: {p}-bit significand, exponent range [-958, 959]"
                candidates = [(int(r["neurons"]), float(r[error_key])) for r in rows
                              if r.get("status") == "ok" and "neurons" in r
                              and np.isfinite(r.get(error_key, np.inf))]
                assert candidates, (target, method, p)
                values, selected = [], []
                for cap in CAPS:
                    eligible = [(n, e) for n, e in candidates if n <= cap]
                    winner = min(eligible, key=lambda item: item[1]) if eligible else None
                    selected.append(winner[0] if winner else None)
                    values.append(winner[1] if winner else None)
                finite = np.array([v for v in values if v is not None])
                assert np.all(finite > 0) and np.all(np.diff(finite) <= 0)
                panel["methods"][method] = {
                    "arithmetic": arithmetic,
                    "saved_candidate_count": len(candidates),
                    "relative_l2": values,
                    "selected_candidate_neurons": selected,
                }
            panels.append(panel)
    return {
        "metric": "minimum saved validation relative L2 error with at most N total hidden neurons",
        "neuron_caps": CAPS,
        "precision_note": "p=24 matches FP32 significand precision, not its exponent range. "
                          "The saved p=24 QUILL width grid is coarser than the native p=53 grid.",
        "source_sha256": sources,
        "panels": panels,
    }


def figure(data):
    # Use the reference figure's colors and hollow circles, sized for a paper.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 12.5, "axes.titlesize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.133, right=.955, bottom=.155, top=.915,
                        wspace=.15, hspace=.32)
    for index, (ax, panel) in enumerate(zip(axes.flat, data["panels"])):
        p, target = panel["precision_bits"], panel["target"]
        precision = "FP32 precision" if p == 24 else "FP64"
        ax.set_title(f"{target.capitalize()} ({precision})",
                     fontweight="bold", pad=10)
        for method, (label, color) in METHODS.items():
            values = [np.nan if v is None else v for v in panel["methods"][method]["relative_l2"]]
            ax.plot(data["neuron_caps"], values, color=color, lw=1.8,
                    marker="o", ms=3.5, mfc="white", mec=color, mew=1.05,
                    label=label, zorder=5 if method == "quills" else 4)
        ax.set_xscale("log", base=2)
        # A small margin keeps the endpoint circles fully visible.
        ax.set_xlim(30, 1080)
        ticks = [32, 64, 128, 256, 512, 1024]
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.xaxis.set_minor_locator(FixedLocator(
            [w * mult for w in ticks[:-1] for mult in (1.25, 1.5, 1.75)]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yscale("log")
        ax.set_ylim(3e-17, 10)
        ax.yaxis.set_major_locator(FixedLocator(10. ** np.arange(-16, 1, 4)))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.,), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if index >= 2:
            ax.set_xlabel(r"Total hidden neurons $W$", labelpad=6)
        if index % 2 == 0:
            ax.set_ylabel(r"Relative $L^2$ error", labelpad=5)
        ax.grid(True, axis="y", which="major", color=".91", lw=.65)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_linewidth(1.65)
        ax.tick_params(which="major", direction="out", length=4.3, width=1.2, pad=4)
        ax.tick_params(which="minor", direction="out", length=2.5, width=.75)
        ax.tick_params(axis="x", labelbottom=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .014),
               ncol=5, frameon=False, fontsize=9,
               handlelength=2.1, handletextpad=.5, columnspacing=1.15)
    return fig


CAPTION = r"""\caption{Measured accuracy versus total hidden-neuron budget for Runge,
$f(x)=1/(1+25x^2)$ (left), and the chirp,
$f(x)=\sin(8\pi(x+1)^2)$ (right), on $[-1,1]$.
Each marker gives the smallest validation relative $L^2$ error among saved
candidates with at most $W$ hidden neurons, including all layers and halo
neurons; plateaus may retain a smaller candidate.
The top row uses 24-bit significands (FP32 precision) with an expanded exponent
range, rather than native binary32 arithmetic. The bottom row uses native
FP64 for QUILL and ChebNet, and the saved 53-bit arithmetic runs for the other
methods. QUILL's 24-bit candidate grid is coarser; budgets without a saved
candidate are left blank. All panels share the same scales.}
"""


def main():
    data = saved_curves()
    (DATA / f"{STEM}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    out = OUT / "figures"
    out.mkdir(exist_ok=True)
    fig = figure(data)
    fig.savefig(out / f"{STEM}.png", dpi=450, facecolor="white")
    fig.savefig(out / f"{STEM}.pdf", facecolor="white")
    plt.close(fig)
    (out / f"{STEM}_caption.tex").write_text(CAPTION)
    (out / f"{STEM}_insert.tex").write_text(
        "% Paste after the final paragraph of the comparison-bounds appendix.\n"
        "% Upload the PNG to figures/. Requires graphicx (already used by the paper).\n"
        "\\begin{figure}[tbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{figures/{STEM}.png}}\n"
        + CAPTION
        + "  \\label{fig:appendix-method-comparison}\n"
        "\\end{figure}\n"
    )
    for panel in data["panels"]:
        print(panel["target"], panel["precision_bits"],
              {m: r["relative_l2"][-1] for m, r in panel["methods"].items()})
    print(out / f"{STEM}.png")


if __name__ == "__main__":
    main()
