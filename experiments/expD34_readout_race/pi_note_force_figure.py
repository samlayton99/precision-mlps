"""Replot archived GD force channels for the PI note; no training or prose output.

Run from the repository root with:
    .venv/bin/python experiments/expD34_readout_race/pi_note_force_figure.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/curated"
OUTPUT = ROOT / "docs/figures/d34_pi_force_decomposition.png"
TARGETS = (
    ("sine", "Sine"), ("runge", "Runge"), ("moment3", "Degree 3"),
    ("moment5", "Degree 5"), ("moment9", "Degree 9"),
    ("mixed_sine", "Mixed sine"), ("localized_sine", "Localized sine"),
    ("chirp", "Chirp"), ("moment4", "Degree 4"),
    ("blend_m010", "Mix s = -0.1"), ("blend_p001", "Mix s = 0.01"),
    ("blend_p010", "Mix s = 0.1"), ("blend_p030", "Mix s = 0.3"),
)
CHANNELS = (
    ("raw_total_norm", "#111827", "Full"),
    ("raw_effective_norm", "#2563eb", "Effective fine"),
    ("raw_tracking_norm", "#d97706", "Coarse tracking"),
)


def main():
    traces = {name: {} for name, _ in TARGETS}
    for seed in range(5):
        folder = SOURCE / f"primary_{seed}"
        manifest = json.loads((folder / "manifest.json").read_text())
        assert (manifest["m"], manifest["width"], manifest["max_updates"]) == (2048, 177, 600000)
        columns = [manifest["metrics"].index(name) for name, _, _ in CHANNELS]
        with np.load(folder / "trace.npz") as archive:
            # Match the original figure's saved pre-update coordinate exactly.
            updates = archive["ends"] - 1
            assert np.all(np.diff(updates) > 0) and updates[-1] == 599999
            for i, case in enumerate(manifest["cases"]):
                if case["optimizer"] != "gd":
                    continue
                assert case["seed"] == seed and case["eta"] == 0.002
                assert seed not in traces[case["target"]]
                values = archive["values"][i][:, columns]
                assert np.all(np.isfinite(values)) and np.all(values > 0)
                traces[case["target"]][seed] = (updates, values)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(5, 3, figsize=(6.9, 7.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.07, top=0.94,
                        hspace=0.72, wspace=0.35)
    for ax, (target, label) in zip(axes.flat, TARGETS):
        records = traces[target]
        assert set(records) == set(range(5)), target
        common = records[0][0]
        for updates, _ in records.values():
            np.testing.assert_array_equal(updates, common)
        values = np.stack([records[seed][1] for seed in range(5)])
        for j, (_, color, channel) in enumerate(CHANNELS):
            # Same median/min/max as adam_summarize.py, on its identical grids.
            yy = values[:, :, j]
            ax.plot(common, np.median(yy, axis=0), color=color, lw=1, label=channel)
            ax.fill_between(common, yy.min(axis=0), yy.max(axis=0), color=color, alpha=0.13, lw=0)
        ax.set_title(label, pad=4)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("log")
        ax.set_xlim(common[0], common[-1])
        ax.set_ylim(1e-12, 10)
        ax.xaxis.set_major_locator(FixedLocator([10, 1000, 100000]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_locator(FixedLocator([1e-12, 1e-8, 1e-4, 1]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.tick_params(which="major", length=2.5, pad=2)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.12, lw=0.4)
    for ax in list(axes.flat)[len(TARGETS):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 0.999), fontsize=8.5, handlelength=2.4)
    fig.supxlabel("Optimizer updates", y=0.015, fontsize=9)
    fig.supylabel("Slope-gradient norm", x=0.008, fontsize=9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Verified 13 targets x 5 seeds x {len(common)} saved states; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
