"""Pair archived GD force channels with slope scales; no training or prose output.

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
OUTPUT = ROOT / "docs/figures"
H = 2 / 128  # Construction spacing uses reference resolution, not actual width 177.
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
SCALES = (
    ("mean_gamma", "#059669", "Mean scale"),
    ("max_gamma", "#7c3aed", "Maximum scale"),
)


def main():
    traces = {name: {} for name, _ in TARGETS}
    for seed in range(5):
        folder = SOURCE / f"primary_{seed}"
        manifest = json.loads((folder / "manifest.json").read_text())
        assert (manifest["m"], manifest["width"], manifest["max_updates"]) == (2048, 177, 600000)
        columns = [manifest["metrics"].index(name) for name, _, _ in CHANNELS + SCALES]
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

    for target, _ in TARGETS:
        records = traces[target]
        assert set(records) == set(range(5)), target
        common = records[0][0]
        for updates, _ in records.values():
            np.testing.assert_array_equal(updates, common)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for page, targets in enumerate((TARGETS[:7], TARGETS[7:]), start=1):
        fig, axes = plt.subplots(len(targets), 2, figsize=(6.9, 7.0), sharex=True)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.065, top=0.91,
                            hspace=0.52, wspace=0.27)
        for row, (target, label) in enumerate(targets):
            records = traces[target]
            common = records[0][0]
            values = np.stack([records[seed][1] for seed in range(5)])
            below = values[:, :, 2] <= values[:, :, 1]
            assert np.all(below.any(axis=1)), target
            # Median of per-seed first saved crossings, not a crossing of medians.
            crossover = np.median(common[np.argmax(below, axis=1)])
            assert np.all(values[:, :, 3] <= values[:, :, 4])
            assert H * values[:, :, 4].max() < 0.3, "Scale axis would clip data"
            for col, channels in enumerate((CHANNELS, SCALES)):
                ax = axes[row, col]
                for j, (_, color, channel) in enumerate(channels):
                    yy = values[:, :, j] if col == 0 else H * values[:, :, 3 + j]
                    assert yy.min() >= (1e-12 if col == 0 else 5e-4)
                    ax.plot(common, np.median(yy, axis=0), color=color, lw=1.1, label=channel)
                    ax.fill_between(common, yy.min(axis=0), yy.max(axis=0),
                                    color=color, alpha=0.15, lw=0)
                ax.axvline(crossover, color="#6b7280", ls=":", lw=0.8)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(common[0], common[-1])
                ax.set_ylim((1e-12, 10) if col == 0 else (5e-4, 0.3))
                ax.xaxis.set_major_locator(FixedLocator([10, 1000, 100000]))
                ax.xaxis.set_major_formatter(LogFormatterMathtext())
                ax.yaxis.set_major_locator(FixedLocator(
                    [1e-12, 1e-8, 1e-4, 1] if col == 0 else [1e-3, 1e-2, 1e-1]))
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
                ax.tick_params(which="major", length=2.5, pad=2)
                ax.minorticks_off()
                ax.grid(axis="y", alpha=0.12, lw=0.4)
            axes[row, 0].set_title(label, loc="left", pad=3)
            axes[row, 1].axhline(0.25, color="#9ca3af", ls="--", lw=0.8)
        for col in range(2):
            handles, labels = axes[0, col].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3 if col == 0 else 2,
                       frameon=False, bbox_to_anchor=(0.285 if col == 0 else 0.785, 0.974),
                       fontsize=7.5, handlelength=1.5, columnspacing=0.8)
        fig.text(0.285, 0.995, "Slope-gradient norm", ha="center", va="top", fontsize=9)
        fig.text(0.785, 0.995, r"Normalized slope $\lambda=h|a|$", ha="center", va="top", fontsize=9)
        fig.supxlabel("Optimizer updates", y=0.012, fontsize=9)
        output = OUTPUT / f"d34_pi_forces_scales_{page}.png"
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)
        print(f"Wrote {output}")
    print(f"Verified 13 targets x 5 seeds x {len(common)} saved states")


if __name__ == "__main__":
    main()
