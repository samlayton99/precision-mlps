"""Replot saved D34 evidence for the PI note; never train or generate prose.

Run from the repository root:
    MPLCONFIGDIR=/tmp/gamma-pi-mpl .venv/bin/python -m \
        experiments.expD34_readout_race.pi_brief_figures
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/checkpoint_D_optimizers/expD34_readout_race"
OUTPUT = ROOT / "docs/assets/gamma_scale_acquisition_pi_brief"
TARGETS = {"sine": "Sine", "moment3": "Coarse + degree 3", "moment9": "Coarse + degree 9"}
INK, BLUE, ORANGE = "#263442", "#2f60a8", "#d4871a"


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def band(ax, x, values, **kwargs):
    ax.plot(x, np.median(values, axis=0), **kwargs)
    ax.fill_between(x, values.min(axis=0), values.max(axis=0),
                    color=kwargs.get("color", INK), alpha=.12, linewidth=0)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK,
                         "axes.titlepad": 8, "axes.linewidth": .6,
                         "pdf.fonttype": 42})
    paths = {
        "rates": SOURCE / "core20k/summary.csv",
        "states": SOURCE / "adam_force_extension/states.csv.gz",
        "windows": SOURCE / "adam_force_extension/windows.csv",
    }
    rates = [r for r in read_rows(paths["rates"])
             if r["target"] == "moment9" and int(r["n"]) == 128
             and int(r["degree"]) == 0 and int(r["seed"]) in range(5)]
    assert len(rates) == 35 and all(r["complete"] == "True" for r in rates)
    kappas = sorted({float(r["kappa"]) for r in rates})
    lookup = {(int(r["seed"]), float(r["kappa"])): r for r in rates}
    assert len(lookup) == len(rates)
    fields = ("coarse_1pct_step", "signed_mean_change")
    values = {key: np.array([[float(lookup[s, k][key]) for k in kappas]
                            for s in range(5)]) for key in fields}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15), layout="constrained")
    for ax, field, color in zip(axes, fields, (ORANGE, BLUE)):
        for row in values[field]:
            ax.plot(kappas, row, color=color, alpha=.2, lw=.7)
        band(ax, kappas, values[field], color=color, marker="o", ms=3, lw=1.5)
        ax.set(xscale="log", yscale="log", xlabel=r"Readout / geometry rate $\kappa$")
        ax.set_xticks([1e-4, 1e-2, 1, 100])
        ax.grid(axis="y", alpha=.15)
    axes[0].set(title="A  Coarse fitting happens sooner", ylabel="Updates to 1%\ncoarse residual")
    axes[1].set(title="B  Less net slope growth", ylabel=r"Mean-slope increase $\Delta\bar\gamma$")
    save(fig, "rate_intervention")

    windows = [r for r in read_rows(paths["windows"])
               if r["bundle"].startswith("primary_") and r["optimizer"] == "gd"
               and int(r["start"]) == 20000 and int(r["end"]) == 600000]
    assert len(windows) == 65
    fractions = np.array([float(r["raw_tracking_fraction"]) for r in windows])
    assert len({(r["target"], r["seed"]) for r in windows}) == 65
    for r in windows:
        a, b = float(r["raw_channel_path_tracking"]), float(r["raw_channel_path_effective"])
        assert np.isclose(float(r["raw_tracking_fraction"]), a / (a + b), rtol=1e-12)

    states = [r for r in read_rows(paths["states"])
              if r["bundle"].startswith("primary_") and r["optimizer"] == "gd"
              and r["target"] in TARGETS and int(r["step"]) >= 20000]
    steps = sorted({int(r["step"]) for r in states})
    state_lookup = {(r["target"], int(r["seed"]), int(r["step"])): r for r in states}
    assert len(state_lookup) == len(states) == 3 * 5 * len(steps)
    assert all(r["gd_balance_resolved"] == "True" and float(r["eta"]) == .002 for r in states)
    for r in states:
        assert np.isclose(float(r["effective_coupling"]),
                          (float(r["effective_norm"]) / float(r["fine_residual_norm"]))**2,
                          rtol=1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 2.8), layout="constrained", sharex=True)
    endpoints = {}
    for col, (target, title) in enumerate(TARGETS.items()):
        def matrix(key):
            return np.array([[float(state_lookup[target, s, n][key]) for n in steps]
                             for s in range(5)])
        time = np.array(steps) * .002
        for field, label, color, ls, lw in (
                ("raw_norm", "Full slope gradient", INK, "-", 2.4),
                ("effective_norm", "Effective fine force", BLUE, "--", 1.5),
                ("tracking_norm", "Coarse disagreement", ORANGE, "-", 1.2)):
            band(axes[0, col], time, matrix(field), color=color,
                 linestyle=ls, lw=lw, label=label)
        band(axes[1, col], time, matrix("effective_coupling"), color=BLUE, lw=1.5)
        axes[0, col].set_title(title)
        for ax in axes[:, col]:
            ax.set(xscale="log", yscale="log")
            ax.grid(axis="y", alpha=.15)
        axes[1, col].set(xlabel=r"Flow clock $t=0.002n$")
        axes[1, col].set_xticks([40, 200, 1200], ["40", "200", "1200"])
        endpoints[target] = {
            "effective_coupling_median": float(np.median(matrix("effective_coupling")[:, -1])),
            "fine_residual_norm_median": float(np.median(matrix("fine_residual_norm")[:, -1])),
        }
    axes[0, 0].set_ylabel("Slope-force norm")
    axes[1, 0].set_ylabel(r"Sensitivity $\mu_a^{\rm eff}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    save(fig, "effective_sensitivity")

    selected_fields = ("target", "seed", "step", "raw_norm", "effective_norm",
                       "tracking_norm", "fine_residual_norm", "effective_coupling")
    evidence = {
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths.values()},
        "rate_protocol": {"width": 177, "training_points": 2048, "steps": 20000,
                          "seeds": list(range(5)), "geometry_rate": .002,
                          "target": "0.3 phi_0 + 0.4 phi_1 + sqrt(0.75) phi_9"},
        "rate_anchors": {str(k): {
            key: float(np.median([float(lookup[s, k][key]) for s in range(5)]))
            for key in (*fields, "coarse_1pct_mean_gamma", "initial_mean_gamma", "heldout_mse")}
            for k in (1e-4, 1., 100.)},
        "mean_growth_ratio_equal_to_fast": float(np.median(values["signed_mean_change"][:, kappas.index(1.)]) /
                                                  np.median(values["signed_mean_change"][:, kappas.index(100.)])),
        "matched_coarse_growth_medians": {str(k): float(np.median([
            float(lookup[s, k]["coarse_1pct_mean_gamma"]) - float(lookup[s, k]["initial_mean_gamma"])
            for s in range(5)])) for k in (1e-4, 1., 100.)},
        "tracking_budget": {"cases": 65, "start": 20000, "end": 600000,
                            "maximum_fraction": float(max(fractions)),
                            "median_fraction": float(np.median(fractions))},
        "sensitivity_endpoints": endpoints,
        "rate_plot_data": [{key: r[key] for key in ("seed", "kappa", *fields)} for r in rates],
        "force_plot_data": [{key: r[key] for key in selected_fields} for r in states],
        "checks": {"unique_rate_cases": 35, "unique_force_states": len(states),
                   "tracking_fraction_identity": "passed", "sensitivity_identity": "passed"},
    }
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("rate_anchors", "mean_growth_ratio_equal_to_fast",
          "matched_coarse_growth_medians", "tracking_budget", "sensitivity_endpoints", "checks")}, indent=2))


if __name__ == "__main__":
    main()
