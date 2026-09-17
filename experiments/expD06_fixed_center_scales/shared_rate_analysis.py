"""Audit the existing shared-rate Adam/envelope experiment; no new training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path

from .analyze import analyze_one
from .consolidate import mechanism_rows, spectral_evidence
from .run import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "shared_rate_analysis"
    (output / "figures").mkdir(parents=True, exist_ok=True)
    with (args.root / "consolidation" / "summary.csv").open() as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = [r for r in reader if r["optimizer"] == "adam" and r["initialization"] == "envelope"
                and r["arm"] in {"both", "raw"} and float(r["rate_r"]) == float(r["rate_g"])
                and r["epsilon_mode"] == "legacy" and r["bias_rate"] == ""
                and r["halo_init"] == r["halo_metric"] == "full"
                and r["target"] == "sine" and r["n"] == "512" and r["seed"] in {"0", "1"}]
    expected = {(arm, rate, seed) for arm, rates in
                [("both", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), ("raw", [1e-4, 1e-3, 1e-2])]
                for rate in rates for seed in [0, 1]}
    assert len(rows) == len(expected)
    assert {(r["arm"], float(r["rate_r"]), int(r["seed"])) for r in rows} == expected
    assert all(r["trace_verified_steps"] == r["latest_step"] == "320000" for r in rows)
    with (output / "summary.csv").open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    keys = {r["key"] for r in rows}
    windows = json.loads((args.root / "consolidation" / "windows.json").read_text())
    write_json(output / "windows.json", [w for w in windows if w["key"] in keys])
    for row in rows:
        for name in ["n", "seed"]:
            row[name] = int(row[name])
        for name in ["rate_r", "rate_g"]:
            row[name] = float(row[name])
    representatives = [r for r in rows if r["rate_r"] == 1e-3
                       or (r["arm"] == "both" and r["rate_r"] == 1e-2)]
    steps = [0, 20000, 80000, 160000, 320000]
    tasks = {(r["folder"], step, 1e-12, False) for r in representatives for step in steps}
    tasks |= {(r["folder"], step, cutoff, False) for r in representatives if r["arm"] == "both"
              for step in [20000, 320000] for cutoff in [1e-10, 1e-14]}
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        metrics = list(pool.map(analyze_one, sorted(tasks)))
    write_json(output / "checkpoint_metrics.json", metrics)
    write_json(output / "mechanism.json", [m for r in representatives for m in mechanism_rows(Path(r["folder"]), steps)])
    # Export histories and numerical checks without the broad campaign's ranked figures.
    choices = [{"cases": [r["key"] for r in representatives], "score_key": "shared_rate_audit"}]
    spectral_evidence(representatives, choices, output)
    write_json(output / "selection.json", {
        "constraint": "One shared trained-coordinate rate; fixed reference metric, bias scale, and epsilon",
        "included": sorted(keys), "representatives": choices[0]["cases"],
        "reason": "All five shared-rate candidates; inspect the lowest paired late-window error, the next larger rate, and the same-rate raw control",
        "new_training_steps": 0})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    horizon = steps[1:]
    groups = [("raw", 1e-3, "Unscaled, eta=0.001", "#777777"),
              ("both", 1e-3, "Prescribed scales, eta=0.001", "#0072B2"),
              ("both", 1e-2, "Prescribed scales, eta=0.01", "#D55E00")]
    indexed = {(m["case"], m["step"], m["cutoff"]): m for m in metrics}
    for arm, rate, label, color in groups:
        for row in representatives:
            if (row["arm"], row["rate_r"]) != (arm, rate):
                continue
            values = [[float(row[f"{field}_{step}"]) for step in horizon]
                      for field in ["lambda_median", "window_rms"]]
            values.append([indexed[row["key"], step, 1e-12]["refit_validation"]["rms"] for step in horizon])
            for ax, series in zip(axes, values):
                ax.loglog(horizon, series, marker="o", linestyle="-" if row["seed"] == 0 else "--",
                          color=color, label=label if row["seed"] == 0 else None)
    axes[0].axhline(.25, color="black", linestyle=":", linewidth=1)
    for ax, title in zip(axes, ["Median |lambda|", "Training RMS, preceding 20k steps", "Detached validation refit RMS"]):
        ax.set(xlabel="Training updates", title=title)
        ax.set_xticks(horizon, labels=["20k", "80k", "160k", "320k"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Shared-rate Adam, fixed reference-envelope initialization; seed 0 solid, seed 1 dashed")
    fig.savefig(output / "figures" / "shared_rate_trajectories.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"included": len(rows), "diagnostic_tasks": len(tasks), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
