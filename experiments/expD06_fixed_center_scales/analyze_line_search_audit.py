"""Summarize matched numerical-audit windows without changing trained states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import run


def summarize(root, output):
    protocols = sorted(root.glob("*/protocol.json"))
    if len(protocols) != 7:
        raise ValueError("Expected the seven selected audit dictionaries")
    paths = [p.parent / method / spec[0] for p in protocols
             for method in ("direct_loss", "loss_change")
             for spec in json.loads(p.read_text())["specifications"]]
    horizon = min(json.loads((p / "latest.json").read_text())["step"] for p in paths)
    horizon = horizon // 20000 * 20000
    if horizon < 20000:
        raise ValueError("Every audit arm must complete at least 20k new updates")
    records, hashes = [], {}
    for protocol_path in protocols:
        protocol = json.loads(protocol_path.read_text())
        folder = protocol_path.parent
        for spec in protocol["specifications"]:
            record = {"dictionary": folder.name, "case": protocol["case"],
                      "solver": spec[0], "source_step": protocol["source_step"],
                      "additional_steps": horizon, "arms": {}}
            initial = []
            for method in ("direct_loss", "loss_change"):
                path = folder / method / spec[0]
                cp0_path = path / "checkpoint_000000000.npz"
                cp_path = path / f"checkpoint_{horizon:09d}.npz"
                with np.load(cp0_path) as cp0, np.load(cp_path) as cp:
                    initial.append({key: cp0[key].copy() for key in cp0.files})
                    motion = float(np.sqrt(np.mean((cp["c"] - cp0["c"])**2)))
                    moved = int(np.count_nonzero(cp["c"] != cp0["c"]))
                metric_path = path / f"metrics_{horizon:09d}.json"
                row = json.loads(metric_path.read_text())
                row.update(physical_net_displacement_rms=motion, changed_coefficients=moved)
                row["initial"] = json.loads((path / "metrics_000000000.json").read_text())
                windows = []
                for right in range(20000, horizon + 1, 20000):
                    losses = run.trace_window_losses(path, right - 20000, right)
                    if losses is None or len(losses) != 20000 or not np.isfinite(losses).all():
                        raise ValueError(f"Incomplete or nonfinite audit window: {path}")
                    windows.append({"end": right, "mse_mean": float(2 * np.mean(losses)),
                                    "mse_quantiles": np.quantile(2 * losses, [0, .5, .9, 1]).tolist()})
                row["windows"] = windows
                record["arms"][method] = row
                for source in (cp0_path, cp_path, metric_path):
                    hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
            assert initial[0].keys() == initial[1].keys()
            assert all(np.array_equal(initial[0][k], initial[1][k]) for k in initial[0])
            control, repaired = (record["arms"][m] for m in ("direct_loss", "loss_change"))
            assert control["gradient_evaluations"] == repaired["gradient_evaluations"] == horizon
            record["window_mse_ratio"] = repaired["windows"][-1]["mse_mean"] / control["windows"][-1]["mse_mean"]
            record["validation_mse_ratio"] = repaired["validation_mse"] / control["validation_mse"]
            records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    run.write_json(output / "evidence.json", {"common_additional_horizon": horizon,
        "source": str(root.resolve()), "paired_states_identical": True, "records": records,
        "source_sha256": hashes, "interpretation": "Combined numerical guards, not an isolated loss-evaluation effect"})
    selected = [("quadratic_N1024_uniform", "differences_adam_armijo"),
                ("sine_N1024_uniform", "differences_adam_armijo"),
                ("quadratic_N512_uniform", "differences_adam_armijo"),
                ("quadratic_N512_adam_both_envelope_s0_012a20b62876_learned", "differences_momentum_armijo")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (dictionary, label) in zip(axes.flat, selected):
        row = next(r for r in records if r["dictionary"] == dictionary and r["solver"] == label)
        for method, name in (("direct_loss", "Original evaluation"), ("loss_change", "Numerical guards")):
            data = row["arms"][method]
            x = [0] + [w["end"] / 1000 for w in data["windows"]]
            y = [data["initial"]["train_mse"]] + [w["mse_mean"] for w in data["windows"]]
            ax.plot(x, np.array(y) / data["initial"]["train_mse"], marker="o", label=name)
        kind = "learned" if dictionary.endswith("learned") else "uniform"
        ax.set(title=f"{row['case']['target']}, N={row['case']['n']}, {kind}; {label.split('_')[1]}",
               xlabel="Additional updates (thousands)", ylabel="Window MSE / initial MSE")
        ax.ticklabel_format(useOffset=False, axis="y")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Neighbor-difference coordinates: paired continuation of recorded numerical stalls\nInitial point is an endpoint; subsequent points are complete 20k-window means")
    fig.savefig(output / "numerical_audit.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"dictionaries": len(protocols), "paired_solvers": len(records),
                      "common_additional_horizon": horizon}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
