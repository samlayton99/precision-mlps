"""Hidden-weight row norms and matched-row rotations from saved D38 runs."""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precision_d38_matplotlib")
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.expD38_init_readout_baseline.run import (
    OUT, TITLES, config, load_data, make_model, mse, predictions,
    same_identity, save_json, select, valid_data_identity,
)


def collect(task, seed):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    cfg = config()
    recipe = select(cfg)[task]
    steps = cfg["comparison_steps"]
    arrays_out, summaries, sources = {}, {}, {}
    for scheme in ("standard", "qi"):
        path = OUT / "data/compare" / f"{task}_{scheme}_w{recipe['width']}_lr{recipe['lr']:g}_seed{seed}_steps{steps}.json"
        run = json.loads(path.read_text())
        identity = run["identity"]
        assert run["complete"] and valid_data_identity(identity, cfg)
        checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        assert same_identity(identity, checkpoint["identity"])
        arrays, metadata = load_data(task, identity["config"])
        assert metadata == run["data"], "Saved run's preprocessing/data must be reproduced exactly"
        inputs = {split: torch.from_numpy(x.copy()) for split, (x, _) in arrays.items()}
        initial, _ = make_model(metadata["d_in"], identity["width"], seed, scheme,
                                inputs["train"], identity["config"])
        initial_prediction, _ = predictions(initial, inputs)
        initial_errors = {split: mse(initial_prediction[split], y) for split, (_, y) in arrays.items()}
        for split, value in initial_errors.items():
            np.testing.assert_allclose(value, run["trace"][0]["learned"][split], rtol=1e-10, atol=1e-12)
        summaries[scheme] = {}
        sources[scheme] = {"checkpoint": str(path.with_suffix(".pt").relative_to(OUT)),
                           "identity": identity, "reconstructed_initial_mse": initial_errors}
        for layer in ("fc1", "fc2"):
            before = getattr(initial, layer).weight.detach().cpu().numpy().copy()
            after = checkpoint["model"][f"{layer}.weight"].numpy().copy()
            assert before.shape == after.shape
            gamma_initial = np.linalg.norm(before, axis=1)
            gamma_final = np.linalg.norm(after, axis=1)
            valid = (gamma_initial > 0) & (gamma_final > 0)
            cosine = np.full(len(before), np.nan)
            cosine[valid] = np.clip(np.einsum("ij,ij->i", before[valid], after[valid]) /
                                     (gamma_initial[valid] * gamma_final[valid]), -1., 1.)
            ratio = gamma_final[valid] / gamma_initial[valid]
            for name, values in (("initial", gamma_initial), ("final", gamma_final), ("cosine", cosine)):
                arrays_out[f"{scheme}_{layer}_{name}"] = values
            summaries[scheme][layer] = {
                "rows": len(before), "undefined_cosines_zero_norm": int((~valid).sum()),
                "initial_gamma_median": float(np.median(gamma_initial)),
                "final_gamma_median": float(np.median(gamma_final)),
                "paired_gamma_ratio_median": float(np.median(ratio)),
                "cosine_min": float(np.nanmin(cosine)), "cosine_median": float(np.nanmedian(cosine)),
                "cosine_max": float(np.nanmax(cosine)),
                "rotation_degrees_median": float(np.nanmedian(np.degrees(np.arccos(cosine)))),
            }
    return arrays_out, {"task": task, "seed": seed, "steps": steps, "width": recipe["width"],
                        "gamma_definition": "L2 norm of each hidden weight-matrix row, excluding bias",
                        "cosine_definition": "Signed cosine between the same indexed row at initialization and finish",
                        "sources": sources, "summary": summaries}


def plot(arrays, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    colors = {"standard": "#1f77b4", "qi": "#e58320"}
    labels = {"standard": "Standard initialization", "qi": "QI initialization"}
    handles = [Patch(facecolor=colors[s], edgecolor=colors[s], alpha=.45, label=labels[s]) for s in colors]
    task, seed, steps = audit["task"], audit["seed"], audit["steps"]
    context = f"{TITLES[task]} · seed {seed} · width {audit['width']} + {audit['width']}"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    gamma_max = max(v.max() for k, v in arrays.items() if not k.endswith("cosine"))
    upper = np.ceil(gamma_max * 2) / 2
    gamma_bins = np.linspace(0, upper, 51)
    gamma_peak = max(np.histogram(v, gamma_bins)[0].max() * 100 / len(v)
                     for k, v in arrays.items() if not k.endswith("cosine"))
    gamma_ymax = np.ceil(gamma_peak * 1.08 / 5) * 5
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row, layer in enumerate(("fc1", "fc2")):
        for col, stage in enumerate(("initial", "final")):
            ax = axes[row, col]
            for scheme in colors:
                values = arrays[f"{scheme}_{layer}_{stage}"]
                ax.hist(values, bins=gamma_bins, weights=np.full(len(values), 100 / len(values)),
                        color=colors[scheme], edgecolor=colors[scheme], alpha=.45, linewidth=.6)
            when = "Initialization · step 0" if stage == "initial" else f"Finish · step {steps:,}"
            ax.set_title(f"Hidden layer {row+1} · {when}")
            ax.set_xlabel(r"Row norm $\gamma_i = \|W_{i,:}\|_2$")
            ax.set_ylabel("Rows in bin (%)")
            ax.set_xlim(0, upper)
            ax.set_ylim(0, gamma_ymax)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"Hidden-layer gamma distributions · {context}", y=.99, fontsize=14)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=2, frameon=False)
    fig.text(.5,.015,f"{audit['width']} rows per histogram. Identical bins and axis limits in all four panels.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.87), h_pad=2)
    fig.savefig(OUT / f"figures/{task}_gamma_histograms_seed{seed}.png", dpi=180)
    plt.close(fig)

    cosine_bins = np.linspace(-1, 1, 81)
    cosine_values = [v[np.isfinite(v)] for k, v in arrays.items() if k.endswith("cosine")]
    cosine_peak = max(np.histogram(v, cosine_bins)[0].max() * 100 / len(v) for v in cosine_values)
    cosine_ymax = np.ceil(cosine_peak * 1.08 / 5) * 5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for col, layer in enumerate(("fc1", "fc2")):
        ax = axes[col]
        for scheme in colors:
            values = arrays[f"{scheme}_{layer}_cosine"]
            values = values[np.isfinite(values)]
            assert len(values), "Cosines require nonzero initial and final rows"
            ax.hist(values, bins=cosine_bins, weights=np.full(len(values), 100 / len(values)),
                    color=colors[scheme], edgecolor=colors[scheme], alpha=.45, linewidth=.6)
        ax.set_title(f"Hidden layer {col+1} · step 0 → {steps:,}")
        ax.set_xlabel("Cosine similarity to the same initial row")
        ax.set_ylabel("Rows in bin (%)")
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_ylim(0, cosine_ymax)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(labelleft=True)
        ax.grid(axis="y", alpha=.15)
    undefined = sum(v["undefined_cosines_zero_norm"] for arm in audit["summary"].values() for v in arm.values())
    note = "1 = same direction; 0 = orthogonal; −1 = reversed. Identical bins and axes; rows matched by index."
    if undefined:
        note += f" {undefined} zero-norm pairs omitted."
    fig.suptitle(f"Hidden-row direction retention · {context}", y=.995, fontsize=14)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.92), ncol=2, frameon=False)
    fig.text(.5,.015,note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.81))
    fig.savefig(OUT / f"figures/{task}_direction_cosines_seed{seed}.png", dpi=180)
    plt.close(fig)
    audit["histogram_bins"] = {"gamma": gamma_bins.tolist(), "cosine": cosine_bins.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TITLES), default="airfoil")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    arrays, audit = collect(args.task, args.seed)
    plot(arrays, audit)
    folder = OUT / "data/weight_geometry"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_seed{args.seed}_steps{audit['steps']}"
    np.savez_compressed(folder / f"{stem}.npz", **arrays)
    save_json(folder / f"{stem}.json", audit)
    print(json.dumps(audit["summary"], indent=2))


if __name__ == "__main__":
    main()
