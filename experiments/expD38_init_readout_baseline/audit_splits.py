"""Split overlap and an untuned sensitivity check on previously unseen inputs."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precision_d38_matplotlib")
from pathlib import Path
import sys
import json
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.expD38_init_readout_baseline.run import (
    ROOT, OUT, TITLES, affine_fit, config, load_data, make_model, mse,
    predictions, recipe_config, save_json, select, valid_data_identity,
)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    cfg = config()
    choices = select(cfg)
    rows = []
    for task in cfg["tasks"]:
        arrays, metadata = load_data(task, cfg)
        fit_x, fit_y = arrays["train"]
        fit = {}
        for x, y in zip(fit_x, fit_y):
            fit.setdefault(tuple(x), set()).add(float(y[0]))
        seen = set(fit) | set(map(tuple, arrays["val"][0]))
        keep = np.array([tuple(x) not in seen for x in arrays["test"][0]])
        overlaps = {}
        for split in ("val", "test"):
            x, y = arrays[split]
            overlaps[split] = {
                "rows": len(x),
                "inputs_matching_fit": sum(tuple(v) in fit for v in x),
                "input_target_pairs_matching_fit": sum(float(t[0]) in fit.get(tuple(v), ()) for v, t in zip(x, y)),
            }
        assert keep.any()
        coef, bias, _ = affine_fit(fit_x, fit_y, cfg["least_squares_rcond"])
        test_x, test_y = arrays["test"]
        affine = mse(test_x[keep] @ coef + bias, test_y[keep])
        inputs = {"train": torch.from_numpy(fit_x), "test": torch.from_numpy(test_x[keep])}
        records = []
        for path in (OUT / "data/compare").glob(f"{task}_*.json"):
            r = json.loads(path.read_text())
            i = r["identity"]
            if not (r.get("complete") and valid_data_identity(i, cfg)
                    and recipe_config(i["config"]) == recipe_config(cfg)
                    and i["seed"] in cfg["comparison_seeds"] and i["steps"] == cfg["comparison_steps"]
                    and i["lr"] == choices[task]["lr"] and i["width"] == choices[task]["width"]):
                continue
            model, _ = make_model(metadata["d_in"], i["width"], i["seed"], i["scheme"], inputs["train"], cfg)
            model.load_state_dict(torch.load(path.with_suffix(".pt"), weights_only=True)["model"])
            pred, _ = predictions(model, {"test": inputs["test"]})
            records.append({"scheme": i["scheme"], "seed": i["seed"],
                            "test_mse": mse(pred["test"], test_y[keep])})
        expected = {(scheme, seed) for scheme in ("standard", "qi") for seed in cfg["comparison_seeds"]}
        assert {(r["scheme"], r["seed"]) for r in records} == expected, task
        rows.append({"task": task, "overlap": overlaps, "unseen_input_test_rows": int(keep.sum()),
                     "unseen_input_affine_mse": affine, "unseen_input_runs": records})
    source = loadmat(ROOT / "data/sarcos/sarcos_inv.mat")["sarcos_inv"]
    supplied = loadmat(ROOT / "data/sarcos/sarcos_inv_test.mat")["sarcos_inv_test"]
    distance, indices = cKDTree(source[:, :21]).query(supplied[:, :21])
    audit = {"scope": "Post-hoc evaluation only; no tuning or checkpoint selection on this subset.",
             "original_sarcos": {"supplied_test_rows": len(supplied),
                                 "inputs_matching_source_train": int((distance == 0).sum()),
                                 "full_rows_matching_source_train": int(np.all(source[indices] == supplied, axis=1).sum())},
             "tasks": rows}
    save_json(OUT / "split_audit.json", audit)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    positions = np.arange(len(rows))
    axes[0].bar(positions, [100*r["overlap"]["test"]["inputs_matching_fit"]/r["overlap"]["test"]["rows"] for r in rows], color="#999999")
    axes[0].set_ylabel("Test inputs also found in fitting split (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Exact input overlap · corrected splits")
    for offset, scheme, color in ((-.18, "standard", "#1f77b4"), (.18, "qi", "#e58320")):
        vals = [[q["test_mse"]/r["unseen_input_affine_mse"] for q in r["unseen_input_runs"] if q["scheme"] == scheme] for r in rows]
        avg = np.array([np.mean(v) for v in vals])
        axes[1].bar(positions+offset, avg, width=.34, color=color, label=scheme.capitalize())
        axes[1].errorbar(positions+offset, avg, yerr=np.array([avg-[min(v) for v in vals], [max(v) for v in vals]-avg]), fmt="none", color="#333333", capsize=3)
    axes[1].axhline(1, color="#666666", ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylim(.005, 2)
    axes[1].set_ylabel("Trained MLP MSE / linear-regression MSE")
    axes[1].set_title("Test inputs absent from fitting and validation")
    for ax in axes:
        ax.set_xticks(positions, [TITLES[r["task"]] for r in rows], rotation=25, ha="right")
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.94), ncol=2, frameon=False)
    fig.suptitle("Input-overlap sensitivity · width 512 + 512 · fixed final 20k models", y=.995)
    fig.text(.5,.01,"Right: mean and range of 3 seeds; same subset for both initializations and linear regression. No retuning.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.86))
    fig.savefig(OUT / "figures/split_overlap_controls.png", dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
