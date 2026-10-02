"""Check Airfoil's extreme LS predictions without changing any training run."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precision_d38_matplotlib")

import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.expD38_init_readout_baseline.run import (
    OUT, config, load_data, make_model, mse, predictions, save_json,
)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    cfg = config()
    arrays, metadata = load_data("airfoil", cfg)
    inputs = {k: torch.from_numpy(x.copy()) for k, (x, _) in arrays.items()}
    targets = {k: y for k, (_, y) in arrays.items()}
    output = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for col, scheme in enumerate(("standard", "qi")):
        file = next((OUT / "data/compare").glob(f"airfoil_{scheme}_*seed0_steps20000.json"))
        recorded = json.loads(file.read_text())
        for row, step in enumerate((0, 20000)):
            model, _ = make_model(metadata["d_in"], 512, 0, scheme, inputs["train"], cfg)
            if step:
                checkpoint = torch.load(file.with_suffix(".pt"), weights_only=True)
                model.load_state_dict(checkpoint["model"])
            _, h = predictions(model, inputs, with_features=True)
            mean, ym = h["train"].mean(0), targets["train"].mean(0)
            a, b = h["train"] - mean, targets["train"] - ym
            u, s, vt = linalg.svd(a, full_matrices=False, lapack_driver="gesvd")
            rows = []
            for cutoff in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]:
                keep = s > s[0] * cutoff
                coef = vt[keep].T @ ((u[:, keep].T @ b) / s[keep, None])
                scores = {k: mse((v - mean) @ coef + ym, targets[k]) for k, v in h.items()}
                rows.append({"rcond": cutoff, "rank": int(keep.sum()), "coef_norm": float(np.linalg.norm(coef)), **scores})
            nominal = next(r for r in rows if r["rcond"] == 1e-12)
            original = next(q for q in recorded["trace"] if q["step"] == step)
            # Independent classical SVD must reproduce the divide-and-conquer
            # solution's errors, including the large held-out predictions.
            relative = {k: abs(nominal[k] / original["solved"][k] - 1) for k in inputs}
            assert max(relative.values()) < 0.01, relative
            output.append({"scheme": scheme, "step": step, "cutoffs": rows,
                           "relative_mse_difference_between_svd_drivers": relative})
            ax = axes[row, col]
            ax.loglog([r["rcond"] for r in rows], [r["train"] for r in rows], color="#1f77b4", marker="o", label="Training MSE")
            ax.loglog([r["rcond"] for r in rows], [r["test"] for r in rows], color="#e58320", marker="o", label="Test MSE")
            ax.axvline(1e-12, color="#777777", ls=":", label="Recorded cutoff")
            ax.set_title(f"{'QI' if scheme == 'qi' else 'Standard'} initialization · step {step:,}")
            ax.set_xlabel("Relative singular-value cutoff")
            ax.set_ylabel("MSE (standardized target)")
            ax.grid(alpha=.15, which="major")
    errors = [r[split] for item in output for r in item["cutoffs"] for split in ("train", "test")]
    y_limits = (10. ** np.floor(np.log10(min(errors))), 10. ** np.ceil(np.log10(max(errors))))
    for ax in axes.flat:
        ax.set_xlim(10. ** -14.5, 10. ** -3.5)
        ax.set_ylim(*y_limits)
        ax.set_xticks(10. ** np.arange(-14, -3, 2))
        ax.set_yticks(10. ** np.arange(np.log10(y_limits[0]), np.log10(y_limits[1]) + 1, 2))
        ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.955), ncol=3, frameon=False)
    fig.suptitle("Airfoil readout sensitivity · fixed saved features · no further training", y=.995)
    fig.text(.5, .01, "Identical logarithmic axis limits and ticks in all four panels.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.90), h_pad=2)
    fig.savefig(OUT / "figures/airfoil_readout_sensitivity.png", dpi=180)
    plt.close(fig)
    save_json(OUT / "readout_audit.json", output)
    print(json.dumps([{k: v for k, v in r.items() if k != "cutoffs"} for r in output], indent=2))


if __name__ == "__main__":
    main()
