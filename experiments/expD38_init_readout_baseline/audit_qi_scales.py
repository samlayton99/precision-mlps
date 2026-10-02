"""Explain initial QI row-scale variation without modifying the initializer."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precision_d38_matplotlib")
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.expD38_init_readout_baseline.run import OUT, load_data, make_model, save_json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    path = OUT / "data/compare/airfoil_qi_w512_lr0.001_seed0_steps20000.json"
    saved = json.loads(path.read_text())
    cfg = saved["identity"]["config"]
    arrays, metadata = load_data("airfoil", cfg)
    assert metadata == saved["data"]
    x = torch.from_numpy(arrays["train"][0])
    model, _ = make_model(metadata["d_in"], 512, 0, "qi", x, cfg)
    result = {}
    for layer, inputs in (("fc1", x), ("fc2", model.hidden1(x).detach())):
        weight = getattr(model, layer).weight.detach()
        bias = getattr(model, layer).bias.detach()
        rows = []
        for start in range(0, 512, 22):
            stop = min(start + 22, 512)
            size = stop - start
            gamma = weight[start].norm().item()
            direction = weight[start] / gamma
            half_range = (inputs @ direction).abs().quantile(.999).item()
            norms = weight[start:stop].norm(dim=1)
            centers = -bias[start:stop] / gamma
            expected = cfg["qi_lambda"] * size / (2 * half_range)
            np.testing.assert_allclose(gamma, expected, rtol=1e-13)
            np.testing.assert_allclose(norms.numpy(), gamma, rtol=1e-13)
            rows.append({"first_row": start, "rows": size, "projection_half_range": half_range,
                         "gamma": gamma, "formula_gamma": expected,
                         "within_bundle_gamma_spread": float(norms.max() - norms.min()),
                         "nominal_h": 2 * half_range / size,
                         "actual_center_spacing": float(torch.diff(centers).mean()),
                         "gamma_times_actual_spacing": float(gamma * torch.diff(centers).mean())})
        result[layer] = rows
    save_json(OUT / "qi_scale_audit.json", result)

    first = result["fc1"]
    full, remainder = first[:-1], first[-1]
    orange, red, gray = "#e58320", "#b54b35", "#666666"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    values = model.fc1.weight.detach().norm(dim=1).numpy()
    axes[0].step(np.arange(506), values[:506], where="mid", color=orange, lw=1.6)
    axes[0].plot(np.arange(506, 512), values[506:], color=red, marker="x", ms=4)
    axes[0].set_xlabel("Hidden-neuron row index")
    axes[0].set_xlim(0, 512)
    axes[0].set_xticks([0, 128, 256, 384, 512])
    axes[0].set_title("Constant within each direction bundle")
    axes[1].scatter([r["projection_half_range"] for r in full], [r["gamma"] for r in full], color=orange, s=36, zorder=3)
    axes[1].scatter([remainder["projection_half_range"]], [remainder["gamma"]], color=red, marker="x", s=70, zorder=3)
    span = np.linspace(2, 5.6, 200)
    axes[1].plot(span, .25 * 22 / (2 * span), color=gray, ls="--", lw=1.3)
    axes[1].set_xlabel(r"Projection half-range $A_m$ (99.9th percentile of $|x\cdot u_m|$)")
    axes[1].set_title("Different projection ranges give different scales")
    for ax in axes:
        ax.set_ylim(0, 1.5)
        ax.set_ylabel(r"Initial row norm $\gamma$")
        ax.tick_params(labelleft=True)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=orange, marker="o", ls="", label="23 full bundles · 22 neurons each"),
               Line2D([], [], color=red, marker="x", ls="", label="Remainder bundle · 6 neurons"),
               Line2D([], [], color=gray, ls="--", label=r"22-neuron formula: $\gamma=0.25\times22/(2A)$")]
    fig.suptitle("Airfoil first-layer QI scale audit · seed 0 · initialization", y=.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=3, frameon=False, fontsize=9)
    fig.text(.5,.015,"The initializer adapts gamma to each direction's projected range and its bundle size.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.82))
    fig.savefig(OUT / "figures/airfoil_qi_scale_audit.png", dpi=180)
    plt.close(fig)
    print("Verified identical row norms within every bundle and the per-direction gamma formula in both layers.")


if __name__ == "__main__":
    main()
