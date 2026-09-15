"""Matched J versus J* figures from cosine-scheduled Xavier trajectories."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD33_current_readout_split import run as current

BASE = ROOT / "results/checkpoint_D_optimizers"
OUTPUT = current.RESULTS / "j_vs_jstar"
LABELS = {"Jstar": r"$J_*$: VarPro split", "J": r"$J$: current-readout split"}
COLORS = {"Jstar": "#482878", "J": "#21918c"}


def load(target, mu, method, steps):
    if method == "J":
        paths = [OUTPUT / "data/trajectories" / f"J__{target}__{mu}.npz"]
    else:
        paths = [BASE / "expD31_split_adam/long_run/data" / f"{target}__{mu}__cosine.npz",
                 OUTPUT / "data/trajectories" / f"Jstar__{target}__{mu}.npz"]
    for path in paths:
        if not path.exists():
            continue
        with np.load(path) as d:
            c = {k: d[k] for k in d.files}
        if c["step"][-1] < steps:
            continue
        cfg = json.loads(str(c["config_json"]))
        assert cfg.get("lr_schedule") == dict(kind="cosine",start_step=0,min_factor=.001)
        assert cfg.get("target_update_ratio") is None
        assert int(c["mu"]) == mu
        x = current.profile.previous.midpoint_grid(cfg["n_train"])
        y = current.profile.previous.matched.target_values(target, x, cfg)
        keep = c["step"] <= steps
        if "refit_train_relative_l2" in c:
            refit = c["refit_train_relative_l2"][keep]
        else:
            with np.load(path.with_name(path.stem + "__refits.npz")) as e:
                np.testing.assert_array_equal(e["steps"], c["step"])
                refit = e["refit_train_relative_l2"][keep]
        return dict(step=c["step"][keep], actual=np.sqrt(2*c["L"][keep]/np.mean(y*y)),
                    refit=refit, gamma=c["mean_gamma"][keep], cfg=cfg,
                    initial={k: c[k][0] for k in ("a", "b", "v")}, source=str(path))
    raise FileNotFoundError(f"No saved {method} trajectory for {target}, mu={mu}, {steps} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mus", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    targets = current.config()["targets"]
    cases, sources = {}, []
    settings = ("resolution", "halo", "seed", "learning_rate", "adam_betas", "adam_epsilon",
                "n_train", "n_eval", "domain", "envelope_sigma", "target_modes",
                "target_amplitudes", "readout_rcond", "lr_schedule", "steps")
    for target in targets:
        for mu in args.mus:
            pair = {method: load(target, mu, method, args.steps) for method in LABELS}
            for k in settings:
                assert pair["J"]["cfg"][k] == pair["Jstar"]["cfg"][k], (target, mu, k)
            for k in ("a", "b", "v"):
                np.testing.assert_array_equal(pair["J"]["initial"][k], pair["Jstar"]["initial"][k])
            np.testing.assert_array_equal(pair["J"]["step"], pair["Jstar"]["step"])
            for method, c in pair.items():
                cases[target, mu, method] = c
                sources.append(dict(target=target, mu=mu, method=method, source=c["source"]))
    figdir = OUTPUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data").mkdir(exist_ok=True)
    (OUTPUT / "data/sources.json").write_text(json.dumps(dict(steps=args.steps, mus=args.mus, sources=sources), indent=2)+"\n")
    for target in targets:
        fig, axes = plt.subplots(3, len(args.mus), figsize=(5*len(args.mus), 11), dpi=180,
                                 sharex=True, sharey="row", squeeze=False)
        cs = [cases[target, mu, method] for mu in args.mus for method in LABELS]
        top_lo = min(float(c["actual"].min()) for c in cs)*.65
        top_hi = max(float(c["actual"].max()) for c in cs)*1.3
        mid_hi = max(2, max(float(c["refit"].max()) for c in cs)*1.3)
        gamma_hi = max(float(c["gamma"].max()) for c in cs)*1.08
        for col, mu in enumerate(args.mus):
            axes[0, col].set_title(rf"$\mu={mu}$", fontsize=17, pad=13)
            for method in LABELS:
                c = cases[target, mu, method]
                for row, key in enumerate(("actual", "refit", "gamma")):
                    axes[row, col].plot(c["step"], c[key], color=COLORS[method],
                                       ls="-" if method == "Jstar" else "--", lw=1.4)
            for row in (0, 1):
                axes[row, col].set_yscale("log")
            axes[0, col].set_ylim(top_lo, top_hi)
            axes[1, col].set_ylim(1e-16, mid_hi)
            axes[1, col].set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
            axes[2, col].set_ylim(0, gamma_hi)
            axes[2, col].yaxis.set_major_locator(MaxNLocator(nbins=5))
            axes[2, col].ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[2, col].set_xlabel("Training updates", fontsize=12)
            for ax in axes[:, col]:
                ax.set_xlim(0, args.steps)
                ax.set_xticks(np.linspace(0, args.steps, 6))
                ax.tick_params(labelsize=9, labelleft=True)
                ax.grid(alpha=.17)
                ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel(r"Trained relative $L_2$ error"+"\n(log scale)", fontsize=13)
        axes[1, 0].set_ylabel(r"Refitted relative $L_2$ error"+"\n(log scale)", fontsize=13)
        axes[2, 0].set_ylabel(r"Mean $\gamma=\mathrm{mean}|a_k|$"+"\n(linear scale)", fontsize=13)
        fig.suptitle(current.profile.LABELS[target] + r" · $J_*$ versus $J$ · matched Xavier initialization"
                     + f"\n{args.steps:,} steps · same cosine decay: 0.002 → 0.000002", fontsize=19, y=.985)
        handles = [Line2D([], [], color=COLORS[m], ls="-" if m == "Jstar" else "--", lw=2,
                          label=LABELS[m]) for m in LABELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .9), ncol=2,
                   frameon=False, fontsize=13)
        fig.subplots_adjust(left=.085, right=.985, top=.795, bottom=.15, hspace=.28, wspace=.26)
        fig.text(.5, .06,
                 "Same initial weights, samples, duration, cosine schedule, Adam settings, and outside μ in each column.\n"
                 "Both error rows use the same 1,024 training points: top retains trained coefficients; middle solves coefficients for evaluation only.\n"
                 r"Top: $\sqrt{2L/\mathrm{mean}(y^2)}$. Middle: reconstructed least-squares residual, not a projection-only loss conversion."
                 + "\nBoth variants retain independent Adam streams and ordinary-Adam readout updates. Saved trajectories only; axes match within each row.",
                 ha="center", va="center", fontsize=10, linespacing=1.5)
        path = figdir / f"{target}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
