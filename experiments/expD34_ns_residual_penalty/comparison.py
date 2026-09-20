"""Add the refined NS loss to the saved, matched cosine J/J* comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD34_ns_residual_penalty import run as exp
from experiments.expD33_current_readout_split import compare_jstar as old
from experiments.expD31_split_adam import long_run as prior_long

OUTPUT = exp.RESULTS / "j_comparison"
MUS = [100, 250, 500, 1000]
LABELS = old.LABELS | {"NS": "Newton–Schulz residual penalty"}
COLORS = old.COLORS | {"NS": "#d48806"}
STYLES = {"Jstar": "-", "J": "--", "NS": "-"}


def config(target):
    return prior_long.config("cosine") | dict(target=target, arm="xavier", mu_values=MUS,
                                             ns_steps=5, refinement_steps=4,
                                             diagnostic_snapshots=201)


def train_target(target):
    cfg = config(target)
    torch.set_num_threads(cfg["threads"])
    directory = OUTPUT / "data"
    directory.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(cfg["threads"]):
        for mu in MUS:
            path = directory / f"{target}__{mu}.npz"
            if path.exists():
                with np.load(path) as saved:
                    assert saved["step"][-1] == cfg["steps"]
                    assert json.loads(str(saved["config_json"])) == cfg
                continue
            case = exp.train(cfg, mu, cfg["refinement_steps"])
            exp.evaluate(case, cfg)
            case["config_json"] = np.array(json.dumps(cfg))
            np.savez_compressed(path, **case)
    print(f"Complete: {target}, four multipliers, {cfg['steps']} steps each", flush=True)


def load_ns(target, mu):
    path = OUTPUT / "data" / f"{target}__{mu}.npz"
    with np.load(path) as archive:
        c = dict(archive)
    cfg = json.loads(str(c["config_json"]))
    assert cfg == config(target)
    assert c["step"][-1] == 10000
    np.testing.assert_allclose(c["learning_rate"],
                              [.002*(.001+.999*.5*(1+np.cos(np.pi*s/10000))) for s in c["step"]],
                              rtol=1e-14, atol=1e-16)
    return dict(step=c["step"], actual=c["relative_l2"], refit=c["eval_refit_train_relative_l2"],
                refit_step=c["eval_step"], gamma=c["mean_gamma"], cfg=cfg,
                initial={k: c[k][0] for k in ("a", "b", "v")}, source=str(path), raw=c)


def plot():
    targets = exp.previous.config()["targets"]
    cases, sources = {}, []
    settings = ("resolution", "halo", "seed", "learning_rate", "adam_betas", "adam_epsilon",
                "n_train", "n_eval", "domain", "envelope_sigma", "target_modes",
                "target_amplitudes", "readout_rcond", "lr_schedule", "steps")
    for target in targets:
        for mu in MUS:
            triple = {m: old.load(target, mu, m, 10000) for m in old.LABELS}
            triple["NS"] = load_ns(target, mu)
            reference = triple["Jstar"]
            for method, case in triple.items():
                for key in settings:
                    assert case["cfg"][key] == reference["cfg"][key], (target, mu, method, key)
                for key in ("a", "b", "v"):
                    np.testing.assert_array_equal(case["initial"][key], reference["initial"][key])
                np.testing.assert_array_equal(case["step"], reference["step"])
                cases[target, mu, method] = case
                sources.append(dict(target=target, mu=mu, method=method, source=case["source"]))
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data/sources.json").write_text(json.dumps(sources, indent=2)+"\n")
    summary = []
    for target in targets:
        fig, axes = plt.subplots(3, len(MUS), figsize=(20, 11), dpi=180,
                                 sharex=True, sharey="row", squeeze=False)
        relevant = [cases[target, mu, method] for mu in MUS for method in LABELS]
        top_lo = min(float(c["actual"].min()) for c in relevant)*.65
        top_hi = max(float(c["actual"].max()) for c in relevant)*1.3
        mid_hi = max(2., max(float(c["refit"].max()) for c in relevant)*1.3)
        gamma_hi = max(float(c["gamma"].max()) for c in relevant)*1.08
        for col, mu in enumerate(MUS):
            axes[0, col].set_title(rf"$\mu={mu}$", fontsize=17, pad=13)
            for method in LABELS:
                c = cases[target, mu, method]
                for row, key in enumerate(("actual", "refit", "gamma")):
                    x = c.get("refit_step", c["step"]) if row == 1 else c["step"]
                    axes[row, col].plot(x, c[key], color=COLORS[method], ls=STYLES[method], lw=1.4)
                summary.append(dict(target=target, mu=mu, method=method,
                                    final_relative_l2=float(c["actual"][-1]),
                                    final_refit_relative_l2=float(c["refit"][-1]),
                                    final_gamma=float(c["gamma"][-1])))
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
                ax.set_xlim(0, 10000)
                ax.set_xticks(np.linspace(0, 10000, 6))
                ax.tick_params(labelsize=9, labelleft=True)
                ax.grid(alpha=.17)
                ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel(r"Trained relative $L_2$ error"+"\n(log scale)", fontsize=13)
        axes[1, 0].set_ylabel(r"Refitted relative $L_2$ error"+"\n(log scale)", fontsize=13)
        axes[2, 0].set_ylabel(r"Mean $\gamma=\mathrm{mean}|a_k|$"+"\n(linear scale)", fontsize=13)
        fig.suptitle(old.current.profile.LABELS[target] + r" · $J_*$, $J$, and Newton–Schulz penalty"
                     + "\nMatched Xavier initialization · 10,000 steps · cosine decay: 0.002 → 0.000002", fontsize=19, y=.985)
        handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=2, label=LABELS[m]) for m in LABELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .9), ncol=3, frameon=False, fontsize=13)
        fig.subplots_adjust(left=.085, right=.985, top=.795, bottom=.15, hspace=.28, wspace=.26)
        fig.text(.5, .06,
                 "Same initial weights, samples, duration, cosine schedule, and Adam settings. Axes match within each row.\n"
                 "J/J*: μ multiplies one stream AFTER its own Adam normalization. NS: one Adam differentiates L + μ F̂.\n"
                 "Both error rows use 1,024 training points. Middle: explicit least-squares evaluation; NS refits at saved states.\n"
                 "NS uses five Muon iterations plus four refinement steps. Training uses no coefficient solve; evaluation solves never affect it.",
                 ha="center", va="center", fontsize=10, linespacing=1.5)
        path = OUTPUT / "figures" / f"{target}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)
    (OUTPUT / "data/summary.json").write_text(json.dumps(summary, indent=2)+"\n")


def validate_endpoints():
    records = []
    with threadpool_limits(2):
        for target in exp.previous.config()["targets"]:
            for mu in MUS:
                case = load_ns(target, mu)
                c, cfg = case["raw"], case["cfg"]
                x = exp.previous.profile.previous.midpoint_grid(cfg["n_train"])
                y = exp.previous.profile.previous.matched.target_values(target, x, cfg)
                for i, step in ((0, 0), (-1, 10000)):
                    p = {k: torch.tensor(c[k][i]) for k in ("a", "b", "v")}
                    prediction = torch.tanh(torch.tensor(x)[:, None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
                    relative = np.linalg.norm(prediction.numpy()-y)/np.linalg.norm(y)
                    np.testing.assert_allclose(relative, c["relative_l2"][step], rtol=1e-12, atol=1e-15)
                state = {k: c[k][-1] for k in ("a", "b", "v")}
                for cutoff in (1e-12, 1e-13, 1e-14):
                    result = exp.refit(state, target, cfg, cutoff=cutoff, n_eval=65536)
                    if cutoff == cfg["readout_rcond"]:
                        np.testing.assert_allclose(result["refit_train_relative_l2"], c["eval_refit_train_relative_l2"][-1], rtol=1e-12, atol=1e-15)
                    records.append(dict(target=target, mu=mu, cutoff=cutoff, **result))
    (OUTPUT / "data/endpoint_checks.json").write_text(json.dumps(records, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=exp.previous.config()["targets"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.target:
        train_target(args.target)
    if args.plot:
        torch.set_num_threads(2)
        validate_endpoints()
        plot()


if __name__ == "__main__":
    main()
