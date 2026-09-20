"""Smaller outside-mu sweep; unchanged expD31 trainer, explicit refit errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD31_split_adam import run as prior

profile = prior.profile
RESULTS = prior.RESULTS / "mu_refit_sweep"


def config():
    cfg = prior.config()
    cfg.update(arms=["xavier"], mu_values=[100, 500, 1000, 2000], diagnostic_snapshots=501)
    return cfg


def refit(state, target, cfg, *, cutoff=None, n_eval=None):
    """Fit only training samples, then reconstruct predictions on a separate grid."""
    x = profile.previous.midpoint_grid(cfg["n_train"])
    xe = profile.previous.midpoint_grid(n_eval or cfg["n_eval"])
    y = profile.previous.matched.target_values(target, x, cfg)
    ye = profile.previous.matched.target_values(target, xe, cfg)
    h = profile.hidden(state["a"], state["b"], x)
    he = profile.hidden(state["a"], state["b"], xe)
    A = np.c_[h, np.ones(len(x))] / np.sqrt(len(x))
    yn = y / np.sqrt(len(x))
    U, s, Vh = sla.svd(A, full_matrices=False, check_finite=False, lapack_driver="gesvd")
    rank = int(np.sum(s > (cfg["readout_rcond"] if cutoff is None else cutoff) * s[0]))
    alpha = U[:, :rank].T @ yn
    vstar = Vh[:rank].T @ (alpha / s[:rank])
    prediction = he @ vstar[:-1] + vstar[-1]
    actual = he @ state["v"][:-1] + state["v"][-1]
    return dict(refit_relative_l2=float(np.linalg.norm(prediction-ye)/np.linalg.norm(ye)),
                refit_train_relative_l2=float(np.linalg.norm(A@vstar-yn)/np.linalg.norm(yn)),
                projection_relative_l2=float(np.linalg.norm(U[:, :rank]@alpha-yn)/np.linalg.norm(yn)),
                actual_relative_l2=float(np.linalg.norm(actual-ye)/np.linalg.norm(ye)),
                refit_linf=float(np.max(abs(prediction-ye))),
                coefficient_norm=float(np.linalg.norm(vstar)), rank=rank)


def evaluate(case, cfg):
    rows = []
    target, mu = str(case["target"]), int(case["mu"])
    for i, step in enumerate(case["saved_steps"]):
        state = {k: case[k][i] for k in ("a", "b", "v")}
        rows.append(refit(state, target, cfg))
        if int(step) in (0, 250, cfg["steps"]):
            print(f"refit {target}/mu={mu}, step={step}: relative L2={rows[-1]['refit_relative_l2']:.5g}", flush=True)
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]} | dict(steps=case["saved_steps"])


def summarize(cases, evaluations, cfg):
    records, checks, references = [], [], {}
    for target in cfg["targets"]:
        references[target] = refit(profile.initial_state("qi_zero", cfg), target, cfg)
        for mu in cfg["mu_values"]:
            c, e = cases[target, mu], evaluations[target, mu]
            best = int(np.argmin(e["refit_relative_l2"]))
            audits = json.loads(str(c["audits_json"]))
            row = dict(target=target, mu=mu, status=str(c["status"]), best_step=int(e["steps"][best]),
                       best_refit_relative_l2=float(e["refit_relative_l2"][best]),
                       final_refit_relative_l2=float(e["refit_relative_l2"][-1]),
                       final_actual_relative_l2=float(e["actual_relative_l2"][-1]),
                       best_gamma=float(c["mean_gamma"][e["steps"][best]]),
                       final_gamma=float(c["mean_gamma"][-1]),
                       best_coefficient_norm=float(e["coefficient_norm"][best]),
                       best_projection_relative_l2=float(e["projection_relative_l2"][best]),
                       initial_step_variation=audits[0]["relative_step_variation"],
                       maximum_step_variation=max(a["relative_step_variation"] for a in audits))
            records.append(row)
            state = {k: c[k][best] for k in ("a", "b", "v")}
            for cutoff in (1e-12, 1e-13, 1e-14):
                checks.append(dict(target=target, mu=mu, step=row["best_step"], cutoff=cutoff,
                                   n_eval=cfg["n_eval"], **refit(state, target, cfg, cutoff=cutoff)))
            checks.append(dict(target=target, mu=mu, step=row["best_step"], cutoff=cfg["readout_rcond"],
                               n_eval=8*cfg["n_eval"], **refit(state, target, cfg, n_eval=8*cfg["n_eval"])))
    summary = dict(runs=records, best_state_checks=checks, qi_reference=references)
    (RESULTS / "data/summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def plot(cases, evaluations, summary, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator

    colors = dict(zip(cfg["mu_values"], plt.cm.viridis(np.linspace(.04, .93, 4))))
    figdir = RESULTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for early in (False, True):
        fig, axes = plt.subplots(3, 4, figsize=(20, 12), dpi=170, sharex=True)
        limit = 30 if early else cfg["steps"]
        for col, target in enumerate(cfg["targets"]):
            axes[0, col].set_title(profile.LABELS[target], fontsize=16, pad=14)
            gammas, losses = [], []
            for mu in cfg["mu_values"]:
                c, e = cases[target, mu], evaluations[target, mu]
                keep, keepe = c["step"] <= limit, e["steps"] <= limit
                axes[0, col].plot(c["step"][keep], c["L"][keep], color=colors[mu], lw=1.7)
                axes[1, col].plot(e["steps"][keepe], e["refit_relative_l2"][keepe], color=colors[mu], lw=1.7,
                                  marker="." if early else None, ms=3.5)
                axes[2, col].plot(c["step"][keep], c["mean_gamma"][keep], color=colors[mu], lw=1.7)
                gammas.extend(c["mean_gamma"][keep]); losses.extend(c["L"][keep])
            control = prior.gd.load(prior.RESULTS / "data" / f"xavier__{target}__0.npz", prior.config())
            keep = control["step"] <= limit
            axes[0, col].plot(control["step"][keep], control["L"][keep], color=".15", ls=":", lw=1.8)
            axes[2, col].plot(control["step"][keep], control["mean_gamma"][keep], color=".15", ls=":", lw=1.8)
            # Control refits are measured on saved states, including step zero.
            ce = evaluations[target, 0]; keep = ce["steps"] <= limit
            axes[1, col].plot(ce["steps"][keep], ce["refit_relative_l2"][keep], color=".15", ls=":", lw=1.8)
            axes[1, col].axhline(summary["qi_reference"][target]["refit_relative_l2"], color=".65", ls="--", lw=1.2)
            for row in (0, 1):
                axes[row, col].set_yscale("log")
                axes[row, col].yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axes[0, col].set_ylim(min(losses)*.6, max(losses)*1.6)
            axes[1, col].set_ylim(1e-16, 2)
            axes[1, col].set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
            pad = max(max(gammas)-min(gammas), .01)*.08
            axes[2, col].set_ylim(max(0, min(gammas)-pad), max(gammas)+pad)
            axes[2, col].yaxis.set_major_locator(MaxNLocator(nbins=5))
            axes[2, col].ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[2, col].set_xlabel("Training updates", fontsize=12)
            for ax in axes[:, col]:
                ax.set_xlim(0, limit)
                ax.set_xticks(np.arange(0, limit+1, 5 if early else 100))
                ax.grid(alpha=.17)
                ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel(r"Actual loss $L=\frac{1}{2}\,\mathrm{mean}(e^2)$"+"\n(log scale)", fontsize=13)
        axes[1, 0].set_ylabel(r"Refitted relative $L_2$: $\|f_{\theta,v_*}-y\|_2/\|y\|_2$"+"\n(independent grid; log scale)", fontsize=12)
        axes[2, 0].set_ylabel(r"Mean scale $\overline{\gamma}=\mathrm{mean}|a_k|$"+"\n(linear scale)", fontsize=13)
        title = "Xavier · original VarPro split Adam · smaller outside multipliers"
        fig.suptitle(title+("\nFirst 30 updates" if early else "\n500 updates"), fontsize=19, y=.985)
        handles = [Line2D([], [], color=colors[mu], lw=2.2, label=f"μ = {mu:,}") for mu in cfg["mu_values"]]
        handles += [Line2D([], [], color=".15", ls=":", lw=1.8, label="Ordinary Adam"),
                    Line2D([], [], color=".65", ls="--", label="QI geometry + refit (middle row)")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=6, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.085, right=.98, top=.815, bottom=.145, hspace=.25, wspace=.30)
        fig.text(.5, .064,
                 "Geometry update: −η[μ Adam(DFτ) + Adam(DGτ)], with independent moment histories. Readout uses ordinary Adam.\n"
                 "Same expD31 rate η=0.002, β=(0.9,0.999), εAdam=10⁻⁸, seed 0, N=128, 24 halo neurons per side.\n"
                 "Middle row: fit coefficients on 1,024 training samples; evaluate predictions on 8,192 separate samples in [−1,1].\n"
                 "Solved coefficients never replace the trained readout. Split-run refit evaluation at every step; SVD cutoff: 10⁻¹³σ₁.",
                 ha="center", va="center", fontsize=10)
        fig.savefig(figdir / ("xavier_early.png" if early else "xavier.png"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    data = RESULTS / "data"
    data.mkdir(parents=True, exist_ok=True)
    cases, evaluations = {}, {}
    torch.set_num_threads(cfg["threads"])
    with threadpool_limits(cfg["threads"]):
        for target in cfg["targets"]:
            for mu in cfg["mu_values"]:
                path = data / f"{target}__{mu}.npz"
                if path.exists():
                    c = prior.gd.load(path, cfg)
                elif args.plot_only:
                    raise FileNotFoundError(path)
                else:
                    start = time.perf_counter()
                    c = prior.train(target, "xavier", mu, cfg)
                    c["seconds"] = np.array(time.perf_counter()-start)
                    prior.gd.save(c, cfg, path)
                assert str(c["status"]) == "complete"
                np.testing.assert_array_equal(c["saved_steps"], np.arange(cfg["steps"]+1))
                initial = profile.initial_state("xavier", cfg)
                for key in ("a", "b", "v"):
                    np.testing.assert_array_equal(c[key][0], initial[key])
                # Saving every state must not change the previous mu=1000 run.
                if mu == 1000:
                    old = prior.gd.load(prior.RESULTS/"data"/f"xavier__{target}__1000.npz", prior.config())
                    for key in ("L", "F", "mean_gamma"):
                        np.testing.assert_array_equal(c[key], old[key])
                ep = data / f"{target}__{mu}__refits.npz"
                if ep.exists():
                    with np.load(ep) as f:
                        e = {k: f[k] for k in f.files}
                else:
                    e = evaluate(c, cfg)
                    np.savez_compressed(ep, **e)
                cases[target, mu], evaluations[target, mu] = c, e
            ep = data / f"{target}__ordinary_refits.npz"
            if ep.exists():
                with np.load(ep) as f:
                    evaluations[target, 0] = {k: f[k] for k in f.files}
            else:
                old = prior.gd.load(prior.RESULTS/"data"/f"xavier__{target}__0.npz", prior.config())
                e = evaluate(old, cfg)
                np.savez_compressed(ep, **e)
                evaluations[target, 0] = e
        summary_path = data / "summary.json"
        summary = json.loads(summary_path.read_text()) if args.plot_only else summarize(cases, evaluations, cfg)
        plot(cases, evaluations, summary, cfg)
    for row in summary["runs"]:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
