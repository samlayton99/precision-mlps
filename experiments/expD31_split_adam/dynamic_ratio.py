"""10k Xavier sweep with a prescribed post-Adam F/G update-norm ratio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD31_split_adam import long_run as fixed

prior, refits = fixed.prior, fixed.refits
RESULTS = prior.RESULTS / "dynamic_ratio"
RATIOS = (.01, .1, 1., 10., 100.)
SCHEDULES = ("constant", "cosine")


def config(schedule, ratio):
    return fixed.config(schedule) | dict(mu_values=[1], ratio_values=list(RATIOS),
                                        target_update_ratio=float(ratio))


def case_path(target, ratio, schedule):
    return RESULTS / "data" / f"{target}__r{ratio:g}__{schedule}.npz"


def check_case(case, target, cfg):
    initial = prior.profile.initial_state("xavier", cfg)
    for k in ("a", "b", "v"):
        np.testing.assert_array_equal(case[k][0], initial[k])
    np.testing.assert_array_equal(case["step"], np.arange(len(case["step"])))
    np.testing.assert_allclose(case["F"] + case["G"], case["L"], rtol=1e-13, atol=1e-15)
    achieved = case["F_step_norm"][:-1] / case["G_step_norm"][:-1]
    np.testing.assert_allclose(achieved, cfg["target_update_ratio"], rtol=1e-12, atol=0)
    return float(np.max(np.abs(achieved/cfg["target_update_ratio"]-1))) if len(achieved) else 0.


def run_case(target, ratio, schedule):
    cfg = config(schedule, ratio)
    path = case_path(target, ratio, schedule)
    if path.exists():
        c = prior.gd.load(path, cfg)
    else:
        start = time.perf_counter()
        c = prior.train(target, "xavier", 1, cfg)
        c["seconds"] = np.array(time.perf_counter()-start)
        c["target_update_ratio"] = np.array(ratio)
        prior.gd.save(c, cfg, path)
    check_case(c, target, cfg)
    ep = path.with_name(path.stem + "__refits.npz")
    if not ep.exists():
        rows = [refits.refit({k: c[k][i] for k in ("a", "b", "v")}, target, cfg)
                for i in range(len(c["saved_steps"]))]
        e = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
        np.savez_compressed(ep, **e, steps=c["saved_steps"])
    print(f"COMPLETE {target}/{schedule}/r={ratio:g}: {c['status']}, steps={c['step'][-1]}", flush=True)


def load_cases():
    cases, evaluations, controls = {}, {}, {}
    for schedule in SCHEDULES:
        for target in fixed.config(schedule)["targets"]:
            cp = fixed.case_path(target, 0, schedule)
            c = prior.gd.load(cp, fixed.config(schedule))
            with np.load(cp.with_name(cp.stem + "__refits.npz")) as d:
                controls[target, schedule] = (c, {k: d[k] for k in d.files})
            for ratio in RATIOS:
                path = case_path(target, ratio, schedule)
                cases[target, ratio, schedule] = prior.gd.load(path, config(schedule, ratio))
                with np.load(path.with_name(path.stem + "__refits.npz")) as d:
                    evaluations[target, ratio, schedule] = {k: d[k] for k in d.files}
    return cases, evaluations, controls


def summarize(cases, evaluations):
    rows, checks = [], []
    for (target, ratio, schedule), c in cases.items():
        cfg = config(schedule, ratio)
        error = check_case(c, target, cfg)
        e = evaluations[target, ratio, schedule]
        best = int(np.argmin(e["refit_relative_l2"]))
        audits = json.loads(str(c["audits_json"]))
        rows.append(dict(target=target, ratio=ratio, schedule=schedule, status=str(c["status"]),
                         final_step=int(c["step"][-1]), maximum_relative_ratio_error=error,
                         effective_mu_min=float(c["effective_mu"][:-1].min()),
                         effective_mu_max=float(c["effective_mu"][:-1].max()),
                         effective_mu_final=float(c["effective_mu"][-1]),
                         final_actual_relative_l2=float(e["actual_relative_l2"][-1]),
                         final_refit_relative_l2=float(e["refit_relative_l2"][-1]),
                         final_gamma=float(c["mean_gamma"][-1]),
                         best_saved_step=int(e["steps"][best]),
                         best_saved_refit_relative_l2=float(e["refit_relative_l2"][best]),
                         maximum_step_variation=max(a["relative_step_variation"] for a in audits),
                         unresolved_audits=sum(not a["gF_resolved"] for a in audits), audit_count=len(audits)))
        for which, index in (("final", len(e["steps"])-1), ("best_saved", best)):
            state = {k: c[k][index] for k in ("a", "b", "v")}
            for cutoff, n_eval in ((1e-13, 65536), (1e-12, cfg["n_eval"]), (1e-14, cfg["n_eval"])):
                checks.append(dict(target=target, ratio=ratio, schedule=schedule, which=which,
                                   step=int(e["steps"][index]), cutoff=cutoff, n_eval=n_eval,
                                   **refits.refit(state, target, cfg, cutoff=cutoff, n_eval=n_eval)))
    qi = json.loads((fixed.RESULTS / "data/summary.json").read_text())["qi_reference"]
    summary = dict(runs=rows, checks=checks, qi_reference=qi)
    (RESULTS / "data/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot(cases, evaluations, controls, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator

    cfg = fixed.config("constant")
    colors = dict(zip(RATIOS, plt.cm.viridis(np.linspace(.05, .9, len(RATIOS)))))
    limits = {}
    for target in cfg["targets"]:
        cs = [c for (t, _, _), c in cases.items() if t == target]
        cs += [controls[target, sc][0] for sc in SCHEDULES]
        loss = np.concatenate([c["L"] for c in cs])
        gamma = np.concatenate([c["mean_gamma"] for c in cs])
        limits[target] = (float(loss.min())*.6, float(loss.max())*1.6,
                          max(.01, float(gamma.max()))*1.08)
    error_max = max(float(e["refit_relative_l2"].max()) for e in evaluations.values())
    handles = [Line2D([], [], color=colors[r], lw=2, label=f"r = {r:g}") for r in RATIOS]
    handles += [Line2D([], [], color=".1", ls=":", lw=1.5, label="Ordinary Adam: same schedule"),
                Line2D([], [], color=".65", ls="--", label="QI refit reference"),
                Line2D([], [], color=".3", ls="none", marker="o", mfc="none", ms=4,
                       label="Independent-grid refit check")]
    figdir = RESULTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for schedule in SCHEDULES:
        fig, axes = plt.subplots(3, 4, figsize=(20, 12), dpi=180, sharex=True)
        for col, target in enumerate(cfg["targets"]):
            axes[0, col].set_title(prior.profile.LABELS[target], fontsize=16, pad=13)
            control, ce = controls[target, schedule]
            axes[0, col].plot(control["step"], control["L"], color=".1", ls=":", lw=1.5)
            axes[1, col].plot(ce["steps"], ce["refit_relative_l2"], color=".1", ls=":", lw=1.5)
            axes[2, col].plot(control["step"], control["mean_gamma"], color=".1", ls=":", lw=1.5)
            for ratio in RATIOS:
                c, e = cases[target, ratio, schedule], evaluations[target, ratio, schedule]
                axes[0, col].plot(c["step"], c["L"], color=colors[ratio], lw=1.25)
                axes[1, col].plot(c["step"], c["refit_train_relative_l2"], color=colors[ratio], lw=1.25)
                axes[1, col].plot(e["steps"], e["refit_relative_l2"], ls="none", marker="o", ms=2,
                                  mfc="none", markeredgewidth=.5, color=colors[ratio])
                axes[2, col].plot(c["step"], c["mean_gamma"], color=colors[ratio], lw=1.25)
                if str(c["status"]) != "complete":
                    axes[0, col].plot(c["step"][-1], c["L"][-1], "x", color=colors[ratio], ms=8)
            axes[1, col].axhline(summary["qi_reference"][target]["refit_relative_l2"], color=".65", ls="--", lw=1)
            for row in (0, 1):
                axes[row, col].set_yscale("log")
                axes[row, col].yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            lo, hi, gmax = limits[target]
            axes[0, col].set_ylim(lo, hi)
            axes[1, col].set_ylim(1e-16, max(2, 1.5*error_max))
            axes[2, col].set_ylim(0, gmax)
            axes[2, col].yaxis.set_major_locator(MaxNLocator(nbins=5))
            axes[2, col].ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[2, col].set_xlabel("Training updates", fontsize=12)
            for ax in axes[:, col]:
                ax.set_xlim(0, 10000)
                ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
                ax.tick_params(labelsize=9)
                ax.grid(alpha=.17)
                ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel(r"Actual loss $L=\frac{1}{2}\mathrm{mean}(e^2)$" + "\n(log scale)", fontsize=12)
        axes[1, 0].set_ylabel(r"Refitted relative $L_2$ error" + "\n(log scale)", fontsize=12)
        axes[2, 0].set_ylabel(r"Mean $\gamma=\mathrm{mean}|a_k|$" + "\n(linear scale)", fontsize=12)
        rate = "Constant learning rate: 0.002" if schedule == "constant" else "Cosine decay over 10,000 steps: 0.002 → 0.000002"
        fig.suptitle(r"Xavier · VarPro split Adam · fixed update ratio $r$, dynamic $\mu_t$" + "\n" + rate,
                     fontsize=19, y=.99)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .918), ncol=4, frameon=False, fontsize=11)
        fig.subplots_adjust(left=.08, right=.985, top=.79, bottom=.15, hspace=.25, wspace=.27)
        fig.text(.5, .061,
                 r"$\mu_t=r\|u_G\|_2/\|u_F\|_2$, so $\|\mu_tu_F\|_2/\|u_G\|_2=r$ after separate Adam normalizations."
                 + "\nGeometry update: −ηt(μt uF + uG). Ordinary-Adam readout; common rate for all streams; continuous moments."
                 + "\nSame Xavier seed, samples, width, Adam ε=10⁻⁸, and original numerical VarPro derivative. Matching panels have identical axes."
                 + "\nMiddle: training-refit relative error each step; circles check saved refits on 8,192 independent samples. Solved coefficients are never installed.",
                 ha="center", va="center", fontsize=10, linespacing=1.5)
        fig.savefig(figdir / f"{schedule}.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = fixed.config("constant")
    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    with threadpool_limits(cfg["threads"]):
        if not (args.analyze_only or args.plot_only):
            for target in args.targets or cfg["targets"]:
                for schedule in SCHEDULES:
                    for ratio in RATIOS:
                        run_case(target, ratio, schedule)
        if args.train_only:
            return
        cases, evaluations, controls = load_cases()
        summary = json.loads((RESULTS / "data/summary.json").read_text()) if args.plot_only else summarize(cases, evaluations)
        plot(cases, evaluations, controls, summary)
    print(f"Recorded {len(summary['runs'])} dynamic-ratio runs; plots saved in {RESULTS / 'figures'}")


if __name__ == "__main__":
    main()
