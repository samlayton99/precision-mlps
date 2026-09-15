"""Matched four-way geometry/readout comparison; refits never enter training.

All targets use the same finite-domain model and samples. Two trajectories per
initialization supply four curves: joint GD, frozen-geometry GD, joint snapshots
with refitted readouts, and one initial-geometry readout solve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum import first_steps_refit as matched

RESULTS = matched.rc.RESULTS.parent / "four_way_comparison"
TARGETS = ("sine", "sine_mixture", "runge", "gaussian_envelope")
ARMS = ("xavier", "gamma_1", "gamma_4", "gamma_16")
ARM_LABELS = ("Xavier · random readout", "γ₀ = 1 · zero readout",
              "γ₀ = 4 · zero readout", "γ₀ = 16 (QI) · zero readout")
COLORS = ("#2274b5", "#d98428", "#20966b", "#9a50a3")
CURVE_LABELS = ("1  Free geometry + GD", "2  Frozen initial geometry + GD",
                "3  Free geometry + GD; LS evaluation", "4  Frozen initial geometry; LS evaluation")


def config():
    return dict(version=1, resolution=128, halo=24, seed=0, steps=2000,
                learning_rate=.002, n_train=1024, n_eval=32768, threads=2,
                domain=[-1, 1], envelope_sigma=.4, target_modes=[2., 6., 14.],
                target_amplitudes=[1., .5, .25], readout_rcond=matched.rc.RCOND,
                targets=list(TARGETS), arms=list(ARMS),
                model="standard tanh(ax+b) @ c + d; unconstrained readout for every target",
                evaluation="same independent midpoint grid for all four curves; refits use training samples")


def midpoint_grid(n):
    return -1 + (np.arange(n) + .5) * 2 / n


def evaluation_steps(total):
    early = np.arange(min(total, 20) + 1)
    later = np.rint(np.geomspace(21, total, 80)).astype(int) if total > 20 else []
    return np.unique(np.r_[early, later, total]).astype(int)


def initial_state(arm, cfg):
    if arm == "xavier":
        arrays, _ = matched.rc.original.initial_arrays(cfg["resolution"], cfg["seed"],
                                                      "xavier", halo=cfg["halo"])
        return dict(a=arrays["a"], b=arrays["b"], v=np.r_[arrays["c"], arrays["d"]])
    gamma = float(arm.split("_")[1])
    centers, _, _ = matched.rc.original.uniform_geometry(cfg["resolution"], cfg["halo"])
    return dict(a=np.full(len(centers), gamma), b=-gamma*centers, v=np.zeros(len(centers)+1))


def design(x, a, b):
    return np.column_stack((np.tanh(x[:, None]*a+b), np.ones(len(x))))


def train_pair(target, arm, cfg):
    x = midpoint_grid(cfg["n_train"])
    y = matched.target_values(target, x, cfg)
    initial = initial_state(arm, cfg)
    params = {key: torch.nn.Parameter(torch.tensor(value)) for key, value in initial.items()}
    tx, ty = torch.tensor(x), torch.tensor(y)
    A0 = torch.tensor(design(x, initial["a"], initial["b"]))
    G, h = A0.T @ A0 / len(x), A0.T @ ty / len(x)
    # Preserve the paired random readout for Xavier; zero only for the uniform arms.
    frozen_v = torch.tensor(initial["v"].copy())
    optimizer = torch.optim.SGD(params.values(), lr=cfg["learning_rate"])
    chosen = evaluation_steps(cfg["steps"])
    chosen_lookup = {int(step): index for index, step in enumerate(chosen)}
    data = {key: np.empty((len(chosen), value.size)) for key, value in initial.items()}
    data.update(steps=chosen, frozen_v=np.empty_like(data["v"]),
                joint_train_loss=np.empty(cfg["steps"]+1),
                frozen_train_loss=np.empty(cfg["steps"]+1))
    max_frozen_gradient_error = 0.
    for step in range(cfg["steps"]+1):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(tx[:, None]*params["a"]+params["b"])
        residual = hidden @ params["v"][:-1]+params["v"][-1]-ty
        loss = .5*torch.mean(residual.square())
        frozen_residual = A0 @ frozen_v-ty
        frozen_gradient = G @ frozen_v-h
        data["joint_train_loss"][step] = loss.item()
        data["frozen_train_loss"][step] = (.5*torch.mean(frozen_residual.square())).item()
        if step in chosen_lookup:
            index = chosen_lookup[step]
            for key, value in params.items():
                data[key][index] = value.detach().numpy()
            data["frozen_v"][index] = frozen_v.numpy()
            direct = A0.T @ frozen_residual / len(x)
            discrepancy = torch.max(torch.abs(direct-frozen_gradient)).item()
            max_frozen_gradient_error = max(max_frozen_gradient_error, discrepancy)
            np.testing.assert_allclose(frozen_gradient.numpy(), direct.numpy(), rtol=1e-8, atol=5e-14)
        if step % 500 == 0:
            energy = torch.mean(ty.square()).item()
            print(f"{target}/{arm}: step {step}, joint L2={np.sqrt(2*loss.item()/energy):.6g}, "
                  f"frozen L2={np.sqrt(2*data['frozen_train_loss'][step]/energy):.6g}", flush=True)
        if step < cfg["steps"]:
            loss.backward()
            optimizer.step()
            frozen_v -= cfg["learning_rate"]*frozen_gradient
    assert np.all(np.isfinite(data["joint_train_loss"]))
    assert np.all(np.isfinite(data["frozen_train_loss"]))
    np.testing.assert_array_equal(data["v"][0], data["frozen_v"][0])
    for key in initial:
        np.testing.assert_array_equal(data[key][0], initial[key])
    if arm != "xavier" and cfg["steps"] >= 1:
        for key in ("a", "b"):
            np.testing.assert_array_equal(data[key][0], data[key][1])
    data["frozen_gradient_max_error"] = np.array(max_frozen_gradient_error)
    return data


def evaluate_pair(data, target, cfg):
    """Only consume completed trajectories; solve a readout at every evaluation."""
    before = {key: data[key].copy() for key in ("a", "b", "v", "frozen_v")}
    x, xx = midpoint_grid(cfg["n_train"]), midpoint_grid(cfg["n_eval"])
    y, yy = (matched.target_values(target, nodes, cfg) for nodes in (x, xx))
    weights = np.full(len(x), 1/len(x))
    A0 = design(x, data["a"][0], data["b"][0])
    initial_refit, initial_info = matched.rc.solve_readout(A0, y, weights)
    count = len(data["steps"])
    result = dict(errors=np.zeros((count, 4)), train_errors=np.zeros((count, 4)),
                  refit_v=np.empty_like(data["v"]), refit_rank=np.empty(count, dtype=int),
                  refit_readout_norm=np.empty(count), refit_stationarity=np.empty(count))
    # Multiplication by a fixed dictionary evaluates all frozen readouts together.
    A0_eval = design(xx, data["a"][0], data["b"][0])
    frozen_eval_residual = A0_eval @ data["frozen_v"].T - yy[:, None]
    result["errors"][:, 1] = np.linalg.norm(frozen_eval_residual, axis=0)/np.linalg.norm(yy)
    initial_eval_error = np.linalg.norm(A0_eval @ initial_refit-yy)/np.linalg.norm(yy)
    initial_train_error = np.linalg.norm(A0 @ initial_refit-y)/np.linalg.norm(y)
    result["errors"][:, 3] = initial_eval_error
    result["train_errors"][:, 3] = initial_train_error
    result["train_errors"][:, :2] = np.sqrt(
        2*np.column_stack((data["joint_train_loss"], data["frozen_train_loss"]))[data["steps"]]
        / np.mean(y*y))
    for index, step in enumerate(data["steps"]):
        A = design(x, data["a"][index], data["b"][index])
        if index == 0:
            refit, info = initial_refit.copy(), initial_info
        else:
            refit, info = matched.rc.solve_readout(A, y, weights)
        result["refit_v"][index] = refit
        result["refit_rank"][index] = info["rank"]
        result["refit_readout_norm"][index] = info["readout_norm"]
        result["refit_stationarity"][index] = info["stationarity"]
        result["train_errors"][index, 2] = np.linalg.norm(A @ refit-y)/np.linalg.norm(y)
        Ae = A0_eval if index == 0 else design(xx, data["a"][index], data["b"][index])
        result["errors"][index, 0] = np.linalg.norm(Ae @ data["v"][index]-yy)/np.linalg.norm(yy)
        result["errors"][index, 2] = np.linalg.norm(Ae @ refit-yy)/np.linalg.norm(yy)
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
    np.testing.assert_allclose(result["errors"][0, 0], result["errors"][0, 1], rtol=2e-14, atol=2e-15)
    np.testing.assert_array_equal(result["errors"][:, 3], np.full(count, result["errors"][0, 2]))
    if np.all(data["v"][0] == 0) and cfg["steps"] >= 1:
        np.testing.assert_array_equal(result["errors"][0, 2], result["errors"][1, 2])
    assert np.all(np.isfinite(result["errors"]))
    return result


def save(cases, cfg, path):
    payload = {"config_json": json.dumps(cfg)}
    payload.update({f"{case_key}__{field}": value
                    for case_key, case in cases.items() for field, value in case.items()})
    temporary = path.with_name("comparison.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as source:
        cfg = json.loads(str(source["config_json"]))
        cases = {}
        for name in source.files:
            if name == "config_json":
                continue
            target, arm, field = name.split("__")
            cases.setdefault(f"{target}__{arm}", {})[field] = source[name]
    return cases, cfg


def plot(cases, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(4, 4, figsize=(19, 14), dpi=170, sharex=True, sharey=True)
    for row, target in enumerate(TARGETS):
        for col, arm in enumerate(ARMS):
            ax, case = axes[row, col], cases[f"{target}__{arm}"]
            steps, values = case["steps"], case["errors"]
            # Draw dashed references first, leaving them visible beside the solid curves.
            for curve, style, width in ((1, "--", 2.7), (0, "-", 1.3), (3, ":", 2.8), (2, "-", 1.25)):
                ax.plot(steps, np.maximum(values[:, curve], 1e-16), style,
                        color=COLORS[curve], lw=width,
                        marker="o" if curve == 2 else None, ms=2, markevery=10)
            ax.set(xscale="symlog", yscale="log", xlim=(0, cfg["steps"]), ylim=(1e-16, 2.5))
            ax.set_xscale("symlog", linthresh=2)
            ax.set_xticks([0, 2, 10, 100, 2000], labels=["0", "2", "10", "100", "2000"])
            ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
            ax.grid(alpha=.18)
            gain = 100*(1-values[-1, 0]/values[-1, 1])
            ax.text(.035, .06, f"Final GD benefit: {gain:+.3f}%\n"
                    f"LS: {values[0, 3]:.3g} → {values[-1, 2]:.3g}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.88))
            if row == 0:
                ax.set_title(ARM_LABELS[col], fontsize=12, pad=14)
            if col == 0:
                ax.set_ylabel(matched.rc.LABELS[target]+"\nRelative L₂ error", fontsize=11)
            if row == 3:
                ax.set_xlabel("GD step (log spacing after 2)")
    fig.suptitle("Does moving geometry improve the GD fit or the best readout fit?", fontsize=21, y=.984)
    handles = [Line2D([], [], color=c, lw=2, ls=s) for c, s in zip(COLORS, ("-", "--", "-", ":"))]
    fig.legend(handles, CURVE_LABELS, loc="upper center", bbox_to_anchor=(.5, .953), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(top=.86, bottom=.10, left=.065, right=.99, hspace=.20, wspace=.13)
    fig.text(.5, .027,
             f"All targets: same 1,024 training midpoints and 32,768 evaluation midpoints in [−1,1], including Gaussian. "
             f"{len(next(iter(cases.values()))['steps'])} shared evaluation states; every evaluation includes an LS solve.\n"
             "Each free/frozen pair starts identically. Ordinary raw (a,b,c,d) GD, rate 0.002, 2,000 steps; N=128 intervals, 177 neurons, halo 24 per side.\n"
             "Readout solves use training samples and SVD cutoff 10⁻¹³; solves never enter training. Curve 4 is the fixed initial refit. All panels share axes; display floor 10⁻¹⁶.",
             ha="center", fontsize=10, linespacing=1.5)
    fig.savefig(RESULTS / "comparison.png")
    plt.close(fig)

    # Compare improvements in the same units. Dividing by tiny LS baselines
    # would amplify numerical variations and hide the actual GD benefit.
    fig, axes = plt.subplots(4, 4, figsize=(19, 13), dpi=170, sharex=True)
    for row, target in enumerate(TARGETS):
        for col, arm in enumerate(ARMS):
            ax, case = axes[row, col], cases[f"{target}__{arm}"]
            steps, values = case["steps"], case["errors"]
            gd_gain = values[:, 1]-values[:, 0]
            ax.plot(steps, gd_gain, color=COLORS[0], lw=1.8)
            ls_gain = values[:, 3]-values[:, 2]
            ax.plot(steps, ls_gain, color=COLORS[2], lw=1.5)
            visible = [gd_gain, ls_gain]
            extent = max(float(np.max(abs(v))) for v in visible)
            extent = max(extent, 1e-10)
            ax.set_ylim(-1.2*extent, 1.2*extent)
            ax.axhline(0, color="#777777", lw=.7)
            ax.set_xscale("symlog", linthresh=2)
            ax.set_xlim(0, cfg["steps"])
            ax.set_xticks([0, 2, 10, 100, 2000], labels=["0", "2", "10", "100", "2000"])
            ax.grid(alpha=.18)
            ax.text(.03, .06, f"Final GD reduction: {gd_gain[-1]:+.3g}\n"
                    f"Final LS reduction: {ls_gain[-1]:+.3g}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
            if row == 0:
                ax.set_title(ARM_LABELS[col], fontsize=12, pad=14)
            if col == 0:
                ax.set_ylabel(matched.rc.LABELS[target]+"\nDecrease in relative L₂ error", fontsize=11)
            if row == 3:
                ax.set_xlabel("GD step (log spacing after 2)")
    fig.suptitle("How much does learned geometry improve each comparison?", fontsize=21, y=.985)
    fig.legend([Line2D([], [], color=COLORS[i], lw=2) for i in (0, 2)],
               ["GD benefit: curve 2 − curve 1", "LS benefit: curve 4 − curve 3"],
               loc="upper center", bbox_to_anchor=(.5, .947), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(top=.865, bottom=.145, left=.065, right=.99, hspace=.24, wspace=.25)
    fig.text(.5, .028, "Same runs and evaluation samples as the four-way comparison. Positive = improvement from allowing geometry to move; negative = worse.\n"
             "Blue and green use the same units: an absolute decrease in relative L₂ error, without dividing by either baseline. Vertical ranges vary by panel; zero is centered.\n"
             "All curves are retained. Small LS changes can reflect finite-precision SVD effects; the main figure shows the underlying errors. Both centers and scales are free in joint GD.",
             ha="center", fontsize=10, linespacing=1.5)
    fig.savefig(RESULTS / "gains.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    path = RESULTS / "data/comparison.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = {}
    if path.exists():
        cases, previous_config = load(path)
        if cfg != previous_config:
            raise ValueError("Saved comparison configuration differs.")
    with threadpool_limits(limits=cfg["threads"]):
        for target in TARGETS:
            for arm in ARMS:
                key = f"{target}__{arm}"
                if key in cases:
                    continue
                if args.plot_only:
                    raise ValueError(f"Missing case {key}")
                case = train_pair(target, arm, cfg)
                case.update(evaluate_pair(case, target, cfg))
                cases[key] = case
                save(cases, cfg, path)
                print("Completed", key, "final four errors", case["errors"][-1].tolist(), flush=True)
        plot(cases, cfg)
    print(RESULTS / "comparison.png", flush=True)
    print(RESULTS / "gains.png", flush=True)


if __name__ == "__main__":
    main()
