"""Branch ordinary Xavier GD into readout-frozen geometry continuations.

Training is raw (a,b,c,d) SGD on half mean squared error. Every diagnostic,
including independent-grid evaluation and numerical readout refits, consumes
completed trajectories and never changes a training state.
"""
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum import four_way_comparison as original

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum"
FIELDS = ("a", "b", "v")
LABELS = original.matched.rc.LABELS


def config():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def numpy_gradients(state, x, y):
    """Independent analytic gradients of L = ||f-y||²/(2n)."""
    hidden = np.tanh(x[:, None] * state["a"] + state["b"])
    residual = hidden @ state["v"][:-1] + state["v"][-1] - y
    common = residual[:, None] * (1 - hidden**2) * state["v"][:-1]
    return dict(a=np.mean(x[:, None] * common, axis=0),
                b=np.mean(common, axis=0),
                v=np.r_[hidden.T @ residual / len(x), np.mean(residual)])


def train_trajectory(initial, x, y, n_updates, learning_rate, *, freeze_readout=False):
    params = {key: torch.nn.Parameter(torch.tensor(initial[key], dtype=torch.float64),
                                      requires_grad=not (freeze_readout and key == "v"))
              for key in FIELDS}
    optimizer = torch.optim.SGD([p for p in params.values() if p.requires_grad], lr=learning_rate)
    tx, ty = torch.tensor(x), torch.tensor(y)
    data = {key: np.empty((n_updates + 1, len(initial[key]))) for key in FIELDS}
    data["train_loss"] = np.empty(n_updates + 1)
    for step in range(n_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(tx[:, None] * params["a"] + params["b"])
        residual = hidden @ params["v"][:-1] + params["v"][-1] - ty
        loss = .5 * torch.mean(residual.square())
        data["train_loss"][step] = loss.item()
        for key in FIELDS:
            data[key][step] = params[key].detach().numpy()
        if step < n_updates:
            loss.backward()
            optimizer.step()
    data["steps"] = np.arange(n_updates + 1)
    data["mean_gamma"] = np.mean(np.abs(data["a"]), axis=1)
    data["mean_gamma_change"] = np.mean(np.abs(np.abs(data["a"]) - np.abs(data["a"][0])), axis=1)
    data["mean_gamma_travel"] = np.r_[0., np.cumsum(np.mean(np.abs(np.diff(np.abs(data["a"]), axis=0)), axis=1))]
    data["readout_norm"] = np.linalg.norm(data["v"], axis=1)
    for key, values in data.items():
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"Nonfinite {key} in trajectory")
    if freeze_readout:
        np.testing.assert_array_equal(data["v"], np.broadcast_to(initial["v"], data["v"].shape))
    return data


def train_branches(target, cfg):
    x = original.midpoint_grid(cfg["n_train"])
    y = original.matched.target_values(target, x, cfg)
    initial = original.initial_state("xavier", cfg)
    total = max(cfg["freeze_steps"]) + cfg["frozen_steps"]
    joint = train_trajectory(initial, x, y, total, cfg["learning_rate"])
    cases = {"joint": joint}
    max_gradient_error = 0.
    for freeze_step in cfg["freeze_steps"]:
        start = {key: joint[key][freeze_step].copy() for key in FIELDS}
        continuation = train_trajectory(start, x, y, cfg["frozen_steps"],
                                        cfg["learning_rate"], freeze_readout=True)
        branch = {key: np.concatenate((joint[key][:freeze_step], continuation[key]))
                  for key in (*FIELDS, "train_loss")}
        branch["steps"] = np.arange(freeze_step + cfg["frozen_steps"] + 1)
        branch["mean_gamma"] = np.mean(np.abs(branch["a"]), axis=1)
        branch["mean_gamma_change"] = np.mean(np.abs(np.abs(branch["a"]) - np.abs(initial["a"])), axis=1)
        branch["mean_gamma_travel"] = np.r_[0., np.cumsum(np.mean(np.abs(np.diff(np.abs(branch["a"]), axis=0)), axis=1))]
        branch["readout_norm"] = np.linalg.norm(branch["v"], axis=1)
        branch["freeze_step"] = np.array(freeze_step)
        gradients = numpy_gradients(start, x, y)
        for key in FIELDS:
            np.testing.assert_array_equal(branch[key][:freeze_step + 1], joint[key][:freeze_step + 1])
            if key == "v":
                np.testing.assert_array_equal(branch[key][freeze_step:],
                                              np.broadcast_to(start[key], branch[key][freeze_step:].shape))
            else:
                expected = start[key] - cfg["learning_rate"] * gradients[key]
                discrepancy = np.max(np.abs(expected - branch[key][freeze_step + 1]))
                max_gradient_error = max(max_gradient_error, float(discrepancy))
                np.testing.assert_allclose(branch[key][freeze_step + 1], expected, rtol=3e-14, atol=3e-15)
        cases[f"freeze_{freeze_step}"] = branch
    return cases, max_gradient_error


def refit_steps(freeze_step, frozen_steps):
    before = [s for s in (0, 2, 10, 50, 150) if s < freeze_step]
    offsets = [s for s in (0, 1, 2, 5, 10, 25, 50, 100, 250, 500) if s <= frozen_steps]
    return np.unique([*before, *(freeze_step + np.array(offsets)), freeze_step + frozen_steps]).astype(int)


def solve_readout(state, x, y, rcond):
    matrix = original.design(x, state["a"], state["b"]) / np.sqrt(len(x))
    rhs = y / np.sqrt(len(x))
    u, s, vh = sla.svd(matrix, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    keep = s > rcond * s[0]
    coefficients = vh[keep].T @ ((u[:, keep].T @ rhs) / s[keep])
    residual = matrix @ coefficients - rhs
    return coefficients, dict(rank=int(np.sum(keep)), readout_norm=np.linalg.norm(coefficients),
                              stationarity=np.linalg.norm(matrix.T @ residual)/(s[0]*np.linalg.norm(rhs)))


def evaluate_target(cases, target, cfg):
    """Dense current-readout errors at every step; sparse, offline LS errors."""
    started = time.perf_counter()
    x, xx = original.midpoint_grid(cfg["n_train"]), original.midpoint_grid(cfg["n_eval"])
    y, yy = (original.matched.target_values(target, nodes, cfg) for nodes in (x, xx))
    y_norm = np.linalg.norm(yy)
    refit_sets = {f"freeze_{step}": refit_steps(step, cfg["frozen_steps"])
                  for step in cfg["freeze_steps"]}
    refit_sets["joint"] = np.unique(np.concatenate(list(refit_sets.values())))
    for name, case in cases.items():
        before = {key: case[key].copy() for key in FIELDS}
        count = len(case["steps"])
        case["relative_l2"] = np.empty(count)
        chosen = refit_sets[name]
        case["refit_steps"] = chosen
        case["refit_relative_l2"] = np.empty(len(chosen))
        case["refit_train_relative_l2"] = np.empty(len(chosen))
        case["refit_rank"] = np.empty(len(chosen), dtype=int)
        case["refit_readout_norm"] = np.empty(len(chosen))
        case["refit_stationarity"] = np.empty(len(chosen))
        case["refit_v"] = np.empty((len(chosen), case["v"].shape[1]))
        lookup = {int(step): index for index, step in enumerate(chosen)}
        for step in range(count):
            # Prefix diagnostics are copied from the common joint run, too.
            if name != "joint" and step <= int(case["freeze_step"]):
                case["relative_l2"][step] = cases["joint"]["relative_l2"][step]
            else:
                Ae = original.design(xx, case["a"][step], case["b"][step])
                case["relative_l2"][step] = np.linalg.norm(Ae @ case["v"][step] - yy)/y_norm
            if step in lookup:
                index = lookup[step]
                if name != "joint" and step <= int(case["freeze_step"]):
                    joint_index = int(np.searchsorted(cases["joint"]["refit_steps"], step))
                    for field in ("refit_relative_l2", "refit_train_relative_l2", "refit_rank",
                                  "refit_readout_norm", "refit_stationarity", "refit_v"):
                        case[field][index] = cases["joint"][field][joint_index]
                else:
                    state = {key: case[key][step] for key in FIELDS}
                    coefficients, info = solve_readout(state, x, y, cfg["readout_rcond"])
                    case["refit_relative_l2"][index] = np.linalg.norm(Ae @ coefficients - yy)/y_norm
                    case["refit_train_relative_l2"][index] = np.linalg.norm(original.design(x, state["a"], state["b"]) @ coefficients - y)/np.linalg.norm(y)
                    case["refit_v"][index] = coefficients
                    for field in ("rank", "readout_norm", "stationarity"):
                        case[f"refit_{field}"][index] = info[field]
        for key, old in before.items():
            np.testing.assert_array_equal(case[key], old)
        for key, value in case.items():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"Nonfinite {name}/{key}")
        print(f"{target}/{name}: E={case['relative_l2'][-1]:.6g}, "
              f"mean gamma={case['mean_gamma'][-1]:.6g}, "
              f"LS={case['refit_relative_l2'][-1]:.6g}", flush=True)
    return time.perf_counter() - started


def save(cases, target, cfg, metadata):
    path = RESULTS / "data" / f"freeze_{target}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(config_json=json.dumps(cfg), metadata_json=json.dumps(metadata))
    payload.update({f"{name}__{key}": value for name, case in cases.items() for key, value in case.items()})
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load(target):
    with np.load(RESULTS / "data" / f"freeze_{target}.npz", allow_pickle=False) as source:
        cfg = json.loads(str(source["config_json"]))
        cases = {}
        for key in source.files:
            if "__" in key:
                name, field = key.split("__")
                cases.setdefault(name, {})[field] = source[key]
    return cases, cfg


def plot_target(cases, target, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

    colors = plt.colormaps["viridis"]([.15, .37, .6, .83])
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8), dpi=170, sharey="row")
    joint = cases["joint"]
    err = np.concatenate([case["relative_l2"] for case in cases.values()])
    gam = np.concatenate([case["mean_gamma"] for case in cases.values()])
    gamma_pad = max(float(np.ptp(gam))*.14, float(np.mean(gam))*.001)
    for col, (freeze_step, color) in enumerate(zip(cfg["freeze_steps"], colors)):
        case = cases[f"freeze_{freeze_step}"]
        endpoint = freeze_step + cfg["frozen_steps"]
        for row, field in enumerate(("relative_l2", "mean_gamma")):
            ax = axes[row, col]
            ax.plot(joint["steps"][:endpoint + 1], joint[field][:endpoint + 1],
                    color="#777777", ls="--", lw=2.2)
            ax.plot(case["steps"], case[field], color=color, lw=1.6)
            ax.axvline(freeze_step, color="#444444", ls=":", lw=1.3)
            ax.set_xlim(0, max(cfg["freeze_steps"]) + cfg["frozen_steps"])
            ax.set_xticks([0, 150, 300, 450, 650])
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_yscale("log")
                lo, hi = max(float(err.min())*.92, 1e-16), float(err.max())*1.06
                ticks = MaxNLocator(nbins=5).tick_values(lo, hi)
                ax.set_yticks(ticks[(ticks >= lo) & (ticks <= hi)])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.5g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_ylim(lo, hi)
                ax.set_title(f"Freeze after {freeze_step} joint-GD steps", fontsize=12, pad=12)
            else:
                ax.set_ylim(float(gam.min())-gamma_pad, float(gam.max())+gamma_pad)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.set_xlabel("Total GD steps")
    axes[0, 0].set_ylabel("Relative $L_2$ error (log scale)", fontsize=12)
    axes[1, 0].set_ylabel(r"Mean scale $\overline{\gamma}=\frac{1}{m}\sum_j |a_j|$", fontsize=12)
    fig.suptitle(f"{LABELS[target]}: freeze the readout, continue geometry GD", fontsize=20, y=.98)
    handles = [Line2D([], [], color="#248a8d", lw=2), Line2D([], [], color="#777777", ls="--", lw=2),
               Line2D([], [], color="#444444", ls=":", lw=1.3)]
    fig.legend(handles, ["Joint GD, then frozen readout", "Continued joint GD", "Readout freeze"],
               loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False, fontsize=12)
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.16, hspace=.27, wspace=.14)
    fig.text(.5, .065,
             f"Same seeded Xavier start; raw (a,b,c,d) GD, learning rate {cfg['learning_rate']} throughout. "
             f"After the marked step, c and output bias d stay fixed for {cfg['frozen_steps']} updates; a and b continue learning.\n"
             f"All errors use {cfg['n_eval']:,} independent midpoints on [−1,1]; training uses {cfg['n_train']:,}. "
             f"Mean scale includes all {joint['a'].shape[1]} neurons. Each row shares both axis limits; its vertical range is enlarged to show movement.",
             ha="center", fontsize=10, linespacing=1.65)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / f"freeze_{target}.png")
    plt.close(fig)


def plot_refits(targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

    available = [target for target in targets if (RESULTS / "data" / f"freeze_{target}.npz").exists()]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 4, figsize=(17.5, 3.1*len(available)+2),
                             dpi=170, sharex=True, sharey="row", squeeze=False)
    colors = plt.colormaps["viridis"]([.15, .37, .6, .83])
    for row, target in enumerate(available):
        cases, cfg = load(target)
        joint = cases["joint"]
        all_errors = np.concatenate([case["refit_relative_l2"] for case in cases.values()])
        log_span = max(np.log10(all_errors.max()/all_errors.min()), .015)
        for col, (freeze_step, color) in enumerate(zip(cfg["freeze_steps"], colors)):
            ax = axes[row, col]
            case = cases[f"freeze_{freeze_step}"]
            active = joint["refit_steps"] <= case["steps"][-1]
            ax.axhline(joint["refit_relative_l2"][0], color="#bbbbbb", ls=":", lw=1)
            ax.plot(joint["refit_steps"][active], joint["refit_relative_l2"][active],
                    color="#777777", ls="--", lw=1.8, marker=".", ms=4)
            ax.plot(case["refit_steps"], case["refit_relative_l2"], color=color, lw=1.4, marker="o", ms=3)
            ax.axvline(freeze_step, color="#444444", ls=":", lw=1.1)
            ax.set(yscale="log", xlim=(0, max(cfg["freeze_steps"])+cfg["frozen_steps"]),
                   ylim=(all_errors.min()/10**(.12*log_span), all_errors.max()*10**(.12*log_span)))
            lo, hi = ax.get_ylim()
            ticks = MaxNLocator(nbins=5).tick_values(lo, hi)
            ax.set_yticks(ticks[(ticks >= lo) & (ticks <= hi)])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.6g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(lo, hi)
            ax.set_xticks([0, 150, 300, 450, 650])
            ax.grid(alpha=.18)
            if row == 0:
                ax.set_title(f"Freeze readout at step {freeze_step}", fontsize=12, pad=12)
            if col == 0:
                ax.set_ylabel(f"{LABELS[target]}\nRefitted relative $L_2$", fontsize=11)
            if row == len(available)-1:
                ax.set_xlabel("Total GD steps")
    fig.suptitle("Does readout freezing improve the learned geometry?", fontsize=21, y=.988)
    fig.legend([Line2D([], [], color="#248a8d", marker="o", ms=3),
                Line2D([], [], color="#777777", ls="--", marker=".", ms=4),
                Line2D([], [], color="#bbbbbb", ls=":")],
               ["Frozen-readout trajectory, then LS evaluation", "Joint-GD trajectory, then LS evaluation", "Initial geometry, LS evaluation"],
               loc="upper center", bbox_to_anchor=(.5, .954), ncol=3, frameon=False, fontsize=10.5)
    fig.subplots_adjust(top=.875, bottom=.105, left=.09, right=.99, hspace=.24, wspace=.15)
    fig.text(.5, .032,
             "Every marker uses a new numerical least-squares readout on the completed trajectory; no solve enters training. "
             "Vertical dotted lines mark when the training readout was frozen.\n"
             "Same independent evaluation grid as the primary figures; SVD cutoff 10⁻¹³. Each target row shares its enlarged y range. "
             "Small variations with large solved coefficients can include numerical cancellation.",
             ha="center", fontsize=10, linespacing=1.7)
    fig.savefig(RESULTS / "figures" / "freeze_refit.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(cfg["threads"])
    torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=cfg["threads"]):
        for target in args.targets or cfg["targets"]:
            if args.plot_only:
                cases, saved_cfg = load(target)
                plot_target(cases, target, saved_cfg)
            else:
                started = time.perf_counter()
                cases, max_gradient_error = train_branches(target, cfg)
                training_seconds = time.perf_counter() - started
                print(f"{target}: training complete in {training_seconds:.2f}s; offline evaluation next", flush=True)
                diagnostic_seconds = evaluate_target(cases, target, cfg)
                metadata = dict(training_seconds=training_seconds, diagnostic_seconds=diagnostic_seconds,
                                max_first_frozen_update_error=max_gradient_error,
                                readout_frozen_bitwise=True, prefix_identical=True, refits_mutate_training=False)
                save(cases, target, cfg, metadata)
                plot_target(cases, target, cfg)
        plot_refits(cfg["targets"])


if __name__ == "__main__":
    main()
