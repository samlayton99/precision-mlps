"""Freeze a GD-trained readout after fitting a nonlinear Runge predictor."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD26_freeze_and_readout_spectrum import freeze as base
from experiments.expD28_loss_gradient_decomposition import run as setup

RESULTS = base.RESULTS/"freeze_mechanism"


def config():
    return base.config() | yaml.safe_load(Path(__file__).with_suffix(".yaml").read_text())


def train(initial, cfg, *, frozen=False, warmup=False):
    x = base.original.midpoint_grid(cfg["n_train"])
    y = base.original.matched.target_values(cfg["target"], x, cfg)
    xe = base.original.midpoint_grid(cfg["n_eval"])
    ye = base.original.matched.target_values(cfg["target"], xe, cfg)
    p = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64), requires_grad=not(frozen and k=="v"))
         for k, v in initial.items()}
    opt = torch.optim.SGD([v for v in p.values() if v.requires_grad], lr=cfg["learning_rate"])
    tx, ty = torch.tensor(x), torch.tensor(y)
    histories, saved, snapshots = [], [], {k: [] for k in p}
    budget = cfg["warmup_cap"] if warmup else cfg["continuation_steps"]
    completed = False
    for step in range(budget+1):
        opt.zero_grad(set_to_none=True)
        pred = torch.tanh(tx[:, None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
        loss = .5*torch.mean((pred-ty).square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}")
        state = {k: v.detach().numpy() for k, v in p.items()}
        relative = float(torch.sqrt(2*loss.detach()/torch.mean(ty.square())))
        eval_error = np.nan
        if step%cfg["evaluation_stride"]==0 or step==budget or (warmup and relative<=cfg["fit_threshold"]):
            predicted = np.tanh(xe[:, None]*state["a"]+state["b"])@state["v"][:-1]+state["v"][-1]
            eval_error = float(np.linalg.norm(predicted-ye)/np.linalg.norm(ye))
        done = warmup and relative<=cfg["fit_threshold"] and eval_error<=cfg["fit_threshold"]
        histories.append(dict(step=step, L=float(loss.detach()), train_relative_l2=relative,
                              eval_relative_l2=eval_error, mean_gamma=float(np.mean(abs(state["a"]))),
                              mean_readout=float(np.mean(abs(state["v"][:-1]))),
                              mean_abs_bias=float(np.mean(abs(state["b"]))), output_bias=float(state["v"][-1])))
        if step%cfg["snapshot_stride"]==0 or step in (0, 1, 2) or step==budget or done:
            saved.append(step)
            for k in p:
                snapshots[k].append(state[k].copy())
        if step%2000==0 or done or step==budget:
            print(f"{'warmup' if warmup else 'frozen' if frozen else 'joint'} {step}: "
                  f"E={relative:.6g}, mean c={histories[-1]['mean_readout']:.7g}, "
                  f"mean gamma={histories[-1]['mean_gamma']:.9g}", flush=True)
        if done or step==budget:
            completed = bool(done or not warmup)
            break
        loss.backward()
        opt.step()
    if not completed:
        raise RuntimeError("Warmup did not reach the prescribed accuracy; do not label it well fitted.")
    c = {key: np.asarray([r[key] for r in histories]) for key in histories[0]}
    c.update({k: np.asarray(v) for k, v in snapshots.items()})
    c["saved_steps"] = np.asarray(saved)
    return c


def verify(cases, cfg):
    warm, joint, frozen = (cases[k] for k in ("warmup", "joint", "frozen"))
    initial = {k: warm[k][-1] for k in base.FIELDS}
    for k in base.FIELDS:
        np.testing.assert_array_equal(joint[k][0], initial[k])
        np.testing.assert_array_equal(frozen[k][0], initial[k])
    # Every frozen readout is saved; immutability also follows from its
    # absence from the optimizer, checked by requires_grad in train().
    np.testing.assert_array_equal(frozen["v"], np.broadcast_to(initial["v"], frozen["v"].shape))
    for k in ("a", "b"):
        np.testing.assert_array_equal(frozen[k][1], joint[k][1])
    x = base.original.midpoint_grid(cfg["n_train"])
    y = base.original.matched.target_values(cfg["target"], x, cfg)
    grads = base.numpy_gradients(initial, x, y)
    for k in ("a", "b"):
        np.testing.assert_allclose(frozen[k][1], initial[k]-cfg["learning_rate"]*grads[k], rtol=2e-14, atol=3e-15)
    assert warm["train_relative_l2"][-1]<=cfg["fit_threshold"]
    assert warm["eval_relative_l2"][-1]<=cfg["fit_threshold"]
    xe = base.original.midpoint_grid(cfg["n_eval"])
    ye = base.original.matched.target_values(cfg["target"], xe, cfg)
    affine = np.mean(ye)+xe*np.mean(xe*ye)/np.mean(xe*xe)
    affine_error = float(np.linalg.norm(affine-ye)/np.linalg.norm(ye))
    assert affine_error > 10*warm["eval_relative_l2"][-1]
    summary = dict(freeze_step=int(warm["step"][-1]), best_affine_error=affine_error,
                   threshold=cfg["fit_threshold"], warmup={k:float(warm[k][-1]) for k in ("train_relative_l2", "eval_relative_l2", "mean_gamma", "mean_readout")})
    for name, c in (("joint", joint), ("frozen", frozen)):
        summary[name] = {k:float(c[k][-1]) for k in ("train_relative_l2", "eval_relative_l2", "mean_gamma", "mean_readout")}
        summary[name]["gamma_change_after_freeze"] = float(c["mean_gamma"][-1]-c["mean_gamma"][0])
        summary[name]["readout_change_after_freeze"] = float(c["mean_readout"][-1]-c["mean_readout"][0])
    return summary


def plot(cases, cfg, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    warm = cases["warmup"]
    freeze_step = summary["freeze_step"]
    colors = {"frozen":"#482878", "joint":"#21918c"}
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), dpi=160, sharex="col")
    keys = ("eval_relative_l2", "mean_readout", "mean_gamma")
    labels = ("Relative $L_2$ error\n(independent grid; log scale)",
              r"Mean neuron readout size $\mathrm{mean}|c_k|$",
              r"Mean scale $\overline{\gamma}=\mathrm{mean}|a_k|$")
    for row, key in enumerate(keys):
        for col in range(2):
            ax = axes[row, col]
            shift = freeze_step if col else 0
            mask = np.isfinite(warm[key])
            ax.plot(warm["step"][mask]-shift, warm[key][mask], color=".45", lw=2)
            for name in ("frozen", "joint"):
                c = cases[name]; valid = np.isfinite(c[key])
                ax.plot(c["step"][valid]+freeze_step-shift, c[key][valid], color=colors[name], lw=2)
            ax.axvline(0 if col else freeze_step, color=".5", ls=":", lw=1.2)
            if row==0:
                ax.set_yscale("log")
                if col==0:
                    ax.axhline(summary["best_affine_error"], color=".6", ls="--", lw=1.2)
            else:
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(y=.12)
            if col:
                ax.set_xlim(-500, cfg["continuation_steps"])
                # Autoscale the zoom's vertical axis using only its visible
                # data, so tiny changes are readable without an offset label.
                values = [warm[key][(warm["step"]>=freeze_step-500)&np.isfinite(warm[key])]]
                values += [cases[n][key][np.isfinite(cases[n][key])] for n in colors]
                values = np.concatenate(values)
                low, high = float(values.min()), float(values.max())
                if row==0:
                    ax.set_ylim(low/1.12, high*1.12)
                else:
                    pad = max((high-low)*.15, abs(high)*1e-9)
                    ax.set_ylim(low-pad, high+pad)
            else:
                ax.set_xlim(0, freeze_step+cfg["continuation_steps"])
        axes[row, 0].set_ylabel(labels[row], fontsize=11)
    axes[0, 0].set_title("Full training history", fontsize=14, pad=14)
    axes[0, 1].set_title("Around and after freezing · expanded vertical ranges", fontsize=14, pad=14)
    axes[2, 0].set_xlabel("Total GD updates", fontsize=11)
    axes[2, 1].set_xlabel("Updates relative to readout freeze", fontsize=11)
    handles = [Line2D([], [], color=".45", lw=2, label="Common joint-GD warmup"),
               Line2D([], [], color=colors["frozen"], lw=2, label="Frozen readout"),
               Line2D([], [], color=colors["joint"], lw=2, label="Continued joint GD"),
               Line2D([], [], color=".6", ls="--", label="Best affine approximation (top left)")]
    fig.suptitle("Readout freezing after fitting a nonlinear Runge model\nStart at uniform γ=16; freeze once relative error reaches 1%", fontsize=17, y=.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .907), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.115, right=.97, top=.80, bottom=.14, hspace=.29, wspace=.24)
    fig.text(.5, .055,
             f"Freeze at update {freeze_step}; 5,000 further updates. Same constant GD rate η=0.002 for every trainable parameter. No coefficient solves.\n"
             "Mean readout size excludes the output bias; that bias is also frozen. Both means include all 177 neurons.\n"
             f"Branch error: {100*summary['warmup']['eval_relative_l2']:.4f}%; best affine error: {100*summary['best_affine_error']:.2f}%. Independent grid has 8,192 points.\n"
             "The first post-freeze geometry update is identical in both branches; differences develop as subsequent readout updates change the residual.",
             ha="center", va="center", fontsize=9.5)
    fig.savefig(RESULTS/"late_freeze_runge.png")
    plt.close(fig)


def main():
    cfg = config()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    (RESULTS/"data").mkdir(parents=True, exist_ok=True)
    path = RESULTS/"data"/"late_freeze_runge.npz"
    with threadpool_limits(limits=cfg["threads"]):
        if path.exists():
            with np.load(path) as source:
                assert json.loads(str(source["config_json"]))==cfg
                cases = {}
                for k in source.files:
                    if "__" in k:
                        name, field = k.split("__", 1)
                        cases.setdefault(name, {})[field] = source[k]
        else:
            initial = setup.initial_state(cfg["initialization"], cfg)
            warm = train(initial, cfg, warmup=True)
            branch = {k:warm[k][-1] for k in base.FIELDS}
            cases = dict(warmup=warm, joint=train(branch, cfg), frozen=train(branch, cfg, frozen=True))
            np.savez_compressed(path, config_json=json.dumps(cfg),
                                **{f"{name}__{key}":value for name,c in cases.items() for key,value in c.items()})
        summary = verify(cases, cfg)
        (RESULTS/"data"/"late_freeze_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        plot(cases, cfg, summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
