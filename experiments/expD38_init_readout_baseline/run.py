"""Matched ordinary-training trajectories, with observational readout solves.

Run from the repository's Python environment. No downloads or test-set tuning.
  python experiments/expD38_init_readout_baseline/run.py pilot
  python experiments/expD38_init_readout_baseline/run.py compare
  python experiments/expD38_init_readout_baseline/run.py plot

For independent processes, use --shards N --shard i, then plot after completion.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precision_d38_matplotlib")

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg
from scipy.io import loadmat
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expF04_qi_init_real_data.model import SimpleMLP2, qi_ridge_init_layer_

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/checkpoint_D_optimizers/expD38_init_readout_baseline"
TITLES = {"airfoil": "Airfoil", "bike_sharing": "Bike Sharing", "kin8nm": "Kin8nm",
          "pol": "Pol", "sarcos": "SARCOS", "superconductivity": "Superconductivity"}


def config():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    assert cfg["hidden_layers"] == 2 and cfg["activation"] == "tanh"
    assert cfg["dtype"] == "float64" and cfg["optimizer"] == "Adam"
    assert 1 <= cfg["width"] <= 2048
    assert cfg["qi_layers"] == "both_hidden"
    return cfg


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def recipe_config(cfg):
    """Campaign membership is not a change to an individual run's recipe.

Actual task, seed, LR, width, phase, and step budget live in the run identity.
This permits adding repeats without replacing completed, identical runs.
"""
    campaign = {"tasks", "learning_rates", "pilot_steps", "comparison_steps", "comparison_seeds",
                "long_budget_tasks", "long_budget_steps", "long_budget_seeds", "task_data_protocols"}
    return {k: v for k, v in cfg.items() if k not in campaign}


def same_identity(a, b):
    return ({k: v for k, v in a.items() if k != "config"} ==
            {k: v for k, v in b.items() if k != "config"} and
            recipe_config(a["config"]) == recipe_config(b["config"]))


def valid_data_identity(identity, cfg):
    expected = cfg.get("task_data_protocols", {}).get(identity["task"], "original")
    return identity.get("data_protocol", "original") == expected


def random_split(x, y):
    idx = np.random.default_rng(0).permutation(len(x))
    n_test = int(0.2 * len(x))
    te, tr = idx[:n_test], idx[n_test:]
    return x[tr], y[tr], x[te], y[te]


def load_data(task, cfg):
    data = ROOT / "data"
    if task in ("kin8nm", "pol"):
        path = data / f"cache_expD20/{task}.npz"
        z = np.load(path)
        x, y, xt, yt = (z[k] for k in ("Xtr", "ytr", "Xte", "yte"))
        sources = [path]
    elif task == "sarcos":
        assert cfg["task_data_protocols"][task] == "train_file_disjoint_random_80_20_v2"
        # The supplied test file duplicates training inputs (RealMLP, C.3.2).
        # Use only the 44,484 distinct source rows and construct disjoint splits.
        sources = [data / "sarcos/sarcos_inv.mat"]
        a = loadmat(sources[0])["sarcos_inv"]
        assert len(np.unique(a[:, :21], axis=0)) == len(a)
        x, y, xt, yt = random_split(a[:, :21], a[:, 21])
    elif task == "bike_sharing":
        import pandas as pd
        sources = [data / "bike_sharing/hour.csv"]
        df = pd.read_csv(sources[0])
        # Known category domains, shared by affine and neural models. Neither
        # target components (casual, registered) nor identifiers enter features.
        cats = {"season": range(1, 5), "mnth": range(1, 13), "hr": range(24),
                "weekday": range(7), "weathersit": range(1, 5)}
        parts = [df[["yr", "holiday", "workingday", "temp", "atemp", "hum", "windspeed"]].to_numpy()]
        for name, domain in cats.items():
            parts.append((df[name].to_numpy()[:, None] == np.array(list(domain))[None, :]).astype(float))
        x, y, xt, yt = random_split(np.hstack(parts), df["cnt"].to_numpy())
    else:
        sources = [data / ("airfoil/airfoil_self_noise.dat" if task == "airfoil" else "superconductivity/train.csv")]
        a = np.loadtxt(sources[0], **({} if task == "airfoil" else {"delimiter": ",", "skiprows": 1}))
        x, y, xt, yt = random_split(a[:, :-1], a[:, -1])
    x, y, xt, yt = (np.asarray(a, dtype=np.float64) for a in (x, y, xt, yt))
    y, yt = y.reshape(-1, 1), yt.reshape(-1, 1)
    idx = np.random.default_rng(cfg["validation_seed"]).permutation(len(x))
    n_val = max(1, int(cfg["validation_fraction"] * len(x)))
    iv, it = idx[:n_val], idx[n_val:]
    mean, scale = x[it].mean(0), x[it].std(0)
    scale[scale < 1e-12] = 1
    ymean, yscale = y[it].mean(), y[it].std()
    arrays = {"train": ((x[it] - mean) / scale, (y[it] - ymean) / yscale),
              "val": ((x[iv] - mean) / scale, (y[iv] - ymean) / yscale),
              "test": ((xt - mean) / scale, (yt - ymean) / yscale)}
    for xx, yy in arrays.values():
        assert np.isfinite(xx).all() and np.isfinite(yy).all()
    metadata = {"n_train": len(it), "n_val": len(iv), "n_test": len(xt), "d_in": x.shape[1],
                "target_mean": float(ymean), "target_std": float(yscale),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "split_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
                "input_mean": mean.tolist(), "input_std": scale.tolist()}
    return arrays, metadata


def affine_fit(features, targets, rcond=1e-12):
    # Centering implements an unpenalized intercept and handles dummy columns.
    mean, ymean = features.mean(0), targets.mean(0)
    coef, _, rank, singular = linalg.lstsq(features - mean, targets - ymean,
                                          cond=rcond, lapack_driver="gelsd")
    bias = ymean - mean @ coef
    return coef, bias, {"rank": int(rank), "singular_max": float(singular[0]),
                        "singular_min": float(singular[-1]), "coef_norm": float(np.linalg.norm(coef))}


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def frozen_ridge(features, ys, alphas, rcond):
    mean, ymean = features["train"].mean(0), ys["train"].mean(0)
    u, s, vt = linalg.svd(features["train"] - mean, full_matrices=False)
    rhs = u.T @ (ys["train"] - ymean)
    rows = []
    for alpha in alphas:
        factor = np.zeros_like(s)
        if alpha == 0:
            keep = s > s[0] * rcond
            factor[keep] = 1 / s[keep]
        else:
            factor = s / (s * s + len(u) * alpha)
        coef = vt.T @ (factor[:, None] * rhs)
        errors = {k: mse((h - mean) @ coef + ymean, ys[k]) for k, h in features.items()}
        rows.append({"alpha": alpha, **errors})
    return min(rows, key=lambda row: row["val"]), rows


def make_model(d, width, seed, scheme, train_x, cfg):
    torch.manual_seed(seed)
    model = SimpleMLP2(d, width, 1, activation="tanh").double()
    with torch.no_grad():
        for layer in (model.fc1, model.fc2):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(model.fc3.weight, gain=1)
        nn.init.zeros_(model.fc3.bias)
    diagnostics = {}
    if scheme == "qi":
        gen = torch.Generator().manual_seed(10000 + seed)
        kwargs = dict(lam=cfg["qi_lambda"], centers_per_dir=int(math.sqrt(width)), generator=gen)
        diagnostics["first"] = qi_ridge_init_layer_(model.fc1, train_x, **kwargs)
        with torch.no_grad():
            hidden = model.hidden1(train_x[:4096])
        diagnostics["second"] = qi_ridge_init_layer_(model.fc2, hidden, **kwargs)
    return model, diagnostics


@torch.no_grad()
def predictions(model, inputs, with_features=False):
    outputs, hidden = {}, {}
    for split, x in inputs.items():
        ys, hs = [], []
        for chunk in x.split(2048):
            h = model.act(model.fc2(model.hidden1(chunk)))
            ys.append(model.fc3(h).numpy())
            if with_features:
                hs.append(h.numpy())
        outputs[split] = np.concatenate(ys)
        if with_features:
            hidden[split] = np.concatenate(hs)
    return outputs, hidden


def run_one(job):
    task, scheme, lr, seed, steps, phase, cfg, width = job
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    path = OUT / "data" / phase / f"{task}_{scheme}_w{width}_lr{lr:g}_seed{seed}_steps{steps}.json"
    identity = dict(task=task, scheme=scheme, lr=lr, seed=seed, steps=steps, width=width, phase=phase, config=cfg)
    if task in cfg.get("task_data_protocols", {}):
        identity["data_protocol"] = cfg["task_data_protocols"][task]
    if path.exists():
        existing = json.loads(path.read_text())
        if same_identity(existing["identity"], identity) and existing.get("complete"):
            return str(path)
        if not same_identity(existing["identity"], identity):
            raise ValueError(f"Refusing to overwrite a different configuration at {path}")
    arrays, metadata = load_data(task, cfg)
    splits = ["train", "val"] if phase == "pilot" else ["train", "val", "test"]
    inputs = {k: torch.from_numpy(arrays[k][0].copy()) for k in splits}
    ys = {k: arrays[k][1] for k in splits}
    target = torch.from_numpy(ys["train"].copy())
    model, init_diagnostics = make_model(metadata["d_in"], width, seed, scheme, inputs["train"], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    flat_steps = int(0.2 * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 if step <= flat_steps else
                        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1., (step-flat_steps)/(steps-flat_steps)))))
    gen = torch.Generator().manual_seed(50000 + seed)
    # Independent batch RNG is identical across initializations, never used by LS.
    batch = cfg["batch_size"]
    eval_steps = sorted(set([0, 1, 10, 30, 100, 300, 1000] + list(range(2000, steps + 1, 1000)) + [steps]))
    solve_steps = set([0, 10, 100, 300, 1000, 3000] + list(range(5000, steps + 1, 5000)) + [steps])
    coef, bias, info = affine_fit(arrays["train"][0], ys["train"], cfg["least_squares_rcond"])
    affine = {k: mse(arrays[k][0] @ coef + bias, ys[k]) for k in splits}
    result = {"identity": identity, "data": metadata, "affine": affine, "affine_diagnostics": info,
              "init_diagnostics": init_diagnostics, "n_parameters": sum(p.numel() for p in model.parameters()),
              "versions": {"torch": torch.__version__, "numpy": np.__version__}, "trace": []}
    start = time.perf_counter()
    for step in range(steps + 1):
        if step in eval_steps:
            do_solve = phase != "pilot" and step in solve_steps
            pred, features = predictions(model, inputs, do_solve)
            row = {"step": step, "lr": opt.param_groups[0]["lr"],
                   "learned": {k: mse(pred[k], ys[k]) for k in splits}}
            if do_solve:
                w, b, diag = affine_fit(features["train"], ys["train"], cfg["least_squares_rcond"])
                row["solved"] = {k: mse(features[k] @ w + b, ys[k]) for k in splits}
                row["solve_diagnostics"] = diag
                assert row["solved"]["train"] <= row["learned"]["train"] + 1e-8, "LS must improve training MSE"
                if step == 0:
                    best, grid = frozen_ridge(features, ys, cfg["feature_ridge_alphas"], cfg["least_squares_rcond"])
                    result["frozen_ridge"] = best
                    result["frozen_ridge_grid"] = grid
            result["trace"].append(row)
            result["elapsed_seconds"] = time.perf_counter() - start
            result["complete"] = step == steps
            save_json(path, result)
        if step == steps:
            break
        idx = torch.randint(len(target), (batch,), generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs["train"][idx]) - target[idx]) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss: {identity}")
        loss.backward()
        opt.step()
        scheduler.step()
    result["best_val_step"] = min(result["trace"], key=lambda r: r["learned"]["val"])["step"]
    save_json(path, result)
    checkpoint = path.with_suffix(".pt")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(), "batch_rng": gen.get_state(), "identity": identity}, checkpoint)
    print(f"DONE {phase} {task} {scheme} lr={lr:g} best_val="
          f"{min(r['learned']['val'] for r in result['trace']):.5g} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return str(path)


def select(cfg):
    choices = {}
    for task in cfg["tasks"]:
        runs = [json.loads(p.read_text()) for p in (OUT / "data/pilot").glob(f"{task}_standard_*.json")]
        runs = [r for r in runs if r.get("complete") and valid_data_identity(r["identity"], cfg)
                and recipe_config(r["identity"]["config"]) == recipe_config(cfg)
                and r["identity"]["steps"] == cfg["pilot_steps"]
                and r["identity"]["width"] == cfg["width"]
                and r["identity"]["lr"] in cfg["learning_rates"]]
        if not runs:
            raise ValueError(f"No completed pilots for {task}")
        best = min(runs, key=lambda r: min(q["learned"]["val"] for q in r["trace"]))
        choices[task] = {"lr": best["identity"]["lr"], "width": best["identity"]["width"],
                         "pilot_best_val": min(q["learned"]["val"] for q in best["trace"])}
    save_json(OUT / "selected_recipe.json", choices)
    return choices


def plot_pilots(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    # Pilot panel displays validation only; no held-out test selection.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for ax, task in zip(axes.flat, cfg["tasks"]):
        runs = [json.loads(p.read_text()) for p in (OUT / "data/pilot").glob(f"{task}_standard_*.json")]
        runs = [r for r in runs if r.get("complete") and valid_data_identity(r["identity"], cfg)
                and recipe_config(r["identity"]["config"]) == recipe_config(cfg)
                and r["identity"]["steps"] == cfg["pilot_steps"]
                and r["identity"]["width"] == cfg["width"]
                and r["identity"]["lr"] in cfg["learning_rates"]]
        for r in sorted(runs, key=lambda r: r["identity"]["lr"]):
            trace = r["trace"]
            ax.plot([q["step"] for q in trace], [q["learned"]["val"] for q in trace],
                    label=f"LR {r['identity']['lr']:g}")
        ax.axhline(runs[0]["affine"]["val"], color="#666666", ls=":", label="Linear regression")
        ax.set_title(f"{TITLES[task]}\nwidth {cfg['width']} + {cfg['width']}", fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(0, cfg["pilot_steps"])
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Validation MSE (standardized target)")
        ax.grid(alpha=.15, which="both")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.94), ncol=4, frameon=False)
    fig.suptitle("Learning-rate pilots · standard initialization · one seed", y=.995, fontsize=15)
    fig.tight_layout(rect=(0,0,1,.88), h_pad=2.2)
    fig.savefig(figs / "pilot_learning_rates.png", dpi=180)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlim(1, cfg["pilot_steps"])
    fig.text(.5, .008, "Step 0 is omitted on the logarithmic step axis.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0,.025,1,.88), h_pad=2.2)
    fig.savefig(figs / "pilot_learning_rates_loglog.png", dpi=180)
    for ax in axes.flat:
        ax.set_ylim(1e-3, 1)
        for line in list(ax.lines):
            if line.get_label() == "Linear regression":
                continue
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            clipped = (x > 0) & (y > 1)
            ax.plot(x[clipped], np.full(clipped.sum(), .96), marker="^", ls="", ms=4,
                    color=line.get_color())
    fig.texts[-1].set_text("Step 0 omitted. Triangles mark sampled values above the MSE ceiling of 1.")
    fig.savefig(figs / "pilot_learning_rates_loglog_capped.png", dpi=180)
    plt.close(fig)


def plot(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_pilots(cfg)
    figs = OUT / "figures"
    colors = {"standard": "#1f77b4", "qi": "#e58320"}
    choices = select(cfg)
    all_runs = [json.loads(p.read_text()) for p in (OUT / "data/compare").glob("*.json")]
    all_runs = [r for r in all_runs if r.get("complete") and valid_data_identity(r["identity"], cfg)
                and recipe_config(r["identity"]["config"]) == recipe_config(cfg)
                and r["identity"]["lr"] == choices[r["identity"]["task"]]["lr"]
                and r["identity"]["width"] == choices[r["identity"]["task"]]["width"]
                and r["identity"]["steps"] == cfg["comparison_steps"]
                and r["identity"]["seed"] in cfg["comparison_seeds"]]
    if not all_runs:
        return
    expected = {(task, scheme, seed) for task in cfg["tasks"]
                for scheme in ("standard", "qi") for seed in cfg["comparison_seeds"]}
    actual = {(r["identity"]["task"], r["identity"]["scheme"], r["identity"]["seed"]) for r in all_runs}
    if actual != expected:
        raise ValueError(f"Comparison incomplete: {len(actual)}/{len(expected)} completed runs")
    for split in ("train", "val", "test"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
        for ax, task in zip(axes.flat, cfg["tasks"]):
            for scheme in ("standard", "qi"):
                runs = [r for r in all_runs if r["identity"]["task"] == task and r["identity"]["scheme"] == scheme]
                if not runs:
                    continue
                for field, style in (("learned", "-"), ("solved", ":")):
                    traces = [[q for q in r["trace"] if field in q] for r in runs]
                    x = [q["step"] for q in traces[0]]
                    y = np.array([[q[field][split] for q in trace] for trace in traces])
                    ax.plot(x, y.mean(0), color=colors[scheme], ls=style, lw=2)
                    if len(y) > 1:
                        ax.fill_between(x, y.min(0), y.max(0), color=colors[scheme], alpha=.10)
            task_runs = [r for r in all_runs if r["identity"]["task"] == task]
            if task_runs:
                ax.axhline(task_runs[0]["affine"][split], color="#666666", ls=":", lw=1.8)
            ax.set_title(f"{TITLES[task]}\nwidth {choices[task]['width']} + {choices[task]['width']}", fontsize=12)
            ax.set_yscale("log")
            ax.set_xlim(0, cfg["comparison_steps"])
            ax.set_xticks(np.linspace(0, cfg["comparison_steps"], 5))
            ax.set_xlabel("Gradient step")
            ax.set_ylabel(f"{split.capitalize()} MSE (standardized target)")
            ax.grid(alpha=.15, which="both")
        handles = [Line2D([], [], color=colors[s], ls=st, lw=2, label=label)
                   for s, st, label in [("standard","-","Standard · trained head"),("qi","-","QI · trained head"),
                                        ("standard",":","Standard · LS head"),("qi",":","QI · LS head")]]
        handles.append(Line2D([], [], color="#666666", ls=":", label="Linear regression"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=5, frameon=False, fontsize=9)
        fig.suptitle(f"Initialization comparison · {split} · mean of {len(cfg['comparison_seeds'])} seeds · ordinary Adam", y=.995, fontsize=15)
        footer = fig.text(.5,.012,"Shading: seed range, not a confidence interval. Same data split for all seeds.", ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0,.035,1,.88), h_pad=2.2)
        fig.savefig(figs / f"comparison_{split}.png", dpi=180)
        for ax in axes.flat:
            ax.set_xscale("log")
            ax.set_xlim(1, cfg["comparison_steps"])
        footer.set_text("Shading: seed range. Step 0 omitted on logarithmic step axis; full initial values are in the linear-step figure.")
        fig.savefig(figs / f"comparison_{split}_loglog.png", dpi=180)
        # A separately labelled view of the trained regime avoids letting huge
        # step-zero LS extrapolation errors hide the later comparison.
        for ax, task in zip(axes.flat, cfg["tasks"]):
            ax.set_xscale("linear")
            ax.set_xlim(3000, cfg["comparison_steps"])
            ax.set_xticks([3000, 5000, 10000, 15000, cfg["comparison_steps"]])
            values = []
            for r in all_runs:
                if r["identity"]["task"] != task:
                    continue
                values.append(r["affine"][split])
                for q in r["trace"]:
                    if q["step"] >= 3000:
                        values.extend(q[field][split] for field in ("learned", "solved") if field in q)
            ax.set_ylim(min(values)/1.2, max(values)*1.2)
        fig.suptitle(f"Trained regime · {split} · mean of {len(cfg['comparison_seeds'])} seeds · ordinary Adam", y=.995, fontsize=15)
        footer.set_text("Detail view: steps 3,000 onward. Shading: seed range. The full-range figures retain all initial LS spikes.")
        fig.savefig(figs / f"comparison_{split}_detail.png", dpi=180)
        plt.close(fig)
    # A separate control figure keeps the requested five-line panels uncluttered.
    fig, axes = plt.subplots(1, 2, figsize=(12,4.8))
    summary = []
    for i, task in enumerate(cfg["tasks"]):
        task_runs = [r for r in all_runs if r["identity"]["task"] == task]
        standard = [r for r in task_runs if r["identity"]["scheme"] == "standard"]
        if not standard or not any(r["identity"]["scheme"] == "qi" for r in task_runs):
            plt.close(fig)
            return  # Main trajectories can be viewed before all pairs finish.
        mean = lambda values: float(np.mean(values))
        record = {"task": task, "affine_test": standard[0]["affine"]["test"],
                  "frozen_ridge_test": mean([r["frozen_ridge"]["test"] for r in standard]),
                  "standard_final_test": mean([r["trace"][-1]["learned"]["test"] for r in standard]),
                  "recipe": choices[task], "arms": {}}
        for scheme in ("standard", "qi"):
            runs = sorted([r for r in task_runs if r["identity"]["scheme"] == scheme],
                          key=lambda r: r["identity"]["seed"])
            best = [min(r["trace"], key=lambda q: q["learned"]["val"]) for r in runs]
            final_test = [r["trace"][-1]["learned"]["test"] for r in runs]
            record["arms"][scheme] = {
                "seeds": [r["identity"]["seed"] for r in runs],
                "final_test": mean(final_test),
                "final_test_by_seed": final_test,
                "final_test_range": [min(final_test), max(final_test)],
                "final_ls_test": mean([r["trace"][-1]["solved"]["test"] for r in runs]),
                "frozen_ridge_test": mean([r["frozen_ridge"]["test"] for r in runs]),
                "best_val_test": mean([q["learned"]["test"] for q in best]),
                "best_val_steps": [q["step"] for q in best]}
        summary.append(record)
        for ax, split in zip(axes, ("val", "test")):
            vals = [mean([r["frozen_ridge"][split] for r in standard]),
                    mean([r["trace"][-1]["learned"][split] for r in standard]),
                    mean([r["trace"][-1]["solved"][split] for r in standard])]
            for j, (v, color) in enumerate(zip(vals, ("#999999", "#1f77b4", "#59a5d8"))):
                ax.bar(i + (j-1)*.24, v/standard[0]["affine"][split], width=.23, color=color)
            ax.set_xticks(range(6), [TITLES[t] for t in cfg["tasks"]], rotation=25, ha="right")
            ax.axhline(1, color="#666666", ls=":")
            ax.set_yscale("log")
            ax.set_ylim(.003, 2)
            ax.set_ylabel(f"{split.capitalize()} MSE / linear-regression MSE")
            ax.set_title(f"{split.capitalize()} · standard initialization")
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color=c,label=l) for c,l in
                        [("#999999","Frozen initial features · validation-tuned ridge"),
                         ("#1f77b4","Trained features · trained head"),("#59a5d8","Trained features · LS head")]],
               loc="upper center", bbox_to_anchor=(.5,1.01), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0,0,1,.89))
    fig.savefig(figs / "feature_learning_controls.png", dpi=180)
    plt.close(fig)
    save_json(OUT / "summary.json", summary)
    plot_long_budget(cfg)


def plot_long_budget(cfg):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    runs = [json.loads(p.read_text()) for p in (OUT / "data/long_budget").glob("*.json")]
    runs = [r for r in runs if r.get("complete") and valid_data_identity(r["identity"], cfg) and
            recipe_config(r["identity"]["config"]) == recipe_config(cfg) and
            r["identity"]["steps"] == cfg["long_budget_steps"]]
    if len(runs) != 2 * len(cfg["long_budget_tasks"]) * len(cfg["long_budget_seeds"]):
        return
    colors = {"standard": "#1f77b4", "qi": "#e58320"}
    summaries = []
    for task in cfg["long_budget_tasks"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
        for ax, split in zip(axes, ("train", "val", "test")):
            for scheme in ("standard", "qi"):
                subset = [r for r in runs if r["identity"]["task"] == task and r["identity"]["scheme"] == scheme]
                for field, style in (("learned", "-"), ("solved", ":")):
                    traces = [[q for q in r["trace"] if field in q and q["step"] > 0] for r in subset]
                    x = [q["step"] for q in traces[0]]
                    y = np.array([[q[field][split] for q in trace] for trace in traces])
                    ax.loglog(x, y.mean(0), color=colors[scheme], ls=style, lw=2)
            ax.axhline(subset[0]["affine"][split], color="#666666", ls=":", lw=1.8)
            ax.set_xlim(1, cfg["long_budget_steps"])
            ax.set_title(f"{TITLES[task]} · {split}\nwidth {cfg['width']} + {cfg['width']}")
            ax.set_xlabel("Gradient step")
            ax.set_ylabel("MSE (standardized target)")
            ax.grid(alpha=.15, which="both")
        handles = [Line2D([], [], color=colors[s], ls=st, lw=2, label=label)
                   for s, st, label in [("standard","-","Standard · trained head"),("qi","-","QI · trained head"),
                                        ("standard",":","Standard · LS head"),("qi",":","QI · LS head")]]
        handles.append(Line2D([], [], color="#666666", ls=":", label="Linear regression"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=5, frameon=False, fontsize=9)
        fig.suptitle(f"Longer-budget check · {cfg['long_budget_steps']:,} steps · seed 0 · fresh training", y=.995)
        fig.text(.5,.01,"Same recipe within each comparison. The 100k budget stretches the flat/cosine schedule; step 0 is omitted.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0,.045,1,.84))
        fig.savefig(OUT / f"figures/{task}_long_budget.png", dpi=180)
        for ax in axes:
            positive_values = [float(y) for line in ax.lines for x, y in zip(line.get_xdata(), line.get_ydata())
                               if x > 0 and y > 0]
            lower = min(1e-3, 10 ** math.floor(math.log10(min(positive_values))))
            ax.set_ylim(lower, 1)
            for line in list(ax.lines):
                x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
                clipped = (x > 0) & (y > 1)
                ax.plot(x[clipped], np.full(clipped.sum(), .96), marker="^", ls="", ms=4,
                        color=line.get_color())
        fig.texts[-1].set_text("Fresh 100k schedule; step 0 omitted. Triangles mark sampled values above the MSE ceiling of 1.")
        fig.savefig(OUT / f"figures/{task}_long_budget_capped.png", dpi=180)
        plt.close(fig)
        for r in runs:
            if r["identity"]["task"] != task:
                continue
            i = r["identity"]
            short_file = OUT / "data/compare" / f"{task}_{i['scheme']}_w{i['width']}_lr{i['lr']:g}_seed{i['seed']}_steps{cfg['comparison_steps']}.json"
            short = json.loads(short_file.read_text())
            best = min(r["trace"], key=lambda q: q["learned"]["val"])
            summaries.append({"task": task, "scheme": i["scheme"], "seed": i["seed"],
                              "short_budget_steps": cfg["comparison_steps"], "long_budget_steps": i["steps"],
                              "short_final": short["trace"][-1], "long_final": r["trace"][-1],
                              "long_best_validation": best})
    save_json(OUT / "long_budget_summary.json", summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["pilot", "compare", "long-budget", "plot", "smoke"])
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    cfg = config()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.yaml").write_text((HERE / "config.yaml").read_text())
    if args.phase == "plot":
        plot(cfg)
        return
    if args.phase == "smoke":
        jobs = [("airfoil", scheme, .001, 0, 3, "smoke", cfg, 32) for scheme in ("standard", "qi")]
    elif args.phase == "pilot":
        jobs = [(task, "standard", lr, 0, cfg["pilot_steps"], "pilot", cfg, cfg["width"])
                for task in cfg["tasks"] for lr in cfg["learning_rates"]]
    elif args.phase == "long-budget":
        recipes = select(cfg)
        jobs = [(task, scheme, recipes[task]["lr"], seed, cfg["long_budget_steps"], "long_budget", cfg, recipes[task]["width"])
                for task in cfg["long_budget_tasks"] for seed in cfg["long_budget_seeds"] for scheme in ("standard", "qi")]
    else:
        recipes = select(cfg)
        jobs = [(task, scheme, recipes[task]["lr"], seed, cfg["comparison_steps"], "compare", cfg, recipes[task]["width"])
                for task in cfg["tasks"] for seed in cfg["comparison_seeds"] for scheme in ("standard", "qi")]
    assert 0 <= args.shard < args.shards
    for job in jobs[args.shard::args.shards]:
        run_one(job)
    if args.shards == 1 and args.phase in ("pilot", "compare"):
        plot(cfg)


if __name__ == "__main__":
    main()
