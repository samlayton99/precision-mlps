"""expD22 -- CD-RGE (zero-order) on the expD16 optimizer-zoo suite.

Same eval suite as expD16, exactly: standard-parameterization tanh MLP
(QIMlp), two inits (qi construction geometry with Glorot readout / xavier
everywhere), 4 targets x N in {64,128,256}, full-batch fp64 MSE on 2003
equispaced points, eval rel L2 on a misaligned 4001-point grid, figures as
4x3 trajectory grids (one per init, fixed y = 1e-16..1e1).

The optimizer under test is CD-RGE from Chaubard's zero_order_rnn repo
(experiments/expD22_cdrge/cdrge.py, a faithful port), replacing expD16's
flawed SPSA datapoint. Variants (lines in the figures) are schedule /
n_perturb / momentum settings chosen by the tuning stages below. Plain Adam
(expD16's protocol) is re-run per cell/seed as the reference line.

Final run: SEEDS per cell, figures show the median with a min-max band.

Usage:
    uv run --extra dev python experiments/expD22_cdrge/run.py --tune stage1
    uv run --extra dev python experiments/expD22_cdrge/run.py --collect --init qi
    uv run --extra dev python experiments/expD22_cdrge/run.py --collect --init qi --targets sine,exp --seeds 0
    uv run --extra dev python experiments/expD22_cdrge/run.py --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config.schema import ModelConfig
from src.construction.qi_mpmath import default_halo
from src.data.targets import get_target
from src.models.mlp import QIMlp

_HERE = Path(__file__).resolve().parent


def _load_local(name):
    spec = importlib.util.spec_from_file_location(f"expD22_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cdrge = _load_local("cdrge")

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD22_cdrge"
DATA_DIR = RESULTS_DIR / "data"
FIG_DIR = RESULTS_DIR / "figures"
TUNE_DIR = RESULTS_DIR / "tuning"

DEVICE = torch.device("cpu")            # fp64 required; MPS is fp32-only
torch.set_default_dtype(torch.float64)

# --- grid (identical to expD16) ---
LAMBDA_STAR = 0.25
TARGETS = ["sine", "exp", "runge", "sine_8pi"]
TARGET_SEEDS = {t: i for i, t in enumerate(TARGETS)}
WIDTHS_N = [64, 128, 256]
INITS = ["qi", "xavier"]
N_TRAIN = 2003
N_EVAL = 4001

ADAM_STEPS = 3000                       # expD16's reference protocol
ADAM_LR = 3e-3
ADAM_WARMUP = 100
LOG_EVERY_ADAM = 10

SEEDS = [0, 1, 2]

# --- CD-RGE variants: finalized by tuning stages 1-5 (see expD22_results.md) ---
VARIANTS = {
    # headline: Adam-style moments on ghat, cosine lr decay (stage 3/4 winner)
    "cdrge_adam_cos": dict(schedule="constant", eps0=1e-3, beta1=0.9,
                           beta2=0.999, adam_lr=0.01, adam_cosine=True,
                           adam_warmup=100, n_perturb=100, max_steps=600),
    # NOTE: a fifth arm, cdrge_adam_anneal (identical + eps halved every 75
    # steps, Sam's eps-anneal hypothesis), was collected on the first ~20
    # cells and produced trajectories identical to cdrge_adam_cos to 3
    # significant figures everywhere; it was dropped from the remaining
    # collection for cost. Its rows remain in the shards; the figures plot
    # the four arms below.
    # author-faithful lr = eps at the largest stable eps (stage 1)
    "cdrge_lr_eq_eps": dict(schedule="constant", eps0=3e-3,
                            n_perturb=150, max_steps=300),
    # author's literal recipe: start eps high, halve both every step
    "cdrge_halve1": dict(schedule="halve_k", halve_every=1, eps0=0.3,
                         n_perturb=300, max_steps=60),
}


# ----------------------------- model & data (expD16 verbatim) -----------------------------

def geometry_for_N(N):
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    idx = np.arange(-halo, N + halo + 1)
    c_uniform = -1.0 + idx.astype(np.float64) * h
    return c_uniform.size, c_uniform, LAMBDA_STAR / h, h, halo


def build_model(init: str, N: int, seed: int) -> QIMlp:
    W, c_uniform, gamma, h, halo = geometry_for_N(N)
    model = QIMlp(ModelConfig(width=W, layer_type="standard")).to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    inner = model.inner_layer.linear
    with torch.no_grad():
        if init == "qi":
            inner.weight.copy_(torch.full((W, 1), gamma))
            inner.bias.copy_(torch.tensor(-gamma * c_uniform))
        elif init == "xavier":
            bound = math.sqrt(6.0 / (1.0 + W))
            inner.weight.copy_((torch.rand(W, 1, generator=g) * 2 - 1) * bound)
            inner.bias.copy_((torch.rand(W, generator=g) * 2 - 1) * bound)
        else:
            raise ValueError(init)
        ob = math.sqrt(6.0 / (W + 1.0))
        model.readout.weight.copy_((torch.rand(1, W, generator=g) * 2 - 1) * ob)
        model.readout.bias.zero_()
    return model


def data_bundle(target: str):
    X_tr = torch.linspace(-1.0, 1.0, N_TRAIN, dtype=torch.float64).reshape(-1, 1)
    X_ev = torch.linspace(-1.0, 1.0, N_EVAL, dtype=torch.float64).reshape(-1, 1)
    t = get_target(target)
    return X_tr, t.fn(X_tr), X_ev, t.fn(X_ev), float(torch.linalg.norm(t.fn(X_ev)))


def get_flat(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model, vec):
    with torch.no_grad():
        torch.nn.utils.vector_to_parameters(vec, model.parameters())


def make_loss_fn(model, X_tr, y_tr):
    def loss_fn(x):
        set_flat(model, x)
        with torch.no_grad():
            return float(((model(X_tr) - y_tr) ** 2).mean())
    return loss_fn


def eval_rel_l2(model, X_ev, y_ev, y_norm):
    with torch.no_grad():
        return float(torch.linalg.norm(model(X_ev) - y_ev) / y_norm)


# ----------------------------- runners -----------------------------

def lr_at(step, peak, warmup, total, end_frac=1e-3):
    end = end_frac * peak
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * prog))


def run_adam(model, bundle, total=ADAM_STEPS):
    """expD16's plain-Adam reference. Returns trace dict."""
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    tr = {"iter": [], "rel_l2": [], "train_mse": []}
    for step in range(total):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, ADAM_LR, ADAM_WARMUP, total)
        opt.zero_grad(set_to_none=True)
        loss = ((model(X_tr) - y_tr) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) % LOG_EVERY_ADAM == 0 or step + 1 == total:
            tr["iter"].append(step + 1)
            tr["rel_l2"].append(eval_rel_l2(model, X_ev, y_ev, y_norm))
            tr["train_mse"].append(float(loss.detach()))
    return tr


def run_cdrge(model, bundle, params, seed):
    """One CD-RGE run with the given variant params. Returns (trace, info)."""
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    loss_fn = make_loss_fn(model, X_tr, y_tr)
    tr = {"iter": [], "rel_l2": [], "train_mse": [], "eps": []}

    def cb(step, x, mean_loss, eps):
        set_flat(model, x)
        tr["iter"].append(step)
        tr["rel_l2"].append(eval_rel_l2(model, X_ev, y_ev, y_norm))
        tr["train_mse"].append(mean_loss)
        tr["eps"].append(eps)

    kw = {k: v for k, v in params.items()}
    x_fin, info = cdrge.cdrge_minimize(get_flat(model), loss_fn, seed=seed,
                                       step_callback=cb, log_every=1, **kw)
    set_flat(model, x_fin)
    info.pop("loss_trace", None)
    return tr, info


# ----------------------------- collect -----------------------------

def run_cell(init, target, N, seed, variants):
    bundle = data_bundle(target)
    model_seed = 100 * seed + TARGET_SEEDS[target]
    rows = []

    t0 = time.time()
    model = build_model(init, N, model_seed)
    tr = run_adam(model, bundle)
    rows.append({"init": init, "target": target, "N": N, "seed": seed,
                 "opt": "adam", "final_rel_l2": tr["rel_l2"][-1],
                 "evals": 2 * ADAM_STEPS, "wall_s": round(time.time() - t0, 1),
                 "trace": tr})

    for name, params in variants.items():
        t0 = time.time()
        model = build_model(init, N, model_seed)
        tr, info = run_cdrge(model, bundle, params, seed=model_seed + 7777)
        rows.append({"init": init, "target": target, "N": N, "seed": seed,
                     "opt": name, "final_rel_l2": tr["rel_l2"][-1],
                     "evals": info["evals"], "final_eps": info["final_eps"],
                     "n_halvings": info.get("n_halvings", 0),
                     "wall_s": round(time.time() - t0, 1),
                     "params": params, "trace": tr})
        print(f"    {init}/{target}/N={N}/s{seed} {name}: "
              f"rel_l2={tr['rel_l2'][-1]:.2e} evals={info['evals']} "
              f"({rows[-1]['wall_s']}s)", flush=True)
    return rows


def collect(inits, targets, widths, seeds, tag=""):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for init in inits:
        out = DATA_DIR / f"trajectories_{init}{tag}.jsonl"
        mode = "a" if tag else "w"
        with open(out, mode) as f:
            for target in targets:
                for N in widths:
                    for seed in seeds:
                        for row in run_cell(init, target, N, seed, VARIANTS):
                            f.write(json.dumps(row) + "\n")
                            f.flush()
        print(f"Saved {out}")


# ----------------------------- tuning -----------------------------

def tune_run(init, target, N, seed, params, label):
    bundle = data_bundle(target)
    model = build_model(init, N, 100 * seed + TARGET_SEEDS[target])
    t0 = time.time()
    tr, info = run_cdrge(model, bundle, params, seed=100 * seed + 7777)
    best = min(tr["rel_l2"])
    print(f"  {init:6s} {target:8s} N={N:3d} {label:42s} "
          f"final={tr['rel_l2'][-1]:.3e} best={best:.3e} "
          f"evals={info['evals']:>7d} eps_end={info['final_eps']:.1e} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"init": init, "target": target, "N": N, "seed": seed,
            "label": label, "params": params, "final_rel_l2": tr["rel_l2"][-1],
            "best_rel_l2": best, "evals": info["evals"],
            "final_eps": info["final_eps"], "trace": tr}


def tune(stage):
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    if stage == "stage1":
        # schedule x eps0 on the cheapest cell (sine, N=64), both inits,
        # np=300 (~m/2), 300-step budget. lr = eps throughout (author advice).
        for init in INITS:
            for eps0 in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
                p = dict(schedule="constant", eps0=eps0,
                         n_perturb=300, max_steps=300)
                rows.append(tune_run(init, "sine", 64, 0, p,
                                     f"const eps={eps0}"))
            for k in [1, 5, 20]:
                p = dict(schedule="halve_k", halve_every=k, eps0=0.3,
                         n_perturb=300, max_steps=300)
                rows.append(tune_run(init, "sine", 64, 0, p,
                                     f"halve/{k} eps0=0.3"))
    elif stage == "stage2":
        # stage-1 verdict: lr = eps is unstable above 3e-3 and ZO-GD-slow at
        # 3e-3. Test the three stabilizers: decoupled lr, momentum, Adam-style.
        for init in INITS:
            for lr_ratio in [3.0, 10.0, 30.0]:
                p = dict(schedule="constant", eps0=1e-3, lr_over_eps=lr_ratio,
                         n_perturb=300, max_steps=300)
                rows.append(tune_run(init, "sine", 64, 0, p,
                                     f"decoupled lr={lr_ratio*1e-3:.0e}"))
            for eps0 in [3e-3, 1e-3]:
                p = dict(schedule="constant", eps0=eps0, beta1=0.9,
                         n_perturb=300, max_steps=300)
                rows.append(tune_run(init, "sine", 64, 0, p,
                                     f"mom.9 eps={eps0}"))
            for adam_lr in [1e-2, 3e-3, 1e-3]:
                p = dict(schedule="constant", eps0=1e-3, beta1=0.9,
                         beta2=0.999, adam_lr=adam_lr,
                         n_perturb=300, max_steps=300)
                rows.append(tune_run(init, "sine", 64, 0, p,
                                     f"adamZO lr={adam_lr}"))
    elif stage == "stage3":
        # adamZO is the live branch. Push lr, lengthen, probe eps/np knobs.
        for init in INITS:
            for adam_lr in [0.01, 0.03]:
                p = dict(schedule="constant", eps0=1e-3, beta1=0.9,
                         beta2=0.999, adam_lr=adam_lr,
                         n_perturb=300, max_steps=1000)
                rows.append(tune_run(init, "sine", 64, 0, p,
                                     f"adamZO lr={adam_lr} T=1000"))
        for np_ in [100, 1000]:
            p = dict(schedule="constant", eps0=1e-3, beta1=0.9,
                     beta2=0.999, adam_lr=0.01,
                     n_perturb=np_, max_steps=1000)
            rows.append(tune_run("qi", "sine", 64, 0, p,
                                 f"adamZO np={np_} T=1000"))
        for eps0 in [1e-2, 1e-4]:
            p = dict(schedule="constant", eps0=eps0, beta1=0.9,
                     beta2=0.999, adam_lr=0.01,
                     n_perturb=300, max_steps=1000)
            rows.append(tune_run("qi", "sine", 64, 0, p,
                                 f"adamZO eps={eps0} T=1000"))
    elif stage == "stage5":
        # Sam's eps-anneal hypothesis: does halving eps during the run break
        # the plateau, or is the barrier iteration count? Identical adamZO-cos
        # runs +- an eps halving schedule (eps only enters the FD estimate).
        base = dict(schedule="halve_k", halve_every=300, eps0=1e-3,
                    eps_floor=1e-7, beta1=0.9, beta2=0.999, adam_lr=0.01,
                    adam_cosine=True, adam_warmup=100,
                    n_perturb=100, max_steps=3000)
        rows.append(tune_run("qi", "sine", 64, 0, base, "adamZO cos +eps-anneal"))
        rows.append(tune_run("qi", "runge", 128, 0, base, "adamZO cos +eps-anneal"))
    elif stage == "stage4":
        # polish + transfer: cosine decay on adam_lr; beta1 push; the tuned
        # config on harder cells (runge N=128 qi, sine_8pi N=64 qi,
        # exp N=256 xavier).
        base = dict(schedule="constant", eps0=1e-3, beta1=0.9, beta2=0.999,
                    adam_lr=0.01, n_perturb=100, max_steps=3000)
        rows.append(tune_run("qi", "sine", 64, 0,
                             dict(base, adam_cosine=True, adam_warmup=100),
                             "adamZO cos T=3000"))
        rows.append(tune_run("qi", "sine", 64, 0, base, "adamZO T=3000"))
        rows.append(tune_run("qi", "sine", 64, 0, dict(base, beta1=0.99),
                             "adamZO b1=.99 T=3000"))
        for init, target, N in [("qi", "runge", 128), ("qi", "sine_8pi", 64),
                                ("xavier", "exp", 256)]:
            rows.append(tune_run(init, target, N, 0,
                                 dict(base, adam_cosine=True, adam_warmup=100),
                                 "adamZO cos T=3000"))
    else:
        raise ValueError(stage)
    out = TUNE_DIR / f"{stage}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {out}")


# ----------------------------- plotting -----------------------------

OPT_STYLE = {
    "adam": ("#7f7f7f", "Adam (reference, 4 passes/it)"),
    "cdrge_adam_cos": ("#d62728", "CD-RGE Adam-style (np=100)"),
    "cdrge_lr_eq_eps": ("#2ca02c", "CD-RGE lr=$\\epsilon$=3e-3 (np=150)"),
    "cdrge_halve1": ("#9467bd", "CD-RGE author recipe (halve/step)"),
}


def _band(rows_v):
    """Median + min/max band across seeds on a shared iteration axis."""
    its = rows_v[0]["trace"]["iter"]
    ys = []
    for r in rows_v:
        y = np.asarray(r["trace"]["rel_l2"])
        it = np.asarray(r["trace"]["iter"])
        ys.append(np.interp(its, it, y))
    Y = np.stack(ys)
    return np.asarray(its), np.median(Y, 0), Y.min(0), Y.max(0)


def plot_panel(init, x_axis="iter"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for path in sorted(DATA_DIR.glob(f"trajectories_{init}*.jsonl")):
        rows += [json.loads(line) for line in open(path)]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TARGETS), len(WIDTHS_N),
                             figsize=(13, 14), sharex=True, sharey=True)
    opt_names = [o for o in OPT_STYLE if any(r["opt"] == o for r in rows)]
    for i, target in enumerate(TARGETS):
        for j, N in enumerate(WIDTHS_N):
            ax = axes[i][j]
            cell_has_data = False
            for opt in opt_names:
                rv = [r for r in rows if r["target"] == target
                      and r["N"] == N and r["opt"] == opt]
                if not rv:
                    continue
                cell_has_data = True
                color, label = OPT_STYLE[opt]
                it, med, lo, hi = _band(rv)
                if x_axis == "evals":
                    scale = (rv[0]["evals"] / it[-1]) if it[-1] else 1
                    it = it * scale
                ax.semilogy(it, med, color=color, lw=1.4, label=label)
                if len(rv) > 1:
                    ax.fill_between(it, lo, hi, color=color, alpha=0.18, lw=0)
            if x_axis == "evals":
                ax.set_xscale("log")
            if not cell_has_data:
                ax.text(0.5, 0.5, "not collected\n(run canceled)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="grey")
            ax.set_ylim(1e-16, 1e1)
            ax.grid(True, alpha=0.3, which="both")
            if i == 0:
                ax.set_title(f"$N={N}$", fontsize=12)
            if i == len(TARGETS) - 1:
                ax.set_xlabel("iteration" if x_axis == "iter"
                              else "full-batch loss evaluations")
            if j == 0:
                ax.set_ylabel(f"{target}\neval rel $L_2$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(opt_names) + 1,
               fontsize=11, bbox_to_anchor=(0.5, 0.985), frameon=False)
    title = ("QI-init geometry (readout Glorot)" if init == "qi"
             else "standard Xavier init (control)")
    n_seeds = len({r["seed"] for r in rows})
    fig.suptitle(f"expD22: CD-RGE zero-order variants, {title} -- "
                 f"median over {n_seeds} seeds, band = min-max; "
                 f"author recipe diverges off-scale from qi init",
                 fontsize=12, y=0.945)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = "" if x_axis == "iter" else "_evals"
    out = FIG_DIR / f"expD22_{init}_init{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", type=str, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--targets", type=str, default=None)
    ap.add_argument("--widths", type=str, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    if args.tune:
        tune(args.tune)
    elif args.collect:
        inits = [args.init] if args.init else INITS
        targets = args.targets.split(",") if args.targets else TARGETS
        widths = [int(w) for w in args.widths.split(",")] if args.widths else WIDTHS_N
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
        collect(inits, targets, widths, seeds, tag=args.tag)
        if not args.tag:
            for init in inits:
                plot_panel(init)
    elif args.plot:
        for init in INITS:
            if any(DATA_DIR.glob(f"trajectories_{init}*.jsonl")):
                plot_panel(init)
