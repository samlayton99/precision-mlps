"""expD16 -- optimizer zoo on the correctly initialized QI geometry.

The paper (QIs_workshop.pdf, Fig. 3) trains end-to-end with Adam and finishes
with a second-order method (headline: Adam -> SSBroyden). This experiment runs
that recipe -- plus NNCG, L-BFGS, and SPSA -- from TWO initializations of the
same standard-parameterization tanh MLP:

  qi:     the paper's construction geometry (uniform grid centers + halo,
          constant gamma = lambda*/h, biases b = -gamma*c), readout at
          standard Glorot init (the optimizer must find the readout).
  xavier: standard Glorot everywhere (the paper's direct-training baseline,
          identical to expD02's `xavier` init).

Optimizers (all train ALL parameters, full-batch fp64 MSE):
  adam            plain Adam, cosine schedule (paper convention).
  adam_ssbroyden  Adam warm start -> SSBroyden (src/training/ssbroyden.py).
  adam_nncg       Adam warm start -> Nystrom Newton-CG (nncg.py, Rathore 2024).
  adam_lbfgs      Adam warm start -> torch LBFGS, strong Wolfe.
  spsa            gradient-free SPSA from init (spsa.py, Spall 1992).

The three finishers branch from ONE shared Adam warm segment per cell, so
their trajectories are directly comparable.

Device: CPU forced (fp64; MPS is fp32-only).

Usage:
    uv run --extra dev python experiments/expD16_optimizer_zoo/run.py            # run grid
    uv run --extra dev python experiments/expD16_optimizer_zoo/run.py --plot     # figures only
    uv run --extra dev python experiments/expD16_optimizer_zoo/run.py --tune-spsa
    uv run --extra dev python experiments/expD16_optimizer_zoo/run.py --init qi  # one init only
"""

from __future__ import annotations

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
from src.training.ssbroyden import ssbroyden_minimize

_HERE = Path(__file__).resolve().parent


def _load_local(name):
    spec = importlib.util.spec_from_file_location(f"expD16_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nncg = _load_local("nncg")
spsa = _load_local("spsa")

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD16_optimizer_zoo"
DATA_DIR = RESULTS_DIR / "data"
FIG_DIR = RESULTS_DIR / "figures"

DEVICE = torch.device("cpu")            # fp64 required; MPS is fp32-only
torch.set_default_dtype(torch.float64)

# --- grid ---
LAMBDA_STAR = 0.25
TARGETS = ["sine", "exp", "runge", "sine_8pi"]      # two easy, two difficult
TARGET_SEEDS = {t: i for i, t in enumerate(TARGETS)}
WIDTHS_N = [64, 128, 256]
INITS = ["qi", "xavier"]
N_TRAIN = 2003                          # prime, > max W (461) -> overdetermined
N_EVAL = 4001                           # prime, misaligned with the train grid

T_TOTAL = 3000                          # shared iteration wall for every optimizer
T_ADAM = 2000                           # warm-start segment for the finishers
T_FINISH = T_TOTAL - T_ADAM
ADAM_LR = 3e-3                          # expD02's better of {1e-3, 3e-3}
ADAM_WARMUP = 100
LOG_EVERY = 10

# SPSA gains chosen by --tune-spsa (see expD16_results.md): a is the sensitive
# knob (a=1e-1 diverges from qi init); c is flat across 1e-2..3e-1, so it is set
# to 1e-1 -- comfortably above fp64 loss-difference noise.
SPSA_GAINS = {"qi": {"a": 0.03, "c": 0.1}, "xavier": {"a": 0.1, "c": 0.1}}

OPT_NAMES = ["adam", "adam_ssbroyden", "adam_nncg", "adam_lbfgs", "spsa"]


# ----------------------------- model & data -----------------------------

def geometry_for_N(N):
    """QI grid+halo geometry at resolution N (expD02's): W, centers, gamma, h, halo."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    idx = np.arange(-halo, N + halo + 1)
    c_uniform = -1.0 + idx.astype(np.float64) * h
    return c_uniform.size, c_uniform, LAMBDA_STAR / h, h, halo


def build_model(init: str, N: int, seed: int) -> QIMlp:
    """Standard-parameterization QIMlp (inner nn.Linear), fp64, on CPU.

    init="qi":     inner weight = gamma (constant), bias = -gamma*c (uniform
                   grid + halo); readout Glorot (seeded), readout bias 0.
    init="xavier": expD02's TanhMLP init -- inner and readout Glorot uniform,
                   readout bias 0.
    """
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
    y_tr = t.fn(X_tr)
    y_ev = t.fn(X_ev)
    return X_tr, y_tr, X_ev, y_ev, float(torch.linalg.norm(y_ev))


# ----------------------------- flat-vector plumbing -----------------------------

def get_flat(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model, vec):
    with torch.no_grad():
        torch.nn.utils.vector_to_parameters(vec, model.parameters())


def make_oracles(model, X_tr, y_tr):
    """(loss_fn, fg, hvp_at) over the flat parameter vector."""
    params = list(model.parameters())

    def loss_at_current():
        return ((model(X_tr) - y_tr) ** 2).mean()

    def loss_fn(x):
        set_flat(model, x)
        with torch.no_grad():
            return loss_at_current()

    def fg(x):
        set_flat(model, x)
        loss = loss_at_current()
        grads = torch.autograd.grad(loss, params)
        return float(loss), torch.cat([gr.reshape(-1) for gr in grads]).detach()

    def hvp_at(x):
        set_flat(model, x)
        loss = loss_at_current()
        g_graph = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = torch.cat([gr.reshape(-1) for gr in g_graph])

        def hvp(v):
            hv = torch.autograd.grad(torch.dot(g_flat, v), params, retain_graph=True)
            return torch.cat([h.reshape(-1) for h in hv]).detach()

        return hvp

    return loss_fn, fg, hvp_at


def eval_rel_l2(model, X_ev, y_ev, y_norm):
    with torch.no_grad():
        pred = model(X_ev)
        return float(torch.linalg.norm(pred - y_ev) / y_norm)


# ----------------------------- optimizer runners -----------------------------
# Each returns a trace: list of [iteration, eval_rel_l2, train_mse].

def lr_at(step, peak, warmup, total, end_frac=1e-3):
    end = end_frac * peak
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * prog))


def run_adam(model, bundle, total, log, offset=0):
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    for step in range(total):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, ADAM_LR, ADAM_WARMUP, total)
        opt.zero_grad(set_to_none=True)
        loss = ((model(X_tr) - y_tr) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) % LOG_EVERY == 0 or step + 1 == total:
            log(offset + step + 1, eval_rel_l2(model, X_ev, y_ev, y_norm),
                float(loss.detach()))


def run_ssbroyden_finisher(model, bundle, total, log, offset):
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    _, fg, _ = make_oracles(model, X_tr, y_tr)

    def cb(step, x, f, gnorm):
        set_flat(model, x)
        log(offset + step, eval_rel_l2(model, X_ev, y_ev, y_norm), f)

    x_star, info = ssbroyden_minimize(get_flat(model), fg, max_steps=total,
                                      step_callback=cb)
    set_flat(model, x_star)
    return info


def run_nncg_finisher(model, bundle, total, log, offset, seed):
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    _, fg, hvp_at = make_oracles(model, X_tr, y_tr)

    def cb(step, x, f, gnorm):
        set_flat(model, x)
        log(offset + step, eval_rel_l2(model, X_ev, y_ev, y_norm), f)

    x_star = nncg.nncg_minimize(get_flat(model), fg, hvp_at, max_steps=total,
                                rank=100, mu=1e-8, precond_freq=25,
                                cg_iters=25, step_callback=cb, seed=seed)
    set_flat(model, x_star)


def run_lbfgs_finisher(model, bundle, total, log, offset):
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1,
                            history_size=100, tolerance_grad=1e-16,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = ((model(X_tr) - y_tr) ** 2).mean()
        loss.backward()
        return loss

    for step in range(total):
        loss = opt.step(closure)
        if not math.isfinite(float(loss)):
            break
        log(offset + step + 1, eval_rel_l2(model, X_ev, y_ev, y_norm), float(loss))


def run_spsa(model, bundle, total, log, gains, seed):
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    loss_fn, _, _ = make_oracles(model, X_tr, y_tr)

    def cb(step, x, f, _gnorm):
        set_flat(model, x)
        log(step, eval_rel_l2(model, X_ev, y_ev, y_norm), f)

    x_final, x_best = spsa.spsa_minimize(get_flat(model), loss_fn, max_steps=total,
                                         a=gains["a"], c=gains["c"],
                                         step_callback=cb, log_every=LOG_EVERY,
                                         seed=seed)
    set_flat(model, x_final)


# ----------------------------- cell protocol -----------------------------

def run_cell(init, target, N):
    """All five optimizer traces for one (init, target, N). Returns rows."""
    seed = TARGET_SEEDS[target]
    bundle = data_bundle(target)
    rows = []
    t_cell = time.time()

    def new_trace():
        tr = {"iter": [], "rel_l2": [], "train_mse": []}

        def log(it, rl2, mse):
            tr["iter"].append(int(it))
            tr["rel_l2"].append(rl2)
            tr["train_mse"].append(mse)

        return tr, log

    def finish_row(opt_name, trace, t0):
        rows.append({"init": init, "target": target, "N": N, "opt": opt_name,
                     "final_rel_l2": trace["rel_l2"][-1] if trace["rel_l2"] else None,
                     "wall_s": round(time.time() - t0, 2), "trace": trace})

    # --- plain Adam, full budget ---
    t0 = time.time()
    model = build_model(init, N, seed)
    trace, log = new_trace()
    run_adam(model, bundle, T_TOTAL, log)
    finish_row("adam", trace, t0)

    # --- shared Adam warm segment ---
    model = build_model(init, N, seed)
    warm_trace, warm_log = new_trace()
    t_warm = time.time()
    run_adam(model, bundle, T_ADAM, warm_log)
    warm_wall = time.time() - t_warm
    theta_warm = get_flat(model)

    for opt_name, runner in [
        ("adam_ssbroyden", lambda m, tr_log: run_ssbroyden_finisher(m, bundle, T_FINISH, tr_log, T_ADAM)),
        ("adam_nncg", lambda m, tr_log: run_nncg_finisher(m, bundle, T_FINISH, tr_log, T_ADAM, seed)),
        ("adam_lbfgs", lambda m, tr_log: run_lbfgs_finisher(m, bundle, T_FINISH, tr_log, T_ADAM)),
    ]:
        t0 = time.time() - warm_wall            # count the shared segment once per row
        set_flat(model, theta_warm)
        trace, log = new_trace()
        trace["iter"] = list(warm_trace["iter"])
        trace["rel_l2"] = list(warm_trace["rel_l2"])
        trace["train_mse"] = list(warm_trace["train_mse"])
        runner(model, log)
        finish_row(opt_name, trace, t0)

    # --- SPSA from init ---
    t0 = time.time()
    model = build_model(init, N, seed)
    trace, log = new_trace()
    run_spsa(model, bundle, T_TOTAL, log, SPSA_GAINS[init], seed)
    finish_row("spsa", trace, t0)

    print(f"  cell ({init},{target},N={N}) done in {time.time()-t_cell:.0f}s: "
          + "  ".join(f"{r['opt']}={r['final_rel_l2']:.2e}" for r in rows))
    return rows


def collect(inits=INITS):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for init in inits:
        out_path = DATA_DIR / f"trajectories_{init}.jsonl"
        with open(out_path, "w") as f:
            for target in TARGETS:
                for N in WIDTHS_N:
                    for row in run_cell(init, target, N):
                        f.write(json.dumps(row) + "\n")
                        f.flush()
        print(f"Saved {out_path}")


# ----------------------------- SPSA tuning -----------------------------

def tune_spsa():
    """Small (a, c) grid on (sine, N=64), both inits, 600 iterations."""
    N, target, iters = 64, "sine", 600
    bundle = data_bundle(target)
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    print("init      a        c        final rel L2")
    for init in INITS:
        results = []
        for a in [3e-3, 1e-2, 3e-2, 1e-1]:
            for c in [1e-2, 1e-1, 3e-1, 1.0]:
                model = build_model(init, N, TARGET_SEEDS[target])
                loss_fn, _, _ = make_oracles(model, X_tr, y_tr)
                x_fin, x_best = spsa.spsa_minimize(get_flat(model), loss_fn,
                                                   max_steps=iters, a=a, c=c, seed=0)
                set_flat(model, x_fin)
                r = eval_rel_l2(model, X_ev, y_ev, y_norm)
                results.append((r, a, c))
                print(f"{init:8s} a={a:.0e}  c={c:.0e}  {r:.3e}")
        best = min(results)
        print(f"--> best for {init}: a={best[1]:.0e} c={best[2]:.0e} ({best[0]:.2e})\n")


# ----------------------------- plotting -----------------------------

OPT_STYLE = {
    "adam": ("#1f77b4", "Adam"),
    "adam_ssbroyden": ("#d62728", "Adam $\\to$ SSBroyden"),
    "adam_nncg": ("#2ca02c", "Adam $\\to$ NNCG"),
    "adam_lbfgs": ("#9467bd", "Adam $\\to$ L-BFGS"),
    "spsa": ("#8c564b", "SPSA"),
}


def plot_panel(init):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in
            open(DATA_DIR / f"trajectories_{init}.jsonl")]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TARGETS), len(WIDTHS_N),
                             figsize=(13, 14), sharex=True, sharey=True)
    for i, target in enumerate(TARGETS):
        for j, N in enumerate(WIDTHS_N):
            ax = axes[i][j]
            for opt in OPT_NAMES:
                row = next((r for r in rows if r["target"] == target
                            and r["N"] == N and r["opt"] == opt), None)
                if row is None:
                    continue
                color, label = OPT_STYLE[opt]
                ax.semilogy(row["trace"]["iter"], row["trace"]["rel_l2"],
                            color=color, lw=1.3, label=label)
            ax.axvline(T_ADAM, color="grey", ls=":", lw=1.0)
            ax.set_xlim(0, T_TOTAL)
            ax.set_ylim(1e-16, 1e1)
            ax.grid(True, alpha=0.3, which="both")
            if i == 0:
                ax.set_title(f"$N={N}$", fontsize=12)
            if i == len(TARGETS) - 1:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel(f"{target}\neval rel $L_2$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.985), frameon=False)
    title = ("QI-init geometry (readout Glorot)" if init == "qi"
             else "standard Xavier init (control)")
    fig.suptitle(f"expD16: optimizer zoo, {title} -- dotted line = Adam$\\to$finisher handoff",
                 fontsize=13, y=0.945)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_DIR / f"expD16_{init}_init.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    if "--tune-spsa" in sys.argv:
        tune_spsa()
    elif "--plot" in sys.argv:
        for init in INITS:
            if (DATA_DIR / f"trajectories_{init}.jsonl").exists():
                plot_panel(init)
    else:
        if "--init" in sys.argv:
            inits = [sys.argv[sys.argv.index("--init") + 1]]
        else:
            inits = INITS
        collect(inits)
        for init in inits:
            plot_panel(init)
