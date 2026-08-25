"""expD17 -- does the geometry even move? (gradient-death / geometry-motion test)

Plain Adam only, one hidden layer everywhere, readout initialized to ZERO in
every arm. Two inits per problem (standard vs QI-family), 4 problem classes x
3 problems each:

  interp1d      sine / runge / sine_8pi, N=128 (W=269). qi = exact 1-D
                construction geometry (uniform grid + halo, gamma = lambda*/h,
                b = -gamma*c, expD16's procedure); standard = Glorot.
  interp2d      gauss_bump / sine2d / mixed2d on the unit disk, N=576 ridges.
                qi = expE01 radon_tensor at that cell's best lambda
                (0.26 / 0.08 / 0.12); standard = Glorot on Linear(2, N).
  pinn_inverse  three manufactured nonlinear 2-D inverse problems on
                [-1,1]^2, each with ONE learnable scalar PDE parameter
                (log-parameterized, started 10x off) trained jointly:
                  burgers_nu   u*u_x + u*u_y = nu*(u_xx+u_yy) + f,  nu* = 0.1
                  bratu_lam    Lap u + lam*exp(u) = f,              lam* = 1.0
                  allencahn_k  Lap u + k*(u - u^3) = f,             k* = 5.0
                Loss = mean sq PDE residual (collocation) + boundary data fit
                + sparse interior data fit (equal weights). qi = radon_tensor
                (512 ridges, lambda=0.15); standard = Glorot.
  tabular       superconductivity / sarcos / parkinsons from the expF04 all20
                cache, width 256, tanh. qi = the all20 best arm `scaled_psqrt`
                (qi_ridge_init_layer_, centers_per_dir=16, uniform_centers=False);
                standard = PyTorch default Linear init.

Geometry vector g_i = first-layer weight + bias, readout excluded (the PDE
scalar is logged separately, never in g). Tracked EVERY iteration:
  drift[i] = ||g_i - g_0|| / ||g_0||      (distance from init)
  step[i]  = ||g_i - g_{i-1}|| / ||g_0||  (per-iteration motion)

Extra instrumentation per run (PROGRAM_FRAMING section 7.1): the lstsq probes.
pre_probe = eval rel L2 of a truncated-SVD readout solve on the INIT geometry;
post_probe = same on the FINAL geometry (trained readout thrown away). For the
PINN class the probe solves the data-fit block only (boundary + interior data
rows) -- a caveat, stated in the writeup.

Usage:
    uv run --extra dev python experiments/expD17_geometry_motion/run.py                # all classes
    uv run --extra dev python experiments/expD17_geometry_motion/run.py --classes interp1d,tabular
    uv run --extra dev python experiments/expD17_geometry_motion/run.py --plot         # figures only
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
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config.schema import ModelConfig
from src.construction.qi_mpmath import default_halo
from src.construction.readout import solve_readout_with_bias
from src.data.targets import get_target
from src.data.targets2d import get_target_2d
from src.data.sampling2d import disk_uniform, disk_grid
from src.models.mlp import QIMlp

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")                    # fp64 required; MPS is fp32-only

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD17_geometry_motion"
DATA_DIR = RESULTS_DIR / "data"
FIG_DIR = RESULTS_DIR / "figures"

LAMBDA_STAR = 0.25
STEPS = 5000
EVAL_EVERY = 50
ADAM_LR = 3e-3                                  # classes 1-3 (expD16 convention)
TABULAR_LR = 1e-3                               # expF04's lr
ADAM_WARMUP = 100

CLASSES = ["interp1d", "interp2d", "pinn_inverse", "tabular"]
PROBLEMS = {
    "interp1d": ["sine", "runge", "sine_8pi"],
    "interp2d": ["gauss_bump", "sine2d", "mixed2d"],
    "pinn_inverse": ["burgers_nu", "bratu_lam", "allencahn_k"],
    "tabular": ["superconductivity", "sarcos", "parkinsons"],
}
ARMS = ["standard", "qi"]                       # two lines per subplot

# expE01 radon_tensor best-lambda at N=576 (results/checkpoint_E_2d/.../data.json)
LAMBDA_2D = {"gauss_bump": 0.26, "sine2d": 0.08, "mixed2d": 0.12}
N_2D = 576
N_PINN = 512
LAMBDA_PINN = 0.15
W_TABULAR = 256

_HERE = Path(__file__).resolve().parent


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# expE01 geometry builders + expF04 ridge init, loaded by explicit path
_geo2d = _load_by_path("expD17_geometries2d",
                       REPO_ROOT / "experiments" / "expE01_geometry_zoo_2d" / "geometries.py")
_f04model = _load_by_path("expD17_f04model",
                          REPO_ROOT / "experiments" / "expF04_qi_init_real_data" / "model.py")


# ----------------------------- generic model -----------------------------

# Activation is a module-level switch so the whole study (builders, probes,
# figures) runs unchanged under a different nonlinearity. Default tanh keeps
# every existing result bit-identical; the width_scaling_gelu driver flips it.
ACT = "tanh"


def act(z):
    """The hidden nonlinearity, read from the module-level ACT at call time."""
    if ACT == "tanh":
        return torch.tanh(z)
    if ACT == "gelu":
        return torch.nn.functional.gelu(z)
    raise ValueError(f"unknown activation {ACT!r}")


class OneHidden(nn.Module):
    """d_in -> width -> act -> 1, fp64. Readout zeroed at build time."""

    def __init__(self, d_in: int, width: int):
        super().__init__()
        self.inner = nn.Linear(d_in, width, dtype=torch.float64)
        self.readout = nn.Linear(width, 1, dtype=torch.float64)

    def features(self, x):
        return act(self.inner(x))

    def forward(self, x):
        return self.readout(self.features(x))


def zero_readout_(model):
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()


def glorot_(linear: nn.Linear, seed: int):
    g = torch.Generator().manual_seed(seed)
    fan_in, fan_out = linear.in_features, linear.out_features
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        linear.weight.copy_((torch.rand(fan_out, fan_in, generator=g) * 2 - 1) * bound)
        linear.bias.copy_((torch.rand(fan_out, generator=g) * 2 - 1) * bound)


def geom_flat(geom_params):
    return torch.cat([p.detach().reshape(-1) for p in geom_params]).clone()


# ----------------------------- probes -----------------------------

def probe_rel_l2(model, X_fit, y_fit, X_ev, y_ev):
    """Truncated-SVD readout solve on the CURRENT geometry; eval rel L2.
    Does not touch model parameters."""
    with torch.no_grad():
        Phi_fit = model.features(X_fit).cpu().numpy()
        Phi_ev = model.features(X_ev).cpu().numpy()
    v, b, _ = solve_readout_with_bias(Phi_fit, y_fit.cpu().numpy().ravel(), method="svd")
    pred = Phi_ev @ v + b
    y = y_ev.cpu().numpy().ravel()
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


# ----------------------------- class builders -----------------------------
# Each returns dict(model, geom_params, loss_fn, eval_fn, probe_fn, lr, extra)

def build_interp1d(problem: str, arm: str):
    N = 128
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + idx.astype(np.float64) * h
    W = centers.size
    gamma = LAMBDA_STAR / h

    model = QIMlp(ModelConfig(width=W, layer_type="standard")).to(DEVICE)
    inner = model.inner_layer.linear
    if arm == "qi":
        with torch.no_grad():
            inner.weight.copy_(torch.full((W, 1), gamma))
            inner.bias.copy_(torch.tensor(-gamma * centers))
    else:
        glorot_(inner, seed=0)
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()

    t = get_target(problem)
    X_tr = torch.linspace(-1, 1, 2003).reshape(-1, 1)
    X_ev = torch.linspace(-1, 1, 4001).reshape(-1, 1)
    y_tr, y_ev = t.fn(X_tr), t.fn(X_ev)
    y_norm = float(torch.linalg.norm(y_ev))

    def loss_fn():
        return ((model(X_tr) - y_tr) ** 2).mean()

    def eval_fn():
        with torch.no_grad():
            return float(torch.linalg.norm(model(X_ev) - y_ev) / y_norm)

    def probe_fn():
        return probe_rel_l2(model, X_tr, y_tr, X_ev, y_ev)

    return dict(model=model, geom_params=[inner.weight, inner.bias],
                loss_fn=loss_fn, eval_fn=eval_fn, probe_fn=probe_fn,
                lr=ADAM_LR, extra={"W": W, "gamma": gamma})


def build_interp2d(problem: str, arm: str):
    # radon_tensor J*M only approximates the requested N; both arms use the
    # actual ridge count so the two lines share one architecture.
    dirs, offs = _geo2d.build_radon_tensor(N_2D, radius=2.5)
    W = dirs.shape[0]
    model = OneHidden(2, W).to(DEVICE)
    lam = LAMBDA_2D[problem]
    h_ref = 2.8 / math.sqrt(N_2D)
    gamma = lam / h_ref
    if arm == "qi":
        with torch.no_grad():
            model.inner.weight.copy_(gamma * torch.tensor(dirs))
            model.inner.bias.copy_(-gamma * torch.tensor(offs))
    else:
        glorot_(model.inner, seed=0)
    zero_readout_(model)

    t = get_target_2d(problem)
    X_tr_np = disk_uniform(8000, radius=1.0, seed=0)
    X_ev_np = disk_grid(120, radius=1.0)
    X_tr = torch.tensor(X_tr_np)
    X_ev = torch.tensor(X_ev_np)
    y_tr = torch.tensor(t.fn_numpy(X_tr_np)).reshape(-1, 1)
    y_ev = torch.tensor(t.fn_numpy(X_ev_np)).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_ev))

    def loss_fn():
        return ((model(X_tr) - y_tr) ** 2).mean()

    def eval_fn():
        with torch.no_grad():
            return float(torch.linalg.norm(model(X_ev) - y_ev) / y_norm)

    def probe_fn():
        return probe_rel_l2(model, X_tr, y_tr, X_ev, y_ev)

    return dict(model=model, geom_params=[model.inner.weight, model.inner.bias],
                loss_fn=loss_fn, eval_fn=eval_fn, probe_fn=probe_fn,
                lr=ADAM_LR, extra={"W": W, "gamma": gamma, "lambda": lam})


# --- PINN inverse problems (manufactured) ---

def _ustar_and_f(problem):
    """Manufactured solution u*, its torch evaluator, and the forcing f
    computed with the TRUE parameter. Returns (u_fn, f_fn, p_true)."""
    pi = math.pi
    if problem == "burgers_nu":
        p_true = 0.1

        def u_fn(X):
            return torch.sin(pi * X[:, 0]) * torch.cos(pi * X[:, 1])

        def f_fn(X):
            x, y = X[:, 0], X[:, 1]
            u = torch.sin(pi * x) * torch.cos(pi * y)
            u_x = pi * torch.cos(pi * x) * torch.cos(pi * y)
            u_y = -pi * torch.sin(pi * x) * torch.sin(pi * y)
            lap = -2 * pi ** 2 * u
            return u * u_x + u * u_y - p_true * lap
    elif problem == "bratu_lam":
        p_true = 1.0

        def u_fn(X):
            x, y = X[:, 0], X[:, 1]
            return (1 - x ** 2) * (1 - y ** 2)

        def f_fn(X):
            x, y = X[:, 0], X[:, 1]
            u = (1 - x ** 2) * (1 - y ** 2)
            lap = -2 * (1 - y ** 2) - 2 * (1 - x ** 2)
            return lap + p_true * torch.exp(u)
    elif problem == "allencahn_k":
        p_true = 5.0

        def u_fn(X):
            return torch.sin(pi * X[:, 0]) * torch.sin(pi * X[:, 1])

        def f_fn(X):
            u = torch.sin(pi * X[:, 0]) * torch.sin(pi * X[:, 1])
            lap = -2 * pi ** 2 * u
            return lap + p_true * (u - u ** 3)
    else:
        raise ValueError(problem)
    return u_fn, f_fn, p_true


class PinnModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = OneHidden(2, width)
        self.log_p = nn.Parameter(torch.tensor(0.0))    # set at init

    def forward(self, x):
        return self.net(x)


def _pde_residual(problem, model, X, f_vals):
    X = X.detach().requires_grad_(True)
    u = model.net(X).squeeze(-1)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    u_x, u_y = g[:, 0], g[:, 1]
    u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), X, create_graph=True)[0][:, 1]
    lap = u_xx + u_yy
    p = torch.exp(model.log_p)
    if problem == "burgers_nu":
        return u * u_x + u * u_y - p * lap - f_vals
    if problem == "bratu_lam":
        return lap + p * torch.exp(u) - f_vals
    if problem == "allencahn_k":
        return lap + p * (u - u ** 3) - f_vals
    raise ValueError(problem)


def build_pinn(problem: str, arm: str):
    dirs, offs = _geo2d.build_radon_tensor(N_PINN, radius=2.5)
    W = dirs.shape[0]
    model = PinnModel(W).to(DEVICE)
    u_fn, f_fn, p_true = _ustar_and_f(problem)
    with torch.no_grad():
        model.log_p.fill_(math.log(p_true / 10.0))       # start 10x low

    h_ref = 2.8 / math.sqrt(N_PINN)
    gamma = LAMBDA_PINN / h_ref
    if arm == "qi":
        with torch.no_grad():
            model.net.inner.weight.copy_(gamma * torch.tensor(dirs))
            model.net.inner.bias.copy_(-gamma * torch.tensor(offs))
    else:
        glorot_(model.net.inner, seed=0)
    zero_readout_(model.net)

    # collocation / boundary / sparse data
    s = torch.linspace(-0.95, 0.95, 28)
    Xc = torch.cartesian_prod(s, s)                       # 784 interior
    tb = torch.linspace(-1, 1, 64)
    ones = torch.ones(64)
    Xb = torch.cat([torch.stack([tb, ones], 1), torch.stack([tb, -ones], 1),
                    torch.stack([ones, tb], 1), torch.stack([-ones, tb], 1)])
    g = torch.Generator().manual_seed(1)
    Xd = (torch.rand(100, 2, generator=g) * 2 - 1) * 0.9  # 100 data points
    f_c = f_fn(Xc)
    u_b = u_fn(Xb)
    u_d = u_fn(Xd)

    se = torch.linspace(-1, 1, 61)
    X_ev = torch.cartesian_prod(se, se)
    u_ev = u_fn(X_ev)
    u_norm = float(torch.linalg.norm(u_ev))

    def loss_fn():
        r = _pde_residual(problem, model, Xc, f_c)
        bc = model.net(Xb).squeeze(-1) - u_b
        dd = model.net(Xd).squeeze(-1) - u_d
        return (r ** 2).mean() + (bc ** 2).mean() + (dd ** 2).mean()

    def eval_fn():
        with torch.no_grad():
            pred = model.net(X_ev).squeeze(-1)
            return float(torch.linalg.norm(pred - u_ev) / u_norm)

    # probe: data-fit block only (boundary + interior data rows)
    X_fit = torch.cat([Xb, Xd])
    y_fit = torch.cat([u_b, u_d]).reshape(-1, 1)

    def probe_fn():
        return probe_rel_l2(model.net, X_fit, y_fit, X_ev, u_ev.reshape(-1, 1))

    def param_now():
        return float(torch.exp(model.log_p).detach())

    return dict(model=model, geom_params=[model.net.inner.weight, model.net.inner.bias],
                loss_fn=loss_fn, eval_fn=eval_fn, probe_fn=probe_fn,
                lr=ADAM_LR, extra={"W": W, "gamma": gamma, "p_true": p_true},
                param_cb=param_now)


def build_tabular(problem: str, arm: str):
    cache = REPO_ROOT / "data" / "cache_all20" / f"{problem}.pt"
    t = torch.load(cache, weights_only=False)
    assert t["kind"] == "regr", problem
    X_tr = t["xtr"].to(torch.float64)
    y_tr = t["ytr"].to(torch.float64).reshape(-1, 1)
    X_te = t["xte"].to(torch.float64)
    y_te = t["yte"].to(torch.float64).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_te))

    model = OneHidden(t["d_in"], W_TABULAR).to(DEVICE)   # default PyTorch init
    info = {}
    if arm == "qi":                                       # all20 best arm: scaled_psqrt
        info = _f04model.qi_ridge_init_layer_(
            model.inner, X_tr, lam=LAMBDA_STAR,
            centers_per_dir=int(math.isqrt(W_TABULAR)), uniform_centers=False,
            generator=torch.Generator().manual_seed(0))
    zero_readout_(model)

    def loss_fn():
        return ((model(X_tr) - y_tr) ** 2).mean()

    def eval_fn():
        with torch.no_grad():
            return float(torch.linalg.norm(model(X_te) - y_te) / y_norm)

    def probe_fn():
        return probe_rel_l2(model, X_tr, y_tr, X_te, y_te)

    return dict(model=model, geom_params=[model.inner.weight, model.inner.bias],
                loss_fn=loss_fn, eval_fn=eval_fn, probe_fn=probe_fn,
                lr=TABULAR_LR, extra={"W": W_TABULAR, "d_in": t["d_in"],
                                      "n_train": int(X_tr.shape[0]), **info})


BUILDERS = {"interp1d": build_interp1d, "interp2d": build_interp2d,
            "pinn_inverse": build_pinn, "tabular": build_tabular}


# ----------------------------- train loop -----------------------------

def lr_at(step, peak, warmup, total, end_frac=1e-3):
    end = end_frac * peak
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * prog))


def run_one(cls: str, problem: str, arm: str):
    b = BUILDERS[cls](problem, arm)
    model, geom = b["model"], b["geom_params"]

    g0 = geom_flat(geom)
    n0 = float(torch.linalg.norm(g0))
    assert n0 > 0, "||g_0|| = 0"
    prev = g0.clone()

    pre_probe = b["probe_fn"]()
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])

    drift, stepsz, evals, param_traj = [], [], [], []
    t0 = time.time()
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, b["lr"], ADAM_WARMUP, STEPS)
        opt.zero_grad(set_to_none=True)
        loss = b["loss_fn"]()
        loss.backward()
        opt.step()

        g = geom_flat(geom)
        drift.append(float(torch.linalg.norm(g - g0)) / n0)
        stepsz.append(float(torch.linalg.norm(g - prev)) / n0)
        prev = g

        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            evals.append([step + 1, b["eval_fn"]()])
            if "param_cb" in b:
                param_traj.append([step + 1, b["param_cb"]()])

    post_probe = b["probe_fn"]()
    final_err = b["eval_fn"]()
    row = {"class": cls, "problem": problem, "arm": arm,
           "steps": STEPS, "g0_norm": n0, "n_geom": int(g0.numel()),
           "pre_probe": pre_probe, "post_probe": post_probe,
           "final_err": final_err, "drift": drift, "step_size": stepsz,
           "evals": evals, "wall_s": round(time.time() - t0, 1),
           "extra": b["extra"]}
    if param_traj:
        row["param_traj"] = param_traj
        row["param_true"] = b["extra"]["p_true"]
        row["param_final"] = param_traj[-1][1]
    print(f"  {cls}/{problem}/{arm}: drift_end={drift[-1]:.3e}  "
          f"pre={pre_probe:.2e} post={post_probe:.2e} run={final_err:.2e} "
          f"({row['wall_s']}s)")
    return row


def collect(classes):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        out = DATA_DIR / f"{cls}.jsonl"
        with open(out, "w") as f:
            for problem in PROBLEMS[cls]:
                for arm in ARMS:
                    row = run_one(cls, problem, arm)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
        print(f"Saved {out}")


# ----------------------------- figures -----------------------------

ARM_STYLE = {"standard": ("#1f77b4", "standard init"),
             "qi": ("#d62728", "QI init")}
CLASS_LABEL = {"interp1d": "1-D interp", "interp2d": "2-D interp",
               "pinn_inverse": "2-D inverse PINN", "tabular": "tabular"}


def plot(metric: str, fname: str, ylabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(13, 13.5), sharex=True, sharey="row")
    for i, cls in enumerate(CLASSES):
        path = DATA_DIR / f"{cls}.jsonl"
        rows = [json.loads(l) for l in open(path)] if path.exists() else []
        row_vals = []                       # positive values across the row
        for j, problem in enumerate(PROBLEMS[cls]):
            ax = axes[i][j]
            for arm in ARMS:
                row = next((r for r in rows if r["problem"] == problem
                            and r["arm"] == arm), None)
                if row is None:
                    continue
                y = np.asarray(row[metric])
                it = np.arange(1, len(y) + 1)
                pos = y > 0                 # step-1 drift is exactly 0 (zero
                color, label = ARM_STYLE[arm]  # readout -> zero geometry grad)
                ax.semilogy(it[pos], y[pos], color=color, lw=1.1, label=label)
                row_vals.append(y[pos])
            ax.set_xlim(0, STEPS)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(problem, fontsize=10)
            if i == 3:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel(f"{CLASS_LABEL[cls]}\n{ylabel}")
        if row_vals:
            allv = np.concatenate(row_vals)
            axes[i][0].set_ylim(allv.min() * 0.5, allv.max() * 3.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.suptitle("expD17: geometry motion under plain Adam (readout zero-init); "
                 "rows share the y scale", fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_all():
    plot("drift", "expD17_drift_from_init.png",
         "$\\|g_i - g_0\\| / \\|g_0\\|$")
    plot("step_size", "expD17_step_size.png",
         "$\\|g_i - g_{i-1}\\| / \\|g_0\\|$")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot_all()
    else:
        if "--classes" in sys.argv:
            classes = sys.argv[sys.argv.index("--classes") + 1].split(",")
        else:
            classes = CLASSES
        collect(classes)
        plot_all()
