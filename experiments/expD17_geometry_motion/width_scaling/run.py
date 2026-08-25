"""expD17/width_scaling -- geometry motion vs width (3 widths per class).

Same protocol as the parent experiment (plain Adam, readout zero-init, one
hidden layer, 5000 steps, drift tracked every iteration), repeated at three
class-appropriate widths per problem class, both arms:

  interp1d      N in {64, 128, 256}
  interp2d      requested ridges in {288, 576, 1152}  (radon_tensor J*M approx)
  pinn_inverse  requested ridges in {256, 512, 1024}
  tabular       width in {128, 256, 512}  (centers_per_dir = isqrt(W))

Adds the gamma instrumentation for Sam's normalization hypothesis: for BOTH
arms, effective gamma = mean_k ||w_k||_1 over first-layer rows at init (each
row factored w_k = s_k u_k with ||u_k||_1 = 1; s_k is the neuron's magnitude).
Per cell we log g0_norm, end absolute/relative drift, gamma_eff, and the
pre/post lstsq probes.

Usage:
    uv run --extra dev python experiments/expD17_geometry_motion/width_scaling/run.py --job interp1d:0
    uv run --extra dev python experiments/expD17_geometry_motion/width_scaling/run.py --driver   # run all 12 jobs
    uv run --extra dev python experiments/expD17_geometry_motion/width_scaling/run.py --plot     # figures + analysis
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

_PARENT = Path(__file__).resolve().parents[1] / "run.py"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_by_path("expD17_parent", _PARENT)     # parent module: helpers + builders

from src.config.schema import ModelConfig                     # noqa: E402
from src.construction.qi_mpmath import default_halo           # noqa: E402
from src.data.targets import get_target                       # noqa: E402
from src.data.targets2d import get_target_2d                  # noqa: E402
from src.data.sampling2d import disk_uniform, disk_grid       # noqa: E402
from src.models.mlp import QIMlp                              # noqa: E402

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD17_geometry_motion" / "width_scaling")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

# Bandwidths. tanh is the study's default; the gelu driver overrides all four
# (its aliasing-rule optimum is lambda* = 0.707, expC07), scaling each class's
# tuned tanh lambda by the same ratio 0.707/0.25.
LAMBDA_2D = dict(P.LAMBDA_2D)
LAMBDA_PINN = P.LAMBDA_PINN

WIDTHS = {
    "interp1d": [64, 128, 256],
    "interp2d": [288, 576, 1152],
    "pinn_inverse": [256, 512, 1024],
    "tabular": [128, 256, 512],
}
CLASSES = list(WIDTHS)
PROBLEMS = P.PROBLEMS
ARMS = P.ARMS
STEPS = P.STEPS
EVAL_EVERY = 10          # denser than parent (50): the loss figure reads as a curve
LAMBDA_STAR = P.LAMBDA_STAR


def make_snap_schedule(total, dense_until=25, factor=1.2, cap=50):
    """Snapshot steps: every step for the first `dense_until`, then the
    interval grows geometrically (x factor) and is CAPPED at `cap` for the
    rest of the run. Used for gamma/bias/update snapshots AND gif frames, so
    a constant gif framerate plays early training in slow motion."""
    out, s, iv = [], 1, 1.0
    while s <= total:
        out.append(s)
        if s >= dense_until:
            iv = min(iv * factor, cap)
        s += max(1, int(round(iv)))
    if out[-1] != total:
        out.append(total)
    return out


SNAP_SCHEDULE = make_snap_schedule(STEPS)


# --------------------- width-parameterized builders ---------------------

def build_interp1d(problem, arm, N):
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + idx.astype(np.float64) * h
    W = centers.size
    gamma = LAMBDA_STAR / h

    # OneHidden(1, W) is Linear(1,W) -> act -> Linear(W,1): the same network as
    # QIMlp(layer_type="standard") but with the module-level activation switch.
    # Every parameter here is explicitly overwritten, so the tanh path is
    # unchanged by the substitution.
    model = P.OneHidden(1, W)
    inner = model.inner
    if arm == "qi":
        with torch.no_grad():
            inner.weight.copy_(torch.full((W, 1), gamma))
            inner.bias.copy_(torch.tensor(-gamma * centers))
    else:
        P.glorot_(inner, seed=0)
    P.zero_readout_(model)

    t = get_target(problem)
    X_tr = torch.linspace(-1, 1, 2003).reshape(-1, 1)
    X_ev = torch.linspace(-1, 1, 4001).reshape(-1, 1)
    y_tr, y_ev = t.fn(X_tr), t.fn(X_ev)
    y_norm = float(torch.linalg.norm(y_ev))

    return dict(
        model=model, inner=inner, geom_params=[inner.weight, inner.bias],
        loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
        eval_fn=lambda: _eval(model, X_ev, y_ev, y_norm),
        probe_fn=lambda: P.probe_rel_l2(model, X_tr, y_tr, X_ev, y_ev),
        x_ref=X_tr,
        lr=P.ADAM_LR, extra={"W": W, "gamma": gamma, "halo": int(halo),
                             "lambda": LAMBDA_STAR})


def build_interp2d(problem, arm, N_req):
    dirs, offs = P._geo2d.build_radon_tensor(N_req, radius=2.5)
    W = dirs.shape[0]
    model = P.OneHidden(2, W)
    lam = LAMBDA_2D[problem]
    h_ref = 2.8 / math.sqrt(N_req)
    gamma = lam / h_ref
    if arm == "qi":
        with torch.no_grad():
            model.inner.weight.copy_(gamma * torch.tensor(dirs))
            model.inner.bias.copy_(-gamma * torch.tensor(offs))
    else:
        P.glorot_(model.inner, seed=0)
    P.zero_readout_(model)

    t = get_target_2d(problem)
    X_tr_np = disk_uniform(8000, radius=1.0, seed=0)
    X_ev_np = disk_grid(120, radius=1.0)
    X_tr, X_ev = torch.tensor(X_tr_np), torch.tensor(X_ev_np)
    y_tr = torch.tensor(t.fn_numpy(X_tr_np)).reshape(-1, 1)
    y_ev = torch.tensor(t.fn_numpy(X_ev_np)).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_ev))

    return dict(
        model=model, inner=model.inner,
        geom_params=[model.inner.weight, model.inner.bias],
        loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
        eval_fn=lambda: _eval(model, X_ev, y_ev, y_norm),
        probe_fn=lambda: P.probe_rel_l2(model, X_tr, y_tr, X_ev, y_ev),
        x_ref=X_tr,
        lr=P.ADAM_LR, extra={"W": W, "gamma": gamma, "lambda": lam})


def build_pinn(problem, arm, N_req):
    dirs, offs = P._geo2d.build_radon_tensor(N_req, radius=2.5)
    W = dirs.shape[0]
    model = P.PinnModel(W)
    u_fn, f_fn, p_true = P._ustar_and_f(problem)
    with torch.no_grad():
        model.log_p.fill_(math.log(p_true / 10.0))

    h_ref = 2.8 / math.sqrt(N_req)
    gamma = LAMBDA_PINN / h_ref
    if arm == "qi":
        with torch.no_grad():
            model.net.inner.weight.copy_(gamma * torch.tensor(dirs))
            model.net.inner.bias.copy_(-gamma * torch.tensor(offs))
    else:
        P.glorot_(model.net.inner, seed=0)
    P.zero_readout_(model.net)

    s = torch.linspace(-0.95, 0.95, 28)
    Xc = torch.cartesian_prod(s, s)
    tb = torch.linspace(-1, 1, 64)
    ones = torch.ones(64)
    Xb = torch.cat([torch.stack([tb, ones], 1), torch.stack([tb, -ones], 1),
                    torch.stack([ones, tb], 1), torch.stack([-ones, tb], 1)])
    g = torch.Generator().manual_seed(1)
    Xd = (torch.rand(100, 2, generator=g) * 2 - 1) * 0.9
    f_c, u_b, u_d = f_fn(Xc), u_fn(Xb), u_fn(Xd)

    se = torch.linspace(-1, 1, 61)
    X_ev = torch.cartesian_prod(se, se)
    u_ev = u_fn(X_ev)
    u_norm = float(torch.linalg.norm(u_ev))

    def loss_fn():
        r = P._pde_residual(problem, model, Xc, f_c)
        bc = model.net(Xb).squeeze(-1) - u_b
        dd = model.net(Xd).squeeze(-1) - u_d
        return (r ** 2).mean() + (bc ** 2).mean() + (dd ** 2).mean()

    def eval_fn():
        with torch.no_grad():
            pred = model.net(X_ev).squeeze(-1)
            return float(torch.linalg.norm(pred - u_ev) / u_norm)

    X_fit = torch.cat([Xb, Xd])
    y_fit = torch.cat([u_b, u_d]).reshape(-1, 1)

    return dict(
        model=model, inner=model.net.inner,
        geom_params=[model.net.inner.weight, model.net.inner.bias],
        loss_fn=loss_fn, eval_fn=eval_fn,
        probe_fn=lambda: P.probe_rel_l2(model.net, X_fit, y_fit,
                                        X_ev, u_ev.reshape(-1, 1)),
        x_ref=torch.cat([Xc, X_fit]),
        lr=P.ADAM_LR, extra={"W": W, "gamma": gamma, "p_true": p_true,
                             "lambda": LAMBDA_PINN},
        param_cb=lambda: float(torch.exp(model.log_p).detach()))


def build_tabular(problem, arm, W_tab):
    cache = REPO_ROOT / "data" / "cache_all20" / f"{problem}.pt"
    t = torch.load(cache, weights_only=False)
    X_tr = t["xtr"].to(torch.float64)
    y_tr = t["ytr"].to(torch.float64).reshape(-1, 1)
    X_te = t["xte"].to(torch.float64)
    y_te = t["yte"].to(torch.float64).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_te))

    model = P.OneHidden(t["d_in"], W_tab)
    info = {}
    if arm == "qi":
        info = P._f04model.qi_ridge_init_layer_(
            model.inner, X_tr, lam=LAMBDA_STAR,
            centers_per_dir=int(math.isqrt(W_tab)), uniform_centers=False,
            generator=torch.Generator().manual_seed(0))
        info = {k: v for k, v in info.items()
                if isinstance(v, (int, float, str))}
    P.zero_readout_(model)

    return dict(
        model=model, inner=model.inner,
        geom_params=[model.inner.weight, model.inner.bias],
        loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
        eval_fn=lambda: _eval(model, X_te, y_te, y_norm),
        probe_fn=lambda: P.probe_rel_l2(model, X_tr, y_tr, X_te, y_te),
        x_ref=X_tr[:4000],
        lr=P.TABULAR_LR, extra={"W": W_tab, "d_in": t["d_in"],
                                "lambda": LAMBDA_STAR, **info})


def _eval(model, X, y, y_norm):
    with torch.no_grad():
        return float(torch.linalg.norm(model(X) - y) / y_norm)


BUILDERS = {"interp1d": build_interp1d, "interp2d": build_interp2d,
            "pinn_inverse": build_pinn, "tabular": build_tabular}


def gamma_eff_l1(inner) -> float:
    """Sam's prescription: factor each first-layer row w_k = s_k * u_k with
    ||u_k||_1 = 1; the neuron's magnitude is s_k = ||w_k||_1. Effective gamma
    of the arm = mean_k s_k, computed at init (weights only, bias excluded)."""
    with torch.no_grad():
        return float(inner.weight.abs().sum(dim=1).mean())


# ----------------------------- run one cell -----------------------------

def preact_sign_class(inner, x_ref):
    """Classify each neuron by the SIGN its preactivation takes over the data,
    at init. A saturating activation can only kill a neuron whose preactivation
    never changes sign, so this is the exact, class-independent generalization
    of "halo neuron":

      'neg'  max_x z_k(x) < 0   (tanh saturates; gelu -> 0, derivative -> 0)
      'pos'  min_x z_k(x) > 0   (tanh saturates; gelu -> identity, derivative -> 1)
      'span' the preactivation crosses zero somewhere in the data

    Returns a list of labels, one per neuron. The gelu asymmetry prediction is
    that 'neg' neurons die and 'pos' ones stay alive as near-linear units.
    """
    with torch.no_grad():
        Z = inner(x_ref)                                  # [n_points, W]
        zmin = Z.min(dim=0).values
        zmax = Z.max(dim=0).values
    out = []
    for lo, hi in zip(zmin.tolist(), zmax.tolist()):
        out.append("neg" if hi < 0 else ("pos" if lo > 0 else "span"))
    return out


def run_one(cls, problem, arm, width):
    b = BUILDERS[cls](problem, arm, width)
    model, geom = b["model"], b["geom_params"]
    sign_class = preact_sign_class(b["inner"], b["x_ref"])

    g0 = P.geom_flat(geom)
    n0 = float(torch.linalg.norm(g0))
    prev = g0.clone()
    gam_eff = gamma_eff_l1(b["inner"])

    pre_probe = b["probe_fn"]()
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    inner = b["inner"]

    drift, stepsz, drift_abs, step_abs = [], [], [], []
    evals, param_traj = [], []
    snap_steps, snap_gamma, snap_upd, snap_bias = [], [], [], []
    sched = set(SNAP_SCHEDULE)
    t0 = time.time()
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp["lr"] = P.lr_at(step, b["lr"], P.ADAM_WARMUP, STEPS)
        snap = (step + 1) in sched
        if snap:
            with torch.no_grad():
                w_before = inner.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        b["loss_fn"]().backward()
        opt.step()
        g = P.geom_flat(geom)
        da = float(torch.linalg.norm(g - g0))
        sa = float(torch.linalg.norm(g - prev))
        drift.append(da / n0)
        stepsz.append(sa / n0)
        drift_abs.append(da)
        step_abs.append(sa)
        prev = g
        if snap:
            with torch.no_grad():
                rows_now = inner.weight.detach()
                gam = torch.linalg.norm(rows_now, dim=1)          # gamma_k = ||row_k||_2
                upd = torch.linalg.norm(rows_now - w_before, dim=1)
                bia = inner.bias.detach()
            snap_steps.append(step + 1)
            snap_gamma.append([float(f"{v:.5g}") for v in gam.tolist()])
            snap_upd.append([float(f"{v:.5g}") for v in upd.tolist()])
            snap_bias.append([float(f"{v:.5g}") for v in bia.tolist()])
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            evals.append([step + 1, b["eval_fn"]()])
            if "param_cb" in b:
                param_traj.append([step + 1, b["param_cb"]()])

    row = {"class": cls, "problem": problem, "arm": arm, "width": width,
           "steps": STEPS, "g0_norm": n0, "n_geom": int(g0.numel()),
           "gamma_eff_l1": gam_eff,
           "abs_drift_end": drift_abs[-1], "rel_drift_end": drift[-1],
           "pre_probe": pre_probe, "post_probe": b["probe_fn"](),
           "final_err": b["eval_fn"](), "drift": drift, "step_size": stepsz,
           "drift_abs": drift_abs, "step_abs": step_abs,
           "snap_steps": snap_steps, "snap_gamma": snap_gamma,
           "snap_upd": snap_upd, "snap_bias": snap_bias,
           "sign_class": sign_class, "activation": P.ACT,
           "evals": evals, "wall_s": round(time.time() - t0, 1),
           "extra": b["extra"]}
    if param_traj:
        row["param_traj"] = param_traj
        row["param_true"] = b["extra"]["p_true"]
        row["param_final"] = param_traj[-1][1]
    print(f"  {cls}/{problem}/{arm}/w{width}: rel={drift[-1]:.3e} "
          f"abs={row['abs_drift_end']:.3e} g0={n0:.3e} gam={gam_eff:.3e} "
          f"pre={pre_probe:.2e} post={row['post_probe']:.2e} ({row['wall_s']}s)",
          flush=True)
    return row


def run_job(cls, wi):
    width = WIDTHS[cls][wi]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{cls}_w{width}.jsonl"
    with open(out, "w") as f:
        for problem in PROBLEMS[cls]:
            for arm in ARMS:
                f.write(json.dumps(run_one(cls, problem, arm, width)) + "\n")
                f.flush()
    print(f"Saved {out}", flush=True)


# ----------------------------- driver -----------------------------

def driver(max_procs=6, script=None):
    """Fan the 12 jobs out over a process pool. `script` is the entry point the
    workers re-invoke; a driver module that rebinds this module's globals (the
    gelu study) MUST pass its own path, or the workers would run this file's
    configuration and overwrite the wrong results directory."""
    import subprocess
    script = str(script or __file__)
    jobs = [(c, i) for c in CLASSES for i in range(3)]
    # longest first so the pool stays busy
    order = {"pinn_inverse": 0, "tabular": 1, "interp2d": 2, "interp1d": 3}
    jobs.sort(key=lambda j: (order[j[0]], -j[1]))
    running, pending = [], list(jobs)
    while pending or running:
        while pending and len(running) < max_procs:
            cls, wi = pending.pop(0)
            p = subprocess.Popen(
                [sys.executable, script, "--job", f"{cls}:{wi}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            running.append((cls, wi, p))
            print(f"launched {cls}:{wi} (pid {p.pid})", flush=True)
        time.sleep(5)
        still = []
        for cls, wi, p in running:
            if p.poll() is None:
                still.append((cls, wi, p))
            else:
                tail = p.stdout.read()[-2000:]
                print(f"finished {cls}:{wi} rc={p.returncode}\n{tail}", flush=True)
        running = still


# ----------------------------- figures + analysis -----------------------------

BLUES = ["#9ecae1", "#4292c6", "#08519c"]      # lighter -> darker = wider
REDS = ["#fcae91", "#fb6a4a", "#a50f15"]
CLASS_LABEL = {"interp1d": "1-D interp", "interp2d": "2-D interp",
               "pinn_inverse": "2-D inverse PINN", "tabular": "tabular"}


def _rows(cls):
    rows = []
    for w in WIDTHS[cls]:
        p = DATA_DIR / f"{cls}_w{w}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p)]
    return rows


def plot(metric, fname, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(13, 13.5), sharex=True, sharey="row")
    for i, cls in enumerate(CLASSES):
        rows = _rows(cls)
        row_vals = []
        for j, problem in enumerate(PROBLEMS[cls]):
            ax = axes[i][j]
            for wi, w in enumerate(WIDTHS[cls]):
                for arm in ARMS:
                    r = next((r for r in rows if r["problem"] == problem
                              and r["arm"] == arm and r["width"] == w), None)
                    if r is None:
                        continue
                    y = np.asarray(r[metric])
                    it = np.arange(1, len(y) + 1)
                    pos = y > 0
                    color = (BLUES if arm == "standard" else REDS)[wi]
                    ax.semilogy(it[pos], y[pos], color=color, lw=1.0)
                    row_vals.append(y[pos])
            ax.set_xlim(0, STEPS)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(problem, fontsize=10)
            if i == 3:
                ax.set_xlabel("iteration")
            if j == 0:
                ws = WIDTHS[cls]
                ax.set_ylabel(f"{CLASS_LABEL[cls]}  W$\\in${{{ws[0]},{ws[1]},{ws[2]}}}"
                              f"\n{ylabel}")
        if row_vals:
            allv = np.concatenate(row_vals)
            axes[i][0].set_ylim(allv.min() * 0.5, allv.max() * 3.0)

    import matplotlib.lines as mlines
    handles = ([mlines.Line2D([], [], color=BLUES[k], lw=2,
                              label=f"standard ({t})")
                for k, t in enumerate(["small", "mid", "large"])] +
               [mlines.Line2D([], [], color=REDS[k], lw=2,
                              label=f"QI ({t})")
                for k, t in enumerate(["small", "mid", "large"])])
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.suptitle(f"expD17 width scaling [{P.ACT}]: geometry motion under plain "
                 "Adam (readout zero-init); lighter = smaller width; rows share the y scale",
                 fontsize=11, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_loss(fname="expD17w_loss.png"):
    """Third figure: the run's error trajectory, same layout/styling as the
    drift figures. y = eval rel L2 (test-set rel L2 for tabular); the PINN row
    plots the FIELD eval rel L2 (u vs u*), stated on the row label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(13, 13.5), sharex=True, sharey="row")
    for i, cls in enumerate(CLASSES):
        rows = _rows(cls)
        row_vals = []
        for j, problem in enumerate(PROBLEMS[cls]):
            ax = axes[i][j]
            for wi, w in enumerate(WIDTHS[cls]):
                for arm in ARMS:
                    r = next((r for r in rows if r["problem"] == problem
                              and r["arm"] == arm and r["width"] == w), None)
                    if r is None:
                        continue
                    ev = np.asarray(r["evals"])
                    color = (BLUES if arm == "standard" else REDS)[wi]
                    ax.semilogy(ev[:, 0], ev[:, 1], color=color, lw=1.0)
                    row_vals.append(ev[:, 1][ev[:, 1] > 0])
            ax.set_xlim(0, STEPS)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(problem, fontsize=10)
            if i == 3:
                ax.set_xlabel("iteration")
            if j == 0:
                ws = WIDTHS[cls]
                lab = ("field eval rel $L_2$" if cls == "pinn_inverse"
                       else "eval rel $L_2$")
                ax.set_ylabel(f"{CLASS_LABEL[cls]}  W$\\in${{{ws[0]},{ws[1]},{ws[2]}}}"
                              f"\n{lab}")
        if row_vals:
            allv = np.concatenate(row_vals)
            axes[i][0].set_ylim(allv.min() * 0.5, allv.max() * 3.0)

    import matplotlib.lines as mlines
    handles = ([mlines.Line2D([], [], color=BLUES[k], lw=2,
                              label=f"standard ({t})")
                for k, t in enumerate(["small", "mid", "large"])] +
               [mlines.Line2D([], [], color=REDS[k], lw=2,
                              label=f"QI ({t})")
                for k, t in enumerate(["small", "mid", "large"])])
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.suptitle(f"expD17 width scaling [{P.ACT}]: run error under plain Adam "
                 "(readout zero-init); lighter = smaller width; rows share the y scale",
                 fontsize=11, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _snap_values(row, kind):
    """Per-snapshot per-neuron values. kind='gamma': gamma_k = ||w_k||_2.
    kind='center': c_k = -b_k / gamma_k (input units)."""
    G = np.asarray(row["snap_gamma"])
    if kind == "gamma":
        return G
    B = np.asarray(row["snap_bias"])
    return -B / np.maximum(G, 1e-300)


def hist_gifs(kind):
    """Deliverables 4 and 6: per width index, a gif of 4x3 per-neuron
    histograms (QI arm only). kind='gamma' or 'center'. Red dotted vline =
    median; mean |value| in the panel title. Axes FIXED across frames, capped
    at robust quantiles [0.1%, 99.9%] over ALL snapshots of the panel; clip
    fraction annotated when nonzero. Log-x for positive-valued panels whose
    robust spread exceeds 30x. Frames follow SNAP_SCHEDULE; constant gif
    framerate -> early training plays slow-motion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    tag = {"gamma": ("gamma_hist", "$\\gamma_k=\\|w_k\\|_2$", "\\gamma"),
           "center": ("center_hist", "$c_k=-b_k/\\gamma_k$ (input units)", "c")}[kind]
    out_dir = FIG_DIR / tag[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    for wi in range(3):
        cells = {}
        for cls in CLASSES:
            w = WIDTHS[cls][wi]
            for r in _rows(cls):
                if r["arm"] == "qi" and r["width"] == w:
                    cells[(cls, r["problem"])] = r
        if not cells:
            continue
        steps = next(iter(cells.values()))["snap_steps"]
        n_frames = len(steps)

        fig, axes = plt.subplots(4, 3, figsize=(13, 13.5))
        panel = {}
        for i, cls in enumerate(CLASSES):
            for j, problem in enumerate(PROBLEMS[cls]):
                ax = axes[i][j]
                r = cells.get((cls, problem))
                if r is None:
                    ax.axis("off")
                    continue
                V = _snap_values(r, kind)                 # [n_snap, W]
                lo, hi = np.quantile(V, [0.001, 0.999])
                clip = float(np.mean((V < lo) | (V > hi)))
                pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1e-6)
                lo, hi = lo - pad, hi + pad
                logx = lo > 0 and hi / max(lo, 1e-300) > 30
                if logx:
                    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
                    ax.set_xscale("log")
                else:
                    bins = np.linspace(lo, hi, 40)
                ymax = max(np.histogram(np.clip(V[k], lo, hi), bins=bins)[0].max()
                           for k in range(V.shape[0]))
                ax.set_xlim(bins[0], bins[-1])
                ax.set_ylim(0, ymax * 1.15)
                if clip > 0:
                    ax.text(0.99, 0.97, f"clip {clip*100:.2g}%",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=6, color="gray")
                if i == 3:
                    ax.set_xlabel(f"per-neuron {tag[1]}")
                if j == 0:
                    ax.set_ylabel(f"{CLASS_LABEL[cls]} (W={WIDTHS[cls][wi]})\nneurons")
                panel[(i, j)] = (ax, V, bins, lo, hi)

        def draw(fi):
            for (i, j), (ax, V, bins, lo, hi) in panel.items():
                for pat in list(ax.patches):
                    pat.remove()
                for ln in list(ax.lines):
                    ln.remove()
                v = V[min(fi, V.shape[0] - 1)]
                ax.hist(np.clip(v, lo, hi), bins=bins, color="#d62728", alpha=0.75)
                ax.axvline(float(np.clip(np.median(v), lo, hi)),
                           color="red", ls=":", lw=1.6)
                cls = CLASSES[i]
                ax.set_title(f"{PROBLEMS[cls][j]}   mean$|{tag[2]}|$="
                             f"{np.abs(v).mean():.3g}", fontsize=9)
            fig.suptitle(f"expD17 width scaling [{P.ACT}]: per-neuron ${tag[2]}$ histogram, "
                         f"QI-init arm, width index {wi+1} -- "
                         f"step {steps[min(fi, n_frames-1)]}",
                         fontsize=12, y=0.99)
            return []

        anim = FuncAnimation(fig, draw, frames=n_frames, interval=120, blit=False)
        out = out_dir / f"expD17w_{kind}_hist_w{wi+1}.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        plt.close(fig)
        print(f"Saved {out}")


def gamma_vs_update(max_pts=3000, seed=0):
    """Deliverable 5: per width index, a 4x3 figure; each panel scatters
    (gamma_k, ||Adam update row||_2) over all (neuron, snapshot) pairs,
    log-log. Color = training step: viridis (standard arm), autumn (QI arm),
    overlaid. Uniform random subsample to max_pts per arm per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    out_dir = FIG_DIR / "gamma_vs_update"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    CMAP = {"standard": "viridis", "qi": "autumn"}

    for wi in range(3):
        fig, axes = plt.subplots(4, 3, figsize=(13.5, 13.5))
        drew = False
        for i, cls in enumerate(CLASSES):
            w = WIDTHS[cls][wi]
            rows = [r for r in _rows(cls) if r["width"] == w]
            for j, problem in enumerate(PROBLEMS[cls]):
                ax = axes[i][j]
                allG, allU = [], []
                for arm in ARMS:
                    r = next((r for r in rows if r["problem"] == problem
                              and r["arm"] == arm), None)
                    if r is None:
                        continue
                    G = np.asarray(r["snap_gamma"]).ravel()
                    U = np.asarray(r["snap_upd"]).ravel()
                    S = np.repeat(np.asarray(r["snap_steps"], dtype=float),
                                  np.asarray(r["snap_gamma"]).shape[1])
                    keep = (G > 0) & (U > 0)
                    G, U, S = G[keep], U[keep], S[keep]
                    if G.size > max_pts:
                        idx = rng.choice(G.size, max_pts, replace=False)
                        G, U, S = G[idx], U[idx], S[idx]
                    allG.append(G)
                    allU.append(U)
                    ax.scatter(G, U, c=S, cmap=CMAP[arm], s=3, alpha=0.45,
                               vmin=0, vmax=STEPS, linewidths=0, rasterized=True)
                    drew = True
                if allG:
                    # robust fixed axes: [0.1%, 99.9%] over both arms; annotate clip
                    G_all, U_all = np.concatenate(allG), np.concatenate(allU)
                    gl, gh = np.quantile(G_all, [0.001, 0.999])
                    ul, uh = np.quantile(U_all, [0.001, 0.999])
                    clip = float(np.mean((G_all < gl) | (G_all > gh)
                                         | (U_all < ul) | (U_all > uh)))
                    ax.set_xlim(gl * 0.8, gh * 1.25)
                    ax.set_ylim(ul * 0.8, uh * 1.25)
                    if clip > 0:
                        ax.text(0.99, 0.03, f"clip {clip*100:.2g}%",
                                transform=ax.transAxes, ha="right", va="bottom",
                                fontsize=6, color="gray")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3, which="both")
                ax.set_title(problem, fontsize=10)
                if i == 3:
                    ax.set_xlabel("per-neuron $\\gamma_k=\\|w_k\\|_2$")
                if j == 0:
                    ax.set_ylabel(f"{CLASS_LABEL[cls]} (W={w})\n"
                                  "$\\|\\Delta w_k\\|_2$ per Adam step")
        if not drew:
            plt.close(fig)
            continue
        fig.subplots_adjust(right=0.86, top=0.90)
        for arm, pos, lab in [("standard", [0.88, 0.55, 0.02, 0.33],
                               "standard init: training step"),
                              ("qi", [0.88, 0.12, 0.02, 0.33],
                               "QI init: training step")]:
            cax = fig.add_axes(pos)
            sm = ScalarMappable(cmap=CMAP[arm], norm=Normalize(0, STEPS))
            fig.colorbar(sm, cax=cax).set_label(lab, fontsize=9)
        fig.suptitle(f"expD17 width scaling [{P.ACT}]: per-neuron $\\gamma$ vs Adam update "
                     f"magnitude, width index {wi+1} (dark-to-light = early-to-late "
                     "within each colormap)", fontsize=11, y=0.965)
        out = out_dir / f"expD17w_gamma_vs_update_w{wi+1}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


TANH_DATA_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
                 / "expD17_geometry_motion" / "width_scaling" / "data")


def dead_stats(row, early_snaps=30):
    """Fraction of neurons receiving no Adam update, split by preactivation
    sign class. Computed post-hoc from the stored snapshots, so it applies
    unchanged to the tanh study's data.

    Snapshot 0 (step 1) is excluded from every mask: the readout is zeroed at
    init, so the first step's inner-layer gradient is identically zero for
    EVERY neuron in both arms. That one step is an artifact of the protocol,
    not of the geometry.
    """
    U = np.asarray(row["snap_upd"])[1:]                  # [n_snap-1, W]
    if U.size == 0:
        return {}
    early = U[:early_snaps]
    dead_early = (early == 0.0).all(axis=0)
    tiny_early = (early < 1e-10).all(axis=0)
    dead_run = (U == 0.0).all(axis=0)
    tiny_run = (U < 1e-10).all(axis=0)
    # Four nested criteria, reported separately because they differ a lot:
    # "no usable update early" is a much larger set than "never moved at all".
    out = {"n": int(U.shape[1]),
           "dead_early_frac": float(dead_early.mean()),
           "tiny_early_frac": float(tiny_early.mean()),
           "dead_run_frac": float(dead_run.mean()),
           "tiny_run_frac": float(tiny_run.mean())}

    # The sign split needs `sign_class`, recorded from the GELU study onward.
    # Runs without it (the original tanh grid) report totals only -- never a
    # fabricated split.
    if not row.get("sign_class"):
        out["has_sign_split"] = False
        return out
    out["has_sign_split"] = True
    sc = np.asarray(row["sign_class"])
    for lab in ("neg", "pos", "span"):
        m = sc == lab
        out[f"frac_{lab}"] = float(m.mean())
        out[f"dead_of_{lab}"] = float(dead_run[m].mean()) if m.any() else float("nan")
    # of the dead neurons, what sign class were they?
    if dead_run.any():
        for lab in ("neg", "pos", "span"):
            out[f"deadpop_{lab}"] = float((sc[dead_run] == lab).mean())
    return out


def _dead_table(data_dir, label):
    rows = []
    for cls in CLASSES:
        for w in WIDTHS[cls]:
            p = data_dir / f"{cls}_w{w}.jsonl"
            if not p.exists():
                continue
            for r in (json.loads(l) for l in open(p)):
                d = dead_stats(r)
                if d:
                    rows.append({"activation": label, "class": cls,
                                 "problem": r["problem"], "arm": r["arm"],
                                 "width": r["width"], **d})
    return rows


def dead_analysis():
    """Dead-neuron accounting for this study, plus the tanh comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    here = _dead_table(DATA_DIR, P.ACT)
    other = _dead_table(TANH_DATA_DIR, "tanh") if DATA_DIR != TANH_DATA_DIR else []
    allrows = here + other
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "dead_neurons.json", "w") as f:
        json.dump(allrows, f, indent=1)

    hdr = (f"{'cell':>34} {'arm':>8} {'dead%':>7} {'tiny%':>7} "
           f"{'neg%':>6} {'pos%':>6} {'dead|neg':>9} {'dead|pos':>9}")
    print("\n" + hdr)
    for r in here:
        if not r.get("has_sign_split"):
            continue
        print(f"{r['class']+'/'+r['problem']+'/w'+str(r['width']):>34} "
              f"{r['arm']:>8} {100*r['dead_run_frac']:7.1f} "
              f"{100*r['tiny_run_frac']:7.1f} {100*r['frac_neg']:6.1f} "
              f"{100*r['frac_pos']:6.1f} {100*r['dead_of_neg']:9.1f} "
              f"{100*r['dead_of_pos']:9.1f}")

    if not other:
        return
    # Figure: dead fraction vs width, QI arm, tanh vs this activation.
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        for lab, rowset, style in [("tanh", other, dict(color="#1f77b4", marker="o")),
                                   (P.ACT, here, dict(color="#d62728", marker="s"))]:
            for arm, ls in [("qi", "-"), ("standard", "--")]:
                xs, ys = [], []
                for w in WIDTHS[cls]:
                    vals = [r["dead_run_frac"] for r in rowset
                            if r["class"] == cls and r["width"] == w
                            and r["arm"] == arm]
                    if vals:
                        xs.append(w)
                        ys.append(100 * float(np.median(vals)))
                if xs:
                    ax.plot(xs, ys, ls=ls, lw=1.6, ms=6,
                            label=f"{lab} / {arm}", **style)
        ax.set_xscale("log")
        ax.set_ylim(-3, 103)
        ax.grid(True, alpha=0.3)
        ax.set_title(CLASS_LABEL[cls], fontsize=10)
        ax.set_xlabel("width $W$")
        if i == 0:
            ax.set_ylabel("neurons with zero update (%)\nmedian over the 3 problems")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.10), frameon=False)
    out = FIG_DIR / f"expD17w_dead_fraction_{P.ACT}_vs_tanh.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def analyze(fname="expD17w_ratio_scatter.png"):
    """The gamma-hypothesis table + ratio scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = []
    for cls in CLASSES:
        rows = _rows(cls)
        for problem in PROBLEMS[cls]:
            for w in WIDTHS[cls]:
                std = next((r for r in rows if r["problem"] == problem
                            and r["arm"] == "standard" and r["width"] == w), None)
                qi = next((r for r in rows if r["problem"] == problem
                           and r["arm"] == "qi" and r["width"] == w), None)
                if std is None or qi is None:
                    continue
                recs.append({
                    "class": cls, "problem": problem, "width": w,
                    "drift_ratio": std["rel_drift_end"] / qi["rel_drift_end"],
                    "gamma_ratio": qi["gamma_eff_l1"] / std["gamma_eff_l1"],
                    "g0_ratio": qi["g0_norm"] / std["g0_norm"],
                    "abs_ratio": std["abs_drift_end"] / qi["abs_drift_end"],
                    "qi_rel": qi["rel_drift_end"], "std_rel": std["rel_drift_end"],
                    "qi_abs": qi["abs_drift_end"], "std_abs": std["abs_drift_end"],
                    "qi_gam": qi["gamma_eff_l1"], "std_gam": std["gamma_eff_l1"],
                    "qi_g0": qi["g0_norm"], "std_g0": std["g0_norm"],
                    "qi_pre": qi["pre_probe"], "qi_post": qi["post_probe"],
                    "std_pre": std["pre_probe"], "std_post": std["post_probe"]})
    with open(DATA_DIR / "analysis.json", "w") as f:
        json.dump(recs, f, indent=1)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    mark = {"interp1d": "o", "interp2d": "s", "pinn_inverse": "^", "tabular": "D"}
    for cls in CLASSES:
        pts = [r for r in recs if r["class"] == cls]
        ax.loglog([r["gamma_ratio"] for r in pts],
                  [r["drift_ratio"] for r in pts],
                  mark[cls], ms=6, alpha=0.8, label=CLASS_LABEL[cls])
    lims = [0.5, 2000]
    ax.loglog(lims, lims, "k--", lw=1, label="$y=x$")
    ax.set_xlabel("$\\gamma_{qi} / \\gamma_{std}$   (effective, mean $L_1$ row norm)")
    ax.set_ylabel("rel-drift ratio  std / QI")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              borderaxespad=0, fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    print(f"\n{'cell':>34} {'driftR':>9} {'gammaR':>9} {'g0R':>9} {'absR':>7}")
    for r in recs:
        print(f"{r['class']+'/'+r['problem']+'/w'+str(r['width']):>34} "
              f"{r['drift_ratio']:9.3g} {r['gamma_ratio']:9.3g} "
              f"{r['g0_ratio']:9.3g} {r['abs_ratio']:7.3g}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot("drift", "expD17w_drift_from_init.png", "$\\|g_i-g_0\\|/\\|g_0\\|$")
        plot("step_size", "expD17w_step_size.png", "$\\|g_i-g_{i-1}\\|/\\|g_0\\|$")
        plot("drift_abs", "expD17w_drift_from_init_abs.png", "$\\|g_i-g_0\\|$  (absolute)")
        plot("step_abs", "expD17w_step_size_abs.png", "$\\|g_i-g_{i-1}\\|$  (absolute)")
        plot_loss()
        hist_gifs("gamma")
        hist_gifs("center")
        gamma_vs_update()
        analyze()
        dead_analysis()
    elif "--driver" in sys.argv:
        driver()
    elif "--job" in sys.argv:
        cls, wi = sys.argv[sys.argv.index("--job") + 1].split(":")
        run_job(cls, int(wi))
    else:
        driver()
