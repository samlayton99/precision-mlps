"""expD17/norm_factorial -- the normalization factorial on the expD17 protocol.

Same protocol as ../width_scaling (plain Adam, readout zero-init, drift tracked
every iteration, 5000 steps, snapshot schedule dense-at-the-start capped at 50),
but the two arms {standard, QI} become SIX, crossing activation, normalization
and init:

  1  tanh / no-norm    / standard          4  gelu / no-norm    / QI
  2  tanh / no-norm    / QI                5  gelu / rms_center / standard
  3  gelu / no-norm    / standard          6  gelu / rms_center / QI

tanh x rms_center is deliberately absent. expD21 measured it as a wasted run
(0.79x, 32x worst case): tanh's saturated halo columns have RMS exactly 1 but
standard deviation ~1e-14, so centering divides by roundoff and amplifies it to
unit variance. Nothing is learned by re-measuring that.

`rms_center` is h -> (h - mean(h_init)) / std(h_init) with BOTH statistics
frozen at init, i.e. exactly a BatchNorm pinned to its initial statistics. It
carries NO learnable parameters, so every arm has an identical parameter count
(gate test). With the readout zeroed at init it is a pure reparameterization:
the represented function is unchanged and only the gradient geometry differs.

Bandwidths follow expC07's aliasing rule per activation: lambda* = 0.25 (tanh),
0.707 (gelu), every class's tuned tanh lambda scaled by the same 2.828 ratio --
identical to ../width_scaling_gelu. The halo is HELD FIXED at the standard rule
in every arm; the halo question is not asked here and must not be conflated.

Two widths per class (the largest of the width_scaling triple is dropped).
The tabular class is TWO hidden layers -- expD20 measured one hidden layer as
width-saturated on real tabular regression (16x width buys under 5% on all 17
tasks), so the headroom there is depth-headroom. Only the FIRST layer is the
geometry: every drift / gamma / center / dead-neuron metric reads layer 1, the
QI-family init is applied to layer 1 only, and layer 2 keeps standard init.
Tabular also uses the expD20 suite and expD20's preprocessing, whose targets are
standardized on the train split, so a mean-predictor scores 1.0 -- the
variance-normalized metric. The other three classes keep the existing relative
L2 unchanged.

Usage:
    uv run --extra dev python experiments/expD17_geometry_motion/norm_factorial/run.py --driver
    uv run --extra dev python experiments/expD17_geometry_motion/norm_factorial/run.py --job interp1d:0:sine
    uv run --extra dev python experiments/expD17_geometry_motion/norm_factorial/run.py --plot
    uv run --extra dev python experiments/expD17_geometry_motion/norm_factorial/run.py --naval-check
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # register before exec: @dataclass resolves its module via sys.modules,
    # and expD20's evaluate.py defines one at import time.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HERE = Path(__file__).resolve().parent
WS = _load_by_path("expD17w_shared", _HERE.parents[0] / "width_scaling" / "run.py")
P = WS.P                                              # parent module

_D20 = _load_by_path("expD20_datasets",
                     REPO_ROOT / "experiments" / "expD20_tabular_suite" / "datasets.py")
_EV = _load_by_path("expD20_evaluate",
                    REPO_ROOT / "experiments" / "expD20_tabular_suite" / "evaluate.py")

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD17_geometry_motion" / "norm_factorial")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

LAMBDA_TANH = 0.25
LAMBDA_GELU = 0.707                                   # expC07 aliasing rule
LAM_RATIO = LAMBDA_GELU / LAMBDA_TANH                 # 2.828

# ------------------------------- the six arms -------------------------------
# key -> (activation, normalization, init)
ARM_SPEC = {
    "tanh_none_std": ("tanh", "none", "standard"),
    "tanh_none_qi": ("tanh", "none", "qi"),
    "gelu_none_std": ("gelu", "none", "standard"),
    "gelu_none_qi": ("gelu", "none", "qi"),
    "gelu_rmsc_std": ("gelu", "rms_center", "standard"),
    "gelu_rmsc_qi": ("gelu", "rms_center", "qi"),
}
ARMS = list(ARM_SPEC)

WIDTHS = {
    "interp1d": [64, 128],
    "interp2d": [288, 576],
    "pinn_inverse": [256, 512],
    "tabular": [128, 256],
}
CLASSES = list(WIDTHS)

# expD20's recommended parity suite plus naval (Sam: naval in).
TAB_TASKS = ["pol", "kin8nm", "parkinsons", "bike_sharing", "airfoil",
             "sarcos", "naval"]
TAB_PLOT = ["pol", "kin8nm", "naval"]                 # the 4x3 figures' row
# Full-batch fp64 on a 2-layer net is quadratic in width; the cap keeps 84
# tabular runs tractable and is applied identically to every arm.
TAB_CAP = 6000

PROBLEMS = {**P.PROBLEMS, "tabular": TAB_TASKS}
STEPS = WS.STEPS
SNAP_SCHEDULE = WS.SNAP_SCHEDULE


# --------------------------- norm-capable models ---------------------------

class NormOneHidden(P.OneHidden):
    """P.OneHidden plus two FIXED buffers applied to the features.

    With nrm_mu = 0 and nrm_inv = 1 this is bitwise identical to the parent
    ((x - 0)*1 == x exactly in IEEE), so rebinding P.OneHidden to this class
    leaves every no-norm arm numerically unchanged.
    """

    def __init__(self, d_in, width):
        super().__init__(d_in, width)
        self.register_buffer("nrm_mu", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("nrm_inv", torch.ones(width, dtype=torch.float64))

    def raw_features(self, x):
        return P.act(self.inner(x))

    def features(self, x):
        return (self.raw_features(x) - self.nrm_mu) * self.nrm_inv

    # the QI-initialized layer's output, post-normalization: what the column
    # statistics are measured on, and for the 1-layer nets also what the
    # readout sees.
    def geom_features(self, x):
        return self.features(x)


# Rebinding here means every width_scaling builder (interp1d / interp2d /
# pinn, and PinnModel's internal net) produces a norm-capable model with no
# change to those builders.
P.OneHidden = NormOneHidden


class TwoHidden(nn.Module):
    """d_in -> W -> act -> [norm] -> W -> act -> 1  (tabular only).

    Layer 1 (`inner`) is the geometry: the QI-family init and every drift,
    gamma, center and dead-neuron metric read it and nothing else. Layer 2
    keeps standard init in every arm. The normalization is applied to layer 1's
    output, which is the layer carrying the column-scale pathology.
    """

    def __init__(self, d_in, width):
        super().__init__()
        self.inner = nn.Linear(d_in, width, dtype=torch.float64)
        self.hidden2 = nn.Linear(width, width, dtype=torch.float64)
        self.readout = nn.Linear(width, 1, dtype=torch.float64)
        self.register_buffer("nrm_mu", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("nrm_inv", torch.ones(width, dtype=torch.float64))

    def raw_features(self, x):
        return P.act(self.inner(x))

    def geom_features(self, x):
        return (self.raw_features(x) - self.nrm_mu) * self.nrm_inv

    def features(self, x):                 # what the readout sees
        return P.act(self.hidden2(self.geom_features(x)))

    def forward(self, x):
        return self.readout(self.features(x))


@torch.no_grad()
def set_rms_center_(model, x_ref):
    """Freeze the init column statistics of the QI layer into the buffers:
    h -> (h - mean(h_init)) / std(h_init). Identical to expD21's rms_center."""
    A = model.raw_features(x_ref)
    mu = A.mean(dim=0)
    sd = (A - mu).pow(2).mean(dim=0).sqrt()
    sd = torch.where(sd < 1e-300, torch.ones_like(sd), sd)
    model.nrm_mu.copy_(mu)
    model.nrm_inv.copy_(1.0 / sd)


@torch.no_grad()
def col_norm_stats(model, x_ref, sign_class):
    """Column RMS of the QI layer's (post-normalization) output."""
    A = model.geom_features(x_ref)
    cn = A.pow(2).mean(dim=0).sqrt().cpu().numpy()
    sc = np.asarray(sign_class)
    live = cn > 1e-300
    out = {"max_over_min": float(cn.max() / max(cn.min(), 1e-300)),
           "max_over_min_live": (float(cn.max() / max(cn[live].min(), 1e-300))
                                 if live.any() else float("inf")),
           "frac_zero": float((~live).mean())}
    for tag, msk in (("span", sc == "span"), ("pos", sc == "pos"),
                     ("neg", sc == "neg")):
        out[f"mean_{tag}"] = float(cn[msk].mean()) if msk.any() else float("nan")
    return out


# ------------------------------- tabular -------------------------------

_TAB_CACHE = {}


def load_tab(name):
    """expD20 loader + expD20 preprocessing. `prep` standardizes the target on
    the train split, so relative L2 against it is the variance-normalized
    metric: a mean-predictor scores 1.0."""
    if name in _TAB_CACHE:
        return _TAB_CACHE[name]
    if name in _D20.INCUMBENTS:
        raw = _D20.load_incumbent(name)
    else:
        raw = _D20.CANDIDATES[name]()
    Xtr, ytr, Xte, yte = _EV.prep(*raw)
    if len(Xtr) > TAB_CAP:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Xtr), TAB_CAP, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    _TAB_CACHE[name] = (Xtr, ytr, Xte, yte)
    return _TAB_CACHE[name]


def build_tabular(problem, init, W_tab):
    Xtr, ytr, Xte, yte = load_tab(problem)
    X_tr = torch.tensor(Xtr)
    y_tr = torch.tensor(ytr).reshape(-1, 1)
    X_te = torch.tensor(Xte)
    y_te = torch.tensor(yte).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_te))
    d_in = X_tr.shape[1]

    model = TwoHidden(d_in, W_tab)
    info = {}
    if init == "qi":
        info = P._f04model.qi_ridge_init_layer_(
            model.inner, X_tr, lam=WS.LAMBDA_STAR,
            centers_per_dir=int(math.isqrt(W_tab)), uniform_centers=False,
            generator=torch.Generator().manual_seed(0))
        info = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
    P.zero_readout_(model)

    return dict(
        model=model, inner=model.inner,
        geom_params=[model.inner.weight, model.inner.bias],
        loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
        eval_fn=lambda: WS._eval(model, X_te, y_te, y_norm),
        probe_fn=lambda: P.probe_rel_l2(model, X_tr, y_tr, X_te, y_te),
        x_ref=X_tr[:4000],
        lr=P.TABULAR_LR,
        extra={"W": W_tab, "d_in": d_in, "n_train": int(X_tr.shape[0]),
               "lambda": WS.LAMBDA_STAR, "depth": 2, **info})


BASE_BUILDERS = {"interp1d": WS.build_interp1d, "interp2d": WS.build_interp2d,
                 "pinn_inverse": WS.build_pinn, "tabular": build_tabular}


def set_activation(act):
    """Activation plus the bandwidths it forces (expC07 aliasing rule)."""
    P.ACT = act
    r = 1.0 if act == "tanh" else LAM_RATIO
    WS.LAMBDA_STAR = round(LAMBDA_TANH * r, 4)
    WS.LAMBDA_2D = {k: round(v * r, 4) for k, v in P.LAMBDA_2D.items()}
    WS.LAMBDA_PINN = round(P.LAMBDA_PINN * r, 4)


def build(cls, problem, arm_key, width):
    act, norm, init = ARM_SPEC[arm_key]
    set_activation(act)
    b = BASE_BUILDERS[cls](problem, init, width)
    if norm == "rms_center":
        set_rms_center_(b["model"] if cls != "pinn_inverse" else b["model"].net,
                        b["x_ref"])
    b["extra"].update({"activation": act, "norm": norm, "init": init})
    return b


# ----------------------------- run one cell -----------------------------

def run_one(cls, problem, arm_key, width):
    act, norm, init = ARM_SPEC[arm_key]
    b = build(cls, problem, arm_key, width)
    model, geom, inner = b["model"], b["geom_params"], b["inner"]
    normed = b["model"].net if cls == "pinn_inverse" else b["model"]

    sign_class = WS.preact_sign_class(inner, b["x_ref"])
    cn_init = col_norm_stats(normed, b["x_ref"], sign_class)
    n_params = int(sum(p.numel() for p in model.parameters()))

    g0 = P.geom_flat(geom)
    n0 = float(torch.linalg.norm(g0))
    prev = g0.clone()
    gam_eff = WS.gamma_eff_l1(inner)

    pre_probe = b["probe_fn"]()
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])

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
                gam = torch.linalg.norm(rows_now, dim=1)
                upd = torch.linalg.norm(rows_now - w_before, dim=1)
                bia = inner.bias.detach()
            snap_steps.append(step + 1)
            snap_gamma.append([float(f"{v:.5g}") for v in gam.tolist()])
            snap_upd.append([float(f"{v:.5g}") for v in upd.tolist()])
            snap_bias.append([float(f"{v:.5g}") for v in bia.tolist()])
        if (step + 1) % WS.EVAL_EVERY == 0 or step == 0:
            evals.append([step + 1, b["eval_fn"]()])
            if "param_cb" in b:
                param_traj.append([step + 1, b["param_cb"]()])

    row = {"class": cls, "problem": problem, "arm": arm_key,
           "activation": act, "norm": norm, "init": init,
           "width": width, "steps": STEPS, "g0_norm": n0,
           "n_geom": int(g0.numel()), "n_params": n_params,
           "gamma_eff_l1": gam_eff,
           "abs_drift_end": drift_abs[-1], "rel_drift_end": drift[-1],
           "pre_probe": pre_probe, "post_probe": b["probe_fn"](),
           "final_err": b["eval_fn"](), "drift": drift, "step_size": stepsz,
           "drift_abs": drift_abs, "step_abs": step_abs,
           "snap_steps": snap_steps, "snap_gamma": snap_gamma,
           "snap_upd": snap_upd, "snap_bias": snap_bias,
           "sign_class": sign_class,
           "colnorm_init": cn_init,
           "colnorm_final": col_norm_stats(normed, b["x_ref"], sign_class),
           "evals": evals, "wall_s": round(time.time() - t0, 1),
           "extra": b["extra"]}
    if param_traj:
        row["param_traj"] = param_traj
        row["param_true"] = b["extra"]["p_true"]
        row["param_final"] = param_traj[-1][1]
    print(f"  {cls}/{problem}/{arm_key}/w{width}: rel={drift[-1]:.3e} "
          f"abs={row['abs_drift_end']:.3e} pre={pre_probe:.2e} "
          f"post={row['post_probe']:.2e} run={row['final_err']:.3e} "
          f"({row['wall_s']}s)", flush=True)
    return row


def run_job(cls, wi, problem):
    width = WIDTHS[cls][wi]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{cls}_w{width}_{problem}.jsonl"
    with open(out, "w") as f:
        for arm in ARMS:
            f.write(json.dumps(run_one(cls, problem, arm, width)) + "\n")
            f.flush()
    print(f"Saved {out}", flush=True)


def driver(max_procs=6):
    import subprocess
    jobs = [(c, i, p) for c in CLASSES for i in range(2) for p in PROBLEMS[c]]
    order = {"pinn_inverse": 0, "tabular": 1, "interp2d": 2, "interp1d": 3}
    jobs.sort(key=lambda j: (order[j[0]], -j[1]))
    running, pending = [], list(jobs)
    done = 0
    while pending or running:
        while pending and len(running) < max_procs:
            cls, wi, prob = pending.pop(0)
            p = subprocess.Popen(
                [sys.executable, str(__file__), "--job", f"{cls}:{wi}:{prob}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            running.append((cls, wi, prob, p))
            print(f"launched {cls}:{wi}:{prob} (pid {p.pid})", flush=True)
        time.sleep(5)
        still = []
        for cls, wi, prob, p in running:
            if p.poll() is None:
                still.append((cls, wi, prob, p))
            else:
                done += 1
                tail = p.stdout.read()[-1500:]
                print(f"[{done}/{len(jobs)}] finished {cls}:{wi}:{prob} "
                      f"rc={p.returncode}\n{tail}", flush=True)
        running = still


# ----------------------------- figures -----------------------------
# colour family = activation + normalization; lightness = init
# (light = standard, dark = QI); linestyle = width (dashed = small).

ARM_COLOR = {
    "tanh_none_std": "#9ecae1", "tanh_none_qi": "#08519c",
    "gelu_none_std": "#fcae91", "gelu_none_qi": "#a50f15",
    "gelu_rmsc_std": "#a1d99b", "gelu_rmsc_qi": "#006d2c",
}
ARM_LABEL = {
    "tanh_none_std": "tanh / no-norm / std", "tanh_none_qi": "tanh / no-norm / QI",
    "gelu_none_std": "gelu / no-norm / std", "gelu_none_qi": "gelu / no-norm / QI",
    "gelu_rmsc_std": "gelu / rms_center / std", "gelu_rmsc_qi": "gelu / rms_center / QI",
}
WIDTH_LS = {0: "--", 1: "-"}
CLASS_LABEL = WS.CLASS_LABEL


def plot_problems(cls):
    return TAB_PLOT if cls == "tabular" else PROBLEMS[cls]


def _rows(cls, problem=None):
    rows = []
    for w in WIDTHS[cls]:
        for prob in ([problem] if problem else PROBLEMS[cls]):
            p = DATA_DIR / f"{cls}_w{w}_{prob}.jsonl"
            if p.exists():
                rows += [json.loads(l) for l in open(p)]
    return rows


def _legend_handles():
    import matplotlib.lines as mlines
    h = [mlines.Line2D([], [], color=ARM_COLOR[a], lw=2, label=ARM_LABEL[a])
         for a in ARMS]
    h += [mlines.Line2D([], [], color="0.35", lw=2, ls=WIDTH_LS[i],
                        label=f"width {t}") for i, t in enumerate(["small", "large"])]
    return h


def plot(metric, fname, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(14, 14), sharex=True, sharey="row")
    for i, cls in enumerate(CLASSES):
        rows = _rows(cls)
        row_vals = []
        for j, problem in enumerate(plot_problems(cls)):
            ax = axes[i][j]
            for wi, w in enumerate(WIDTHS[cls]):
                for arm in ARMS:
                    r = next((r for r in rows if r["problem"] == problem
                              and r["arm"] == arm and r["width"] == w), None)
                    if r is None:
                        continue
                    y = np.asarray(r[metric] if metric != "evals"
                                   else np.asarray(r["evals"])[:, 1])
                    it = (np.arange(1, len(y) + 1) if metric != "evals"
                          else np.asarray(r["evals"])[:, 0])
                    pos = y > 0
                    ax.semilogy(it[pos], y[pos], color=ARM_COLOR[arm],
                                ls=WIDTH_LS[wi], lw=1.1)
                    row_vals.append(y[pos])
            ax.set_xlim(0, STEPS)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(problem, fontsize=10)
            if i == 3:
                ax.set_xlabel("iteration")
            if j == 0:
                ws = WIDTHS[cls]
                ax.set_ylabel(f"{CLASS_LABEL[cls]}  W$\\in${{{ws[0]},{ws[1]}}}"
                              f"\n{ylabel}")
        if row_vals:
            allv = np.concatenate(row_vals)
            axes[i][0].set_ylim(allv.min() * 0.5, allv.max() * 3.0)
    fig.legend(handles=_legend_handles(), loc="upper center", ncol=4,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle(title, fontsize=11, y=0.945)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_loss():
    plot("evals", "expD17n_loss.png", "eval error",
         "expD17 norm factorial: run error under plain Adam (readout zero-init).  "
         "tabular row = variance-normalized test error (mean-predictor = 1.0), "
         "PINN row = field rel $L_2$;  rows share the y scale")


# ---- gifs and scatter: QI-init arms, reusing the width_scaling renderers ----

def _snap_values(row, kind):
    G = np.asarray(row["snap_gamma"])
    if kind == "gamma":
        return G
    B = np.asarray(row["snap_bias"])
    return -B / np.maximum(G, 1e-300)


QI_ARMS = [a for a in ARMS if ARM_SPEC[a][2] == "qi"]


def hist_gifs(kind):
    """4x3 per-neuron histograms per width, QI-init arms only, one gif per
    (width, arm). Red dotted vline = median, mean in the title, axes fixed
    across frames at robust quantiles, frames on SNAP_SCHEDULE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    tag = {"gamma": ("gamma_hist", "$\\gamma_k=\\|w_k\\|_2$", "\\gamma"),
           "center": ("center_hist", "$c_k=-b_k/\\gamma_k$ (input units)", "c")}[kind]
    out_dir = FIG_DIR / tag[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    for wi in range(2):
        for arm in QI_ARMS:
            cells = {}
            for cls in CLASSES:
                w = WIDTHS[cls][wi]
                for prob in plot_problems(cls):
                    for r in _rows(cls, prob):
                        if r["arm"] == arm and r["width"] == w:
                            cells[(cls, prob)] = r
            if not cells:
                continue
            steps = next(iter(cells.values()))["snap_steps"]
            n_frames = len(steps)
            fig, axes = plt.subplots(4, 3, figsize=(13, 13.5))
            panel = {}
            for i, cls in enumerate(CLASSES):
                for j, problem in enumerate(plot_problems(cls)):
                    ax = axes[i][j]
                    r = cells.get((cls, problem))
                    if r is None:
                        ax.axis("off")
                        continue
                    V = _snap_values(r, kind)
                    lo, hi = np.quantile(V, [0.001, 0.999])
                    clip = float(np.mean((V < lo) | (V > hi)))
                    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1e-6)
                    lo, hi = lo - pad, hi + pad
                    if lo > 0 and hi / max(lo, 1e-300) > 30:
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
                        ax.set_ylabel(f"{CLASS_LABEL[cls]} (W={WIDTHS[cls][wi]})"
                                      f"\nneurons")
                    panel[(i, j)] = (ax, V, bins, lo, hi)

            def draw(fi):
                for (i, j), (ax, V, bins, lo, hi) in panel.items():
                    for pat in list(ax.patches):
                        pat.remove()
                    for ln in list(ax.lines):
                        ln.remove()
                    v = V[min(fi, V.shape[0] - 1)]
                    ax.hist(np.clip(v, lo, hi), bins=bins,
                            color=ARM_COLOR[arm], alpha=0.75)
                    ax.axvline(float(np.clip(np.median(v), lo, hi)),
                               color="red", ls=":", lw=1.6)
                    ax.set_title(f"{plot_problems(CLASSES[i])[j]}   "
                                 f"mean$|{tag[2]}|$={np.abs(v).mean():.3g}",
                                 fontsize=9)
                fig.suptitle(f"expD17 norm factorial: per-neuron ${tag[2]}$, "
                             f"{ARM_LABEL[arm]}, width index {wi+1} -- "
                             f"step {steps[min(fi, n_frames-1)]}",
                             fontsize=12, y=0.99)
                return []

            anim = FuncAnimation(fig, draw, frames=n_frames, interval=120)
            out = out_dir / f"expD17n_{kind}_hist_{arm}_w{wi+1}.gif"
            anim.save(out, writer=PillowWriter(fps=10))
            plt.close(fig)
            print(f"Saved {out}")


def gamma_vs_update(max_pts=2500, seed=0):
    """Per width, a 4x3 scatter of (gamma_k, ||Adam update row||_2) over all
    (neuron, snapshot) pairs, log-log, all six arms overlaid, colour = arm,
    alpha ramped by training step (early transparent -> late opaque)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    out_dir = FIG_DIR / "gamma_vs_update"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for wi in range(2):
        fig, axes = plt.subplots(4, 3, figsize=(14, 14))
        drew = False
        for i, cls in enumerate(CLASSES):
            w = WIDTHS[cls][wi]
            for j, problem in enumerate(plot_problems(cls)):
                ax = axes[i][j]
                rows = _rows(cls, problem)
                allG, allU = [], []
                for arm in ARMS:
                    r = next((r for r in rows if r["arm"] == arm
                              and r["width"] == w), None)
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
                    if G.size == 0:
                        continue
                    base = to_rgba(ARM_COLOR[arm])
                    cols = np.tile(base, (G.size, 1))
                    cols[:, 3] = 0.12 + 0.5 * (S / STEPS)
                    ax.scatter(G, U, c=cols, s=3, linewidths=0, rasterized=True)
                    allG.append(G)
                    allU.append(U)
                    drew = True
                if allG:
                    G_all, U_all = np.concatenate(allG), np.concatenate(allU)
                    gl, gh = np.quantile(G_all, [0.001, 0.999])
                    ul, uh = np.quantile(U_all, [0.001, 0.999])
                    clip = float(np.mean((G_all < gl) | (G_all > gh)
                                         | (U_all < ul) | (U_all > uh)))
                    ax.set_xlim(gl * 0.8, gh * 1.25)
                    ax.set_ylim(ul * 0.8, uh * 1.25)
                    if clip > 0:
                        ax.text(0.99, 0.03, f"clip {clip*100:.2g}%",
                                transform=ax.transAxes, ha="right",
                                va="bottom", fontsize=6, color="gray")
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
        fig.legend(handles=_legend_handles()[:6], loc="upper center", ncol=3,
                   fontsize=9.5, bbox_to_anchor=(0.5, 0.995), frameon=False)
        fig.suptitle(f"expD17 norm factorial: per-neuron $\\gamma$ vs Adam update "
                     f"magnitude, width index {wi+1} "
                     "(opacity ramps with training step)", fontsize=11, y=0.945)
        fig.tight_layout(rect=[0, 0, 1, 0.925])
        out = out_dir / f"expD17n_gamma_vs_update_w{wi+1}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


# ----------------------------- analysis -----------------------------

def analyze():
    recs = []
    for cls in CLASSES:
        for problem in PROBLEMS[cls]:
            rows = _rows(cls, problem)
            for w in WIDTHS[cls]:
                for arm in ARMS:
                    r = next((r for r in rows if r["arm"] == arm
                              and r["width"] == w), None)
                    if r is None:
                        continue
                    d = WS.dead_stats(r)
                    rec = {"class": cls, "problem": problem, "width": w,
                           "arm": arm, "activation": r["activation"],
                           "norm": r["norm"], "init": r["init"],
                           "n_params": r["n_params"],
                           "g0_norm": r["g0_norm"],
                           "rel_drift": r["rel_drift_end"],
                           "abs_drift": r["abs_drift_end"],
                           "pre_probe": r["pre_probe"],
                           "post_probe": r["post_probe"],
                           "final_err": r["final_err"],
                           "cn_init": r["colnorm_init"]["max_over_min_live"],
                           "cn_final": r["colnorm_final"]["max_over_min_live"],
                           "frac_zero": r["colnorm_init"]["frac_zero"],
                           "dead_run": d.get("dead_run_frac"),
                           "dead_neg": d.get("dead_of_neg"),
                           "dead_pos": d.get("dead_of_pos")}
                    if "param_final" in r:
                        rec["param_true"] = r["param_true"]
                        rec["param_final"] = r["param_final"]
                    recs.append(rec)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "analysis.json", "w") as f:
        json.dump(recs, f, indent=1)
    print(f"Saved {DATA_DIR/'analysis.json'}  ({len(recs)} rows)")
    return recs


def naval_check():
    """Is naval's 1e-3 plateau an approximation limit or label quantization?"""
    Xtr, ytr, Xte, yte = _D20.load_incumbent("airfoil") if False else \
        _D20.CANDIDATES["naval"]()
    y = np.concatenate([ytr, yte])
    u = np.unique(y)
    d = np.diff(u)
    print(f"naval target: n={y.size}, {u.size} distinct values")
    print(f"  range [{u.min():.6f}, {u.max():.6f}]")
    print(f"  spacing: min {d.min():.3e}  median {np.median(d):.3e}  max {d.max():.3e}")
    print(f"  std {y.std():.6f}")
    Xp, yp, Xep, yep = _EV.prep(Xtr, ytr, Xte, yte)
    yy = np.concatenate([yp, yep])
    uu = np.unique(yy)
    dd = np.diff(uu)
    print(f"  standardized: spacing median {np.median(dd):.3e}, std {yy.std():.4f}")
    print(f"  => a model landing at rel L2 = 1.0e-3 has residual "
          f"{1e-3*np.linalg.norm(yep)/np.sqrt(yep.size):.3e} per point "
          f"(standardized), i.e. {np.median(dd)/ (1e-3*np.linalg.norm(yep)/np.sqrt(yep.size)):.1f}x "
          f"SMALLER than the label grid step" if True else "")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot("drift", "expD17n_drift_from_init.png",
             "$\\|g_i-g_0\\|/\\|g_0\\|$",
             "expD17 norm factorial: geometry motion (relative), plain Adam; "
             "colour = activation+norm, lightness = init, dashed = smaller width")
        plot("step_size", "expD17n_step_size.png",
             "$\\|g_i-g_{i-1}\\|/\\|g_0\\|$",
             "expD17 norm factorial: per-iteration geometry motion (relative)")
        plot("drift_abs", "expD17n_drift_from_init_abs.png",
             "$\\|g_i-g_0\\|$  (absolute)",
             "expD17 norm factorial: geometry motion (absolute units)")
        plot("step_abs", "expD17n_step_size_abs.png",
             "$\\|g_i-g_{i-1}\\|$  (absolute)",
             "expD17 norm factorial: per-iteration geometry motion (absolute)")
        plot_loss()
        analyze()
        hist_gifs("gamma")
        hist_gifs("center")
        gamma_vs_update()
    elif "--analyze" in sys.argv:
        analyze()
    elif "--naval-check" in sys.argv:
        naval_check()
    elif "--job" in sys.argv:
        cls, wi, prob = sys.argv[sys.argv.index("--job") + 1].split(":")
        run_job(cls, int(wi), prob)
    else:
        driver()
