"""Experiment 17 -- does Adam learn a useful geometry? (1D, checkpoint 2)

Trains a standard tanh MLP with Adam, then asks whether the *geometry* it learned
is any good by refitting only the readout with fp64 least squares (the readout
barrier removed). Apples-to-apples with the other experiments: at each resolution
N the width is W = N + 2*halo + 1 (grid + halo); every geometry compared has the
same W neurons.

Four L2 lines per function (relative L2 on a dense eval grid):
  1. trained      -- the fully-trained net's own eval error (learned readout).
  2. trained+lstsq-- the net's BEST-checkpoint geometry, readout re-solved by lstsq
                     (Adam's best shot at a geometry).
  3. random+lstsq -- an untrained Xavier net's geometry, readout by lstsq.
  4. uniform+lstsq-- the QI grid+halo geometry (gamma=lambda*/h), readout by lstsq.

Gaps: (1)->(2) = the readout barrier; (2) vs (3) = did Adam beat random; (2/3)->(4)
= distance from the ideal geometry.

Precision/device: Adam trains in float32 (MPS if available; MPS has no float64).
The geometry is just positions/weights, so float32 training is fine; every refit
and every reported error is computed in fp64 (numpy).

Stages: (1) HP tuning [--tune], (2) train all 20 runs + (3) select best checkpoint +
(4) refit, [collect], (5) plot [--plot].

Usage:
    python experiments/expD02_adam_geometry/run.py --tune     # stage 1 HP tuning
    python experiments/expD02_adam_geometry/run.py            # stages 2-5 (collect + plot)
    python experiments/expD02_adam_geometry/run.py --plot     # plot from saved data
"""

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

from src.construction.qi_mpmath import default_halo
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
DATA_PATH = RESULTS_DIR / "data.json"

# --- experiment grid (apples-to-apples: width = W = N + 2*halo + 1) ---
WIDTHS_N = [32, 64, 128, 256, 512]
TARGETS = ["sine", "sine_8pi", "runge", "exp"]
TARGET_SEEDS = {t: i for i, t in enumerate(TARGETS)}   # one seed per function
LAMBDA_STAR = 0.25
N_TRAIN = 2003                  # prime, > max W (921) -> overdetermined refit
N_EVAL = 4001                   # prime, misaligned with the train grid
RCOND = 1e-13

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
TDTYPE = torch.float32          # training dtype (MPS has no float64)


def geometry_for_N(N):
    """The QI grid+halo geometry at resolution N: returns W, uniform centers,
    the uniform gamma (lambda*/h), h, halo. W = N + 2*halo + 1 is THE width."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    idx = np.arange(-halo, N + halo + 1)
    c_uniform = -1.0 + idx.astype(np.float64) * h
    return c_uniform.size, c_uniform, LAMBDA_STAR / h, h, halo


# ----------------------------- the trained net -----------------------------

class TanhMLP(nn.Module):
    """Standard single-hidden-layer tanh MLP, W neurons, float32."""

    def __init__(self, W, seed, init_scale=1.0):
        super().__init__()
        self.inner = nn.Linear(1, W, dtype=TDTYPE)
        self.readout = nn.Linear(W, 1, dtype=TDTYPE)
        g = torch.Generator().manual_seed(seed)
        bound = math.sqrt(6.0 / (1.0 + W))                # inner Glorot bound
        with torch.no_grad():
            # init_scale multiplies inner weight AND bias: scales gamma (=|w|) while
            # leaving the center (=-b/w) unchanged.
            self.inner.weight.copy_((torch.rand(W, 1, generator=g) * 2 - 1) * bound * init_scale)
            self.inner.bias.copy_((torch.rand(W, generator=g) * 2 - 1) * bound * init_scale)
            ob = math.sqrt(6.0 / (W + 1.0))               # readout Glorot bound
            self.readout.weight.copy_((torch.rand(1, W, generator=g) * 2 - 1) * ob)
            self.readout.bias.zero_()

    def forward(self, x):
        return self.readout(torch.tanh(self.inner(x)))

    def inner_wb(self):
        """Inner (w, b) as fp64 numpy, shape [W] each (preact = w*x + b)."""
        w = self.inner.weight.detach().cpu().numpy().astype(np.float64).ravel()
        b = self.inner.bias.detach().cpu().numpy().astype(np.float64).ravel()
        return w, b

    def readout_vb(self):
        v = self.readout.weight.detach().cpu().numpy().astype(np.float64).ravel()
        bias = float(self.readout.bias.detach().cpu().numpy().ravel()[0])
        return v, bias


# ----------------------------- fp64 evaluation -----------------------------

def features_np(w, b, X):
    """Phi[i,j] = tanh(w_j * x_i + b_j), fp64. X:[n], w,b:[W]."""
    return np.tanh(X[:, None] * w[None, :] + b[None, :])


def rel_l2_with_readout(w, b, v, bias, X_ev, y_ev, y_norm):
    """Eval rel L2 of the net f(x) = bias + sum v_j tanh(w_j x + b_j) in fp64."""
    pred = features_np(w, b, X_ev) @ v + bias
    return float(np.linalg.norm(pred - y_ev) / y_norm)


def refit_rel_l2(w, b, X_tr, y_tr, X_ev, y_ev, y_norm):
    """Freeze geometry (w,b); solve the readout by fp64 truncated-SVD lstsq
    (with bias column); return eval rel L2."""
    Phi_tr = features_np(w, b, X_tr)
    Aug = np.hstack([Phi_tr, np.ones((X_tr.size, 1))])
    U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
    s_inv = np.where(s > RCOND * s[0], 1.0 / np.where(s > 0, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv * (U.T @ y_tr))
    pred = features_np(w, b, X_ev) @ sol[:-1] + sol[-1]
    return float(np.linalg.norm(pred - y_ev) / y_norm)


# ----------------------------- training -----------------------------

def lr_at(step, hp):
    peak, warm, total = hp["lr"], hp["warmup"], hp["steps"]
    if hp["schedule"] == "constant":
        return peak
    end = hp.get("end_frac", 1e-3) * peak                 # cosine to a small floor
    if step < warm:
        return peak * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * prog))


def eval_steps(total, k=50):
    pts = np.unique(np.geomspace(max(1, total // 200), total, k).astype(int))
    return set(int(p) for p in pts) | {total}


def train_adam(W, seed, hp, X_tr_t, y_tr_t, cpu_eval):
    """Train an Adam net; track the trained-net eval (line 1, final) and the
    best-over-training lstsq-refit (line 2). cpu_eval bundles the fp64 eval data.
    Returns dict with line1/line2 errors and the trace."""
    X_tr_np, y_tr_np, X_ev_np, y_ev_np, y_norm = cpu_eval
    net = TanhMLP(W, seed, hp.get("init_scale", 1.0)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    es = eval_steps(hp["steps"])

    best_refit = math.inf
    trace = {"step": [], "train_mse": [], "refit": []}
    for step in range(hp["steps"]):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, hp)
        opt.zero_grad(set_to_none=True)
        loss = ((net(X_tr_t) - y_tr_t) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) in es:
            w, b = net.inner_wb()
            r = refit_rel_l2(w, b, X_tr_np, y_tr_np, X_ev_np, y_ev_np, y_norm)
            trace["step"].append(step + 1)
            trace["train_mse"].append(float(loss.detach().cpu()))
            trace["refit"].append(r)
            best_refit = min(best_refit, r)

    w, b = net.inner_wb(); v, bias = net.readout_vb()
    line1 = rel_l2_with_readout(w, b, v, bias, X_ev_np, y_ev_np, y_norm)  # trained net
    return {"line1_trained": line1, "line2_refit_best": best_refit, "trace": trace}


def eval_bundle(target):
    """fp64 train/eval data for one target."""
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    y_tr = get_target(target).fn_numpy(X_tr)
    y_ev = get_target(target).fn_numpy(X_ev)
    return X_tr, y_tr, X_ev, y_ev, float(np.linalg.norm(y_ev))


def to_device_train(X_tr, y_tr):
    Xt = torch.tensor(X_tr, dtype=TDTYPE, device=DEVICE).reshape(-1, 1)
    yt = torch.tensor(y_tr, dtype=TDTYPE, device=DEVICE).reshape(-1, 1)
    return Xt, yt


# ----------------------------- stage 1: HP tuning -----------------------------

def stage1_tune():
    """Expanded HP search at two widths (mid + large) on two functions (easy +
    hard): LR x init-gamma-scale (cosine), then a 50k convergence check on the
    best recipe per width. init_scale multiplies the inner (w, b) -> scales the
    starting gamma while keeping the centers, probing barrier 1."""
    print(f"device={DEVICE}, train dtype={TDTYPE}")
    tune_N = [128, 512]
    tune_fns = ["sine", "runge"]
    lrs = [1e-3, 2e-3, 3e-3]
    init_scales = [1.0, 8.0, 64.0]

    results = []
    t0 = time.time()
    # --- Phase A: recipe grid (cosine, 12k steps) ---
    for N in tune_N:
        W = geometry_for_N(N)[0]
        for fn in tune_fns:
            cpu = eval_bundle(fn)
            Xt, yt = to_device_train(cpu[0], cpu[1])
            for lr in lrs:
                for isc in init_scales:
                    hp = {"lr": lr, "schedule": "cosine", "steps": 12000,
                          "warmup": 300, "init_scale": isc}
                    r = train_adam(W, TARGET_SEEDS[fn], hp, Xt, yt, cpu)
                    results.append({"phase": "A", "N": N, "W": W, "fn": fn, "lr": lr,
                                    "init_scale": isc, "line1": r["line1_trained"],
                                    "line2": r["line2_refit_best"]})
                    print(f"[{time.time()-t0:5.0f}s] A N={N} {fn:5s} lr={lr:.0e} "
                          f"init={isc:4.0f}x | trained={r['line1_trained']:.2e} "
                          f"best-refit={r['line2_refit_best']:.2e}")

    best_hp = {}
    for N in tune_N:
        agg = {}
        for r in results:
            if r["phase"] == "A" and r["N"] == N:
                agg.setdefault((r["lr"], r["init_scale"]), []).append(r["line2"])
        bk = min(agg.items(), key=lambda kv: np.mean(kv[1]))
        best_hp[N] = bk[0]
        print(f"N={N}: best (lr,init) by mean refit = lr={bk[0][0]:.0e} "
              f"init={bk[0][1]:.0f}x  (mean {np.mean(bk[1]):.2e})")

    # --- Phase B: convergence to 50k for the best recipe per width ---
    for N in tune_N:
        W = geometry_for_N(N)[0]
        lr, isc = best_hp[N]
        for fn in tune_fns:
            cpu = eval_bundle(fn)
            Xt, yt = to_device_train(cpu[0], cpu[1])
            hp = {"lr": lr, "schedule": "cosine", "steps": 50000, "warmup": 500,
                  "init_scale": isc}
            r = train_adam(W, TARGET_SEEDS[fn], hp, Xt, yt, cpu)
            steps = np.array(r["trace"]["step"]); refit = np.array(r["trace"]["refit"])
            traj = " ".join(f"{m//1000}k:{refit[steps <= m].min():.1e}"
                            for m in [5000, 12000, 25000, 50000] if (steps <= m).any())
            print(f"[{time.time()-t0:5.0f}s] B N={N} {fn:5s} lr={lr:.0e} init={isc:.0f}x 50k | "
                  f"trained={r['line1_trained']:.2e} best-refit={r['line2_refit_best']:.2e} | {traj}")
            results.append({"phase": "B", "N": N, "W": W, "fn": fn, "lr": lr,
                            "init_scale": isc, "steps": 50000, "line1": r["line1_trained"],
                            "line2": r["line2_refit_best"], "trace": r["trace"]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "stage1_tune.json", "w") as f:
        json.dump(results, f)
    print(f"\nSaved {RESULTS_DIR/'stage1_tune.json'}")


# ----------------------------- stages 2-4: collect -----------------------------

def collect():
    """Train each (function, width) cell with Adam (best over two LRs, standard
    1x Xavier init), and refit the readout (fp64 lstsq) on the trained
    best-checkpoint geometry, the untrained random geometry, and the uniform QI
    geometry. Width = W = N + 2*halo + 1 for every line. Four L2 lines per cell."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LRS = [1e-3, 3e-3]
    STEPS = 50000
    rows = []
    t0 = time.time()
    for N in WIDTHS_N:
        W, c_uniform, gamma_uniform, h, halo = geometry_for_N(N)
        for fn in TARGETS:
            cpu = eval_bundle(fn)
            Xt, yt = to_device_train(cpu[0], cpu[1])
            seed = TARGET_SEEDS[fn]
            # lines 1 & 2: Adam (best over the two LRs), standard 1x init.
            line1, line2 = math.inf, math.inf
            for lr in LRS:
                hp = {"lr": lr, "schedule": "cosine", "steps": STEPS,
                      "warmup": 500, "init_scale": 1.0}
                r = train_adam(W, seed, hp, Xt, yt, cpu)
                line1 = min(line1, r["line1_trained"])
                line2 = min(line2, r["line2_refit_best"])
            # line 3: untrained random (1x Xavier) geometry, readout by lstsq.
            w_r, b_r = TanhMLP(W, seed, 1.0).inner_wb()
            line3 = refit_rel_l2(w_r, b_r, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
            # line 4: uniform QI geometry (gamma = lambda*/h), readout by lstsq.
            w_u = np.full(W, gamma_uniform)
            b_u = -gamma_uniform * c_uniform
            line4 = refit_rel_l2(w_u, b_u, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
            rows.append({"N": N, "W": int(W), "fn": fn, "trained": line1,
                         "learned_refit": line2, "random_refit": line3,
                         "uniform_refit": line4})
            print(f"[{time.time()-t0:5.0f}s] N={N:4d} W={W} {fn:8s} | "
                  f"trained={line1:.2e} learned={line2:.2e} "
                  f"random={line3:.2e} uniform={line4:.2e}")
    out = {"config": {"widths_n": WIDTHS_N, "targets": TARGETS, "lrs": LRS,
                      "steps": STEPS, "init_scale": 1.0, "n_train": N_TRAIN,
                      "n_eval": N_EVAL, "lambda_star": LAMBDA_STAR}, "rows": rows}
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- stage 5: plot -----------------------------

def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    rows, widths = data["rows"], data["config"]["widths_n"]
    targets = data["config"]["targets"]
    lines = [("trained", "trained net (learned readout)", "#d62728"),
             ("learned_refit", "learned geometry + lstsq", "#ff7f0e"),
             ("random_refit", "random init + lstsq", "#1f77b4"),
             ("uniform_refit", "uniform (QI) + lstsq", "#2ca02c")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Exp17: does Adam learn a useful geometry?  (relative $L_2$, "
                 "fp64 refit; standard Xavier init)", fontsize=14)
    for ax, fn in zip(axes.ravel(), targets):
        for key, label, color in lines:
            ys = [next(r[key] for r in rows if r["fn"] == fn and r["N"] == N)
                  for N in widths]
            ax.loglog(widths, ys, "-o", color=color, ms=5, label=label)
        ax.set_title(fn, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$ neurons)")
        ax.set_ylabel(r"relative $L_2$")
        ax.set_xticks(widths)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0][0].legend(fontsize=9, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / "adam_geometry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    if "--tune" in sys.argv:
        stage1_tune()
    elif "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            plot(json.load(f))
    else:
        plot(collect())
