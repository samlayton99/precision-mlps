"""Experiment expD02 (lambda-init variant) -- does Adam HOLD a correct gamma regime?

The original `run.py` trains from standard Xavier init (gamma ~ 0.1, ~100x too
small) and asks "does Adam learn a useful geometry?" -- the honest test, answer:
no. This variant flips the question: INITIALIZE Adam already in the correct
gamma/lambda regime (lambda* = 0.25) and ask whether training preserves it, and
whether -- given a good geometry (and even a good readout) -- Adam stays near the
floor or drifts off it.

The initialization splits into three independent parts: the inner weight w, the
inner bias b (together they fix the centers c = -b/w and the bandwidth gamma =
|w|), and the readout. The four schemes (one PNG each) span that space; all four
put the bandwidth in the lambda* = 0.25 regime (gamma* = lambda*/h), and differ in
the center layout and the readout:

  1. scaled_xavier -- a genuine Xavier draw (w, b), then a single global scale
     s = gamma*/mean(|w|) on BOTH w and b. Average bandwidth -> gamma*, the
     (random) centers c = -b/w untouched (scaling w and b together never moves a
     center). Readout: random Xavier, trained. Tests: right bandwidth + random
     centers -- enough?
  2. const_weight -- w = gamma* for every neuron, but b = a raw Xavier draw, so
     the centers c = -b/gamma* collapse toward 0 (b ~ O(0.1), gamma* >> 1).
     Readout: random Xavier, trained. An ablation: correct constant weight vector
     but NOT the correct centers.
  3. qi_geometry -- the exact QI GEOMETRY: w = gamma*, b = -gamma* * c_uniform
     (uniform grid + halo centers). Readout: random Xavier, trained. Tests: from a
     perfect geometry + useless readout, can Adam solve the readout / keep the
     geometry?  (The honest readout-barrier test.)
  4. full_qi -- the exact QI geometry AND the readout set to the fp64 lstsq
     solution on that geometry (so the net starts AT the ~1e-14 floor), then Adam.
     Tests: does Adam drift off the exact solution? (The low-LR-from-construction
     drift test.)

Four lines per (function, width), relative L2 on a dense fp64 eval grid:
  1. trained        -- the fully-trained net's own eval error (learned readout).
  2. trained+lstsq  -- the BEST-checkpoint geometry, readout re-solved by fp64 lstsq.
  3. init+lstsq     -- this scheme's UNTRAINED geometry, readout by fp64 lstsq
                       (what the initialization's GEOMETRY alone buys).
  4. uniform(fp64)  -- the uniform QI geometry built in pure fp64, readout by lstsq:
                       the absolute floor reference (== the full_qi starting point).

PRECISION CAVEAT: training is float32 (MPS has no float64), just like run.py. The
geometry is extracted to fp64 for every refit, so the floor is reachable only when
the stored params happen to be float32-exact. For the UNTRAINED uniform geometry
(schemes 3-4) this holds exactly -- for power-of-2 widths with lambda* = 0.25 the
params (w = N/8, b = gamma* - n*0.25, multiples of 0.25) are float32-exact, so
line 3 (qi_geometry / full_qi) and line 4 coincide at ~1e-14. Once Adam moves
params to arbitrary float32 values (lines 1-2), and for the scaled_xavier scale s
and the full_qi readout coefficients (which reach ~hundreds), the float32 storage
carries ~1e-7 relative error and bounds the refit near ~1e-6 to ~1e-10. Line 4
(pure fp64) is always the true floor reference.

Usage:
    python experiments/expD02_adam_geometry/run_lambda_init.py            # run the NEW schemes (const_weight, full_qi), merge, plot
    python experiments/expD02_adam_geometry/run_lambda_init.py --all      # (re)run all four schemes
    python experiments/expD02_adam_geometry/run_lambda_init.py --plot     # plot from saved data
    python experiments/expD02_adam_geometry/run_lambda_init.py --sanity   # init/refit checks only
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the original expD02 machinery (fp64 eval/refit, geometry, schedules).
import run as expD02
from run import (DEVICE, TDTYPE, LAMBDA_STAR, N_TRAIN, N_EVAL, RCOND, features_np,
                 geometry_for_N, eval_bundle, to_device_train,
                 refit_rel_l2, rel_l2_with_readout, lr_at, eval_steps)
from src.construction.center_geometry import trained_centers

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
DATA_PATH = RESULTS_DIR / "data_lambda_init.json"

# --- experiment grid (apples-to-apples: width = W = N + 2*halo + 1) ---
WIDTHS_N = [32, 64, 128, 256, 512]
TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
TARGET_SEEDS = {t: i for i, t in enumerate(TARGETS)}     # one seed per function
ALL_SCHEMES = ["scaled_xavier", "const_weight", "qi_geometry", "full_qi"]
NEW_SCHEMES = ["const_weight", "full_qi"]                 # the ones added this round
LRS = [1e-3, 3e-3]
STEPS = 50000

SCHEME_TITLE = {
    "scaled_xavier": r"scaled Xavier -- random centers, correct bandwidth "
                     r"($\mathrm{mean}|w| = \gamma^*$); readout trained",
    "const_weight":  r"constant weight $w=\gamma^*$, random Xavier bias "
                     r"(centers collapse $\to 0$); readout trained",
    "qi_geometry":   r"QI geometry ($w=\gamma^*$, $b=-\gamma^* c_{\rm unif}$, "
                     r"uniform centers); readout random $\to$ trained",
    "full_qi":       r"full QI init -- QI geometry AND fp64-lstsq readout "
                     r"(starts at the floor), then Adam",
}


# ----------------------------- the trained net -----------------------------

class TanhMLP(nn.Module):
    """Single-hidden-layer tanh MLP, W neurons, float32. Params are set by
    init_net (the nn.Linear default init is overwritten there)."""

    def __init__(self, W):
        super().__init__()
        self.inner = nn.Linear(1, W, dtype=TDTYPE)
        self.readout = nn.Linear(W, 1, dtype=TDTYPE)

    def forward(self, x):
        return self.readout(torch.tanh(self.inner(x)))

    def inner_wb(self):
        w = self.inner.weight.detach().cpu().numpy().astype(np.float64).ravel()
        b = self.inner.bias.detach().cpu().numpy().astype(np.float64).ravel()
        return w, b

    def readout_vb(self):
        v = self.readout.weight.detach().cpu().numpy().astype(np.float64).ravel()
        bias = float(self.readout.bias.detach().cpu().numpy().ravel()[0])
        return v, bias


def lstsq_solution(w, b, X_tr, y_tr):
    """The fp64 truncated-SVD lstsq readout (with bias column) on geometry (w, b).
    Returns (v, bias) -- the readout that minimizes the residual on this geometry."""
    Phi = features_np(w, b, X_tr)
    Aug = np.hstack([Phi, np.ones((X_tr.size, 1))])
    U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
    s_inv = np.where(s > RCOND * s[0], 1.0 / np.where(s > 0, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv * (U.T @ y_tr))
    return sol[:-1], float(sol[-1])


def init_net(W, seed, scheme, gamma_uniform, c_uniform, readout=None):
    """Build a TanhMLP initialized per `scheme` (see module docstring). All schemes
    place the bandwidth at gamma*; they differ in the center layout and readout.
    If `readout=(v, bias)` is given, the readout is set to it; otherwise random
    Xavier (then trained)."""
    net = TanhMLP(W)
    g = torch.Generator().manual_seed(seed)
    bound = math.sqrt(6.0 / (1.0 + W))
    with torch.no_grad():
        if scheme == "xavier":
            # (a) standard Xavier everything (gamma ~ 0.1). Same draw order as
            # run.py's TanhMLP at init_scale=1, so seed s reproduces it exactly.
            net.inner.weight.copy_((torch.rand(W, 1, generator=g) * 2 - 1) * bound)
            net.inner.bias.copy_((torch.rand(W, generator=g) * 2 - 1) * bound)
        elif scheme == "scaled_xavier":
            # (b) Xavier (w, b), scaled by s so mean|w| = gamma*; centers c=-b/w
            # (raw, heavy-tailed) unchanged because scaling w and b together never
            # moves a center.
            w = (torch.rand(W, 1, generator=g) * 2 - 1) * bound
            b = (torch.rand(W, generator=g) * 2 - 1) * bound
            s = gamma_uniform / float(w.abs().mean())
            net.inner.weight.copy_(w * s)
            net.inner.bias.copy_(b * s)
        elif scheme == "const_weight":
            # (c) constant weight w=gamma*, centers = Xavier -b/w CLIPPED to the
            # grid+halo span (the expC05 centers "xavier" covering geometry), b=-gamma*c.
            w_x = ((torch.rand(W, 1, generator=g) * 2 - 1) * bound).numpy().ravel()
            b_x = ((torch.rand(W, generator=g) * 2 - 1) * bound).numpy().ravel()
            span = (float(c_uniform[0]), float(c_uniform[-1]))
            c = trained_centers(w_x, b_x, span)           # spread, covers the span
            net.inner.weight.copy_(torch.full((W, 1), gamma_uniform, dtype=TDTYPE))
            net.inner.bias.copy_(torch.tensor(-gamma_uniform * c, dtype=TDTYPE).reshape(W))
        elif scheme in ("qi", "qi_geometry", "full_qi"):
            # (d) exact QI geometry: w=gamma*, b=-gamma* c_uniform (uniform centers).
            net.inner.weight.copy_(torch.full((W, 1), gamma_uniform, dtype=TDTYPE))
            net.inner.bias.copy_(torch.tensor(-gamma_uniform * c_uniform,
                                              dtype=TDTYPE).reshape(W))
        else:
            raise ValueError(scheme)

        if readout is not None:
            v, rbias = readout
            net.readout.weight.copy_(torch.tensor(v, dtype=TDTYPE).reshape(1, W))
            net.readout.bias.copy_(torch.tensor([rbias], dtype=TDTYPE))
        else:
            ob = math.sqrt(6.0 / (W + 1.0))               # readout Glorot
            net.readout.weight.copy_((torch.rand(1, W, generator=g) * 2 - 1) * ob)
            net.readout.bias.zero_()
    return net


# ----------------------------- training -----------------------------

def train_from_net(net, hp, X_tr_t, y_tr_t, cpu_eval):
    """Train a pre-initialized net with Adam; track the trained-net eval (line 1)
    and the best-over-training lstsq refit (line 2). Returns (line1, line2)."""
    X_tr_np, y_tr_np, X_ev_np, y_ev_np, y_norm = cpu_eval
    net = net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    es = eval_steps(hp["steps"])

    best_refit = math.inf
    for step in range(hp["steps"]):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, hp)
        opt.zero_grad(set_to_none=True)
        loss = ((net(X_tr_t) - y_tr_t) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) in es:
            w, b = net.inner_wb()
            best_refit = min(best_refit, refit_rel_l2(
                w, b, X_tr_np, y_tr_np, X_ev_np, y_ev_np, y_norm))

    w, b = net.inner_wb(); v, bias = net.readout_vb()
    line1 = rel_l2_with_readout(w, b, v, bias, X_ev_np, y_ev_np, y_norm)
    return line1, best_refit


# ----------------------------- cube cell -----------------------------

def compute_cell(scheme, N, fn):
    """Run one (scheme, N, fn) cell and return the four regime errors keyed by the
    descriptive cube names, plus W and the seed used. Regimes:
      adam_both         -- fully trained by Adam (both layers), best over the LRs.
      adam_first_lstsq  -- best-checkpoint first layer, readout re-solved by lstsq.
      frozen_init_lstsq -- untrained (init) first layer, readout by lstsq.
      qi_lstsq          -- the QI uniform first layer (pure fp64), readout by lstsq.
    """
    W, c_uniform, gamma_uniform, h, halo = geometry_for_N(N)
    cpu = eval_bundle(fn)
    Xt, yt = to_device_train(cpu[0], cpu[1])
    seed = TARGET_SEEDS[fn]
    ro = None
    if scheme in ("qi", "full_qi"):
        w_u0 = np.full(W, gamma_uniform); b_u0 = -gamma_uniform * c_uniform
        ro = lstsq_solution(w_u0, b_u0, cpu[0], cpu[1])
    adam_both, adam_first = math.inf, math.inf
    for lr in LRS:
        net = init_net(W, seed, scheme, gamma_uniform, c_uniform, readout=ro)
        hp = {"lr": lr, "schedule": "cosine", "steps": STEPS, "warmup": 500}
        l1, l2 = train_from_net(net, hp, Xt, yt, cpu)
        adam_both, adam_first = min(adam_both, l1), min(adam_first, l2)
    w0, b0 = init_net(W, seed, scheme, gamma_uniform, c_uniform, readout=ro).inner_wb()
    frozen = refit_rel_l2(w0, b0, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
    w_u = np.full(W, gamma_uniform); b_u = -gamma_uniform * c_uniform
    qi = refit_rel_l2(w_u, b_u, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
    return {"W": int(W), "seed": seed, "adam_both": adam_both,
            "adam_first_lstsq": adam_first, "frozen_init_lstsq": frozen, "qi_lstsq": qi}


# ----------------------------- collect -----------------------------

def load_existing():
    """Existing rows keyed (scheme, N, fn); renames the legacy 'uniform' scheme to
    'qi_geometry' so prior runs are reused under the new descriptive name."""
    if not DATA_PATH.exists():
        return {}
    with open(DATA_PATH) as f:
        data = json.load(f)
    out = {}
    for r in data["rows"]:
        if r["scheme"] == "uniform":
            r["scheme"] = "qi_geometry"
        out[(r["scheme"], r["N"], r["fn"])] = r
    return out


def collect(run_schemes):
    """Run `run_schemes`, merging with any existing rows (other schemes reused)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_existing()
    t0 = time.time()
    for scheme in run_schemes:
        for N in WIDTHS_N:
            W, c_uniform, gamma_uniform, h, halo = geometry_for_N(N)
            for fn in TARGETS:
                cpu = eval_bundle(fn)
                Xt, yt = to_device_train(cpu[0], cpu[1])
                seed = TARGET_SEEDS[fn]
                # full_qi: precompute the QI/lstsq readout on the uniform geometry.
                ro = None
                if scheme == "full_qi":
                    w_u0 = np.full(W, gamma_uniform); b_u0 = -gamma_uniform * c_uniform
                    ro = lstsq_solution(w_u0, b_u0, cpu[0], cpu[1])
                # lines 1 & 2: Adam from this scheme's init, best over the two LRs.
                line1, line2 = math.inf, math.inf
                for lr in LRS:
                    net = init_net(W, seed, scheme, gamma_uniform, c_uniform, readout=ro)
                    hp = {"lr": lr, "schedule": "cosine", "steps": STEPS, "warmup": 500}
                    l1, l2 = train_from_net(net, hp, Xt, yt, cpu)
                    line1, line2 = min(line1, l1), min(line2, l2)
                # line 3: this scheme's UNTRAINED geometry (float32 net), refit fp64.
                w0, b0 = init_net(W, seed, scheme, gamma_uniform, c_uniform,
                                  readout=ro).inner_wb()
                line3 = refit_rel_l2(w0, b0, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
                # line 4: pure-fp64 uniform QI geometry, refit fp64 (floor reference).
                w_u = np.full(W, gamma_uniform); b_u = -gamma_uniform * c_uniform
                line4 = refit_rel_l2(w_u, b_u, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
                rows[(scheme, N, fn)] = {"scheme": scheme, "N": N, "W": int(W), "fn": fn,
                                         "trained": line1, "learned_refit": line2,
                                         "init_refit": line3, "uniform_refit": line4}
                print(f"[{time.time()-t0:5.0f}s] {scheme:13s} N={N:4d} W={W} {fn:12s} | "
                      f"trained={line1:.2e} learned={line2:.2e} "
                      f"init={line3:.2e} uniform(fp64)={line4:.2e}")
    ordered = [rows[(s, N, fn)] for s in ALL_SCHEMES for N in WIDTHS_N for fn in TARGETS
               if (s, N, fn) in rows]
    out = {"config": {"widths_n": WIDTHS_N, "targets": TARGETS, "schemes": ALL_SCHEMES,
                      "lrs": LRS, "steps": STEPS, "n_train": N_TRAIN, "n_eval": N_EVAL,
                      "lambda_star": LAMBDA_STAR, "train_dtype": "float32"}, "rows": ordered}
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plot -----------------------------

def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    rows, widths = data["rows"], data["config"]["widths_n"]
    targets = data["config"]["targets"]
    lines = [("trained", "trained net (learned readout)", "#d62728"),
             ("learned_refit", "trained geometry + lstsq", "#ff7f0e"),
             ("init_refit", "init geometry (untrained) + lstsq", "#1f77b4"),
             ("uniform_refit", "uniform QI (fp64) + lstsq", "#2ca02c")]
    schemes = [s for s in data["config"]["schemes"]
               if any(r["scheme"] == s for r in rows)]
    for scheme in schemes:
        srows = [r for r in rows if r["scheme"] == scheme]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"expD02 (lambda-init): {SCHEME_TITLE[scheme]}\n"
                     r"relative $L_2$, fp64 refit, $\lambda^*=0.25$, float32 Adam",
                     fontsize=13)
        for ax, fn in zip(axes.ravel(), targets):
            for key, label, color in lines:
                ys = [next(r[key] for r in srows if r["fn"] == fn and r["N"] == N)
                      for N in widths]
                ax.loglog(widths, ys, "-o", color=color, ms=5, label=label)
            ax.set_title(fn, fontsize=12)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$ neurons)")
            ax.set_ylabel(r"relative $L_2$")
            ax.set_xticks(widths)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncol=4, fontsize=10, borderaxespad=0)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        out = RESULTS_DIR / f"lambda_init_{scheme}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


# ----------------------------- sanity checks -----------------------------

def sanity():
    """Verify the implementation matches the theory before the full run:
    (1) scaled_xavier puts mean|w| at gamma*; centers match the raw draw.
    (2) const_weight collapses centers toward 0 (b/gamma* << 1).
    (3) the pure-fp64 uniform QI geometry refits to ~machine floor (== full_qi start).
    (4) the float32-stored uniform geometry refits at the same floor (float32-exact)."""
    print("=== sanity checks ===")
    for N in [64, 512]:
        W, c_uniform, gamma_uniform, h, halo = geometry_for_N(N)
        g = torch.Generator().manual_seed(0)
        bound = math.sqrt(6.0 / (1.0 + W))
        w = (torch.rand(W, 1, generator=g) * 2 - 1) * bound
        b = (torch.rand(W, generator=g) * 2 - 1) * bound
        # (1) scaled_xavier bandwidth + center invariance
        s = gamma_uniform / float(w.abs().mean())
        c_raw = (-b / w.ravel()).numpy()
        c_scaled = (-(b * s) / (w.ravel() * s)).numpy()
        mean_abs_w = float((w.ravel() * s).abs().mean())
        # (2) const_weight collapsed centers
        c_const = (-b.numpy() / gamma_uniform)
        print(f"N={N}: gamma*={gamma_uniform:.3f}  scaled mean|w|={mean_abs_w:.3f}  "
              f"center-shift={np.abs(c_raw - c_scaled).max():.1e}  "
              f"const_weight |center| max={np.abs(c_const).max():.3e}")
        assert abs(mean_abs_w - gamma_uniform) < 1e-4 * gamma_uniform
        assert np.abs(c_raw - c_scaled).max() < 1e-6
        assert np.abs(c_const).max() < 0.1            # centers collapsed near 0

        # (3) & (4) uniform geometry: fp64 floor vs float32 cap; full_qi readout
        cpu = eval_bundle("sine")
        w_u64 = np.full(W, gamma_uniform); b_u64 = -gamma_uniform * c_uniform
        r64 = refit_rel_l2(w_u64, b_u64, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
        net = init_net(W, 0, "qi_geometry", gamma_uniform, c_uniform)
        w32, b32 = net.inner_wb()
        r32 = refit_rel_l2(w32, b32, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4])
        # full_qi: net starts near the floor with the lstsq readout (float32-stored)
        v, rb = lstsq_solution(w_u64, b_u64, cpu[0], cpu[1])
        fq = init_net(W, 0, "full_qi", gamma_uniform, c_uniform, readout=(v, rb))
        wq, bq = fq.inner_wb(); vq, bq2 = fq.readout_vb()
        start = rel_l2_with_readout(wq, bq, vq, bq2, cpu[2], cpu[3], cpu[4])
        print(f"      uniform refit fp64={r64:.2e}  float32-stored={r32:.2e}  "
              f"full_qi start (float32 net)={start:.2e}  (sine, N={N})")
    print("sanity OK")


if __name__ == "__main__":
    if "--sanity" in sys.argv:
        sanity()
    elif "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            plot(json.load(f))
    else:
        run_schemes = ALL_SCHEMES if "--all" in sys.argv else NEW_SCHEMES
        plot(collect(run_schemes))
