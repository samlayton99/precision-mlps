"""expD02 variant -- QI init, Adam to plateau, LBFGS finishes (fp64).

Recreates the QI-init (green) line of initialization_slice/init_slice_adam_both
for all 6 functions, with one change to the training recipe: Adam does NOT run
all the way. Adam trains both layers from the full QI init (exact geometry
w = gamma*, b = -gamma* c_uniform, readout = fp64 lstsq solution) until its
train loss plateaus (or MAX_STEPS cap), then LBFGS (PyTorch, strong-Wolfe
closure) finishes the optimization from Adam's final state.

Everything runs in fp64 on CPU -- unlike the cube's float32/MPS runs -- because
the question is whether the *optimizer* can hold/recover machine precision, and
float32 storage alone caps eval error near 1e-7.

Lines per (function, N), eval relative L2 on the dense fp64 grid:
  qi_lstsq          -- the construction floor (QI geometry + lstsq), reference.
  after_adam        -- the net's own eval error when Adam plateaus.
  after_lbfgs       -- the net's own eval error after the LBFGS finish (headline).
  refit_after_lbfgs -- final geometry + lstsq refit (did the geometry survive?).
  cube fp32 Adam-only reference overlaid from cube.json when present.

Usage:
    python experiments/expD02_adam_geometry/run_qi_adam_lbfgs.py          # run + plot
    python experiments/expD02_adam_geometry/run_qi_adam_lbfgs.py --plot   # plot only
"""

import copy
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

# Load this experiment's run.py by explicit path: several experiments have a
# run.py, and a bare `from run import ...` picks up whichever one is already in
# sys.modules, which breaks pytest collection when two are imported together.
import importlib.util as _ilu
_rs = _ilu.spec_from_file_location("expd02_run", Path(__file__).resolve().parent / "run.py")
_d02run = _ilu.module_from_spec(_rs); _rs.loader.exec_module(_d02run)
geometry_for_N, eval_bundle = _d02run.geometry_for_N, _d02run.eval_bundle
refit_rel_l2, rel_l2_with_readout = _d02run.refit_rel_l2, _d02run.rel_l2_with_readout
from run_lambda_init import lstsq_solution

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
DATA_PATH = RESULTS_DIR / "data_qi_adam_lbfgs.json"
CUBE_PATH = RESULTS_DIR / "cube.json"

WIDTHS_N = [32, 64, 128, 256, 512]
TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
LRS = [1e-3, 3e-3]

MAX_STEPS = 1_500_000        # hard cap; plateau stopping fires long before this
WARMUP = 500                 # linear LR warmup, then constant (no fixed-horizon cosine)
PATIENCE = 5000              # stop Adam after this many steps without a real improvement
REL_IMPROVE = 1e-3           # "real improvement" = loss < best * (1 - REL_IMPROVE)
LBFGS_CHUNK = 100            # LBFGS iterations per .step(closure) call
LBFGS_MAX_ITERS = 3000
LBFGS_REL_IMPROVE = 1e-3     # stop when a chunk improves the loss by less than this


class TanhMLP64(nn.Module):
    """Single-hidden-layer tanh MLP, W neurons, fp64 (CPU)."""

    def __init__(self, W):
        super().__init__()
        self.inner = nn.Linear(1, W, dtype=torch.float64)
        self.readout = nn.Linear(W, 1, dtype=torch.float64)

    def forward(self, x):
        return self.readout(torch.tanh(self.inner(x)))

    def inner_wb(self):
        w = self.inner.weight.detach().numpy().astype(np.float64).ravel()
        b = self.inner.bias.detach().numpy().astype(np.float64).ravel()
        return w.copy(), b.copy()

    def readout_vb(self):
        v = self.readout.weight.detach().numpy().astype(np.float64).ravel().copy()
        bias = float(self.readout.bias.detach().numpy().ravel()[0])
        return v, bias


def init_qi_net(W, gamma_uniform, c_uniform, readout):
    """The full QI init in fp64: w = gamma*, b = -gamma* c_uniform, readout =
    (v, bias) from the fp64 lstsq solve. The net starts AT the construction."""
    net = TanhMLP64(W)
    v, rbias = readout
    with torch.no_grad():
        net.inner.weight.copy_(torch.full((W, 1), gamma_uniform, dtype=torch.float64))
        net.inner.bias.copy_(torch.tensor(-gamma_uniform * c_uniform,
                                          dtype=torch.float64).reshape(W))
        net.readout.weight.copy_(torch.tensor(v, dtype=torch.float64).reshape(1, W))
        net.readout.bias.copy_(torch.tensor([rbias], dtype=torch.float64))
    return net


def adam_stage(net, lr, Xt, yt, max_steps=MAX_STEPS, warmup=WARMUP,
               patience=PATIENCE, rel_improve=REL_IMPROVE, log_every=100):
    """Adam on both layers until the train MSE plateaus: stop when `patience`
    steps pass without the loss improving on its best by a relative
    `rel_improve`. Linear warmup then constant LR (a fixed-horizon cosine makes
    no sense with an adaptive stopping time). Returns steps run + loss trace."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_step = math.inf, 0
    trace = {"step": [], "loss": []}
    step, l = 0, math.nan
    for step in range(max_steps):
        cur = lr * (step + 1) / warmup if step < warmup else lr
        for grp in opt.param_groups:
            grp["lr"] = cur
        opt.zero_grad(set_to_none=True)
        loss = ((net(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
        l = float(loss)
        if step % log_every == 0:
            trace["step"].append(step)
            trace["loss"].append(l)
        if l < best * (1.0 - rel_improve):
            best_step = step
        if l < best:
            best = l
        if step - best_step >= patience:
            break
    return {"steps_run": step + 1, "final_loss": l, "best_loss": best, "trace": trace}


def lbfgs_stage(net, Xt, yt, max_iters=LBFGS_MAX_ITERS, chunk=LBFGS_CHUNK,
                rel_improve=LBFGS_REL_IMPROVE):
    """LBFGS (strong Wolfe) from the net's current state, in chunks of `chunk`
    iterations; stop when a chunk improves the loss by < `rel_improve`
    (relative) or the loss goes non-finite (state restored). fp64 throughout."""
    opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=chunk,
                            history_size=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-30, tolerance_change=1e-30)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = ((net(Xt) - yt) ** 2).mean()
        loss.backward()
        return loss

    def cur_loss():
        with torch.no_grad():
            return float(((net(Xt) - yt) ** 2).mean())

    prev = cur_loss()
    trace = [prev]
    backup = copy.deepcopy(net.state_dict())
    iters = 0
    for _ in range(max(1, max_iters // chunk)):
        opt.step(closure)
        l = cur_loss()
        if not math.isfinite(l) or l > prev:
            net.load_state_dict(backup)     # keep the best finite state
            l = prev
            break
        iters += chunk
        trace.append(l)
        improved = l < prev * (1.0 - rel_improve)
        backup = copy.deepcopy(net.state_dict())
        prev = l
        if not improved:
            break
    return {"iters": iters, "final_loss": prev, "trace": trace}


def run_cell(fn, N, lr, adam_kw=None, lbfgs_kw=None):
    """One (function, N, lr) cell: QI init -> Adam to plateau -> LBFGS finish.
    All reported errors are eval relative L2 in fp64."""
    W, c_uniform, gamma_uniform, h, halo = geometry_for_N(N)
    X_tr, y_tr, X_ev, y_ev, y_norm = eval_bundle(fn)
    Xt = torch.tensor(X_tr, dtype=torch.float64).reshape(-1, 1)
    yt = torch.tensor(y_tr, dtype=torch.float64).reshape(-1, 1)

    w_u = np.full(W, gamma_uniform)
    b_u = -gamma_uniform * c_uniform
    ro = lstsq_solution(w_u, b_u, X_tr, y_tr)
    qi_floor = refit_rel_l2(w_u, b_u, X_tr, y_tr, X_ev, y_ev, y_norm)

    net = init_qi_net(W, gamma_uniform, c_uniform, ro)
    w0, b0 = net.inner_wb(); v0, rb0 = net.readout_vb()
    err_init = rel_l2_with_readout(w0, b0, v0, rb0, X_ev, y_ev, y_norm)

    adam = adam_stage(net, lr, Xt, yt, **(adam_kw or {}))
    w1, b1 = net.inner_wb(); v1, rb1 = net.readout_vb()
    err_after_adam = rel_l2_with_readout(w1, b1, v1, rb1, X_ev, y_ev, y_norm)

    lbfgs = lbfgs_stage(net, Xt, yt, **(lbfgs_kw or {}))
    w2, b2 = net.inner_wb(); v2, rb2 = net.readout_vb()
    err_after_lbfgs = rel_l2_with_readout(w2, b2, v2, rb2, X_ev, y_ev, y_norm)
    refit_after_lbfgs = refit_rel_l2(w2, b2, X_tr, y_tr, X_ev, y_ev, y_norm)

    return {"fn": fn, "N": N, "W": int(W), "lr": lr,
            "qi_lstsq": qi_floor, "err_init": err_init,
            "adam_steps": adam["steps_run"], "adam_final_loss": adam["final_loss"],
            "err_after_adam": err_after_adam,
            "lbfgs_iters": lbfgs["iters"], "lbfgs_final_loss": lbfgs["final_loss"],
            "err_after_lbfgs": err_after_lbfgs,
            "refit_after_lbfgs": refit_after_lbfgs,
            "adam_trace": adam["trace"], "lbfgs_trace": lbfgs["trace"]}


def collect():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for N in WIDTHS_N:
        for fn in TARGETS:
            for lr in LRS:
                r = run_cell(fn, N, lr)
                rows.append(r)
                print(f"[{time.time()-t0:6.0f}s] N={N:4d} W={r['W']:4d} {fn:12s} "
                      f"lr={lr:.0e} | adam {r['adam_steps']:6d} steps "
                      f"-> {r['err_after_adam']:.2e} | lbfgs {r['lbfgs_iters']:4d} "
                      f"iters -> {r['err_after_lbfgs']:.2e} "
                      f"(refit {r['refit_after_lbfgs']:.2e}, floor {r['qi_lstsq']:.2e})",
                      flush=True)
    out = {"config": {"widths_n": WIDTHS_N, "targets": TARGETS, "lrs": LRS,
                      "max_steps": MAX_STEPS, "warmup": WARMUP,
                      "patience": PATIENCE, "rel_improve": REL_IMPROVE,
                      "lbfgs_chunk": LBFGS_CHUNK, "lbfgs_max_iters": LBFGS_MAX_ITERS,
                      "lbfgs_rel_improve": LBFGS_REL_IMPROVE,
                      "train_dtype": "float64", "device": "cpu",
                      "init": "full QI (w=gamma*, b=-gamma*c_uniform, lstsq readout)",
                      "metric": "eval relative L2 on the dense fp64 grid"},
           "rows": rows}
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plot -----------------------------

def _best_rows(rows, fn, widths):
    """Per width, the LR branch with the best post-LBFGS error (the whole
    branch is reported, so after_adam and after_lbfgs stay consistent)."""
    out = []
    for N in widths:
        cands = [r for r in rows if r["fn"] == fn and r["N"] == N]
        out.append(min(cands, key=lambda r: r["err_after_lbfgs"]))
    return out


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    rows, widths = data["rows"], data["config"]["widths_n"]
    targets = data["config"]["targets"]

    cube_ref = None
    if CUBE_PATH.exists():
        cube = json.load(open(CUBE_PATH))
        cube_ref = {(r["function"], r["N"]): r["adam_both"]
                    for r in cube["rows"] if r["init"] == "qi"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for ax, fn in zip(axes.ravel(), targets):
        best = _best_rows(rows, fn, widths)
        ax.loglog(widths, [r["qi_lstsq"] for r in best], "--", color="k",
                  lw=1.6, label="QI + lstsq (construction floor)")
        if cube_ref is not None:
            ys = [cube_ref.get((fn, N)) for N in widths]
            if all(y is not None for y in ys):
                ax.loglog(widths, ys, ":", color="#2ca02c", lw=1.6,
                          label="Adam-only 50k, fp32 (cube ref)")
        ax.loglog(widths, [r["err_after_adam"] for r in best], "-o",
                  color="#ff7f0e", ms=5, label="Adam at plateau (own readout)")
        ax.loglog(widths, [r["err_after_lbfgs"] for r in best], "-o",
                  color="#d62728", ms=5, label="Adam -> LBFGS finish (own readout)")
        ax.loglog(widths, [r["refit_after_lbfgs"] for r in best], ":",
                  color="#9467bd", lw=1.6, label="final geometry + lstsq refit")
        ax.set_title(fn, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$ neurons)")
        ax.set_ylabel(r"relative $L_2$")
        ax.set_xticks(widths)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle("expD02 (QI init): Adam to plateau, LBFGS finishes -- fp64, "
                 r"$\lambda^*=0.25$, best over LRs $\{10^{-3},3\times10^{-3}\}$",
                 fontsize=13, y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=5, fontsize=9, frameon=True, borderaxespad=0)
    out = RESULTS_DIR / "qi_adam_lbfgs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            plot(json.load(f))
    else:
        plot(collect())
