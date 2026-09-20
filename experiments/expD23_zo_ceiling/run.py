"""expD23 -- the zero-order ceiling: every solver in Fchaubard/zero_order_rnn
(the four expD22 never ported, plus its binary LR search) on the expD16 suite,
and the two oracle zero-order arms that bound the class.

Same suite as expD16/expD22 (their build_model/data_bundle are imported by
path): standard-parameterization tanh MLP, qi init (construction geometry,
Glorot readout), full-batch fp64 MSE on 2003 equispaced points, eval rel L2
on a misaligned 4001-point grid.  Budgets are counted in loss-oracle
evaluations (one full-batch forward each).

Modes:
  --tune                 short-budget hyperparameter grids for the new arms (cell A)
  --arms [--only a,b]    the arms at the expD22 headline budget (120k evals) on cell A
  --long                 the two lines that matter at 3000 iterations
  --ceiling              Picard curves + the loss-oracle Newton floor on the readout
                         block, all 12 qi cells; Hessian spectra on cell A
  --grid                 the new arms on 4 targets x N in {64,128} (qi)
  --plot                 all figures from whatever data exists
Cell A = sine / N=64 / qi (m = 616), the cell every expD22 tuning stage used.

    uv run --extra dev python experiments/expD23_zo_ceiling/run.py --tune
    OMP_NUM_THREADS=2 uv run --extra dev python experiments/expD23_zo_ceiling/run.py --arms --only spsa15,lrsearch
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
_HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d22 = _load("expD23_d22", REPO_ROOT / "experiments" / "expD22_cdrge" / "run.py")
zo = _load("expD23_zo", _HERE / "zo.py")

torch.set_default_dtype(torch.float64)

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD23_zo_ceiling"
DATA_DIR = RESULTS_DIR / "data"
FIG_DIR = RESULTS_DIR / "figures"
D22_DATA = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD22_cdrge" / "data"

TARGETS = ["sine", "exp", "runge", "sine_8pi"]
WIDTHS_N = [64, 128, 256]
CELL_A = ("sine", 64, "qi")
BUDGET = 120_000                      # expD22 headline: 600 steps x 2 x 100 probes
ADAM_STEPS, ADAM_LR, ADAM_WARMUP = 3000, 3e-3, 100
ADAM_EVALS_PER_STEP = 3               # forward + backward ~ 3 forwards (evals axis only)


# ----------------------------------------------------------------------------- cell plumbing

class Cell:
    def __init__(self, target, N, init, seed=0):
        self.target, self.N, self.init, self.seed = target, N, init, seed
        model_seed = 100 * seed + d22.TARGET_SEEDS[target]
        self.model = d22.build_model(init, N, model_seed)
        self.X_tr, self.y_tr, self.X_ev, self.y_ev, self.y_norm = d22.data_bundle(target)
        self.W = self.model.readout.weight.shape[1]
        self.oracle = zo.FlatLoss(self.X_tr, self.y_tr, self.W)
        self.x0 = d22.get_flat(self.model)
        self.m = self.x0.numel()
        self.probe_seed = model_seed + 7777           # expD22's probe seed

    def rel_l2(self, theta):
        return self.oracle.rel_l2(theta, self.X_ev, self.y_ev, self.y_norm)

    def key(self):
        return f"{self.target}_{self.N}_{self.init}_s{self.seed}"


class Trace:
    def __init__(self, cell):
        self.cell = cell
        self.d = {"iter": [], "evals": [], "rel_l2": [], "train_mse": [], "eps": []}

    def __call__(self, step, theta, mean_loss, eps, evals):
        self.d["iter"].append(int(step))
        self.d["evals"].append(int(evals))
        self.d["rel_l2"].append(self.cell.rel_l2(theta))
        self.d["train_mse"].append(float(mean_loss))
        self.d["eps"].append(float(eps))


def run_adam(cell, steps=ADAM_STEPS, grad="autograd", fd_eps=1e-5, log_every=1, sched_total=ADAM_STEPS):
    """expD16/expD22's plain-Adam reference on the flat vector (cosine, warmup),
    with the gradient from autograd or from coordinate central differences of
    the loss oracle (grad='fd': the n_perturb -> infinity zero-order oracle)."""
    theta = cell.x0.clone().requires_grad_(grad == "autograd")
    opt = torch.optim.Adam([theta], lr=ADAM_LR)
    tr = Trace(cell)
    evals0 = cell.oracle.n_evals
    fd_err = []
    for step in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = d22.lr_at(step, ADAM_LR, ADAM_WARMUP, sched_total)   # Adam's schedule, truncated
        opt.zero_grad(set_to_none=True)
        if grad == "autograd":
            loss = ((cell.oracle.predict(theta[None])[0] - cell.oracle.y) ** 2).mean()
            loss.backward()
            cell.oracle.n_evals += ADAM_EVALS_PER_STEP
            mean_loss = float(loss)
        else:
            with torch.no_grad():
                g = zo.fd_gradient(theta, cell.oracle, fd_eps)
                mean_loss = cell.oracle.loss(theta)
                if step % 50 == 0:
                    with torch.enable_grad():
                        th = theta.detach().clone().requires_grad_(True)
                        l_ = ((cell.oracle.predict(th[None])[0] - cell.oracle.y) ** 2).mean()
                        g_true = torch.autograd.grad(l_, th)[0]
                    fd_err.append((step, float((g - g_true).norm() / g_true.norm())))
                theta.grad = g
        opt.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            tr(step + 1, theta.detach(), mean_loss, float("nan"), cell.oracle.n_evals - evals0)
    info = {"evals": cell.oracle.n_evals - evals0, "steps_run": steps}
    if fd_err:
        info["fd_grad_rel_err"] = fd_err
    return theta.detach(), tr.d, info


# ----------------------------------------------------------------------------- arms

def _cdrge(**kw):
    return ("cdrge", kw)


LRSEARCH = dict(lr_min=1e-5, lr_max=0.1, depth=4, probe_steps=1, patience=15,
                threshold=0.005, ema_alpha=0.1)          # upstream README section 6.3

ARMS = {
    # references
    "adam": ("adam", dict(steps=ADAM_STEPS)),
    "fdgrad_adam": ("adam", dict(steps=400, grad="fd")),                # 2m evals/step, Adam's 3000-step schedule truncated
    # expD22 headline (regression line) and the author tie
    "cdrge_adam": _cdrge(n_perturb=100, eps=1e-3, beta1=0.9, beta2=0.999, adam_lr=0.01,
                         cosine=True, warmup=100),
    "cdrge_lr_eq_eps": _cdrge(n_perturb=100, eps=3e-3),
    # the four solvers expD22 never ported, and the LR search
    "spsa15_asrun": _cdrge(n_perturb=100, eps=3e-3),                     # == lr = eps
    "spsa15": _cdrge(n_perturb=100, eps=1e-1, curv_alpha=0.5, lam_reg=1.0),  # tuned: alpha, eps grid
    "lrsearch_1spsa": _cdrge(n_perturb=100, eps=1e-2, lr_search=LRSEARCH),
    "lrsearch_spsa15": _cdrge(n_perturb=100, eps=1e-2, curv_alpha=0.5, lr_search=LRSEARCH),
    "spsa2": ("spsa2", dict(n_perturb=33, eps=1e-2)),
    "sanger": ("sanger", dict(n_perturb=100, eps=1e-3, lr=3e-3, rank=8)),
    "bandit": _cdrge(n_perturb=100, eps=3e-5, bandit=dict(temperature=1e-4, min_fd=1e-3, ema=0.5)),
    "bandit_active": _cdrge(n_perturb=100, eps=3e-5, bandit=dict(temperature=1e-4, min_fd=0.0, ema=0.5)),
    "upstream_rmsprop": _cdrge(n_perturb=100, eps=1e-3, lr=1e-2, beta1=0.9, beta2=0.999,
                               upstream_beta2=True),                      # lr tuned (1e-3, 3e-3, 1e-2)
    "probe_precond": _cdrge(n_perturb=100, eps=1e-3, lr=1e-2, beta1=0.9, beta2=0.999,
                            upstream_beta2=True, probe_precond=True),
}

LONG_ARMS = {
    "adam": ("adam", dict(steps=ADAM_STEPS)),
    "cdrge_adam_3000": _cdrge(n_perturb=100, eps=1e-3, beta1=0.9, beta2=0.999, adam_lr=0.01,
                              cosine=True, warmup=100),
    "spsa15_3000": _cdrge(n_perturb=100, eps=1e-1, curv_alpha=0.5, lam_reg=1.0),
}


def run_arm(cell, name, kind, kw, budget=BUDGET, steps=None):
    t0 = time.time()
    tr = Trace(cell)
    kw = dict(kw)
    if kind == "adam":
        theta, d, info = run_adam(cell, **kw)
        tr.d = d
    else:
        n_per_step = 2 * kw["n_perturb"] if kind != "spsa2" else 6 * kw["n_perturb"]
        if steps is None:
            steps = max(1, budget // n_per_step)
        if kind == "cdrge":
            theta, info = zo.cdrge_minimize(cell.x0, cell.oracle, steps=steps, seed=cell.probe_seed,
                                            callback=tr, max_evals=budget if steps * n_per_step > budget else None, **kw)
        elif kind == "spsa2":
            theta, info = zo.spsa2_minimize(cell.x0, cell.oracle, steps=steps, seed=cell.probe_seed,
                                            callback=tr, **kw)
        elif kind == "sanger":
            theta, info = zo.sanger_minimize(cell.x0, cell.oracle, steps=steps, seed=cell.probe_seed,
                                             callback=tr, **kw)
        else:
            raise ValueError(kind)
        info.pop("loss_trace", None)
        info.pop("var_ratio_trace", None)
    row = {"target": cell.target, "N": cell.N, "init": cell.init, "seed": cell.seed,
           "opt": name, "kind": kind, "params": {k: v for k, v in kw.items()},
           "final_rel_l2": tr.d["rel_l2"][-1], "best_rel_l2": min(tr.d["rel_l2"]),
           "evals": info["evals"], "info": info,
           "wall_s": round(time.time() - t0, 1), "trace": tr.d}
    print(f"  {cell.key():22s} {name:18s} final={row['final_rel_l2']:.3e} best={row['best_rel_l2']:.3e} "
          f"evals={row['evals']:>8d} ({row['wall_s']}s)", flush=True)
    return row


def _append(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_rows(pattern):
    rows = []
    for p in sorted(DATA_DIR.glob(pattern)):
        rows += [json.loads(l) for l in open(p)]
    return rows


# ----------------------------------------------------------------------------- tuning

def tune(only=None, budget=40_000, seed=0):
    """Short-budget grids on cell A for the new arms.  Criterion: best rel L2
    within 40k evals (200 steps at 100 probes)."""
    cell = Cell(*CELL_A, seed=seed)
    grids = {
        "spsa15": [_cdrge(n_perturb=100, eps=e, curv_alpha=a, lam_reg=1.0)
                   for a in (1.0, 0.5, 0.1) for e in (1e-1, 1e-2, 1e-3)],
        "spsa2": [("spsa2", dict(n_perturb=33, eps=e)) for e in (1e-1, 1e-2, 1e-3)],
        "sanger": [("sanger", dict(n_perturb=100, eps=1e-3, lr=lr, rank=r))
                   for r in (1, 8) for lr in (1e-3, 3e-3, 1e-2)],
        "bandit": [_cdrge(n_perturb=100, eps=e, bandit=dict(temperature=1e-4, min_fd=mf, ema=0.5))
                   for e in (1e-4, 3e-5, 1e-5) for mf in (1e-3, 0.0)],
        "upstream_rmsprop": [_cdrge(n_perturb=100, eps=1e-3, lr=lr, beta1=0.9, beta2=0.999,
                                    upstream_beta2=True) for lr in (1e-3, 3e-3, 1e-2)],
    }
    rows = []
    for arm, cfgs in grids.items():
        if only and arm not in only:
            continue
        for i, (kind, kw) in enumerate(cfgs):
            rows.append(run_arm(cell, f"tune_{arm}_{i}", kind, kw, budget=budget))
            rows[-1]["tune_arm"] = arm
    _append(DATA_DIR / f"tuning_{'_'.join(only) if only else 'all'}.jsonl", rows)


# ----------------------------------------------------------------------------- ceiling analysis

def picard_curve(Phi_tr, y_tr, Phi_ev, y_ev, y_norm, taus):
    """Truncated-SVD lstsq of [Phi, 1] v = y at relative cutoffs tau; returns
    (rel L2 on the eval grid per tau, sigma/sigma_1, |u_k^T y|/|y|)."""
    Aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
    U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
    yv = y_tr.ravel()
    coef = U.T @ yv
    errs = []
    for tau in taus:
        keep = s > tau * s[0]
        sol = Vt.T @ (np.where(keep, coef / np.where(keep, s, 1.0), 0.0))
        pred = Phi_ev @ sol[:-1] + sol[-1]
        errs.append(float(np.linalg.norm(pred - y_ev.ravel()) / y_norm))
    return np.array(errs), s / s[0], np.abs(coef) / np.linalg.norm(yv)


def zo_newton_readout(cell, taus, fd_eps=1e-1, refine=3):
    """The strongest zero-order method on the most favourable subproblem: the
    readout block (loss exactly quadratic, Phi frozen at the QI geometry),
    coordinate FD gradient + coordinate FD Hessian from the fp64 loss oracle,
    one eigen-truncated Newton step at each cutoff tau; then `refine` further
    FD-Newton steps at the best tau.  Returns dict of results."""
    Phi = cell.oracle.features(cell.x0)
    ro = zo.ReadoutLoss(Phi, cell.y_tr)
    v0 = torch.cat([cell.x0[2 * cell.W:3 * cell.W], cell.x0[3 * cell.W:]])
    Phi_ev = cell.oracle.features(cell.x0, x=cell.X_ev)

    def rel(v):
        pred = Phi_ev @ v[:-1] + v[-1]
        return float(torch.linalg.norm(pred - cell.y_ev.reshape(-1)) / cell.y_norm)

    t0 = time.time()
    g = zo.fd_gradient(v0, ro, 1e-3)
    H = zo.fd_hessian(v0, ro, fd_eps)
    evals_H = ro.n_evals
    Aug = torch.cat([Phi, torch.ones(Phi.shape[0], 1)], 1)
    H_true = 2 * Aug.T @ Aug / Phi.shape[0]
    h_err = float((H - H_true).abs().max())
    lam_true = torch.linalg.eigvalsh(H_true)
    lam_fd = torch.linalg.eigvalsh(H)
    out = {"evals_hessian": evals_H, "hessian_abs_err": h_err,
           "hessian_diag_max": float(H_true.diagonal().max()),
           "lam_max": float(lam_true.max()), "taus": list(map(float, taus)), "one_step": [], "kept": []}
    for tau in taus:
        v1, kept = zo.truncated_newton_step(v0, g, H, tau)
        out["one_step"].append(rel(v1))
        out["kept"].append(kept)
    best_i = int(np.argmin(out["one_step"]))
    tau_b = taus[best_i]
    v = v0.clone()
    traj = [rel(v)]
    for _ in range(refine):
        g = zo.fd_gradient(v, ro, 1e-3)
        H = zo.fd_hessian(v, ro, fd_eps)
        v, _ = zo.truncated_newton_step(v, g, H, tau_b)
        traj.append(rel(v))
    out.update({"best_tau": float(tau_b), "best_one_step": out["one_step"][best_i],
                "refine_traj": traj, "evals_total": ro.n_evals, "wall_s": round(time.time() - t0, 1),
                "lam_fd_neg_count": int((lam_fd < 0).sum()),
                "lam_true_rel": (lam_true / lam_true.max()).tolist()})
    return out


def ceiling(cells=None, seed=0):
    taus = np.logspace(-18, 0, 37)
    rows = []
    for target in TARGETS:
        for N in WIDTHS_N:
            if cells and (target, N) not in cells:
                continue
            cell = Cell(target, N, "qi", seed=seed)
            Phi_tr = cell.oracle.features(cell.x0).numpy()
            Phi_ev = cell.oracle.features(cell.x0, x=cell.X_ev).numpy()
            errs, sig, coef = picard_curve(Phi_tr, cell.y_tr.numpy(), Phi_ev, cell.y_ev.numpy(),
                                           cell.y_norm, taus)
            t0 = time.time()
            zn = zo_newton_readout(cell, taus)
            row = {"target": target, "N": N, "init": "qi", "seed": seed, "m_readout": cell.W + 1,
                   "taus": taus.tolist(), "picard_rel_l2": errs.tolist(), "sigma_rel": sig.tolist(),
                   "picard_coef": coef.tolist(), "zo_newton": zn,
                   "lstsq_floor": float(errs[np.argmin(np.abs(taus - 1e-13))])}
            print(f"  {target:8s} N={N:3d}  lstsq(1e-13)={row['lstsq_floor']:.2e}  "
                  f"picard(1e-8)={errs[np.argmin(np.abs(taus-1e-8))]:.2e}  "
                  f"ZO-Newton best={zn['best_one_step']:.2e} at tau={zn['best_tau']:.0e} "
                  f"(H err {zn['hessian_abs_err']:.1e}, {zn['evals_total']} evals, {time.time()-t0:.0f}s)",
                  flush=True)
            rows.append(row)
    _append(DATA_DIR / "ceiling.jsonl", rows)


def spectra(seed=0):
    """Cell A: full-parameter Hessian spectrum at the qi init and at Adam's
    3000-step plateau; the equilibrated spectrum; the z^T H z distribution."""
    cell = Cell(*CELL_A, seed=seed)

    def L(th):
        return ((cell.oracle.predict(th[None])[0] - cell.oracle.y) ** 2).mean()

    out = {"target": cell.target, "N": cell.N, "m": cell.m}
    theta_plateau, tr, _ = run_adam(cell, log_every=100)
    out["adam_final_rel_l2"] = tr["rel_l2"][-1]
    for tag, th in [("init", cell.x0), ("adam_plateau", theta_plateau)]:
        H = torch.autograd.functional.hessian(L, th)
        lam = torch.linalg.eigvalsh(H)
        D = H.diagonal().abs().sqrt()
        le = torch.linalg.eigvalsh(H / (D[:, None] * D[None, :]))
        Z = zo.rademacher(cell.m, torch.Generator().manual_seed(1), 4000)
        q = torch.einsum("jm,mk,jk->j", Z, H, Z)
        th_ = th.clone().requires_grad_(True)
        g = torch.autograd.grad(L(th_), th_)[0]
        out[tag] = {"lam": lam.tolist(), "lam_equilibrated": le.tolist(), "trace": float(H.trace()),
                    "zHz_over_trH": (q / H.trace()).tolist(), "loss": float(L(th)),
                    "grad_norm": float(g.norm()), "rel_l2": cell.rel_l2(th),
                    "zHz_std_pred": float(torch.sqrt(2 * ((H ** 2).sum() - (H.diagonal() ** 2).sum())))}
        print(f"  {tag}: lam_max={float(lam.max()):.3e} lam_min={float(lam.min()):.3e} "
              f"trH={float(H.trace()):.3e} kappa={float(lam.max()/lam.abs().min()):.2e} "
              f"kappa_eq={float(le.max()/le.abs().min()):.2e} rel_l2={out[tag]['rel_l2']:.2e}")
    _append(DATA_DIR / "spectra.jsonl", [out])


# ----------------------------------------------------------------------------- plotting

STYLE = {
    "adam": ("#7f7f7f", "-", "Adam (autograd gradient)"),
    "fdgrad_adam": ("#000000", "--", "Adam on coordinate-FD gradient (2m evals/step)"),
    "cdrge_adam": ("#d62728", "-", "CD-RGE Adam-style, n=100 (expD22 headline)"),
    "cdrge_adam_3000": ("#d62728", "-", "CD-RGE Adam-style, n=100, 3000 it"),
    "cdrge_lr_eq_eps": ("#2ca02c", "-", "CD-RGE lr=eps=3e-3"),
    "spsa15_asrun": ("#2ca02c", ":", "1.5-SPSA as run (== lr=eps)"),
    "spsa15": ("#ff7f0e", "-", "1.5-SPSA curvature-normalised (tuned)"),
    "spsa15_3000": ("#ff7f0e", "-", "1.5-SPSA curvature-normalised, 3000 it"),
    "lrsearch_1spsa": ("#1f77b4", "-", "CD-RGE + binary LR search (upstream 6.3)"),
    "lrsearch_spsa15": ("#1f77b4", "--", "1.5-SPSA + binary LR search"),
    "spsa2": ("#9467bd", "-", "2SPSA (upstream)"),
    "sanger": ("#8c564b", "-", "Sanger-SPSA (tuned)"),
    "bandit": ("#e377c2", "-", "Bandit-SPSA (upstream thresholds)"),
    "bandit_active": ("#e377c2", "--", "Bandit-SPSA (reservoir active)"),
    "upstream_rmsprop": ("#bcbd22", "-", "CD-RGE upstream beta1/beta2 (v init ones)"),
    "probe_precond": ("#17becf", "-", "CD-RGE + probe preconditioning"),
}


def _setup(ax, xlab, ylab=None):
    ax.set_ylim(1e-16, 1e1)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)


def plot_cell_a():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _load_rows("arms_*.jsonl") + _load_rows("long_*.jsonl")
    rows = [r for r in rows if (r["target"], r["N"], r["init"]) == CELL_A and r["seed"] == 0]
    if not rows:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    seen = {}
    for r in rows:                              # latest row per arm wins (files sort by tag)
        if r["opt"] in STYLE:
            seen[r["opt"]] = r
    for name, r in seen.items():
        c, ls, lab = STYLE[name]
        tr = r["trace"]
        axes[0].semilogy(tr["iter"], tr["rel_l2"], color=c, ls=ls, lw=1.3, label=lab)
        axes[1].loglog(np.maximum(tr["evals"], 1), tr["rel_l2"], color=c, ls=ls, lw=1.3, label=lab)
    _setup(axes[0], "iteration", "eval rel $L_2$")
    _setup(axes[1], "loss-oracle evaluations (Adam: 3 per iteration)")
    axes[0].set_xlim(0, 3000)
    axes[1].set_xlim(1e2, 2e6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle("expD23 cell A: sine, $N=64$, qi init -- every zero_order_rnn solver "
                 "and the FD-gradient oracle vs Adam", y=0.80, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig(FIG_DIR / "expD23_cellA.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", FIG_DIR / "expD23_cellA.png")


def plot_ceiling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _load_rows("ceiling.jsonl")
    if not rows:
        return
    arms = _load_rows("arms_*.jsonl") + _load_rows("grid_*.jsonl") + _load_rows("long_*.jsonl")
    d22rows = []
    for p in sorted(D22_DATA.glob("trajectories_qi__*.jsonl")):
        d22rows += [json.loads(l) for l in open(p)]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TARGETS), len(WIDTHS_N), figsize=(14, 14), sharex=True, sharey=True)
    for i, target in enumerate(TARGETS):
        for j, N in enumerate(WIDTHS_N):
            ax = axes[i][j]
            rv = [r for r in rows if r["target"] == target and r["N"] == N]
            if not rv:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color="grey")
                _setup(ax, "")
                continue
            r = rv[-1]
            taus = np.array(r["taus"])
            ax.loglog(taus, r["picard_rel_l2"], color="k", lw=1.6,
                      label="truncated-SVD lstsq on $\\Phi$ (the geometry's floor curve)")
            zn = r["zo_newton"]
            ax.loglog(taus, zn["one_step"], color="#d62728", lw=1.4, ls="--",
                      label="one FD-Hessian Newton step from the loss oracle")
            ax.axvline(2 ** -52, color="k", ls=":", lw=1, label="$\\varepsilon_{mach}$ (lstsq on $\\Phi$ can go here)")
            ax.axvline(2 ** -26, color="#d62728", ls=":", lw=1, label="$\\sqrt{\\varepsilon_{mach}}$ (loss-oracle curvature floor)")
            best_zo = [a["best_rel_l2"] for a in arms if a["target"] == target and a["N"] == N
                       and a["init"] == "qi" and a["kind"] != "adam"]
            if best_zo:
                ax.axhline(min(best_zo), color="#ff7f0e", lw=1.2, label="best zero-order run (this experiment)")
            ad = [a["final_rel_l2"] for a in d22rows if a["target"] == target and a["N"] == N and a["opt"] == "adam"]
            if ad:
                ax.axhline(np.median(ad), color="#7f7f7f", lw=1.2, label="Adam, 3000 it (expD22 median)")
            ax.set_ylim(1e-16, 1e1)
            ax.set_xlim(1e-18, 1)
            ax.grid(True, alpha=0.3, which="both")
            if i == 0:
                ax.set_title(f"$N={N}$")
            if i == len(TARGETS) - 1:
                ax.set_xlabel("singular-value cutoff $\\sigma/\\sigma_1$")
            if j == 0:
                ax.set_ylabel(f"{target}\neval rel $L_2$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle("expD23: what the loss oracle can resolve -- Picard curves of the QI geometry vs the "
                 "finite-difference Hessian floor", y=0.925, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIG_DIR / "expD23_ceiling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", FIG_DIR / "expD23_ceiling.png")


def plot_spectra():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _load_rows("spectra.jsonl")
    if not rows:
        return
    r = rows[-1]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for tag, c in [("init", "#1f77b4"), ("adam_plateau", "#d62728")]:
        lam = np.abs(np.array(r[tag]["lam"]))[::-1]
        le = np.abs(np.array(r[tag]["lam_equilibrated"]))[::-1]
        lam.sort(); le.sort()
        lam, le = lam[::-1], le[::-1]
        axes[0].semilogy(np.arange(1, lam.size + 1), lam / lam[0], color=c, lw=1.4,
                         label=f"$|\\lambda_k|/\\lambda_1$ at {tag.replace('_', ' ')}")
        axes[0].semilogy(np.arange(1, le.size + 1), le / le[0], color=c, lw=1.2, ls="--",
                         label=f"same, diagonally equilibrated")
        q = np.array(r[tag]["zHz_over_trH"])
        axes[1].hist(q, bins=80, range=(-1, 6), histtype="step", color=c, lw=1.4,
                     label=f"$z^T H z/\\mathrm{{tr}}H$ at {tag.replace('_', ' ')} (std {q.std():.2f})")
    axes[0].axhline(2 ** -52, color="k", ls=":", lw=1, label="$\\varepsilon_{mach}$")
    axes[0].set_xlabel("eigenvalue index $k$"); axes[0].set_ylabel("relative magnitude")
    axes[0].set_ylim(1e-26, 2); axes[0].grid(True, alpha=0.3, which="both")
    axes[1].axvline(1.0, color="k", ls=":", lw=1)
    axes[1].set_xlabel("1.5-SPSA curvature normaliser $z^T H z$, in units of tr$H$"); axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.3)
    cr = _load_rows("ceiling.jsonl")
    for N, c in zip(WIDTHS_N, ["#2ca02c", "#ff7f0e", "#9467bd"]):
        rv = [x for x in cr if x["target"] == "sine" and x["N"] == N]
        if rv:
            s = np.array(rv[-1]["sigma_rel"])
            axes[2].semilogy(np.arange(1, s.size + 1), s, color=c, lw=1.4, label=f"$\\sigma_k/\\sigma_1$ of $[\\Phi,1]$, $N={N}$")
            p = np.array(rv[-1]["picard_coef"])
            axes[2].semilogy(np.arange(1, p.size + 1), p, color=c, lw=0.7, ls="--", alpha=0.6, label=f"$|u_k^T y|/\\|y\\|$, $N={N}$")
    axes[2].axhline(2 ** -52, color="k", ls=":", lw=1, label="$\\varepsilon_{mach}$")
    axes[2].axhline(2 ** -26, color="#d62728", ls=":", lw=1, label="$\\sqrt{\\varepsilon_{mach}}$")
    axes[2].set_ylim(1e-20, 2); axes[2].set_xlabel("singular index $k$ (sine)"); axes[2].grid(True, alpha=0.3, which="both")
    for ax, nc in zip(axes, (2, 1, 2)):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=nc, fontsize=8, frameon=False, borderaxespad=0)
    fig.suptitle("expD23 cell A spectra: the Hessian is gapless over 23 decades; the 1.5-SPSA normaliser is a random scalar", y=1.06, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIG_DIR / "expD23_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", FIG_DIR / "expD23_spectra.png")


def plot_grid():
    """Breadth at N = 64 (the grid was cut to one width for cost): the three new
    arms that make progress, against expD22's Adam / CD-RGE lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _load_rows("grid_*.jsonl")
    if not rows:
        return
    d22rows = []
    for p in sorted(D22_DATA.glob("trajectories_qi__*.jsonl")):
        d22rows += [json.loads(l) for l in open(p)]
    N = 64
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 5), sharey=True)
    labels_grid = {"spsa15": "1.5-SPSA curvature-normalised ($\\alpha=0.5$, $\\epsilon=10^{-2}$)",
                   "lrsearch_1spsa": "CD-RGE + binary LR search (upstream 6.3)",
                   "sanger": "Sanger-SPSA (rank 8)"}
    for j, target in enumerate(TARGETS):
        ax = axes[j]
        for opt, (c, lab) in {"adam": ("#7f7f7f", "Adam (expD22, median of seeds, band = min-max)"),
                              "cdrge_adam_cos": ("#d62728", "CD-RGE Adam-style (expD22, median)")}.items():
            rv = [r for r in d22rows if r["target"] == target and r["N"] == N and r["opt"] == opt]
            if rv:
                it, med, lo, hi = d22._band(rv)
                ax.semilogy(it, med, color=c, lw=1.4, label=lab)
                ax.fill_between(it, lo, hi, color=c, alpha=0.15, lw=0)
        rv_all = [r for r in rows if r["target"] == target and r["N"] == N and r["init"] == "qi"]
        for name, lab in labels_grid.items():
            rv = [r for r in rv_all if r["opt"] == name]
            if rv:
                c, ls, _ = STYLE[name]
                tr = rv[-1]["trace"]
                ax.semilogy(tr["iter"], tr["rel_l2"], color=c, ls=ls, lw=1.3, label=lab)
        if not rv_all:
            ax.text(0.5, 0.5, "not run\n(grid cut for cost)", transform=ax.transAxes, ha="center", color="grey")
        ax.set_ylim(1e-16, 1e1); ax.set_xlim(0, 3000); ax.grid(True, alpha=0.3, which="both")
        ax.set_title(f"{target}, $N={N}$"); ax.set_xlabel("iteration")
        if j == 0:
            ax.set_ylabel("eval rel $L_2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle("expD23 breadth at $N=64$, qi init, seed 0: the new zero-order arms vs expD22's lines (600 steps = 120k evaluations)",
                 y=0.86, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(FIG_DIR / "expD23_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", FIG_DIR / "expD23_grid.png")


# ----------------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--spectra", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--cells", type=str, default=None, help="e.g. sine:64,runge:128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in args.cells.split(",")] if args.cells else None

    if args.tune:
        tune(only=only, seed=args.seed)
    if args.arms:
        cell = Cell(*CELL_A, seed=args.seed)
        rows = [run_arm(cell, n, k, kw, budget=args.budget) for n, (k, kw) in ARMS.items()
                if not only or n in only]
        _append(DATA_DIR / f"arms_{args.tag or ('_'.join(only) if only else 'all')}_s{args.seed}.jsonl", rows)
    if args.long:
        cell = Cell(*CELL_A, seed=args.seed)
        rows = [run_arm(cell, n, k, kw, steps=3000, budget=10 ** 9) for n, (k, kw) in LONG_ARMS.items()
                if not only or n in only]
        _append(DATA_DIR / f"long_{args.tag or ('_'.join(only) if only else 'all')}_s{args.seed}.jsonl", rows)
    if args.ceiling:
        ceiling(cells=cells, seed=args.seed)
    if args.spectra:
        spectra(seed=args.seed)
    if args.grid:
        arms = only or ["spsa15", "lrsearch_1spsa", "sanger", "spsa2", "bandit_active"]
        rows = []
        for target in TARGETS:
            for N in [64, 128]:
                if cells and (target, N) not in cells:
                    continue
                cell = Cell(target, N, "qi", seed=args.seed)
                for n in arms:
                    k, kw = ARMS[n]
                    rows.append(run_arm(cell, n, k, kw, budget=args.budget))
                _append(DATA_DIR / f"grid_{args.tag or 'all'}_s{args.seed}.jsonl", rows)
                rows = []
    if args.plot:
        plot_cell_a()
        plot_ceiling()
        plot_spectra()
        plot_grid()
