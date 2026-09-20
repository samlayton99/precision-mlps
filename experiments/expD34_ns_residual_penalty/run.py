"""A differentiable matrix-product residual penalty, with one ordinary Adam.

Training uses no SVD or coefficient solve. Saved-state evaluations do use SVD.
The penalty is a smooth spectral surrogate, not an exact VarPro floor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD31_split_adam import run as previous
from experiments.expD31_split_adam.mu_refit_sweep import refit

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD34_ns_residual_penalty"
COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def config():
    return previous.config() | yaml.safe_load((HERE / "config.yaml").read_text())


def ns_factor(A, steps=5, refinements=0):
    """Differentiate through the Frobenius normalization and every iteration."""
    if A.ndim != 2 or not A.is_floating_point():
        raise ValueError("A must be a real floating-point matrix")
    norm = torch.linalg.vector_norm(A)
    if not bool(torch.isfinite(norm)) or bool(norm == 0):
        raise ValueError("A must have a finite nonzero Frobenius norm")
    # Work with the smaller Gram matrix. Transposition retains autograd.
    transposed = A.shape[0] < A.shape[1]
    X = (A.T if transposed else A) / norm
    a, b, c = COEFFICIENTS
    for _ in range(steps):
        gram = X.T @ X
        X = a * X + X @ (b * gram + c * (gram @ gram))
    for _ in range(refinements):
        X = 1.5 * X - 0.5 * (X @ (X.T @ X))
    return X.T if transposed else X


def residual_penalty(A, y, steps=5, refinements=0):
    X = ns_factor(A, steps, refinements)
    residual = y - X @ (X.T @ y)
    return 0.5 * residual.square().sum()


def scalar_response(s, steps=5, refinements=0):
    s = np.asarray(s, dtype=np.float64).copy()
    a, b, c = COEFFICIENTS
    for _ in range(steps):
        s = a*s + b*s**3 + c*s**5
    for _ in range(refinements):
        s = 1.5*s - 0.5*s**3
    return s


def loss_terms(p, x, y, cfg, refinements):
    h = torch.tanh(x[:, None] * p["a"] + p["b"])
    prediction = h @ p["v"][:-1] + p["v"][-1]
    L = 0.5 * (prediction - y).square().mean()
    A = torch.cat((h, torch.ones_like(x[:, None])), dim=1) / len(x)**0.5
    Fhat = residual_penalty(A, y / len(x)**0.5, cfg["ns_steps"], refinements)
    return L, Fhat


def train(cfg, mu, refinements):
    x = previous.profile.previous.midpoint_grid(cfg["n_train"])
    y = previous.profile.previous.matched.target_values(cfg["target"], x, cfg)
    initial = previous.profile.initial_state(cfg["arm"], cfg)
    p = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for k, v in initial.items()}
    tx, ty = torch.tensor(x), torch.tensor(y)
    optimizer = torch.optim.Adam(p.values(), lr=cfg["learning_rate"],
                                 betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"])
    chosen = set(previous.profile.snapshots(cfg["steps"], cfg["diagnostic_snapshots"]))
    history, states, saved = [], {k: [] for k in p}, []
    start = time.perf_counter()
    for step in range(cfg["steps"] + 1):
        eta = previous.learning_rate_at(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = eta
        optimizer.zero_grad(set_to_none=True)
        if mu:
            L, Fhat = loss_terms(p, tx, ty, cfg, refinements)
            objective = L + mu * Fhat
        else:
            prediction = torch.tanh(tx[:, None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
            L = .5*(prediction-ty).square().mean()
            objective, Fhat = L, None
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError(f"nonfinite objective: mu={mu}, step={step}")
        history.append(dict(step=step, learning_rate=eta, L=float(L.detach()),
                            objective=float(objective.detach()),
                            Fhat=float(Fhat.detach()) if Fhat is not None else np.nan,
                            relative_l2=float(torch.sqrt(2*L.detach()/ty.square().mean())),
                            mean_gamma=float(p["a"].detach().abs().mean())))
        if step in chosen:
            saved.append(step)
            for k in p:
                states[k].append(p[k].detach().numpy().copy())
        if step == cfg["steps"]:
            break
        objective.backward()
        if not all(bool(torch.isfinite(v.grad).all()) for v in p.values()):
            raise FloatingPointError(f"nonfinite gradient: mu={mu}, step={step}")
        optimizer.step()
    case = {k: np.asarray([r[k] for r in history]) for k in history[0]}
    case.update({k: np.asarray(v) for k, v in states.items()})
    case.update(saved_steps=np.asarray(saved), mu=np.array(mu), refinements=np.array(refinements),
                training_seconds=np.array(time.perf_counter()-start))
    return case


def evaluate(case, cfg):
    """Offline only: solve readouts and evaluate both spectral surrogates."""
    records = []
    tx = torch.tensor(previous.profile.previous.midpoint_grid(cfg["n_train"]))
    y = previous.profile.previous.matched.target_values(cfg["target"], tx.numpy(), cfg)
    ty = torch.tensor(y)
    for i, step in enumerate(case["saved_steps"]):
        state = {k: case[k][i] for k in ("a", "b", "v")}
        row = refit(state, cfg["target"], cfg)
        p = {k: torch.tensor(v) for k, v in state.items()}
        with torch.no_grad():
            for label, count in (("literal", 0), ("refined", cfg["refinement_steps"])):
                _, Fhat = loss_terms(p, tx, ty, cfg, count)
                row[f"{label}_penalty_relative"] = float(torch.sqrt(2*Fhat/ty.square().mean()))
        records.append(dict(step=int(step), **row))
    for k in records[0]:
        case[f"eval_{k}"] = np.asarray([r[k] for r in records])
    return case


def spectral_figure(cfg):
    import matplotlib.pyplot as plt
    s = np.geomspace(1e-6, 1, 4000)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    for label, refinements, color in (("Five Muon iterations", 0, "#a34783"),
                                      ("Then four refinement steps", cfg["refinement_steps"], "#17887d")):
        q = scalar_response(s, cfg["ns_steps"], refinements)**2
        axes[0].semilogx(s, q, color=color, label=label, lw=1.8)
        axes[1].loglog(s, np.maximum((1-q)**2, 1e-32), color=color, lw=1.8)
    axes[0].axhline(1, color="black", ls=":", label="Exact projector: response = 1")
    axes[0].set(ylabel=r"Response of $X X^T$: $q(s)$", title="Does an in-span component pass unchanged?", ylim=(0, 1.7))
    axes[1].set(ylabel=r"Remaining squared-error weight: $[1-q(s)]^2$",
                title="Penalty on an already representable component", ylim=(1e-32, 2))
    axes[1].text(.03, .06, "Exact floor: zero for every in-span component\nZero weights displayed at $10^{-32}$", transform=axes[1].transAxes, fontsize=10)
    for ax in axes:
        ax.set(xlabel=r"Normalized singular value $s=\sigma/\|A\|_F$", xlim=(s[0], 1))
        ax.grid(alpha=.2)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Refinement improves the projector approximation; very small modes still remain unresolved", fontsize=12)
    fig.savefig(RESULTS / "figures/spectral_response.png", dpi=170)
    plt.close(fig)


def training_figure(cases, cfg):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = plt.cm.viridis(np.linspace(.12, .86, len(cfg["mu_values"])))
    palette = dict(zip(cfg["mu_values"], colors))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for label, case in cases.items():
        mu, refinements = int(case["mu"]), int(case["refinements"])
        color = palette[mu] if mu else ".25"
        style = "--" if mu and not refinements else "-"
        opts = dict(color=color, ls=style, lw=1.6, alpha=.92)
        axes[0, 0].semilogy(case["step"], case["relative_l2"], **opts)
        axes[0, 1].semilogy(case["eval_step"], case["eval_refit_train_relative_l2"], **opts)
        penalty = "refined" if refinements else "literal"
        if mu:
            axes[1, 0].semilogy(case["eval_step"], case[f"eval_{penalty}_penalty_relative"], **opts)
        axes[1, 1].plot(case["step"], case["mean_gamma"], **opts)
    axes[0, 0].set(title="Actual network: trained coefficients", ylabel=r"Relative $L_2$: $\|Av-y\|/\|y\|$")
    axes[0, 1].set(title="Geometry: least-squares evaluation only", ylabel=r"Relative $L_2$: $\|Av_* - y\|/\|y\|$")
    axes[1, 0].set(title="Approximate penalty used by each method", ylabel=r"$\sqrt{2\widehat F}/\|y\|$")
    axes[1, 1].set(title="Scale movement", ylabel=r"Mean scale: $\mathrm{mean}_k\,|a_k|$")
    for ax in axes.flat:
        ax.set(xlabel="Training step", xlim=(0, cfg["steps"]))
        ax.grid(alpha=.2)
    handles = [Line2D([], [], color=".25", label="Ordinary Adam (μ=0)")]
    handles += [Line2D([], [], color=palette[mu], label=f"μ={mu}") for mu in cfg["mu_values"]]
    handles += [Line2D([], [], color=".4", ls="--", label="Five Muon steps"),
                Line2D([], [], color=".4", label="+ four refinement steps")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle(r"Mixed sine · Xavier · one Adam on $L+\mu\widehat F$ · learning rate 0.002" "\n"
                 "All relative errors use the same training samples; coefficient solves never affect training", fontsize=13)
    fig.savefig(RESULTS / "figures/mixed_sine.png", dpi=170)
    plt.close(fig)


def summarize(cases, cfg):
    summary = []
    for name, case in cases.items():
        state = {k: case[k][-1] for k in ("a", "b", "v")}
        checks = [dict(cutoff=cutoff, **refit(state, cfg["target"], cfg, cutoff=cutoff, n_eval=65536))
                  for cutoff in (1e-12, 1e-13, 1e-14)]
        summary.append(dict(name=name, mu=int(case["mu"]), refinements=int(case["refinements"]),
                            final_relative_l2=float(case["relative_l2"][-1]),
                            initial_refit_relative_l2=float(case["eval_refit_train_relative_l2"][0]),
                            final_refit_relative_l2=float(case["eval_refit_train_relative_l2"][-1]),
                            initial_gamma=float(case["mean_gamma"][0]), final_gamma=float(case["mean_gamma"][-1]),
                            training_seconds=float(case["training_seconds"]), final_checks=checks))
    (RESULTS / "data/summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(cfg["threads"])
    for directory in ("figures", "data"):
        (RESULTS / directory).mkdir(parents=True, exist_ok=True)
    specs = [("ordinary", 0, 0)] + [(f"{method}__mu{mu}", mu, count)
             for method, count in (("literal", 0), ("refined", cfg["refinement_steps"]))
             for mu in cfg["mu_values"]]
    cases = {}
    with threadpool_limits(limits=cfg["threads"]):
        for name, mu, refinements in specs:
            path = RESULTS / "data" / f"{name}.npz"
            if args.plot_only:
                with np.load(path) as archive:
                    cases[name] = dict(archive)
            else:
                case = train(cfg, mu, refinements)
                evaluate(case, cfg)
                np.savez_compressed(path, **case)
                cases[name] = case
        reference = cases["ordinary"]
        for case in cases.values():
            for key in ("a", "b", "v"):
                np.testing.assert_array_equal(case[key][0], reference[key][0])
        summary = summarize(cases, cfg)
    (RESULTS / "data/config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    spectral_figure(cfg)
    training_figure(cases, cfg)
    print(json.dumps([{k: v for k, v in row.items() if k != "final_checks"} for row in summary], indent=2))


if __name__ == "__main__":
    main()
