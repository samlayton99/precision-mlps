"""Standard second-order optimizers vs Adam, in the two expD02 regimes.

Objective: compare Adam to standard second-order optimizers on the SAME init
(free QI geometry) in the two regimes the expD02 init_slice plots use:

  regime "both"  -- all params optimized, NO least squares
                    (the analogue of expD02 `adam_both`: init + pure GD)
  regime "lstsq" -- geometry optimized, readout solved by lstsq every step
                    (the analogue of expD02 `adam_first_lstsq`: GD + lstsq on top)

Optimizer here is LBFGS (the standard quasi-Newton, and the repo's documented
second stage). Gauss-Newton results for the same two regimes already come from
adam_gn_fullnet_probe.py ("both") and varpro_gn_probe.py ("lstsq").

torch fp64 on DEVICE. Usage:
    uv run --extra dev python experiments/expD04_varpro/second_order_probe.py --mode smoke
    uv run --extra dev python experiments/expD04_varpro/second_order_probe.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_D05_PATH = REPO_ROOT / "experiments" / "expD05_scale_init_story" / "run.py"
_spec = importlib.util.spec_from_file_location("expd05_run", _D05_PATH)
d05 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["expd05_run"] = d05
_spec.loader.exec_module(d05)  # type: ignore[union-attr]

from src import DEVICE  # noqa: E402
from src.data.targets import get_target  # noqa: E402
from src.training.ssbroyden import ssbroyden_minimize  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro"


def out_path(optimizers):
    """Per-optimizer file when a single optimizer is requested, else the combined one."""
    if optimizers and len(optimizers) == 1:
        return RESULTS_DIR / f"second_order_{optimizers[0]}_summary.csv"
    return RESULTS_DIR / "second_order_summary.csv"

TARGETS = ("sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed")  # expD02's 6
RESOLUTIONS = (32, 64, 128, 256, 512)
SEEDS = (0, 1)
FAMILY = "qi_geom_random_readout"   # free QI geometry (readout not handed over)
REGIMES = ("both", "lstsq")
OPTIMIZERS = ("lbfgs", "ssbroyden")
N_TRAIN, N_EVAL = 2003, 4001
LBFGS_ROUNDS, LBFGS_MAX_ITER = 8, 250
SSB_STEPS = 2000       # SSBroyden outer steps (each with its own Wolfe search)
TOL = 1e-23            # convergence tolerance for second-order optimizers


def rel_l2_t(pred, y):
    return float(torch.linalg.norm(pred - y) / torch.linalg.norm(y))


def solve_readout_t(phi, y):
    """Readout solve on [phi, 1] via QR (backward-stable).

    Normal equations square the condition number and cap the attainable readout
    accuracy around 1e-9 -- which would masquerade as an optimizer 'wall'. QR
    keeps the full fp64 accuracy. Fall back to a bounded ridge solve only if the
    geometry has driven the feature matrix degenerate (so it stays finite).
    """
    A = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)], dim=1)
    try:
        Q, R = torch.linalg.qr(A)
        rdiag = torch.diagonal(R).abs()
        if float(rdiag.min()) > 1e-13 * float(rdiag.max()):
            sol = torch.linalg.solve_triangular(R, Q.T @ y.reshape(-1, 1), upper=True).reshape(-1)
            if torch.isfinite(sol).all():
                return A, sol
    except RuntimeError:
        pass
    AtA = A.T @ A
    Aty = A.T @ y.reshape(-1)
    eps = 1e-14 * torch.diagonal(AtA).abs().max().clamp_min(1e-300)
    ridge = eps * torch.eye(AtA.shape[0], dtype=A.dtype, device=A.device)
    sol = torch.linalg.solve(AtA + ridge, Aty)
    return A, sol


def run_case_ssbroyden(regime, target, resolution, seed):
    """SSBroyden on a flat parameter vector.
    regime 'both'  -> x = [a, b, v, c]   (readout learned, no lstsq)
    regime 'lstsq' -> x = [a, b]         (readout solved by lstsq each eval)
    """
    geom = d05.geometry_for_resolution(resolution)
    W = geom.width
    st = d05.build_initial_state(FAMILY, target, geom, seed)
    t = get_target(target)
    xt = torch.linspace(-1, 1, N_TRAIN, dtype=torch.float64, device=DEVICE)
    yt = torch.as_tensor(t.fn_numpy(xt.cpu().numpy()), dtype=torch.float64, device=DEVICE)
    xe = torch.linspace(-1, 1, N_EVAL, dtype=torch.float64, device=DEVICE)
    ye = torch.as_tensor(t.fn_numpy(xe.cpu().numpy()), dtype=torch.float64, device=DEVICE)

    parts = [st.input_weights, st.input_biases]
    if regime == "both":
        parts += [st.readout_weights, np.array([st.readout_bias])]
    x0 = torch.as_tensor(np.concatenate(parts), dtype=torch.float64, device=DEVICE)

    def predict(x, xq):
        a, b = x[:W], x[W:2 * W]
        phi_q = torch.tanh(xq.reshape(-1, 1) * a.reshape(1, -1) + b.reshape(1, -1))
        if regime == "both":
            return phi_q @ x[2 * W:3 * W] + x[3 * W]
        phi_t = torch.tanh(xt.reshape(-1, 1) * a.reshape(1, -1) + b.reshape(1, -1))
        with torch.no_grad():
            _, sol = solve_readout_t(phi_t, yt)
        return phi_q @ sol[:W] + sol[W]

    def fg(x):
        xx = x.detach().clone().requires_grad_(True)
        loss = torch.mean((predict(xx, xt) - yt) ** 2)
        if not torch.isfinite(loss):
            return float("inf"), torch.zeros_like(x)
        (g,) = torch.autograd.grad(loss, xx)
        return float(loss), g.detach()

    @torch.no_grad()
    def eval_at(x):
        return rel_l2_t(predict(x, xe), ye)

    best = eval_at(x0)
    seen = {"best": best}

    # track the best eval error along the trajectory, not just the endpoint
    def wrapped_fg(x):
        f, g = fg(x)
        return f, g

    x_star, info = ssbroyden_minimize(x0, wrapped_fg, max_steps=SSB_STEPS, tol=TOL)
    seen["best"] = min(seen["best"], eval_at(x_star))
    return W, seen["best"]


def run_case(regime, target, resolution, seed):
    geom = d05.geometry_for_resolution(resolution)
    W = geom.width
    st = d05.build_initial_state(FAMILY, target, geom, seed)
    t = get_target(target)
    xt = torch.linspace(-1, 1, N_TRAIN, dtype=torch.float64, device=DEVICE)
    yt = torch.as_tensor(t.fn_numpy(xt.cpu().numpy()), dtype=torch.float64, device=DEVICE)
    xe = torch.linspace(-1, 1, N_EVAL, dtype=torch.float64, device=DEVICE)
    ye = torch.as_tensor(t.fn_numpy(xe.cpu().numpy()), dtype=torch.float64, device=DEVICE)

    a = torch.tensor(st.input_weights, dtype=torch.float64, device=DEVICE, requires_grad=True)
    b = torch.tensor(st.input_biases, dtype=torch.float64, device=DEVICE, requires_grad=True)
    if regime == "both":
        v = torch.tensor(st.readout_weights, dtype=torch.float64, device=DEVICE, requires_grad=True)
        c = torch.tensor([st.readout_bias], dtype=torch.float64, device=DEVICE, requires_grad=True)
        params = [a, b, v, c]
    else:
        params = [a, b]

    def feats(x):
        return torch.tanh(x.reshape(-1, 1) * a.reshape(1, -1) + b.reshape(1, -1))

    def train_loss():
        phi = feats(xt)
        if regime == "both":
            pred = phi @ v + c
        else:
            # Solve the readout WITHOUT autograd. At the lstsq solution the residual
            # is orthogonal to col([phi,1]), so the gradient taken with the readout
            # held fixed IS the Kaufman/VarPro reduced gradient -- and this avoids
            # the unstable backward through torch.linalg.lstsq.
            with torch.no_grad():
                _, sol = solve_readout_t(phi, yt)
            pred = phi @ sol[:W] + sol[W]
        return torch.mean((pred - yt) ** 2)

    @torch.no_grad()
    def eval_rel():
        phi_t = feats(xt)
        if regime == "both":
            pe = feats(xe) @ v + c
        else:
            _, sol = solve_readout_t(phi_t, yt)
            pe = feats(xe) @ sol[:W] + sol[W]
        return rel_l2_t(pe, ye)

    best = eval_rel()
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_MAX_ITER,
                            history_size=100, line_search_fn="strong_wolfe",
                            tolerance_grad=TOL, tolerance_change=TOL)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = train_loss()
        if not torch.isfinite(loss):
            # make the strong-Wolfe line search back off instead of diverging
            loss = torch.full_like(loss, 1e30)
            return loss
        loss.backward()
        return loss

    for _ in range(LBFGS_ROUNDS):
        try:
            opt.step(closure)
        except RuntimeError:
            break
        r = eval_rel()
        if not np.isfinite(r):
            break
        best = min(best, r)
    return W, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--optimizers", type=lambda s: tuple(x.strip() for x in s.split(",")),
                    default=None, help="comma list from: lbfgs,ssbroyden")
    args = ap.parse_args(argv)
    targets = ("runge",) if args.mode == "smoke" else TARGETS
    resolutions = (64, 128) if args.mode == "smoke" else RESOLUTIONS
    seeds = (0,) if args.mode == "smoke" else SEEDS

    optimizers = args.optimizers or OPTIMIZERS
    rows = []
    total = len(REGIMES) * len(optimizers) * len(targets) * len(resolutions) * len(seeds)
    i = 0
    for regime in REGIMES:
        for opt_name in optimizers:
            for target in targets:
                for resolution in resolutions:
                    for seed in seeds:
                        i += 1
                        if opt_name == "ssbroyden":
                            W, best = run_case_ssbroyden(regime, target, resolution, seed)
                        else:
                            W, best = run_case(regime, target, resolution, seed)
                        rows.append({"regime": regime, "optimizer": opt_name, "target": target,
                                     "resolution": resolution, "width": W, "seed": seed,
                                     "best_eval_rel_l2": best})
                        print(f"[{i:3d}/{total}] {regime:>5s} {opt_name:>9s} {target:>12s} "
                              f"N={resolution:<4d} W={W:<4d} seed={seed} best={best:.2e}", flush=True)
    out = out_path(optimizers)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
