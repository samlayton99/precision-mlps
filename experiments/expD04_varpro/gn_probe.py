"""Gauss-Newton probe -- can a second-order optimizer close QI-geom's stall?

expD06 showed QI-geometry + random readout trained by pure Adam stalls at ~1e-4
(the geometry stays correct, but Adam cannot solve the readout / exploit the
least-squares curvature).  The MSE objective IS a nonlinear least-squares
problem, so Gauss-Newton -- which approximates the Hessian by J^T J -- is the
natural second-order method.  This probe swaps Adam for damped (Levenberg-
Marquardt) Gauss-Newton on the SAME free end-to-end params (a, b, v, c) and the
SAME QI-geom init, and asks whether it reaches the oracle floor.

This is a fast, exploratory precursor to the full VarPro + SSBroyden build; it
reuses expD05's init builders and is pure numpy fp64 (analytic Jacobian).

Model:   f(x) = sum_j v_j tanh(a_j x + b_j) + c
Params:  p = [a (W), b (W), v (W), c (1)]   -> 3W+1

Usage:
    uv run --extra dev python experiments/expD04_varpro/gn_probe.py --mode smoke
    uv run --extra dev python experiments/expD04_varpro/gn_probe.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_D05_PATH = REPO_ROOT / "experiments" / "expD05_scale_init_story" / "run.py"
_spec = importlib.util.spec_from_file_location("expd05_run", _D05_PATH)
d05 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["expd05_run"] = d05
_spec.loader.exec_module(d05)  # type: ignore[union-attr]

from src.data.targets import get_target  # noqa: E402

EXPERIMENT = "expD04_varpro"
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / EXPERIMENT
SUMMARY_CSV = RESULTS_DIR / "gn_probe_summary.csv"

FAMILIES = ("standard_xavier_affine", "qi_geom_random_readout", "exact_qi_oracle")
DEFAULT_TARGETS = ("runge", "exp")
DEFAULT_RESOLUTIONS = (64, 128, 256, 512)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ITERS = 200
DEFAULT_N_TRAIN = 6007   # prime, > 3W+1 at the largest width (keeps GN overdetermined)
DEFAULT_N_EVAL = 4001


def rel_l2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def predict(p: np.ndarray, W: int, x: np.ndarray) -> np.ndarray:
    a, b, v, c = p[:W], p[W:2 * W], p[2 * W:3 * W], p[3 * W]
    phi = np.tanh(x[:, None] * a[None, :] + b[None, :])
    return phi @ v + c


def residual_and_jac(p: np.ndarray, W: int, x: np.ndarray, y: np.ndarray):
    a, b, v, c = p[:W], p[W:2 * W], p[2 * W:3 * W], p[3 * W]
    z = x[:, None] * a[None, :] + b[None, :]
    phi = np.tanh(z)
    sech2 = 1.0 - phi * phi                      # d tanh / d z
    pred = phi @ v + c
    r = pred - y
    vs = v[None, :] * sech2                       # (n, W)
    dA = vs * x[:, None]                           # d pred / d a_j
    dB = vs                                        # d pred / d b_j
    dV = phi                                       # d pred / d v_j
    dC = np.ones((x.size, 1))                      # d pred / d c
    J = np.concatenate([dA, dB, dV, dC], axis=1)   # (n, 3W+1)
    return r, J


def lm_gauss_newton(p0, W, x, y, xe, ye, iters, mu0=1e-3):
    """Levenberg-Marquardt Gauss-Newton with scaled diagonal damping.
    Returns (best_eval_rel_l2, trained_eval_rel_l2, history)."""
    p = p0.copy()
    r, _ = residual_and_jac(p, W, x, y)
    loss = float(r @ r)
    mu = mu0
    best_eval = rel_l2(predict(p, W, xe), ye)
    history = [best_eval]
    for _ in range(iters):
        r, J = residual_and_jac(p, W, x, y)
        JtJ = J.T @ J
        Jtr = J.T @ r
        diag = np.clip(np.diag(JtJ), 1e-12, None)
        accepted = False
        for _ls in range(30):
            A = JtJ + mu * np.diag(diag)
            try:
                delta = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(A, Jtr, rcond=None)[0]
            p_new = p + delta
            r_new = predict(p_new, W, x) - y
            loss_new = float(r_new @ r_new)
            if np.isfinite(loss_new) and loss_new < loss:
                p, loss = p_new, loss_new
                mu = max(mu * 0.5, 1e-14)
                accepted = True
                break
            mu *= 4.0
            if mu > 1e14:
                break
        eval_r = rel_l2(predict(p, W, xe), ye)
        best_eval = min(best_eval, eval_r)
        history.append(eval_r)
        if not accepted:      # damping ran away -> converged or stuck
            break
    trained_eval = rel_l2(predict(p, W, xe), ye)
    return best_eval, trained_eval, history


def make_numpy_data(target_name, n_train, n_eval):
    target = get_target(target_name)
    x = np.linspace(-1.0, 1.0, n_train, dtype=np.float64)
    xe = np.linspace(-1.0, 1.0, n_eval, dtype=np.float64)
    return x, target.fn_numpy(x).astype(np.float64), xe, target.fn_numpy(xe).astype(np.float64)


def run(targets, resolutions, seeds, iters, n_train, n_eval):
    rows = []
    total = len(targets) * len(resolutions) * len(seeds) * len(FAMILIES)
    i = 0
    for target in targets:
        x, y, xe, ye = make_numpy_data(target, n_train, n_eval)
        for resolution in resolutions:
            geom = d05.geometry_for_resolution(resolution)
            W = geom.width
            for seed in seeds:
                for family in FAMILIES:
                    i += 1
                    state = d05.build_initial_state(family, target, geom, seed)
                    p0 = np.concatenate([
                        state.input_weights, state.input_biases,
                        state.readout_weights, [state.readout_bias],
                    ]).astype(np.float64)
                    best, trained, _ = lm_gauss_newton(p0, W, x, y, xe, ye, iters)
                    rows.append({
                        "target": target, "resolution": resolution, "width": W,
                        "seed": seed, "family": family,
                        "gn_best_eval_rel_l2": best, "gn_final_eval_rel_l2": trained,
                    })
                    print(f"[{i:3d}/{total}] {target:>6s} N={resolution:<4d} W={W:<4d} "
                          f"seed={seed} {family:<24s} gn_best={best:.2e}", flush=True)
    return rows


def write_summary(rows):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        SUMMARY_CSV.write_text("")
        return
    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args(argv)
    if args.mode == "smoke":
        rows = run(("runge",), (64, 128), (0,), args.iters or 60, 1543, 1001)
    else:
        rows = run(DEFAULT_TARGETS, DEFAULT_RESOLUTIONS, DEFAULT_SEEDS,
                   args.iters or DEFAULT_ITERS, DEFAULT_N_TRAIN, DEFAULT_N_EVAL)
    write_summary(rows)
    print(f"wrote {SUMMARY_CSV} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
