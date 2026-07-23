"""Full-network Adam -> Gauss-Newton (NO VarPro, NO least-squares readout).

The optimizer-only counterpart to varpro_gn_probe: here ALL params (a, b, v, c)
are trained end-to-end -- Adam warmup then damped (LM) Gauss-Newton -- so the
readout is *learned*, never solved by lstsq. This isolates whether a second-order
optimizer alone (no readout projection) can reach the floor from QI geometry.

For the init-slice overlay's runge and sine_mixture panels. Init = the free QI
geometry (qi_geom_random_readout). pure numpy fp64, analytic Jacobian.
Params p = [a (W), b (W), v (W), c (1)] -> 3W+1.

Usage: uv run --extra dev python experiments/expD04_varpro/adam_gn_fullnet_probe.py
"""

from __future__ import annotations

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

OUT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "adam_gn_fullnet_summary.csv"

TARGETS = ("runge", "sine_mixture")
RESOLUTIONS = (32, 64, 128, 256, 512)
SEEDS = (0, 1)
FAMILY = "qi_geom_random_readout"     # free QI geometry, readout learned (not projected)
WARMUP, GN_ITERS = 5000, 60           # longer Adam warmup (fair vs the pure-Adam line) then GN
N_TRAIN, N_EVAL = 6007, 4001          # > 3W+1 at the largest width


def rel_l2(pred, y):
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def predict(p, W, x):
    a, b, v, c = p[:W], p[W:2 * W], p[2 * W:3 * W], p[3 * W]
    return np.tanh(x[:, None] * a[None, :] + b[None, :]) @ v + c


def residual_and_jac(p, W, x, y):
    a, b, v, c = p[:W], p[W:2 * W], p[2 * W:3 * W], p[3 * W]
    phi = np.tanh(x[:, None] * a[None, :] + b[None, :])
    sech2 = 1.0 - phi * phi
    r = phi @ v + c - y
    vs = v[None, :] * sech2
    J = np.concatenate([vs * x[:, None], vs, phi, np.ones((x.size, 1))], axis=1)  # (n, 3W+1)
    return r, J


def residual_and_grad(p, W, x, y):
    """Residual and gradient J^T r WITHOUT forming the (n, 3W+1) Jacobian --
    cheap enough for a long Adam warmup."""
    a, b, v, c = p[:W], p[W:2 * W], p[2 * W:3 * W], p[3 * W]
    phi = np.tanh(x[:, None] * a[None, :] + b[None, :])
    sech2 = 1.0 - phi * phi
    r = phi @ v + c - y
    vs = v[None, :] * sech2
    g_a = (r[:, None] * vs * x[:, None]).sum(0)
    g_b = (r[:, None] * vs).sum(0)
    g_v = phi.T @ r
    g_c = r.sum()
    return r, np.concatenate([g_a, g_b, g_v, [g_c]])


def adam_then_gn(p0, W, xt, yt, xe, ye, warmup, gn_iters, adam_lr=1e-3, mu0=1e-3):
    p = p0.copy()
    best = rel_l2(predict(p, W, xe), ye)
    # ---- Adam warmup on ALL params ----
    m = np.zeros_like(p); vv = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-12
    for t in range(1, warmup + 1):
        r, g = residual_and_grad(p, W, xt, yt)
        if not np.all(np.isfinite(g)):
            break
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        p = p - adam_lr * (m / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + eps)
        if t % 50 == 0:
            best = min(best, rel_l2(predict(p, W, xe), ye))
    # ---- LM Gauss-Newton on ALL params ----
    r, _ = residual_and_jac(p, W, xt, yt)
    loss = float(r @ r); mu = mu0
    for _ in range(gn_iters):
        r, J = residual_and_jac(p, W, xt, yt)
        JtJ = J.T @ J; Jtr = J.T @ r
        diag = np.clip(np.diag(JtJ), 1e-12, None)
        accepted = False
        for _ls in range(30):
            try:
                delta = np.linalg.solve(JtJ + mu * np.diag(diag), -Jtr)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(JtJ + mu * np.diag(diag), Jtr, rcond=None)[0]
            pn = p + delta
            rn = predict(pn, W, xt) - yt
            ln = float(rn @ rn)
            if np.isfinite(ln) and ln < loss:
                p, loss = pn, ln; mu = max(mu * 0.5, 1e-14); accepted = True; break
            mu *= 4.0
            if mu > 1e14:
                break
        best = min(best, rel_l2(predict(p, W, xe), ye))
        if not accepted:
            break
    return best, rel_l2(predict(p, W, xe), ye)


def main():
    rows = []
    total = len(TARGETS) * len(RESOLUTIONS) * len(SEEDS)
    i = 0
    for target in TARGETS:
        t = get_target(target)
        xt = np.linspace(-1, 1, N_TRAIN); yt = t.fn_numpy(xt).astype(np.float64)
        xe = np.linspace(-1, 1, N_EVAL); ye = t.fn_numpy(xe).astype(np.float64)
        for resolution in RESOLUTIONS:
            geom = d05.geometry_for_resolution(resolution)
            W = geom.width
            for seed in SEEDS:
                i += 1
                st = d05.build_initial_state(FAMILY, target, geom, seed)
                p0 = np.concatenate([st.input_weights, st.input_biases,
                                     st.readout_weights, [st.readout_bias]]).astype(np.float64)
                best, final = adam_then_gn(p0, W, xt, yt, xe, ye, WARMUP, GN_ITERS)
                rows.append({"target": target, "resolution": resolution, "width": W,
                             "seed": seed, "adamgn_best_eval_rel_l2": best})
                print(f"[{i:2d}/{total}] {target:>12s} N={resolution:<4d} W={W:<4d} "
                      f"seed={seed} adamgn_best={best:.2e}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
