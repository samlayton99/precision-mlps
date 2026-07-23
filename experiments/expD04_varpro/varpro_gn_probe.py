"""VarPro + (Adam -> Gauss-Newton) probe.

expD06 + the GN probe showed that on the FULL end-to-end problem a random
readout couples into the geometry Jacobian (columns scaled by v) and stalls both
Adam (~1e-4) and Gauss-Newton (~1e-5) well above the oracle floor. Variable
Projection removes that coupling: the linear readout v(theta) is solved exactly
every step, so the optimizer only moves the nonlinear geometry theta=(a,b) and
always sees the correctly-scaled curvature.

This probe runs VarPro with a two-stage optimizer -- Adam warmup then damped
(Levenberg-Marquardt) Gauss-Newton -- on theta only, and asks whether it
CONVERGES QI geometry to the fp64 floor (the thing full-network GN could not do).
It is the precursor to VarPro + Adam->SSBroyden.

Because the readout is always solved, only the GEOMETRY init matters, so the two
inits are:
  - qi   : QI construction geometry (centers, gamma)   -- expect convergence to floor
  - xavier: standard Xavier affine geometry             -- reference

Reduced residual (Kaufman VarPro):
  A(theta) = [tanh(a x + b), 1]           (n, W+1)
  c_hat    = A^+ y   (lstsq, via QR)      readout+bias, solved
  r        = A c_hat - y                  reduced residual
  J_theta  = d r / d(a,b) holding c_hat fixed          (n, 2W)
  J        = (I - Q Q^T) J_theta          project off col(A)  [Kaufman]
  grad     = J^T r,   GN normal eqn: (J^T J + mu diag) d = -J^T r

pure numpy fp64.  Usage:
    uv run --extra dev python experiments/expD04_varpro/varpro_gn_probe.py --mode smoke
    uv run --extra dev python experiments/expD04_varpro/varpro_gn_probe.py --mode full
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
SUMMARY_CSV = RESULTS_DIR / "varpro_gn_summary.csv"

# geometry-init label -> expD05 family to pull (a, b) from (readout is discarded).
# A ladder that fixes progressively more of the geometry; VarPro+GN recovers the rest.
GEOM_INITS = {
    "xavier": "standard_xavier_affine",              # random centers (collapsed) + random gamma
    "center_spread": "center_spread_xavier",         # grid centers + random gamma
    "scale_corrected": "scale_corrected_xavier_l030",# grid centers + correct-scale (spread) gamma
    "qi": "qi_geom_random_readout",                  # grid centers + exact uniform gamma
}
ALL_TARGETS = ("sine", "cosine", "sine_8pi", "runge", "tanh_steep",
               "sine_mixture", "exp", "poly5", "abs_cubed")
DEFAULT_TARGETS = ALL_TARGETS
DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)
DEFAULT_SEEDS = (0, 1)
FLOOR = 5e-14
GEOM_COLORS = {"xavier": "#d62728", "center_spread": "#ff7f0e",
               "scale_corrected": "#1f77b4", "qi": "#2ca02c"}
GEOM_LABELS = {"xavier": "Xavier (random centers+γ)",
               "center_spread": "grid centers, random γ",
               "scale_corrected": "grid centers, correct-scale γ",
               "qi": "QI geometry (grid + exact γ)"}
DEFAULT_WARMUP = 500      # Adam warmup steps on theta
DEFAULT_GN = 80           # LM Gauss-Newton iterations
DEFAULT_N_TRAIN = 2003    # matches expD06; > 2W at all widths
DEFAULT_N_EVAL = 4001


def rel_l2(pred, y):
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def _A_and_derivs(theta, W, x):
    a, b = theta[:W], theta[W:]
    z = x[:, None] * a[None, :] + b[None, :]
    phi = np.tanh(z)
    sech2 = 1.0 - phi * phi
    A = np.concatenate([phi, np.ones((x.size, 1))], axis=1)   # (n, W+1)
    return A, phi, sech2


def solve_readout(theta, W, x, y):
    """Exact readout solve. One QR (also gives Q for the Kaufman projection);
    solve via R in the well-conditioned common case, fall back to truncated
    lstsq only when the geometry drives A near-singular (keeps c_hat bounded)."""
    A, phi, sech2 = _A_and_derivs(theta, W, x)
    Q, R = np.linalg.qr(A)
    rdiag = np.abs(np.diag(R))
    if rdiag.size and rdiag.min() > 1e-12 * rdiag.max():
        c_hat = np.linalg.solve(R, Q.T @ y)
    else:
        c_hat, *_ = np.linalg.lstsq(A, y, rcond=1e-13)
    r = A @ c_hat - y
    return A, phi, sech2, Q, c_hat, r


def reduced_jac(phi, sech2, Q, c_hat, W, x):
    """Kaufman VarPro Jacobian of the reduced residual wrt theta=(a,b)."""
    v = c_hat[:W]                       # solved readout weights
    vs = v[None, :] * sech2             # (n, W)
    dA = vs * x[:, None]
    dB = vs
    J_theta = np.concatenate([dA, dB], axis=1)   # (n, 2W)
    J = J_theta - Q @ (Q.T @ J_theta)            # project off col(A)
    return J


def eval_rel_l2(theta, W, xt, yt, xe, ye):
    """Solve readout on train points, evaluate on eval points."""
    A, _, _, Q, c_hat, _ = solve_readout(theta, W, xt, yt)
    a, b = theta[:W], theta[W:]
    phi_e = np.tanh(xe[:, None] * a[None, :] + b[None, :])
    pred_e = phi_e @ c_hat[:W] + c_hat[W]
    return rel_l2(pred_e, ye)


def varpro_adam_then_gn(theta0, W, xt, yt, xe, ye, warmup, gn_iters,
                        adam_lr=1e-2, mu0=1e-3):
    theta = theta0.copy()
    best = eval_rel_l2(theta, W, xt, yt, xe, ye)

    # ---- stage 1: Adam warmup on theta (VarPro reduced gradient) ----
    m = np.zeros_like(theta); vv = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-12
    for t in range(1, warmup + 1):
        _, phi, sech2, Q, c_hat, r = solve_readout(theta, W, xt, yt)
        J = reduced_jac(phi, sech2, Q, c_hat, W, xt)
        g = J.T @ r
        if not np.all(np.isfinite(g)):
            break
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = vv / (1 - b2 ** t)
        theta = theta - adam_lr * mh / (np.sqrt(vh) + eps)
        if t % 50 == 0:
            best = min(best, eval_rel_l2(theta, W, xt, yt, xe, ye))

    # ---- stage 2: LM Gauss-Newton on theta ----
    _, phi, sech2, Q, c_hat, r = solve_readout(theta, W, xt, yt)
    loss = float(r @ r)
    mu = mu0
    for _ in range(gn_iters):
        J = reduced_jac(phi, sech2, Q, c_hat, W, xt)
        JtJ = J.T @ J
        Jtr = J.T @ r
        diag = np.clip(np.diag(JtJ), 1e-12, None)
        accepted = False
        for _ls in range(30):
            try:
                delta = np.linalg.solve(JtJ + mu * np.diag(diag), -Jtr)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(JtJ + mu * np.diag(diag), Jtr, rcond=None)[0]
            theta_new = theta + delta
            _, phi_n, sech2_n, Q_n, c_hat_n, r_n = solve_readout(theta_new, W, xt, yt)
            loss_new = float(r_n @ r_n)
            if np.isfinite(loss_new) and loss_new < loss:
                theta, phi, sech2, Q, c_hat, r, loss = theta_new, phi_n, sech2_n, Q_n, c_hat_n, r_n, loss_new
                mu = max(mu * 0.5, 1e-14)
                accepted = True
                break
            mu *= 4.0
            if mu > 1e14:
                break
        best = min(best, eval_rel_l2(theta, W, xt, yt, xe, ye))
        if not accepted:
            break
    return best, eval_rel_l2(theta, W, xt, yt, xe, ye)


def make_data(target_name, n_train, n_eval):
    t = get_target(target_name)
    xt = np.linspace(-1, 1, n_train); xe = np.linspace(-1, 1, n_eval)
    return xt, t.fn_numpy(xt).astype(np.float64), xe, t.fn_numpy(xe).astype(np.float64)


def run(targets, resolutions, seeds, warmup, gn_iters, n_train, n_eval, inits=None):
    init_items = [(k, GEOM_INITS[k]) for k in (inits or GEOM_INITS)]
    rows = []
    total = len(targets) * len(resolutions) * len(seeds) * len(init_items)
    i = 0
    for target in targets:
        xt, yt, xe, ye = make_data(target, n_train, n_eval)
        for resolution in resolutions:
            geom = d05.geometry_for_resolution(resolution)
            W = geom.width
            for seed in seeds:
                for label, family in init_items:
                    i += 1
                    st = d05.build_initial_state(family, target, geom, seed)
                    theta0 = np.concatenate([st.input_weights, st.input_biases]).astype(np.float64)
                    best, final = varpro_adam_then_gn(theta0, W, xt, yt, xe, ye, warmup, gn_iters)
                    rows.append({"target": target, "resolution": resolution, "width": W,
                                 "seed": seed, "geom_init": label,
                                 "vpgn_best_eval_rel_l2": best, "vpgn_final_eval_rel_l2": final})
                    print(f"[{i:3d}/{total}] {target:>6s} N={resolution:<4d} W={W:<4d} "
                          f"seed={seed} geom={label:<7s} best={best:.2e}", flush=True)
    return rows


def write_summary(rows):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        SUMMARY_CSV.write_text(""); return
    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def read_summary():
    with SUMMARY_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def merge_and_write(new_rows):
    """Merge new rows into summary.csv, keyed by target/resolution/seed/geom_init."""
    def key(r):
        return (r["target"], int(r["resolution"]), int(r["seed"]), r["geom_init"])
    existing = {}
    if SUMMARY_CSV.exists() and SUMMARY_CSV.stat().st_size > 0:
        for r in read_summary():
            existing[key(r)] = dict(r)
    for r in new_rows:
        existing[key(r)] = r
    merged = list(existing.values())
    write_summary(merged)
    return merged


def plot(rows):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    # keep the registry order for facets
    order = [t for t in ALL_TARGETS if any(r["target"] == t for r in rows)]
    order += [t for t in dict.fromkeys(r["target"] for r in rows) if t not in order]
    ncol = 3
    nrow = math.ceil(len(order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.3 * nrow), squeeze=False)

    def med(vals):
        vals = [v for v in vals if v == v]
        return sorted(vals)[len(vals) // 2] if vals else float("nan")

    for idx, target in enumerate(order):
        ax = axes[idx // ncol][idx % ncol]
        trows = [r for r in rows if r["target"] == target]
        widths = sorted({int(r["width"]) for r in trows})
        present = [g for g in GEOM_INITS if any(r["geom_init"] == g for r in trows)]
        for geom in present:
            med_y, lo, hi = [], [], []
            for w in widths:
                vals = [float(r["vpgn_best_eval_rel_l2"]) for r in trows
                        if int(r["width"]) == w and r["geom_init"] == geom]
                med_y.append(med(vals))
                lo.append(min(vals) if vals else float("nan"))
                hi.append(max(vals) if vals else float("nan"))
            c = GEOM_COLORS[geom]
            ax.plot(widths, med_y, "-o", color=c, label=GEOM_LABELS[geom], zorder=3)
            ax.fill_between(widths, lo, hi, color=c, alpha=0.15, zorder=1)
        ax.axhline(FLOOR, color="0.6", ls=":", lw=1.1, zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_title(target)
        ax.set_xlabel("hidden width $W$"); ax.set_ylabel("eval rel $L_2$")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="0.6", ls=":", lw=1.1))
    labels.append(f"fp64 floor ({FLOOR:.0e})")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0), fontsize=9, framealpha=0.9)
    fig.suptitle("expD04 VarPro + Adam→Gauss-Newton: convergence vs width, "
                 "QI vs Xavier geometry (readout projected out)", y=1.03, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS_DIR / "figures" / "varpro_gn_convergence_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--append", action="store_true", help="merge into existing summary.csv")
    ap.add_argument("--inits", type=lambda s: tuple(x.strip() for x in s.split(",")), default=None,
                    help="comma list of geom-init labels to run (default: all four)")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--gn", type=int, default=None)
    args = ap.parse_args(argv)
    if args.plot_only:
        plot(read_summary())
        return 0
    if args.mode == "smoke":
        rows = run(("runge", "exp"), (64, 128), (0,), args.warmup or 200, args.gn or 40, 1543, 1001,
                   inits=args.inits)
    else:
        rows = run(DEFAULT_TARGETS, DEFAULT_RESOLUTIONS, DEFAULT_SEEDS,
                   args.warmup or DEFAULT_WARMUP, args.gn or DEFAULT_GN,
                   DEFAULT_N_TRAIN, DEFAULT_N_EVAL, inits=args.inits)
    all_rows = merge_and_write(rows) if args.append else (write_summary(rows) or rows)
    print(f"wrote {SUMMARY_CSV} ({len(rows)} new rows)")
    plot(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
