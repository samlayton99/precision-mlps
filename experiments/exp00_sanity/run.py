"""Experiment 0: Numerics Sanity Checks.

Goal: verify that linear solves, function evaluation, and tolerance choices
are not the precision bottleneck before attributing failures to training.

Tests
-----
1. Construction baseline: QI fp64 vs QI mpmath L_inf on dense eval grids,
   with and without Kahan-compensated evaluation.
2. Solver comparison: with gamma and interior centers fixed at the QI values,
   solve for (a_n, c0) via lstsq / qr / svd / ridge. Compare residual norms,
   eval L_inf, and reproducibility across backends.
3. Eval density sweep: evaluate each solution on grids of increasing density
   {256, 512, 1024, 2048, 4096, 8192} and verify the reported L_inf is stable.
4. Conditioning: cond(Phi) vs cond(Phi^T Phi); track the squaring.
5. tanh stability: compare fp64 vs mpmath evaluation of Phi entries at large
   gamma*(x-x_m), which reach O(N/2 * lambda) arguments near edges.
6. Extended-precision solve (small widths only): solve normal equations in
   mpmath, keep fp64 v, and see if the floor drops.

Outputs
-------
results/exp00_sanity/
    construction.jsonl      per-width construction-baseline metrics
    readout_solves.jsonl    per-(width, method) readout-solve metrics
    density_sweep.jsonl     per-(width, method, n_eval) L_inf
    conditioning.jsonl      per-width Phi conditioning
    tanh_stability.jsonl    per-width Phi fp64-vs-mpmath max diff
    mp_solve.jsonl          per-width mpmath normal-equations solve
    summary.txt             human-readable summary table
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.config.loader import load_config
from src.construction import (
    construct_qi,
    evaluate_qi,
    default_halo,
    build_phi,
    solve_readout,
    solve_readout_with_bias,
)
from src.data.targets import get_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / den if den > 0 else float("nan")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def eval_mlp_at(x: np.ndarray, gamma: float, centers: np.ndarray,
                a: np.ndarray, c0: float) -> np.ndarray:
    """Evaluate q(x) = c0 + sum_n a_n * tanh(gamma*(x-c_n)) in fp64."""
    z = gamma * (x[:, None] - centers[None, :])
    return c0 + np.sum(a[None, :] * np.tanh(z), axis=1)


# ---------------------------------------------------------------------------
# Build a Phi matrix + RHS for the readout solve
# ---------------------------------------------------------------------------

def make_train_grid(n_train: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n_train, dtype=np.float64)


def make_eval_grid(n_eval: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n_eval, dtype=np.float64)


# ---------------------------------------------------------------------------
# mpmath helpers (for stability + high-precision solve)
# ---------------------------------------------------------------------------

def phi_mpmath_max_diff(gamma: float, centers: np.ndarray, x_eval: np.ndarray,
                        dps: int = 50) -> float:
    """Max |tanh_fp64(gamma*(x-c)) - tanh_mp(gamma*(x-c))| over a subset of points.

    To keep cost manageable we sample a random-ish subset of entries.
    """
    import mpmath as mp
    mp.mp.dps = int(dps)

    # Evaluate at a stride (spare entries) to keep it fast
    n_x = x_eval.shape[0]
    n_c = centers.shape[0]
    # Sample up to ~4000 (x, c) pairs
    stride_x = max(1, n_x // 64)
    stride_c = max(1, n_c // 64)
    xs = x_eval[::stride_x]
    cs = centers[::stride_c]

    gamma_mp = mp.mpf(float(gamma))
    max_diff = 0.0
    for xv in xs:
        for cv in cs:
            z_fp = float(gamma) * (float(xv) - float(cv))
            t_fp = math.tanh(z_fp)
            t_mp = float(mp.tanh(gamma_mp * (mp.mpf(float(xv)) - mp.mpf(float(cv)))))
            d = abs(t_fp - t_mp)
            if d > max_diff:
                max_diff = d
    return max_diff


def mp_solve_normal_equations(Phi: np.ndarray, y: np.ndarray, dps: int = 50) -> dict:
    """Solve (Phi^T Phi) v = Phi^T y in mpmath, return fp64 v + diagnostics."""
    import mpmath as mp
    mp.mp.dps = int(dps)

    n, w = Phi.shape
    Phi_mp = mp.matrix(n, w)
    for i in range(n):
        for j in range(w):
            Phi_mp[i, j] = mp.mpf(float(Phi[i, j]))
    y_mp = mp.matrix(n, 1)
    for i in range(n):
        y_mp[i] = mp.mpf(float(y[i]))

    # Normal equations
    A = Phi_mp.T * Phi_mp
    b = Phi_mp.T * y_mp
    v_mp = mp.lu_solve(A, b)
    v = np.array([float(v_mp[i]) for i in range(w)], dtype=np.float64)

    # Residual in fp64
    resid = Phi @ v - y
    return {"v": v, "residual_norm": float(np.linalg.norm(resid))}


# ---------------------------------------------------------------------------
# Per-width tests
# ---------------------------------------------------------------------------

def test_construction_baseline(target, N: int, n_eval_list: list[int]) -> list[dict]:
    """Construct QI in fp64 and mpmath, evaluate on each eval density."""
    rows = []
    # fp64 construction
    qi_f = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, precision="fp64")
    # mpmath construction
    qi_m = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, precision="mpmath")

    for n_eval in n_eval_list:
        x = make_eval_grid(n_eval)
        y_true = target.fn_numpy(x)
        for label, qi in [("fp64", qi_f), ("mpmath", qi_m)]:
            for kahan in (False, True):
                y_pred = evaluate_qi(qi, x, kahan=kahan)
                rows.append({
                    "N": N,
                    "precision": label,
                    "kahan": kahan,
                    "n_eval": n_eval,
                    "linf": linf(y_pred, y_true),
                    "rel_l2": rel_l2(y_pred, y_true),
                    "gamma": qi.gamma,
                    "lambda": qi.lambda_val,
                    "halo": qi.halo,
                    "width_total": int(qi.centers.shape[0]),
                    "width_interior": int(qi.interior_centers.shape[0]),
                    "toeplitz_residual": qi.toeplitz_residual,
                })
    return rows


def test_readout_solves(target, N: int, n_train: int, n_eval_list: list[int],
                        geometry: str = "full") -> dict:
    """Fix gamma and centers from QI; solve readout with each method.

    geometry: "full" (halo+interior) | "interior" (interior-only).
    """
    # Use mpmath construction geometry (machine-eps reference)
    qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, precision="mpmath")
    gamma = qi.gamma
    if geometry == "full":
        centers = qi.centers
    elif geometry == "interior":
        centers = qi.interior_centers
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Train grid + rhs
    x_train = make_train_grid(n_train)
    y_train = target.fn_numpy(x_train)

    # Build Phi
    Phi = build_phi(x_train, gamma * np.ones_like(centers), centers)  # [n_train, N+1]

    # Conditioning
    s = np.linalg.svd(Phi, compute_uv=False)
    smin, smax = float(s[-1]), float(s[0])
    cond_Phi = smax / smin if smin > 0 else float("inf")
    s_gram = np.linalg.svd(Phi.T @ Phi, compute_uv=False)
    cond_gram = float(s_gram[0] / s_gram[-1]) if s_gram[-1] > 0 else float("inf")

    solver_rows: list[dict] = []
    density_rows: list[dict] = []

    methods = [
        ("lstsq", 0.0),
        ("qr", 0.0),
        ("svd", 0.0),
        ("ridge_0", 0.0),
        ("ridge_1e-14", 1e-14),
        ("ridge_1e-12", 1e-12),
    ]

    for method_label, ridge_alpha in methods:
        try:
            if method_label.startswith("ridge"):
                v, bias, info = solve_readout_with_bias(
                    Phi, y_train, method="ridge", ridge_alpha=ridge_alpha,
                )
            else:
                v, bias, info = solve_readout_with_bias(
                    Phi, y_train, method=method_label, ridge_alpha=0.0,
                )
        except np.linalg.LinAlgError as e:
            solver_rows.append({
                "N": N, "geometry": geometry, "method": method_label,
                "ridge_alpha": float(ridge_alpha),
                "train_linf": float("nan"), "residual_norm": float("nan"),
                "eval_linf_max_density": float("nan"),
                "eval_rel_l2_max_density": float("nan"),
                "v_max_abs": float("nan"), "v_l2": float("nan"),
                "bias": float("nan"), "n_train": n_train,
                "error": str(e),
            })
            continue

        # Train residual
        y_pred_train = Phi @ v + bias
        train_linf = linf(y_pred_train, y_train)
        residual_norm = info.get("residual_norm", float("nan"))

        # Dense eval metrics (at max density) for a quick scalar
        x_eval_max = make_eval_grid(n_eval_list[-1])
        y_eval_max = target.fn_numpy(x_eval_max)
        y_pred_max = eval_mlp_at(x_eval_max, gamma, centers, v, bias)
        eval_linf_max = linf(y_pred_max, y_eval_max)
        eval_rel_l2_max = rel_l2(y_pred_max, y_eval_max)

        solver_rows.append({
            "N": N,
            "geometry": geometry,
            "method": method_label,
            "ridge_alpha": float(ridge_alpha),
            "train_linf": train_linf,
            "residual_norm": float(residual_norm),
            "eval_linf_max_density": eval_linf_max,
            "eval_rel_l2_max_density": eval_rel_l2_max,
            "v_max_abs": float(np.max(np.abs(v))),
            "v_l2": float(np.linalg.norm(v)),
            "bias": float(bias),
            "n_train": n_train,
        })

        # Density sweep
        for n_eval in n_eval_list:
            x = make_eval_grid(n_eval)
            y_true = target.fn_numpy(x)
            y_pred = eval_mlp_at(x, gamma, centers, v, bias)
            density_rows.append({
                "N": N,
                "geometry": geometry,
                "method": method_label,
                "n_eval": n_eval,
                "linf": linf(y_pred, y_true),
                "rel_l2": rel_l2(y_pred, y_true),
            })

    cond_row = {
        "N": N,
        "geometry": geometry,
        "n_train": n_train,
        "width_interior": int(centers.shape[0]),
        "gamma": float(gamma),
        "lambda": float(qi.lambda_val),
        "phi_smin": smin,
        "phi_smax": smax,
        "cond_Phi": cond_Phi,
        "cond_gram": cond_gram,
        "cond_ratio": cond_gram / cond_Phi if cond_Phi > 0 else float("nan"),
    }

    return {
        "solvers": solver_rows,
        "density": density_rows,
        "conditioning": cond_row,
        "Phi": Phi,
        "y_train": y_train,
        "gamma": gamma,
        "centers": centers,
    }


def cross_solver_reproducibility(Phi: np.ndarray, y_train: np.ndarray,
                                 gamma: float, centers: np.ndarray,
                                 x_eval: np.ndarray) -> dict:
    """Compute max |pred_a - pred_b| across solver pairs on a dense eval grid."""
    preds = {}
    methods = [("lstsq", 0.0), ("qr", 0.0), ("svd", 0.0), ("ridge_0", 0.0)]
    for label, ridge_alpha in methods:
        try:
            if label.startswith("ridge"):
                v, bias, _ = solve_readout_with_bias(
                    Phi, y_train, method="ridge", ridge_alpha=ridge_alpha,
                )
            else:
                v, bias, _ = solve_readout_with_bias(
                    Phi, y_train, method=label, ridge_alpha=0.0,
                )
            preds[label] = eval_mlp_at(x_eval, gamma, centers, v, bias)
        except np.linalg.LinAlgError:
            continue

    names = list(preds.keys())
    pair_diffs: dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_diffs[f"{a}_vs_{b}"] = float(np.max(np.abs(preds[a] - preds[b])))
    return pair_diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(str(config_path))
    out_dir = Path("results") / "exp00_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = get_target(cfg.target)
    n_eval_list = [256, 512, 1024, 2048, 4096, 8192]

    construction_rows: list[dict] = []
    solver_rows: list[dict] = []
    density_rows: list[dict] = []
    conditioning_rows: list[dict] = []
    tanh_rows: list[dict] = []
    mp_rows: list[dict] = []
    repro_rows: list[dict] = []

    print(f"[exp00] target={cfg.target}, widths={cfg.widths}")

    for N in cfg.widths:
        print(f"\n[exp00] --- N = {N} ---")

        # 1. Construction baseline
        print(f"  [construction] fp64 + mpmath at N={N}")
        crows = test_construction_baseline(target, N, n_eval_list)
        construction_rows.extend(crows)

        # Print a compact summary for this width
        for r in crows:
            if r["n_eval"] == n_eval_list[-1]:
                print(
                    f"    precision={r['precision']:<6} kahan={str(r['kahan']):<5} "
                    f"n_eval={r['n_eval']} linf={r['linf']:.3e} rel_l2={r['rel_l2']:.3e} "
                    f"gamma={r['gamma']:.2f} lambda={r['lambda']}"
                )

        # 2-4. Readout solves + density sweep + conditioning (two geometries)
        halo = default_halo(N)
        width_full = N + 2 * halo + 1
        # n_train must exceed the widest solve (full geometry)
        n_train = max(cfg.n_train, 4 * width_full)
        print(f"  [readout solves] n_train={n_train}, width_full={width_full}, width_interior={N+1}")

        results_full = test_readout_solves(
            target, N, n_train, n_eval_list, geometry="full",
        )
        results_int = test_readout_solves(
            target, N, n_train, n_eval_list, geometry="interior",
        )
        solver_rows.extend(results_full["solvers"])
        solver_rows.extend(results_int["solvers"])
        density_rows.extend(results_full["density"])
        density_rows.extend(results_int["density"])
        conditioning_rows.append(results_full["conditioning"])
        conditioning_rows.append(results_int["conditioning"])
        results = results_full  # use full for cross-solver repro / stability

        for res_label, res in [("full", results_full), ("interior", results_int)]:
            cond = res["conditioning"]
            print(
                f"    [{res_label}] cond(Phi)={cond['cond_Phi']:.3e}  "
                f"cond(Phi^T Phi)={cond['cond_gram']:.3e}  ratio={cond['cond_ratio']:.3e}"
            )
            for s in res["solvers"]:
                print(
                    f"    [{res_label}] {s['method']:<12} resid={s['residual_norm']:.3e} "
                    f"eval_linf={s['eval_linf_max_density']:.3e} "
                    f"v_max={s['v_max_abs']:.3e}"
                )

        # Cross-solver reproducibility
        x_eval_max = make_eval_grid(n_eval_list[-1])
        pair_diffs = cross_solver_reproducibility(
            results["Phi"], results["y_train"],
            results["gamma"], results["centers"], x_eval_max,
        )
        for pair, diff in pair_diffs.items():
            repro_rows.append({"N": N, "pair": pair, "max_diff": diff})
        print(f"    [repro] lstsq-vs-svd max |diff| = {pair_diffs.get('lstsq_vs_svd', float('nan')):.3e}")

        # 5. tanh stability (slow-ish due to mpmath)
        print(f"  [tanh stability] fp64 vs mpmath on Phi subset")
        max_diff = phi_mpmath_max_diff(
            results["gamma"], results["centers"], x_eval_max, dps=50,
        )
        tanh_rows.append({
            "N": N,
            "gamma": float(results["gamma"]),
            "max_phi_diff": max_diff,
            "gamma_x_max": float(results["gamma"] * 2.0),  # worst |gamma*(x-c)|
        })
        print(f"    max |tanh_fp64 - tanh_mp| = {max_diff:.3e}")

        # 6. Extended-precision solve for small N only (cost ~ w^3 * dps)
        # Use interior geometry to keep the problem size small.
        if N <= 64:
            print(f"  [mp solve] normal equations in mpmath (N={N}, interior geometry)")
            Phi_int = results_int["Phi"]
            Phi_aug = np.hstack([Phi_int, np.ones((Phi_int.shape[0], 1))])
            mp_info = mp_solve_normal_equations(Phi_aug, results_int["y_train"], dps=50)
            v_all = mp_info["v"]
            v = v_all[:-1]
            bias = float(v_all[-1])
            y_pred = eval_mlp_at(
                x_eval_max, results_int["gamma"], results_int["centers"], v, bias,
            )
            y_true = target.fn_numpy(x_eval_max)
            mp_rows.append({
                "N": N,
                "residual_norm": mp_info["residual_norm"],
                "eval_linf": linf(y_pred, y_true),
                "eval_rel_l2": rel_l2(y_pred, y_true),
                "v_max_abs": float(np.max(np.abs(v))),
            })
            print(f"    mp-solve eval_linf={mp_rows[-1]['eval_linf']:.3e}")

    # --- Write outputs ---
    write_jsonl(out_dir / "construction.jsonl", construction_rows)
    write_jsonl(out_dir / "readout_solves.jsonl", solver_rows)
    write_jsonl(out_dir / "density_sweep.jsonl", density_rows)
    write_jsonl(out_dir / "conditioning.jsonl", conditioning_rows)
    write_jsonl(out_dir / "tanh_stability.jsonl", tanh_rows)
    write_jsonl(out_dir / "mp_solve.jsonl", mp_rows)
    write_jsonl(out_dir / "reproducibility.jsonl", repro_rows)

    # --- Summary ---
    summary_lines = []
    summary_lines.append(f"exp00 sanity checks: target={cfg.target}")
    summary_lines.append("=" * 80)

    summary_lines.append("\n[1] Construction baseline (L_inf, n_eval=8192):")
    summary_lines.append(f"  {'N':>4} {'precision':>9} {'kahan':>6} {'linf':>12} {'rel_l2':>12}")
    for r in construction_rows:
        if r["n_eval"] == 8192:
            summary_lines.append(
                f"  {r['N']:>4} {r['precision']:>9} {str(r['kahan']):>6} "
                f"{r['linf']:>12.3e} {r['rel_l2']:>12.3e}"
            )

    summary_lines.append("\n[2] Readout solvers (eval_linf at n_eval=8192):")
    summary_lines.append(
        f"  {'N':>4} {'geom':>8} {'method':>13} {'resid':>12} {'eval_linf':>12} {'v_max':>12}"
    )
    for s in solver_rows:
        summary_lines.append(
            f"  {s['N']:>4} {s['geometry']:>8} {s['method']:>13} "
            f"{s['residual_norm']:>12.3e} "
            f"{s['eval_linf_max_density']:>12.3e} {s['v_max_abs']:>12.3e}"
        )

    summary_lines.append("\n[3] Conditioning:")
    summary_lines.append(
        f"  {'N':>4} {'geom':>8} {'cond(Phi)':>12} {'cond(Phi^TPhi)':>16} {'ratio':>12}"
    )
    for c in conditioning_rows:
        summary_lines.append(
            f"  {c['N']:>4} {c['geometry']:>8} {c['cond_Phi']:>12.3e} "
            f"{c['cond_gram']:>16.3e} {c['cond_ratio']:>12.3e}"
        )

    summary_lines.append("\n[4] tanh stability (max |tanh_fp64 - tanh_mp| over sampled Phi):")
    summary_lines.append(f"  {'N':>4} {'gamma':>10} {'max_diff':>12}")
    for t in tanh_rows:
        summary_lines.append(
            f"  {t['N']:>4} {t['gamma']:>10.2f} {t['max_phi_diff']:>12.3e}"
        )

    if mp_rows:
        summary_lines.append("\n[5] mpmath normal-equations solve:")
        summary_lines.append(f"  {'N':>4} {'resid':>12} {'eval_linf':>12} {'rel_l2':>12}")
        for m in mp_rows:
            summary_lines.append(
                f"  {m['N']:>4} {m['residual_norm']:>12.3e} "
                f"{m['eval_linf']:>12.3e} {m['eval_rel_l2']:>12.3e}"
            )

    summary_lines.append("\n[6] Density sweep: eval_linf changes across n_eval (mpmath QI construction):")
    summary_lines.append(f"  {'N':>4} " + " ".join(f"{n:>12}" for n in n_eval_list))
    for N in cfg.widths:
        row = [f"  {N:>4}"]
        for n in n_eval_list:
            match = [
                r for r in construction_rows
                if r["N"] == N and r["precision"] == "mpmath" and not r["kahan"] and r["n_eval"] == n
            ]
            if match:
                row.append(f"{match[0]['linf']:>12.3e}")
            else:
                row.append(f"{'-':>12}")
        summary_lines.append(" ".join(row))

    summary = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\n[exp00] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
