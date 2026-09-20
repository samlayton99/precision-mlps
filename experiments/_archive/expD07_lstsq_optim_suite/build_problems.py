"""Build the fixed expD07 problem registry (synthetic + phis) with fp64 floors.

Synthetic (data/expD07_problems/synthetic/): A = U diag(s) V^T with a controlled
singular spectrum; y = scale * A v* (+ optional off-range component). The axes
that matter for an optimizer on a quadratic are the spectrum and the alignment
of the solution with the singular directions -- everything else is cosmetic.
10 cells x N in {64,128,256} x R/N in {1,4} = 60 problems, v0 = 0, no seeds
(each problem is a fixed artifact; its build seed is recorded).

Phis (data/expD07_problems/phis/): the QI-geometry feature system, repo
conventions exactly: uniform centers + halo (default_halo, lambda*=0.25),
gamma = lambda*/h, augmented [Phi, 1], truncated-SVD floor at common.RCOND,
eval on 4001 equispaced points. Rows R = ratio * cols, ratio in {0.5,1,2,4},
6 target functions x N in {64,128,256} = 72 problems.

Floors stored per problem: L* (train MSE of the direct solve), L0 (loss at
v=0), smax (for lr scaling), and for phis the eval rel-L2 of the SVD solution.

Usage: uv run python experiments/expD07_lstsq_optim_suite/build_problems.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    LAMBDA_STAR, MANIFEST_PATH, N_EVAL, PROBLEMS_DIR, RCOND, build_phi_aug,
)
from src.construction.qi_mpmath import default_halo  # noqa: E402
from src.construction.readout import solve_readout  # noqa: E402
from src.data.targets import get_target  # noqa: E402

SYN_N = [64, 128, 256]
SYN_RATIOS = [1, 4]
PHI_N = [64, 128, 256]
PHI_RATIOS = [0.5, 1, 2, 4]
PHI_FNS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]

# cell -> spectrum / solution-alignment / scale / consistency
CELLS = {
    "wellcond_iso":       dict(spec="log", kappa=1e2, align="iso"),
    "illcond8_iso":       dict(spec="log", kappa=1e8, align="iso"),
    "illcond14_iso":      dict(spec="log", kappa=1e14, align="iso"),
    "phispec_iso":        dict(spec="phi", align="iso"),
    "illcond8_head":      dict(spec="log", kappa=1e8, align="head"),
    "illcond8_tail":      dict(spec="log", kappa=1e8, align="tail"),
    "rankdef_consistent": dict(spec="rankdef", align="iso"),
    "rankdef_inconsistent": dict(spec="rankdef", align="iso", inconsistent=True),
    "scale_tiny":         dict(spec="log", kappa=1e2, align="iso", scale=1e-3),
    "scale_huge":         dict(spec="log", kappa=1e2, align="iso", scale=1e3),
}


def _orth(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, m)))
    return q[:, :m]


def _spectrum(kind: str, m: int, kappa: float | None) -> np.ndarray:
    if kind == "log":
        return np.logspace(0.0, -np.log10(kappa), m)
    if kind == "phi":
        # matches the measured [Phi, 1] spectrum (analyze_phi_regime.py):
        # exponential decay from the very first index, then a hard rank
        # cliff at 0.6*m (rank ~ N + bounded null space, expA04)
        r = int(0.6 * m)
        s = np.zeros(m)
        s[:r] = np.logspace(0.0, -10.0, r)
        return s
    if kind == "rankdef":
        r = int(0.6 * m)
        s = np.zeros(m)
        s[:r] = np.logspace(0.0, -3.0, r)
        return s
    raise ValueError(kind)


def make_synthetic(cell: str, cfg: dict, N: int, R: int, seed: int):
    """Returns (A, y). Deterministic given (cell, N, R, seed)."""
    rng = np.random.default_rng(seed)
    m = min(N, R)
    s = _spectrum(cfg["spec"], m, cfg.get("kappa"))
    U, V = _orth(rng, R, m), _orth(rng, N, m)
    A = (U * s) @ V.T
    e = rng.standard_normal(m)
    if cfg["align"] == "head":
        e[int(0.2 * m):] = 0.0
    elif cfg["align"] == "tail":
        e[: int(0.8 * m)] = 0.0
    e[s == 0.0] = 0.0  # keep v* in the row space when rank-deficient
    v_star = V @ (e / np.linalg.norm(e))
    y = cfg.get("scale", 1.0) * (A @ v_star)
    if cfg.get("inconsistent"):
        g = rng.standard_normal(R)
        g -= U @ (U.T @ g)          # project off the range of A
        g /= np.linalg.norm(g)
        y = y + 0.5 * np.linalg.norm(y) * g
    return A, y


def make_phi(fn_name: str, N: int, ratio: float):
    """Returns (arrays_for_npz, meta). QI geometry, repo conventions."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    centers = -1.0 + np.arange(-halo, N + halo + 1, dtype=np.float64) * h
    gamma = LAMBDA_STAR / h
    cols = centers.size + 1                      # [Phi, 1]
    R = int(round(ratio * cols))
    x_tr = np.linspace(-1.0, 1.0, R)
    x_ev = np.linspace(-1.0, 1.0, N_EVAL)
    f = get_target(fn_name).fn_numpy
    y_tr, y_ev = f(x_tr), f(x_ev)

    A = build_phi_aug(x_tr, gamma, centers)
    sol, info = solve_readout(A, y_tr, method="svd", rcond=RCOND)
    A_ev = build_phi_aug(x_ev, gamma, centers)
    resid_ev = A_ev @ sol - y_ev
    meta = {
        "cols": cols, "R": R, "halo": halo,
        "L_star": float(np.mean((A @ sol - y_tr) ** 2)),
        "L0": float(np.mean(y_tr ** 2)),
        "smax": info["smax"], "rank": info["rank"],
        "floor_eval_rel_l2": float(
            np.linalg.norm(resid_ev) / np.linalg.norm(y_ev)),
        "floor_eval_linf": float(np.abs(resid_ev).max()),
    }
    arrays = dict(x_tr=x_tr, y_tr=y_tr, x_ev=x_ev, y_ev=y_ev,
                  centers=centers, gamma=np.float64(gamma))
    return arrays, meta


def main():
    t0 = time.time()
    (PROBLEMS_DIR / "synthetic").mkdir(parents=True, exist_ok=True)
    (PROBLEMS_DIR / "phis").mkdir(parents=True, exist_ok=True)
    manifest = []

    seed = 0
    for cell, cfg in CELLS.items():
        for N in SYN_N:
            for ratio in SYN_RATIOS:
                R = ratio * N
                seed += 1
                A, y = make_synthetic(cell, cfg, N, R, seed)
                v_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
                s = np.linalg.svd(A, compute_uv=False)
                s_nz = s[s > 1e-15 * s[0]]
                pid = f"syn_{cell}_N{N}_R{R}"
                np.savez_compressed(
                    PROBLEMS_DIR / "synthetic" / f"{pid}.npz", A=A, y=y
                )
                manifest.append({
                    "id": pid, "kind": "synthetic", "cls": cell,
                    "N": N, "R": R, "cols": N, "ratio": ratio, "seed": seed,
                    "path": f"synthetic/{pid}.npz",
                    "L_star": float(np.mean((A @ v_hat - y) ** 2)),
                    "L0": float(np.mean(y ** 2)),
                    "smax": float(s[0]),
                    "cond": float(s_nz[0] / s_nz[-1]),
                    "rank": int(s_nz.size),
                })
        print(f"[{time.time()-t0:5.1f}s] synthetic cell {cell} done")

    for fn in PHI_FNS:
        for N in PHI_N:
            for ratio in PHI_RATIOS:
                arrays, meta = make_phi(fn, N, ratio)
                pid = f"phi_{fn}_N{N}_r{ratio:g}"
                np.savez_compressed(PROBLEMS_DIR / "phis" / f"{pid}.npz", **arrays)
                manifest.append({
                    "id": pid, "kind": "phi", "cls": fn,
                    "N": N, "R": meta["R"], "cols": meta["cols"],
                    "ratio": ratio, "seed": None,
                    "path": f"phis/{pid}.npz",
                    "L_star": meta["L_star"], "L0": meta["L0"],
                    "smax": meta["smax"], "cond": None, "rank": meta["rank"],
                    "floor_eval_rel_l2": meta["floor_eval_rel_l2"],
                    "floor_eval_linf": meta["floor_eval_linf"],
                })
        print(f"[{time.time()-t0:5.1f}s] phi target {fn} done")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    n_syn = sum(1 for e in manifest if e["kind"] == "synthetic")
    n_phi = len(manifest) - n_syn
    print(f"\nSaved {MANIFEST_PATH}: {n_syn} synthetic + {n_phi} phi problems "
          f"in {time.time()-t0:.1f}s")
    print("\nPhi floors (eval rel L2 of the truncated-SVD solve):")
    for e in manifest:
        if e["kind"] == "phi" and e["ratio"] == 4:
            print(f"  {e['id']:28s} floor={e['floor_eval_rel_l2']:.2e}")


if __name__ == "__main__":
    main()
