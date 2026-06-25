"""Shared helpers for the geometry-interpolation experiment family (centers /
weights / weight+bias). Three run scripts import from here:

  run_centers.py     -- interpolate center positions (Xavier -> uniform), sweep gamma.
  run_weights.py     -- interpolate inner weights (Xavier -> 1*gamma), sweep gamma.
  run_weightbias.py  -- interpolate both (Xavier (w,b) -> ideal), gamma fixed per (N,target).

All three share the same uniform QI geometry, the same Glorot draw / neuron ordering,
the same truncated-SVD least-squares readout, and the same dense train/eval grids, so
their results are directly comparable. The functions here are byte-for-byte the inline
logic the scripts used before the merge -- the readout and geometry are unchanged.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import default_halo          # noqa: E402  (re-exported)
from src.data.targets import get_target                       # noqa: E402  (re-exported)

# Base results dir for the merged experiment; each mode writes to a subdir.
EXP_RESULTS = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC05_geometry_interpolation"

# --- shared knobs (identical across the three modes) ---
HALO_LAMBDA = 0.25                                # sizes the uniform grid's halo
N_TRAIN = 2003                                    # prime, overdetermined for all W (W<=921 at N=512)
N_EVAL = 4001                                     # prime, misaligned with the train grid
RCOND = 1e-13                                     # SVD truncation (relative to s_max)


def uniform_geometry(N, halo_lambda=HALO_LAMBDA):
    """QI grid + halo: the ideal uniform centers and h = 2/N for this width.

    These FIXED, ideal positions are the geometry the t=1 / s=1 corners target:
    centers = c_uniform, bias = -gamma * c_uniform. gamma-independent."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=halo_lambda)
    idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + idx.astype(np.float64) * h          # ascending, x_n = -1 + n*h
    return centers, h, halo


def xavier_draw(W, seed, N):
    """The genuine Glorot/Xavier draw shared by all three modes: per-neuron weights
    and biases, both uniform on +-sqrt(6/(1+W)), seeded per (folder seed, width).
    Returns (w, b) in draw order (w first, then b) -- identical RNG stream to the
    pre-merge scripts."""
    bound = float(np.sqrt(6.0 / (1.0 + W)))
    rng = np.random.default_rng([seed, N])
    w = rng.uniform(-bound, bound, size=W)
    b = rng.uniform(-bound, bound, size=W)
    return w.astype(np.float64), b.astype(np.float64)


def center_order(w, b, eps=1e-8):
    """Ascending neuron order by Xavier center c = -b/w, dead |w|<eps parked by bias
    sign (at -1/+1), stable argsort -- the ordering the weights / weight+bias modes use
    to pin the i-th neuron to the i-th ascending uniform grid center."""
    w = np.asarray(w); b = np.asarray(b)
    safe = np.abs(w) >= eps
    centers = np.empty_like(w)
    centers[safe] = -b[safe] / w[safe]
    centers[~safe] = np.where(b[~safe] >= 0, -1.0, 1.0)
    return np.argsort(centers, kind="stable")


def solve_relL2(Phi_tr, Phi_ev, Y_tr_cols, Y_ev_cols, y_norms, rcond=RCOND):
    """Truncated-SVD readout (explicit bias column) for one or more RHS; returns the
    per-column eval relative L2. Identical to the inline solve each script used."""
    Aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
    U, sv, Vt = np.linalg.svd(Aug, full_matrices=False)
    s_inv = np.where(sv > rcond * sv[0], 1.0 / np.where(sv > 0, sv, 1.0), 0.0)
    sol = Vt.T @ (s_inv[:, None] * (U.T @ Y_tr_cols))        # [W+1, ncols]
    resid = Phi_ev @ sol[:-1] + sol[-1] - Y_ev_cols          # [n_ev, ncols]
    return np.linalg.norm(resid, axis=0) / y_norms
