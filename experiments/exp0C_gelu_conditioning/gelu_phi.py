"""GELU feature matrix and conditioning helpers for exp0C.

Mirrors src/construction/readout.build_phi but with an exact GELU activation
(gelu(z) = 0.5 z (1 + erf(z/sqrt(2)))) instead of tanh. Pure numpy/scipy;
imported by run.py and exercised by tests/test_exp0C.py.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def gelu(z) -> np.ndarray:
    """Exact GELU: 0.5 * z * (1 + erf(z / sqrt(2)))."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))


def build_phi_gelu(x: np.ndarray, gamma: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Phi[i, k] = gelu(gamma_k * (x_i - centers_k)). Shape [n_points, width]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    centers = np.asarray(centers, dtype=np.float64).ravel()
    return gelu(gamma[None, :] * (x[:, None] - centers[None, :]))


def conditioning(A: np.ndarray, rcond: float = 1e-13) -> dict:
    """SVD-based conditioning of A.

    rank   = number of singular values above rcond*s_max (numerically nonzero),
    null   = ncols - rank,
    cond   = ratio of the largest to the smallest KEPT singular value (the
             effective condition number; the full s_max/s_min is often inf for
             GELU because the smallest singular values underflow to 0),
    smax/smin = largest / smallest singular value (smin may be 0).
    """
    s = np.linalg.svd(np.asarray(A), compute_uv=False)
    smax = s[0] if s.size else 0.0
    keep = s > rcond * smax
    skept = s[keep]
    cond = float(skept[0] / skept[-1]) if skept.size else float("inf")
    return {"rank": int(keep.sum()),
            "null": int(A.shape[1] - keep.sum()),
            "cond": cond,
            "smax": float(smax),
            "smin": float(s[-1]) if s.size else 0.0}
