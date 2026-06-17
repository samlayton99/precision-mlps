"""Coefficient-comparison metrics for exp0B.

On a FIXED tanh geometry (shared centers + gamma), the QI construction and a
least-squares readout both produce an augmented coefficient vector
beta = [outer weights; bias] that solves A @ beta = y, with A = [Phi, 1].

But A is heavily rank-deficient (~half its columns are a null space the data
cannot see). So "are QI and lstsq the same solution?" has three different
answers depending on where you measure:

    raw_diff       distance in the full coefficient space  -- sees the null
                   space, so it is LARGE even when the functions agree.
    rowspace_diff  distance projected onto A's row space   -- the only part of
                   the coefficient difference the data constrains.
    function_diff  distance of the functions they compute  -- the honest
                   "same solution" measure.

All pure numpy; imported by run.py and exercised by tests/test_exp0B.py.
"""

from __future__ import annotations

import numpy as np


def raw_diff(beta_qi: np.ndarray, beta_ls: np.ndarray) -> dict:
    """L2 / Linf / relative-L2 of the full coefficient difference."""
    d = np.asarray(beta_qi) - np.asarray(beta_ls)
    nrm_qi = float(np.linalg.norm(beta_qi)) or 1.0
    l2 = float(np.linalg.norm(d))
    return {"l2": l2, "linf": float(np.max(np.abs(d))), "rel_l2": l2 / nrm_qi}


def rowspace_diff(A: np.ndarray, beta_qi: np.ndarray, beta_ls: np.ndarray,
                  rcond: float = 1e-13) -> dict:
    """Coefficient difference projected onto the row space of A.

    The null space of A is invisible to the data, so two coefficient vectors
    that differ only there compute the same values on the training points. We
    keep only the right singular vectors with singular value above rcond*s_max
    and report the norm of the difference within that span.
    """
    d = np.asarray(beta_qi) - np.asarray(beta_ls)
    s, Vt = np.linalg.svd(np.asarray(A), full_matrices=False)[1:]
    smax = s[0] if s.size else 0.0
    keep = s > rcond * smax
    Vr = Vt[keep]                       # [rank, p], orthonormal rows
    coords = Vr @ d                     # difference expressed in the row basis
    recon = Vr.T @ coords              # row-space component back in coeff space
    return {"l2": float(np.linalg.norm(coords)),
            "linf": float(np.max(np.abs(recon))) if keep.any() else 0.0,
            "rank": int(keep.sum())}


def function_diff(A_eval: np.ndarray, beta_qi: np.ndarray, beta_ls: np.ndarray,
                  y_eval: np.ndarray | None = None) -> dict:
    """L2 / Linf of (f_qi - f_ls) evaluated through A_eval = [Phi_eval, 1]."""
    d = np.asarray(A_eval) @ (np.asarray(beta_qi) - np.asarray(beta_ls))
    out = {"l2": float(np.linalg.norm(d)), "linf": float(np.max(np.abs(d)))}
    if y_eval is not None:
        out["rel_l2"] = float(np.linalg.norm(d) / (np.linalg.norm(y_eval) or 1.0))
    return out


def fit_residual(A: np.ndarray, beta: np.ndarray, y: np.ndarray) -> float:
    """||A @ beta - y||_2 -- how well a coefficient vector fits the data."""
    return float(np.linalg.norm(np.asarray(A) @ np.asarray(beta) - np.asarray(y)))


# ---------------------------------------------------------------------------
# Headline metric: largest element-wise deviation vs. the average coefficient.
# ---------------------------------------------------------------------------

def deviation_ratio(beta_qi: np.ndarray, beta_ls: np.ndarray) -> float:
    """linf(beta_qi - beta_ls) / max(mean|beta_qi|, mean|beta_ls|).

    "How big is the largest disagreement between corresponding coefficients,
    measured in units of the typical coefficient size?" A value near 0 means the
    two vectors are effectively identical; a value of O(1) means the worst
    element disagrees by as much as a typical coefficient is large.
    """
    beta_qi = np.asarray(beta_qi)
    beta_ls = np.asarray(beta_ls)
    denom = max(float(np.mean(np.abs(beta_qi))), float(np.mean(np.abs(beta_ls))))
    linf = float(np.max(np.abs(beta_qi - beta_ls)))
    return linf / denom if denom else float("inf")


def project_rowspace(A: np.ndarray, beta: np.ndarray,
                     rcond: float = 1e-11) -> tuple[np.ndarray, int]:
    """Orthogonal projection of beta onto the row space of A.

    P = V_r V_r^T, where V_r are the right singular vectors with singular value
    above rcond*s_max. P*beta is the part of the coefficient vector the data can
    see -- equivalently, the minimum-norm vector that produces the same function
    A@beta. The discarded (null-space) part does not affect the function at all.

    The default rcond is 1e-11, NOT machine eps: the last decade of singular
    values sits at the fp64 floor and carries spurious disagreement, so we treat
    those directions as effectively null. Returns (P*beta, rank).
    """
    s, Vt = np.linalg.svd(np.asarray(A), full_matrices=False)[1:]
    smax = s[0] if s.size else 0.0
    keep = s > rcond * smax
    Vr = Vt[keep]
    return Vr.T @ (Vr @ np.asarray(beta)), int(keep.sum())


def rowspace_ratio(A: np.ndarray, beta_qi: np.ndarray, beta_ls: np.ndarray,
                   rcond: float = 1e-11) -> dict:
    """deviation_ratio computed on the row-space projections of both vectors.

    This compares the two solutions only in the subspace the data constrains.
    NOTE (do not oversell): the row-space projection is the min-norm
    representative of each function, so a near-zero ratio here is the SAME fact
    as the two functions agreeing -- not independent evidence. The independent
    finding is that the full-space deviation_ratio stays O(1): the methods reach
    the same function through representations that differ freely in the null
    space.
    """
    Pq, rank = project_rowspace(A, beta_qi, rcond)
    Pl, _ = project_rowspace(A, beta_ls, rcond)
    return {"ratio": deviation_ratio(Pq, Pl),
            "mean_qi": float(np.mean(np.abs(Pq))),
            "mean_ls": float(np.mean(np.abs(Pl))),
            "rank": rank}
