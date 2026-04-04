"""Feature matrix construction and exact readout solve.

Uses numpy/scipy (not torch) since this is in the construction context.

Phi[i, k] = tanh(gamma_k * (x_i - centers_k))

Solve methods: lstsq, qr, svd, ridge.
"""

from __future__ import annotations

import numpy as np


def build_phi(x: np.ndarray, gamma: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Build feature matrix Phi[i, k] = tanh(gamma_k * (x_i - centers_k)).

    x: [n_points], gamma: [width], centers: [width]
    Returns: [n_points, width]
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    centers = np.asarray(centers, dtype=np.float64).ravel()
    return np.tanh(gamma[None, :] * (x[:, None] - centers[None, :]))


def solve_readout(Phi: np.ndarray, y: np.ndarray, method: str = "lstsq",
                  ridge_alpha: float = 0.0) -> tuple[np.ndarray, dict]:
    """Solve Phi @ v = y. Returns (v, info_dict)."""
    y = np.asarray(y, dtype=np.float64).ravel()

    if method == "lstsq":
        result = np.linalg.lstsq(Phi, y, rcond=None)
        v = result[0]
        info = {"residual_norm": float(np.linalg.norm(Phi @ v - y)),
                "rank": int(result[2]) if len(result) > 2 else Phi.shape[1]}
    elif method == "qr":
        Q, R = np.linalg.qr(Phi)
        v = np.linalg.solve(R, Q.T @ y)
        info = {"residual_norm": float(np.linalg.norm(Phi @ v - y))}
    elif method == "svd":
        U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
        v = Vt.T @ (np.diag(1.0 / s) @ (U.T @ y))
        info = {"residual_norm": float(np.linalg.norm(Phi @ v - y)),
                "cond": float(s[0] / s[-1]) if s[-1] > 0 else float("inf"),
                "smin": float(s[-1]), "smax": float(s[0])}
    elif method == "ridge":
        # (Phi^T Phi + alpha I) v = Phi^T y
        A = Phi.T @ Phi + ridge_alpha * np.eye(Phi.shape[1])
        v = np.linalg.solve(A, Phi.T @ y)
        info = {"residual_norm": float(np.linalg.norm(Phi @ v - y)),
                "ridge_alpha": ridge_alpha}
    else:
        raise ValueError(f"Unknown method: {method}")

    return v, info


def solve_readout_with_bias(Phi: np.ndarray, y: np.ndarray, method: str = "lstsq",
                            ridge_alpha: float = 0.0) -> tuple[np.ndarray, float, dict]:
    """Solve [Phi, 1] @ [v; b] = y. Returns (v, bias, info_dict).

    Bias is unpenalized in ridge mode.
    """
    n, w = Phi.shape
    ones = np.ones((n, 1), dtype=np.float64)
    Phi_aug = np.hstack([Phi, ones])

    if method == "ridge":
        # Penalize only the v part, not the bias
        reg = np.zeros((w + 1, w + 1), dtype=np.float64)
        np.fill_diagonal(reg[:w, :w], ridge_alpha)
        A = Phi_aug.T @ Phi_aug + reg
        sol = np.linalg.solve(A, Phi_aug.T @ y.ravel())
    else:
        sol, info = solve_readout(Phi_aug, y, method=method)
        v = sol[:w]
        bias = float(sol[w])
        info["residual_norm"] = float(np.linalg.norm(Phi @ v + bias - y.ravel()))
        return v, bias, info

    v = sol[:w]
    bias = float(sol[w])
    info = {"residual_norm": float(np.linalg.norm(Phi @ v + bias - y.ravel())),
            "ridge_alpha": ridge_alpha}
    return v, bias, info
