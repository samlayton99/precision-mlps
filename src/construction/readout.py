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
    # TODO: implement
    raise NotImplementedError


def solve_readout(Phi: np.ndarray, y: np.ndarray, method: str = "lstsq",
                  ridge_alpha: float = 0.0) -> tuple[np.ndarray, dict]:
    """Solve Phi @ v = y. Returns (v, info_dict)."""
    # TODO: implement
    raise NotImplementedError


def solve_readout_with_bias(Phi: np.ndarray, y: np.ndarray, method: str = "lstsq",
                            ridge_alpha: float = 0.0) -> tuple[np.ndarray, float, dict]:
    """Solve [Phi, 1] @ [v; b] = y. Returns (v, bias, info_dict).

    Bias is unpenalized in ridge mode.
    """
    # TODO: implement
    raise NotImplementedError
