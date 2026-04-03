"""High-precision QI construction using mpmath.

Constructs a single-hidden-layer tanh MLP that approximates a target function
to machine precision (~10^-15) on [-1, 1].

Algorithm (from the paper):
    1. Set up uniform grid: x_k = -1 + k*h, k = 0, ..., N, with h = 2/N.
    2. Set gamma = lambda_star / h (bandwidth scaling with width).
    3. Solve Toeplitz system for cardinal function coefficients c_k:
       sum_j c_j * h * Kd((r-j)*h) = h * delta_{r,0}
       where Kd(x) = gamma * sech^2(gamma * x) is the derivative kernel.
    4. Convolve c_k with target derivative values g'(x_k) using Kahan summation
       to get outer weights a_n (coefficients of the quasi-interpolant).
    5. Compute bias c0 from boundary condition: q(-1) = g(-1).
    6. Return QIResult with all construction data.

The MLP then computes:
    q(x) = c0 + sum_n a_n * tanh(gamma * (x - x_n))

Key implementation notes:
- Use mpmath for the Toeplitz solve (arbitrary precision arithmetic).
- Use Kahan compensated summation for the convolution step.
- Ghost nodes (halo) extend the grid beyond [-1, 1] for boundary accuracy.
- Project final coefficients to fp64 for use in JAX models.

Adapted from continuous-mlps/src/construction/explicit_quasi_interpolant.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class QIResult:
    """Immutable result of QI construction.

    This is a pure data object -- it never references a model.
    Use initialize.py to copy values into a QIMlp.

    Fields:
        N:                  Number of interior grid points (width).
        h:                  Grid spacing 2/N.
        gamma:              Bandwidth parameter lambda_star / h.
        lambda_val:         Dimensionless bandwidth gamma * h = lambda_star.
        centers:            Grid positions including halo, shape [N + 2*halo + 1].
        interior_centers:   Grid positions without halo, shape [N + 1].
        a_coeffs:           Outer weights (QI coefficients), shape [N + 2*halo + 1].
        interior_a_coeffs:  Outer weights without halo, shape [N + 1].
        c0:                 Bias constant from boundary condition.
        toeplitz_residual:  Residual norm of the Toeplitz solve (for diagnostics).
        halo:               Number of ghost nodes on each side.
        Kc:                 Toeplitz stencil half-width.
    """
    N: int
    h: float
    gamma: float
    lambda_val: float
    centers: np.ndarray
    interior_centers: np.ndarray
    a_coeffs: np.ndarray
    interior_a_coeffs: np.ndarray
    c0: float
    toeplitz_residual: float
    halo: int
    Kc: int


def construct_qi(
    target_fn: Callable[[float], float],
    target_deriv: Callable[[float], float],
    N: int,
    *,
    lambda_star: float = 1.5,
    Kc: int = 12,
    halo: int = 8,
    mp_dps: int = 50,
) -> QIResult:
    """Construct a QI MLP approximation using mpmath.

    Implementation steps:
    1. Compute grid: x_k = -1 + k*h for k in range(-halo, N + halo + 1), h = 2/N.
    2. Compute gamma = lambda_star / h.
    3. Build Toeplitz matrix T[r, j] = h * Kd((r-j)*h) where Kd(x) = gamma * sech^2(gamma*x).
       Use mpmath.sech for high-precision evaluation. Matrix size: (2*Kc + 1) x (2*Kc + 1).
    4. Solve T @ c = e_Kc (unit vector at center) for cardinal coefficients c.
    5. Evaluate target derivative at grid points: g'(x_k) for all k including halo.
    6. Convolve: a_n = sum_j c_j * g'(x_{n+j}) * h, using Kahan compensated summation.
    7. Compute bias: c0 = g(-1) - sum_n a_n * tanh(gamma * (-1 - x_n)).
    8. Project all coefficients to float64.
    9. Return QIResult.

    Args:
        target_fn:    Target function g(x). Must be evaluable at grid points.
        target_deriv: Derivative g'(x). Needed for the quasi-interpolant construction.
        N:            Number of interior grid intervals (= width of MLP).
        lambda_star:  Target dimensionless bandwidth (default 1.5).
        Kc:           Toeplitz stencil half-width (default 12).
        halo:         Ghost nodes on each side for boundary accuracy (default 8).
        mp_dps:       mpmath decimal places for high-precision arithmetic (default 50).

    Returns:
        QIResult with all construction data.
    """
    # TODO: implement
    raise NotImplementedError
