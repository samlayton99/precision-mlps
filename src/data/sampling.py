"""Sampling functions for training and evaluation points.

All return [n_points, 1] float64 torch tensors, sorted.
"""

from __future__ import annotations

import torch


def equispaced(n_points: int, domain: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """linspace(a, b, n) including endpoints."""
    # TODO: implement
    raise NotImplementedError


def uniform_random(n_points: int, domain: tuple[float, float] = (-1.0, 1.0),
                   seed: int = 0) -> torch.Tensor:
    """Uniform random in [a, b] with endpoints always included, sorted."""
    # TODO: implement
    raise NotImplementedError


def chebyshev(n_points: int, domain: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Chebyshev nodes cos(pi*k/(n-1)) mapped to [a, b], sorted."""
    # TODO: implement
    raise NotImplementedError


def qi_grid(N: int, domain: tuple[float, float] = (-1.0, 1.0),
            halo: int = 0) -> torch.Tensor:
    """QI grid x_k = a + k*h for k in range(-halo, N + halo + 1)."""
    # TODO: implement
    raise NotImplementedError


def get_sampling_fn(name: str):
    """Look up sampling function by name: "equispaced" | "uniform" | "chebyshev" | "qi_grid"."""
    fns = {"equispaced": equispaced, "uniform": uniform_random,
           "chebyshev": chebyshev, "qi_grid": qi_grid}
    return fns[name]
