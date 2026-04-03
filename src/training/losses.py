"""Loss functions for training.

All follow signature: (model, x, y) -> scalar tensor.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    # TODO: implement
    raise NotImplementedError


def lp_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, p: float = 6.0) -> torch.Tensor:
    """Lp loss: mean(|f(x) - y|^p). Approximates L_inf as p -> inf."""
    # TODO: implement
    raise NotImplementedError


def hybrid_boundary_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         boundary_weight: float = 5.0, boundary_fraction: float = 0.1) -> torch.Tensor:
    """MSE with extra weight on points near domain boundaries."""
    # TODO: implement
    raise NotImplementedError


def get_loss_fn(name: str, **kwargs) -> Callable:
    """Look up loss function by name. Returns (model, x, y) -> scalar.

    Usage:
        loss_fn = get_loss_fn("mse")
        loss_fn = get_loss_fn("lp", p=6.0)
    """
    if name == "mse":
        return mse
    elif name == "lp":
        return partial(lp_loss, p=kwargs.get("p", 6.0))
    elif name == "hybrid_boundary":
        return partial(hybrid_boundary_loss, **kwargs)
    raise ValueError(f"Unknown loss: {name}")
