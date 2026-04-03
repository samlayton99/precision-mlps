"""Inner-layer parameterizations for single-hidden-layer tanh MLPs.

Three layer types, each representing the pre-activation gamma * (x - center):

GammaLinear:     Raw parameterization. Stores gamma and centers directly.
                 Forward: gamma * (x - centers).
                 Natural match to QI construction but gradients vanish at large gamma.

GammaExpLinear:  Log-gamma reparameterization. Stores log_gamma and centers.
                 Forward: exp(log_gamma) / h * (x - centers).
                 Gradient d(phi)/d(log_gamma) = O(1) instead of O(1/N).

StandardLinear:  Standard weight/bias parameterization (nn.Linear wrapper).
                 Forward: x @ weight + bias.
                 Baseline for reparameterization experiments (Experiment 8).

All layers: input [batch, 1], output [batch, width], dtype float64.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GammaLinear(nn.Module):
    """Linear layer parameterized as gamma * (x - center).

    Parameters:
        gamma:   nn.Parameter, shape [1, width].
        centers: nn.Parameter, shape [1, width].

    QI construction sets:
        gamma = lambda_star / h  (scalar, broadcast to all neurons)
        centers = [-1 + k*h for k in range(N)]  (uniform grid)
    """

    def __init__(self, width: int, gamma_init=None, center_init=None):
        # TODO: implement
        # self.gamma = nn.Parameter(gamma_init or torch.ones(1, width))
        # self.centers = nn.Parameter(center_init or torch.zeros(1, width))
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """gamma * (x - centers). x: [batch, 1] -> [batch, width]."""
        # TODO: implement
        raise NotImplementedError


class GammaExpLinear(nn.Module):
    """Linear layer with gamma = exp(log_gamma) / h.

    Reparameterization ensuring gradient d(loss)/d(log_gamma) = O(1).

    Parameters:
        log_gamma: nn.Parameter, shape [1, width].
        centers:   nn.Parameter, shape [1, width].
        h:         float (not a parameter). Grid spacing 2/N.

    For QI init: log_gamma = log(lambda_star), h = 2/N,
    so gamma = lambda_star / h = lambda_star * N / 2.
    """

    def __init__(self, width: int, h: float = 1.0, log_gamma_init=None, center_init=None):
        # TODO: implement
        # self.log_gamma = nn.Parameter(log_gamma_init or torch.zeros(1, width))
        # self.centers = nn.Parameter(center_init or torch.zeros(1, width))
        # self.h = h
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """exp(log_gamma)/h * (x - centers). x: [batch, 1] -> [batch, width]."""
        # TODO: implement
        raise NotImplementedError

    def get_gamma(self) -> torch.Tensor:
        """Return effective gamma = exp(log_gamma) / h."""
        # TODO: implement
        raise NotImplementedError


class StandardLinear(nn.Module):
    """Standard weight/bias linear layer. Baseline for Experiment 8.

    Wraps nn.Linear(1, width). Does not separate gamma and center.
    """

    def __init__(self, width: int):
        # TODO: implement
        # self.linear = nn.Linear(1, width)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x @ weight + bias. x: [batch, 1] -> [batch, width]."""
        # TODO: implement
        raise NotImplementedError


def get_layer(layer_type: str, width: int, **kwargs) -> nn.Module:
    """Factory for inner layer types.

    Args:
        layer_type: "gamma_linear" | "gamma_exp" | "standard"
        width: Number of hidden neurons.
        **kwargs: Layer-specific kwargs (h for GammaExpLinear, inits).

    Returns:
        Instantiated layer module.
    """
    # TODO: implement
    raise NotImplementedError
