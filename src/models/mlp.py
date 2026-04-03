"""Single-hidden-layer tanh MLP for quasi-interpolant experiments.

QIMlp implements:  f(x) = readout(tanh(inner_layer(x)))

The inner layer type (GammaLinear, GammaExpLinear, StandardLinear) determines
the parameterization. The readout is always nn.Linear(width, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schema import ModelConfig
from src.models.layers import get_layer


class QIMlp(nn.Module):
    """Single-hidden-layer tanh MLP.

    Architecture:
        x: [batch, 1] -> inner_layer: [batch, width] -> tanh -> readout: [batch, 1]
    """

    def __init__(self, config: ModelConfig, **layer_kwargs):
        # TODO: implement
        # super().__init__()
        # self.config = config
        # self.inner_layer = get_layer(config.layer_type, config.width, **layer_kwargs)
        # self.readout = nn.Linear(config.width, 1)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: [batch, 1] -> [batch, 1]."""
        # TODO: implement
        # return self.readout(torch.tanh(self.inner_layer(x)))
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature matrix Phi = tanh(inner_layer(x)). Shape [n_points, width]."""
        # TODO: implement
        raise NotImplementedError

    def get_gamma(self) -> torch.Tensor:
        """Effective gamma values, shape [width]. Handles all layer types."""
        # TODO: implement
        # GammaLinear: self.inner_layer.gamma.data.squeeze(0)
        # GammaExpLinear: self.inner_layer.get_gamma().squeeze(0)
        # StandardLinear: self.inner_layer.linear.weight.data.squeeze(1) (interpretation differs)
        raise NotImplementedError

    def get_centers(self) -> torch.Tensor:
        """Center positions, shape [width]."""
        # TODO: implement
        raise NotImplementedError

    def get_readout_weights(self) -> torch.Tensor:
        """Outer weights v_k, shape [width]."""
        # TODO: implement
        # return self.readout.weight.data.squeeze(0)
        raise NotImplementedError

    def get_readout_bias(self) -> float:
        """Readout bias (scalar)."""
        # TODO: implement
        raise NotImplementedError

    def get_lambda(self, h: float) -> torch.Tensor:
        """lambda = gamma * h for each neuron, shape [width]."""
        # TODO: implement
        raise NotImplementedError
