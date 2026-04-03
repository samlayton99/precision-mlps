"""Parameter freezing via requires_grad.

PyTorch freezing is simple: param.requires_grad_(False) excludes a parameter
from gradient computation. No custom variable types needed.

Usage:
    model = QIMlp(config)
    initialize_from_construction(model, qi_result)
    freeze_inner_layer(model)       # gamma + centers frozen
    # Training only updates readout weights
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


def freeze_params(model: nn.Module, filter_fn: Callable[[str, nn.Parameter], bool]) -> None:
    """Freeze parameters where filter_fn returns True.

    Args:
        model: Any nn.Module.
        filter_fn: (name, param) -> bool. True = freeze.
    """
    # TODO: implement
    # for name, param in model.named_parameters():
    #     if filter_fn(name, param):
    #         param.requires_grad_(False)
    raise NotImplementedError


def unfreeze_params(model: nn.Module, filter_fn: Callable[[str, nn.Parameter], bool]) -> None:
    """Unfreeze parameters where filter_fn returns True."""
    # TODO: implement
    raise NotImplementedError


def freeze_inner_layer(model: nn.Module) -> None:
    """Freeze all inner layer parameters (gamma/log_gamma and centers)."""
    # TODO: implement
    raise NotImplementedError


def freeze_gamma(model: nn.Module) -> None:
    """Freeze only gamma (or log_gamma). Keep centers trainable."""
    # TODO: implement
    raise NotImplementedError


def freeze_centers(model: nn.Module) -> None:
    """Freeze only centers."""
    # TODO: implement
    raise NotImplementedError


def freeze_readout(model: nn.Module) -> None:
    """Freeze readout weights and bias."""
    # TODO: implement
    raise NotImplementedError


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    # TODO: implement
    raise NotImplementedError


def get_trainable_param_count(model: nn.Module) -> int:
    """Count parameters with requires_grad=True."""
    # TODO: implement
    raise NotImplementedError
