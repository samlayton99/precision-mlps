"""Optimizer dispatch using PyTorch built-in optimizers.

Supports: Adam, SGD, LBFGS. No custom optimizers.
LBFGS uses PyTorch's built-in closure pattern with strong Wolfe line search,
replacing the need for SSBroyden or custom second-order methods.

Usage:
    optimizer = build_optimizer(stage_config, model)
    scheduler = build_scheduler(optimizer, stage_config, total_steps)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.config.schema import OptimizerStageConfig


def build_optimizer(config: OptimizerStageConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Build optimizer for a training stage.

    Only passes parameters with requires_grad=True to the optimizer.

    Args:
        config: OptimizerStageConfig (name, lr, kwargs).
        model: nn.Module instance.

    Returns:
        PyTorch optimizer.
    """
    # TODO: implement
    # trainable = [p for p in model.parameters() if p.requires_grad]
    # if config.name == "adam":
    #     return torch.optim.Adam(trainable, lr=config.learning_rate)
    # elif config.name == "lbfgs":
    #     return torch.optim.LBFGS(trainable, lr=config.learning_rate,
    #                               max_iter=config.kwargs.get("max_iter", 20),
    #                               line_search_fn="strong_wolfe")
    # elif config.name == "sgd":
    #     return torch.optim.SGD(trainable, lr=config.learning_rate)
    raise NotImplementedError


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizerStageConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build LR scheduler. Returns None if no schedule configured.

    Args:
        optimizer: PyTorch optimizer.
        config: OptimizerStageConfig.
        total_steps: Total steps in this stage.

    Returns:
        Scheduler or None.
    """
    # TODO: implement
    # if config.use_cosine_schedule:
    #     return torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=total_steps, eta_min=config.min_learning_rate)
    # return None
    raise NotImplementedError
