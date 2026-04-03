"""Multi-stage training loop.

Supports Adam, SGD (standard step) and LBFGS (closure-based step).

Usage:
    result = run_training(config, model, dataset)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from src.config.schema import ExperimentConfig
from src.data.dataset import Dataset
from src.training.metrics import MetricsCollector


@dataclass
class TrainResult:
    """Result of a complete training run."""
    loss_history: list[float] = field(default_factory=list)
    eval_metrics: dict[str, list[float]] = field(default_factory=dict)
    eval_steps: list[int] = field(default_factory=list)
    stage_boundaries: list[int] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> float:
    """Standard training step (Adam, SGD). Returns loss value."""
    # TODO: implement
    raise NotImplementedError


def train_step_lbfgs(model: nn.Module, optimizer: torch.optim.LBFGS,
                     x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> float:
    """LBFGS training step with closure."""
    # TODO: implement
    raise NotImplementedError


def run_training(config: ExperimentConfig, model: nn.Module, dataset: Dataset) -> TrainResult:
    """Execute full multi-stage training.

    For each stage in config.training.stages:
    1. Build optimizer and scheduler.
    2. Select step function (standard or LBFGS closure).
    3. Run training steps, evaluate at config.training.eval_interval.
    4. Optionally solve readout between stages.
    """
    # TODO: implement
    raise NotImplementedError
