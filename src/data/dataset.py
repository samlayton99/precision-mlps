"""Dataset construction from config.

Eval points are always dense equispaced for consistent L_inf measurement.
Noise applied only to training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config.schema import ExperimentConfig
from src.data.targets import TargetFn


@dataclass
class Dataset:
    """Train/eval data. All tensors float64, shape [n, 1]."""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_eval: torch.Tensor
    y_eval: torch.Tensor
    target_fn: TargetFn


def build_dataset(config: ExperimentConfig) -> Dataset:
    """Build train/eval dataset from experiment config.

    1. Look up target function via config.target.
    2. Sample x_train using config.sampling strategy.
    3. Compute y_train = target.fn(x_train).
    4. If config.x_noise_std > 0: perturb x_train, recompute y on perturbed x.
    5. If config.y_noise_std > 0: add Gaussian noise to y_train.
    6. x_eval = equispaced(config.n_eval, config.domain). No noise on eval.
    """
    # TODO: implement
    raise NotImplementedError
