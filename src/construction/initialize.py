"""Project QI construction results into model parameters.

Three patterns:
1. Full construction init: copy gamma, centers, readout from QIResult.
2. Partial construction: copy only some parameter groups.
3. Readout solve: given current inner layer, solve readout via numpy lstsq.

All parameter writes use torch.no_grad().
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.construction.qi_mpmath import QIResult
from src.config.schema import FreezeConfig, InitConfig


def initialize_from_construction(
    model: nn.Module,
    qi_result: QIResult,
    *,
    construct_gamma: bool = True,
    construct_centers: bool = True,
    construct_readout: bool = True,
) -> None:
    """Copy QIResult values into model parameters in-place.

    Uses torch.no_grad() for all parameter writes.

    For GammaLinear: gamma.data = qi_result.gamma (broadcast to [1, width])
    For GammaExpLinear: log_gamma.data = log(lambda_star)
    Readout: weight.data = interior_a_coeffs reshaped to [1, width]
             bias.data = c0
    """
    # TODO: implement
    raise NotImplementedError


def initialize_and_freeze(
    model: nn.Module,
    qi_result: QIResult,
    freeze_config: FreezeConfig,
    init_config: InitConfig,
) -> None:
    """Initialize from construction, then freeze per config.

    Geometry ladder entry point (Experiment 3).
    """
    # TODO: implement
    raise NotImplementedError


def initialize_with_readout_solve(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    method: str = "lstsq",
    ridge_alpha: float = 0.0,
) -> dict:
    """Solve readout weights via numpy lstsq given current inner layer.

    Detaches Phi to numpy, solves, copies weights back with torch.no_grad().
    Returns info dict with solve diagnostics.
    """
    # TODO: implement
    raise NotImplementedError
