"""Target function registry for approximation experiments.

6 categories: low_freq, high_freq, boundary_layer, mixed_scale, polynomial, rough.

Each TargetFn provides fn (torch) and deriv (torch) for training/evaluation,
plus fn_numpy and deriv_numpy for construction (mpmath needs numpy-compatible callables).

Usage:
    target = get_target("sine")
    y = target.fn(x)          # x is torch.Tensor
    y_np = target.fn_numpy(x_np)  # x_np is numpy array
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import numpy as np


@dataclass(frozen=True)
class TargetFn:
    """A target function with analytic derivative.

    Fields:
        fn:          f(x) -> y, operates on torch tensors.
        deriv:       f'(x) -> y', analytic derivative (torch).
        fn_numpy:    f(x) -> y, operates on numpy arrays (for construction).
        deriv_numpy: f'(x) -> y', numpy version (for construction).
        name:        Registry key.
        category:    One of the 6 target categories.
    """
    fn: Callable[[torch.Tensor], torch.Tensor]
    deriv: Callable[[torch.Tensor], torch.Tensor]
    fn_numpy: Callable[[np.ndarray], np.ndarray]
    deriv_numpy: Callable[[np.ndarray], np.ndarray]
    name: str
    category: str


# TODO: populate TARGET_REGISTRY with functions across all 6 categories.
#
# Low-frequency analytic:
#   "sine"         f(x) = sin(2*pi*x),         f'(x) = 2*pi*cos(2*pi*x)
#   "cosine"       f(x) = cos(2*pi*x),         f'(x) = -2*pi*sin(2*pi*x)
#
# High-frequency analytic:
#   "sine_8pi"     f(x) = sin(8*pi*x),         f'(x) = 8*pi*cos(8*pi*x)
#
# Boundary-layer / steep transition:
#   "runge"        f(x) = 1/(1 + 25*x^2),      f'(x) = -50*x/(1 + 25*x^2)^2
#   "tanh_steep"   f(x) = tanh(20*x),           f'(x) = 20*sech^2(20*x)
#
# Mixed-scale analytic:
#   "sine_mixture" f(x) = sin(2*pi*x) + 0.5*sin(6*pi*x) + 0.25*sin(14*pi*x)
#
# Polynomial / entire function:
#   "exp"          f(x) = exp(x),               f'(x) = exp(x)
#   "poly5"        f(x) = x^5 - 3*x^3 + x,     f'(x) = 5*x^4 - 9*x^2 + 1
#
# Slightly rough:
#   "abs_cubed"    f(x) = |x|^3,                f'(x) = 3*x*|x|

TARGET_REGISTRY: dict[str, TargetFn] = {}


def get_target(name: str) -> TargetFn:
    """Look up target function by name."""
    # TODO: implement
    raise NotImplementedError


def get_all_targets() -> list[TargetFn]:
    """Return all registered targets."""
    return list(TARGET_REGISTRY.values())
