"""Smooth synthetic targets for the dimension ladder (expF04/ladder).

Three structural regimes, each defined for any input dimension d on [-1,1]^d:

  ridge_mix   sum of 3 fixed random ridge sines      -- ridge-structured (favorable
              f(x) = (1/3) sum_j sin(pi u_j.x)          to the ridge init by design)
  gauss_bump  exp(-3 ||x||^2 / d)                    -- radial, NOT ridge; the 3/d
              scaling keeps it O(1)-varying at all d    (E||x||^2 = d/3 on the cube)
  tensor4     prod_{i<=min(d,4)} sin(pi x_i)         -- multiplicative low-dim
                                                        structure embedded in d dims

Target directions u_j are fixed per (target, d) with a dedicated seed, so every arm
and every training seed fits the SAME function. Precision is measurable: rel L2 on a
held-out uniform sample.
"""
from __future__ import annotations

import math

import torch

TARGET_SEED = 1234
TARGET_ORDER = ["ridge_mix", "gauss_bump", "tensor4"]
PRETTY = {"ridge_mix": "ridge mix (3 sines)",
          "gauss_bump": "Gaussian bump (radial)",
          "tensor4": "tensor sine (4-dim core)"}


def _ridge_dirs(d: int, n: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(TARGET_SEED + d)
    U = torch.randn(n, d, generator=g)
    return U / U.norm(dim=1, keepdim=True)


def target_fn(name: str, d: int):
    if name == "ridge_mix":
        U = _ridge_dirs(d)
        return lambda x: torch.sin(math.pi * (x @ U.t())).mean(dim=1)
    if name == "gauss_bump":
        return lambda x: torch.exp(-3.0 * x.pow(2).sum(dim=1) / d)
    if name == "tensor4":
        k = min(d, 4)
        return lambda x: torch.sin(math.pi * x[:, :k]).prod(dim=1)
    raise KeyError(name)


def make_dataset(name: str, d: int, n_train: int, n_eval: int, seed: int):
    """Uniform samples on [-1,1]^d, y = f(x). Data varies with seed; f does not."""
    g = torch.Generator().manual_seed(seed)
    f = target_fn(name, d)
    x_train = torch.empty(n_train, d).uniform_(-1, 1, generator=g)
    x_eval = torch.empty(n_eval, d).uniform_(-1, 1, generator=g)
    return x_train, f(x_train), x_eval, f(x_eval)
