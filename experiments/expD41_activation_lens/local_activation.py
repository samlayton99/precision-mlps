"""Evaluation of a frozen, compact spline kernel and its integrated activation."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, PPoly
import torch


def load_design(path) -> dict:
    """Read the frozen design JSON; numerical arrays remain ordinary lists."""
    return json.loads(Path(path).read_text())


class LocalActivation:
    """Cache the spline and its exact antiderivative for one frozen design.

    Design coefficients are fixed. Training derivatives act only on the input,
    so ordinary affine network parameters remain trainable.
    """

    def __init__(self, design: dict):
        self.design = design
        self.support = tuple(float(x) for x in design["support"])
        self.kernel = BSpline(np.asarray(design["knots"], dtype=float),
                              np.asarray(design["coefficients"], dtype=float),
                              int(design["degree"]), extrapolate=False)
        self.antiderivative = self.kernel.antiderivative()
        # Even kernels give odd activations. Recenter the antiderivative's
        # constant terms to integrate from zero, avoiding subtraction near zero.
        integrated = PPoly.from_spline(self.antiderivative)
        coefficients = integrated.c.copy()
        coefficients[-1] -= float(self.antiderivative(0.0))
        self.positive_integral = PPoly(coefficients, integrated.x, extrapolate=False)

    def numpy(self, z) -> np.ndarray:
        """Return the odd integrated kernel, with exact saturation at +/-1."""
        z = np.asarray(z, dtype=np.float64)
        magnitude = np.abs(z)
        result = np.sign(z) * self.positive_integral(np.minimum(magnitude, self.support[1]))
        result = np.where(magnitude >= self.support[1], np.sign(z), result)
        return np.asarray(result)

    def derivative_numpy(self, z) -> np.ndarray:
        """Evaluate K inside its support and zero outside, preserving shape."""
        z = np.asarray(z, dtype=np.float64)
        inside = (z >= self.support[0]) & (z <= self.support[1])
        result = np.where(inside, self.kernel(z), 0.0)
        return np.asarray(np.where(np.isnan(z), np.nan, result))

    def torch(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate on CPU using SciPy, with the exact kernel as backward."""
        return _IntegratedKernel.apply(z, self)


class _IntegratedKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, activation):
        ctx.save_for_backward(z)
        ctx.activation = activation
        result = activation.numpy(z.detach().cpu().numpy())
        return torch.as_tensor(result, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        values = ctx.activation.derivative_numpy(z.detach().cpu().numpy())
        derivative = torch.as_tensor(values, dtype=z.dtype, device=z.device)
        return grad_output * derivative, None


@lru_cache(maxsize=16)
def _cached_activation(degree, knots, coefficients, support):
    return LocalActivation({"degree": degree, "knots": knots,
                            "coefficients": coefficients, "support": support})


def _activation(design):
    if isinstance(design, LocalActivation):
        return design
    return _cached_activation(int(design["degree"]), tuple(design["knots"]),
                              tuple(design["coefficients"]), tuple(design["support"]))


def local_np(z, design) -> np.ndarray:
    return _activation(design).numpy(z)


def local_derivative_np(z, design) -> np.ndarray:
    return _activation(design).derivative_numpy(z)


def local_torch(z: torch.Tensor, design) -> torch.Tensor:
    return _activation(design).torch(z)
