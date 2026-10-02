"""Fixed saturating activations and Fourier transforms of their derivatives.

The transform convention is integral k(z) exp(-i omega z) dz. Each derivative
has total mass 2; the sinc integral and its transform use improper limits.
"""

from __future__ import annotations

import numpy as np
from scipy import special
import torch


ACTIVATIONS = ("tanh", "notch", "sinc")
_SQRT_TWO_OVER_PI = np.sqrt(2.0 / np.pi)


def _validate(name: str) -> None:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; expected one of {ACTIVATIONS}")


def activation_np(name: str, z) -> np.ndarray:
    """Evaluate an activation in float64, preserving the input shape."""
    _validate(name)
    z = np.asarray(z, dtype=np.float64)
    if name == "tanh":
        result = np.tanh(z)
    elif name == "notch":
        # Equivalent to erf(z/sqrt(2)) - sqrt(2/pi)*z*exp(-z*z/2),
        # without the subtraction of two nearly equal terms near zero.
        with np.errstate(over="ignore"):
            result = np.sign(z) * special.gammainc(1.5, 0.5 * z**2)
    else:
        result = (2.0 / np.pi) * special.sici(np.pi * z)[0]
    return np.asarray(result)


def derivative_np(name: str, z) -> np.ndarray:
    """Evaluate the derivative kernel with its removable limits filled in."""
    _validate(name)
    z = np.asarray(z, dtype=np.float64)
    if name == "tanh":
        decay = np.exp(-np.abs(z)) ** 2
        result = 4.0 * decay / (1.0 + decay) ** 2
    elif name == "notch":
        # Beyond 40 the kernel has already underflowed in float64. Clipping
        # prevents inf*0 at extreme inputs without changing representable values.
        bounded = np.clip(z, -40.0, 40.0)
        square = bounded**2
        result = _SQRT_TWO_OVER_PI * square * np.exp(-0.5 * square)
    else:
        result = 2.0 * np.sinc(z)
    return np.asarray(result)


class _Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.tanh(z)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        decay = torch.exp(-torch.abs(z)).square()
        return grad_output * (4.0 * decay / (1.0 + decay).square())


class _Notch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.sign(z) * torch.special.gammainc(
            torch.full_like(z, 1.5), 0.5 * z.square()
        )

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        square = z.clamp(-40.0, 40.0).square()
        return grad_output * (_SQRT_TWO_OVER_PI * square * torch.exp(-0.5 * square))


class _SincIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        # SciPy supplies Si; PyTorch supplies the exact differentiable backward.
        values = activation_np("sinc", z.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        return grad_output * (2.0 * torch.sinc(z))


def activation_torch(name: str, z: torch.Tensor) -> torch.Tensor:
    """Evaluate an activation with an analytic first derivative for training.

    The sinc forward uses SciPy on the CPU and returns the original dtype/device.
    Tanh also uses a custom backward to retain its small, nonzero tail derivative.
    """
    _validate(name)
    functions = {"tanh": _Tanh, "notch": _Notch, "sinc": _SincIntegral}
    return functions[name].apply(z)


def kernel_fourier(name: str, omega) -> np.ndarray:
    """Transform the derivative kernel; omega is angular frequency, not cycles."""
    _validate(name)
    omega = np.asarray(omega, dtype=np.float64)
    if name == "tanh":
        with np.errstate(over="ignore", invalid="ignore"):
            a = (np.pi / 2.0) * np.abs(omega)
            result = np.divide(
                4.0 * (a * np.exp(-a)),
                -np.expm1(-2.0 * a),
                out=np.full_like(a, 2.0),
                where=a != 0.0,
            )
            result = np.where(np.isinf(a), 0.0, result)
    elif name == "notch":
        with np.errstate(over="ignore", invalid="ignore"):
            square = omega**2
            result = 2.0 * ((1.0 - square) * np.exp(-0.5 * square))
            result = np.where(np.isinf(square), 0.0, result)
    else:
        magnitude = np.abs(omega)
        result = np.where(magnitude < np.pi, 2.0, np.where(magnitude == np.pi, 1.0, 0.0))
        result = np.where(np.isnan(omega), np.nan, result)
    return np.asarray(result)
