"""Explicit alias-block spectrum of smoothed periodic square-wave features.

J[l,j] = phi_gamma(l*period/m - j*period/W)/sqrt(m), with an optional
ones/sqrt(m) bias column. Coefficient truncation has an analytic absolute
tail bound; floating-point arithmetic itself is not interval certified.
"""

from dataclasses import dataclass

import numpy as np


def attenuation(frequency, gamma):
    """Logistic smoothing multiplier, with angular frequencies."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    z = np.abs(np.asarray(frequency, dtype=float)) * np.pi / (2 * gamma)
    denominator = -np.expm1(-2 * z)
    return np.divide(2 * z * np.exp(-z), denominator,
                     out=np.ones_like(z), where=z != 0)


def square_coefficients(indices, gamma, period):
    """Coefficients of smoothed sign(sin(2*pi*x/period))."""
    if period <= 0:
        raise ValueError("period must be positive")
    indices = np.asarray(indices, dtype=int)
    result = np.zeros(indices.shape, dtype=complex)
    odd = indices % 2 != 0
    result[odd] = (-2j / (np.pi * indices[odd])
                   * attenuation(2 * np.pi * indices[odd] / period, gamma))
    return result


def square_tail_bound(cutoff, gamma, period):
    """Upper bound on sum_{|n|>cutoff} |a_n| (all omitted odd modes)."""
    if cutoff < 0 or gamma <= 0 or period <= 0:
        raise ValueError("require cutoff >= 0 and positive gamma, period")
    first = cutoff + 1 + (cutoff % 2)
    a = np.pi**2 / (gamma * period)
    return (8 * a / np.pi * np.exp(-a * first)
            / (-np.expm1(-2 * a) * -np.expm1(-2 * a * first)))


@dataclass
class PeriodicSpectrum:
    eigenvalues: np.ndarray
    fourier_modes: np.ndarray
    aliases: np.ndarray
    feature_tail_bound: float
    kernel_error_bound: float

    def target_weights(self, target):
        """Weights and perpendicular floor for this coefficient truncation."""
        target = np.asarray(target)
        if target.shape != self.aliases.shape or np.linalg.norm(target) == 0:
            raise ValueError("target must be nonzero and match the sample grid")
        target_hat = np.fft.fft(target, norm="ortho")
        weights = (np.abs(self.fourier_modes.conj().T @ target_hat)**2
                   / np.vdot(target, target).real)
        floor = max(0.0, 1.0 - float(weights.sum()))
        return weights, floor

    def relative_residual_squared(self, target, step, updates):
        """Exact GD law of the truncated feature model, initialized at zero."""
        rates = step * self.eigenvalues
        if step < 0 or np.any(rates > 1) or updates < 0:
            raise ValueError("require updates >= 0 and rates in [0, 1]")
        weights, floor = self.target_weights(target)
        return float(floor + np.dot(weights, (1 - rates)**(2 * updates)))


def aligned_square_spectrum(*, width, samples, gamma, period=2.0,
                            cutoff=1024, bias=True):
    """Full alias blocks for m=qW aligned uniform grids, raw readout scaling.

    Eigenpairs are exact for the finite Fourier sum |n| <= cutoff, up to
    roundoff. kernel_error_bound controls its difference from the infinite
    smoothed-square kernel in spectral norm, including the unchanged bias.
    """
    if width < 1 or samples < width or samples % width:
        raise ValueError("samples must be a positive integer multiple of width")
    if cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    n = np.arange(1, cutoff + 1, 2)
    coefficients = square_coefficients(n, gamma, period)
    aliases = np.zeros(samples, dtype=complex)
    np.add.at(aliases, n % samples, coefficients)
    np.add.at(aliases, (-n) % samples, coefficients.conj())
    # Preserve exact odd symmetry at the self-conjugate DC/Nyquist bins.
    aliases = (aliases - aliases[(-np.arange(samples)) % samples]) / 2
    values, modes = [], []
    for residue in range(width):
        indices = np.arange(residue, samples, width)
        length = np.linalg.norm(aliases[indices])
        if length > 0:
            mode = np.zeros(samples, dtype=complex)
            mode[indices] = aliases[indices] / length
            values.append(width * length**2)
            modes.append(mode)
    if bias:
        mode = np.zeros(samples, dtype=complex)
        mode[0] = 1
        values.append(1.0)
        modes.append(mode)
    eigenvalues = np.asarray(values)
    fourier_modes = (np.column_stack(modes) if modes else
                     np.zeros((samples, 0), dtype=complex))
    tail = float(square_tail_bound(cutoff, gamma, period))
    feature_error = np.sqrt(width) * tail
    norm = np.sqrt(eigenvalues.max(initial=0))
    kernel_error = 2 * norm * feature_error + feature_error**2
    return PeriodicSpectrum(eigenvalues, fourier_modes, aliases,
                            tail, float(kernel_error))
