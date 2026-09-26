"""Extra targets from the eight-target comparison, preserving the original suite."""
from functools import lru_cache

import numpy as np

from experiments.expC09_bandwidth_figures.targets import (
    TITLES as ORIGINAL_TITLES, target_values as original_values,
    frequency_scale as original_frequency_scale,
)

EXTRA_NAMES = ("chirp", "runge100", "smooth_bump")
TITLES = dict(ORIGINAL_TITLES, **{
    "chirp": r"Chirp: $f(x)=\sin(8\pi(x+1)^2)$",
    "runge100": r"Runge 100: $f(x)=1/(1+100x^2)$",
    "smooth_bump": r"Smooth compact bump: $f(x)=e^{-1/(1-4x^2)}$ for $|x|<1/2$, zero otherwise",
})


def target_values(x, name):
    x = np.asarray(x)
    if name == "chirp":
        return np.sin(8*np.pi*(x+1)**2)
    if name == "runge100":
        return 1/(1+100*x*x)
    if name == "smooth_bump":
        u = x/.5
        result = np.zeros_like(u, dtype=float)
        inside = np.abs(u) < 1
        result[inside] = np.exp(-1/(1-u[inside]**2))
        return result
    return original_values(x, name)


def bump_frequency_scale(length=128., points=131072):
    """Quadrature of integral w*|fhat| / integral |fhat| on [0,1000].

    Zero padding resolves frequency with spacing 2*pi/length. The spatial
    step is 1/1024; truncating w at 1000 excludes the numerical noise tail.
    Doubling the padding from length 64 changes the result by under 2e-5.
    """
    x = np.arange(points)*length/points-length/2
    omega = 2*np.pi*np.fft.rfftfreq(points, d=length/points)
    amplitude = np.abs(np.fft.rfft(target_values(x, "smooth_bump")))
    keep = omega <= 1000
    return float(np.trapezoid(omega[keep]*amplitude[keep], omega[keep])
                 / np.trapezoid(amplitude[keep], omega[keep]))


@lru_cache(maxsize=None)
def frequency_scale(name):
    if name == "runge100":
        return 10.
    if name == "chirp":
        # Spatial mean of |phase'(x)| on [-1,1], not a Fourier moment.
        return 16*np.pi
    if name == "smooth_bump":
        return bump_frequency_scale()
    return original_frequency_scale(name)
