"""D24 targets with regular sine changed to sin(4*pi*x), all scored on [-1, 1]."""
from math import erf, pi, sqrt
import numpy as np

MODES = (2, 6, 14)
AMPLITUDES = (1.0, 0.5, 0.25)
SIGMA = 0.4
NAMES = ("runge", "mixed_sine", "sine", "gaussian_envelope")
TITLES = {
    "runge": r"Runge: $f(x)=1/(1+25x^2)$",
    "mixed_sine": r"Mixed sine: $f(x)=\sin(2\pi x)+\frac{1}{2}\sin(6\pi x)+\frac{1}{4}\sin(14\pi x)$",
    "sine": r"Sine: $f(x)=\sin(4\pi x)$",
    "gaussian_envelope": r"Gaussian envelope: $f(x)=e^{-x^2/(2\cdot0.4^2)}[\sin(2\pi x)+\frac{1}{2}\sin(6\pi x)+\frac{1}{4}\sin(14\pi x)]$",
}


def target_values(x, name):
    x = np.asarray(x)
    if name == "runge":
        return 1.0 / (1.0 + 25.0*x**2)
    if name == "sine":
        return np.sin(4*np.pi*x)
    mixture = sum(a*np.sin(k*np.pi*x) for a, k in zip(AMPLITUDES, MODES))
    if name == "mixed_sine":
        return mixture
    if name == "gaussian_envelope":
        return np.exp(-.5*(x/SIGMA)**2)*mixture
    raise ValueError(name)


def frequency_scale(name):
    """Mean absolute angular frequency weighted by Fourier amplitude."""
    if name == "runge":
        return 5.0
    if name == "sine":
        return 4*pi
    numerator = sum(a*k*pi for a, k in zip(AMPLITUDES, MODES))
    if name == "mixed_sine":
        return numerator / sum(AMPLITUDES)
    if name == "gaussian_envelope":
        # For positive omega, each shifted-Gaussian difference is positive.
        # Integrating its mass gives erf(sigma*k*pi/sqrt(2)); its first moment
        # gives k*pi. Thus this ratio includes cancellation near omega=0.
        return numerator / sum(a*erf(SIGMA*k*pi/sqrt(2)) for a, k in zip(AMPLITUDES, MODES))
    raise ValueError(name)
