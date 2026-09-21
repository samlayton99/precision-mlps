"""Independent periodic tanh-difference laws; all norms use sample means.

Finite alias and spatial sums return approximations, not certified spectra.
The explicit remainder functions bound their corresponding infinite tails.
"""
from __future__ import annotations

import numpy as np


def multiplier(theta, lam):
    if lam <= 0:
        raise ValueError('The relative bandwidth must be positive')
    z = np.abs(np.asarray(theta, dtype=float))*np.pi/(2*lam)
    small = z < 1e-4
    safe = np.where(small, 1., z)
    regular = 2*safe*np.exp(-safe)/(-np.expm1(-2*safe))
    return np.where(small, 1-z*z/6+7*z**4/360, regular)


def amplitude(theta, lam):
    theta = np.asarray(theta, dtype=float)
    return 2*np.exp(-.5j*theta)*np.sinc(theta/(2*np.pi))*multiplier(theta, lam)


def continuous_spectrum(n, gamma, aliases=32):
    theta = 2*np.pi*np.arange(n)/n
    theta = (theta+np.pi) % (2*np.pi)-np.pi
    frequencies = theta[:, None]+2*np.pi*np.arange(-aliases, aliases+1)
    eigenvalues = np.sum(np.abs(amplitude(frequencies, 2*gamma/n))**2, axis=1)/n
    upper = multiplier(theta, 2*gamma/n)**2
    lower = np.sinc(theta/(2*np.pi))**2*upper
    return eigenvalues, lower, upper


def periodized_design(n, gamma, density, offset=0., images=8):
    """Direct spatial synthesis, independent of the alias formula."""
    m = density*n
    t = (np.arange(m)+offset)/density
    z = t[:, None]-np.arange(n)
    lam = 2*gamma/n
    values = np.zeros((m, n))
    for q in range(-images, images+1):
        u = lam*(z-q*n)
        values += np.tanh(u)-np.tanh(u-lam)
    return values/np.sqrt(m)


def sampled_design(n, gamma, density, offset=0., aliases=32):
    """Fourier synthesis on the shifted sampling grid with coherent aliases."""
    m = n*density
    ell = np.arange(-aliases*n, (aliases+1)*n)
    coefficients = amplitude(2*np.pi*ell/n, 2*gamma/n)/n
    grid = np.arange(m)
    folded = np.zeros(m, dtype=complex)
    np.add.at(folded, ell % m, coefficients*np.exp(2j*np.pi*ell*offset/m))
    one = np.fft.ifft(folded)*m
    return np.column_stack([one[(grid-j*density) % m].real for j in range(n)])/np.sqrt(m)


def sampled_spectrum(n, gamma, density, offset=0., aliases=32):
    r = np.arange(n)
    theta = 2*np.pi*r/n
    # Each residue modulo d collects amplitudes before taking squared moduli.
    packets = np.zeros((n, density), dtype=complex)
    for b in range(density):
        q = np.arange(-aliases, aliases+1)
        frequency = theta[:, None]+2*np.pi*(b+density*q)
        packets[:, b] = np.sum(amplitude(frequency, 2*gamma/n)
                               *np.exp(2j*np.pi*q*offset), axis=1)
    return np.sum(np.abs(packets)**2, axis=1)/n


def alias_tail(theta, lam, aliases, power=2):
    """Absolute amplitude (p=1) or energy (p=2) tail for |q|>aliases.

    Uses |A(u)| <= (4*pi/lam) exp(-a|u|)/(1-exp(-2a|u|)),
    a=pi/(2*lam), valid for |theta|<=pi. The geometric envelope
    deliberately keeps constants independent of sin(theta/2).
    """
    if power not in (1, 2) or aliases < 0 or np.any(np.abs(theta) > np.pi+1e-14):
        raise ValueError('Require power 1 or 2, nonnegative aliases, principal theta')
    a = np.pi/(2*lam)
    first = 2*np.pi*(aliases+1)-np.abs(np.asarray(theta))
    coefficient = 4*np.pi/lam/(-np.expm1(-2*a*first))
    return 2*coefficient**power*np.exp(-power*a*first)/(-np.expm1(-power*a*2*np.pi))


def spatial_tail(n, gamma, images):
    """Uniform bound for direct spatial images on 0<=t<N, 0<=j<N."""
    if images < 1:
        raise ValueError('At least one image is required')
    lam = 2*gamma/n
    return 4*np.exp(-2*lam*n*(images-1))/(-np.expm1(-2*lam*n))
