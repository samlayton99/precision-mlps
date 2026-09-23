"""Independent sine synthesis checks the periodic alias-block theorem."""

import numpy as np
import pytest

from experiments.expD36_frozen_gamma_probe.uniform_periodic import (
    aligned_square_spectrum, attenuation, square_tail_bound,
)


def direct_features(width, samples, gamma, period, cutoff, bias):
    x = np.arange(samples) * period / samples
    centers = np.arange(width) * period / width
    features = np.zeros((samples, width))
    for n in range(1, cutoff + 1, 2):
        z = np.pi**2 * n / (gamma * period)
        multiplier = z / np.sinh(z) if z < 700 else 0.0
        features += (4 / (np.pi * n) * multiplier
                     * np.sin(2 * np.pi * n * (x[:, None] - centers) / period))
    if bias:
        features = np.column_stack((features, np.ones(samples)))
    return features / np.sqrt(samples)


@pytest.mark.parametrize("width", [7, 8])
@pytest.mark.parametrize("q", [1, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_alias_eigenpairs_match_direct_finite_synthesis(width, q, bias):
    samples = q * width
    parameters = dict(width=width, samples=samples, gamma=7., period=3.,
                      cutoff=55, bias=bias)
    spectrum = aligned_square_spectrum(**parameters)
    j = direct_features(**parameters)
    k = j @ j.T
    modes = np.fft.ifft(spectrum.fourier_modes, axis=0, norm="ortho")
    reconstructed = (modes * spectrum.eigenvalues) @ modes.conj().T
    np.testing.assert_allclose(reconstructed, k, atol=2e-14)
    np.testing.assert_allclose(modes.conj().T @ modes,
                               np.eye(len(spectrum.eigenvalues)), atol=2e-14)
    target = np.random.default_rng(123).normal(size=samples)
    step = .4 / np.linalg.norm(k, 2)
    residual = target.copy()
    for _ in range(37):
        residual -= step * (k @ residual)
    measured = np.dot(residual, residual) / np.dot(target, target)
    assert spectrum.relative_residual_squared(target, step, 37) == pytest.approx(
        measured, abs=3e-14)


def test_square_parity_nullspace_and_bias():
    spectrum = aligned_square_spectrum(width=8, samples=8, gamma=5., cutoff=101)
    x = 2 * np.pi * np.arange(8) / 8
    target = np.cos(x) + np.cos(2 * x)
    weights, floor = spectrum.target_weights(target)
    assert floor == pytest.approx(.5, abs=1e-14)
    assert weights.sum() == pytest.approx(.5, abs=1e-14)
    assert len(spectrum.eigenvalues) == 5  # four odd modes, plus the bias
    _, constant_floor = spectrum.target_weights(np.ones(8))
    assert constant_floor == pytest.approx(0, abs=1e-14)
    no_bias = aligned_square_spectrum(width=8, samples=8, gamma=5., bias=False)
    assert no_bias.target_weights(np.ones(8))[1] == pytest.approx(1)


def test_absolute_tail_and_kernel_transfer_bound():
    parameters = dict(width=8, samples=32, gamma=12., period=3., bias=True)
    coarse = aligned_square_spectrum(**parameters, cutoff=11)
    fine = aligned_square_spectrum(**parameters, cutoff=1001)
    j_coarse = direct_features(**parameters, cutoff=11)
    j_fine = direct_features(**parameters, cutoff=1001)
    assert np.max(np.abs(j_fine - j_coarse)) * np.sqrt(32) <= coarse.feature_tail_bound
    assert np.linalg.norm(j_fine @ j_fine.T - j_coarse @ j_coarse.T, 2) <= coarse.kernel_error_bound
    assert np.sum(np.abs(fine.aliases - coarse.aliases)) <= coarse.feature_tail_bound
    assert square_tail_bound(12, 12., 3.) == square_tail_bound(11, 12., 3.)


def test_multiplier_cap_and_width_scaling():
    frequency = np.array([0., 1., 10., 100.])
    assert attenuation(0, 3.) == 1
    assert np.all(attenuation(frequency, 3.) <= attenuation(frequency, 8.))
    np.testing.assert_allclose(attenuation(2 * np.pi * .2 * 80 / 3, .7 * 80),
                               attenuation(2 * np.pi * .2 * 160 / 3, .7 * 160))
