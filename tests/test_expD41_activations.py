"""Value, kernel, and training-gradient checks for the activation comparison."""

import numpy as np
import pytest
from scipy import integrate, special
import torch

from experiments.expD41_activation_lens.activations import (
    ACTIVATIONS,
    activation_np,
    activation_torch,
    derivative_np,
    kernel_fourier,
)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_numpy_torch_values_and_analytic_backward(name):
    z = np.array([-1000.0, -20.0, -3.2, -0.2, -1e-8, 0.0,
                  1e-8, 0.2, 3.2, 20.0, 1000.0, 0.8]).reshape(2, 2, 3)
    tensor = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    actual = activation_torch(name, tensor)
    assert actual.shape == tensor.shape
    assert actual.dtype == tensor.dtype
    np.testing.assert_allclose(actual.detach().numpy(), activation_np(name, z), rtol=3e-14, atol=1e-16)
    actual.sum().backward()
    np.testing.assert_allclose(tensor.grad.numpy(), derivative_np(name, z), rtol=3e-14, atol=0.0)
    assert np.isfinite(actual.detach().numpy()).all()


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_autograd_gradcheck_at_zero_small_and_large_inputs(name):
    z = torch.tensor([-30.0, -3.0, -0.1, -1e-7, 0.0, 1e-7, 0.1, 3.0, 30.0],
                     dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: activation_torch(name, values),
                                    (z,), eps=1e-6, atol=2e-9, rtol=2e-6)


def test_notch_near_zero_and_closed_form():
    small = np.array([-1e-12, -1e-8, 1e-8, 1e-12])
    # The leading cubic term detects a cancellation-damaged forward function.
    np.testing.assert_allclose(activation_np("notch", small),
                               np.sqrt(2.0 / np.pi) * small**3 / 3.0, rtol=2e-14)
    z = np.array([-4.0, -1.0, -0.3, 0.3, 1.0, 4.0])
    expected = special.erf(z / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * z * np.exp(-z**2 / 2.0)
    np.testing.assert_allclose(activation_np("notch", z), expected, rtol=1e-14, atol=1e-16)
    np.testing.assert_array_equal(activation_np("notch", [-100.0, 0.0, 100.0]), [-1.0, 0.0, 1.0])
    assert derivative_np("notch", 0.0) == 0.0
    assert derivative_np("sinc", 0.0) == 2.0
    assert derivative_np("tanh", 0.0) == 1.0
    assert derivative_np("tanh", 20.0) > 0.0


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_weighted_network_parameter_gradients(name):
    """Check weights, centers, scales, and intercept against two references."""
    x = np.array([-1.0, -0.4, 0.0, 0.25, 0.9])
    target = np.array([0.2, -0.5, 0.1, 0.6, -0.2])
    weights = np.array([0.07, 0.21, 0.31, 0.19, 0.22])
    # theta = readout a (3), centers c (3), log-scales ell (3), intercept b.
    theta = np.array([0.4, -0.7, 0.2, -0.4, 0.0, 0.35,
                      np.log(0.8), np.log(1.7), np.log(2.3), -0.15])

    def numpy_loss(parameters):
        a, c, scale, b = parameters[:3], parameters[3:6], np.exp(parameters[6:9]), parameters[9]
        z = (x[:, None] - c) * scale
        residual = activation_np(name, z) @ a + b - target
        return 0.5 * np.sum(weights * residual**2)

    parameters = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    scale = parameters[6:9].exp()
    z_torch = (torch.tensor(x[:, None]) - parameters[3:6]) * scale
    residual = activation_torch(name, z_torch) @ parameters[:3] + parameters[9] - torch.tensor(target)
    loss = 0.5 * (torch.tensor(weights) * residual.square()).sum()
    (gradient,) = torch.autograd.grad(loss, parameters)

    a, c, scale = theta[:3], theta[3:6], np.exp(theta[6:9])
    z = (x[:, None] - c) * scale
    features = activation_np(name, z)
    weighted_residual = weights * (features @ a + theta[9] - target)
    hidden_gradient = weighted_residual[:, None] * a * derivative_np(name, z)
    analytic = np.concatenate([
        features.T @ weighted_residual,
        (-hidden_gradient * scale).sum(axis=0),
        (hidden_gradient * z).sum(axis=0),
        [weighted_residual.sum()],
    ])
    numerical = np.empty_like(theta)
    step = 2e-6
    for j in range(len(theta)):
        direction = np.zeros_like(theta)
        direction[j] = step
        numerical[j] = (numpy_loss(theta + direction) - numpy_loss(theta - direction)) / (2.0 * step)
    np.testing.assert_allclose(gradient.detach().numpy(), analytic, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(gradient.detach().numpy(), numerical, rtol=3e-7, atol=2e-10)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_scalar_and_empty_shapes(name):
    for shape in [(), (0,), (2, 0, 3)]:
        z = np.zeros(shape)
        assert activation_np(name, z).shape == shape
        assert derivative_np(name, z).shape == shape
        assert kernel_fourier(name, z).shape == shape
        tensor = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        result = activation_torch(name, tensor)
        assert result.shape == tensor.shape
        result.sum().backward()
        assert tensor.grad.shape == tensor.shape


@pytest.mark.parametrize("name", ["tanh", "notch"])
def test_fourier_formula_against_kernel_integral(name):
    for omega in [0.0, 0.35, 1.0, 1.8, 4.0]:
        numerical, _ = integrate.quad(
            lambda z: 2.0 * float(derivative_np(name, z)) * np.cos(omega * z),
            0.0, np.inf, epsabs=2e-11, epsrel=2e-11,
        )
        np.testing.assert_allclose(kernel_fourier(name, omega), numerical, atol=3e-11, rtol=2e-10)


def test_fourier_notch_zero_sinc_edges_and_stable_tanh_limit():
    for name in ACTIVATIONS:
        assert kernel_fourier(name, 0.0) == 2.0
    np.testing.assert_array_equal(kernel_fourier("notch", [-1.0, 1.0]), [0.0, 0.0])
    assert kernel_fourier("notch", 1.1) < 0.0
    assert kernel_fourier("notch", 0.9) > 0.0
    np.testing.assert_array_equal(kernel_fourier("sinc", [-4.0, -np.pi, 0.0, np.pi, 4.0]),
                                  [0.0, 1.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(kernel_fourier("tanh", [-1e-16, 1e-16]), [2.0, 2.0], rtol=2e-16)
    for name in ["tanh", "notch"]:
        assert np.isfinite(kernel_fourier(name, [-1e308, -1000.0, 1000.0, 1e308])).all()


@pytest.mark.parametrize("function", [activation_np, derivative_np, kernel_fourier])
def test_unknown_numpy_activation_raises(function):
    with pytest.raises(ValueError, match="Unknown activation"):
        function("unknown", 0.0)


def test_unknown_torch_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        activation_torch("unknown", torch.tensor(0.0))
