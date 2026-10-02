"""Independent checks of the variational gradients and frozen-kernel training."""
import numpy as np
import pytest
from scipy import integrate
import torch

from experiments.expD41_activation_lens.local_activation import (
    LocalActivation, local_np, local_derivative_np, local_torch,
)
from experiments.expD41_activation_lens.local_design import DesignProblem, audit_scaled_gram


@pytest.fixture(scope="module")
def problem():
    return DesignProblem(theta_count=32, alias_half_width=4, fourier_order=16, gram_theta_count=65)


@pytest.fixture(scope="module")
def design(problem):
    return problem.design(problem.starts()["narrow_gaussian"])


def test_variational_objective_and_constraint_gradients(problem):
    a = problem.starts()["windowed_sinc"] + 0.003 * np.cos(np.arange(problem.dimension))
    a *= 2.0 / (problem.mass @ a)
    _, gradient = problem.objective_gradient(a)
    jacobian = problem.constraint_jacobian(a)
    step = 2e-6
    numerical = np.zeros_like(gradient)
    constraint_numerical = np.zeros_like(jacobian)
    for j in range(len(a)):
        direction = np.zeros_like(a)
        direction[j] = step
        numerical[j] = (problem.objective_gradient(a + direction)[0] - problem.objective_gradient(a - direction)[0]) / (2 * step)
        constraint_numerical[:, j] = (problem.constraint(a + direction) - problem.constraint(a - direction)) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-10)
    np.testing.assert_allclose(jacobian, constraint_numerical, rtol=3e-7, atol=2e-9)


def test_spline_constraints_and_exact_quadratic_forms(problem):
    assert problem.dimension == 16
    a = problem.starts()["windowed_sinc"]
    activation = LocalActivation(problem.design(a))
    np.testing.assert_array_equal(activation.kernel.c, activation.kernel.c[::-1])
    np.testing.assert_array_equal(activation.kernel.c[:2], [0.0, 0.0])
    np.testing.assert_array_equal(activation.kernel.c[-2:], [0.0, 0.0])
    assert abs(activation.kernel.integrate(-4, 4) - 2.0) < 2e-15
    np.testing.assert_allclose(activation.kernel.derivative()([-4.0, 4.0]), [0.0, 0.0], atol=1e-15)
    energy = integrate.quad(lambda x: float(activation.kernel(x))**2, -4, 4,
                            points=problem.breaks[1:-1], epsabs=1e-12)[0]
    roughness = integrate.quad(lambda x: float(activation.kernel.derivative()(x))**2, -4, 4,
                               points=problem.breaks[1:-1], epsabs=1e-12)[0]
    np.testing.assert_allclose(a @ problem.energy @ a, energy, rtol=2e-14)
    np.testing.assert_allclose(a @ problem.roughness @ a, roughness, rtol=2e-14)
    extrema = problem.gram_extrema(a)
    dense = np.einsum("a,tab,b->t", a, problem.gram_matrices(np.linspace(0, np.pi, 20001)), a)
    assert extrema["minimum"] <= dense.min() + 1e-12
    assert dense.min() - extrema["minimum"] < 1e-7
    scaled = audit_scaled_gram(problem.design(a), scale=1.0)
    np.testing.assert_allclose(scaled["kernel_energy"], energy, rtol=2e-14)
    np.testing.assert_allclose(scaled["minimum"], extrema["minimum"], atol=2e-6)


def test_local_activation_is_exact_integral_and_stable_near_zero(design):
    activation = LocalActivation(design)
    z = np.array([-6.0, -4.0, -3.0, -0.125, -1e-12, 0.0, 1e-12, 0.125, 3.0, 4.0, 6.0])
    expected = np.array([activation.kernel.integrate(-4.0, np.clip(value, -4.0, 4.0)) - 1.0 for value in z])
    np.testing.assert_allclose(local_np(z, design), expected, atol=4e-15, rtol=2e-14)
    np.testing.assert_array_equal(local_np(z, design), -local_np(-z, design))
    assert local_np(0.0, design) == 0.0
    np.testing.assert_allclose(local_np(1e-12, design) / 1e-12,
                               local_derivative_np(0.0, design), rtol=2e-14)
    assert local_np(-5.0, design) == -1.0
    assert local_np(5.0, design) == 1.0
    np.testing.assert_array_equal(local_derivative_np([-5.0, -4.0, 4.0, 5.0], design), [0.0] * 4)


def test_local_torch_gradient_at_knots_zero_and_support_edges(design):
    values = torch.tensor([-5.0, -4.0, -1.0, -1e-8, 0.0, 1e-8, 0.25, 1.0, 4.0, 5.0],
                          dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: local_torch(z, design), (values,),
                                    eps=1e-6, atol=3e-9, rtol=1e-6)
    result = local_torch(values, design)
    result.sum().backward()
    np.testing.assert_allclose(result.detach().numpy(), local_np(values.detach().numpy(), design), atol=0.0)
    np.testing.assert_allclose(values.grad.numpy(), local_derivative_np(values.detach().numpy(), design), atol=0.0)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3, 4)])
def test_shapes_and_dtype(shape, design):
    values = np.zeros(shape)
    assert local_np(values, design).shape == shape
    assert local_derivative_np(values, design).shape == shape
    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    output = local_torch(tensor, design)
    assert output.shape == tensor.shape
    assert output.dtype == torch.float64
    assert output.device.type == "cpu"
    output.sum().backward()
    assert tensor.grad.shape == tensor.shape


def test_weighted_network_parameter_gradients(design):
    activation = LocalActivation(design)
    x = np.array([-1.0, -0.3, 0.0, 0.4, 1.0])
    target = np.array([0.2, -0.1, 0.3, 0.8, -0.2])
    quadrature = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
    # Three input weights, three hidden biases, three readouts, one output bias.
    theta = np.array([0.7, -1.1, 2.4, 0.1, -0.2, 0.0, 0.4, -0.3, 0.5, -0.1])

    def objective(parameters):
        phi = activation.numpy(x[:, None] * parameters[:3] + parameters[3:6])
        residual = phi @ parameters[6:9] + parameters[9] - target
        return 0.5 * np.sum(quadrature * residual**2)

    parameters = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(x[:, None]) * parameters[:3] + parameters[3:6]
    residual = activation.torch(z) @ parameters[6:9] + parameters[9] - torch.tensor(target)
    loss = 0.5 * (torch.tensor(quadrature) * residual.square()).sum()
    (actual,) = torch.autograd.grad(loss, parameters)
    z_np = x[:, None] * theta[:3] + theta[3:6]
    phi = activation.numpy(z_np)
    weighted_residual = quadrature * (phi @ theta[6:9] + theta[9] - target)
    hidden = weighted_residual[:, None] * theta[6:9] * activation.derivative_numpy(z_np)
    expected = np.r_[(hidden * x[:, None]).sum(axis=0), hidden.sum(axis=0), phi.T @ weighted_residual,
                      weighted_residual.sum()]
    numerical = np.zeros_like(theta)
    for j in range(theta.size):
        offset = np.zeros_like(theta)
        offset[j] = 2e-6
        numerical[j] = (objective(theta + offset) - objective(theta - offset)) / 4e-6
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(actual.detach().numpy(), numerical, rtol=2e-7, atol=2e-10)
