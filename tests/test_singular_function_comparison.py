import numpy as np
import pytest
import torch
from experiments.expK05_singular_function_comparison.construction import (
    TARGETS, TargetExtension, derivative, build, metrics,
)
from experiments.expK04_npu_elementary_suite.problem import target


@pytest.mark.parametrize('name', TARGETS)
def test_derivative_and_partition(name):
    x = torch.tensor(np.geomspace(.0001, 1, 2001), requires_grad=True)
    exact = torch.autograd.grad(target(name, x).sum(), x)[0].detach().numpy()
    np.testing.assert_allclose(derivative(name, x.detach().numpy()), exact, rtol=1e-15)
    e = TargetExtension(name=name)
    xx = x.detach().numpy()
    total = sum(s**-1.5*e.block_derivative(xx/(2*s)-1, s)/2 for s in e.scales)
    np.testing.assert_allclose(total, exact, rtol=3e-15)
    np.testing.assert_array_equal(e.value(xx), target(name, xx))
    assert np.isfinite(e.value(np.array([-10., 0., 10.]))).all()
    assert np.all(e.derivative(np.array([-10., 0., 10.])) == 0)


@pytest.mark.parametrize('name', TARGETS)
def test_general_physical_metric(name):
    class Perturbed:
        eps = .0001
        def predict(self, x): return 1.01*target(name, x)
    assert abs(metrics(Perturbed(), name)-.01) < 1e-15


@pytest.mark.parametrize('name', TARGETS)
def test_constructed_network_precision(name, tmp_path):
    net = build(TargetExtension(name=name), 128, tmp_path)
    assert net.width == 2895
    assert metrics(net, name, 128) < 2e-12
