"""Tests for QIMlp and layer types."""

import math

import pytest
import torch

from src.config.schema import ModelConfig
from src.models.layers import GammaLinear, GammaExpLinear, StandardLinear, get_layer
from src.models.mlp import QIMlp


def test_qimlp_forward_shape():
    cfg = ModelConfig(width=16, layer_type="gamma_linear")
    m = QIMlp(cfg)
    x = torch.linspace(-1, 1, 32).reshape(-1, 1)
    out = m(x)
    assert out.shape == (32, 1)
    assert out.dtype == torch.float64


def test_features_shape():
    cfg = ModelConfig(width=24, layer_type="gamma_linear")
    m = QIMlp(cfg)
    x = torch.linspace(-1, 1, 10).reshape(-1, 1)
    phi = m.features(x)
    assert phi.shape == (10, 24)


def test_gamma_linear_math():
    gamma = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    centers = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    layer = GammaLinear(2, gamma_init=gamma, center_init=centers)
    x = torch.tensor([[1.0]], dtype=torch.float64)
    out = layer(x)
    # gamma * (x - centers) = [2*(1-0.5), 3*(1-(-0.5))] = [1.0, 4.5]
    expected = torch.tensor([[1.0, 4.5]], dtype=torch.float64)
    assert torch.allclose(out, expected)


def test_gamma_exp_linear_math():
    log_gamma = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float64)
    centers = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    layer = GammaExpLinear(2, h=2.0, log_gamma_init=log_gamma, center_init=centers)
    # Effective gamma = exp(log_gamma)/h = [1.0, 2.0]
    x = torch.tensor([[3.0]], dtype=torch.float64)
    out = layer(x)
    expected = torch.tensor([[3.0, 6.0]], dtype=torch.float64)
    assert torch.allclose(out, expected)
    assert torch.allclose(layer.get_gamma(), torch.tensor([[1.0, 2.0]], dtype=torch.float64))


def test_standard_linear_shape():
    layer = StandardLinear(8)
    x = torch.randn(5, 1, dtype=torch.float64)
    out = layer(x)
    assert out.shape == (5, 8)


def test_get_layer_factory():
    assert isinstance(get_layer("gamma_linear", 4), GammaLinear)
    assert isinstance(get_layer("gamma_exp", 4, h=1.0), GammaExpLinear)
    assert isinstance(get_layer("standard", 4), StandardLinear)
    with pytest.raises(ValueError):
        get_layer("bogus", 4)


def test_accessors_gamma_linear():
    cfg = ModelConfig(width=8, layer_type="gamma_linear")
    m = QIMlp(cfg)
    with torch.no_grad():
        m.inner_layer.gamma.data.fill_(3.0)
        m.inner_layer.centers.data.copy_(torch.linspace(-1, 1, 8).reshape(1, 8))
    g = m.get_gamma()
    c = m.get_centers()
    assert g.shape == (8,)
    assert c.shape == (8,)
    assert torch.allclose(g, torch.full((8,), 3.0, dtype=torch.float64))


def test_accessors_gamma_exp():
    cfg = ModelConfig(width=4, layer_type="gamma_exp")
    m = QIMlp(cfg, h=0.5)
    with torch.no_grad():
        m.inner_layer.log_gamma.data.fill_(0.0)  # exp(0)=1, /0.5 = 2
    g = m.get_gamma()
    assert torch.allclose(g, torch.full((4,), 2.0, dtype=torch.float64))


def test_lambda_from_gamma():
    cfg = ModelConfig(width=4, layer_type="gamma_linear")
    m = QIMlp(cfg)
    with torch.no_grad():
        m.inner_layer.gamma.data.fill_(10.0)
    lam = m.get_lambda(h=0.1)
    assert torch.allclose(lam, torch.ones(4, dtype=torch.float64))


def test_all_float64():
    cfg = ModelConfig(width=16, layer_type="gamma_linear")
    m = QIMlp(cfg)
    for p in m.parameters():
        assert p.dtype == torch.float64
