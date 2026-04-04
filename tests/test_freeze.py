"""Tests for parameter freezing via requires_grad."""

import torch

from src.config.schema import ModelConfig
from src.models.mlp import QIMlp
from src.models.freeze import (
    freeze_gamma, freeze_centers, freeze_readout, freeze_inner_layer,
    unfreeze_all, get_trainable_param_count,
)


def _make_model(width=8, layer_type="gamma_linear"):
    return QIMlp(ModelConfig(width=width, layer_type=layer_type))


def test_freeze_gamma_leaves_centers_trainable():
    m = _make_model()
    freeze_gamma(m)
    assert not m.inner_layer.gamma.requires_grad
    assert m.inner_layer.centers.requires_grad
    assert m.readout.weight.requires_grad
    assert m.readout.bias.requires_grad


def test_freeze_centers():
    m = _make_model()
    freeze_centers(m)
    assert m.inner_layer.gamma.requires_grad
    assert not m.inner_layer.centers.requires_grad


def test_freeze_readout():
    m = _make_model()
    freeze_readout(m)
    assert m.inner_layer.gamma.requires_grad
    assert not m.readout.weight.requires_grad
    assert not m.readout.bias.requires_grad


def test_freeze_inner_layer_freezes_both():
    m = _make_model()
    freeze_inner_layer(m)
    assert not m.inner_layer.gamma.requires_grad
    assert not m.inner_layer.centers.requires_grad
    assert m.readout.weight.requires_grad


def test_freeze_gamma_exp():
    m = _make_model(layer_type="gamma_exp")
    freeze_gamma(m)
    assert not m.inner_layer.log_gamma.requires_grad
    assert m.inner_layer.centers.requires_grad


def test_param_count():
    m = _make_model(width=8)
    full = get_trainable_param_count(m)
    # 8 (gamma) + 8 (centers) + 8 (readout w) + 1 (bias) = 25
    assert full == 25
    freeze_gamma(m)
    assert get_trainable_param_count(m) == 25 - 8
    unfreeze_all(m)
    assert get_trainable_param_count(m) == 25


def test_frozen_params_have_no_gradient():
    m = _make_model(width=4)
    freeze_gamma(m)
    x = torch.linspace(-1, 1, 8).reshape(-1, 1)
    y = torch.sin(x)
    loss = ((m(x) - y) ** 2).mean()
    loss.backward()
    assert m.inner_layer.gamma.grad is None
    assert m.inner_layer.centers.grad is not None
    assert m.readout.weight.grad is not None


def test_forward_unchanged_by_freezing():
    m = _make_model(width=8)
    x = torch.linspace(-1, 1, 10).reshape(-1, 1)
    out_before = m(x).detach().clone()
    freeze_gamma(m)
    out_after = m(x).detach()
    assert torch.allclose(out_before, out_after)


def test_freeze_then_unfreeze_identity():
    m = _make_model(width=6)
    freeze_inner_layer(m)
    unfreeze_all(m)
    for p in m.parameters():
        assert p.requires_grad
