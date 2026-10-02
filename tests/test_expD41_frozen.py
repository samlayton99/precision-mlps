"""Readout-only updates must match the full network with frozen geometry."""
import numpy as np
import torch

from experiments.expD41_activation_lens import run
from experiments.expD41_activation_lens.frozen import FrozenReadout, fixed_features, frozen_config


def test_cached_features_match_frozen_network_updates():
    x = torch.linspace(-1, 1, 31, dtype=torch.float64)
    y = torch.sin(2 * torch.pi * x)
    for activation in ("tanh", "notch", "local", "sinc"):
        w = np.full(9, 3.0)
        b = -w * np.linspace(-1.5, 1.5, 9)
        full = run.Network(activation, (w, b, np.zeros(9), 0.0))
        full.w.requires_grad_(False)
        full.b.requires_grad_(False)
        cached = FrozenReadout(np.zeros(9), 0.0)
        phi = fixed_features(activation, w, b, x.numpy())
        before = phi.clone()
        opt1 = torch.optim.Adam([full.v, full.bias], lr=0.002)
        opt2 = torch.optim.Adam(cached.parameters(), lr=0.002)
        for _ in range(3):
            run.train_step(full, opt1, x, y, run.mse)
            run.train_step(cached, opt2, phi, y, run.mse)
        torch.testing.assert_close(cached.v, full.v, atol=2e-15, rtol=2e-14)
        torch.testing.assert_close(cached.bias, full.bias, atol=2e-15, rtol=2e-14)
        torch.testing.assert_close(phi, before, atol=0, rtol=0)
        np.testing.assert_array_equal(full.w.detach().numpy(), w)
        np.testing.assert_array_equal(full.b.detach().numpy(), b)
        assert list(dict(cached.named_parameters())) == ["v", "bias"]


def test_frozen_protocol_uses_zero_readout():
    cfg = frozen_config()
    assert "zero_readout" in cfg["frozen_geometry"]["initializations"]
    assert cfg["frozen_geometry"]["trainable_parameters"] == ["v", "output_bias"]
    m = FrozenReadout(np.zeros(10), 0.0)
    assert torch.count_nonzero(m.v) == 0 and m.bias == 0
    assert torch.count_nonzero(m(torch.randn(17, 10, dtype=torch.float64))) == 0
