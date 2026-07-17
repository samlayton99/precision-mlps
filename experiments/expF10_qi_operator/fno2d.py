"""Minimal 2D Fourier Neural Operator (torch). Standard FNO2d: lift -> K
spectral-conv+pointwise layers -> project. Verified forward/backward at 64^2."""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        # two corners of the rfft2 spectrum (low + high vertical frequencies)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, 2))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, 2))

    @staticmethod
    def _cmul(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(w))

    def forward(self, x):
        b, c, h, wd = x.shape
        m = self.modes
        xft = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, wd // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = self._cmul(xft[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = self._cmul(xft[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(h, wd))


class FNO2d(nn.Module):
    def __init__(self, width=32, modes=12, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(1, width, 1)
        self.specs = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, x):
        x = self.lift(x)
        for sp, w in zip(self.specs, self.ws):
            x = torch.nn.functional.gelu(sp(x) + w(x))
        return self.proj(x)
