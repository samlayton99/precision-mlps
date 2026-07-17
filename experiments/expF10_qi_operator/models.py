"""expF10 config models. A = coeff-space MLP + fixed QI decode; B/C = FNO."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import fno2d


class CoeffMLP(nn.Module):
    """c_a [b,D] -> MLP -> c_u [b,D] -> field [b, n_out] via fixed Phi_out."""
    def __init__(self, D, Phi_out, hidden=1024, n_layers=4):
        super().__init__()
        layers, d = [], D
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, D)]
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("Phi_out",
                             torch.tensor(np.asarray(Phi_out), dtype=torch.float32))

    def forward(self, c_a):
        c_u = self.mlp(c_a)                 # [b, D]
        return c_u @ self.Phi_out.t()       # [b, n_out]  (decoded field, flat)


def build_model(config, D=None, Phi_out=None, fno_kw=None):
    fno_kw = fno_kw or dict(width=32, modes=12, n_layers=4)
    if config == "A":
        return CoeffMLP(D, Phi_out), "coeff"
    if config in ("B", "C"):
        return fno2d.FNO2d(**fno_kw), "field"
    raise ValueError(config)
