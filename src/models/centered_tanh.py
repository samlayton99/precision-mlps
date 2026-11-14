"""Ordinary tanh MLP with QUILL's fixed affine input centering."""
import torch
from torch import nn


class CenteredTanhMLP(nn.Module):
    def __init__(self, width, *, eps=1e-4, seed=0, dtype=torch.float64):
        super().__init__()
        self.eps = eps
        self.radius = (1 - eps) / 2
        # Preserve the caller's RNG; retain nn.Linear's default initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(1, width, dtype=dtype), nn.Tanh(),
                                     nn.Linear(width, 1, dtype=dtype))

    def forward(self, x):
        return self.net((x - self.eps) / self.radius - 1)
