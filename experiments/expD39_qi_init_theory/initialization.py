"""Versioned ridge-bank hypotheses; D38's historical initializer is untouched.

These are geometry-informed starting points for ordinary dense MLP training,
not complete QI constructions or guarantees about noisy held-out data.
"""
from __future__ import annotations

import math
import torch
from torch import nn


VARIANTS = {
    "spacing": dict(balanced=False),
    "balanced": dict(),
    "centered": dict(centered=True),
    "collar": dict(centered=True, collar=1.25),
    "common": dict(centered=True, common=True),
    "lambda05": dict(centered=True, lam=0.5),
    "lambda10": dict(centered=True, lam=1.0),
    "directions64": dict(centered=True, n_dirs=64),
}


@torch.no_grad()
def initialize_layer(linear: nn.Linear, x: torch.Tensor, *, generator,
                     balanced=True, centered=False, collar=1., common=False,
                     lam=.25, n_dirs=None):
    width, dim = linear.weight.shape
    p = int(math.sqrt(width))
    if n_dirs is None:
        n_dirs = math.ceil(width / p)
    if not 1 <= n_dirs <= width // 2:
        raise ValueError("Each bank needs at least two centers")
    if balanced:
        q, r = divmod(width, n_dirs)
        sizes = [q + (j < r) for j in range(n_dirs)]
    else:
        sizes = [p] * (width // p) + ([width % p] if width % p else [])
        if min(sizes) < 2:
            raise ValueError("Unbalanced control has a singleton bank")
    xs = x
    if len(xs) > 4096:
        xs = xs[torch.randperm(len(xs), generator=generator)[:4096]]
    banks = []
    for size in sizes:
        u = torch.randn(dim, dtype=x.dtype, device=x.device, generator=generator)
        u /= u.norm()
        t = xs @ u
        if centered:
            lo, hi = torch.quantile(t, t.new_tensor([.0005, .9995])).tolist()
            mid = .5 * (lo + hi)
            half = max(.5 * (hi - lo), 1e-6)
        else:
            mid = 0.
            half = max(t.abs().quantile(.999).item(), 1e-6)
        half *= collar
        h = 2 * half / (size - 1)
        banks.append((size, u, mid, h, t))
    # A shared gamma needs a shared spacing. Choose the largest required h so
    # every bank still covers its estimated projection interval.
    shared_h = max(b[3] for b in banks) if common else None
    records = []
    start = 0
    for size, u, mid, h, t in banks:
        if shared_h is not None:
            h = shared_h
        gamma = lam / h
        centers = mid + (torch.arange(size, dtype=x.dtype, device=x.device) - (size - 1) / 2) * h
        linear.weight[start:start+size] = gamma * u
        linear.bias[start:start+size] = -gamma * centers
        records.append(dict(start=start, size=size, midpoint=mid, h=h, gamma=gamma,
                            lambda_actual=gamma * h,
                            lo=centers[0].item(), hi=centers[-1].item(),
                            coverage=((t >= centers[0]) & (t <= centers[-1])).double().mean().item()))
        start += size
    return dict(n_dirs=len(banks), lam=lam, banks=records,
                gamma_min=min(b['gamma'] for b in records),
                gamma_max=max(b['gamma'] for b in records))
