"""Single-hidden-layer GELU MLP + the QI-style initialization (expF04).

The QI init is the repo's 1D construction gamma*(x - c_k), lifted to high-d WITHOUT
swamping:

  - each weight row is  w_k = gamma * u_k  with u_k a UNIT direction, so ||w_k|| = gamma
    (in 1D the "direction" is the scalar 1, so w_k = gamma -- identical to GammaLinear).
  - centers c_k are swept across the ACTUAL range of the projection t = u_k . x on the
    [-1,1]-normalized data, so gamma*(u.x) and gamma*c_k are the same magnitude and the
    center sweep is never swamped by the dot product.
  - gamma = lambda / h, lambda = 0.25, h = center spacing in the projection coordinate.
    With projection half-range A this is gamma = 0.125 * N / A, recovering 0.125*N when
    A = 1 (the 1D domain [-1,1]).

Bias b_k = -gamma * c_k so the pre-activation is gamma*(u_k . x - c_k). Readout (fc2)
keeps default init in both arms; the only thing the QI arm changes is fc1's scale/bias.
"""
from __future__ import annotations

import torch
import torch.nn as nn


LAMBDA_STAR = 0.25


class SimpleMLP(nn.Module):
    """d_in -> width -> d_out, single hidden layer. Activation configurable:
    GELU (modern default) or tanh (the construction's activation -- expA04 bounded
    null space; the d=1 control showed the QI-init advantage is tanh-bound)."""

    def __init__(self, d_in: int, width: int, d_out: int, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width)
        self.act = nn.Tanh() if activation == "tanh" else nn.GELU()
        self.fc2 = nn.Linear(width, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@torch.no_grad()
def qi_init_(
    model: SimpleMLP,
    x_train: torch.Tensor,
    lam: float = LAMBDA_STAR,
    proj_quantile: float = 0.999,
    max_proj_rows: int = 4096,
) -> dict:
    """Apply the QI init to model.fc1 in-place. Returns diagnostics.

    Steps:
      1. take the already random-initialized fc1 weight, normalize each row to unit L2
         -> unit directions u_k (keeps the random directions, drops the random scale).
      2. project a sample of the [-1,1]-normalized training data onto every u_k and take
         a robust half-range A = quantile(|u.x|, proj_quantile) over the pooled sample.
      3. gamma = lam / h with h = 2A / N ; centers c_k = linspace(-A, A, N).
      4. w_k = gamma * u_k ; b_k = -gamma * c_k.
    """
    W = model.fc1.weight              # [N, d]
    N, d = W.shape
    dtype, device = W.dtype, W.device

    U = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)      # unit directions [N, d]

    xs = x_train
    if xs.shape[0] > max_proj_rows:
        idx = torch.randperm(xs.shape[0], device=xs.device)[:max_proj_rows]
        xs = xs[idx]
    proj = xs.to(device=device, dtype=dtype) @ U.t()          # [rows, N]
    A = proj.abs().flatten().quantile(proj_quantile).item()
    A = max(A, 1e-6)

    h = 2.0 * A / N
    gamma = lam / h                                           # = 0.125 * N / A
    centers = torch.linspace(-A, A, N, dtype=dtype, device=device)

    W.copy_(gamma * U)                                        # ||w_k|| = gamma
    model.fc1.bias.copy_(-gamma * centers)
    return {"gamma": gamma, "A": A, "h": h, "lambda": lam}


@torch.no_grad()
def qi_ridge_init_layer_(
    linear: nn.Linear,
    x_input: torch.Tensor,
    lam: float = LAMBDA_STAR,
    centers_per_dir: int = 16,
    uniform_centers: bool = True,
    proj_quantile: float = 0.999,
    max_proj_rows: int = 4096,
    generator: torch.Generator | None = None,
) -> dict:
    """Ridge-bundle QI init on a single nn.Linear, using x_input for projection ranges.

    x_input is whatever feeds this layer: raw features for layer 1, or the previous
    layer's post-activation outputs for a deeper layer (so a 2nd-layer init tiles the
    hidden representation, with fresh random directions).

    Ridge-bundle QI init: the faithful transport of the 1D construction.

    Neurons are partitioned into M bundles of P = centers_per_dir. Each bundle shares
    ONE random unit direction u_m and carries a full 1D QI sweep along it: centers
    uniform over the direction's own projection range [-A_m, A_m], spacing h_m = 2A_m/P,
    gamma_m = lam / h_m = O(P). gamma and center packing grow TOGETHER, per ridge --
    unlike qi_init_ (the M=N, P=1 corner) where gamma=O(N) with nothing packing.

    At d=1 with P=N this IS the repo construction: gamma = lam/(2/N) = 0.125*N.

    uniform_centers=False gives the 'scaled' variant (D02 win #3 analog): identical
    directions and gamma scale, but centers sampled randomly from the data projections
    instead of the uniform sweep -- isolates scale vs exact-sweep.
    """
    W = linear.weight                 # [N, d]
    N, d = W.shape
    dtype, device = W.dtype, W.device
    P = max(1, min(centers_per_dir, N))

    xs = x_input
    if xs.shape[0] > max_proj_rows:
        idx = torch.randperm(xs.shape[0], device=xs.device, generator=generator)[:max_proj_rows]
        xs = xs[idx]
    xs = xs.to(device=device, dtype=dtype)

    # bundle sizes: N split into ceil(N/P) bundles as evenly as full P allows
    sizes = [P] * (N // P)
    if N % P:
        sizes.append(N % P)

    W_new = torch.empty_like(W)
    b_new = torch.empty(N, dtype=dtype, device=device)
    gammas = []
    k = 0
    for size in sizes:
        u = torch.randn(d, dtype=dtype, device=device, generator=generator)
        u = u / u.norm().clamp_min(1e-12)
        t = xs @ u                                            # projections along u
        A = max(t.abs().quantile(proj_quantile).item(), 1e-6)
        h = 2.0 * A / size
        gamma = lam / h                                       # = lam * size / (2A)
        if uniform_centers:
            c = (torch.linspace(-A, A, size, dtype=dtype, device=device)
                 if size > 1 else torch.zeros(1, dtype=dtype, device=device))
        else:
            ridx = torch.randint(0, t.shape[0], (size,), device=device, generator=generator)
            c = t[ridx]
        W_new[k:k + size] = gamma * u.unsqueeze(0)
        b_new[k:k + size] = -gamma * c
        gammas.append(gamma)
        k += size

    W.copy_(W_new)
    linear.bias.copy_(b_new)
    g = torch.tensor(gammas)
    return {"n_dirs": len(sizes), "centers_per_dir": P, "lambda": lam,
            "gamma_mean": g.mean().item(), "gamma_min": g.min().item(),
            "gamma_max": g.max().item()}


@torch.no_grad()
def qi_ridge_init_(model, x_train, **kwargs) -> dict:
    """Back-compat wrapper: ridge init on model.fc1 (used by ridge_real / all20)."""
    return qi_ridge_init_layer_(model.fc1, x_train, **kwargs)


class SimpleMLP2(nn.Module):
    """Two-hidden-layer MLP: d_in -> width -> width -> d_out, configurable activation.
    fc1, fc2 are the two hidden layers; fc3 is the readout."""

    def __init__(self, d_in: int, width: int, d_out: int, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, d_out)
        self.act = nn.Tanh() if activation == "tanh" else nn.GELU()

    def hidden1(self, x):
        return self.act(self.fc1(x))

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))
