"""SPSA (simultaneous-perturbation stochastic approximation), Spall 1992.

Two full-batch loss evaluations per iteration:
    ghat = (L(x + c_k*Delta) - L(x - c_k*Delta)) / (2 c_k) * Delta
with Delta a Rademacher (+-1) vector (Delta^{-1} = Delta elementwise).

Standard gain schedules:
    a_k = a / (k + 1 + A)^0.602,   c_k = c / (k + 1)^0.101,   A = 0.1 * T.

Not expected to be precision-competitive; included as a gradient-free
datapoint. The perturbation scale c is the sensitive knob (per Sam:
"make the perturbation size large enough") -- tune it via spsa_tune().
"""

from __future__ import annotations

import torch


def spsa_minimize(x0, loss_fn, max_steps=3000, a=0.01, c=0.1,
                  step_callback=None, log_every=10, seed=0):
    """Minimise loss_fn (x -> float tensor) from x0. Returns final x."""
    x = x0.detach().clone()
    m = x.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = 0.1 * max_steps
    best_x, best_f = x.clone(), float("inf")
    for k in range(int(max_steps)):
        a_k = a / (k + 1 + A) ** 0.602
        c_k = c / (k + 1) ** 0.101
        delta = (torch.randint(0, 2, (m,), generator=gen, dtype=torch.int64)
                 .to(torch.float64) * 2.0 - 1.0).to(x.device)
        f_plus = float(loss_fn(x + c_k * delta))
        f_minus = float(loss_fn(x - c_k * delta))
        ghat = (f_plus - f_minus) / (2.0 * c_k) * delta
        x = x - a_k * ghat
        f_mid = 0.5 * (f_plus + f_minus)
        if f_mid < best_f:
            best_f, best_x = f_mid, x.clone()
        if step_callback is not None and ((k + 1) % log_every == 0 or k + 1 == max_steps):
            step_callback(k + 1, x, f_mid, float("nan"))
    return x, best_x
