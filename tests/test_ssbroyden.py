"""Validation for the PyTorch SSBroyden port (src/training/ssbroyden.py).

Checks the ported SSB-II + strong-Wolfe optimizer on standard problems where the
correct answer is known, so we can tell the port actually works before trusting
it on the QI objective:

  1. well-conditioned quadratic  -> machine precision
  2. ill-conditioned quadratic   -> still converges (tests the self-scaling)
  3. Rosenbrock (2D and 10D)     -> converges to the global min at all-ones
  4. least-squares residual form -> the shape our experiments actually use

Run: uv run --extra dev python tests/test_ssbroyden.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.ssbroyden import ssbroyden_minimize  # noqa: E402

torch.set_default_dtype(torch.float64)


def make_fg(loss_fn):
    def fg(x):
        xx = x.detach().clone().requires_grad_(True)
        f = loss_fn(xx)
        (g,) = torch.autograd.grad(f, xx)
        return float(f), g.detach()
    return fg


def test_quadratic(n=20, cond=1.0):
    torch.manual_seed(0)
    d = torch.logspace(0, torch.log10(torch.tensor(cond)).item(), n) if cond > 1 else torch.ones(n)
    b = torch.randn(n)
    loss = lambda x: 0.5 * torch.sum(d * (x - b) ** 2)
    x, info = ssbroyden_minimize(torch.zeros(n), make_fg(loss), max_steps=500)
    err = float(torch.linalg.norm(x - b))
    print(f"  quadratic n={n} cond={cond:g}: |x-x*|={err:.3e} f={info.f:.3e} "
          f"grad={info.grad_norm:.3e} steps={info.steps} converged={info.converged}")
    return err


def test_rosenbrock(n=2, max_steps=2000):
    def loss(x):
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    x0 = torch.full((n,), -1.2)
    x, info = ssbroyden_minimize(x0, make_fg(loss), max_steps=max_steps)
    err = float(torch.linalg.norm(x - torch.ones(n)))
    print(f"  rosenbrock n={n}: |x-1|={err:.3e} f={info.f:.3e} "
          f"grad={info.grad_norm:.3e} steps={info.steps}")
    return err


def test_least_squares(n=30, m=200):
    """The residual/least-squares shape our QI objective has."""
    torch.manual_seed(1)
    A = torch.randn(m, n)
    x_true = torch.randn(n)
    y = A @ x_true
    loss = lambda x: torch.mean((A @ x - y) ** 2)
    x, info = ssbroyden_minimize(torch.zeros(n), make_fg(loss), max_steps=800)
    err = float(torch.linalg.norm(x - x_true) / torch.linalg.norm(x_true))
    print(f"  least-squares m={m} n={n}: rel|x-x*|={err:.3e} f={info.f:.3e} "
          f"grad={info.grad_norm:.3e} steps={info.steps}")
    return err


if __name__ == "__main__":
    print("SSBroyden port validation")
    ok = True

    e = test_quadratic(n=20, cond=1.0);      ok &= e < 1e-10
    e = test_quadratic(n=20, cond=1e6);      ok &= e < 1e-6
    e = test_rosenbrock(n=2);                ok &= e < 1e-6
    e = test_rosenbrock(n=10);               ok &= e < 1e-4
    e = test_least_squares();                ok &= e < 1e-8

    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
