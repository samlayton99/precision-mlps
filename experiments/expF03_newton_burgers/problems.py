"""Steady viscous Burgers on [-1,1]^2, manufactured Taylor-Green solution.

  F_u(u,v) = u u_x + v u_y - nu lap u - f_u = 0
  F_v(u,v) = u v_x + v v_y - nu lap v - f_v = 0
  u* = -cos(pi x) sin(pi y),  v* = sin(pi x) cos(pi y)   (period 2)
Dirichlet BCs from the exact solution on all four edges.
"""
from __future__ import annotations

import numpy as np

PI = np.pi


def u_exact(P):
    return -np.cos(PI * P[:, 0]) * np.sin(PI * P[:, 1])


def v_exact(P):
    return np.sin(PI * P[:, 0]) * np.cos(PI * P[:, 1])


def u_x(P):
    return PI * np.sin(PI * P[:, 0]) * np.sin(PI * P[:, 1])


def u_y(P):
    return -PI * np.cos(PI * P[:, 0]) * np.cos(PI * P[:, 1])


def v_x(P):
    return PI * np.cos(PI * P[:, 0]) * np.cos(PI * P[:, 1])


def v_y(P):
    return -PI * np.sin(PI * P[:, 0]) * np.sin(PI * P[:, 1])


def lap_u(P):
    return -2 * PI**2 * u_exact(P)


def lap_v(P):
    return -2 * PI**2 * v_exact(P)


def f_u(P, nu):
    return u_exact(P) * u_x(P) + v_exact(P) * u_y(P) - nu * lap_u(P)


def f_v(P, nu):
    return u_exact(P) * v_x(P) + v_exact(P) * v_y(P) - nu * lap_v(P)


def _fd(f, P, col, h=1e-5):
    Pp, Pm = P.copy(), P.copy()
    Pp[:, col] += h
    Pm[:, col] -= h
    return (f(Pp) - f(Pm)) / (2 * h)


def _fd2(f, P, col, h=1e-4):
    Pp, Pm = P.copy(), P.copy()
    Pp[:, col] += h
    Pm[:, col] -= h
    return (f(Pp) - 2 * f(P) + f(Pm)) / h**2


def verify_all(nu, tol=2e-4):
    rng = np.random.default_rng(7)
    P = rng.uniform(-0.9, 0.9, (60, 2))
    checks = [
        ("u_x", u_x(P), _fd(u_exact, P, 0)),
        ("u_y", u_y(P), _fd(u_exact, P, 1)),
        ("v_x", v_x(P), _fd(v_exact, P, 0)),
        ("v_y", v_y(P), _fd(v_exact, P, 1)),
        ("lap_u", lap_u(P), _fd2(u_exact, P, 0) + _fd2(u_exact, P, 1)),
        ("lap_v", lap_v(P), _fd2(v_exact, P, 0) + _fd2(v_exact, P, 1)),
        ("f_u", f_u(P, nu),
         u_exact(P) * _fd(u_exact, P, 0) + v_exact(P) * _fd(u_exact, P, 1)
         - nu * (_fd2(u_exact, P, 0) + _fd2(u_exact, P, 1))),
        ("f_v", f_v(P, nu),
         u_exact(P) * _fd(v_exact, P, 0) + v_exact(P) * _fd(v_exact, P, 1)
         - nu * (_fd2(v_exact, P, 0) + _fd2(v_exact, P, 1))),
    ]
    for name, ours, fd in checks:
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(ours - fd)) / scale
        assert err < tol, f"burgers check '{name}' failed: rel err {err:.2e}"
