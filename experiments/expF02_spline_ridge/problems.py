"""Part-A problems on [-1,1]^2, all with u* = 0 on the boundary.

  poisson:        -lap u = f
  darcy_control:  -a lap u - grad a . grad u = f   (smooth manufactured a)

Every hand-coded derivative/forcing is FD-verified by verify_all().
"""
from __future__ import annotations

import numpy as np

PI = np.pi


def ustar(P):
    x, y = P[:, 0], P[:, 1]
    return np.sin(PI * x) * np.sin(PI * y) + 0.5 * np.sin(2 * PI * x) * np.sin(PI * y)


def ustar_x(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.cos(PI * x) * np.sin(PI * y) + PI * np.cos(2 * PI * x) * np.sin(PI * y)


def ustar_y(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.sin(PI * x) * np.cos(PI * y) + 0.5 * PI * np.sin(2 * PI * x) * np.cos(PI * y)


def ustar_lap(P):
    x, y = P[:, 0], P[:, 1]
    uxx = -PI**2 * np.sin(PI * x) * np.sin(PI * y) - 2 * PI**2 * np.sin(2 * PI * x) * np.sin(PI * y)
    uyy = -PI**2 * np.sin(PI * x) * np.sin(PI * y) - 0.5 * PI**2 * np.sin(2 * PI * x) * np.sin(PI * y)
    return uxx + uyy


def a_ctrl(P):
    x, y = P[:, 0], P[:, 1]
    return 3.0 + np.exp(np.sin(PI * x) * np.sin(PI * y))


def a_ctrl_x(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(np.sin(PI * x) * np.sin(PI * y)) * PI * np.cos(PI * x) * np.sin(PI * y)


def a_ctrl_y(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(np.sin(PI * x) * np.sin(PI * y)) * PI * np.sin(PI * x) * np.cos(PI * y)


def poisson():
    terms = [((2, 0), -1.0), ((0, 2), -1.0)]
    forcing = lambda P: -ustar_lap(P)
    return dict(name="poisson", terms=terms, forcing=forcing, exact=ustar)


def darcy_control():
    terms = [((2, 0), lambda P: -a_ctrl(P)),
             ((0, 2), lambda P: -a_ctrl(P)),
             ((1, 0), lambda P: -a_ctrl_x(P)),
             ((0, 1), lambda P: -a_ctrl_y(P))]
    forcing = lambda P: -(a_ctrl(P) * ustar_lap(P)
                          + a_ctrl_x(P) * ustar_x(P)
                          + a_ctrl_y(P) * ustar_y(P))
    return dict(name="darcy_control", terms=terms, forcing=forcing, exact=ustar)


PROBLEMS = [poisson, darcy_control]


def _fd_partial(f, P, ax, ay, h=1e-5):
    if ax == 0 and ay == 0:
        return f(P)
    if ax + ay == 1:
        col = 0 if ax else 1
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - f(Pm)) / (2 * h)
    if (ax, ay) in [(2, 0), (0, 2)]:
        col = 0 if ax else 1
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - 2 * f(P) + f(Pm)) / h**2
    raise ValueError((ax, ay))


def verify_all(tol=2e-4):
    rng = np.random.default_rng(7)
    P = rng.uniform(-0.9, 0.9, (60, 2))
    checks = [
        ("u_x", ustar_x(P), _fd_partial(ustar, P, 1, 0)),
        ("u_y", ustar_y(P), _fd_partial(ustar, P, 0, 1)),
        ("lap", ustar_lap(P),
         _fd_partial(ustar, P, 2, 0, h=1e-4) + _fd_partial(ustar, P, 0, 2, h=1e-4)),
        ("a_x", a_ctrl_x(P), _fd_partial(a_ctrl, P, 1, 0)),
        ("a_y", a_ctrl_y(P), _fd_partial(a_ctrl, P, 0, 1)),
    ]
    # boundary condition: u* = 0 on all four edges
    s = np.linspace(-1, 1, 41)
    for edge in [np.stack([s, np.full_like(s, -1.0)], 1), np.stack([s, np.full_like(s, 1.0)], 1),
                 np.stack([np.full_like(s, -1.0), s], 1), np.stack([np.full_like(s, 1.0), s], 1)]:
        assert np.max(np.abs(ustar(edge))) < 1e-12
    for name, ours, fd in checks:
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(ours - fd)) / scale
        assert err < tol, f"check '{name}' failed: rel err {err:.2e}"
    # forcing identity for both problems: L[u*] evaluated by FD == forcing
    for prob_fn in PROBLEMS:
        prob = prob_fn()
        lhs = np.zeros(len(P))
        for (ax, ay), coeff in prob["terms"]:
            c = coeff(P) if callable(coeff) else coeff
            lhs += c * _fd_partial(ustar, P, ax, ay, h=1e-4)
        err = np.max(np.abs(lhs - prob["forcing"](P))) / max(1.0, np.max(np.abs(lhs)))
        assert err < tol, f"forcing identity '{prob['name']}': rel err {err:.2e}"
