"""The two-wall rule, frozen for expC10 (docs/lambda_two_wall_rule.md). Do not edit during the experiment."""
from __future__ import annotations

import numpy as np

C_CONST = 10.0
HALO = 32
M_MAX = 8
LAM_GRID_RULE = np.geomspace(0.03, 1.5, 600)
KERNEL_ORDER = {"tanh": 1, "gelu": 2, "swish": 2}
EPS = {"fp64": float(np.finfo(np.float64).eps), "fp32": float(np.finfo(np.float32).eps)}


def log_khat(act: str, xi):
    """log of the normalized kernel transform |Khat(xi)|/|Khat(0)|, xi > 0, overflow-safe."""
    xi = np.asarray(xi, dtype=float)
    if act == "tanh":
        a = np.pi * xi / 2
        return np.log(a) - (a + np.log1p(-np.exp(-2 * a)) - np.log(2))
    if act == "gelu":
        return np.log1p(xi ** 2) - xi ** 2 / 2
    if act == "swish":
        a = np.pi * xi
        return 2 * np.log(np.pi * xi) + (a + np.log1p(np.exp(-2 * a)) - np.log(2)) \
            - 2 * (a + np.log1p(-np.exp(-2 * a)) - np.log(2))
    raise ValueError(act)


def E_alias(act: str, theta_b: float, lam: float) -> float:
    r = KERNEL_ORDER[act]
    ms = np.arange(-M_MAX, M_MAX + 1)
    tt = theta_b + 2 * np.pi * ms
    lt = 2 * log_khat(act, np.abs(tt) / lam) - 2 * r * np.log(np.abs(tt))
    lt -= lt.max()
    v = np.exp(lt)
    return float(np.sqrt(v[ms != 0].sum() / v.sum()))


def E_comp(act: str, theta_b: float, lam: float, N: int, eps: float) -> float:
    r = KERNEL_ORDER[act]
    W = N + 2 * HALO + 1
    with np.errstate(over="ignore"):
        amp = theta_b ** r / (lam ** (r - 1) * np.exp(log_khat(act, theta_b / lam)))
    return float(C_CONST * eps * (1.0 + W * amp))


def two_wall(act: str, N: int, omega: float, precision: str):
    """Returns (lambda_star, E_star, theta_b) for band edge omega (rad per unit x)."""
    eps = EPS[precision]
    theta_b = 2.0 * omega / N
    if theta_b >= np.pi:
        return None, None, theta_b
    tot = np.array([E_alias(act, theta_b, l) + E_comp(act, theta_b, l, N, eps) for l in LAM_GRID_RULE])
    i = int(np.argmin(tot))
    return float(LAM_GRID_RULE[i]), float(tot[i]), float(theta_b)


def constant_rule(act: str, precision: str) -> float:
    """Baseline S3: largest lambda with |Khat(2 pi / lambda)| / |Khat(0)| = eps."""
    eps = EPS[precision]
    g = LAM_GRID_RULE
    vals = log_khat(act, 2 * np.pi / g) - np.log(eps)
    ok = np.where(vals <= 0)[0]
    return float(g[ok.max()])
