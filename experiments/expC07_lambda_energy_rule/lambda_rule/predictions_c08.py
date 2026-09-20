"""Lambda predictions for expC08: the frequency-resolved aliasing rule (new) vs the expC07 constant rule.

The NEW rule (Sam's table, 2026-09-08) uses the fp64 aliasing budget eps = 2^-52 and the target's
frequency per grid cell q = h*Omega/pi with h = 2/(N-1). For tanh it is the explicit equation

    8 z e^{-z} / (1-e^{-z})^3 * sinh(qz)/(qz) = eps / B,   z = pi^2 / lambda,           (eq. 2)

For gelu and swish the table entries were produced by "the full kernel-ratio constraint"; the
function `solve_kernel_ratio` below reproduces them (checked in `check_tables`).

The PREVIOUS rule (expC07) is frequency- and width-independent: |Khat(2 pi/lambda)| / |Khat(0)| = eps*
with eps* = A_tanh(0.25) = 5.65e-16, giving tanh 0.25, gelu 0.7070, swish 0.4554.
"""
from __future__ import annotations

import math

import mpmath as mp

EPS64 = 2.0 ** -52
PREV = {"tanh": 0.25, "gelu": 0.7070, "swish": 0.4554}          # expC07 aliasing rule
LIMIT_NEW = {"tanh": 0.23579, "gelu": 0.68662, "swish": 0.43127}  # q -> 0 limits quoted with the table

WIDTHS = [16, 32, 64, 128, 256, 512]
# target name -> (Omega / pi, B)
TARGETS = {
    "sin_1pi": (1, 1.0),
    "cos_4pi": (4, 1.0),
    "f8": (8, 1.0),
    "sin_16pi": (16, 1.0),
    "f32": (32, 1.0),
    "sin_100pi": (100, 1.0),
}
_D = None
TABLE = {
    "tanh": {
        "sin_1pi":   [0.217, 0.230, 0.234, 0.235, 0.235, 0.235],
        "cos_4pi":   [0.122, 0.189, 0.218, 0.230, 0.234, 0.235],
        "f8":        [_D, 0.126, 0.190, 0.218, 0.230, 0.234],
        "sin_16pi":  [_D, _D, 0.128, 0.190, 0.218, 0.230],
        "f32":       [_D, _D, _D, 0.129, 0.191, 0.218],
        "sin_100pi": [_D, _D, _D, _D, 0.057, 0.157],
    },
    "gelu": {
        "sin_1pi":   [0.647, 0.670, 0.681, 0.685, 0.686, 0.686],
        "cos_4pi":   [0.487, 0.604, 0.649, 0.671, 0.681, 0.685],
        "f8":        [_D, 0.496, 0.605, 0.649, 0.671, 0.681],
        "sin_16pi":  [_D, _D, 0.500, 0.606, 0.650, 0.671],
        "f32":       [_D, _D, _D, 0.502, 0.606, 0.650],
        "sin_100pi": [_D, _D, _D, _D, 0.336, 0.552],
    },
    "swish": {
        "sin_1pi":   [0.407, 0.425, 0.429, 0.430, 0.431, 0.431],
        "cos_4pi":   [0.237, 0.361, 0.409, 0.425, 0.429, 0.430],
        "f8":        [_D, 0.245, 0.362, 0.409, 0.425, 0.429],
        "sin_16pi":  [_D, _D, 0.249, 0.363, 0.409, 0.425],
        "f32":       [_D, _D, _D, 0.251, 0.363, 0.409],
        "sin_100pi": [_D, _D, _D, _D, 0.113, 0.303],
    },
}


def khat(act: str, xi):
    """Normalized kernel transform |Khat(xi)| / |Khat(0)| for K = psi^(r) (expC07 / lambda_rule_theory.md)."""
    xi = abs(mp.mpf(xi))
    if xi == 0:
        return mp.mpf(1)
    if act == "tanh":
        return (mp.pi * xi / 2) / mp.sinh(mp.pi * xi / 2)
    if act == "gelu":
        return (1 + xi ** 2) * mp.exp(-xi ** 2 / 2)
    if act == "swish":
        return mp.pi ** 2 * xi ** 2 * mp.cosh(mp.pi * xi) / mp.sinh(mp.pi * xi) ** 2
    raise ValueError(act)


def q_of(N: int, omega_over_pi: float, h_convention: str = "N-1") -> float:
    h = 2.0 / (N - 1) if h_convention == "N-1" else 2.0 / N
    return h * omega_over_pi  # q = h*Omega/pi with Omega = omega_over_pi * pi


def _largest_root(g, lo=0.02, hi=2.0, n=4000):
    """Largest lambda in [lo, hi] with g(lambda) = 0, g increasing through the root (log-scanned bisection)."""
    xs = [lo * (hi / lo) ** (i / n) for i in range(n + 1)]
    vals = [g(x) for x in xs]
    for i in range(n, 0, -1):
        if vals[i - 1] < 0 <= vals[i]:
            return float(mp.findroot(g, (xs[i - 1], xs[i]), solver="bisect", tol=1e-30))
    return None


def solve_tanh_eq2(N: int, omega_over_pi: float, B: float = 1.0, eps: float = EPS64, h_convention="N-1"):
    """Sam's equation (2) for tanh. Returns None when q >= 1 (frequency at or past Nyquist)."""
    q = q_of(N, omega_over_pi, h_convention)
    if q >= 1:
        return None

    def g(lam):
        z = mp.pi ** 2 / lam
        s = q * z
        corr = mp.sinh(s) / s if s > 0 else mp.mpf(1)
        return mp.log(8 * z * mp.exp(-z) / (1 - mp.exp(-z)) ** 3 * corr) - mp.log(mp.mpf(eps) / B)

    return _largest_root(g)


def solve_kernel_ratio(act: str, N: int, omega_over_pi: float, B: float = 1.0, eps: float = EPS64,
                       h_convention="N-1", variant="edge4"):
    """Candidate general-activation constraints (used to identify how the gelu/swish table was made).

    variant 'edge4':  4 * Khat((2pi - theta0)/lam) / Khat(theta0/lam) = eps/B        (two ghosts, both sides)
    variant 'edge2':  2 * Khat((2pi - theta0)/lam) / Khat(theta0/lam) = eps/B
    variant 'sum4':   4 * [Khat((2pi-theta0)/lam) + Khat((2pi+theta0)/lam)] / (2 Khat(theta0/lam)) = eps/B
    variant 'zero4':  4 * Khat(2pi/lam) = eps/B   (frequency-independent; the expC07 rule with eps/4)
    with theta0 = q*pi. Returns None when q >= 1.
    """
    q = q_of(N, omega_over_pi, h_convention)
    if q >= 1:
        return None
    th = q * mp.pi

    def ratio(lam):
        if variant == "zero4":
            return 4 * khat(act, 2 * mp.pi / lam)
        sig = khat(act, th / lam)
        g1 = khat(act, (2 * mp.pi - th) / lam)
        if variant == "edge4":
            return 4 * g1 / sig
        if variant == "edge2":
            return 2 * g1 / sig
        if variant == "sum4":
            g2 = khat(act, (2 * mp.pi + th) / lam)
            return 2 * (g1 + g2) / sig
        raise ValueError(variant)

    return _largest_root(lambda lam: mp.log(ratio(lam)) - mp.log(mp.mpf(eps) / B))


def floor3(x):
    return None if x is None else math.floor(x * 1000 + 1e-9) / 1000


def check_tables():
    mp.mp.dps = 30
    print("tanh: eq. (2) recomputed (floor to 3 dp) vs table")
    for name, (w, B) in TARGETS.items():
        rec = [floor3(solve_tanh_eq2(N, w, B)) for N in WIDTHS]
        print(f"  {name:10s} rec {rec}\n  {'':10s} tab {TABLE['tanh'][name]}")
    for act in ("gelu", "swish"):
        for variant in ("edge4", "edge2", "sum4", "zero4"):
            print(f"{act}: variant {variant}")
            for name, (w, B) in TARGETS.items():
                rec = [floor3(solve_kernel_ratio(act, N, w, B, variant=variant)) for N in WIDTHS]
                print(f"  {name:10s} rec {rec}\n  {'':10s} tab {TABLE[act][name]}")
    print("q->0 limits: tanh eq2", solve_tanh_eq2(10**9, 1), "| gelu 4A=eps", solve_kernel_ratio("gelu", 10**9, 1, variant="zero4"),
          "| swish 4A=eps", solve_kernel_ratio("swish", 10**9, 1, variant="zero4"))


if __name__ == "__main__":
    check_tables()
