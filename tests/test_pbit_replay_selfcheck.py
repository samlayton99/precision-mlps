"""Self-checks of the independent gmpy2 reference (tests/pbit_replay_reference.py); no C code involved."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

gmpy2 = pytest.importorskip("gmpy2")
sys.path.insert(0, str(Path(__file__).parent))
import pbit_replay_reference as R  # noqa: E402

N, HALO, M, LAM = 8, 2, 41, 0.3


def _problem(p, M=M, Me=57):
    x = np.linspace(-1, 1, M)
    y = np.sin(8 * np.pi * (x + 1) ** 2)
    xe = np.linspace(-1, 1, Me)
    q = np.vectorize(lambda v: R.q(v, p))
    return q(x), q(y), q(xe), R.q(LAM, p)


def _ulp(v, p):
    # ulp of a p-bit number of magnitude |v| (unbounded exponent)
    e = math.frexp(abs(v))[1] if v != 0 else 0
    return math.ldexp(1.0, e - p)


def test_features_match_numpy_p53():
    x, y, xe, lam = _problem(53)
    phi = R.features(53, x, N, HALO, lam)
    h = 2.0 / N
    c = -1.0 + np.arange(-HALO, N + HALO + 1) * h
    ref = np.tanh((lam / h) * (x[:, None] - c[None, :]))
    assert phi.shape == (M, N + 2 * HALO + 1)
    assert np.max(np.abs(phi - ref)) <= 1e-14


def test_tanh_p53_vs_math():
    zs = np.concatenate([np.linspace(-40, 40, 801), [1e-300, -1e-12, 0.1, 0.35, 0.3466, 19.4, -19.41]])
    for z in zs:
        z = float(z)
        got = R.tanh_p(z, 53)
        assert abs(got - math.tanh(z)) <= 4 * _ulp(math.tanh(z), 53), z


def test_tanh_p24_vs_true():
    p = 24
    zs = [R.q(float(v), p) for v in np.linspace(-40, 40, 801)] + [R.q(1e-6, p), R.q(-0.2, p)]
    for z in zs:
        got = R.tanh_p(z, p)
        with gmpy2.context(precision=300):
            true = gmpy2.tanh(gmpy2.mpfr(z))
            err = abs(gmpy2.mpfr(got) - true)
        assert float(err) <= 4 * _ulp(float(true), p), z


@pytest.mark.parametrize("p", [8, 11, 24, 53])
def test_outputs_are_pbit(p):
    x, y, xe, lam = _problem(p, M=25, Me=17)
    phi = R.features(p, x, N, HALO, lam)
    w = [R.q(v, p) for v in np.linspace(-0.3, 0.4, N + 2 * HALO + 2)]
    fit = R.forward(p, xe, N, HALO, lam, w)
    for v in [*np.ravel(phi), *fit]:
        assert np.isfinite(v) and float(v) == R.q(float(v), p)
    C = R.constants(p)
    for key, v in C.items():
        for vv in v if isinstance(v, list) else [v]:
            assert vv == R.q(vv, p), (key, vv)
    assert C["coeffs"][0] == 1.0


def test_constants_hand_values():
    C = R.constants(8)
    assert C["LN2HI"] == 0.6875  # ln2 to 4 bits (k_b = 4)
    assert len(C["coeffs"]) == 3
    assert R.q(Fraction(1, 3), 8) == 0.333984375
