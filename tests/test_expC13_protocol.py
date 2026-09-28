"""expC13 shared protocol: the pfloat arithmetic path against the expC11/expC12 implementations it
replaces, the oracle, the meter, the forward passes, and exact model storage."""
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pfloat = pytest.importorskip("pfloat")
gmpy2 = pytest.importorskip("gmpy2")
import mpmath as mp  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.methods import chebyshev, mhaskar, quills  # noqa: E402


@pytest.mark.parametrize("p", [12, 24, 53])
def test_oracle_is_one_correct_rounding(p):
    """sample() returns Q_p(f(x)) for p-bit x, f evaluated far beyond p bits (checked with MPFR)."""
    F = C.fmt(p)
    x = C.inputs(C.midpoint_points(64), F)
    y = C.sample("chirp", x)
    for xi, yi in zip(x.to_fractions(), y.to_fractions()):
        with gmpy2.context(precision=400):
            v = gmpy2.sin(8 * gmpy2.const_pi() * (gmpy2.mpfr(gmpy2.mpq(xi.numerator, xi.denominator)) + 1) ** 2)
        with gmpy2.context(precision=p, round=gmpy2.RoundToNearest):
            want = gmpy2.mpfr(v)
        assert Fraction(*want.as_integer_ratio()) == yi


def test_meter_is_stable_and_exact():
    """Errors computed at 320 and 640 bits agree far below the reported digits, and a p-bit output
    that equals the rounded truth has the error of that rounding alone."""
    pts = C.linspace_points(257)
    F = C.fmt(20)
    out = C.round_values(C.exact_values("runge", pts, 200), F, 200)
    e1 = C.Meter("runge", pts, 320).errors(out)["rel_l2"]
    e2 = C.Meter("runge", pts, 640).errors(out)["rel_l2"]
    assert abs(e1 - e2) <= 1e-14 * e2 and 0 < e1 < 2.0 ** -20


@pytest.mark.parametrize("p", [20, 32, 53])
def test_quills_arithmetic_matches_expC11_implementation(p):
    """The same QUILLS model (expC11's centered features) through pfloat and through src/precision
    (an independent C implementation of the p-bit model and DGELSS port): identical readouts."""
    from src.precision import pbit
    W, H, M = 64, 4, 301
    N = W - 2 * H - 1
    lam = 0.4
    F = C.fmt(p)
    x = C.inputs(C.linspace_points(M), F)
    y = C.sample("runge", x)
    h = C.constant(2, F) / C.constant(N, F)
    centers = C.constant(-1, F) + pfloat.array(np.arange(-H, N + H + 1), F) * h
    gamma = C.constant(Fraction(lam), F) / h
    phi = pfloat.tanh((x.reshape(-1, 1) - centers.reshape(1, -1)) * gamma)
    A = pfloat.concatenate([phi, pfloat.ones((M, 1), F)], axis=1)
    w = pfloat.lstsq(A, y, rcond=Fraction(2) ** (1 - p))[0]
    ref = pbit.run(x.to_numpy(), y.to_numpy(), x.to_numpy()[:5], N, H, lam, pbit.panel_format(p))
    np.testing.assert_array_equal(w.to_numpy(), ref["weights"])


@pytest.mark.parametrize("p", [16, 24, 53])
def test_mhaskar_port_matches_expC12_kernels(p):
    """Projection, monomial conversion, Taylor recurrence and stencil assembly: bit-identical to the
    reviewed expC12 MPFR kernels on the same p-bit nodes and labels (tanh(b0) injected, since
    expC12 used a correctly rounded tanh and expC13 uses tanh_p)."""
    from experiments.expC12_mhaskar_comparison import pbit as old
    F = C.fmt(p)
    deg, M = 14, 64
    x = chebyshev.nodes(p, M)
    y = C.sample("runge", x)
    c_old, poly_old, t_old, b_old = old.polynomial_data(x.to_numpy(), y.to_numpy(), deg, p)
    cheb = chebyshev.projection("runge", p, deg, count=M)
    polys = chebyshev.to_monomials(cheb)
    b0 = mhaskar.bias_point(F)
    tay = chebyshev.tanh_taylor(b0, deg, c0=pfloat.array(t_old[0], F))
    np.testing.assert_array_equal(cheb.to_numpy(), c_old)
    np.testing.assert_array_equal(np.stack([q.to_numpy() for q in polys]), poly_old)
    np.testing.assert_array_equal(tay.to_numpy(), t_old)
    assert float(b0) == b_old
    degs = [3, 8, 14]
    for step in (0.05, 0.5):
        a, slopes, ok = mhaskar.stencils(polys, tay, degs, step)
        for row, d in enumerate(degs):
            net = mhaskar.network(a[row], slopes, b0, d)
            m_old = old.construct(poly_old[d], t_old, d, step, b_old, p)
            np.testing.assert_array_equal(net.readout.to_numpy(), m_old.readout)
            np.testing.assert_array_equal(net.slope.to_numpy(), m_old.slope)


def _mpfr_ctx(p):
    return gmpy2.context(precision=p, round=gmpy2.RoundToNearest, subnormalize=True,
                         emin=C.EMIN - p + 2, emax=C.EMAX + 1)


@pytest.mark.parametrize("p", [11, 24, 53])
def test_tanh_forward_matches_independent_mpfr_replay(p):
    """TanhNet.forward against a replay written from SPEC alone: gmpy2 operations in the format and
    the independent tanh_p of tests/pbit_replay_reference.py."""
    sys.path.insert(0, str(ROOT / "tests"))
    import pbit_replay_reference as ref
    F = C.fmt(p)
    rng = np.random.default_rng(p)
    net = C.TanhNet(pfloat.array(rng.uniform(-40, 40, 9), F), pfloat.array(rng.uniform(-5, 5, 9), F),
                    pfloat.array(rng.standard_normal(9), F), pfloat.array(0.25, F))
    x = pfloat.array(np.linspace(-1, 1, 41), F)
    got = net.forward(x).to_numpy()
    for xi, gi in zip(x.to_numpy(), got):
        with _mpfr_ctx(p):
            s = gmpy2.mpfr(float(net.offset))
            for w, b, a in zip(net.slope.to_numpy(), net.bias.to_numpy(), net.readout.to_numpy()):
                z = gmpy2.mpfr(w) * gmpy2.mpfr(xi) + gmpy2.mpfr(b)
                t = ref.tanh_p(float(z), p)
                s = s + gmpy2.mpfr(t) * gmpy2.mpfr(a)
        assert float(s) == gi


def test_saved_models_replay_bit_for_bit(tmp_path):
    for p in (24, 80):
        F = C.fmt(p)
        rng = np.random.default_rng(1)
        net = C.TanhNet(pfloat.array(rng.uniform(-9, 9, 5), F), pfloat.array(rng.uniform(-1, 1, 5), F),
                        pfloat.array(rng.standard_normal(5), F), pfloat.array(Fraction(1, 3), F))
        C.save_model(tmp_path / f"m{p}.npz", net, p, {"k": 1})
        back = C.load_model(tmp_path / f"m{p}.npz")
        x = pfloat.array(np.linspace(-1, 1, 17), F)
        assert C.same_bits(net.forward(x), back.forward(x))


def test_oracle_resolves_a_near_midpoint_by_ziv_retries():
    """At p = 113 the rounded training point x = Q(1/5) puts 1/(1+25x^2) within ~2^-230 of a 113-bit
    rounding midpoint (the first-order offset from 1/2 is exactly -2^-115); the oracle must re-evaluate
    at higher precision and agree with exact rational rounding."""
    F = C.fmt(113)
    x = C.inputs([Fraction(1, 5), Fraction(-1, 5), Fraction(1, 3)], F)
    y = C.sample("runge", x)
    for xi, yi in zip(x.to_fractions(), y.to_fractions()):
        exact = 1 / (1 + 25 * xi * xi)
        with gmpy2.context(precision=113, round=gmpy2.RoundToNearest):
            want = gmpy2.mpfr(gmpy2.mpq(exact.numerator, exact.denominator))
        assert Fraction(*want.as_integer_ratio()) == yi
