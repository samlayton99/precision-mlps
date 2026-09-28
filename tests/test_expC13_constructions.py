"""expC13 constructions: each network equals its mathematical definition when the arithmetic is wide
(p = 200), has the size the source paper gives, and behaves as the theory predicts at p = 53."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pfloat = pytest.importorskip("pfloat")
import mpmath as mp  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.methods import chebnet  # noqa: E402

WIDE = 200


def _m(v):
    """An exact number as an mpf at 400 bits, independent of mpmath's current precision."""
    with mp.workprec(400):
        return C._mpf(Fraction(v))


# ---------------------------------------------------------------- ChebNet

def _hier_matrix(m):
    """S_m of (2.22) in arXiv:1911.05467 v3: S_0 = I_2, S_j = (I_2 kron S_{j-1}) [[I, -A], [0, 2I]]."""
    S = np.eye(2)
    for j in range(1, m + 1):
        n1, n2 = 2 ** j + 1, 2 ** j - 1
        A = np.zeros((n1, n2))
        A[1:1 + n2] = np.fliplr(np.eye(n2))
        B = np.block([[np.eye(n1), -A], [np.zeros((n2, n1)), 2 * np.eye(n2)]])
        S = np.kron(np.eye(2), S) @ B
    return S


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_chebnet_split_is_the_paper_transform(m):
    F = C.fmt(WIDE)
    c = np.random.default_rng(m).integers(-9, 9, 2 ** (m + 1)).astype(float)
    got = [float(v) for v in chebnet.hier([pfloat.array(v, F) for v in c])]
    np.testing.assert_array_equal(got, _hier_matrix(m) @ c)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 13, 16, 31, 32, 48, 63, 64])
@pytest.mark.parametrize("normalize", [False, True])
def test_chebnet_network_is_the_chebyshev_sum(n, normalize):
    """At 200 bits the ReQU forward pass equals sum c_j T_j(x) to the 200-bit rounding level."""
    F = C.fmt(WIDE)
    coef = pfloat.array(np.random.default_rng(n).standard_normal(n + 1) / (1 + np.arange(n + 1)), F)
    net = chebnet.build(coef, n, normalize)
    x = pfloat.array(np.linspace(-1, 1, 17), F)
    out = net.forward(x)
    with mp.workprec(400):
        cs = [_m(v) for v in coef.to_fractions()]
        for xi, oi in zip(x.to_fractions(), out.to_fractions()):
            val = sum(cs[j] * mp.chebyt(j, _m(xi)) for j in range(n + 1))
            assert abs(_m(oi) - val) < mp.mpf(2) ** -180


@pytest.mark.parametrize("n,neurons,params,depth", [
    (3, 14, 66, 2), (7, 32, 200, 3), (15, 66, 462, 4), (31, 132, 980, 5), (63, 262, 2010, 6),   # padded tree
    (4, 24, 120, 3), (8, 42, 254, 4), (16, 76, 516, 5), (20, 96, 664, 5), (32, 142, 1034, 6),
    (48, 206, 1546, 6), (64, 272, 2064, 7)])                                                    # pruned tree
def test_chebnet_sizes(n, neurons, params, depth):
    """Padded sizes: neurons 2^(m+3) + 2m - 4 and nonzeros 64 2^m + 6m - 68 (derived from the layer
    algorithm; for n = 3 these equal Theorem 3's explicit network); depth floor(log2 n) + 1."""
    F = C.fmt(WIDE)
    coef = pfloat.array(np.random.default_rng(n).uniform(0.1, 0.9, n + 1), F)   # generic coefficients
    net = chebnet.build(coef, n, False)
    assert (net.neurons(), net.params(), net.depth()) == (neurons, params, depth)


# ---------------------------------------------------------------- staircase and Costarelli-Spigler

from experiments.expC13_five_method_comparison.methods import costarelli, mhaskar, staircase  # noqa: E402
from experiments.expC13_five_method_comparison.methods import chebyshev as cheb_mod  # noqa: E402


def _S(t):
    return (1 + mp.tanh(t)) / 2


@pytest.mark.parametrize("jumps", ["sample", "midpoint"])
@pytest.mark.parametrize("kappa", [Fraction(1, 2), Fraction(1), Fraction(4)])
def test_staircase_network_is_the_formula(jumps, kappa):
    """At 200 bits the tanh network (with its telescoped offset) equals
    f(-1) + sum_j [f(x_j) - f(x_{j-1})] S(kappa N (x - t_j)) on the same samples."""
    N = 24
    net = staircase.build("runge", WIDE, N, kappa, jumps)
    F = C.fmt(WIDE)
    xs = C.inputs([Fraction(-1) + Fraction(2 * j, N) for j in range(N + 1)], F)
    ys = [_m(v) for v in C.sample("runge", xs).to_fractions()]
    t = [Fraction(-1) + Fraction(2 * j, N) for j in range(1, N + 1)] if jumps == "sample" else \
        [Fraction(-1) + Fraction(2 * j - 1, N) for j in range(1, N + 1)]
    x = pfloat.array(np.linspace(-1, 1, 23), F)
    with mp.workprec(400):
        for xi, oi in zip(x.to_fractions(), net.forward(x).to_fractions()):
            val = ys[0] + sum((ys[j] - ys[j - 1]) * _S(_m(kappa * N) * (_m(xi) - _m(t[j - 1]))) for j in range(1, N + 1))
            assert abs(_m(oi) - val) < mp.mpf(2) ** -185


@pytest.mark.parametrize("centred", [False, True])
@pytest.mark.parametrize("trunc", ["2w", "tight"])
def test_costarelli_network_is_the_series(centred, trunc):
    """At 200 bits the merged tanh network equals sum_{|k|<=K} A_k psi(w x - k), psi(t) = S(t+1) - S(t),
    with A_k the oracle's samples inside [-1,1] and the linear taper outside."""
    w = 10
    K = costarelli.truncations(w, WIDE)[trunc]
    net = costarelli.build("exp", WIDE, w, K, centred)
    A = [_m(v) for v in costarelli.coefficients("exp", WIDE, w, K, centred).to_fractions()]
    x = pfloat.array(np.linspace(-1, 1, 21), C.fmt(WIDE))
    with mp.workprec(400):
        for xi, oi in zip(x.to_fractions(), net.forward(x).to_fractions()):
            u = _m(w) * _m(xi)
            val = sum(A[k + K] * (_S(u - k + 1) - _S(u - k)) for k in range(-K, K + 1))
            assert abs(_m(oi) - val) < mp.mpf(2) ** -185
        # the taper: F(u) = f(1) (2 - u) beyond x = 1
        u_last = Fraction(2 * K - 1, 2 * w) if centred else Fraction(K, w)
        if u_last > 1:
            assert abs(A[-1] - mp.e * (2 - _m(u_last))) < mp.mpf(2) ** -190


def test_psi_partition_of_unity():
    """Lemma 5.1 of Costarelli-Spigler (2015): sum_k psi(x - k) = 1."""
    with mp.workprec(200):
        for x in (mp.mpf("0.3"), mp.mpf("-7.25"), mp.mpf("100.5")):
            assert abs(sum(_S(x - k + 1) - _S(x - k) for k in range(-300, 301)) - 1) < mp.mpf(2) ** -150


def _rate(errors, sizes):
    return [np.log2(errors[i] / errors[i + 1]) / np.log2(sizes[i + 1] / sizes[i]) for i in range(len(sizes) - 1)]


def test_faithful_first_order_rates_at_p53():
    """The appendix forms converge at O(1/N) (classical staircase, Theorem 5.4's sampled series)."""
    from experiments.expC13_five_method_comparison.selection import Context
    ctx = Context("exp", 53)
    sizes = [32, 64, 128, 256]
    e_st = [ctx.val_meter.rel_l2(staircase.build("exp", 53, N, Fraction(1), "sample").forward(ctx.val_x)) for N in sizes]
    e_cs = [ctx.val_meter.rel_l2(costarelli.build("exp", 53, w, 2 * w, False).forward(ctx.val_x)) for w in sizes]
    for r in _rate(e_st, sizes) + _rate(e_cs, sizes):
        assert 0.9 < r < 1.1


def test_mhaskar_network_converges_to_its_polynomial_like_h_squared():
    """Lemma 3.2's centered differences: at 200 bits the degree-d network differs from the degree-d
    Chebyshev truncation by O(h^2) (halving h divides the difference by about 4)."""
    p, d = WIDE, 6
    F = C.fmt(p)
    cheb = cheb_mod.projection("exp", p, d, count=64)
    polys = cheb_mod.to_monomials(cheb)
    b0 = mhaskar.bias_point(F)
    tay = cheb_mod.tanh_taylor(b0, d)
    x = pfloat.array(np.linspace(-1, 1, 41), F)
    with mp.workprec(400):
        poly = [_m(v) for v in polys[d].to_fractions()]
        target = [sum(poly[r] * _m(xi) ** r for r in range(d + 1)) for xi in x.to_fractions()]
    diffs = []
    for h in (Fraction(1, 64), Fraction(1, 128), Fraction(1, 256)):
        a, slopes, ok = mhaskar.stencils(polys, tay, [d], float(h))
        out = mhaskar.network(a[0], slopes, b0, d).forward(x).to_fractions()
        with mp.workprec(400):
            diffs.append(max(abs(_m(o) - t) for o, t in zip(out, target)))
    assert 3.6 < diffs[0] / diffs[1] < 4.4 and 3.6 < diffs[1] / diffs[2] < 4.4


# ---------------------------------------------------------------- Mhaskar, common grid

@pytest.mark.parametrize("m", [1, 3, 8])
def test_lagrange_table_matches_moments(m):
    """sum_j L_kj j^i = [i == k] for i, k = 0..2m: exact maximal-order difference weights."""
    cols = mhaskar._lagrange_exact(m)
    for k in range(2 * m + 1):
        for i in range(2 * m + 1):
            assert sum(cols[j][k] * (j - m) ** i for j in range(2 * m + 1)) == (1 if i == k else 0)


def test_mhaskar_common_grid_converges_to_its_polynomial():
    """At 200 bits the common-grid network of degree d approaches the degree-d truncation as h -> 0,
    and uses 2 ceil(d/2) + 1 neurons."""
    p, d = WIDE, 6
    F = C.fmt(p)
    cheb = cheb_mod.projection("exp", p, d, count=64)
    polys = cheb_mod.to_monomials(cheb)
    b0 = mhaskar.bias_point(F)
    tay = cheb_mod.tanh_taylor(b0, d)
    x = pfloat.array(np.linspace(-1, 1, 21), F)
    with mp.workprec(400):
        poly = [_m(v) for v in polys[d].to_fractions()]
        target = [sum(poly[r] * _m(xi) ** r for r in range(d + 1)) for xi in x.to_fractions()]
    diffs = []
    for h in (Fraction(1, 16), Fraction(1, 32), Fraction(1, 64)):
        a = mhaskar.common_readout(polys[d], tay, d, float(h))
        m = (d + 1) // 2
        assert a.shape == (2 * m + 1,)
        net = C.TanhNet(mhaskar.common_slopes(F, float(h), m), pfloat.ones(2 * m + 1, F) * b0, a, pfloat.array(0, F))
        with mp.workprec(400):
            diffs.append(max(abs(_m(o) - t) for o, t in zip(net.forward(x).to_fractions(), target)))
    assert diffs[0] > diffs[1] > diffs[2] and diffs[1] / diffs[2] > 3.5


def test_staircase_halo_extrapolation_is_exact_for_quadratics():
    """The halo's quadratic Lagrange extrapolation reproduces a quadratic exactly (at 200 bits)."""
    F = C.fmt(WIDE)
    q = lambda t: 3 * t * t - 2 * t + Fraction(1, 7)  # noqa: E731
    ys = [pfloat.array(q(Fraction(k)), F) for k in range(3)]
    for i in range(1, 9):
        got = staircase._extrapolate(ys[0], ys[1], ys[2], i, F).to_fractions()[()]
        assert abs(Fraction(got) - q(Fraction(-i))) < Fraction(1, 2 ** 190)
