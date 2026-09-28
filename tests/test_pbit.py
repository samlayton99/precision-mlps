"""Tests of src/precision: the p-bit format arithmetic, the model, and the ported LAPACK DGELSS.

What each group establishes:
- primitives: every +, -, *, /, sqrt of the emulator is the correctly rounded result in the format
  (MPFR with IEEE subnormals), for p = 2..53, in the sweep's exponent range and in binary32;
  binary32/binary64 also match the hardware (numpy) and the native builds bit for bit.
- constants: the LAPACK machine parameters equal what gfortran computes from the reference source.
- model: tanh, features and readout equal an independent MPFR replay written from the spec alone;
  tanh is within 3 ulp at every p.
- solve: the ported DGELSS equals netlib reference LAPACK S/DGELSS (compiled from source)
  bit for bit, on tanh design matrices and on random, rank-deficient and graded matrices; the
  native builds equal the emulator.

Needs gmpy2 (``uv run --extra dev --extra precision python -m pytest tests/test_pbit.py``); the
reference-LAPACK tests also need gfortran and src/precision/_build/lapack-3.12.1 (see
src/precision/reference_lapack.py) and are skipped without them.
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

gmpy2 = pytest.importorskip("gmpy2", reason="install the 'precision' extra (gmpy2)")
sys.path.insert(0, str(Path(__file__).parent))
import pbit_replay_reference as REPLAY  # noqa: E402
from src.precision import pbit, reference_lapack  # noqa: E402

HAVE_REFERENCE = shutil.which("gfortran") is not None and reference_lapack.SOURCE.exists()
needs_reference = pytest.mark.skipif(not HAVE_REFERENCE, reason="gfortran or reference LAPACK source missing")
ALL_P = list(range(2, 54))


def target(x):
    return np.sin(8 * np.pi * (x + 1) ** 2)


# ---------------------------------------------------------------- primitives

def mpfr_context(fmt):
    """gmpy2 context equal to the IEEE-style format (p, emin, emax) with gradual underflow."""
    p, emin, emax = fmt
    return gmpy2.context(precision=p, emin=emin - p + 2, emax=emax + 1, subnormalize=True,
                         round=gmpy2.RoundToNearest)


def random_values(rng, fmt, count, lo, hi):
    """Random format values with exponents in [lo, hi] (subnormal quanta below emin)."""
    p, emin, _ = fmt
    e = rng.integers(lo, hi + 1, count)
    m = rng.integers(2 ** (p - 1), 2 ** p, count, dtype=np.int64) if p < 63 else None
    v = np.ldexp(m.astype(np.float64), e - p + 1) * rng.choice([-1.0, 1.0], count)
    return pbit.round_p(v, fmt)


def structured_cases(p):
    """Operand pairs whose exact results sit on or next to rounding midpoints."""
    ulp = 2.0 ** (1 - p)
    a, b = [], []
    for k in range(4):
        for j in (1, 2, 3):
            for s in (0, 1, 2, 3):
                a.append(1 + k * ulp)
                b.append(j * ulp / 2 ** (s + 1))
                a.append(1 + k * ulp)
                b.append(-j * ulp / 2 ** (s + 1))
    if p >= 3:
        for i in range(1, 6):
            for j in range(1, 6):
                a.append(1 + i * ulp)
                b.append(1 + j * ulp)
                a.append(1 - i * ulp / 2)
                b.append(1 + j * ulp)
    return np.array(a), np.array(b)


def expected(fmt, name, a, b):
    ops = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
           "div": lambda x, y: x / y, "sqrt": lambda x, y: gmpy2.sqrt(x)}
    out = []
    with mpfr_context(fmt):
        for x, y in zip(a, b):
            out.append(float(ops[name](gmpy2.mpfr(float(x)), gmpy2.mpfr(float(y)))))
    return np.array(out)


@pytest.mark.parametrize("p", ALL_P)
def test_primitives_match_mpfr_sweep_format(p):
    fmt = pbit.panel_format(p)
    rng = np.random.default_rng(p)
    a = random_values(rng, fmt, 1500, -8, 8)
    b = random_values(rng, fmt, 1500, -8, 8)
    sa, sb = structured_cases(p)
    sa, sb = pbit.round_p(sa, fmt), pbit.round_p(sb, fmt)
    # operands near and inside the subnormal range of the format
    ta = random_values(rng, fmt, 300, fmt[1] - p + 2, fmt[1] + 3)
    tb = random_values(rng, fmt, 300, -3, 3)
    A, B = np.concatenate([a, sa, ta]), np.concatenate([b, sb, tb])
    for name in ("add", "sub", "mul", "div", "sqrt"):
        x = np.abs(A) if name == "sqrt" else A
        np.testing.assert_array_equal(pbit.op(fmt, name, x, B), expected(fmt, name, x, B), err_msg=name)


def test_primitives_match_mpfr_binary32_with_subnormals():
    fmt = pbit.FP32
    rng = np.random.default_rng(32)
    a = np.concatenate([random_values(rng, fmt, 4000, -150, 20), random_values(rng, fmt, 2000, -3, 3)])
    b = np.concatenate([random_values(rng, fmt, 4000, -20, 20), random_values(rng, fmt, 2000, -130, -100)])
    for name in ("add", "sub", "mul", "div", "sqrt"):
        x = np.abs(a) if name == "sqrt" else a
        np.testing.assert_array_equal(pbit.op(fmt, name, x, b), expected(fmt, name, x, b), err_msg=name)


@pytest.mark.parametrize("fmt,dtype,backend", [(pbit.FP32, np.float32, "f32"), (pbit.FP64, np.float64, "f64")])
def test_primitives_match_hardware(fmt, dtype, backend):
    rng = np.random.default_rng(7)
    lo = -150 if dtype == np.float32 else -1070
    a = np.concatenate([rng.standard_normal(20000), np.ldexp(rng.standard_normal(2000), rng.integers(lo, 0, 2000))]).astype(dtype)
    b = np.concatenate([rng.standard_normal(20000), rng.standard_normal(2000)]).astype(dtype)
    hw = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "sqrt": np.sqrt(np.abs(a))}
    for name, ref in hw.items():
        x = np.abs(a) if name == "sqrt" else a
        x64, b64 = x.astype(np.float64), b.astype(np.float64)
        np.testing.assert_array_equal(pbit.op(fmt, name, x64, b64), ref.astype(np.float64), err_msg=name)
        np.testing.assert_array_equal(pbit.op(fmt, name, x64, b64, backend=backend), ref.astype(np.float64), err_msg=name)


def test_double_rounding_trap():
    # The exact product lies just below a p = 52 midpoint; rounding through binary64 first would
    # land on the midpoint and round up. The emulator must not.
    a, b = 1 + 2.0 ** -26, 1 + 2.0 ** -26 - 2.0 ** -51
    fmt = pbit.panel_format(52)
    np.testing.assert_array_equal(pbit.op(fmt, "mul", [a], [b]), expected(fmt, "mul", [a], [b]))
    assert pbit.op(fmt, "mul", [a], [b])[0] != float(pbit.round_p([a * b], fmt)[0])


# ---------------------------------------------------------------- constants

@needs_reference
def test_lapack_constants_match_gfortran():
    ref = reference_lapack.machine_constants()
    for name, fmt in (("fp64", pbit.FP64), ("fp32", pbit.FP32)):
        c = pbit.constants(fmt)
        for key in ("eps", "prec", "sfmin", "huge", "safmin", "safmax", "tsml", "tbig", "ssml", "sbig",
                    "epspow", "hndrth"):
            assert c[key] == ref[name][key], (name, key)


@pytest.mark.parametrize("p", ALL_P)
def test_constants_are_format_values_and_match_replay(p):
    fmt = pbit.panel_format(p)
    c = pbit.constants(fmt)
    values = [v for k, v in c.items() if k != "format" and k != "coeffs"] + c["coeffs"]
    np.testing.assert_array_equal(pbit.round_p(values, fmt), values)
    r = REPLAY.constants(p)
    for key in ("INVLN2", "LN2HI", "LN2LO", "SAT", "coeffs"):
        assert c[key] == r[key], key
    assert c["eps"] == 2.0 ** -p and c["prec"] == 2.0 ** (1 - p)


# ---------------------------------------------------------------- model

@pytest.mark.parametrize("p", [3, 8, 11, 17, 24, 31, 40, 47, 52, 53])
def test_tanh_matches_replay(p):
    fmt = pbit.panel_format(p)
    rng = np.random.default_rng(p)
    z = pbit.round_p(np.concatenate([rng.uniform(-14, 14, 400), rng.uniform(-1e-3, 1e-3, 50), [0.0]]), fmt)
    got = pbit.tanh(z, fmt)
    want = np.array([REPLAY.tanh_p(v, p) for v in z])
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("p", list(range(8, 54)))
def test_tanh_within_three_ulp(p):
    fmt = pbit.panel_format(p)
    rng = np.random.default_rng(1000 + p)
    z = np.unique(pbit.round_p(np.concatenate([rng.uniform(-12, 12, 600), np.exp2(rng.uniform(-30, 3, 200))]), fmt))
    t = pbit.tanh(z, fmt)
    worst = 0.0
    with gmpy2.context(precision=300):
        for zi, ti in zip(z, t):
            true = gmpy2.tanh(gmpy2.mpfr(float(zi)))
            if true == 0:
                assert ti == 0
                continue
            ulp = gmpy2.exp2(gmpy2.floor(gmpy2.log2(abs(true))) - p + 1)
            worst = max(worst, float(abs(gmpy2.mpfr(float(ti)) - true) / ulp))
    assert worst <= 3.0, worst


@pytest.mark.parametrize("p", [8, 13, 24, 37, 53])
def test_features_and_readout_match_replay(p):
    fmt = pbit.panel_format(p)
    N, H, lam = 10, 3, 0.4
    x, xe = np.linspace(-1, 1, 23), np.linspace(-1, 1, 17)
    phi = pbit.features(x, N, H, lam, fmt)
    np.testing.assert_array_equal(phi, REPLAY.features(p, pbit.round_p(x, fmt), N, H, pbit.round_p([lam], fmt)[0]))
    w = pbit.round_p(np.random.default_rng(p).standard_normal(N + 2 * H + 2), fmt)
    np.testing.assert_array_equal(pbit.evaluate(w, xe, N, H, lam, fmt),
                                  REPLAY.forward(p, pbit.round_p(xe, fmt), N, H, pbit.round_p([lam], fmt)[0], w))


# ---------------------------------------------------------------- solve

def _matrices(rng, fmt):
    """Test matrices for the solver: random, tall and nearly square (both DGELSS paths),
    rank-deficient, and graded columns (singular values over many decades)."""
    out = []
    for m, n in ((30, 7), (12, 10), (60, 25)):
        out.append(("random", rng.standard_normal((m, n))))
    u = rng.standard_normal((40, 4))
    out.append(("rank4", u @ rng.standard_normal((4, 12))))
    g = rng.standard_normal((50, 16)) * np.logspace(0, -12, 16)[None, :]
    out.append(("graded", g))
    return [(name, pbit.round_p(a.ravel(), fmt).reshape(a.shape)) for name, a in out]


@needs_reference
@pytest.mark.parametrize("fmt,dtype,backend", [(pbit.FP32, np.float32, "f32"), (pbit.FP64, np.float64, "f64")])
def test_gelss_port_matches_reference_on_matrices(fmt, dtype, backend):
    rng = np.random.default_rng(11)
    p = fmt[0]
    for name, a in _matrices(rng, fmt):
        b = pbit.round_p(rng.standard_normal(a.shape[0]), fmt)
        for rcond in (2.0 ** (1 - p), 2.0 ** (8 - p), 1e-3):
            rc = float(pbit.round_p([rcond], fmt)[0])
            ref = reference_lapack.gelss(a.astype(dtype), b.astype(dtype), rc, dtype)
            got = pbit.gelss(a, b, fmt, [rc])
            nat = pbit.gelss(a, b, fmt, [rc], backend=backend)
            assert ref["info"] == 0 and got["status"] == 0, name
            np.testing.assert_array_equal(got["sigma"], ref["sigma"], err_msg=name)
            np.testing.assert_array_equal(got["x"][0], ref["x"], err_msg=name)
            assert got["rank"][0] == ref["rank"], name
            np.testing.assert_array_equal(nat["x"], got["x"], err_msg=name)


@needs_reference
@pytest.mark.parametrize("fmt,dtype,lam", [(pbit.FP32, np.float32, 0.5), (pbit.FP64, np.float64, 0.3)])
@pytest.mark.parametrize("N,H,M", [(8, 2, 41), (40, 6, 301), (120, 12, 700)])
def test_model_and_solve_match_reference_lapack(fmt, dtype, lam, N, H, M):
    p = fmt[0]
    x, xe = np.linspace(-1, 1, M), np.linspace(-1, 1, 301)
    r = pbit.run(x, target(x), xe, N, H, lam, fmt)
    A = np.hstack([pbit.features(x, N, H, lam, fmt), np.ones((M, 1))])
    ref = reference_lapack.gelss(A.astype(dtype), pbit.round_p(target(x), fmt).astype(dtype), 2.0 ** (1 - p), dtype)
    np.testing.assert_array_equal(r["sigma"], ref["sigma"])
    np.testing.assert_array_equal(r["weights"], ref["x"])
    np.testing.assert_array_equal(r["fit"], pbit.evaluate(ref["x"], xe, N, H, lam, fmt))
    native = pbit.run(x, target(x), xe, N, H, lam, fmt, backend="f32" if p == 24 else "f64")
    np.testing.assert_array_equal(native["fit"], r["fit"])


def test_sweep_range_is_binary64_at_p53():
    x, xe = np.linspace(-1, 1, 301), np.linspace(-1, 1, 101)
    a = pbit.run(x, target(x), xe, 40, 6, 0.3, pbit.FP64)
    b = pbit.run(x, target(x), xe, 40, 6, 0.3, pbit.panel_format(53))
    np.testing.assert_array_equal(a["weights"], b["weights"])
    np.testing.assert_array_equal(a["fit"], b["fit"])


@pytest.mark.parametrize("p", [8, 16, 24, 40])
def test_run_outputs_are_format_values(p):
    fmt = pbit.panel_format(p)
    x, xe = np.linspace(-1, 1, 201), np.linspace(-1, 1, 77)
    r = pbit.run(x, target(x), xe, 30, 4, 0.6, fmt, rconds=[2.0 ** (1 - p), 2.0 ** (3 - p)])
    for arr in (r["weights_all"], r["fit_all"], r["sigma"]):
        np.testing.assert_array_equal(pbit.round_p(arr.ravel(), fmt), arr.ravel())
    assert r["events"]["nonfinite"] == 0 and r["events"]["overflow"] == 0
    assert np.all(np.diff(r["sigma"]) <= 0)


def test_fp64_solve_agrees_with_numpy_to_the_floor():
    """At p = 53 the port is a standard backward-stable solver: same singular values as numpy's
    SVD and the same fitted function to about the fp64 floor (not the same bits)."""
    smooth = lambda x: np.sin(np.pi * x)  # resolved at this width, so both solvers reach the floor
    x, xe = np.linspace(-1, 1, 801), np.linspace(-1, 1, 401)
    N, H, lam = 150, 12, 0.3
    r = pbit.run(x, smooth(x), xe, N, H, lam, pbit.FP64)
    A = np.hstack([pbit.features(x, N, H, lam, pbit.FP64), np.ones((x.size, 1))])
    np.testing.assert_allclose(r["sigma"], np.linalg.svd(A, compute_uv=False), rtol=0, atol=1e-12 * r["sigma"][0])
    w = np.linalg.lstsq(A, smooth(x), rcond=2.0 ** -52)[0]
    fit_np = np.hstack([pbit.features(xe, N, H, lam, pbit.FP64), np.ones((xe.size, 1))]) @ w
    assert np.max(np.abs(r["fit"] - smooth(xe))) < 1e-12
    assert np.max(np.abs(r["fit"] - fit_np)) < 1e-12


# ---------------------------------------------------------------- no leakage

from src.precision import audit as AUDIT  # noqa: E402


def test_static_audit_finds_only_the_documented_exception():
    """Outside the rounding layer the emulator build contains no floating +, -, *, /, no unrounded
    conversions, no math calls other than exact ones, and no raw literals, except LAPACK's integer
    crossover MNTHR = INT(REAL(MIN(M,N))*1.6E0)."""
    assert AUDIT.audit() == AUDIT.ALLOWED


@pytest.mark.parametrize("old,new,expect", [
    ("T dtemp = ADD(MUL(c, x[ix]), MUL(s, y[iy]));", "T dtemp = c * x[ix] + MUL(s, y[iy]);", "floating '*'"),
    ("T d = SQRT(ADD(MUL(f, f), MUL(g, g)));", "T d = sqrt(ADD(MUL(f, f), MUL(g, g)));", "call sqrt"),
    ("", "", None),  # control: an unchanged copy is clean
    ("sminoa = DIV(sminoa, SQRT(I2T(n)));", "sminoa = DIV(sminoa, SQRT((double)n));", "cast IntegralToFloating"),
])
def test_static_audit_catches_injected_leaks(tmp_path, old, new, expect):
    """Negative controls: a raw multiply, a libm call, or an unrounded conversion planted in the
    ported LAPACK code is reported."""
    for name in ("pbit_emul.c", "pbit_algo.h", "lapack_gelss.h"):
        (tmp_path / name).write_text((AUDIT.HERE / name).read_text())
    target_file = "lapack_gelss.h"
    text = (tmp_path / target_file).read_text()
    if expect is not None:
        assert old in text
        (tmp_path / target_file).write_text(text.replace(old, new, 1))
    extra = [f for f in AUDIT.audit(tmp_path) if f not in AUDIT.ALLOWED]
    if expect is None:
        assert extra == []
    else:
        assert any(expect in f["what"] for f in extra), extra


@pytest.mark.parametrize("p", [11, 24, 40, 52, 53])
def test_primitives_exact_for_tiny_results(p):
    """Products, quotients and roots of magnitude 2^-1100 .. 2^-850 (where an unscaled FMA residual
    could underflow) against MPFR, in the sweep format."""
    fmt = pbit.panel_format(p)
    rng = np.random.default_rng(500 + p)
    a = random_values(rng, fmt, 3000, -560, -420)
    b = random_values(rng, fmt, 3000, -560, -420)
    for name in ("mul", "div", "sqrt"):
        x = np.abs(a) if name == "sqrt" else a
        y = np.ldexp(b, 480) if name == "div" else b  # quotients land near 2^-1000 .. 2^-850
        y = pbit.round_p(y, fmt)
        np.testing.assert_array_equal(pbit.op(fmt, name, x, y), expected(fmt, name, x, y), err_msg=name)
