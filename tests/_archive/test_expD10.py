"""expD10 -- tests that the hardened step-2 part matches the research claims.

Each test pins one claim from the expD10 writeup:
  1. compensated (dd) primitives agree with the reference implementation
  2. dd triangular solve is exact to fp64 on a well-conditioned system
  3. finalist solve (whiten k=128 + LSMR + Fong-Saunders stop) reaches the
     floor on a clean cell, with NO oracle
  4. seriation recovers a randomly destroyed column order end to end
  5. implicit compensated whitening matches materialized B; fp64-implicit
     control fails by orders (recipe detail 2)
  6. the same pipeline at fp32 lands near the fp32 floor (precision-agnostic)
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.sparse.linalg import lsmr

HERE = Path(__file__).resolve().parent
E10 = HERE.parent / "experiments" / "expD10_step2_hardening"
sys.path.insert(0, str(E10))

import core10 as c   # noqa: E402
import dd            # noqa: E402


@pytest.fixture(scope="module")
def prob():
    return c.cv.make_problem("sine", 64)


def test_dd_primitives_match_reference():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((37, 23))
    X = (rng.standard_normal(23), 1e-18 * rng.standard_normal(23))
    ref = dd.matvec(M, X)
    fast = c.dd_mat_ddvec(M, X)
    assert np.abs(ref[0] - fast[0]).max() == 0.0
    assert np.abs(ref[1] - fast[1]).max() < 1e-28


def test_dd_trisolve_exact():
    rng = np.random.default_rng(1)
    R = np.triu(rng.standard_normal((30, 30))) + 8 * np.eye(30)
    b = rng.standard_normal(30)
    xh, xl = c.dd_trisolve_upper(R, b)
    assert np.abs(xh + xl - sla.solve_triangular(R, b)).max() < 1e-14


def test_finalist_reaches_floor_no_oracle(prob):
    A, y = prob["A"], prob["y"]
    pieces, B = c.whiten(A, 128)
    res = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=3000)
    w = c.unwhiten(pieces, res[0], A.shape[1])
    # floor_svd for sine N=64 is ~7e-15; stopped lands within a small factor
    assert c.eval_err(prob, w) < 5e-14


def test_seriation_recovers_destroyed_order(prob):
    rng = np.random.default_rng(7)
    A, y = prob["A"], prob["y"]
    d = A.shape[1]
    perm = rng.permutation(d)
    Ap = A[:, perm]
    order = c.seriate(Ap[rng.choice(A.shape[0], 256, replace=False)])
    pieces, B = c.whiten(Ap, 128, order=order)
    res = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=3000)
    w = np.zeros(d)
    w[perm] = c.unwhiten(pieces, res[0], d)
    assert c.eval_err(prob, w) < 1e-11      # permuted-naive is ~1e-12 at N=64;
    # the discriminating cell is N=128 (1.4e-8 naive) -- covered in evidence.json;
    # here we pin that seriation does not DEGRADE the easy cell
    order_naive = np.arange(d)
    pieces2, B2 = c.whiten(Ap, 128, order=order_naive)
    res2 = lsmr(B2, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=3000)
    w2 = np.zeros(d)
    w2[perm] = c.unwhiten(pieces2, res2[0], d)
    assert c.eval_err(prob, w) <= 10 * max(c.eval_err(prob, w2), 1e-15)


def test_implicit_dd_parity_and_fp64_control(prob):
    A, y = prob["A"], prob["y"]
    d = A.shape[1]
    pieces, B = c.whiten(A, 128)
    e_mat = c.eval_err(prob, c.unwhiten(
        pieces, lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=2000)[0], d))
    opdd = c.implicit_dd_operator(A, pieces)
    e_dd = c.eval_err(prob, c.unwhiten(
        pieces, lsmr(opdd, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=2000)[0], d))
    assert e_dd < 50 * max(e_mat, 1e-15)    # parity up to stopping variance

    # fp64-implicit control: injects eps*kappa(R) per matvec, fails by orders
    from run_evidence import fp64_implicit
    e_64 = c.eval_err(prob, c.unwhiten(
        pieces, lsmr(fp64_implicit(A, pieces), y, atol=1e-15, btol=1e-15,
                     conlim=0, maxiter=2000)[0], d))
    assert e_64 > 1e3 * e_dd


def test_precision_agnostic_fp32(prob):
    A = prob["A"].astype(np.float32)
    y = prob["y"].astype(np.float32)
    d = A.shape[1]
    pieces = []
    B = np.zeros(A.shape, dtype=np.float32)
    for t in range(0, d, 128):
        C = np.arange(t, min(t + 128, d))
        Q, R, piv = sla.qr(A[:, C], mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        nk = int((dg > 1e-6 * max(dg[0], 1e-30)).sum())
        B[:, C[:nk]] = Q[:, :nk]
        pieces.append((C, piv, R[:nk, :], nk))
    res = lsmr(B, y, atol=1e-7, btol=1e-7, conlim=0, maxiter=3000)
    z = res[0].astype(np.float32)
    w = np.zeros(d, dtype=np.float32)
    for C, piv, R, nk in pieces:
        if nk == 0:
            continue
        cc = sla.solve_triangular(R[:, :nk], z[C[:nk]], lower=False)
        out = np.zeros(len(C), dtype=np.float32)
        out[piv[:nk]] = cc
        w[C] = out
    # fp32 floor on this cell is ~2.5e-7; land within a small factor of it
    assert c.eval_err(prob, w.astype(np.float64)) < 5e-6


def test_agglomerative_blocking_recovers_permuted(prob):
    """The shipped blocking (cluster_blocks) recovers a destroyed order and
    the kappa gate separates it from a structure-free matrix."""
    rng = np.random.default_rng(11)
    A, y = prob["A"], prob["y"]
    d = A.shape[1]
    perm = rng.permutation(d)
    Ap = A[:, perm]
    sr = rng.choice(A.shape[0], 256, replace=False)
    blocks = c.cluster_blocks(Ap[sr], kmax=128)
    pieces, B = c.whiten(Ap, 128, blocks=blocks)
    res = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=5000)
    w = np.zeros(d)
    w[perm] = c.unwhiten(pieces, res[0], d)
    assert c.eval_err(prob, w) < 1e-12

    # kappa gate: structured (QI) vs structure-free spectral twin
    kap_qi = c.kappa_gate(Ap[sr], blocks)
    rng2 = np.random.default_rng(3)
    r = int(0.6 * d)
    U = np.linalg.qr(rng2.standard_normal((256, r)))[0]
    V = np.linalg.qr(rng2.standard_normal((d, r)))[0]
    Atw = (U * np.logspace(0, -14, r)) @ V.T
    blocks_tw = c.cluster_blocks(Atw, kmax=128)
    kap_tw = c.kappa_gate(Atw, blocks_tw)
    assert kap_tw > 1e2 * kap_qi
