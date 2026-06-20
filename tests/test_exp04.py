"""Tests for exp04 coefficient-comparison metrics.

The heart of exp04: on a FIXED tanh geometry, the QI and lstsq readouts solve
the same Phi @ beta = y, but Phi is ~50% null space. So the two coefficient
vectors can differ wildly in the null space while computing the same function.
These tests pin the three metrics that disentangle that:

  raw_diff       -- distance in the full coefficient space (sees null space)
  rowspace_diff  -- distance projected onto Phi's row space (data-visible part)
  function_diff  -- distance of the functions they compute (the honest measure)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp04_coeff_nullspace"))

import coeff_compare as cc


def test_raw_diff_matches_hand_computation():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 0.0, -1.0])  # diff = [0, 2, 4]
    out = cc.raw_diff(a, b)
    assert out["l2"] == pytest.approx(np.sqrt(0 + 4 + 16))
    assert out["linf"] == pytest.approx(4.0)
    assert out["rel_l2"] == pytest.approx(np.sqrt(20) / np.linalg.norm(a))


def test_rowspace_diff_kills_nullspace_component():
    # A has rank 1: row space spanned by [1,1,1]/sqrt(3); null space is 2-dim.
    A = np.ones((4, 3))
    # A delta purely in the null space (orthogonal to [1,1,1]) is invisible.
    null_delta = np.array([1.0, -1.0, 0.0])  # sums to 0 -> in null space of A
    qi = np.array([5.0, 5.0, 5.0])
    ls = qi - null_delta
    out = cc.rowspace_diff(A, qi, ls)
    assert out["l2"] == pytest.approx(0.0, abs=1e-12)
    assert out["linf"] == pytest.approx(0.0, abs=1e-12)
    assert out["rank"] == 1


def test_rowspace_diff_keeps_rowspace_component():
    # A delta entirely in the row space is fully retained.
    A = np.ones((4, 3))
    row_delta = np.array([2.0, 2.0, 2.0])  # parallel to row space
    qi = np.array([5.0, 5.0, 5.0])
    ls = qi - row_delta
    out = cc.rowspace_diff(A, qi, ls)
    assert out["l2"] == pytest.approx(np.linalg.norm(row_delta))


def test_function_diff_equals_design_times_delta():
    rng = np.random.default_rng(0)
    A_eval = rng.standard_normal((10, 3))
    qi = rng.standard_normal(3)
    ls = rng.standard_normal(3)
    y = rng.standard_normal(10)
    out = cc.function_diff(A_eval, qi, ls, y_eval=y)
    expected = A_eval @ (qi - ls)
    assert out["linf"] == pytest.approx(np.max(np.abs(expected)))
    assert out["l2"] == pytest.approx(np.linalg.norm(expected))
    assert out["rel_l2"] == pytest.approx(np.linalg.norm(expected) / np.linalg.norm(y))


def test_fit_residual_zero_for_exact_solution():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 3))
    beta = rng.standard_normal(3)
    y = A @ beta  # beta fits exactly
    assert cc.fit_residual(A, beta, y) == pytest.approx(0.0, abs=1e-12)


def test_deviation_ratio_matches_hand_computation():
    a = np.array([1.0, 2.0, 3.0])   # mean|a| = 2
    b = np.array([1.0, 0.0, -1.0])  # mean|b| = 2/3, diff = [0, 2, 4], linf = 4
    # ratio = linf(diff) / max(mean|a|, mean|b|) = 4 / 2 = 2
    assert cc.deviation_ratio(a, b) == pytest.approx(2.0)


def test_project_rowspace_removes_nullspace_keeps_rowspace():
    A = np.ones((4, 3))  # row space = span([1,1,1]); rank 1
    row_vec, rank = cc.project_rowspace(A, np.array([2.0, 2.0, 2.0]))
    null_vec, _ = cc.project_rowspace(A, np.array([1.0, -1.0, 0.0]))
    assert rank == 1
    assert row_vec == pytest.approx([2.0, 2.0, 2.0])          # kept
    assert null_vec == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)  # removed


def test_project_rowspace_idempotent():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((8, 4))
    v = rng.standard_normal(4)
    Pv, _ = cc.project_rowspace(A, v)
    PPv, _ = cc.project_rowspace(A, Pv)
    assert PPv == pytest.approx(Pv)


def test_rowspace_ratio_zero_when_difference_is_pure_nullspace():
    # Two vectors that differ only in the null space compute the same function,
    # so their row-space projections coincide -> row-space ratio ~ 0, even though
    # the full-space ratio is not.
    A = np.ones((4, 3))
    beta_ls = np.array([2.0, 2.0, 2.0])          # already in row space
    beta_qi = beta_ls + np.array([1.0, -1.0, 0.0])  # + a null-space component
    out = cc.rowspace_ratio(A, beta_qi, beta_ls)
    assert out["ratio"] == pytest.approx(0.0, abs=1e-12)
    assert out["rank"] == 1
    assert cc.deviation_ratio(beta_qi, beta_ls) > 0.1  # full-space ratio is large
