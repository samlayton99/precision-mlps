"""Tests for exp06 frequency-ladder helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp06_lambda_vs_frequency"))

import sweep_utils as su


def test_sin_target_values():
    fn, deriv = su.sin_target(2)
    assert fn(0.25) == pytest.approx(1.0)        # sin(2*pi*0.25) = sin(pi/2) = 1
    assert deriv(0.0) == pytest.approx(2 * np.pi)  # 2*pi*cos(0) = 2*pi


def test_sin_target_derivative_matches_finite_difference():
    fn, deriv = su.sin_target(3)
    x0, e = 0.3, 1e-6
    fd = (fn(x0 + e) - fn(x0 - e)) / (2 * e)
    assert deriv(x0) == pytest.approx(fd, rel=1e-5)


def test_best_lambda_picks_argmin():
    lam, err = su.best_lambda([0.1, 0.2, 0.3], [1e-3, 1e-7, 1e-5])
    assert lam == pytest.approx(0.2)
    assert err == pytest.approx(1e-7)


def test_best_lambda_ignores_nan():
    lam, err = su.best_lambda([0.1, 0.2, 0.3], [np.nan, 1e-4, np.nan])
    assert lam == pytest.approx(0.2)


def test_best_lambda_all_nan_returns_nan():
    lam, err = su.best_lambda([0.1, 0.2], [np.nan, np.nan])
    assert np.isnan(lam) and np.isnan(err)
