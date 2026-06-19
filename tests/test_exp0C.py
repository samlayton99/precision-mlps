"""Tests for exp0C GELU feature matrix and conditioning helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp0C_gelu_conditioning"))

import gelu_phi as gp


def test_gelu_known_values():
    assert gp.gelu(0.0) == pytest.approx(0.0)
    # gelu(1) = 0.5 * (1 + erf(1/sqrt2)) = 0.8413447...
    assert gp.gelu(1.0) == pytest.approx(0.84134475, abs=1e-7)


def test_gelu_asymptotes():
    # gelu(z) -> z for large positive z, -> 0 for large negative z
    assert gp.gelu(12.0) == pytest.approx(12.0, abs=1e-6)
    assert gp.gelu(-12.0) == pytest.approx(0.0, abs=1e-6)


def test_build_phi_gelu_shape_and_value():
    x = np.array([0.0, 0.5])
    centers = np.array([0.0, 1.0, -1.0])
    gamma = np.full(3, 2.0)
    Phi = gp.build_phi_gelu(x, gamma, centers)
    assert Phi.shape == (2, 3)
    # entry [0,0] = gelu(2*(0-0)) = gelu(0) = 0
    assert Phi[0, 0] == pytest.approx(0.0)


def test_conditioning_rank_deficient_matrix():
    # columns: c1, c2, c1+c2  -> rank 2, one null direction
    A = np.array([[1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 1.0, 2.0],
                  [2.0, 0.0, 2.0]])
    info = gp.conditioning(A)
    assert info["rank"] == 2
    assert info["null"] == 1
    assert np.isfinite(info["cond"])  # ratio of kept singular values is finite
