"""expG03 numeric core: fixed uniform-gamma construction + SVD readout.

Mirrors expG01's geometry() (app.py:103) and the src.construction solver path,
as a small importable core for the batch experiment. fp64 throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi, solve_readout_with_bias

HALO_LAMBDA = 0.25  # reference lambda for halo sizing (as in expG01/expC04)


def geometry(N, lam, halo=None):
    """Center lattice on [-1,1] plus a halo of ghost nodes per side, and a
    per-center gamma = lam/h (h = 2/N). Returns (centers, gamma_vec)."""
    h = 2.0 / N
    if halo is None:
        halo = default_halo(N, lambda_star=HALO_LAMBDA)
    halo = int(halo)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h
    gamma_vec = np.full(centers.size, lam / h)
    return centers, gamma_vec


def fit(x_train, y_train, centers, gamma_vec):
    """Truncated-SVD min-norm readout with bias on the training points.
    Returns (v, bias, info)."""
    Phi = build_phi(x_train, gamma_vec, centers)
    return solve_readout_with_bias(Phi, np.asarray(y_train, float),
                                   method="svd")


def predict(x, centers, gamma_vec, v, bias):
    return build_phi(x, gamma_vec, centers) @ v + bias


def basis_contributions(x, centers, gamma_vec, v, bias):
    """Per-center weighted ridges c_k*phi_k(x) as columns [n_x, width] and the
    scalar bias. contributions.sum(axis=1) + bias == predict(...)."""
    Phi = build_phi(x, gamma_vec, centers)
    return Phi * np.asarray(v, float)[None, :], float(bias)


def rel_l2(u_hat, u_true):
    u_hat, u_true = np.asarray(u_hat), np.asarray(u_true)
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


def linf(u_hat, u_true):
    return float(np.max(np.abs(np.asarray(u_hat) - np.asarray(u_true))))
