"""expG04 cascade multi-band geometry.

Stacks single-band expG03 geometries at decreasing bandwidth and increasing
center spacing into one (centers, gamma_vec) pair, plus a band index. The
concatenated geometry is a drop-in for expG03's solver.{fit,predict,
basis_contributions}, so a cascade is solved by the same stacked SVD readout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver  # expG03 numeric core


def cascade_geometry(N, lambdas, coarsen=2):
    """Multi-band geometry: band k is expG03's uniform geometry on grid
    N // coarsen**k at bandwidth lambdas[k] (sharp band = full N; softer bands
    coarser). Returns (centers, gamma_vec, band_idx), all length = total width.
    """
    centers_list, gamma_list, band_list = [], [], []
    for k, lam in enumerate(lambdas):
        Nk = max(4, N // coarsen**k)
        c, g = solver.geometry(Nk, lam)
        centers_list.append(c)
        gamma_list.append(g)
        band_list.append(np.full(c.size, k, dtype=int))
    return (np.concatenate(centers_list),
            np.concatenate(gamma_list),
            np.concatenate(band_list))
