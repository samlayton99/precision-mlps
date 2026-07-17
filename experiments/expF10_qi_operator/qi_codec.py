"""Fixed QI ridge encoder/decoder (numpy) for expF10.

encode(a_grid) = pinv(basis(grid)) @ a  -- a fixed linear analysis transform.
decode(c, P)   = basis(P) @ c           -- synthesis at any points/resolution.
basis(P) = rows_2d(P, ..., [((0,0),1.0)]) -- ridge values + degree-3 poly tail.
Pseudo-inverses are cached per grid resolution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))

import core  # expF08 scalar primitives


class QICodec:
    def __init__(self, W=576, lam=0.25, rcond=1e-10):
        self.dirs, self.offs, self.gamma = core.radon_geometry(W, lam)
        self.D = len(self.offs) + len(core.MONO_2D)
        self.rcond = rcond
        self._pinv = {}

    def grid(self, n):
        g = np.linspace(-1.0, 1.0, n)
        X, Y = np.meshgrid(g, g, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def basis(self, P):
        return core.rows_2d(np.asarray(P, float), self.dirs, self.offs,
                            self.gamma, [((0, 0), 1.0)])

    def pinv(self, n):
        if n not in self._pinv:
            self._pinv[n] = np.linalg.pinv(self.basis(self.grid(n)),
                                           rcond=self.rcond)
        return self._pinv[n]

    def encode(self, a_grid, n):
        """a_grid: [n*n] flattened (row-major, indexing='ij') -> coeffs [D]."""
        return self.pinv(n) @ np.asarray(a_grid, float).ravel()

    def decode(self, c, P):
        return self.basis(P) @ np.asarray(c, float)

    @staticmethod
    def rel_l2(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                     / np.linalg.norm(np.asarray(b)))
