"""darcy_421 benchmark loading + spline coefficient surrogate.

Benchmark convention: -div_s(a grad_s u) = 1 on [0,1]^2, u = 0 on the boundary.
On p = 2s - 1: -a lap_p u - grad_p a . grad_p u = 1/4, so DARCY_FORCING = 0.25.
darcy_421 stores CELL-CENTERED grids (node k at (k+1/2)h).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

DARCY_FORCING = 0.25
DEFAULT_NPZ = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_test_421_jax.npz"


def grid_1d(n, cell_centered=False):
    if cell_centered:
        return -1.0 + (2.0 * np.arange(n) + 1.0) / n
    return np.linspace(-1.0, 1.0, n)


class DarcyCoefficient:
    """Cubic-spline surrogate of a gridded coefficient on [-1,1]^2."""

    def __init__(self, a_grid, sigma_px=0.0, cell_centered=False):
        a = np.asarray(a_grid, dtype=np.float64)
        if sigma_px > 0:
            a = gaussian_filter(a, sigma_px, mode="nearest")
        n0, n1 = a.shape
        self._sp = RectBivariateSpline(grid_1d(n0, cell_centered),
                                       grid_1d(n1, cell_centered), a, kx=3, ky=3)

    def a(self, P):
        return self._sp.ev(P[:, 0], P[:, 1])

    def ax(self, P):
        return self._sp.ev(P[:, 0], P[:, 1], dx=1)

    def ay(self, P):
        return self._sp.ev(P[:, 0], P[:, 1], dy=1)

    def terms(self):
        return [((2, 0), lambda P: -self.a(P)),
                ((0, 2), lambda P: -self.a(P)),
                ((1, 0), lambda P: -self.ax(P)),
                ((0, 1), lambda P: -self.ay(P))]


def load_darcy_test(path=DEFAULT_NPZ, n_instances=16):
    d = np.load(path)
    keys = set(d.keys())
    if {"x", "y"} <= keys:
        a, u = d["x"], d["y"]
    elif {"a", "u"} <= keys:
        a, u = d["a"], d["u"]
    else:
        raise KeyError(f"unrecognized darcy npz keys: {sorted(keys)}")
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    a = a.reshape(a.shape[0], a.shape[-2], a.shape[-1])
    u = u.reshape(u.shape[0], u.shape[-2], u.shape[-1])
    return a[:n_instances], u[:n_instances]


def eval_points_and_ref(u_grid, stride=3):
    """Cell-centered eval points and reference values, subsampled by stride."""
    n = u_grid.shape[0]
    g = grid_1d(n, cell_centered=True)[::stride]
    P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return P, u_grid[::stride, ::stride].ravel()
