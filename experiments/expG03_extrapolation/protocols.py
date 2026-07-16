"""Hold-out protocols for expG03.

Each protocol returns (x_train, x_test, holdout_regions):
  x_train, x_test  1D float64 grids, guaranteed disjoint;
  holdout_regions  list of (lo, hi, include_lo, include_hi) intervals scored as
                   "held-out". The inner (training-side) boundary is open and
                   the outer boundary is closed, so training endpoints that sit
                   exactly on a boundary are NOT counted as held-out while the
                   extreme test points ARE.

N (geometry resolution) is separate from the training-sample count: the counts
below are the number of collocation samples, chosen to exceed the geometry's
effective DOF (~N) so the unmasked region reaches the fp64 floor.
"""
from __future__ import annotations

import numpy as np


def in_regions(x, regions):
    """Boolean mask: True where x falls in any (lo, hi, include_lo, include_hi)."""
    x = np.asarray(x, float)
    m = np.zeros(x.shape, dtype=bool)
    for lo, hi, inc_lo, inc_hi in regions:
        left = x >= lo if inc_lo else x > lo
        right = x <= hi if inc_hi else x < hi
        m |= left & right
    return m


def _disjoint_grid(lo, hi, n, x_train, tol=1e-9):
    """Equispaced n points on [lo, hi] with any point coinciding with a train
    point removed, so x_train and the result are disjoint."""
    x = np.linspace(lo, hi, n)
    keep = np.min(np.abs(x[:, None] - np.asarray(x_train)[None, :]), axis=1) > tol
    return x[keep]


def edge_holdout(c=0.5, n_train=300, n_test=400):
    """Train densely on [-1, c]; held-out (c, 1] (one-sided extrapolation)."""
    x_train = np.linspace(-1.0, c, n_train)
    x_test = _disjoint_grid(-1.0, 1.0, n_test, x_train)
    return x_train, x_test, [(c, 1.0, False, True)]


def beyond_domain(delta=0.3, n_train=400, n_test=400):
    """Train densely on [-1, 1]; held-out [-1-delta, -1) and (1, 1+delta] (past
    the last neuron)."""
    x_train = np.linspace(-1.0, 1.0, n_train)
    x_test = _disjoint_grid(-1.0 - delta, 1.0 + delta, n_test, x_train)
    return x_train, x_test, [(-1.0 - delta, -1.0, True, False),
                             (1.0, 1.0 + delta, False, True)]


def sparse_half(n_dense=250, n_sparse=3, n_test=400):
    """Dense on [-1, 0] plus n_sparse interior points on (0, 1); held-out
    (0, 1] scored as the data-poor half (data-poor, NOT data-free)."""
    x_dense = np.linspace(-1.0, 0.0, n_dense)
    x_sparse = np.linspace(0.0, 1.0, n_sparse + 2)[1:-1]
    x_train = np.concatenate([x_dense, x_sparse])
    x_test = _disjoint_grid(-1.0, 1.0, n_test, x_train)
    return x_train, x_test, [(0.0, 1.0, False, True)]


PROTOCOLS = {
    "edge_holdout": edge_holdout,
    "beyond_domain": beyond_domain,
    "sparse_half": sparse_half,
}
