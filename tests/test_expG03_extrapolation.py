import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver


def test_no_holdout_sine_reaches_fp64_floor():
    """Dense full-grid fit of sine at lambda=0.25, N=128 hits the fp64 floor,
    matching expG01's 3.6e-14 sanity number. N sets the geometry (center
    spacing + halo); the number of training samples is an independent knob and
    must exceed the geometry's effective DOF (~N) to reach the floor."""
    N, lam = 128, 0.25
    centers, gamma_vec = solver.geometry(N, lam)
    x = np.linspace(-1.0, 1.0, 400)
    f = lambda t: np.sin(2 * np.pi * t)
    v, bias, info = solver.fit(x, f(x), centers, gamma_vec)
    x_test = np.linspace(-1.0, 1.0, 257)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    assert solver.rel_l2(u_hat, f(x_test)) < 1e-12


import protocols


def _min_gap(a, b):
    return float(np.min(np.abs(a[:, None] - b[None, :])))


def test_protocols_disjoint_and_holdout_coverage():
    # edge_holdout and beyond_domain: NO train point in the held-out region.
    for maker in (protocols.edge_holdout, protocols.beyond_domain):
        x_tr, x_te, regions = maker()
        assert _min_gap(x_tr, x_te) > 1e-9                       # disjoint grids
        assert protocols.in_regions(x_tr, regions).sum() == 0    # data-free hold-out

    # sparse_half: data-POOR, at most n_sparse train points in (0, 1].
    x_tr, x_te, regions = protocols.sparse_half(n_sparse=3)
    assert _min_gap(x_tr, x_te) > 1e-9
    assert protocols.in_regions(x_tr, regions).sum() <= 3


def test_in_regions_masks_correctly():
    # (lo, hi, include_lo, include_hi): inner boundary open, outer closed.
    x = np.array([-0.9, 0.0, 0.5, 0.6, 1.0, 1.2])
    m = protocols.in_regions(x, [(0.5, 1.0, False, True)])
    assert list(m) == [False, False, False, True, True, False]
