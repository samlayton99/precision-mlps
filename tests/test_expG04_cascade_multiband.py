import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG04_cascade_multiband"))

import solver          # expG03 numeric core
import cascade


def test_n_bands_1_reproduces_expg03_geometry():
    """A single-band cascade must equal expG03's uniform geometry, and a dense
    fit of sine must reach the fp64 floor."""
    c1, g1, b1 = cascade.cascade_geometry(128, [0.25], coarsen=2)
    c0, g0 = solver.geometry(128, 0.25)
    assert np.array_equal(c1, c0) and np.array_equal(g1, g0)
    assert set(np.unique(b1)) == {0}
    x = np.linspace(-1.0, 1.0, 400)
    f = lambda t: np.sin(2 * np.pi * t)
    v, bias, _ = solver.fit(x, f(x), c1, g1)
    xt = np.linspace(-1.0, 1.0, 257)
    assert solver.rel_l2(solver.predict(xt, c1, g1, v, bias), f(xt)) < 1e-12


def test_cascade_concatenation_and_band_index():
    """Three bands concatenate to the known per-band sizes with a matching
    band index; softer bands have fewer in-grid centers."""
    c, g, b = cascade.cascade_geometry(128, [0.25, 0.10, 0.05], coarsen=2)
    counts = [int((b == k).sum()) for k in range(3)]
    assert counts == [269, 205, 173]
    assert c.size == g.size == b.size == 647
    # per-band gamma is constant and strictly decreasing across bands
    gammas = [g[b == k][0] for k in range(3)]
    assert gammas[0] > gammas[1] > gammas[2]
    assert all(np.allclose(g[b == k], gammas[k]) for k in range(3))
