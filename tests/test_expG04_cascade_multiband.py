import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Append (never insert-0): these dirs hold generically-named modules (run/viz)
# shared with sibling experiments; keeping them off sys.path[0] avoids shadowing
# another test's bare imports in a shared pytest process. solver/protocols are
# unique to expG03, cascade to expG04; run/viz are loaded by explicit path below.
sys.path.append(str(REPO_ROOT / "experiments" / "expG03_extrapolation"))
sys.path.append(str(REPO_ROOT / "experiments" / "expG04_cascade_multiband"))

import importlib.util

import solver          # expG03 numeric core
import cascade


def _load_g04(name, relpath):
    """Load an expG04 module by explicit path under a unique name, so the
    generic `run`/`viz` names do not collide with expG03's modules in a shared
    pytest process."""
    path = REPO_ROOT / "experiments" / "expG04_cascade_multiband" / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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


g04run = _load_g04("g04_run", "run.py")


def test_evaluate_cell_precision_preserved_and_per_band():
    """3-band cascade keeps the trained region near the floor, splits ||v|| by
    band, and the held-out region is finite and no easier than unmasked."""
    rec = g04run.evaluate_cell(3, "edge_holdout", "sine")
    assert rec["rel_l2_unmasked"] < 1e-8            # precision preserved
    assert np.isfinite(rec["rel_l2_held"])
    assert rec["rel_l2_held"] >= rec["rel_l2_unmasked"]
    assert sorted(rec["per_band_norm"]) == [0, 1, 2]
    assert np.isclose(
        np.sqrt(sum(n**2 for n in rec["per_band_norm"].values())),
        rec["coeff_norm"], rtol=1e-9)


def test_evaluate_cell_basis_sum_identity():
    """With a cascade geometry, sum_k c_k*phi_k + bias == predict(...)."""
    c, g, _ = g04run.C.cascade_geometry(128, g04run.LAMBDAS, g04run.COARSEN)
    x = np.linspace(-1.0, 1.0, 300)
    y = 1.0 / (1.0 + 25.0 * x**2)
    v, bias, _ = solver.fit(x, y, c, g)
    xd = np.linspace(-1.3, 1.3, 411)
    contrib, b = solver.basis_contributions(xd, c, g, v, bias)
    assert np.max(np.abs(contrib.sum(axis=1) + b
                         - solver.predict(xd, c, g, v, bias))) < 1e-9


def test_viz_writes_figures(tmp_path):
    viz = _load_g04("g04_viz", "viz.py")
    import protocols
    recs = [g04run.evaluate_cell(3, "edge_holdout", "runge"),
            g04run.evaluate_cell(1, "edge_holdout", "runge")]
    viz.make_all_figures(recs, tmp_path, g04run.TARGETS, protocols, cascade,
                         solver, 128, g04run.LAMBDAS, g04run.COARSEN)
    assert (tmp_path / "summary_held_vs_nbands.png").exists()
    assert any(tmp_path.glob("basis_nb3_edge_holdout_runge.png"))
    assert any(tmp_path.glob("fit_nb3_edge_holdout_runge.png"))
