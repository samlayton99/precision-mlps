"""Tests for 1D center-placement strategies (expC04 geometry comparison)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.center_geometry import (
    _cluster_offsets,
    clustered_centers,
    random_centers,
    regular_clustered_centers,
    trained_centers,
    uniform_centers,
)

SPAN = (-1.8, 1.8)


@pytest.mark.parametrize("gen", [uniform_centers, random_centers, clustered_centers])
@pytest.mark.parametrize("W", [33, 128, 357])
def test_count_sorted_within_span(gen, W):
    c = (gen(W, SPAN, seed=0) if gen is not uniform_centers else gen(W, SPAN))
    assert c.shape == (W,)
    assert np.all(np.diff(c) >= 0)                 # sorted
    assert c.min() >= SPAN[0] - 1e-9 and c.max() <= SPAN[1] + 1e-9


def test_uniform_is_equispaced():
    c = uniform_centers(50, SPAN)
    assert np.allclose(np.diff(c), np.diff(c)[0])
    assert c[0] == pytest.approx(SPAN[0]) and c[-1] == pytest.approx(SPAN[1])


def test_random_reproducible_and_distinct_seeds():
    a = random_centers(200, SPAN, seed=1)
    b = random_centers(200, SPAN, seed=1)
    c = random_centers(200, SPAN, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_clustered_has_no_large_gaps():
    """Pseudo-clustered should still cover the span -- no super-large gaps."""
    W = 400
    c = clustered_centers(W, SPAN, seed=0)
    mean_spacing = (SPAN[1] - SPAN[0]) / W
    max_gap = np.diff(c).max()
    assert max_gap < 12 * mean_spacing, f"gap {max_gap:.3f} too large"


def test_clustered_more_clumped_than_uniform():
    """Clustered nearest-neighbour spacing varies more than uniform (it clumps)."""
    W = 400
    u = np.diff(uniform_centers(W, SPAN))
    cl = np.diff(clustered_centers(W, SPAN, seed=0))
    assert cl.std() > u.std()


def test_cluster_offsets_matches_spec_example():
    """The exact example from the spec: n=7, ratio=0.5, s=1."""
    off = _cluster_offsets(7, 0.5, 1.0)
    expected = np.array([-0.875, -0.375, -0.125, 0.0, 0.125, 0.375, 0.875])
    assert np.allclose(off, expected)


@pytest.mark.parametrize("W", [120, 200, 461])
def test_regular_clustered_count_and_span(W):
    c = regular_clustered_centers(W, SPAN, cluster_size=8, ratio=0.75)
    assert c.shape == (W,)
    assert np.all(np.diff(c) >= -1e-12)
    assert c.min() >= SPAN[0] - 1e-9 and c.max() <= SPAN[1] + 1e-9


def test_regular_clustered_ratio_one_is_uniform():
    """ratio=1 reproduces an (essentially) equispaced grid."""
    c = regular_clustered_centers(200, SPAN, cluster_size=8, ratio=1.0)
    d = np.diff(c)
    assert d.std() / d.mean() < 1e-6


def test_regular_clustered_clumps_more_as_ratio_drops():
    mild = np.diff(regular_clustered_centers(240, SPAN, 8, 0.9))
    tight = np.diff(regular_clustered_centers(240, SPAN, 8, 0.5))
    assert tight.std() > mild.std()           # smaller ratio -> more clumped


def test_trained_centers_formula():
    w = np.array([2.0, -4.0, 1.0])
    b = np.array([-1.0, 2.0, 0.5])
    c = trained_centers(w, b, SPAN)
    # x0 = -b/w = [0.5, 0.5, -0.5]; sorted -> [-0.5, 0.5, 0.5]
    assert np.allclose(c, np.sort(np.array([0.5, 0.5, -0.5])))


def test_trained_centers_handles_dead_neurons():
    w = np.array([1e-12, 3.0])            # first neuron is dead (|w| ~ 0)
    b = np.array([1.0, -3.0])
    c = trained_centers(w, b, SPAN)
    assert c.shape == (2,)
    assert c.min() >= SPAN[0] - 1e-9 and c.max() <= SPAN[1] + 1e-9
    assert np.isclose(c, 1.0).any()       # live neuron -> -b/w = 1.0 (in span)
    assert np.isclose(c, SPAN[0]).any()   # dead neuron -> parked at lo (b >= 0)
