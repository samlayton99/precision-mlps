"""expD19 gate: the static diagnosis the training study is built on.

Pins the feature-matrix pathology (GELU's column norms spanning ~300 orders
because the right halo underflows while the left halo grows like gamma), the
fact that lstsq is nonetheless scale-robust, and the halo-sizing result that
holds for BOTH activations.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SP = _load("expD19_static_probe",
           REPO_ROOT / "experiments" / "expD19_gelu_init" / "static_probe.py")

N = 128
X = np.linspace(-1, 1, 2003)
XE = np.linspace(-1, 1, 4001)
F = lambda t: np.sin(np.pi * t)
RFULL = max(59, int(np.ceil(0.4 * N)))


def _v(act, hl, hr, colnorm=False, linear=False):
    return SP.variant(act, N, hl, hr, colnorm, linear, X, XE, F)


def test_gelu_geometry_reaches_the_floor_despite_the_scale_spread():
    """The geometry is not the problem: lstsq is scale-robust and gets there."""
    v = _v("gelu", RFULL, RFULL)
    assert v["floor"] < 1e-12, v["floor"]
    # ... while the column norms span hundreds of orders (right halo underflows)
    assert v["colnorm_ratio"] > 1e100, v["colnorm_ratio"]


def test_gelu_column_norms_are_ordered_left_halo_gg_interior_gg_right_halo():
    """The pathology itself: gelu(z)->z inflates the left halo, gelu(z)->0
    underflows the right one. RMS norms, so they are comparable to the study."""
    v = _v("gelu", RFULL, RFULL)
    n = np.sqrt(len(X))
    interior, left, right = SP.region_means(v["centers"], v["col_norms"] / n)
    assert left > interior > 1.0 > right
    assert left / interior > 2.0
    assert right < 1e-1


def test_tanh_column_norms_are_flat():
    """The contrast: a bounded activation gives an automatically scaled Phi."""
    v = _v("tanh", RFULL, RFULL)
    n = np.sqrt(len(X))
    interior, left, right = SP.region_means(v["centers"], v["col_norms"] / n)
    assert max(interior, left, right) / min(interior, left, right) < 1.2


def test_column_normalization_improves_the_gelu_floor():
    base = _v("gelu", RFULL, RFULL)["floor"]
    fixed = _v("gelu", RFULL, RFULL, colnorm=True, linear=True)["floor"]
    assert fixed < base / 3, (base, fixed)


def test_small_halo_matches_the_full_halo_at_40_percent_fewer_neurons():
    full = _v("gelu", RFULL, RFULL, colnorm=True, linear=True)
    small = _v("gelu", 8, 8, colnorm=True, linear=True)
    assert small["W"] < 0.65 * full["W"]
    assert small["floor"] < 3 * full["floor"]


def test_both_halo_sides_are_required_under_gelu():
    """Naive reading of the asymmetry ('the right halo is dead, drop it') is
    wrong: either side alone loses 3+ orders."""
    both = _v("gelu", 8, 8)["floor"]
    for hl, hr in [(8, 0), (0, 8), (0, 0)]:
        assert _v("gelu", hl, hr)["floor"] > 100 * both, (hl, hr)


def test_the_halo_rule_is_oversized_for_tanh_too():
    """R ~ 0.4N is not needed: halo 8 beats halo 59 with 41% fewer neurons."""
    full = _v("tanh", RFULL, RFULL)
    small = _v("tanh", 8, 8)
    assert small["W"] < 0.65 * full["W"]
    assert small["floor"] < full["floor"]


def test_no_halo_fails_for_both_activations():
    """The halo is load-bearing; this is what small-but-nonzero is trading against."""
    for act in ("tanh", "gelu"):
        assert _v(act, 0, 0)["floor"] > 1e-8, act
