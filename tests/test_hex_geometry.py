"""Tests for the 2D hexagonal tangent-line geometry.

Verifies the implementation matches the agreed math:
1. Packing: exact count N, all within radius, min-spacing ~ spacing_d, and the
   centered-hex numbers reproduced for full-ring widths.
2. Tangent-line correctness: W_m . c_m + beta_m = 0 (transition passes through
   the grid point), W_m parallel to c_m (radial normal), ||W_m|| = gamma.
3. Bandwidth duality: lambda_star -> gamma = lambda/d, and the gamma path
   round-trips lambda = gamma*d.
4. build_phi_2d matches the explicit tanh(gamma*(n.x - r)) formula.
5. Regime sanity: the plane-wave control fits to ~eps; a smooth target's lstsq
   residual decreases as N grows.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.hex_geometry import (
    build_hex_geometry,
    build_phi_2d,
    centered_hex_number,
    hex_pack,
    tangent_geometry,
)
from src.construction.readout import solve_readout_with_bias
from src.data.sampling2d import disk_grid, disk_uniform
from src.data.targets2d import get_target_2d

HEX_NUMBERS = [centered_hex_number(K) for K in range(1, 7)]  # 7,19,37,61,91,127
RADIUS = 1.4


# --- 1. packing ---------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 7, 19, 37, 61, 50, 100, 271])
def test_pack_exact_count_and_within_radius(N):
    centers, radii, d = hex_pack(N, RADIUS)
    assert centers.shape == (N, 2)
    assert radii.shape == (N,)
    assert np.all(radii <= RADIUS + 1e-9), f"max radius {radii.max()} > {RADIUS}"
    if N > 1:
        assert d > 0


@pytest.mark.parametrize("N", HEX_NUMBERS)
def test_full_ring_spacing_reaches_radius(N):
    """For complete-ring widths, the outer ring corners sit at `radius`."""
    centers, radii, d = hex_pack(N, RADIUS)
    assert radii.max() == pytest.approx(RADIUS, rel=1e-9)


@pytest.mark.parametrize("N", [7, 19, 37, 61, 100])
def test_min_pairwise_distance_matches_spacing(N):
    """Hex packing is compact: nearest-neighbour distance ~ spacing_d."""
    centers, radii, d = hex_pack(N, RADIUS)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist[np.arange(N), np.arange(N)] = np.inf
    min_dist = dist.min()
    # No two points closer than the lattice spacing (allow tiny slack); and the
    # closest pair is on the order of the spacing (not collapsed).
    assert min_dist >= d - 1e-9
    assert min_dist <= d + 1e-9 or min_dist <= 1.001 * d


def test_centered_hex_numbers():
    assert [centered_hex_number(K) for K in range(5)] == [1, 7, 19, 37, 61]


# --- 2. tangent-line correctness ---------------------------------------------

@pytest.mark.parametrize("N", [7, 37, 61, 91])
@pytest.mark.parametrize("lam", [0.1, 0.25, 0.5])
def test_transition_passes_through_grid_point(N, lam):
    """Preactivation is exactly zero at each neuron's own grid point."""
    geo = build_hex_geometry(N, radius=RADIUS, lambda_star=lam)
    W, beta, centers = geo["W"], geo["beta"], geo["centers"]
    pre = np.einsum("md,md->m", W, centers) + beta   # W_m . c_m + beta_m
    assert np.allclose(pre, 0.0, atol=1e-12)


@pytest.mark.parametrize("N", [7, 61])
def test_weight_is_radial_and_norm_gamma(N):
    geo = build_hex_geometry(N, radius=RADIUS, lambda_star=0.25)
    W, centers, radii, gamma = geo["W"], geo["centers"], geo["radii"], geo["gamma"]
    norms = np.linalg.norm(W, axis=1)
    assert np.allclose(norms, gamma, atol=1e-12)
    interior = radii > 0
    # W_m parallel to c_m: cross product (2D scalar) is zero for interior points.
    cross = W[interior, 0] * centers[interior, 1] - W[interior, 1] * centers[interior, 0]
    assert np.allclose(cross, 0.0, atol=1e-12)
    # And points outward (positive dot with the radial direction).
    dot = np.einsum("md,md->m", W[interior], centers[interior])
    assert np.all(dot > 0)


def test_center_point_handled():
    """The r=0 center gets beta=0 and a unit-norm (arbitrary) direction."""
    geo = build_hex_geometry(7, radius=RADIUS, lambda_star=0.25)
    radii = geo["radii"]
    i0 = int(np.argmin(radii))
    assert radii[i0] == pytest.approx(0.0)
    assert geo["beta"][i0] == pytest.approx(0.0)
    assert np.linalg.norm(geo["W"][i0]) == pytest.approx(geo["gamma"], abs=1e-12)


# --- 3. bandwidth duality ----------------------------------------------------

@pytest.mark.parametrize("N", [19, 61, 127])
def test_lambda_gamma_duality(N):
    centers, radii, d = hex_pack(N, RADIUS)
    lam = 0.3
    _, _, gamma_from_lambda, lam_back = tangent_geometry(
        centers, radii, d, lambda_star=lam)
    assert gamma_from_lambda == pytest.approx(lam / d, rel=1e-12)
    assert lam_back == pytest.approx(lam, rel=1e-12)
    # Round-trip: feeding that gamma back recovers lambda.
    _, _, gamma2, lam2 = tangent_geometry(centers, radii, d, gamma=gamma_from_lambda)
    assert gamma2 == pytest.approx(gamma_from_lambda, rel=1e-12)
    assert lam2 == pytest.approx(lam, rel=1e-12)


def test_requires_exactly_one_bandwidth():
    centers, radii, d = hex_pack(19, RADIUS)
    with pytest.raises(ValueError):
        tangent_geometry(centers, radii, d)
    with pytest.raises(ValueError):
        tangent_geometry(centers, radii, d, lambda_star=0.3, gamma=5.0)


def test_gamma_scales_with_width_at_fixed_lambda():
    """Fixed lambda -> gamma grows as N increases (spacing shrinks)."""
    gammas = [build_hex_geometry(N, radius=RADIUS, lambda_star=0.25)["gamma"]
              for N in HEX_NUMBERS]
    assert all(b > a for a, b in zip(gammas, gammas[1:]))


# --- 4. phi formula ----------------------------------------------------------

def test_build_phi_2d_matches_explicit_formula():
    geo = build_hex_geometry(37, radius=RADIUS, lambda_star=0.25)
    W, beta, centers, radii, gamma = (
        geo["W"], geo["beta"], geo["centers"], geo["radii"], geo["gamma"])
    X = disk_uniform(200, radius=1.0, seed=1)
    Phi = build_phi_2d(X, W, beta)
    assert Phi.shape == (200, 37)
    # Rebuild from n.x - r explicitly for interior neurons.
    interior = radii > 0
    nhat = centers[interior] / radii[interior, None]
    expected = np.tanh(gamma * (X @ nhat.T - radii[interior][None, :]))
    assert np.allclose(Phi[:, interior], expected, atol=1e-12)


# --- 5. regime sanity --------------------------------------------------------

def test_in_span_target_recovered_near_eps():
    """Control: a target that lies in the Phi span (a known readout) must be
    recovered as a FUNCTION to ~machine precision, even though the coefficients
    are underdetermined (the 2D analog of exp04's function_diff)."""
    geo = build_hex_geometry(127, radius=RADIUS, lambda_star=0.25)
    X = disk_uniform(4000, radius=1.0, seed=0)
    Phi = build_phi_2d(X, geo["W"], geo["beta"])
    rng = np.random.default_rng(0)
    v_true = rng.standard_normal(Phi.shape[1])
    b_true = 0.37
    y = Phi @ v_true + b_true
    v, b, info = solve_readout_with_bias(Phi, y, method="lstsq")
    Xev = disk_grid(120, radius=1.0)
    Phiev = build_phi_2d(Xev, geo["W"], geo["beta"])
    yev = Phiev @ v_true + b_true
    linf = np.abs(Phiev @ v + b - yev).max()
    assert linf < 1e-9, f"in-span recovery Linf={linf:.2e}"


def test_smooth_target_residual_decreases_with_width():
    """lstsq residual on a smooth target should fall as N grows."""
    X = disk_uniform(6000, radius=1.0, seed=0)
    Xev = disk_grid(120, radius=1.0)
    tgt = get_target_2d("gauss_bump")
    y, yev = tgt.fn_numpy(X), tgt.fn_numpy(Xev)
    linfs = []
    for N in [19, 61, 127]:
        geo = build_hex_geometry(N, radius=RADIUS, lambda_star=0.25)
        Phi = build_phi_2d(X, geo["W"], geo["beta"])
        v, b, _ = solve_readout_with_bias(Phi, y, method="lstsq")
        Phiev = build_phi_2d(Xev, geo["W"], geo["beta"])
        linfs.append(np.abs(Phiev @ v + b - yev).max())
    assert linfs[-1] < linfs[0], f"no improvement with width: {linfs}"
